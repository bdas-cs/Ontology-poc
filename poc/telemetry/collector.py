"""Phase 4 step 1: per-device telemetry collector (paper §7, layer 1).

Samples the one "device" in this fleet -- the machine serving the local LLM --
every INTERVAL_MS and appends rows to DuckDB:

  samples       one row per tick: hardware metrics (no sudo) +
                LLM-serving counters for the tick (requests completed, tokens,
                POSTs / errors / prompt-processing lines seen in server.log,
                prompt-cache size; vLLM scheduler running/waiting + KV-cache use)

Hardware backend (TELEMETRY_BACKEND=auto|macmon|nvidia, default auto):
  macmon  Apple silicon (`macmon pipe`): SoC/GPU/CPU/ANE power, temps, fans, RAM, swap.
  nvidia  NVIDIA GPU server (`nvidia-smi --query-gpu`, picked when nvidia-smi is on PATH).
          Same columns where a metric exists: gpu_active = GPU utilisation, gpu_freq_mhz =
          SM clock, gpu_power_w = all_power_w = GPU board power (summed over GPUs; this is
          the "SoC power" basis used by aggregate/view), gpu_temp_c = hottest GPU, fan_* =
          fan % (N/A on SXM boards), ram/swap/cpu_usage from /proc. Apple-only columns stay
          NULL; GPU memory, HBM temperature and memory clock go to the extra gpu_mem_* columns.
  llm_requests  every completed request found in runs/llm-*.jsonl ledgers
  log_events    server.log lines that look like errors / warnings

DuckDB allows one writer process at a time, so the collector buffers rows and
opens the database only for a short write every FLUSH_S seconds; readers
(aggregate / explain) open it between flushes.

Run detached:  python -m poc.telemetry.collector --daemon
Stop:          python -m poc.telemetry.collector --stop
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from poc import config

_BACKEND_ENV = os.getenv("TELEMETRY_BACKEND", "auto")
BACKEND = (("nvidia" if shutil.which("nvidia-smi") else "macmon") if _BACKEND_ENV == "auto" else _BACKEND_ENV)
DEVICE_ID = os.getenv("TELEMETRY_DEVICE_ID",
                      "m5max-laptop-gpu0" if BACKEND == "macmon" else f"{socket.gethostname()}-gpu0")
DB_PATH = config.TELEMETRY_DIR / "telemetry.duckdb"
TEL_DIR = config.TELEMETRY_DIR
PIDFILE = TEL_DIR / "collector.pid"
LOGFILE = TEL_DIR / "collector.log"
SERVER_LOG = config.DATA_DIR / "server.log"
INTERVAL_MS = int(os.getenv("TELEMETRY_INTERVAL_MS", "1000"))
FLUSH_S = 10

ERR_RE = re.compile(r"(ERROR|Traceback|Exception|exception|CRITICAL|\" 5\d\d |\" 4\d\d |Killed|out of memory|OOM)")
WARN_RE = re.compile(r"(WARNING|Warning)")
# final line of a Python traceback, e.g. "BrokenPipeError: [Errno 32] Broken pipe" -- one per exception
CLASS_RE = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception|Exit|Interrupt)\b")
CONTEXT_RE = re.compile(r"^(Traceback|Exception occurred during processing)")
CACHE_RE = re.compile(r"Prompt Cache: (\d+) sequences, ([\d.]+) GB")
# vLLM periodic stats line: "... Running: 12 reqs, Waiting: 0 reqs, ... GPU KV cache usage: 41.3%, ..."
VLLM_STATS_RE = re.compile(r"Running: (\d+) reqs, Waiting: (\d+) reqs.*?KV cache usage: ([\d.]+)%")

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
  ts TIMESTAMPTZ, epoch DOUBLE, device_id VARCHAR,
  gpu_active DOUBLE, gpu_freq_mhz DOUBLE, gpu_power_w DOUBLE, gpu_ram_power_w DOUBLE,
  cpu_usage DOUBLE, pcpu_freq_mhz DOUBLE, cpu_power_w DOUBLE, ane_power_w DOUBLE,
  ram_power_w DOUBLE, all_power_w DOUBLE, sys_power_w DOUBLE,
  gpu_temp_c DOUBLE, cpu_temp_c DOUBLE, fan_rpm DOUBLE, fan_max_rpm DOUBLE,
  ram_used_gb DOUBLE, swap_used_gb DOUBLE,
  llm_reqs_done INTEGER, llm_prompt_tok INTEGER, llm_completion_tok INTEGER,
  log_posts INTEGER, log_prompt_progress INTEGER, log_errors INTEGER, log_warnings INTEGER,
  cache_seqs INTEGER, cache_gb DOUBLE
);
ALTER TABLE samples ADD COLUMN IF NOT EXISTS gpu_mem_used_gb DOUBLE;
ALTER TABLE samples ADD COLUMN IF NOT EXISTS gpu_mem_total_gb DOUBLE;
ALTER TABLE samples ADD COLUMN IF NOT EXISTS gpu_mem_util DOUBLE;
ALTER TABLE samples ADD COLUMN IF NOT EXISTS gpu_mem_temp_c DOUBLE;
ALTER TABLE samples ADD COLUMN IF NOT EXISTS gpu_mem_clock_mhz DOUBLE;
ALTER TABLE samples ADD COLUMN IF NOT EXISTS kv_cache_pct DOUBLE;
ALTER TABLE samples ADD COLUMN IF NOT EXISTS sched_running INTEGER;
ALTER TABLE samples ADD COLUMN IF NOT EXISTS sched_waiting INTEGER;
CREATE TABLE IF NOT EXISTS llm_requests (
  ts TIMESTAMPTZ, epoch DOUBLE, device_id VARCHAR, tag VARCHAR, wall_s DOUBLE,
  prompt_tokens INTEGER, completion_tokens INTEGER, thinking BOOLEAN, finish_reason VARCHAR
);
CREATE TABLE IF NOT EXISTS proc_snapshots (
  ts TIMESTAMPTZ, epoch DOUBLE, device_id VARCHAR, pid INTEGER, cpu_pct DOUBLE, mem_pct DOUBLE, command VARCHAR
);
CREATE TABLE IF NOT EXISTS log_events (
  ts TIMESTAMPTZ, epoch DOUBLE, device_id VARCHAR, kind VARCHAR, line VARCHAR
);
"""


def connect(read_only: bool = False, tries: int = 60) -> duckdb.DuckDBPyConnection:
    """Open the DB, retrying while another process holds the write lock."""
    last = None
    for _ in range(tries):
        try:
            return duckdb.connect(str(DB_PATH), read_only=read_only)
        except duckdb.IOException as e:  # lock held by the other process
            last = e
            time.sleep(0.5)
    raise last  # type: ignore[misc]


class LedgerTail:
    """Incrementally reads new rows from every runs/llm-*.jsonl ledger."""

    def __init__(self, start_at_end: bool):
        self.offsets: dict[Path, int] = {}
        if start_at_end:
            for p in config.DATA_DIR.glob("*/runs/llm-*.jsonl"):
                self.offsets[p] = p.stat().st_size

    def poll(self) -> list[dict]:
        rows = []
        for p in config.DATA_DIR.glob("*/runs/llm-*.jsonl"):
            off = self.offsets.get(p, 0)
            size = p.stat().st_size
            if size < off:
                off = 0
            if size == off:
                continue
            with p.open("rb") as f:
                f.seek(off)
                chunk = f.read()
            # only consume complete lines
            end = chunk.rfind(b"\n") + 1
            self.offsets[p] = off + end
            for line in chunk[:end].splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                r.setdefault("tag", p.stem.removeprefix("llm-"))
                rows.append(r)
        return rows


class LogTail:
    def __init__(self, path: Path):
        self.path = path
        self.off = path.stat().st_size if path.exists() else 0
        self.cache_seqs, self.cache_gb = 0, 0.0
        self.running = self.waiting = self.kv_pct = None

    def poll(self) -> list[str]:
        if not self.path.exists():
            return []
        size = self.path.stat().st_size
        if size < self.off:
            self.off = 0
        if size == self.off:
            return []
        with self.path.open("rb") as f:
            f.seek(self.off)
            chunk = f.read()
        end = chunk.rfind(b"\n") + 1
        self.off += end
        # progress bars use \r; split on both
        return [l for l in re.split(r"[\r\n]+", chunk[:end].decode("utf-8", "replace")) if l.strip()]


def top_procs(n: int = 6) -> list[tuple[int, float, float, str]]:
    """Top-n processes by CPU (no sudo). Commands are trimmed to the module/binary name + args."""
    try:
        cmd = (["ps", "-eo", "pid=,pcpu=,pmem=,args=", "--sort=-pcpu"] if sys.platform.startswith("linux")
               else ["ps", "-Ao", "pid=,pcpu=,pmem=,command=", "-r"])  # BSD -r = sort by CPU; GNU -r = running only
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return []
    rows = []
    for i, line in enumerate(out.splitlines()):
        parts = line.split(None, 3)
        if len(parts) != 4:
            continue
        # top-n by CPU, plus every python job (GPU work via MPS/MLX shows little CPU)
        if i < n or ("python" in parts[3] and " -m " in parts[3]):
            cmd = re.sub(r"^\S*/", "", parts[3])[:160]
            rows.append((int(parts[0]), float(parts[1]), float(parts[2]), cmd))
    return rows


SAMPLE_COLS = [
    "ts", "epoch", "device_id",
    "gpu_active", "gpu_freq_mhz", "gpu_power_w", "gpu_ram_power_w",
    "cpu_usage", "pcpu_freq_mhz", "cpu_power_w", "ane_power_w",
    "ram_power_w", "all_power_w", "sys_power_w",
    "gpu_temp_c", "cpu_temp_c", "fan_rpm", "fan_max_rpm",
    "ram_used_gb", "swap_used_gb",
    "llm_reqs_done", "llm_prompt_tok", "llm_completion_tok",
    "log_posts", "log_prompt_progress", "log_errors", "log_warnings",
    "cache_seqs", "cache_gb",
    "gpu_mem_used_gb", "gpu_mem_total_gb", "gpu_mem_util", "gpu_mem_temp_c", "gpu_mem_clock_mhz",
    "kv_cache_pct", "sched_running", "sched_waiting",
]
INSERT_SAMPLES = f"INSERT INTO samples ({', '.join(SAMPLE_COLS)}) VALUES ({','.join('?' * len(SAMPLE_COLS))})"


def macmon_source(stop):
    """Yields one hardware dict per tick from `macmon pipe` (Apple silicon, no sudo)."""
    proc = subprocess.Popen(["macmon", "pipe", "-i", str(INTERVAL_MS)], stdout=subprocess.PIPE, text=True, bufsize=1)
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            if stop():
                break
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            temp = m.get("temp", {}) or {}
            fans = m.get("fans", []) or []
            mem = m.get("memory", {}) or {}
            yield {
                "gpu_active": m.get("gpu_active_ratio"), "gpu_freq_mhz": (m.get("gpu_usage") or [None])[0],
                "gpu_power_w": m.get("gpu_power"), "gpu_ram_power_w": m.get("gpu_ram_power"),
                "cpu_usage": m.get("cpu_usage_pct"), "pcpu_freq_mhz": m.get("pcpu_freq_mhz"),
                "cpu_power_w": m.get("cpu_power"), "ane_power_w": m.get("ane_power"),
                "ram_power_w": m.get("ram_power"), "all_power_w": m.get("all_power"), "sys_power_w": m.get("sys_power"),
                "gpu_temp_c": temp.get("gpu_temp_avg"), "cpu_temp_c": temp.get("cpu_temp_avg"),
                "fan_rpm": sum(f.get("rpm", 0) for f in fans) / max(len(fans), 1),
                "fan_max_rpm": sum(f.get("max_rpm", 0) for f in fans) / max(len(fans), 1),
                "ram_used_gb": mem.get("ram_usage", 0) / 2**30, "swap_used_gb": mem.get("swap_usage", 0) / 2**30,
            }
    finally:
        proc.terminate()


NV_FIELDS = ["utilization.gpu", "utilization.memory", "power.draw", "temperature.gpu", "clocks.sm",
             "memory.used", "memory.total", "fan.speed", "temperature.memory", "clocks.mem"]
NV_CORE = 8  # the first 8 fields exist on every driver; the last two are dropped if a driver rejects them


def _num(v: str) -> float | None:
    try:
        return float(v.strip())
    except ValueError:  # "[N/A]", "[Not Supported]", ""
        return None


def _proc_cpu() -> tuple[int, int] | None:
    try:
        f = [int(x) for x in Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]]
    except Exception:
        return None
    return sum(f), f[3] + (f[4] if len(f) > 4 else 0)  # (total, idle + iowait) jiffies


def _proc_mem() -> tuple[float | None, float | None]:
    try:
        kv = {l.split(":")[0]: int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines() if ":" in l}
    except Exception:
        return None, None
    return ((kv.get("MemTotal", 0) - kv.get("MemAvailable", 0)) / 2**20,
            (kv.get("SwapTotal", 0) - kv.get("SwapFree", 0)) / 2**20)


def nvidia_source(stop):
    """Polls `nvidia-smi --query-gpu` every INTERVAL_MS; aggregates over GPUs (sum power/memory,
    mean utilisation/clock, max temperature). Host CPU / RAM / swap come from /proc."""
    fields = list(NV_FIELDS)
    prev_cpu = _proc_cpu()
    period = INTERVAL_MS / 1000
    nxt = time.time()
    while not stop():
        try:
            r = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
        except subprocess.TimeoutExpired:
            print("nvidia-smi timed out", flush=True)
            continue
        if r.returncode != 0 and len(fields) > NV_CORE:
            print(f"nvidia-smi rejected optional fields, dropping them: {r.stderr.strip()[:200]}", flush=True)
            fields = fields[:NV_CORE]
            continue
        gpus = [dict(zip(fields, map(_num, l.split(",")))) for l in r.stdout.strip().splitlines() if l.strip()]
        cur_cpu = _proc_cpu()
        cpu = None
        if prev_cpu and cur_cpu and cur_cpu[0] > prev_cpu[0]:
            cpu = 1 - (cur_cpu[1] - prev_cpu[1]) / (cur_cpu[0] - prev_cpu[0])
        prev_cpu = cur_cpu
        ram, swap = _proc_mem()

        def agg(key, fn):
            vals = [g[key] for g in gpus if g.get(key) is not None]
            return fn(vals) if vals else None

        def mean(v):
            return sum(v) / len(v)

        if gpus:
            util, mutil, power, fan = (agg("utilization.gpu", mean), agg("utilization.memory", mean),
                                       agg("power.draw", sum), agg("fan.speed", mean))
            mused, mtot = agg("memory.used", sum), agg("memory.total", sum)
            yield {
                "gpu_active": util / 100 if util is not None else None, "gpu_freq_mhz": agg("clocks.sm", mean),
                "gpu_power_w": power, "all_power_w": power, "gpu_temp_c": agg("temperature.gpu", max),
                "fan_rpm": fan, "fan_max_rpm": 100.0 if fan is not None else None,
                "cpu_usage": cpu, "ram_used_gb": ram, "swap_used_gb": swap,
                "gpu_mem_used_gb": mused / 1024 if mused is not None else None,   # MiB -> GiB
                "gpu_mem_total_gb": mtot / 1024 if mtot is not None else None,
                "gpu_mem_util": mutil / 100 if mutil is not None else None,
                "gpu_mem_temp_c": agg("temperature.memory", max), "gpu_mem_clock_mhz": agg("clocks.mem", mean),
            }
        else:
            print(f"nvidia-smi returned no GPUs (rc={r.returncode}): {r.stderr.strip()[:200]}", flush=True)
        nxt = max(nxt + period, time.time() - period)  # keep cadence, but don't burst after a stall
        time.sleep(max(0.0, nxt - time.time()))


def run() -> None:
    TEL_DIR.mkdir(parents=True, exist_ok=True)
    con = connect()
    con.execute(SCHEMA)
    con.close()

    # Resume: only read ledger rows newer than what is already stored (backfill on first run).
    con = connect()
    last_req = con.execute("SELECT coalesce(max(epoch), 0) FROM llm_requests").fetchone()[0]
    con.close()
    ledger = LedgerTail(start_at_end=False)
    logt = LogTail(SERVER_LOG)
    print(f"collector backend={BACKEND} device_id={DEVICE_ID} interval={INTERVAL_MS}ms", flush=True)

    stop = False

    def _sig(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    buf_s, buf_r, buf_e, buf_p = [], [], [], []
    last_flush = time.time()
    n_total = 0
    source = {"macmon": macmon_source, "nvidia": nvidia_source}[BACKEND](lambda: stop)
    for hw in source:
        if stop:
            break
        now = time.time()
        ts = datetime.fromtimestamp(now, tz=timezone.utc)

        reqs = [r for r in ledger.poll() if float(r.get("ts", now)) > last_req]
        for r in reqs:
            rts = float(r.get("ts", now))
            buf_r.append((datetime.fromtimestamp(rts, tz=timezone.utc), rts, DEVICE_ID, r.get("tag"),
                          r.get("wall_s"), r.get("prompt_tokens", 0), r.get("completion_tokens", 0),
                          r.get("thinking"), r.get("finish_reason")))
        # Count only requests that completed in (roughly) this tick for the per-sample counter;
        # backfilled history is kept in llm_requests only.
        fresh = [r for r in reqs if now - float(r.get("ts", now)) < 5]

        posts = progress = errs = warns = 0
        for l in logt.poll():
            if "POST /v1/" in l:
                posts += 1
            if "Prompt processing progress" in l:
                progress += 1
            c = CACHE_RE.search(l)
            if c:
                logt.cache_seqs, logt.cache_gb = int(c.group(1)), float(c.group(2))
            v = VLLM_STATS_RE.search(l)
            if v:
                logt.running, logt.waiting, logt.kv_pct = int(v.group(1)), int(v.group(2)), float(v.group(3))
            if CLASS_RE.match(l):
                errs += 1
                buf_e.append((ts, now, DEVICE_ID, "exception", l[:500]))
            elif CONTEXT_RE.match(l):  # traceback framing: kept for context, not counted
                buf_e.append((ts, now, DEVICE_ID, "error_context", l[:500]))
            elif ERR_RE.search(l) and not l.startswith("  "):
                errs += 1
                buf_e.append((ts, now, DEVICE_ID, "error", l[:500]))
            elif WARN_RE.search(l):
                warns += 1
                buf_e.append((ts, now, DEVICE_ID, "warning", l[:500]))

        row = {
            **hw, "ts": ts, "epoch": now, "device_id": DEVICE_ID,
            "llm_reqs_done": len(fresh), "llm_prompt_tok": sum(int(r.get("prompt_tokens", 0)) for r in fresh),
            "llm_completion_tok": sum(int(r.get("completion_tokens", 0)) for r in fresh),
            "log_posts": posts, "log_prompt_progress": progress, "log_errors": errs, "log_warnings": warns,
            "cache_seqs": logt.cache_seqs, "cache_gb": logt.cache_gb,
            "kv_cache_pct": logt.kv_pct, "sched_running": logt.running, "sched_waiting": logt.waiting,
        }
        buf_s.append(tuple(row.get(c) for c in SAMPLE_COLS))

        if now - last_flush >= FLUSH_S:
            buf_p.extend((ts, now, DEVICE_ID, *p) for p in top_procs())
            try:
                con = connect(tries=10)
                if buf_s:
                    con.executemany(INSERT_SAMPLES, buf_s)
                if buf_r:
                    con.executemany(f"INSERT INTO llm_requests VALUES ({','.join('?' * 9)})", buf_r)
                if buf_e:
                    con.executemany(f"INSERT INTO log_events VALUES ({','.join('?' * 5)})", buf_e)
                if buf_p:
                    con.executemany(f"INSERT INTO proc_snapshots VALUES ({','.join('?' * 7)})", buf_p)
                con.close()
                n_total += len(buf_s)
                print(f"{ts.isoformat()} flushed samples={len(buf_s)} reqs={len(buf_r)} events={len(buf_e)} total={n_total}", flush=True)
                buf_s, buf_r, buf_e, buf_p = [], [], [], []
            except Exception as e:  # keep buffering; retry next tick
                print(f"{ts.isoformat()} flush failed: {e!r}", flush=True)
            last_flush = now
    source.close()  # stops the macmon subprocess
    if buf_s or buf_r or buf_e:
        con = connect()
        if buf_s:
            con.executemany(INSERT_SAMPLES, buf_s)
        if buf_r:
            con.executemany(f"INSERT INTO llm_requests VALUES ({','.join('?' * 9)})", buf_r)
        if buf_e:
            con.executemany(f"INSERT INTO log_events VALUES ({','.join('?' * 5)})", buf_e)
        con.close()
    PIDFILE.unlink(missing_ok=True)
    print("collector stopped", flush=True)


def backfill_exception_lines() -> int:
    """One-off repair for events collected before CLASS_RE existed: map each
    "Exception occurred ... from (host, port)" event to its traceback's final
    exception line in server.log and store it as an 'exception' event."""
    text = SERVER_LOG.read_text(errors="replace").splitlines()
    by_port, port = {}, None
    for l in text:
        m = re.search(r"Exception occurred during processing of request from \('[\d.]+', (\d+)\)", l)
        if m:
            port = m.group(1)
        elif port and CLASS_RE.match(l):
            by_port[port] = l[:500]
            port = None
    con = connect()
    rows = con.execute("""SELECT epoch, device_id, regexp_extract(line, ', (\\d+)\\)', 1) FROM log_events
                          WHERE line LIKE 'Exception occurred%'""").fetchall()
    have = {r[0] for r in con.execute("SELECT epoch FROM log_events WHERE kind = 'exception'").fetchall()}
    con.execute("UPDATE log_events SET kind = 'error_context' WHERE kind = 'error' AND (line LIKE 'Traceback%' OR line LIKE 'Exception occurred%')")
    new = [(ep, ep, dev, "exception", by_port[p]) for ep, dev, p in rows if p in by_port and ep not in have]
    if new:
        con.executemany("INSERT INTO log_events SELECT to_timestamp(?), ?, ?, ?, ?", new)
    con.close()
    return len(new)


def daemon() -> None:
    TEL_DIR.mkdir(parents=True, exist_ok=True)
    if PIDFILE.exists():
        pid = int(PIDFILE.read_text())
        try:
            os.kill(pid, 0)
            print(f"collector already running (pid {pid})")
            return
        except ProcessLookupError:
            pass
    log = LOGFILE.open("a")
    p = subprocess.Popen([sys.executable, "-m", "poc.telemetry.collector"], stdout=log, stderr=log,
                         stdin=subprocess.DEVNULL, start_new_session=True, cwd=str(config.ROOT))
    PIDFILE.write_text(str(p.pid))
    print(f"collector started pid={p.pid} db={DB_PATH} log={LOGFILE}")


def stop() -> None:
    if not PIDFILE.exists():
        print("no pidfile")
        return
    pid = int(PIDFILE.read_text())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"sent SIGTERM to {pid}")
    except ProcessLookupError:
        print("not running")
        PIDFILE.unlink(missing_ok=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--daemon", action="store_true")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--backfill-exceptions", action="store_true")
    a = ap.parse_args()
    if a.backfill_exceptions:
        print(f"backfilled {backfill_exception_lines()} exception lines")
        sys.exit(0)
    if a.daemon:
        daemon()
    elif a.stop:
        stop()
    else:
        run()
