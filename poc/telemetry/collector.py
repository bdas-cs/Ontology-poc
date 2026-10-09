"""Phase 4 step 1: per-device telemetry collector (paper §7, layer 1).

Samples the one "device" in this fleet -- the M5 Max laptop serving the local
LLM -- every INTERVAL_S seconds and appends rows to DuckDB:

  samples       one row per tick: macmon hardware metrics (no sudo) +
                LLM-serving counters for the tick (requests completed, tokens,
                POSTs / errors / prompt-processing lines seen in server.log,
                prompt-cache size)
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
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from poc import config

DEVICE_ID = os.getenv("TELEMETRY_DEVICE_ID", "m5max-laptop-gpu0")
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
        out = subprocess.run(["ps", "-Ao", "pid=,pcpu=,pmem=,command=", "-r"], capture_output=True, text=True, timeout=5).stdout
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
    proc = subprocess.Popen(["macmon", "pipe", "-i", str(INTERVAL_MS)], stdout=subprocess.PIPE, text=True, bufsize=1)

    stop = False

    def _sig(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    buf_s, buf_r, buf_e, buf_p = [], [], [], []
    last_flush = time.time()
    n_total = 0
    for line in proc.stdout:  # type: ignore[union-attr]
        if stop:
            break
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
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

        temp = m.get("temp", {}) or {}
        fans = m.get("fans", []) or []
        mem = m.get("memory", {}) or {}
        buf_s.append((
            ts, now, DEVICE_ID,
            m.get("gpu_active_ratio"), (m.get("gpu_usage") or [None])[0], m.get("gpu_power"), m.get("gpu_ram_power"),
            m.get("cpu_usage_pct"), m.get("pcpu_freq_mhz"), m.get("cpu_power"), m.get("ane_power"),
            m.get("ram_power"), m.get("all_power"), m.get("sys_power"),
            temp.get("gpu_temp_avg"), temp.get("cpu_temp_avg"),
            sum(f.get("rpm", 0) for f in fans) / max(len(fans), 1),
            sum(f.get("max_rpm", 0) for f in fans) / max(len(fans), 1),
            mem.get("ram_usage", 0) / 2**30, mem.get("swap_usage", 0) / 2**30,
            len(fresh), sum(int(r.get("prompt_tokens", 0)) for r in fresh),
            sum(int(r.get("completion_tokens", 0)) for r in fresh),
            posts, progress, errs, warns, logt.cache_seqs, logt.cache_gb,
        ))

        if now - last_flush >= FLUSH_S:
            buf_p.extend((ts, now, DEVICE_ID, *p) for p in top_procs())
            try:
                con = connect(tries=10)
                if buf_s:
                    con.executemany(f"INSERT INTO samples VALUES ({','.join('?' * 29)})", buf_s)
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
    proc.terminate()
    if buf_s or buf_r or buf_e:
        con = connect()
        if buf_s:
            con.executemany(f"INSERT INTO samples VALUES ({','.join('?' * 29)})", buf_s)
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
