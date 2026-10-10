"""Phase 4 step 4b: inference only on exceptions.

For every exception window in device_state_view that has no explanation yet:

  carry_forward  previous window of the same size had the same anomaly
                 signature and is explained -> reuse it (no model call), up to
                 MAX_CHAIN consecutive windows, then refresh with a model call
  rollup         a 5m/15m window whose signature equals a contained, explained
                 finer window -> reuse that explanation (no model call)
  model          otherwise call the local LLM (tag "telemetry-explain",
                 thinking off, small max_tokens) with ONLY the window's stats,
                 baseline, flags, a few samples around the event, log lines
                 and top processes; write likely_cause / explanation /
                 confidence back.
"""
from __future__ import annotations

import asyncio
import json
import os

os.environ.setdefault("LLM_MAX_INFLIGHT", "2")

from poc import llm  # noqa: E402
from poc.telemetry import aggregate, collector, view  # noqa: E402

TAG = "telemetry-explain"
MAX_TOKENS = 200
MAX_CHAIN = 10

_PLATFORM = (
    "an Apple M5 Max laptop (one GPU, 128 GB unified memory) serving a local LLM with mlx_lm.server "
    "while batch jobs (document ingestion, embeddings on the GPU via PyTorch MPS, question answering) run on it"
    if collector.BACKEND == "macmon" else
    "one NVIDIA GPU server serving a local LLM with vLLM while batch jobs (document ingestion, "
    "embeddings on the same GPU via PyTorch CUDA, question answering) run on it"
)

SYSTEM = (
    f"You are a site-reliability assistant for a one-device fleet: {_PLATFORM}. "
    "Deterministic rules flagged ONE telemetry window as an exception (see 'flags'). Using only the "
    "evidence given, state the most likely cause of the flagged condition. Do not claim anything the "
    "numbers contradict; a process whose cpu_pct is close to its cpu_pct_typical is background, not a "
    "cause. If the evidence is thin, say so and use confidence <= 0.5. Reply with JSON only: "
    '{"likely_cause": "<at most 12 words>", "explanation": "<at most 50 words, cite numbers>", '
    '"confidence": <0.0-1.0>}'
)

STAT_COLS = [
    "operating_state", "thermal_state", "capacity_pressure", "inference_efficiency",
    "n_samples", "gpu_active_avg", "gpu_busy_frac", "soc_power_avg", "soc_power_max", "sys_power_avg",
    "gpu_temp_avg", "gpu_temp_max", "gpu_temp_slope_c_per_min", "fan_frac_max", "cpu_usage_avg",
    "gen_tok_s", "prompt_tok_s", "concurrency_avg", "reqs_completed", "per_req_tok_s", "tok_per_joule",
    "log_errors", "ram_used_max_gb", "swap_delta_gb", "tags",
    "baseline_n", "bl_gpu_temp", "bl_soc_power", "bl_busy_tok_s", "bl_tok_per_joule",
    "z_gpu_temp", "z_soc_power", "z_w_per_util", "tok_s_vs_baseline", "tok_per_joule_vs_baseline",
]
SAMPLE_COLS = "strftime(ts, '%H:%M:%S') AS t, round(gpu_active,2) AS gpu, round(all_power_w,1) AS soc_w, " \
              "round(gpu_temp_c,1) AS gpu_c, round(cpu_usage,2) AS cpu, round(fan_rpm) AS fan, " \
              "llm_reqs_done AS reqs, llm_completion_tok AS gen_tok, log_errors AS errs"


def _r(v):
    return round(v, 3) if isinstance(v, float) else v


def gather(con) -> list[dict]:
    """Pending exception windows with the context a model call would need."""
    rows = con.execute(
        f"""SELECT r.device_id, r.window_s, r.ws, r.we, r.signature, r.anomaly_flags,
                   {', '.join('r.' + c for c in STAT_COLS)}
            FROM _rules r JOIN device_state_view v USING (device_id, window_s, ws)
            WHERE v.is_exception AND v.explained_by = 'pending'
            ORDER BY r.window_s, r.ws"""
    ).fetchall()
    cols = ["device_id", "window_s", "ws", "we", "signature", "anomaly_flags"] + STAT_COLS
    out = []
    for row in rows:
        d = dict(zip(cols, row))
        flags = d["anomaly_flags"]
        order = ("gpu_temp_c DESC" if "thermal_excursion" in flags
                 else "all_power_w DESC" if "power_excursion" in flags
                 else "log_errors DESC" if "error_burst" in flags else None)
        if order:  # few samples around the peak
            peak = con.execute(f"SELECT epoch FROM samples WHERE device_id=? AND epoch>=? AND epoch<? ORDER BY {order} LIMIT 1",
                               [d["device_id"], d["ws"], d["we"]]).fetchone()[0]
            samp = con.execute(f"SELECT {SAMPLE_COLS} FROM samples WHERE device_id=? AND epoch BETWEEN ? AND ? ORDER BY epoch",
                               [d["device_id"], peak - 4, peak + 4]).fetchall()
        else:  # 8 evenly spaced samples across the window
            samp = con.execute(
                f"""SELECT {SAMPLE_COLS} FROM (SELECT *, ntile(8) OVER (ORDER BY epoch) AS b,
                       row_number() OVER (PARTITION BY 1 ORDER BY epoch) AS rn FROM samples
                     WHERE device_id=? AND epoch>=? AND epoch<?) QUALIFY row_number() OVER (PARTITION BY b ORDER BY epoch)=1
                    ORDER BY epoch""", [d["device_id"], d["ws"], d["we"]]).fetchall()
        scols = ["t", "gpu", "soc_w", "gpu_c", "cpu", "fan", "reqs", "gen_tok", "errs"]
        # distinct error lines with counts (traceback framing excluded; final exception line kept)
        logs = [f"{n}x {line}" for line, n in con.execute(
            """SELECT line, count(*) FROM log_events WHERE device_id=? AND epoch>=? AND epoch<?
                 AND line NOT LIKE 'Traceback%' GROUP BY 1 ORDER BY 2 DESC LIMIT 5""",
            [d["device_id"], d["ws"], d["we"]]).fetchall()]
        # python jobs first (GPU work via MPS/MLX shows little CPU), then top CPU; each with its
        # typical CPU over the whole collection so the model can tell unusual from always-on
        procs = con.execute(
            """WITH p AS (SELECT regexp_replace(command, '\\s+--?\\S+.*$', '') AS cmd, epoch, cpu_pct
                          FROM proc_snapshots WHERE device_id = ? AND command NOT LIKE 'ps %'
                            AND command NOT LIKE 'uv run%' AND command NOT LIKE 'zsh%'
                            AND command NOT LIKE '%poc.telemetry%')
               SELECT cmd, round(avg(cpu_pct) FILTER (WHERE epoch >= ? AND epoch < ?)) AS cpu_now,
                      round(avg(cpu_pct)) AS cpu_typical
               FROM p GROUP BY 1 HAVING count(*) FILTER (WHERE epoch >= ? AND epoch < ?) > 0
               ORDER BY (cmd LIKE '%python%') DESC, 2 DESC LIMIT 6""",
            [d["device_id"], d["ws"], d["we"], d["ws"], d["we"]]).fetchall()
        d["context"] = {
            "window": {"size_s": d["window_s"],
                       "start_utc": con.execute("SELECT strftime(to_timestamp(?), '%H:%M:%S')", [d["ws"]]).fetchone()[0]},
            "flags": list(flags),
            "stats": {c: _r(d[c]) for c in STAT_COLS if d[c] is not None},
            "samples": {"cols": scols, "rows": [[_r(v) for v in s] for s in samp]},
            "log_lines": logs,
            "processes": [{"cmd": c, "cpu_pct": p, "cpu_pct_typical": t} for c, p, t in procs],
        }
        out.append(d)
    return out


def known_explanations(con) -> dict:
    return {(r[0], r[1], r[2]): {"signature": r[3], "chain": r[4]} for r in con.execute(
        """SELECT device_id, window_s, ws, signature,
                  CASE WHEN explained_by = 'carry_forward' THEN 1 ELSE 0 END FROM window_explanations""").fetchall()}


def plan(pending: list[dict], known: dict) -> tuple[list[dict], list[tuple[dict, str, tuple]]]:
    """Split pending windows into model-call heads and reuse (carry/rollup) dependents."""
    pend = {(p["device_id"], p["window_s"], p["ws"]): p for p in pending}
    heads, reuse = [], []

    def chain_len(key, sig):
        n, (dev, w, ws) = 0, key
        while True:
            prev = (dev, w, ws - w)
            src = pend.get(prev) or known.get(prev)
            if not src or src["signature"] != sig:
                return n
            n += 1
            ws -= w

    for key, p in pend.items():
        dev, w, ws = key
        sig = p["signature"]
        prev = (dev, w, ws - w)
        src = pend.get(prev) or known.get(prev)
        if src and src["signature"] == sig and chain_len(key, sig) % MAX_CHAIN != 0:
            reuse.append((p, "carry_forward", prev))
            continue
        finer = [k for k, v in list(known.items()) + list(pend.items())
                 if k[0] == dev and k[1] < w and ws <= k[2] < ws + w and v["signature"] == sig]
        if finer:
            reuse.append((p, "rollup", min(finer, key=lambda k: (k[1], k[2]))))
            continue
        heads.append(p)
    return heads, reuse


async def call_model(p: dict) -> dict:
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(p["context"], separators=(",", ":"))}]
    res = await llm.chat(msgs, tag=TAG, max_tokens=MAX_TOKENS, thinking=False, temperature=0.1)
    try:
        j = llm.parse_json(res["content"])
        cause, expl, conf = str(j.get("likely_cause", ""))[:200], str(j.get("explanation", ""))[:600], float(j.get("confidence", 0.5))
    except Exception:
        cause, expl, conf = "unparsed model output", res["content"][:600], 0.3
    u = res["usage"]
    return dict(likely_cause=cause, explanation=expl, confidence=max(0.0, min(conf, 1.0)),
                prompt_tokens=u.get("prompt_tokens", 0), completion_tokens=u.get("completion_tokens", 0),
                wall_s=round(res["wall_s"], 3))


async def explain_pending(dry_run: bool = False) -> dict:
    con = collector.connect()
    aggregate.build(con)
    view.build(con)
    pending = gather(con)
    known = known_explanations(con)
    con.close()  # never hold the DB lock across model calls (collector writes every 10 s)

    heads, reuse = plan(pending, known)
    stats = {"pending": len(pending), "model_calls": len(heads), "reused": len(reuse)}
    if dry_run or not pending:
        return stats
    results = await asyncio.gather(*(call_model(h) for h in heads))

    rows = []
    done: dict[tuple, dict] = {}
    for h, r in zip(heads, results):
        key = (h["device_id"], h["window_s"], h["ws"])
        done[key] = r
        rows.append((*key, h["signature"], r["likely_cause"], r["explanation"], r["confidence"], "model",
                     None, None, r["prompt_tokens"], r["completion_tokens"], r["wall_s"]))

    con = collector.connect()
    # resolve reuse in time order (sources are either heads, earlier reuses, or already-stored rows)
    for p, how, src in sorted(reuse, key=lambda t: (t[0]["window_s"], t[0]["ws"])):
        key = (p["device_id"], p["window_s"], p["ws"])
        s = done.get(src)
        if s is None:
            q = con.execute("SELECT likely_cause, explanation, confidence FROM window_explanations "
                            "WHERE device_id=? AND window_s=? AND ws=?", list(src)).fetchone()
            s = dict(likely_cause=q[0], explanation=q[1], confidence=q[2]) if q else None
        if s is None:
            continue
        done[key] = s
        rows.append((*key, p["signature"], s["likely_cause"], s["explanation"], s["confidence"], how,
                     src[2], src[1], 0, 0, 0.0))
    con.executemany(
        """INSERT INTO window_explanations (device_id, window_s, ws, signature, likely_cause, explanation,
             confidence, explained_by, source_ws, source_window_s, prompt_tokens, completion_tokens, wall_s)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
    view.build(con)  # write the explanations back into the view
    con.close()
    return stats


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    print(asyncio.run(explain_pending(dry_run=ap.parse_args().dry_run)))
