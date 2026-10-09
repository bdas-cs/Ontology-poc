"""Cost ledger: tokens per phase from runs/llm-*.jsonl, costed three ways (paper §5).

- Hosted API equivalent: the paper's basis (GPT-5.4 nano $0.20/1M in, $1.25/1M out,
  text-embedding-3-large $0.13/1M), override with env vars.
- Laptop: measured package power (telemetry.duckdb, if Phase 4 collected it) × busy hours
  × electricity price, plus optional hardware amortisation per hour.
- Rented GPU: busy hours × GPU_HOURLY_USD (default $4.41/h, one H100 GPU Droplet on demand per DO pricing page 2026-10-09;
  power is included in the rental). Also printed: the whole session at that rate, from the first
  to the last ledger row, because the droplet bills while idle too.

Busy hours per tag are estimated as the union of call intervals in the ledger, so
concurrent calls are not double counted.

  uv run python -m poc.bench.cost
  uv run python -m poc.bench.cost --runs ~/ontology-poc-data/financebench/runs-gpu \
      --telemetry ~/ontology-poc-data/telemetry-gpu/telemetry.duckdb      # results pulled from the droplet
"""
import argparse
import json
import os

import duckdb
import pandas as pd

from poc import config

API_IN = float(os.getenv("API_IN_PER_M", "0.20"))
API_OUT = float(os.getenv("API_OUT_PER_M", "1.25"))
API_EMBED = float(os.getenv("API_EMBED_PER_M", "0.13"))
KWH_PRICE = float(os.getenv("KWH_PRICE", "0.17"))
DEFAULT_WATTS = float(os.getenv("LAPTOP_WATTS", "90"))
# e.g. $5,000 machine over 3 years at 50% duty ≈ $0.38/h; 0 = energy only
HW_PER_HOUR = float(os.getenv("HW_PER_HOUR", "0"))
GPU_HOURLY_USD = float(os.getenv("GPU_HOURLY_USD", "4.41"))


def busy_hours(df: pd.DataFrame) -> float:
    iv = sorted(zip(df.ts - df.wall_s, df.ts))
    total, cur_s, cur_e = 0.0, None, None
    for s, e in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total / 3600


def measured_watts(path=None) -> float | None:
    """Mean power while busy (gpu_active > 0.3) from telemetry samples: package power on the Mac,
    GPU board power on an NVIDIA box (see poc.telemetry.collector)."""
    path = path or config.TELEMETRY_DIR / "telemetry.duckdb"
    if not path.exists():
        return None
    try:
        con = duckdb.connect(str(path), read_only=True)
        tables = {r[0] for r in con.execute("show tables").fetchall()}
        for t in ("samples", "raw_samples", "telemetry_raw"):
            if t in tables:
                cols = {r[0] for r in con.execute(f"describe {t}").fetchall()}
                for c in ("all_power_w", "all_power", "sys_power", "package_power", "total_power_w"):
                    if c in cols:
                        busy = next((u for u in ("gpu_active", "gpu_usage") if u in cols), None)
                        return con.execute(f"select avg({c}) from {t} where {busy} > 0.3").fetchone()[0] \
                            if busy else con.execute(f"select avg({c}) from {t}").fetchone()[0]
    except Exception:
        return None
    return None


def main() -> None:
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=config.RUNS_DIR, help="runs dir with llm-*.jsonl ledgers")
    ap.add_argument("--telemetry", type=Path, default=None, help="telemetry.duckdb for measured watts")
    a = ap.parse_args()
    runs = a.runs.expanduser()
    frames = []
    for p in sorted(runs.glob("llm-*.jsonl")):
        df = pd.read_json(p, lines=True)
        if len(df):
            frames.append(df)
    if not frames:
        print("no ledgers yet")
        return
    df = pd.concat(frames)
    measured = measured_watts(a.telemetry.expanduser() if a.telemetry else None)
    watts = measured or DEFAULT_WATTS
    rows = []
    for tag, g in df.groupby("tag"):
        h = busy_hours(g)
        tin, tout = int(g.prompt_tokens.sum()), int(g.completion_tokens.sum())
        api = tin / 1e6 * API_IN + tout / 1e6 * API_OUT
        local = h * (watts / 1000 * KWH_PRICE + HW_PER_HOUR)
        rows.append({"tag": tag, "calls": len(g), "in_tok": tin, "out_tok": tout,
                     "busy_h": round(h, 3), "agg_out_tok_s": round(tout / (h * 3600), 1) if h else 0,
                     "api_usd": round(api, 3), "laptop_usd": round(local, 3),
                     "gpu_usd": round(h * GPU_HOURLY_USD, 3)})
    out = pd.DataFrame(rows)
    for name in ("ingest-chunk.json", "ingest-ontology.json"):
        p = runs / name
        if p.exists():
            stats = json.loads(p.read_text())
            emb = stats.get("embedding_tokens") or stats.get("embed_tokens")
            if emb:
                print(f"{name}: embedding tokens {emb:,} → API ${emb / 1e6 * API_EMBED:.3f}")
    print(f"power basis: {watts:.0f} W ({'measured' if measured else 'assumed'}), "
          f"${KWH_PRICE}/kWh, hw ${HW_PER_HOUR}/h;  GPU basis: ${GPU_HOURLY_USD}/h (GPU_HOURLY_USD)")
    print(out.to_string(index=False))
    session_h = (df.ts.max() - (df.ts - df.wall_s).min()) / 3600
    print(f"\nTOTAL api ${out.api_usd.sum():.2f}  vs laptop ${out.laptop_usd.sum():.2f}"
          f"  vs rented GPU ${out.gpu_usd.sum():.2f} busy-hours"
          f" (${session_h * GPU_HOURLY_USD:.2f} for the {session_h:.1f} h ledger span; the droplet bills idle time too)")
    out.to_csv(runs / "cost-summary.csv", index=False)


if __name__ == "__main__":
    main()
