"""Phase 4 pipeline driver: aggregate -> view (rules) -> explain exceptions.

  python -m poc.telemetry.run              # one pass over everything collected so far
  python -m poc.telemetry.run --watch 60   # continuous: re-run every 60 s (Ctrl-C to stop)
  python -m poc.telemetry.run --no-explain # rules only, no model calls

Each pass is idempotent: closed windows are recomputed identically, and only
exception windows without a persisted explanation trigger work.
"""
from __future__ import annotations

import argparse
import asyncio
import time

from poc.telemetry import explain


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0, help="seconds between passes (0 = single pass)")
    ap.add_argument("--no-explain", action="store_true")
    a = ap.parse_args()
    while True:
        t0 = time.time()
        stats = asyncio.run(explain.explain_pending(dry_run=a.no_explain))
        print(f"{time.strftime('%H:%M:%S')} pass {time.time() - t0:.1f}s {stats}", flush=True)
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
