"""Bharat question generation CLI.

  uv run python -m poc.qgen.run plan        # stratified document sample -> runs/qgen-plan.jsonl
  uv run python -m poc.qgen.run estimate    # count input tokens, print cost estimate (no generation)
  uv run python -m poc.qgen.run generate    # Claude calls (resumable; stops at --budget USD)
  uv run python -m poc.qgen.run filter      # -> questions.jsonl (150, balanced)
  uv run python -m poc.qgen.run spotcheck   # -> spotcheck.md / spotcheck.csv
  uv run python -m poc.qgen.run all         # plan, estimate, generate, filter, spotcheck
"""
import argparse
import asyncio
import json
from collections import Counter

from poc.qgen import BUDGET_USD, PLAN, generate, spent


def _plan(fresh: bool) -> list[dict]:
    if PLAN.exists() and not fresh:
        return generate.load_plan()
    plan = generate.sample_docs()
    generate.save_plan(plan)
    print(f"plan: {len(plan)} documents; topics {dict(Counter(p['topic'] for p in plan))}; "
          f"languages {dict(Counter(p['language'] for p in plan))}")
    return plan


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["plan", "estimate", "generate", "filter", "spotcheck", "all"])
    p.add_argument("--budget", type=float, default=BUDGET_USD, help="hard cap on total qgen spend (USD)")
    p.add_argument("--fresh-plan", action="store_true")
    a = p.parse_args()

    if a.step in ("plan", "estimate", "generate", "all"):
        plan = _plan(a.fresh_plan)
    if a.step in ("estimate", "all"):
        est = generate.estimate(plan)
        print(f"estimate: {est}  (already spent ${spent():.2f}, cap ${a.budget:.2f})")
        if a.step == "all" and est["est_usd"] + spent() > a.budget * 1.2:
            raise SystemExit("estimate exceeds the budget; reduce the plan or raise --budget")
    if a.step in ("generate", "all"):
        asyncio.run(generate.run(plan, a.budget))
    if a.step in ("filter", "all"):
        from poc.qgen import filter as qfilter
        print(json.dumps(qfilter.run(), indent=1, default=str, ensure_ascii=False))
    if a.step in ("spotcheck", "all"):
        from poc.qgen import spotcheck
        print("spot-check pack:", spotcheck.run())


if __name__ == "__main__":
    main()
