"""Benchmark question generation with Claude (currently for the Bharat corpus).

  generate.py   stratified document sample -> one Claude call per document -> candidates
  filter.py     verbatim-evidence / dedupe / title-triviality filters -> balanced 150
  spotcheck.py  30-question human review pack (md + csv)
  run.py        CLI: estimate | generate | filter | spotcheck | all

Everything it writes (candidates, questions, ledger, spot-check) stays under
POC_DATA_DIR/bharat/ -- never in the repo (source texts have no blanket licence).
"""
import json
import time

from poc.datasets import bharat

MODEL = "claude-opus-5-5"
# USD per million tokens (claude-opus-5-5 list price).
PRICE = {"input": 4.00, "output": 20.00, "cache_write": 5.00, "cache_read": 0.40}
BUDGET_USD = 10.0

OUT_DIR = bharat.BHARAT_DIR
RUNS = OUT_DIR / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
LEDGER = RUNS / "qgen-ledger.jsonl"
CANDIDATES = RUNS / "qgen-candidates.jsonl"
PLAN = RUNS / "qgen-plan.jsonl"
FILTER_LOG = RUNS / "qgen-filter.jsonl"
QUESTIONS = bharat.QUESTIONS_PATH

QUESTION_TYPES = ("numeric", "eligibility_fact", "narrative")


def cost(usage: dict) -> float:
    return (usage.get("input_tokens", 0) * PRICE["input"]
            + usage.get("output_tokens", 0) * PRICE["output"]
            + usage.get("cache_creation_input_tokens", 0) * PRICE["cache_write"]
            + usage.get("cache_read_input_tokens", 0) * PRICE["cache_read"]) / 1e6


def spent() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(l).get("cost_usd", 0.0) for l in LEDGER.open() if l.strip())


def ledger(row: dict) -> None:
    with LEDGER.open("a") as f:
        f.write(json.dumps({"ts": time.time(), **row}) + "\n")
