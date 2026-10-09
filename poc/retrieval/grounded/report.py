"""Summarise the ontology arm: resolution accuracy, tokens, wall-clock, judged accuracy dev vs full.

  uv run python -m poc.retrieval.grounded.report [--arm ontology] [--judged judged-ontology-localjudge.jsonl]
"""
import argparse
import json

import pandas as pd

from poc import config, data
from poc.retrieval.grounded.run import dev_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="ontology")
    ap.add_argument("--judged", default=None)
    a = ap.parse_args()
    qt = {q["qid"]: q["question_type"] for q in data.questions()}
    df = pd.DataFrame([json.loads(l) for l in (config.RUNS_DIR / f"answers-{a.arm}.jsonl").open()])
    df["question_type"] = df.qid.map(qt)
    df["dev"] = df.qid.isin(set(dev_ids()))
    print(f"n={len(df)}  resolved_correct_filing={df.resolved_correct_filing.mean():.1%}  "
          f"primary_correct={df.resolved_primary_correct.mean():.1%}")
    print(df.groupby("question_type")[["resolved_correct_filing"]].mean().round(3).to_string())
    print(f"tokens in={df.prompt_tokens.sum():,} out={df.completion_tokens.sum():,}  "
          f"mean wall/q={df.wall_s.mean():.1f}s  used structured records: {(df.records_used.str.len() > 0).mean():.1%}")
    jp = config.RUNS_DIR / (a.judged or f"judged-{a.arm}-localjudge.jsonl")
    if jp.exists():
        j = pd.DataFrame([json.loads(l) for l in jp.open()])
        j["dev"] = j.qid.isin(set(dev_ids()))
        for name, sub in (("full", j), ("dev", j[j.dev]), ("held-out", j[~j.dev])):
            print(f"\n== {name} n={len(sub)} strict={sub.strict.mean():.1%} expanded={sub.expanded.mean():.1%}")
            print(sub.groupby("question_type")[["strict", "expanded"]].mean().round(3).to_string())
        j["resolved_ok"] = j.resolved_correct_filing
        print("\nstrict by resolution:", j.groupby("resolved_ok").strict.agg(["mean", "count"]).round(3).to_dict())


if __name__ == "__main__":
    main()
