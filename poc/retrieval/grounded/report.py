"""Summarise the ontology arm for the active dataset (POC_DATASET): resolution accuracy, tokens, wall-clock,
judged accuracy dev vs held-out, by question_type.

  uv run python -m poc.retrieval.grounded.report [--arm ontology] [--judged judged-ontology.jsonl]
  POC_DATASET=cuad uv run python -m poc.retrieval.grounded.report --arm ontology-r2
"""
import argparse
import json

import pandas as pd

from poc import config, data
from poc.retrieval.grounded.run import DEV_IDS_PATH


def _dev() -> set:
    return set(json.loads(DEV_IDS_PATH.read_text())) if DEV_IDS_PATH.exists() else set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="ontology")
    ap.add_argument("--judged", default=None)
    a = ap.parse_args()
    qs = {q["qid"]: q for q in data.questions()}
    qt = {k: q["question_type"] for k, q in qs.items()}
    df = pd.DataFrame([json.loads(l) for l in (config.RUNS_DIR / f"answers-{a.arm}.jsonl").open()])
    df["question_type"] = df.qid.map(qt)
    # Bharat: evidence mirrored in other corpus documents (evidence_also_in) also counts as a correct resolution.
    mirrored = {k: {q["doc_name"], *(q.get("evidence_also_in") or [])} for k, q in qs.items()}
    if any(len(v) > 1 for v in mirrored.values()):
        def ok(r, primary=False):
            f = (r.get("resolved") or {}).get("filings") or []
            return bool(f) and f[0] in mirrored[r["qid"]] if primary else bool(set(f) & mirrored[r["qid"]])
        df["resolved_gold_only"] = df.resolved_correct_filing
        df["resolved_correct_filing"] = [ok(r) for r in df.to_dict("records")]
        df["resolved_primary_correct"] = [ok(r, True) for r in df.to_dict("records")]
        print(f"(resolution counts evidence_also_in mirrors; gold doc only: {df.resolved_gold_only.mean():.1%})")
        rc = dict(zip(df.qid, df.resolved_correct_filing))
    dev = _dev()
    df["dev"] = df.qid.isin(dev)
    print(f"dataset={config.DATASET} arm={a.arm}")
    print(f"n={len(df)}  resolved_correct_doc={df.resolved_correct_filing.mean():.1%}  "
          f"primary_correct={df.resolved_primary_correct.mean():.1%}")
    print(df.groupby("question_type")[["resolved_correct_filing"]].mean().round(3).to_string())
    errs = df.answer.str.startswith("(pipeline error").sum()
    print(f"tokens in={df.prompt_tokens.sum():,} out={df.completion_tokens.sum():,}  "
          f"mean wall/q={df.wall_s.mean():.1f}s  used structured records: {(df.records_used.str.len() > 0).mean():.1%}"
          f"  pipeline errors: {errs}")
    jp = config.RUNS_DIR / (a.judged or f"judged-{a.arm}.jsonl")
    if not jp.exists() and not a.judged:
        jp = config.RUNS_DIR / f"judged-{a.arm}-localjudge.jsonl"
    if jp.exists():
        j = pd.DataFrame([json.loads(l) for l in jp.open()])
        j["dev"] = j.qid.isin(dev)
        if "resolved_gold_only" in df:
            j["resolved_correct_filing"] = j.qid.map(rc)
        groups = (("full", j), ("dev", j[j.dev]), ("held-out", j[~j.dev])) if dev else (("full", j),)
        print(f"\njudged: {jp.name}")
        for name, sub in groups:
            if len(sub) == 0:
                continue
            print(f"\n== {name} n={len(sub)} strict={sub.strict.mean():.1%} expanded={sub.expanded.mean():.1%}")
            print(sub.groupby("question_type")[["strict", "expanded"]].mean().round(3).to_string())
        if "resolved_correct_filing" in j:
            print("\nstrict by resolution:", j.groupby("resolved_correct_filing").strict.agg(["mean", "count"]).round(3).to_dict())
        extra = [c for c in ("present", "category") if c in j]
        if "present" in extra:
            print("\nstrict by present/absent:", j.groupby("present").strict.agg(["mean", "count"]).round(3).to_dict())


if __name__ == "__main__":
    main()
