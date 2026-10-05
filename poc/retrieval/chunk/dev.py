"""Retrieval-only dev sweep (no LLM, no gold answers): doc- and page-level hit rate @k.

Config is chosen on a fixed DEV slice of 20 questions; `--all` reports the same metrics
on all 150 as a diagnostic after the choice is made.

  uv run python -m poc.retrieval.chunk.dev --tables chunks chunks_c600 --ks 5 10 20
"""
import argparse

import pandas as pd

from poc import data
from poc.retrieval.chunk.retrieve import hit_stats, search


def dev_slice(n: int = 20) -> list[dict]:
    qs = data.questions()
    step = len(qs) / n
    return [qs[int(i * step)] for i in range(n)]  # deterministic, spread over companies / types


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", nargs="+", default=["chunks"])
    ap.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20])
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    qs = data.questions() if a.all else dev_slice()
    rows = []
    for t in a.tables:
        hits = search([q["question"] for q in qs], k=max(a.ks), table=t)
        for k in a.ks:
            st = pd.DataFrame([hit_stats(q, h[:k]) for q, h in zip(qs, hits)])
            rows.append({"table": t, "k": k, "n": len(qs), "doc_hit": st.gold_doc_hit.mean(),
                         "page_hit": st.gold_page_hit.mean()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
