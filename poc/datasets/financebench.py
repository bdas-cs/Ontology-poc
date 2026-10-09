"""FinanceBench open set: 150 questions over SEC filings (PDFs).

Corpus defaults to the 84 filings the questions reference; FB_CORPUS=full uses
all 368 filings in the repo (harder entity resolution, closer to production).
"""
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import pandas as pd
import pymupdf

from poc import config

NAME = "financebench"
REPO = config.FINANCEBENCH_DIR / "repo"
PDF_DIR = REPO / "pdfs"
PAGES_PATH = config.DB_DIR / "pages.parquet"


def download() -> None:
    subprocess.run([str(config.ROOT / "scripts" / "get_financebench.sh"), str(config.FINANCEBENCH_DIR)], check=True)


@lru_cache
def questions() -> list[dict]:
    with (REPO / "data" / "financebench_open_source.jsonl").open() as f:
        rows = [json.loads(line) for line in f]
    return [{
        "qid": r["financebench_id"],
        "question": r["question"],
        "answer": r["answer"],
        "justification": r.get("justification") or "",
        "question_type": r["question_type"],
        "doc_name": r["doc_name"],
        "evidence": [{"evidence_text": e.get("evidence_text", ""), "doc_name": e.get("doc_name", r["doc_name"]),
                      "evidence_page_num": e.get("evidence_page_num")} for e in r.get("evidence", [])],
        "company": r.get("company"),
    } for r in rows]


@lru_cache
def doc_info() -> dict[str, dict]:
    with (REPO / "data" / "financebench_document_information.jsonl").open() as f:
        rows = [json.loads(line) for line in f]
    return {r["doc_name"]: r for r in rows}


def corpus_docs() -> list[str]:
    if os.getenv("FB_CORPUS") == "full":
        return sorted(p.stem for p in PDF_DIR.glob("*.pdf"))
    return sorted({q["doc_name"] for q in questions()})


def _parse(doc_name: str) -> list[dict]:
    rows = []
    with pymupdf.open(PDF_DIR / f"{doc_name}.pdf") as pdf:
        for i, page in enumerate(pdf, start=1):
            rows.append({"doc_name": doc_name, "page": i,
                         "text": page.get_text("text", sort=True), "n_images": len(page.get_images())})
    return rows


def build_pages() -> None:
    docs = corpus_docs()
    t0 = time.perf_counter()
    with ProcessPoolExecutor() as ex:
        rows = [r for doc_rows in ex.map(_parse, docs) for r in doc_rows]
    df = pd.DataFrame(rows)
    df.to_parquet(PAGES_PATH)
    print(f"{len(docs)} docs, {len(df)} pages, {df.text.str.len().sum() / 1e6:.1f}M chars "
          f"in {time.perf_counter() - t0:.1f}s → {PAGES_PATH}")
