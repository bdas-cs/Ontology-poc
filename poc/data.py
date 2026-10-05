"""FinanceBench loaders and the shared page store.

`make pages` parses every PDF referenced by the 150 open-source questions into
db/pages.parquet (doc_name, page [1-based], text). Both retrieval arms read it.
"""
import json
from functools import lru_cache

import pandas as pd

from poc import config

REPO = config.FINANCEBENCH_DIR / "repo"
PDF_DIR = REPO / "pdfs"
PAGES_PATH = config.DB_DIR / "pages.parquet"


@lru_cache
def questions() -> list[dict]:
    with (REPO / "data" / "financebench_open_source.jsonl").open() as f:
        return [json.loads(line) for line in f]


@lru_cache
def doc_info() -> dict[str, dict]:
    with (REPO / "data" / "financebench_document_information.jsonl").open() as f:
        rows = [json.loads(line) for line in f]
    return {r["doc_name"]: r for r in rows}


def corpus_docs() -> list[str]:
    """The evaluation corpus: every filing referenced by the 150 questions (84 docs)."""
    return sorted({q["doc_name"] for q in questions()})


def pages() -> pd.DataFrame:
    return pd.read_parquet(PAGES_PATH)
