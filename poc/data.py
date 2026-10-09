"""Active-dataset facade: `from poc import data` keeps working for every benchmark.

The adapter is chosen by POC_DATASET (default financebench); see poc/datasets/__init__.py
for the interface and the common question schema.
"""
import pandas as pd

from poc import config, datasets

_ds = datasets.get()
NAME = _ds.NAME
PAGES_PATH = config.DB_DIR / "pages.parquet"

questions = _ds.questions
doc_info = _ds.doc_info
corpus_docs = _ds.corpus_docs
build_pages = _ds.build_pages
download = _ds.download


def pages() -> pd.DataFrame:
    return pd.read_parquet(PAGES_PATH)
