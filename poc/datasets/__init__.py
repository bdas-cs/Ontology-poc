"""Benchmark registry. POC_DATASET (see poc.config) selects the active adapter.

Every adapter module exposes the same interface, so ingestion, retrieval and the
judge are dataset-agnostic:

  NAME: str
  questions() -> list[dict]       common question schema (below)
  doc_info()  -> dict[str, dict]  per-document metadata keyed by doc_name
                                  (entity/company, doc_type, period, title, ... — free-form)
  corpus_docs() -> list[str]      doc_names that form the retrieval corpus
  build_pages() -> None           writes DB_DIR/pages.parquet: doc_name, page (1-based), text, n_images
  download() -> None              fetches raw data into DATASET_DIR/raw (idempotent)

Common question schema:
  qid            stable question id (str)
  question       question text — the ONLY field pipelines may use at answer time
  answer         gold answer (str)
  justification  gold reasoning, may be ""
  question_type  dataset-specific category used for per-type reporting
  doc_name       gold document (diagnostics only)
  evidence       [{evidence_text, doc_name, evidence_page_num (0-based)}]
"""
import importlib

from poc import config

ADAPTERS = {
    "financebench": "poc.datasets.financebench",
    "docfinqa": "poc.datasets.docfinqa",
    "cuad": "poc.datasets.cuad",
    "bharat": "poc.datasets.bharat",
}


def get(name: str | None = None):
    name = name or config.DATASET
    if name not in ADAPTERS:
        raise SystemExit(f"unknown POC_DATASET={name!r}; choose from {sorted(ADAPTERS)}")
    return importlib.import_module(ADAPTERS[name])
