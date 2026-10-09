"""Ontology store (DuckDB) — generic part.

Every domain plugin (poc/ontology/domains/<name>.py, see docs/ONTOLOGY_PLUGINS.md) contributes its own
typed object tables via `DDL`; this module adds the two tables every domain shares:

  page   (filing_id, doc_name, page, text)                                   -- scoped text search source
  chunk  (id, filing_id, doc_name, page, page_end, text, emb FLOAT[1024])    -- dense + BM25 within a scope

`filing_id` is the *scope id* of the document: the id of the domain record that owns the document
(finance: "F:<doc_name>" = Filing.id; contracts: "CT:<doc_name>" = Contract.id). The column keeps its
historical name so existing FinanceBench databases stay valid. Scoped search filters on it.

The finance tables (company, filing, statement, line_item, section) are documented in
poc/ontology/domains/finance.py; `DDL`, `METRICS` and `SCHEMA_DOC` are still importable from here for
backwards compatibility and resolve to the finance plugin.
"""
from pathlib import Path

import duckdb

from poc import config

DB_PATH: Path = config.DB_DIR / "ontology.duckdb"

GENERIC_DDL = """
CREATE TABLE IF NOT EXISTS page (
  filing_id VARCHAR, doc_name VARCHAR, page INTEGER, text VARCHAR
);
CREATE TABLE IF NOT EXISTS chunk (
  id VARCHAR PRIMARY KEY, filing_id VARCHAR, doc_name VARCHAR, page INTEGER, page_end INTEGER, text VARCHAR,
  emb FLOAT[1024]
);
"""


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(DB_PATH), read_only=read_only)
    if not read_only:
        from poc.ontology import domains

        con.execute(domains.get().DDL)
        con.execute(GENERIC_DDL)
    return con


def __getattr__(name: str):  # backwards compatibility: schema.DDL / METRICS / SCHEMA_DOC -> finance plugin
    if name in ("DDL", "METRICS", "SCHEMA_DOC"):
        from poc.ontology.domains import finance

        return finance.DDL + GENERIC_DDL if name == "DDL" else getattr(finance, name)
    raise AttributeError(name)
