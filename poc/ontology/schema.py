"""Ontology schema (DuckDB) for the ontology-grounded arm.

Object types (every record carries provenance: doc_name + page):

  company    (id, name, aliases[], ticker, sector)
  filing     (id, company_id, doc_name, doc_type, fiscal_year, fiscal_quarter, filed_date, n_pages)
  statement  (id, filing_id, kind income|balance|cashflow, page, title, scale, doc_name)
  line_item  (id, statement_id, filing_id, kind, label, normalized_metric, value, unit, scale,
              period, period_end, fiscal_year, duration, column_label, doc_name, page, row_idx)
  section    (id, filing_id, title, page_start, page_end, doc_name)  -- 10-K/10-Q "Item" ranges
  page       (filing_id, doc_name, page, text)                       -- scoped text search source
  chunk      (id, filing_id, doc_name, page, page_end, text, emb FLOAT[1024])  -- dense + BM25 within a filing
             (vectors reused from the shared Phase-1 LanceDB index when present: same model, same pages)

Links are explicit foreign keys (company -> filing -> statement -> line_item; filing -> section/page/chunk).
"""
from pathlib import Path

import duckdb

from poc import config

DB_PATH: Path = config.DB_DIR / "ontology.duckdb"

DDL = """
CREATE TABLE IF NOT EXISTS company (
  id VARCHAR PRIMARY KEY, name VARCHAR, aliases VARCHAR[], ticker VARCHAR, sector VARCHAR
);
CREATE TABLE IF NOT EXISTS filing (
  id VARCHAR PRIMARY KEY, company_id VARCHAR, doc_name VARCHAR, doc_type VARCHAR,
  fiscal_year INTEGER, fiscal_quarter INTEGER, filed_date DATE, n_pages INTEGER
);
CREATE TABLE IF NOT EXISTS statement (
  id VARCHAR PRIMARY KEY, filing_id VARCHAR, kind VARCHAR, page INTEGER, title VARCHAR,
  scale VARCHAR, doc_name VARCHAR
);
CREATE TABLE IF NOT EXISTS line_item (
  id VARCHAR PRIMARY KEY, statement_id VARCHAR, filing_id VARCHAR, kind VARCHAR,
  label VARCHAR, normalized_metric VARCHAR, value DOUBLE, unit VARCHAR, scale DOUBLE,
  period VARCHAR, period_end VARCHAR, fiscal_year INTEGER, duration VARCHAR,
  column_label VARCHAR, doc_name VARCHAR, page INTEGER, row_idx INTEGER
);
CREATE TABLE IF NOT EXISTS section (
  id VARCHAR PRIMARY KEY, filing_id VARCHAR, title VARCHAR, page_start INTEGER, page_end INTEGER,
  doc_name VARCHAR
);
CREATE TABLE IF NOT EXISTS page (
  filing_id VARCHAR, doc_name VARCHAR, page INTEGER, text VARCHAR
);
CREATE TABLE IF NOT EXISTS chunk (
  id VARCHAR PRIMARY KEY, filing_id VARCHAR, doc_name VARCHAR, page INTEGER, page_end INTEGER, text VARCHAR,
  emb FLOAT[1024]
);
"""

# Normalized metric vocabulary handed to the extractor (raw label is always kept too).
METRICS = [
    "revenue", "cogs", "gross_profit", "sga", "rnd", "restructuring", "operating_expenses",
    "operating_income", "interest_expense", "interest_income", "pretax_income", "income_tax",
    "net_income", "net_income_attributable", "net_income_noncontrolling", "eps_basic", "eps_diluted",
    "shares_basic", "shares_diluted", "dividends_per_share",
    "cash", "short_term_investments", "receivables", "inventory", "total_current_assets",
    "ppe_net", "goodwill", "intangibles", "total_assets", "accounts_payable", "short_term_debt",
    "current_portion_long_term_debt", "total_current_liabilities", "long_term_debt",
    "total_liabilities", "total_equity", "total_liabilities_and_equity", "retained_earnings",
    "treasury_stock", "deposits",
    "depreciation_amortization", "stock_based_compensation", "cash_from_operations", "capex",
    "acquisitions", "cash_from_investing", "dividends_paid", "share_repurchases",
    "debt_issued", "debt_repaid", "cash_from_financing", "net_change_in_cash", "other",
]


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(DB_PATH), read_only=read_only)
    if not read_only:
        con.execute(DDL)
    return con


SCHEMA_DOC = """Tables (DuckDB):
company(id, name, aliases, ticker, sector)
filing(id, company_id, doc_name, doc_type, fiscal_year, fiscal_quarter, filed_date)
statement(id, filing_id, kind['income'|'balance'|'cashflow'], page, title, scale)
line_item(id, statement_id, filing_id, kind, label, normalized_metric, value, unit, scale,
          period, period_end, fiscal_year, duration['12M'|'9M'|'6M'|'3M'|'instant'], doc_name, page)
"""
