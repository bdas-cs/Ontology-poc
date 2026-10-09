"""Finance domain plugin (FinanceBench, DocFinQA): SEC filings -> companies, filings, statements, line items.

Object types (every record carries provenance: doc_name + page):

  company    (id, name, aliases[], ticker, sector)
  filing     (id, company_id, doc_name, doc_type, fiscal_year, fiscal_quarter, filed_date, n_pages)
  statement  (id, filing_id, kind income|balance|cashflow, page, title, scale, doc_name)
  line_item  (id, statement_id, filing_id, kind, label, normalized_metric, value, unit, scale,
              period, period_end, fiscal_year, duration, column_label, doc_name, page, row_idx)
  section    (id, filing_id, title, page_start, page_end, doc_name)  -- 10-K/10-Q "Item" ranges
  fin_table  (id, filing_id, doc_name, page, caption, n_rows, n_cols)   -- text corpora only (see below)
  table_cell (table_id, filing_id, doc_name, page, row_idx, row_label, col_idx, col_label, value, raw)

Links are explicit foreign keys (company -> filing -> statement -> line_item; filing -> section/page/chunk).
Scope id of a document = Filing.id = "F:<doc_name>".

FinanceBench behaviour is the pre-plugin pipeline, moved here verbatim (same prompts, same tables, same
rules). Additions that only fire on text corpora with markdown pipe tables (DocFinQA) and so never
change FinanceBench:
  * locate.py takes the statement title from the text right above a page's largest pipe table;
  * every pipe table outside the primary statements is parsed deterministically (no LLM) into
    fin_table / table_cell. At query time the tables of the resolved filing that best match the
    question (keyword overlap on caption + row/column labels) are added to the planner's value-free
    catalogue as T<n> blocks (row keys T<n>r<i>, column codes c<j>) and executed like line items;
  * resolve honours an explicit "10-K for fiscal <year>" reference (DocFinQA question prefix);
  * the company-alias LLM call is batched (40 companies per call) for corpora with many companies;
    a ticker given by doc_info is kept.
"""
import asyncio
import json
import math
import re
import time
from collections import Counter

import pandas as pd

from poc import config, data, llm
from poc.ontology.locate import locate
from poc.ontology.runtime import (INGEST_TAG, ANSWER_TAG, Usage, chat, has_table, page_view, ro, rw, safe_eval,
                                  slug, squeeze)

NAME = "finance"
TAG = INGEST_TAG
CACHE = config.DB_DIR / "extract_cache"
SCALES = {"units": 1.0, "ones": 1.0, "thousands": 1e3, "millions": 1e6, "billions": 1e9}

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
CREATE TABLE IF NOT EXISTS fin_table (
  id VARCHAR PRIMARY KEY, filing_id VARCHAR, doc_name VARCHAR, page INTEGER, caption VARCHAR,
  n_rows INTEGER, n_cols INTEGER
);
CREATE TABLE IF NOT EXISTS table_cell (
  table_id VARCHAR, filing_id VARCHAR, doc_name VARCHAR, page INTEGER, row_idx INTEGER, row_label VARCHAR,
  col_idx INTEGER, col_label VARCHAR, value DOUBLE, raw VARCHAR
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

SCHEMA_DOC = """Tables (DuckDB):
company(id, name, aliases, ticker, sector)
filing(id, company_id, doc_name, doc_type, fiscal_year, fiscal_quarter, filed_date)
statement(id, filing_id, kind['income'|'balance'|'cashflow'], page, title, scale)
line_item(id, statement_id, filing_id, kind, label, normalized_metric, value, unit, scale,
          period, period_end, fiscal_year, duration['12M'|'9M'|'6M'|'3M'|'instant'], doc_name, page)
"""


def scope_id(doc_name: str) -> str:
    return "F:" + doc_name


def doc_of(scope: str) -> str:
    return scope[2:]


# ================================================================== ingest: base objects
ALIAS_PROMPT = """For each company below give its primary US stock ticker and the short names / former names /
abbreviations an analyst might use for it in a question (e.g. "JnJ", "JPM", "AMEX", "Square").
Return JSON: {"<company>": {"ticker": "...", "aliases": ["...", ...]}, ...}
Companies:
"""
ALIAS_BATCH = 40


async def _company_aliases(names: list[str]) -> tuple[dict, dict]:
    async def one(batch):
        out = await llm.chat([{"role": "user", "content": ALIAS_PROMPT + "\n".join(batch)}],
                             tag=TAG, max_tokens=2500, temperature=0.0)
        return llm.parse_json(out["content"]), out["usage"]

    if len(names) <= ALIAS_BATCH:  # FinanceBench: one call, exactly as before
        return await one(names)
    res = await asyncio.gather(*(one(names[i:i + ALIAS_BATCH]) for i in range(0, len(names), ALIAS_BATCH)),
                               return_exceptions=True)
    merged, usage = {}, {"prompt_tokens": 0, "completion_tokens": 0}
    for r in res:
        if isinstance(r, Exception):
            continue
        js, u = r
        if isinstance(js, dict):
            merged.update(js)
        for k in usage:
            usage[k] += u.get(k, 0)
    return merged, usage


SECTION_RE = re.compile(r"^\s*item\s+(\d{1,2}[abc]?)\s*[\.:\-–—]?\s*([A-Z][^\n]{3,80})", re.I | re.M)


def _sections(doc: str, fid: str, pages: list[tuple[int, str]]) -> list[dict]:
    starts = []
    for p, t in pages:
        head = t[:600]
        ms = SECTION_RE.findall(head)
        if len(ms) == 1 or (ms and len(ms) <= 2 and len(SECTION_RE.findall(t)) <= 3):
            num, title = ms[0]
            starts.append((p, f"Item {num.upper()}. {title.strip()[:80]}"))
    rows, last = [], max((p for p, _ in pages), default=0)
    for i, (p, title) in enumerate(starts):
        end = (starts[i + 1][0] - 1) if i + 1 < len(starts) else last
        rows.append({"id": f"S:{doc}:{p}", "filing_id": fid, "title": title, "page_start": p,
                     "page_end": max(p, end), "doc_name": doc})
    return rows


def step_base(docs: list[str], info: dict, pg: pd.DataFrame) -> dict:
    """Company / Filing / Section objects from the system of record (doc_info) + page text."""
    companies = sorted({info[d]["company"] for d in docs})
    aliases, usage = asyncio.run(_company_aliases(companies))
    comp_rows = []
    for c in companies:
        a = aliases.get(c, {}) if isinstance(aliases, dict) else {}
        a = a if isinstance(a, dict) else {}
        prefixes = {d.split("_20")[0].replace("_", " ") for d in docs if info[d]["company"] == c}
        known = next((info[d].get("ticker") for d in docs if info[d]["company"] == c and info[d].get("ticker")), None)
        ticker = known or a.get("ticker")
        al = sorted({c, c.replace("&", "and"), *prefixes, *(a.get("aliases") or []), ticker or c} - {""})
        sector = next(info[d].get("gics_sector") for d in docs if info[d]["company"] == c)
        comp_rows.append({"id": "C:" + slug(c), "name": c, "aliases": al, "ticker": ticker, "sector": sector})
    con = rw()
    con.execute("DELETE FROM company")
    con.executemany("INSERT INTO company VALUES (?,?,?,?,?)",
                    [[r["id"], r["name"], r["aliases"], r["ticker"], r["sector"]] for r in comp_rows])
    fil, secs = [], []
    con.execute("DELETE FROM filing"); con.execute("DELETE FROM section")
    for d in docs:
        i = info[d]
        q = re.search(r"\d{4}Q(\d)", d)
        dt = re.search(r"dated[-_](\d{4}-\d{2}-\d{2})", d)
        g = pg[pg.doc_name == d].sort_values("page")
        fid = scope_id(d)
        fil.append([fid, "C:" + slug(i["company"]), d, i["doc_type"].upper(), int(i["doc_period"]),
                    int(q.group(1)) if q else None, dt.group(1) if dt else None, len(g)])
        secs += _sections(d, fid, list(zip(g.page, g.text)))
    con.executemany("INSERT INTO filing VALUES (?,?,?,?,?,?,?,?)", fil)
    if secs:
        sdf = pd.DataFrame(secs)
        con.execute("INSERT INTO section SELECT id, filing_id, title, page_start, page_end, doc_name FROM sdf")
    con.close()
    return {"companies": len(comp_rows), "filings": len(fil), "sections": len(secs),
            "base_llm_prompt_tokens": usage.get("prompt_tokens", 0),
            "base_llm_completion_tokens": usage.get("completion_tokens", 0)}


# ================================================================== ingest: statement extraction
EXTRACT_SYS = """You convert one primary financial statement (text layer of a PDF page) into JSON. Be exact.
Rules:
- One entry in "columns" per numeric period column, left to right, as printed. For each column give:
  "label" (header as printed), "period_end" (YYYY-MM-DD if derivable, else ""), "fiscal_year" (the fiscal year
  the company uses for that column, int), "duration": "12M" | "9M" | "6M" | "3M" | "instant" (balance sheet = instant).
  Ignore "% change" / "% of sales" columns.
- "scale": the stated unit of the table: "thousands" | "millions" | "billions" | "units".
- "rows": every line item that has numbers, in order: [label, metric, [one value per column], unit]
  * label: the line item text as printed (prefix the parent heading if the label alone is ambiguous,
    e.g. "Inventories (change in)" in the cash flow statement).
  * metric: one of METRICS or "other". Use a metric only for the line that IS that concept on this statement
    (e.g. revenue = total revenues / net sales line; capex = purchases of property, plant and equipment;
    depreciation_amortization = the D&A line in the cash flow statement; dividends_paid = cash dividends paid).
    Each metric at most once per statement; everything else "other".
  * values: numbers exactly as printed with commas removed, parentheses => negative, "—"/"-"/blank => null.
    Do NOT rescale.
  * unit: "$" for amounts in the table scale, "$/sh" for per-share amounts, "sh" for share counts, "%" for percents.
Output only the JSON object: {"title": str, "scale": str, "columns": [...], "rows": [...]}"""


async def _extract(doc: str, kind: str, pages: list[int], texts: list[str], usage_acc: list) -> dict | None:
    CACHE.mkdir(exist_ok=True)
    cpath = CACHE / f"{doc}__{kind}.json"
    if cpath.exists():
        return json.loads(cpath.read_text())
    body = "\n\n".join(f"[page {p}]\n{squeeze(t)}" for p, t in zip(pages, texts))
    msgs = [{"role": "system", "content": EXTRACT_SYS.replace("METRICS", ", ".join(METRICS))},
            {"role": "user", "content": f"Statement type: {kind}\nFiling: {doc}\n\n{body}"}]
    for attempt in range(2):
        out = await llm.chat(msgs, tag=TAG, max_tokens=6000, temperature=0.0)
        usage_acc.append(out["usage"])
        try:
            js = llm.parse_json(out["content"])
            if isinstance(js, dict) and js.get("rows"):
                js["_pages"] = pages
                cpath.write_text(json.dumps(js))
                return js
        except Exception:
            pass
    print("extract failed", doc, kind)
    return None


def _num(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "").replace("$", "")
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        x = float(s)
    except ValueError:
        return None
    return -x if neg else x


def _period(col: dict) -> str:
    fy, dur = col.get("fiscal_year"), col.get("duration") or ""
    if dur == "instant":
        return f"FY{fy}-end" if not col.get("period_end") else f"{col['period_end']}"
    return f"FY{fy}" if dur == "12M" else f"FY{fy}-{dur}"


def rows_from_extract(doc: str, kind: str, js: dict) -> tuple[dict, list[list]]:
    fid, sid = scope_id(doc), f"ST:{doc}:{kind}"
    pages = js["_pages"]
    scale = SCALES.get(str(js.get("scale", "units")).lower(), 1.0)
    st = {"id": sid, "filing_id": fid, "kind": kind, "page": pages[0], "title": js.get("title", ""),
          "scale": js.get("scale", ""), "doc_name": doc}
    cols = js.get("columns") or []
    out = []
    for ri, row in enumerate(js.get("rows") or []):
        if not isinstance(row, list) or len(row) < 3:
            continue
        label, metric, vals = row[0], row[1], row[2]
        unit = row[3] if len(row) > 3 else "$"
        if not isinstance(vals, list):
            vals = [vals]
        metric = metric if metric in METRICS else "other"
        sc = scale if unit in ("$", "sh") else 1.0
        for ci, v in enumerate(vals[: len(cols)] if cols else vals):
            x = _num(v)
            if x is None:
                continue
            col = cols[ci] if ci < len(cols) else {}
            # page provenance: continuation pages hold the later rows; we keep the first page unless the
            # label is found on a later page's text (resolved in step_extract)
            out.append([f"LI:{doc}:{kind}:{ri}:{ci}", sid, fid, kind, str(label), metric, x,
                        unit, sc, _period(col), col.get("period_end", ""), col.get("fiscal_year"),
                        col.get("duration", ""), col.get("label", ""), doc, pages[0], ri])
    return st, out


async def _extract_all(jobs, usage_acc):
    return await asyncio.gather(*(_extract(d, k, ps, ts, usage_acc) for d, k, ps, ts in jobs))


def step_extract(pg: pd.DataFrame, only: list[str] | None = None, cache_only: bool = False) -> dict:
    t0 = time.perf_counter()
    jobs = []
    n_stmt_pages = 0
    # dev-slice filings first so prompt development can start while the rest extracts (order only)
    dev = set()
    dpath = config.RUNS_DIR / "ontology-dev-ids.json"
    if dpath.exists():
        ids = set(json.loads(dpath.read_text()))
        dev = {q["doc_name"] for q in data.questions() if q["qid"] in ids}
    groups = sorted(pg.groupby("doc_name"), key=lambda kv: (kv[0] not in dev, kv[0]))
    for d, g in groups:
        if only and d not in only:
            continue
        g = g.sort_values("page")
        texts = dict(zip(g.page, g.text))
        for kind, ps in locate(list(texts.items())).items():
            n_stmt_pages += len(ps)
            if cache_only and not (CACHE / f"{d}__{kind}.json").exists():
                continue
            jobs.append((d, kind, ps, [texts[p] for p in ps]))
    usage_acc: list = []
    results = asyncio.run(_extract_all(jobs, usage_acc))
    stmts, items = [], []
    for (d, kind, ps, ts), js in zip(jobs, results):
        if not js:
            continue
        st, rows = rows_from_extract(d, kind, js)
        if len(ps) > 1:  # assign rows to the continuation page where the label appears only there
            low = [t.lower() for t in ts]
            for r in rows:
                key = r[4].lower()[:30]
                if key and key not in low[0] and any(key in t for t in low[1:]):
                    r[15] = ps[1]
        stmts.append(st)
        items += rows
    con = rw()
    if only:
        for d in only:
            con.execute("DELETE FROM statement WHERE doc_name = ?", [d]); con.execute("DELETE FROM line_item WHERE doc_name = ?", [d])
    else:
        con.execute("DELETE FROM statement"); con.execute("DELETE FROM line_item")
    if stmts:
        sdf = pd.DataFrame(stmts)
        con.execute("INSERT INTO statement SELECT id, filing_id, kind, page, title, scale, doc_name FROM sdf")
    con.executemany("INSERT INTO line_item VALUES (" + ",".join("?" * 17) + ")", items)
    # quality check: balance sheet identity per filing/period
    chk = con.execute("""
        SELECT count(*) FILTER (WHERE abs(a.value - b.value) <= 0.005 * abs(a.value)), count(*)
        FROM line_item a JOIN line_item b USING (filing_id, period)
        WHERE a.normalized_metric = 'total_assets' AND b.normalized_metric = 'total_liabilities_and_equity'
    """).fetchone()
    con.close()
    tstats = step_tables(pg, only)
    return {"statement_jobs": len(jobs), "statement_pages": n_stmt_pages, "statements": len(stmts),
            "line_items": len(items),
            "line_items_normalized": sum(1 for r in items if r[5] != "other"),
            "extract_llm_calls": len(usage_acc),
            "extract_llm_prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage_acc),
            "extract_llm_completion_tokens": sum(u.get("completion_tokens", 0) for u in usage_acc),
            "balance_identity_ok": f"{chk[0]}/{chk[1]}",
            **tstats,
            "extract_wall_s": round(time.perf_counter() - t0, 1)}


# ================================================================== ingest: pipe tables (text corpora only)
_SEP = re.compile(r"^\|?\s*:?-{3,}")
_CELL_NUM = re.compile(r"^\(?-?\$?\s*\(?\d[\d,]*(\.\d+)?\)?\s*%?\)?$")
_YEARISH = re.compile(r"^(?:(?:fiscal|fy|year|years?\s+ended)?\s*'?(19|20)\d{2}[a-z]?|[a-z]{3,9}\.?\s+\d{1,2},?(<br>)?\s*(19|20)\d{2}|q[1-4].*)$", re.I)


def _cells(line: str) -> list[str]:
    t = line.strip()
    if t.startswith("|"):
        t = t[1:]
    if t.endswith("|"):
        t = t[:-1]
    return [c.strip().replace("<br>", " ").replace("&amp;", "&") for c in t.split("|")]


def cell_value(raw: str):
    s = raw.strip().replace("$", "").replace(",", "").replace(" ", "")
    if not s or s in ("-", "—", "–", "n/a", "N/A", "nm", "NM", "*"):
        return None
    neg = s.startswith("(") and s.endswith(")") or (s.startswith("(") and s.endswith(")%"))
    s = s.strip("()%")
    try:
        x = float(s)
    except ValueError:
        return None
    return -x if neg else x


def parse_tables(text: str) -> list[dict]:
    """Pipe tables on a page -> [{caption, cols[], rows: [(label, [raw cells])]}]."""
    lines = text.split("\n")
    out, i = [], 0
    while i < len(lines):
        if not lines[i].lstrip().startswith("|"):
            i += 1
            continue
        j = i
        while j < len(lines) and lines[j].lstrip().startswith("|"):
            j += 1
        block = [l for l in lines[i:j] if not _SEP.match(l.strip().lstrip("|").strip() and l.strip()) and not _SEP.match(l.strip())]
        pre = [l.strip("# ").strip() for l in lines[max(0, i - 6):i] if l.strip() and not l.lstrip().startswith("|")]
        caption = " ".join(pre[-2:])[-240:]
        rows = [_cells(l) for l in block]
        rows = [r for r in rows if any(r)]
        if len(rows) >= 2:
            ncol = max(len(r) for r in rows)
            head_parts: list[list[str]] = [[] for _ in range(ncol)]
            body = []
            for r in rows:
                r = r + [""] * (ncol - len(r))
                vals = r[1:]
                is_head = (not body) and (not r[0] or not any(cell_value(v) is not None for v in vals)
                                          or all(_YEARISH.match(v) or not v for v in vals))
                if is_head and any(vals):
                    for k, v in enumerate(r):
                        if v:
                            head_parts[k].append(v)
                    continue
                body.append((r[0], vals))
            cols = [" ".join(h) for h in head_parts[1:]]
            if body and any(cell_value(v) is not None for _, vals in body for v in vals):
                unit = re.search(r"\((?:dollars |\$ ?|amounts )?in (millions|thousands|billions)[^)]*\)|in (millions|thousands|billions)",
                                 (caption + " " + " ".join(h for hp in head_parts for h in hp)).lower())
                out.append({"caption": caption, "row_head": " ".join(head_parts[0]), "cols": cols, "rows": body,
                            "unit": (unit.group(1) or unit.group(2)) if unit else ""})
        i = j
    return out


def step_tables(pg: pd.DataFrame, only: list[str] | None = None) -> dict:
    """Deterministic pipe-table parse into fin_table / table_cell (no LLM). FinanceBench PDFs have no pipe
    tables, so this writes nothing there."""
    tabs, cells = [], []
    for d, g in pg.groupby("doc_name"):
        if only and d not in only:
            continue
        for p, t in zip(g.page, g.text):
            if "|" not in t or sum(1 for l in t.split("\n") if l.lstrip().startswith("|")) < 2:
                continue
            for k, tb in enumerate(parse_tables(t)):
                tid = f"TB:{d}:{int(p)}:{k}"
                cap = tb["caption"] + (f" [{tb['row_head']}]" if tb["row_head"] else "") + \
                    (f" (in {tb['unit']})" if tb["unit"] and tb["unit"] not in tb["caption"].lower() else "")
                tabs.append([tid, scope_id(d), d, int(p), cap.strip(), len(tb["rows"]), len(tb["cols"])])
                for ri, (label, vals) in enumerate(tb["rows"]):
                    for ci, raw in enumerate(vals):
                        x = cell_value(raw)
                        if x is None:
                            continue
                        cells.append([tid, scope_id(d), d, int(p), ri, label, ci,
                                      tb["cols"][ci] if ci < len(tb["cols"]) else f"col{ci}", x, raw])
    if not tabs and not only:
        return {}
    con = rw()
    if only:
        for d in only:
            con.execute("DELETE FROM fin_table WHERE doc_name = ?", [d]); con.execute("DELETE FROM table_cell WHERE doc_name = ?", [d])
    else:
        con.execute("DELETE FROM fin_table"); con.execute("DELETE FROM table_cell")
    if tabs:
        tdf = pd.DataFrame(tabs, columns=["id", "filing_id", "doc_name", "page", "caption", "n_rows", "n_cols"])
        con.execute("INSERT INTO fin_table SELECT * FROM tdf")
        cdf = pd.DataFrame(cells, columns=["table_id", "filing_id", "doc_name", "page", "row_idx", "row_label",
                                           "col_idx", "col_label", "value", "raw"])
        con.execute("INSERT INTO table_cell SELECT * FROM cdf")
    con.close()
    return {"tables": len(tabs), "table_cells": len(cells)}


# ================================================================== query: 1. entity resolution
def _companies() -> list[dict]:
    return ro().execute("SELECT id, name, aliases, ticker FROM company").df().to_dict("records")


def match_companies(question: str) -> list[dict]:
    hits = []
    for c in _companies():
        best = 0
        for a in c["aliases"]:
            a = a.strip()
            if len(a) <= 4:  # short aliases / tickers: case-sensitive, must contain an uppercase letter
                if not re.search(r"[A-Z]", a):
                    continue
                if re.search(rf"(?<![A-Za-z0-9]){re.escape(a)}(?![A-Za-z0-9])", question):
                    best = max(best, len(a))
            elif re.search(rf"(?<![a-z0-9]){re.escape(a.lower())}(?![a-z0-9])", question.lower()):
                best = max(best, len(a))
        if best:
            hits.append({**c, "_len": best})
    return sorted(hits, key=lambda c: -c["_len"])


RESOLVE_SYS = """You map a financial question to the filing record(s) that answer it.
Filing records are listed as: id | type | fiscal_year | quarter | filed_date.
Conventions:
- "FY2022" / "fiscal 2022" / "in 2022" (annual) -> the 10-K with that fiscal_year.
- Several years mentioned (e.g. "FY2019 - FY2021", "change from FY2021 to FY2022") -> the 10-K of the LATEST
  year mentioned (it contains the prior-year comparatives).
- A quarter ("Q2 of FY2023", "2022 Q2", "H1 FY2023") -> the 10-Q or earnings release for that quarter/year.
- Adjusted / non-GAAP figures, guidance, or quarterly results commentary -> an EARNINGS release if one exists
  for the period.
- An event on a specific date, a shareholder vote, an agreement, an executive appointment -> the 8-K nearest
  that date.
- No period mentioned -> the most recent filing that could contain the answer.
Return only JSON {"filings": [<best id>, <optional second id only if it is genuinely also needed>]}"""


_ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4}
LLM_ROUTE = re.compile(r"\b(adjusted|non[- ]?gaap|guidance|8-?k|dated|annual (general )?meeting|agm|vote[sd]?|"
                       r"announce[sd]?|appoint\w*|ceo|nominees?|press release|earnings release)\b", re.I)
EXPLICIT_10K = re.compile(r"\b10-?K for fiscal ((?:19|20)\d{2})\b", re.I)


def period_mentions(question: str) -> tuple[list[int], list[tuple[int, int | None]]]:
    """Deterministic parse of fiscal years and quarters mentioned in the question."""
    years = [int(y) for y in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", question)]
    years += [2000 + int(y) for y in re.findall(r"\bFY\s?'?(\d{2})\b", question)]
    quarters = []
    for q, y in re.findall(r"\bQ([1-4])\s*(?:of\s+)?(?:FY\s*)?['’]?\s*((?:19|20)?\d{2})?(?!\d)", question):
        quarters.append((int(q), (int(y) if len(y) == 4 else 2000 + int(y)) if y else None))
    for y, q in re.findall(r"((?:19|20)\d{2})\s*Q([1-4])", question):
        quarters.append((int(q), int(y)))
    for w in re.findall(r"\b(first|second|third|fourth)\s+(?:fiscal\s+)?quarter", question, re.I):
        quarters.append((_ORD[w.lower()], None))
    if re.search(r"\bH1\b|first half", question, re.I):
        quarters.append((2, None))
    return years, quarters


def rule_resolve(question: str, fil: list[tuple]) -> list[str]:
    """Unambiguous cases only; returns [] to defer to the LLM."""
    if (m := EXPLICIT_10K.search(question)):  # "In X's 10-K for fiscal 2016: ..." (DocFinQA prefix)
        c = [f[0] for f in fil if f[1] == "10K" and f[2] == int(m.group(1))]
        if len(c) == 1:
            return c
    if LLM_ROUTE.search(question):
        return []
    years, quarters = period_mentions(question)
    if quarters:
        q, y = quarters[0]
        y = y or (max(years) if years else None)
        c = [f[0] for f in fil if f[1] in ("10Q", "EARNINGS") and f[3] == q and (y is None or f[2] == y)]
        tenq = [x for x in c if x.endswith("10Q")]
        return tenq[:1] if len(tenq) == 1 else (c if len(c) == 1 else [])
    if years:
        c = [f[0] for f in fil if f[1] == "10K" and f[2] == max(years)]
        return c if len(c) == 1 else []
    return []


async def resolve(question: str, usage: Usage) -> dict:
    hits = match_companies(question)
    comp = hits[0] if hits else None
    if comp is None:
        names = "\n".join(f"{c['id']} | {c['name']} | {c['ticker']}" for c in _companies())
        out = await chat([{"role": "user", "content":
                           f"Which company is this question about?\n{question}\n\nCompanies:\n{names}\n"
                           'Return JSON {"company_id": "..."}'}],
                         tag=ANSWER_TAG, max_tokens=60, temperature=0.0)
        usage.add(out)
        try:
            cid = llm.parse_json(out["content"])["company_id"]
            comp = next(c for c in _companies() if c["id"] == cid)
        except Exception:
            return {"company": None, "filings": [], "entity": None, "scope": []}
    fil = ro().execute(
        "SELECT id, doc_type, fiscal_year, fiscal_quarter, filed_date FROM filing WHERE company_id = ? "
        "ORDER BY fiscal_year, coalesce(fiscal_quarter, 5), filed_date", [comp["id"]]).fetchall()
    lines = "\n".join(f"{f[0]} | {f[1]} | {f[2]} | {('Q' + str(f[3])) if f[3] else '-'} | {f[4] or '-'}" for f in fil)
    ids = [f[0] for f in fil]
    if len(ids) == 1:
        chosen = ids
    elif rule := rule_resolve(question, fil):
        chosen = rule
    else:
        out = await chat([{"role": "system", "content": RESOLVE_SYS},
                          {"role": "user", "content": f"Question: {question}\nCompany: {comp['name']}\nFilings:\n{lines}"}],
                         tag=ANSWER_TAG, max_tokens=400, temperature=0.0)
        usage.add(out)
        try:
            chosen = [f for f in llm.parse_json(out["content"]).get("filings", []) if f in ids][:2]
        except Exception:  # truncated / malformed JSON: take filing ids in the order they appear
            chosen = [f for f in dict.fromkeys(re.findall(r"F:[\w.\-]+", out["content"])) if f in ids][:2]
        if not chosen:
            chosen = [ids[-1]]
    return {"company": comp["name"], "company_id": comp["id"], "filings": chosen,
            "entity": comp["name"], "scope": chosen}


# ================================================================== query: 2. catalogue (value-free)
KIND_CODE = {"income": "I", "balance": "B", "cashflow": "C"}
EMPTY_CATALOGUE = "(no structured statements for this filing)"
MAX_TABLES, MAX_TABLE_ROWS = 4, 30
_STOP = set("the a an of in for to and or is was were what how much many by on at as from with its it this that "
            "which are be did does do company s total percent percentage change ratio between during year years "
            "fiscal 10 k".split())


def _toks(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z]+|(?:19|20)\d{2}", s.lower()) if t not in _STOP and len(t) > 1]


def relevant_tables(question: str, fid: str, k: int = MAX_TABLES) -> list[tuple]:
    """Tables of a filing ranked by keyword overlap (BM25-like, idf within the filing) between the question and
    caption + row labels + column labels. Returns [(table_id, page, caption)]."""
    c = ro()
    if not has_table(c, "fin_table"):
        return []
    tabs = c.execute("SELECT id, page, caption FROM fin_table WHERE filing_id = ?", [fid]).fetchall()
    if not tabs:
        return []
    lab = c.execute("SELECT table_id, string_agg(DISTINCT row_label, ' '), string_agg(DISTINCT col_label, ' ') "
                    "FROM table_cell WHERE filing_id = ? GROUP BY table_id", [fid]).fetchall()
    text = {t: f"{r or ''} {cl or ''}" for t, r, cl in lab}
    docs = {tid: Counter(_toks(f"{cap} {cap} {text.get(tid, '')}")) for tid, _, cap in tabs}
    df = Counter(t for d in docs.values() for t in d)
    n = len(docs)
    q = set(_toks(question))
    years = set(re.findall(r"(?:19|20)\d{2}", question))
    scored = []
    for tid, page, cap in tabs:
        d = docs[tid]
        length = sum(d.values()) or 1
        s = sum(math.log(1 + n / df[t]) * d[t] * 2.2 / (d[t] + 1.2 * (0.25 + 0.75 * length / 60)) for t in q if t in d)
        s += 0.5 * sum(1 for y in years if y in text.get(tid, ""))
        if s > 0:
            scored.append((s, tid, page, cap))
    scored.sort(key=lambda x: -x[0])
    return [(tid, page, cap) for _, tid, page, cap in scored[:k]]


def _table_catalogue(question: str, fid: str, pre: str, keymap: dict) -> list[str]:
    c = ro()
    parts = []
    for n, (tid, page, cap) in enumerate(relevant_tables(question, fid), start=1):
        cols = c.execute("SELECT col_idx, any_value(col_label) FROM table_cell WHERE table_id = ? GROUP BY col_idx "
                         "ORDER BY col_idx", [tid]).fetchall()
        rows = c.execute("SELECT row_idx, any_value(row_label), list(col_idx ORDER BY col_idx) FROM table_cell "
                         "WHERE table_id = ? GROUP BY row_idx ORDER BY row_idx", [tid]).fetchall()
        tkey = f"{pre}T{n}"
        head = f"## {tkey} = table '{cap}' ({fid[2:]} p.{page})\n   columns: " + \
            "; ".join(f"c{ci} = {lab}" for ci, lab in cols)
        lines = []
        for ri, label, cis in rows[:MAX_TABLE_ROWS]:
            key = f"{tkey}r{ri}"
            keymap[key] = ("table", tid, ri, {f"c{ci}": ci for ci in cis})
            miss = "" if len(cis) == len(cols) else " cols:" + ",".join(f"c{ci}" for ci in cis)
            lines.append(f"   {key}: {label or '(no label)'}{miss}")
        if len(rows) > MAX_TABLE_ROWS:
            lines.append(f"   ... {len(rows) - MAX_TABLE_ROWS} more rows")
        parts.append(head + "\n" + "\n".join(lines))
    return parts


def catalogue(filing_ids: list[str], question: str = "") -> tuple[str, dict]:
    """Text catalogue of statements for the plan prompt + key -> (filing, kind, row_idx, {col: period}) map.
    On text corpora the best-matching pipe tables of the filing are appended as T<n> blocks."""
    c = ro()
    parts, keymap = [], {}
    for fi, fid in enumerate(filing_ids):
        pre = "" if fi == 0 else f"{fi + 1}:"
        sts = c.execute("SELECT id, kind, page, title, scale FROM statement WHERE filing_id = ? ORDER BY kind", [fid]).fetchall()
        for sid, kind, page, title, scale in sts:
            cols = c.execute(
                "SELECT DISTINCT period, column_label, duration, fiscal_year FROM line_item WHERE statement_id = ? "
                "ORDER BY fiscal_year DESC NULLS LAST, period DESC", [sid]).fetchall()
            colcode = {p: f"c{i}" for i, (p, *_r) in enumerate(cols)}
            head = f"## {pre}{KIND_CODE[kind]} = {kind} statement '{title}' ({fid[2:]} p.{page}, amounts in {scale})\n   columns: " + \
                "; ".join(f"{colcode[p]} = {p} [{lab}, {dur}, FY{fy}]" for p, lab, dur, fy in cols)
            rows = c.execute(
                "SELECT row_idx, any_value(label), any_value(normalized_metric), any_value(unit), list(DISTINCT period) "
                "FROM line_item WHERE statement_id = ? GROUP BY row_idx ORDER BY row_idx", [sid]).fetchall()
            lines = []
            for ri, label, metric, unit, periods in rows:
                key = f"{pre}{KIND_CODE[kind]}{ri}"
                keymap[key] = (fid, kind, ri, {colcode[p]: p for p in periods})
                tag = f" [{metric}]" if metric != "other" else ""
                u = f" ({unit})" if unit not in ("$", None) else ""
                miss = "" if len(periods) == len(cols) else " cols:" + ",".join(sorted(colcode[p] for p in periods))
                lines.append(f"   {key}: {label}{tag}{u}{miss}")
            parts.append(head + "\n" + "\n".join(lines))
        if question:
            parts += _table_catalogue(question, fid, pre, keymap)
    return "\n".join(parts), keymap


PLAN_SYS = """You are the query planner of an ontology-grounded financial QA system. You do NOT answer the question.
You get the question and a catalogue of the structured financial-statement records (line items) of the resolved
filing: each row key (e.g. B12) is a line item, each column code (c0, c1, ...) a period. Values are hidden; they
will be fetched by SQL and any arithmetic is computed by Python from your plan.

Return JSON:
{
 "needs_numbers": true|false,
 "vars": {"<name>": "<rowkey>.<col>", ...},      // every input value needed, e.g. "rev_fy19": "I0.c0"
 "calc": {"<name>": "<python arithmetic over var names>", ...},  // e.g. "fat": "rev_fy19 / ((ppe_fy19 + ppe_fy18) / 2)"
 "search": ["<query>", ...]                        // 1-3 short text-search queries for narrative evidence
}
Rules:
- Use the question's own definition of a metric when it gives one. Otherwise use standard definitions
  (e.g. quick ratio = (cash + short-term investments + receivables) / current liabilities; working capital =
  current assets - current liabilities; gross margin = (revenue - cogs) / revenue; operating margin = operating
  income / revenue; EBITDA = operating income + D&A from the cash flow statement; FCF = CFO - capex;
  capex/revenue; DPO, DSO, DIO, inventory turnover with averages when asked; payout ratio = dividends paid /
  net income attributable; ROA = net income / average total assets; effective tax rate = income tax / pretax income).
- "average between FY2018 and FY2019" = (x_FY2018 + x_FY2019) / 2. Multiply by 100 only when the result is
  asked for as a percent / margin / growth rate; ratios (turnover, coverage, current ratio, X/Y ratio) stay plain.
- Pick each input by its meaning, not position: e.g. cash from operations is the "net cash provided by operating
  activities" total, not net income at the top of the cash flow statement.
- Cash-flow outflows such as capex / dividends are stored negative; use abs() when the definition needs a
  positive amount.
- For a trend / "improving" / "consistent" / "change" question compute the metric for every available period.
- Pick columns of the right duration (12M for annual, 3M for a quarter unless year-to-date is asked).
- Use only row keys and column codes that exist in the catalogue. If no structured record fits (narrative,
  segment, non-GAAP, notes, guidance, events), set needs_numbers false, leave vars/calc empty and rely on search.
- search queries: specific phrases likely to appear in the filing text (e.g. "segment net sales organic growth",
  "legal proceedings litigation", "registered pursuant to Section 12(b)"). Always give at least one."""

# Appended to PLAN_SYS only when the catalogue holds T<n> table blocks (text corpora); FinanceBench never sees it.
PLAN_TABLES = """
The catalogue may also list other tables of the filing (notes, MD&A, selected data) as T<n> blocks: row keys
like T2r5, column codes c0, c1, ... as printed in that table (values as printed, in the table's own units).
Reference them exactly like statement rows, e.g. "rev_2016": "T2r5.c0". Prefer the table whose caption and row
labels match the question's wording; a primary statement row only when it is the same figure."""


def plan_system(catalogue_text: str) -> str:
    return PLAN_SYS + (PLAN_TABLES if re.search(r"^## (?:\d:)?T\d+ = table", catalogue_text, re.M) else "")


# ================================================================== query: 3. structured retrieval + compute
REF_IN_EXPR = re.compile(r"(?<![\w.])((?:\d:)?(?:[IBC]\d+|T\d+r\d+))\.(c\d+)\b")


def execute(p: dict, keymap: dict) -> dict:
    refs, errors = {}, []
    variables = dict(p.get("vars") or {})
    calc = {}
    for name, expr in (p.get("calc") or {}).items():  # tolerate row refs used directly inside calc
        def sub(m):
            v = f"{m.group(1).replace(':', '_')}_{m.group(2)}"
            variables.setdefault(v, f"{m.group(1)}.{m.group(2)}")
            return v
        calc[name] = REF_IN_EXPR.sub(sub, str(expr))
    p = {**p, "calc": calc}
    trefs = {}
    for name, ref in variables.items():
        m = re.fullmatch(r"\s*((?:\d:)?(?:[IBC]\d+|T\d+r\d+))\.(c\d+)\s*", str(ref))
        if not m or m.group(1) not in keymap or m.group(2) not in keymap[m.group(1)][3]:
            errors.append(f"bad ref {name}={ref}")
            continue
        ent = keymap[m.group(1)]
        if ent[0] == "table":
            trefs[name] = (ent[1], ent[2], ent[3][m.group(2)])
            continue
        fid, kind, ri, cols = ent
        refs[name] = (fid, kind, ri, cols[m.group(2)])
    sql = None
    records = {}
    if refs:
        conds = " OR ".join(
            f"(filing_id = '{fid}' AND kind = '{kind}' AND row_idx = {ri} AND period = '{per}')"
            for fid, kind, ri, per in refs.values())
        sql = ("SELECT id, label, normalized_metric, value, unit, scale, period, column_label, doc_name, page, "
               "filing_id, kind, row_idx FROM line_item WHERE " + conds)
        rows = ro().execute(sql).fetchall()
        idx = {(r[10], r[11], r[12], r[6]): r for r in rows}
        for name, key in refs.items():
            r = idx.get(key)
            if r:
                records[name] = {"id": r[0], "label": r[1], "metric": r[2], "value": r[3], "unit": r[4],
                                 "scale": r[5], "period": r[6], "column": r[7], "doc": r[8], "page": r[9]}
    if trefs:
        conds = " OR ".join(f"(table_id = '{tid}' AND row_idx = {ri} AND col_idx = {ci})"
                            for tid, ri, ci in trefs.values())
        tsql = ("SELECT table_id, row_idx, col_idx, row_label, col_label, value, raw, doc_name, page "
                "FROM table_cell WHERE " + conds)
        sql = (sql + ";\n" if sql else "") + tsql
        idx = {(r[0], r[1], r[2]): r for r in ro().execute(tsql).fetchall()}
        for name, key in trefs.items():
            r = idx.get(key)
            if r:
                records[name] = {"id": f"TC:{r[0][3:]}:{r[1]}:{r[2]}", "label": r[3], "metric": "table",
                                 "value": r[5], "unit": "table", "scale": 1.0, "period": r[4],
                                 "column": f"as printed: {r[6]}", "doc": r[7], "page": r[8]}
    env = {k: v["value"] for k, v in records.items()}
    results = {}
    for name, expr in (p.get("calc") or {}).items():
        try:
            val = safe_eval(str(expr), env)
            env[name] = val
            results[name] = {"expr": expr, "value": val}
        except Exception as e:
            errors.append(f"calc {name}: {type(e).__name__} {e}"[:160])
    return {"sql": sql, "records": records, "results": results, "errors": errors}


# ================================================================== query: 5. grounded generation
ANSWER_SYS = """You answer financial questions using ONLY the evidence given: structured ontology records (line
items with IDs) with values computed by the system, and text passages from the resolved filing.
- Use the computed results as given; do not redo arithmetic differently. Convert units only if the question asks
  (amounts are shown with their scale). Round as the question requests.
- For judgement questions (e.g. "is it capital intensive", "improving margin", "healthy liquidity") give a clear
  yes/no first, then the supporting figures. If a metric is not meaningful for this kind of company (e.g. gross
  margin for a bank), say so and why.
- Read tables carefully: match each number to its row label and column header. When comparing items, list
  each item's value, then pick the one that satisfies the question.
- If the evidence does not contain the answer, say what is missing; do not invent figures.
- Cite record IDs like [LI:...] and pages like (<doc_name> p.N).
Format: first line "Answer: <direct answer>", then 1-4 short lines of support with citations."""


def _fmt_num(x: float) -> str:
    if x is None:
        return "null"
    if abs(x) >= 100:
        return f"{x:,.2f}"
    return f"{x:.6g}"


SCALE_NAME = {1.0: "", 1e3: " thousand", 1e6: " million", 1e9: " billion"}


def evidence_block(question: str, res: dict, ex: dict, passages: list[dict]) -> str:
    lines = [f"Question: {question}", f"Resolved: company={res.get('company')} filings={res.get('filings')}", ""]
    if ex["records"]:
        lines.append("Structured records (SQL over ontology.line_item):")
        for name, r in ex["records"].items():
            unit = {"$": "USD" + SCALE_NAME.get(r["scale"], f" x{r['scale']:g}"), "$/sh": "USD per share",
                    "sh": "shares" + SCALE_NAME.get(r["scale"], ""), "%": "%",
                    "table": "(units as in the table caption)"}.get(r["unit"], r["unit"])
            lines.append(f"- {name} = {_fmt_num(r['value'])} {unit} | {r['label']} | {r['period']} ({r['column']}) "
                         f"| [{r['id']}] ({r['doc']} p.{r['page']})")
    if ex["results"]:
        lines.append("Computed by the system (exact):")
        for name, r in ex["results"].items():
            lines.append(f"- {name} = {r['expr']} = {_fmt_num(r['value'])}")
    if ex["errors"]:
        lines.append("Plan issues: " + "; ".join(ex["errors"]))
    lines.append("")
    lines.append("Passages (text search scoped to the resolved filing):")
    for p in passages:
        lines.append(f"--- ({p['doc']} p.{p['page']})\n{p['text']}")
    return "\n".join(lines)
