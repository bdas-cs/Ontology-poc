"""Build the ontology (DuckDB) from the parsed corpus.

Steps (each idempotent; `all` runs them in order):
  base     Company / Filing / Page / Section objects from the system of record (doc_info) + page text
  extract  locate statement pages (keyword heuristics, poc.ontology.locate) and LLM-extract every
           line item x period column into LineItem rows (ledger tag "ontology-ingest")
  embed    chunk every page and embed with Qwen3-Embedding-0.6B (MPS) for scoped dense search
  fts      BM25 index over chunks for scoped keyword search

  uv run python -m poc.ontology.ingest all
Stats -> runs/ingest-ontology.json
"""
import argparse
import asyncio
import json
import re
import time

import pandas as pd

from poc import config, data, llm
from poc.ontology import schema
from poc.ontology.locate import locate

TAG = "ontology-ingest"
CACHE = config.DB_DIR / "extract_cache"
STATS_PATH = config.RUNS_DIR / "ingest-ontology.json"
SCALES = {"units": 1.0, "ones": 1.0, "thousands": 1e3, "millions": 1e6, "billions": 1e9}


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _squeeze(t: str) -> str:
    t = re.sub(r"[ \t]{3,}", "  ", t)
    return re.sub(r"\n\s*\n+", "\n", t)


def _con():
    for i in range(60):  # other ingest steps may hold the write lock briefly
        try:
            return schema.connect()
        except Exception as e:  # duckdb.IOException on lock conflict
            if "lock" not in str(e).lower():
                raise
            time.sleep(5)
    return schema.connect()


def _load_stats() -> dict:
    return json.loads(STATS_PATH.read_text()) if STATS_PATH.exists() else {}


def _save_stats(upd: dict) -> None:
    s = _load_stats()
    s.update(upd)
    STATS_PATH.write_text(json.dumps(s, indent=2))


# ---------------------------------------------------------------- base objects
ALIAS_PROMPT = """For each company below give its primary US stock ticker and the short names / former names /
abbreviations an analyst might use for it in a question (e.g. "JnJ", "JPM", "AMEX", "Square").
Return JSON: {"<company>": {"ticker": "...", "aliases": ["...", ...]}, ...}
Companies:
"""


async def _company_aliases(names: list[str]) -> tuple[dict, dict]:
    out = await llm.chat([{"role": "user", "content": ALIAS_PROMPT + "\n".join(names)}],
                         tag=TAG, max_tokens=2500, temperature=0.0)
    return llm.parse_json(out["content"]), out["usage"]


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


def step_base() -> dict:
    t0 = time.perf_counter()
    info = data.doc_info()
    docs = data.corpus_docs()
    pg = data.pages()
    companies = sorted({info[d]["company"] for d in docs})
    aliases, usage = asyncio.run(_company_aliases(companies))
    comp_rows = []
    for c in companies:
        a = aliases.get(c, {}) if isinstance(aliases, dict) else {}
        prefixes = {d.split("_20")[0].replace("_", " ") for d in docs if info[d]["company"] == c}
        al = sorted({c, c.replace("&", "and"), *prefixes, *(a.get("aliases") or []), a.get("ticker") or c} - {""})
        sector = next(info[d].get("gics_sector") for d in docs if info[d]["company"] == c)
        comp_rows.append({"id": "C:" + _slug(c), "name": c, "aliases": al, "ticker": a.get("ticker"), "sector": sector})
    con = _con()
    con.execute("DELETE FROM company")
    con.executemany("INSERT INTO company VALUES (?,?,?,?,?)",
                    [[r["id"], r["name"], r["aliases"], r["ticker"], r["sector"]] for r in comp_rows])
    fil, secs = [], []
    con.execute("DELETE FROM filing"); con.execute("DELETE FROM page"); con.execute("DELETE FROM section")
    for d in docs:
        i = info[d]
        q = re.search(r"\d{4}Q(\d)", d)
        dt = re.search(r"dated[-_](\d{4}-\d{2}-\d{2})", d)
        g = pg[pg.doc_name == d].sort_values("page")
        fid = "F:" + d
        fil.append([fid, "C:" + _slug(i["company"]), d, i["doc_type"].upper(), int(i["doc_period"]),
                    int(q.group(1)) if q else None, dt.group(1) if dt else None, len(g)])
        secs += _sections(d, fid, list(zip(g.page, g.text)))
    con.executemany("INSERT INTO filing VALUES (?,?,?,?,?,?,?,?)", fil)
    pdf = pg.assign(filing_id="F:" + pg.doc_name)[["filing_id", "doc_name", "page", "text"]]
    con.execute("INSERT INTO page SELECT * FROM pdf")
    if secs:
        sdf = pd.DataFrame(secs)
        con.execute("INSERT INTO section SELECT id, filing_id, title, page_start, page_end, doc_name FROM sdf")
    st = {"companies": len(comp_rows), "filings": len(fil), "pages": len(pdf), "sections": len(secs),
          "base_llm_prompt_tokens": usage.get("prompt_tokens", 0),
          "base_llm_completion_tokens": usage.get("completion_tokens", 0),
          "base_wall_s": round(time.perf_counter() - t0, 1)}
    con.close()
    return st


# ---------------------------------------------------------------- statement extraction
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
    body = "\n\n".join(f"[page {p}]\n{_squeeze(t)}" for p, t in zip(pages, texts))
    msgs = [{"role": "system", "content": EXTRACT_SYS.replace("METRICS", ", ".join(schema.METRICS))},
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
    fid, sid = "F:" + doc, f"ST:{doc}:{kind}"
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
        metric = metric if metric in schema.METRICS else "other"
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


def step_extract(only: list[str] | None = None, cache_only: bool = False) -> dict:
    t0 = time.perf_counter()
    pg = data.pages()
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
    con = _con()
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
    return {"statement_jobs": len(jobs), "statement_pages": n_stmt_pages, "statements": len(stmts),
            "line_items": len(items),
            "line_items_normalized": sum(1 for r in items if r[5] != "other"),
            "extract_llm_calls": len(usage_acc),
            "extract_llm_prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage_acc),
            "extract_llm_completion_tokens": sum(u.get("completion_tokens", 0) for u in usage_acc),
            "balance_identity_ok": f"{chk[0]}/{chk[1]}",
            "extract_wall_s": round(time.perf_counter() - t0, 1)}


# ---------------------------------------------------------------- chunks / embeddings / FTS
def _chunks(text: str, size: int = 1400, overlap: int = 200) -> list[str]:
    t = _squeeze(text).strip()
    if not t:
        return []
    out, i = [], 0
    while i < len(t):
        out.append(t[i : i + size])
        if i + size >= len(t):
            break
        i += size - overlap
    return out


_EMB = None


def embedder():
    global _EMB
    if _EMB is None:
        from sentence_transformers import SentenceTransformer
        _EMB = SentenceTransformer(config.EMBED_MODEL, device=config.torch_device())
        _EMB.max_seq_length = 512
    return _EMB


def step_embed() -> dict:
    """Chunk vectors for scoped dense search. Reuses the Phase-1 LanceDB table (same embedding model over
    the same pages) when it exists, so both arms share identical text vectors; otherwise embeds locally."""
    t0 = time.perf_counter()
    lpath = config.DB_DIR / "lancedb"
    try:
        import lancedb
        tbl = lancedb.connect(str(lpath)).open_table("chunks")
        src = tbl.to_pandas()
        df = pd.DataFrame({"id": "K:" + src.doc_name + ":" + src.chunk_id.astype(str), "filing_id": "F:" + src.doc_name,
                           "doc_name": src.doc_name, "page": src.page_start.astype(int), "page_end": src.page_end.astype(int),
                           "text": src.text})
        df["emb"] = [v.astype("float32").tolist() for v in src.vector]
        origin = "reused lancedb:chunks (Phase-1 index)"
    except Exception as e:
        print("lancedb reuse failed, embedding locally:", e)
        pg = data.pages()
        rows = []
        for d, p, t in zip(pg.doc_name, pg.page, pg.text):
            for j, c in enumerate(_chunks(t)):
                rows.append((f"K:{d}:{p}:{j}", "F:" + d, d, int(p), int(p), c))
        embs = embedder().encode([r[5] for r in rows], batch_size=32, normalize_embeddings=True, show_progress_bar=True)
        df = pd.DataFrame(rows, columns=["id", "filing_id", "doc_name", "page", "page_end", "text"])
        df["emb"] = [e.astype("float32").tolist() for e in embs]
        origin = "embedded locally"
    con = _con()
    con.execute("DROP TABLE IF EXISTS chunk")
    con.execute(schema.DDL)
    con.execute("INSERT INTO chunk SELECT id, filing_id, doc_name, page, page_end, text, emb::FLOAT[1024] FROM df")
    con.close()
    return {"chunks": len(df), "chunk_vectors": origin, "embed_wall_s": round(time.perf_counter() - t0, 1)}


def step_fts() -> dict:
    t0 = time.perf_counter()
    con = _con()
    con.execute("INSTALL fts; LOAD fts")
    con.execute("PRAGMA create_fts_index('chunk', 'id', 'text', overwrite=1)")
    con.close()
    return {"fts_wall_s": round(time.perf_counter() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["base", "extract", "load", "embed", "fts", "all"])
    ap.add_argument("--only", nargs="*", help="restrict extract to these doc_names (dev)")
    a = ap.parse_args()
    steps = ["base", "extract", "embed", "fts"] if a.step == "all" else [a.step]
    for s in steps:
        print(f"== {s}")
        st = {"base": step_base, "extract": lambda: step_extract(a.only),
              "load": lambda: step_extract(cache_only=True),
              "embed": step_embed, "fts": step_fts}[s]()
        print(json.dumps(st, indent=1))
        if not a.only and s != "load":
            _save_stats(st)
    if not a.only:
        s = _load_stats()
        s["total_wall_s"] = round(sum(v for k, v in s.items() if k.endswith("_wall_s") and k != "total_wall_s"), 1)
        s["total_llm_prompt_tokens"] = s.get("base_llm_prompt_tokens", 0) + s.get("extract_llm_prompt_tokens", 0)
        s["total_llm_completion_tokens"] = s.get("base_llm_completion_tokens", 0) + s.get("extract_llm_completion_tokens", 0)
        STATS_PATH.write_text(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
