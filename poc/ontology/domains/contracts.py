"""Contracts domain plugin (CUAD): commercial contracts -> contract, party, clause, key dates, amounts, law.

Object types (every record carries provenance: doc_name + page):

  contract       (id, doc_name, title, label, filer, contract_type, filing_date, n_pages, aliases[], printed_title)
  party          (id, contract_id, name, role, doc_name, page)
  clause         (id, contract_id, category, text, doc_name, page)      -- category from CLAUSE_TYPES
  key_date       (id, contract_id, kind, value, raw, doc_name, page)    -- agreement / effective / expiration /
                                                                          renewal / notice_to_terminate_renewal
  amount         (id, contract_id, kind, value, currency, raw, doc_name, page)
  governing_law  (id, contract_id, jurisdiction, raw, doc_name, page)

Scope id of a document = Contract.id = "CT:<doc_name>". Contract rows come from the dataset's doc_info (filer,
contract type, filing date — the system of record); everything else is extracted by the local LLM per contract
in page windows of <= WINDOW_CHARS (bounded tokens; capped at MAX_WINDOWS per contract), cached per window in
db/extract_cache/<doc>__w<k>.json. Quotes are located back to their page by text match.

The clause vocabulary is a contract-law taxonomy (the CUAD categories — a standard commercial-contract review
checklist); no CUAD annotations are used at ingest or query time.

Query path: resolve the contract from the question (filer / type / filing-date match, LLM fallback over the
contract list) -> planner picks clause categories + fields from a value-free catalogue (which categories have
records, how many, on which pages) -> SQL -> text search scoped to that contract -> answer quoting clause text.
"""
import asyncio
import json
import re
import time

import pandas as pd

from poc import config, llm
from poc.ontology.runtime import INGEST_TAG, ANSWER_TAG, Usage, chat, ro, rw

NAME = "contracts"
TAG = INGEST_TAG
CACHE = config.DB_DIR / "extract_cache"
WINDOW_CHARS = 16000
MAX_WINDOWS = 12
SEARCH_K = 6

DDL = """
CREATE TABLE IF NOT EXISTS contract (
  id VARCHAR PRIMARY KEY, doc_name VARCHAR, title VARCHAR, label VARCHAR, filer VARCHAR, contract_type VARCHAR,
  filing_date VARCHAR, n_pages INTEGER, aliases VARCHAR[], printed_title VARCHAR
);
CREATE TABLE IF NOT EXISTS party (
  id VARCHAR PRIMARY KEY, contract_id VARCHAR, name VARCHAR, role VARCHAR, doc_name VARCHAR, page INTEGER
);
CREATE TABLE IF NOT EXISTS clause (
  id VARCHAR PRIMARY KEY, contract_id VARCHAR, category VARCHAR, text VARCHAR, doc_name VARCHAR, page INTEGER
);
CREATE TABLE IF NOT EXISTS key_date (
  id VARCHAR PRIMARY KEY, contract_id VARCHAR, kind VARCHAR, value VARCHAR, raw VARCHAR, doc_name VARCHAR, page INTEGER
);
CREATE TABLE IF NOT EXISTS amount (
  id VARCHAR PRIMARY KEY, contract_id VARCHAR, kind VARCHAR, value DOUBLE, currency VARCHAR, raw VARCHAR,
  doc_name VARCHAR, page INTEGER
);
CREATE TABLE IF NOT EXISTS governing_law (
  id VARCHAR PRIMARY KEY, contract_id VARCHAR, jurisdiction VARCHAR, raw VARCHAR, doc_name VARCHAR, page INTEGER
);
"""

CLAUSE_TYPES = {
    "Most Favored Nation": "a party must get terms at least as good as any third party",
    "Competitive Restriction Exception": "exceptions / carve-outs to non-compete, exclusivity or no-solicit",
    "Non-Compete": "restriction on competing with the counterparty or operating in a geography / business",
    "Exclusivity": "exclusive dealing; restriction on buying from / selling to / licensing others",
    "No-Solicit Of Customers": "restriction on soliciting or contracting with the counterparty's customers / partners",
    "No-Solicit Of Employees": "restriction on soliciting or hiring the counterparty's employees",
    "Non-Disparagement": "obligation not to disparage the counterparty",
    "Termination For Convenience": "right to terminate without cause / at will by giving notice",
    "Rofr/Rofo/Rofn": "right of first refusal, first offer or first negotiation",
    "Change Of Control": "termination right, consent or notice on a change of control, merger or asset sale",
    "Anti-Assignment": "consent or notice required to assign the contract",
    "Revenue/Profit Sharing": "sharing of revenue or profit, royalties, revenue share",
    "Price Restrictions": "restriction on raising or reducing prices",
    "Minimum Commitment": "minimum order size, purchase amount or units per period",
    "Volume Restriction": "fee increase or consent needed when use exceeds a threshold",
    "Ip Ownership Assignment": "IP created by one party becomes the property of the counterparty",
    "Joint Ip Ownership": "joint or shared ownership of IP",
    "License Grant": "a license granted by one party to the other",
    "Non-Transferable License": "license may not be transferred / sublicensed",
    "Affiliate License-Licensor": "license grant by the licensor's affiliates / covering their IP",
    "Affiliate License-Licensee": "license extends to the licensee's affiliates",
    "Unlimited/All-You-Can-Eat-License": "unlimited, enterprise-wide, all-you-can-eat license",
    "Irrevocable Or Perpetual License": "irrevocable or perpetual license",
    "Source Code Escrow": "source code deposited into escrow",
    "Post-Termination Services": "obligations after termination / expiration (transition, wind-down, payments, last-buy)",
    "Audit Rights": "right to audit books, records or facilities",
    "Uncapped Liability": "liability excluded from any cap / unlimited for certain breaches",
    "Cap On Liability": "cap or limitation on liability",
    "Liquidated Damages": "liquidated damages, break-up or termination fee",
    "Warranty Duration": "duration of a warranty",
    "Insurance": "requirement to maintain insurance",
    "Covenant Not To Sue": "covenant not to sue / not to challenge the counterparty's IP",
    "Third Party Beneficiary": "non-party beneficiary of the contract",
}
DATE_KINDS = ["agreement", "effective", "expiration", "renewal", "notice_to_terminate_renewal"]
FIELDS = ["title", "parties", "dates", "governing_law", "amounts"]


def scope_id(doc_name: str) -> str:
    return "CT:" + doc_name


def doc_of(scope: str) -> str:
    return scope[3:]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


# ================================================================== ingest: base
def step_base(docs: list[str], info: dict, pg: pd.DataFrame) -> dict:
    rows = []
    for d in docs:
        i = info[d]
        n = int((pg.doc_name == d).sum())
        al = sorted({x for x in (i.get("filer"), i.get("label"), i.get("title"), i.get("company")) if x})
        rows.append([scope_id(d), d, i.get("title", d), i.get("label", d), i.get("filer"), i.get("contract_type"),
                     i.get("filing_date"), n, al, None])
    con = rw()
    con.execute("DELETE FROM contract")
    con.executemany("INSERT INTO contract VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    con.close()
    return {"contracts": len(rows), "base_llm_prompt_tokens": 0, "base_llm_completion_tokens": 0}


# ================================================================== ingest: LLM extraction per window
EXTRACT_SYS = """You extract structured records from one part of a commercial contract. Be exact and conservative:
include only what this text explicitly states.
Return only JSON:
{
 "title": "<the contract's own title as printed, only if this part shows it, else null>",
 "parties": [{"name": "<legal name>", "role": "<defined term / role, e.g. 'Distributor', 'Licensee'>"}],
 "dates": [{"kind": "agreement|effective|expiration|renewal|notice_to_terminate_renewal",
            "value": "<YYYY-MM-DD if a calendar date, else the term as stated, e.g. '5 years from the Effective Date', 'successive 1-year terms', '90 days'>",
            "quote": "<verbatim sentence>"}],
 "governing_law": [{"jurisdiction": "<state / country>", "quote": "<verbatim sentence>"}],
 "amounts": [{"kind": "<what it is, e.g. 'minimum purchase', 'royalty rate', 'license fee', 'liability cap', 'insurance coverage'>",
              "value": <number or null>, "currency": "<USD, %, units, ...>", "quote": "<verbatim sentence>"}],
 "clauses": [{"category": "<one of the categories below>", "quote": "<verbatim operative sentence(s), <= 80 words>"}]
}
Clause categories (use the exact name):
CATEGORIES
Rules:
- Quotes are copied verbatim from the text (you may cut with "..." but do not paraphrase).
- One sentence may appear under several categories when it fits each (e.g. an exclusive license is both
  "License Grant" and "Exclusivity").
- Omit a category entirely when this text has no such clause. Empty lists are fine."""


def _windows(pages: list[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    out, cur, n = [], [], 0
    for p, t in pages:
        if cur and n + len(t) > WINDOW_CHARS:
            out.append(cur)
            cur, n = [], 0
        cur.append((p, t))
        n += len(t)
    if cur:
        out.append(cur)
    return out[:MAX_WINDOWS]


async def _extract_window(doc: str, k: int, win: list[tuple[int, str]], usage_acc: list) -> dict | None:
    CACHE.mkdir(exist_ok=True)
    cpath = CACHE / f"{doc}__w{k}.json"
    if cpath.exists():
        return json.loads(cpath.read_text())
    body = "\n\n".join(f"[page {p}]\n{t}" for p, t in win)
    cats = "\n".join(f"- {c}: {d}" for c, d in CLAUSE_TYPES.items())
    msgs = [{"role": "system", "content": EXTRACT_SYS.replace("CATEGORIES", cats)},
            {"role": "user", "content": f"Contract: {doc} (part {k + 1})\n\n{body}"}]
    for attempt in range(2):
        out = await llm.chat(msgs, tag=TAG, max_tokens=4000, temperature=0.0)
        usage_acc.append(out["usage"])
        try:
            js = llm.parse_json(out["content"])
            if isinstance(js, dict):
                js["_pages"] = [p for p, _ in win]
                cpath.write_text(json.dumps(js))
                return js
        except Exception:
            pass
    print("extract failed", doc, k)
    return None


def _locate(quote: str, win: dict[int, str], default: int) -> int:
    q = _norm(quote.split("...")[0])[:60]
    if len(q) >= 12:
        for p, t in win.items():
            if q in _norm(t):
                return p
    return default


def _records(doc: str, k: int, js: dict, texts: dict[int, str]) -> dict[str, list]:
    cid = scope_id(doc)
    pages = js.get("_pages") or [min(texts)]
    win = {p: texts.get(p, "") for p in pages}
    p0 = pages[0]
    out = {"party": [], "clause": [], "key_date": [], "amount": [], "governing_law": [], "title": []}

    def lst(key):
        v = js.get(key) or []
        return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []

    for i, x in enumerate(lst("parties")):
        if x.get("name"):
            out["party"].append([f"PA:{doc}:{k}:{i}", cid, str(x["name"])[:200], str(x.get("role") or "")[:120], doc,
                                 _locate(str(x["name"]), win, p0)])
    for i, x in enumerate(lst("clauses")):
        cat, q = x.get("category"), str(x.get("quote") or "").strip()
        if cat in CLAUSE_TYPES and q:
            out["clause"].append([f"CL:{doc}:{k}:{i}", cid, cat, q[:1500], doc, _locate(q, win, p0)])
    for i, x in enumerate(lst("dates")):
        kind = str(x.get("kind") or "").strip().lower()
        if kind in DATE_KINDS and (x.get("value") or x.get("quote")):
            q = str(x.get("quote") or "")
            out["key_date"].append([f"KD:{doc}:{k}:{i}", cid, kind, str(x.get("value") or "")[:200], q[:1000], doc,
                                    _locate(q, win, p0)])
    for i, x in enumerate(lst("amounts")):
        q = str(x.get("quote") or "")
        try:
            v = float(x["value"]) if x.get("value") is not None else None
        except (TypeError, ValueError):
            v = None
        out["amount"].append([f"AM:{doc}:{k}:{i}", cid, str(x.get("kind") or "")[:120], v,
                              str(x.get("currency") or "")[:20], q[:1000], doc, _locate(q, win, p0)])
    for i, x in enumerate(lst("governing_law")):
        if x.get("jurisdiction"):
            q = str(x.get("quote") or "")
            out["governing_law"].append([f"GL:{doc}:{k}:{i}", cid, str(x["jurisdiction"])[:120], q[:1000], doc,
                                         _locate(q, win, p0)])
    if isinstance(js.get("title"), str) and js["title"].strip():
        out["title"].append(js["title"].strip()[:300])
    return out


def step_extract(pg: pd.DataFrame, only: list[str] | None = None, cache_only: bool = False) -> dict:
    t0 = time.perf_counter()
    jobs, texts_of = [], {}
    for d, g in sorted(pg.groupby("doc_name"), key=lambda kv: kv[0]):
        if only and d not in only:
            continue
        g = g.sort_values("page")
        texts_of[d] = dict(zip(g.page.astype(int), g.text))
        for k, win in enumerate(_windows(list(texts_of[d].items()))):
            if cache_only and not (CACHE / f"{d}__w{k}.json").exists():
                continue
            jobs.append((d, k, win))
    usage_acc: list = []

    async def run():
        return await asyncio.gather(*(_extract_window(d, k, w, usage_acc) for d, k, w in jobs))

    results = asyncio.run(run())
    tables = {"party": [], "clause": [], "key_date": [], "amount": [], "governing_law": []}
    titles = {}
    for (d, k, _), js in zip(jobs, results):
        if not js:
            continue
        recs = _records(d, k, js, texts_of[d])
        for t in tables:
            tables[t] += recs[t]
        if recs["title"] and d not in titles:
            titles[d] = recs["title"][0]
    con = rw()
    for t in tables:
        if only:
            for d in only:
                con.execute(f"DELETE FROM {t} WHERE doc_name = ?", [d])
        else:
            con.execute(f"DELETE FROM {t}")
        if tables[t]:
            con.executemany(f"INSERT INTO {t} VALUES ({','.join('?' * len(tables[t][0]))})", tables[t])
    for d, title in titles.items():
        con.execute("UPDATE contract SET printed_title = ? WHERE id = ?", [title, scope_id(d)])
    con.close()
    return {"extract_jobs": len(jobs), "contracts_extracted": len(texts_of),
            **{f"{t}_records": len(v) for t, v in tables.items()},
            "clause_categories": len({r[2] for r in tables["clause"]}),
            "extract_llm_calls": len(usage_acc),
            "extract_llm_prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage_acc),
            "extract_llm_completion_tokens": sum(u.get("completion_tokens", 0) for u in usage_acc),
            "extract_wall_s": round(time.perf_counter() - t0, 1)}


# ================================================================== query: 1. resolve the contract
def _contracts() -> list[dict]:
    return ro().execute("SELECT id, label, filer, contract_type, filing_date, title FROM contract").df().to_dict("records")


def match_contracts(question: str) -> list[tuple[float, dict]]:
    qn, ql = _norm(question), question.lower()
    scored = []
    for c in _contracts():
        s = 0.0
        if c["label"] and _norm(c["label"]) in qn:
            s += 10
        if c["filer"] and len(_norm(c["filer"])) >= 3 and _norm(c["filer"]) in qn:
            s += 4
        if c["filing_date"] and c["filing_date"] in question:
            s += 2
        if c["contract_type"] and c["contract_type"].lower().replace(" agreement", "") in ql:
            s += 1
        if s:
            scored.append((s, c))
    return sorted(scored, key=lambda x: -x[0])


async def resolve(question: str, usage: Usage) -> dict:
    hits = match_contracts(question)
    if hits and (len(hits) == 1 or hits[0][0] > hits[1][0]):
        c = hits[0][1]
    else:
        pool = [h[1] for h in hits[:15]] or _contracts()
        lines = "\n".join(f"{c['id']} | {c['label']}" for c in pool)
        out = await chat([{"role": "user", "content":
                           f"Which contract is this question about?\n{question}\n\nContracts:\n{lines}\n"
                           'Return JSON {"contract_id": "..."}'}],
                         tag=ANSWER_TAG, max_tokens=120, temperature=0.0)
        usage.add(out)
        try:
            cid = llm.parse_json(out["content"])["contract_id"]
            c = next(x for x in pool if x["id"] == cid)
        except Exception:
            return {"entity": None, "contract_id": None, "scope": []}
    return {"entity": c["label"], "contract_id": c["id"], "scope": [c["id"]]}


# ================================================================== query: 2. catalogue (value-free)
EMPTY_CATALOGUE = "(no structured records extracted for this contract)"


def catalogue(scope: list[str], question: str = "") -> tuple[str, dict]:
    c = ro()
    parts, keymap = [], {}
    for cid in scope:
        row = c.execute("SELECT label, contract_type, filing_date, n_pages, printed_title FROM contract WHERE id = ?",
                        [cid]).fetchone()
        if not row:
            continue
        label, ctype, fdate, n, ptitle = row
        parts.append(f"## contract {cid} = {label} ({ctype}, filed {fdate or '?'}, {n} pages)"
                     + (" [printed title recorded]" if ptitle else ""))
        npar = c.execute("SELECT count(*) FROM party WHERE contract_id = ?", [cid]).fetchone()[0]
        dk = c.execute("SELECT kind, count(*) FROM key_date WHERE contract_id = ? GROUP BY kind ORDER BY kind", [cid]).fetchall()
        ngl = c.execute("SELECT count(*) FROM governing_law WHERE contract_id = ?", [cid]).fetchone()[0]
        ak = c.execute("SELECT kind, count(*) FROM amount WHERE contract_id = ? GROUP BY kind ORDER BY 2 DESC LIMIT 12",
                       [cid]).fetchall()
        parts.append("fields:")
        parts.append(f"   title: {'1 record' if ptitle else 'none'}")
        parts.append(f"   parties: {npar} records")
        parts.append("   dates: " + (", ".join(f"{k} ({n})" for k, n in dk) if dk else "none"))
        parts.append(f"   governing_law: {ngl} records")
        parts.append("   amounts: " + (", ".join(f"{k} ({n})" for k, n in ak) if ak else "none"))
        cl = dict((cat, (n, pages)) for cat, n, pages in c.execute(
            "SELECT category, count(*), list(DISTINCT page ORDER BY page) FROM clause WHERE contract_id = ? "
            "GROUP BY category", [cid]).fetchall())
        parts.append("clauses (category: records, pages):")
        for cat in CLAUSE_TYPES:
            if cat in cl:
                n, pages = cl[cat]
                parts.append(f"   {cat}: {n} (p.{','.join(map(str, pages[:8]))})")
        missing = [cat for cat in CLAUSE_TYPES if cat not in cl]
        parts.append("no records extracted for: " + ", ".join(missing))
        keymap[cid] = {"categories": set(cl), "fields": set(FIELDS)}
    return "\n".join(parts), keymap


PLAN_SYS = """You are the query planner of an ontology-grounded contract QA system. You do NOT answer the question.
You get the question and a catalogue of the structured records extracted from the resolved contract: which
fields exist (title, parties, dates by kind, governing law, amounts by kind) and how many clause records exist
per clause category. Record text is hidden; it will be fetched by SQL from your plan.

Return JSON:
{
 "fields": ["title" | "parties" | "dates" | "governing_law" | "amounts", ...],   // fields to fetch
 "categories": ["<clause category exactly as listed>", ...],                        // clause records to fetch
 "search": ["<query>", ...]   // 1-3 short phrases likely to appear in the clause text itself
}
Rules:
- Fetch the category the question asks about plus closely related ones that often hold the same clause
  (e.g. Non-Compete with Exclusivity and Competitive Restriction Exception; Expiration / Renewal questions with
  the "dates" field and Termination For Convenience; Cap On Liability with Uncapped Liability; License Grant
  with Non-Transferable License and the Affiliate License categories).
- Use "dates" for agreement / effective / expiration / renewal / notice questions, "parties" for who signed,
  "governing_law" for which law governs, "title" for the document name, "amounts" for fees, minimums, caps.
- A category listed under "no records extracted" may still be present in the text (extraction can miss);
  still give search queries for it.
- search queries: contract wording, e.g. "shall not compete", "governed by the laws of", "right of first
  refusal", "may not assign", "audit the books and records", "liability shall not exceed". Always give at least one."""


def default_plan(question: str) -> dict:
    return {"fields": [], "categories": [], "search": [question]}


# ================================================================== query: 3. SQL
def execute(p: dict, keymap: dict) -> dict:
    errors, records = [], {}
    if not keymap:
        return {"sql": None, "records": {}, "results": {}, "errors": ["no contract resolved"]}
    cid = next(iter(keymap))
    cats = [x for x in (p.get("categories") or []) if isinstance(x, str)]
    bad = [x for x in cats if x not in CLAUSE_TYPES]
    if bad:
        errors.append(f"unknown categories {bad}")
    cats = [x for x in cats if x in CLAUSE_TYPES]
    fields = [f for f in (p.get("fields") or []) if f in FIELDS]
    c = ro()
    sqls = []

    def q(sql, args):
        sqls.append(sql.replace("?", f"'{cid}'"))
        return c.execute(sql, args).fetchall()

    if cats:
        ph = ",".join("?" * len(cats))
        sql = f"SELECT id, category, text, doc_name, page FROM clause WHERE contract_id = ? AND category IN ({ph}) ORDER BY category, page"
        rows = c.execute(sql, [cid, *cats]).fetchall()
        sqls.append(sql.replace("?", f"'{cid}'", 1).replace(f"({ph})", "(" + ",".join(f"'{x}'" for x in cats) + ")"))
        for r in rows:
            records[r[0]] = {"id": r[0], "type": "clause", "label": r[1], "value": r[2], "doc": r[3], "page": r[4]}
        empty = [x for x in cats if not any(r[1] == x for r in rows)]
        if empty:
            errors.append("no extracted records for: " + ", ".join(empty))
    if "title" in fields:
        r = c.execute("SELECT printed_title, doc_name FROM contract WHERE id = ?", [cid]).fetchone()
        sqls.append(f"SELECT printed_title FROM contract WHERE id = '{cid}'")
        if r and r[0]:
            records[f"TI:{r[1]}"] = {"id": f"TI:{r[1]}", "type": "title", "label": "printed title", "value": r[0],
                                    "doc": r[1], "page": 1}
    if "parties" in fields:
        for r in q("SELECT id, name, role, doc_name, page FROM party WHERE contract_id = ? ORDER BY page", [cid]):
            records[r[0]] = {"id": r[0], "type": "party", "label": r[2] or "party", "value": r[1], "doc": r[3], "page": r[4]}
    if "dates" in fields:
        for r in q("SELECT id, kind, value, raw, doc_name, page FROM key_date WHERE contract_id = ? ORDER BY kind", [cid]):
            records[r[0]] = {"id": r[0], "type": "date", "label": r[1], "value": r[2], "quote": r[3], "doc": r[4], "page": r[5]}
    if "governing_law" in fields:
        for r in q("SELECT id, jurisdiction, raw, doc_name, page FROM governing_law WHERE contract_id = ?", [cid]):
            records[r[0]] = {"id": r[0], "type": "governing_law", "label": "governing law", "value": r[1],
                             "quote": r[2], "doc": r[3], "page": r[4]}
    if "amounts" in fields:
        for r in q("SELECT id, kind, value, currency, raw, doc_name, page FROM amount WHERE contract_id = ? LIMIT 20", [cid]):
            records[r[0]] = {"id": r[0], "type": "amount", "label": r[1], "value": f"{r[2]} {r[3]}".strip(),
                             "quote": r[4], "doc": r[5], "page": r[6]}
    return {"sql": ";\n".join(sqls) or None, "records": records, "results": {}, "errors": errors}


# ================================================================== query: 5. grounded generation
ANSWER_SYS = """You answer questions about one commercial contract using ONLY the evidence given: structured records
extracted from the contract (clauses with their text, parties, dates, governing law, amounts; IDs in brackets) and
text passages from the same contract.
- "Does the contract contain X? If so, quote it." -> start with Yes or No. If yes, quote the operative clause text
  verbatim (the most relevant sentence or two) and cite it. Records were extracted by a model and can be
  mislabelled: check the text really is such a clause; passages may confirm it or show a clause the records missed.
- If neither the records nor the passages show such a clause, answer "No - not present in this contract."
- Dates, parties, governing law, renewal terms and notice periods: give the value as the contract states it,
  then quote the supporting sentence. If the contract does not state it, say it is not present.
- Never invent contract text.
Format: first line "Answer: <direct answer>", then 1-4 short lines of support with citations like [CL:...] (<doc> p.N)."""


def evidence_block(question: str, res: dict, ex: dict, passages: list[dict]) -> str:
    lines = [f"Question: {question}", f"Resolved contract: {res.get('entity')} ({res.get('contract_id')})", ""]
    if ex["records"]:
        lines.append("Structured records (SQL over the contract ontology):")
        for r in ex["records"].values():
            q = f' | quote: "{r["quote"][:600]}"' if r.get("quote") else ""
            lines.append(f"- [{r['id']}] {r['type']} / {r['label']}: {str(r['value'])[:1500]}{q} ({r['doc']} p.{r['page']})")
    else:
        lines.append("Structured records: none matched the plan.")
    if ex["errors"]:
        lines.append("Notes: " + "; ".join(ex["errors"]))
    lines.append("")
    lines.append("Passages (text search scoped to the resolved contract):")
    for p in passages:
        lines.append(f"--- ({p['doc']} p.{p['page']})\n{p['text']}")
    return "\n".join(lines)
