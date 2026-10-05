"""Ontology-grounded query path (paper's 5 steps), self-hosted Qwen3.6 via poc.llm.

  1. resolve   question text -> Company (alias match, LLM fallback) -> Filing(s) (LLM over the
               company's filing records: type / fiscal year / quarter / date)
  2. plan      LLM sees the resolved filing's statement catalogue (row keys, labels, normalized
               metrics, period columns — no values) and emits a structured plan:
               {vars: {name: "<row>.<col>"}, calc: {name: expr}, search: [queries]}
  3. execute   plan vars compile to SQL over line_item; calc expressions are evaluated in Python
               (safe AST evaluator) — numbers are computed, never estimated by the model
  4. enrich    hybrid BM25 + dense search over chunks of the resolved filing(s) ONLY
  5. generate  answer only from the records + passages, citing record IDs and pages

The pipeline sees only the question text. Gold metadata is used solely afterwards for the
resolved_correct_filing diagnostic (in run.py).
"""
import ast
import json
import math
import operator
import re
import time

import duckdb

from poc import llm
from poc.ontology import schema

TAG = "ontology-answer"
_CON = None


async def _chat(messages, **kw) -> dict:
    """poc.llm.chat with longer infra retries: the supervised server needs ~1 min to come back after a
    Metal crash, longer than poc.llm's own 3 quick retries. Pipeline logic is unchanged."""
    import asyncio
    for attempt in range(8):
        try:
            return await llm.chat(messages, **kw)
        except Exception:
            if attempt == 7:
                raise
            await asyncio.sleep(30)


def con() -> duckdb.DuckDBPyConnection:
    global _CON
    if _CON is None:
        _CON = duckdb.connect(str(schema.DB_PATH), read_only=True)
        _CON.execute("LOAD fts")
    return _CON


class Usage:
    def __init__(self):
        self.prompt = self.completion = self.calls = 0

    def add(self, out: dict):
        u = out.get("usage") or {}
        self.prompt += u.get("prompt_tokens", 0)
        self.completion += u.get("completion_tokens", 0)
        self.calls += 1


# ------------------------------------------------------------------ 1. entity resolution
def _companies() -> list[dict]:
    return con().execute("SELECT id, name, aliases, ticker FROM company").df().to_dict("records")


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
        out = await _chat([{"role": "user", "content":
                               f"Which company is this question about?\n{question}\n\nCompanies:\n{names}\n"
                               'Return JSON {"company_id": "..."}'}],
                             tag=TAG, max_tokens=60, temperature=0.0)
        usage.add(out)
        try:
            cid = llm.parse_json(out["content"])["company_id"]
            comp = next(c for c in _companies() if c["id"] == cid)
        except Exception:
            return {"company": None, "filings": []}
    fil = con().execute(
        "SELECT id, doc_type, fiscal_year, fiscal_quarter, filed_date FROM filing WHERE company_id = ? "
        "ORDER BY fiscal_year, coalesce(fiscal_quarter, 5), filed_date", [comp["id"]]).fetchall()
    lines = "\n".join(f"{f[0]} | {f[1]} | {f[2]} | {('Q' + str(f[3])) if f[3] else '-'} | {f[4] or '-'}" for f in fil)
    ids = [f[0] for f in fil]
    if len(ids) == 1:
        chosen = ids
    elif rule := rule_resolve(question, fil):
        chosen = rule
    else:
        out = await _chat([{"role": "system", "content": RESOLVE_SYS},
                              {"role": "user", "content": f"Question: {question}\nCompany: {comp['name']}\nFilings:\n{lines}"}],
                             tag=TAG, max_tokens=400, temperature=0.0)
        usage.add(out)
        try:
            chosen = [f for f in llm.parse_json(out["content"]).get("filings", []) if f in ids][:2]
        except Exception:  # truncated / malformed JSON: take filing ids in the order they appear
            chosen = [f for f in dict.fromkeys(re.findall(r"F:[\w.\-]+", out["content"])) if f in ids][:2]
        if not chosen:
            chosen = [ids[-1]]
    return {"company": comp["name"], "company_id": comp["id"], "filings": chosen}


# ------------------------------------------------------------------ 2. query planning
KIND_CODE = {"income": "I", "balance": "B", "cashflow": "C"}


def catalogue(filing_ids: list[str]) -> tuple[str, dict]:
    """Text catalogue of statements for the plan prompt + key -> (filing, kind, row_idx, {col: period}) map."""
    c = con()
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


async def plan(question: str, filing_ids: list[str], usage: Usage, thinking: bool = False) -> tuple[dict, dict, str]:
    cat, keymap = catalogue(filing_ids)
    if not cat:
        cat = "(no structured statements for this filing)"
    msgs = [{"role": "system", "content": PLAN_SYS},
            {"role": "user", "content": f"Question: {question}\n\nCatalogue:\n{cat}"}]
    out = await _chat(msgs, tag=TAG, max_tokens=5000 if thinking else 900, temperature=0.0, thinking=thinking)
    usage.add(out)
    if thinking and "{" not in out["content"]:  # reasoning exhausted the budget -> plan directly
        out = await _chat(msgs, tag=TAG, max_tokens=900, temperature=0.0)
        usage.add(out)
    try:
        p = llm.parse_json(out["content"])
        if not isinstance(p, dict):
            raise ValueError
    except Exception:
        p = {"needs_numbers": False, "vars": {}, "calc": {}, "search": [question]}
    return p, keymap, cat


# ------------------------------------------------------------------ 3. structured retrieval + compute
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"abs": abs, "min": min, "max": max, "round": round, "sum": lambda *a: sum(a), "avg": lambda *a: sum(a) / len(a),
          "sqrt": math.sqrt}


def safe_eval(expr: str, env: dict) -> float:
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.Name):
            return env[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ValueError(f"unsupported expression: {ast.dump(n)[:80]}")
    return ev(ast.parse(expr, mode="eval"))


REF_IN_EXPR = re.compile(r"(?<![\w.])((?:\d:)?[IBC]\d+)\.(c\d+)\b")


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
    for name, ref in variables.items():
        m = re.fullmatch(r"\s*((?:\d:)?[IBC]\d+)\.(c\d+)\s*", str(ref))
        if not m or m.group(1) not in keymap or m.group(2) not in keymap[m.group(1)][3]:
            errors.append(f"bad ref {name}={ref}")
            continue
        fid, kind, ri, cols = keymap[m.group(1)]
        refs[name] = (fid, kind, ri, cols[m.group(2)])
    sql = None
    records = {}
    if refs:
        conds = " OR ".join(
            f"(filing_id = '{fid}' AND kind = '{kind}' AND row_idx = {ri} AND period = '{per}')"
            for fid, kind, ri, per in refs.values())
        sql = ("SELECT id, label, normalized_metric, value, unit, scale, period, column_label, doc_name, page, "
               "filing_id, kind, row_idx FROM line_item WHERE " + conds)
        rows = con().execute(sql).fetchall()
        idx = {(r[10], r[11], r[12], r[6]): r for r in rows}
        for name, key in refs.items():
            r = idx.get(key)
            if r:
                records[name] = {"id": r[0], "label": r[1], "metric": r[2], "value": r[3], "unit": r[4],
                                 "scale": r[5], "period": r[6], "column": r[7], "doc": r[8], "page": r[9]}
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


# ------------------------------------------------------------------ 4. scoped text search
_EMB = None


def _embedder():
    global _EMB
    if _EMB is None:
        from sentence_transformers import SentenceTransformer
        from poc import config
        _EMB = SentenceTransformer(config.EMBED_MODEL, device="mps")
    return _EMB


def scoped_search(queries: list[str], filing_ids: list[str], k: int = 6) -> list[dict]:
    if not filing_ids:
        return []
    c = con()
    ph = ",".join("?" * len(filing_ids))
    scores: dict[str, float] = {}
    qv = _embedder().encode(queries, prompt_name="query", normalize_embeddings=True)
    for q, v in zip(queries, qv):
        dense = c.execute(
            f"SELECT id FROM chunk WHERE filing_id IN ({ph}) "
            f"ORDER BY array_cosine_similarity(emb, ?::FLOAT[1024]) DESC LIMIT 20",
            [*filing_ids, v.tolist()]).fetchall()
        sparse = c.execute(
            f"SELECT id FROM (SELECT id, fts_main_chunk.match_bm25(id, ?) AS s FROM chunk WHERE filing_id IN ({ph})) "
            f"WHERE s IS NOT NULL ORDER BY s DESC LIMIT 20", [q, *filing_ids]).fetchall()
        for lst in (dense, sparse):
            for rank, (cid,) in enumerate(lst):
                scores[cid] = scores.get(cid, 0) + 1 / (60 + rank)
    if not scores:
        return []
    # Chunk hits vote for the pages they span; evidence is rendered from the page's own text layer with
    # column gaps kept (" | "), so table rows stay aligned (the shared chunk index collapses whitespace).
    ids = list(scores)
    spans = c.execute(f"SELECT id, filing_id, doc_name, page, page_end FROM chunk WHERE id IN ({','.join('?' * len(ids))})",
                      ids).fetchall()
    pscore: dict[tuple, float] = {}
    for cid, fid, doc, p0, p1 in spans:
        for pg in range(p0, p1 + 1):
            pscore[(fid, doc, pg)] = pscore.get((fid, doc, pg), 0) + scores[cid]
    top = sorted(pscore, key=lambda x: -pscore[x])[:k]
    out = []
    for fid, doc, pg in top:
        t = c.execute("SELECT text FROM page WHERE filing_id = ? AND page = ?", [fid, pg]).fetchone()
        if not t or not t[0]:
            continue
        txt = re.sub(r"[ \t]{3,}", " | ", t[0])
        txt = re.sub(r"\n\s*\n+", "\n", txt)[:6000]
        out.append({"id": f"P:{doc}:{pg}", "doc": doc, "page": pg, "text": txt})
    return out


# ------------------------------------------------------------------ 5. grounded generation
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
                    "sh": "shares" + SCALE_NAME.get(r["scale"], ""), "%": "%"}.get(r["unit"], r["unit"])
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


ANSWER_THINK_BUDGET = 4000


async def answer(question: str, plan_thinking: bool = True, answer_thinking: bool = True) -> dict:
    t0 = time.perf_counter()
    usage = Usage()
    res = await resolve(question, usage)
    p, keymap, _ = await plan(question, res["filings"], usage, thinking=plan_thinking)
    ex = execute(p, keymap)
    queries = [q for q in (p.get("search") or []) if isinstance(q, str)][:3] + [question]
    passages = scoped_search(queries, res["filings"], k=6)
    ev = evidence_block(question, res, ex, passages)
    msgs = [{"role": "system", "content": ANSWER_SYS}, {"role": "user", "content": ev}]
    out = await _chat(msgs, tag=TAG, max_tokens=ANSWER_THINK_BUDGET if answer_thinking else 700,
                         temperature=0.0, thinking=answer_thinking)
    usage.add(out)
    if not out["content"].strip():  # reasoning ran out of budget -> answer directly
        out = await _chat(msgs, tag=TAG, max_tokens=700, temperature=0.0)
        usage.add(out)
    pages_used = sorted({f"{r['doc']} p.{r['page']}" for r in ex["records"].values()} |
                        {f"{p['doc']} p.{p['page']}" for p in passages})
    return {
        "answer": out["content"],
        "resolved": {"company": res.get("company"), "filings": [f[2:] for f in res["filings"]]},
        "plan": p,
        "sql": ex["sql"],
        "records_used": [r["id"] for r in ex["records"].values()],
        "computed": {k: v["value"] for k, v in ex["results"].items()},
        "plan_errors": ex["errors"],
        "pages_used": pages_used,
        "wall_s": round(time.perf_counter() - t0, 2),
        "prompt_tokens": usage.prompt,
        "completion_tokens": usage.completion,
        "llm_calls": usage.calls,
    }
