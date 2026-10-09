"""Ontology-grounded query path (paper's 5 steps), self-hosted Qwen3.6 via poc.llm — generic over the
active ontology domain plugin (poc.ontology.domains: finance / contracts / gov; see docs/ONTOLOGY_PLUGINS.md).

  1. resolve   plugin.resolve: question text -> entity -> scope record(s) (finance: Company -> Filing(s);
               contracts: Contract)
  2. plan      LLM sees the plugin's value-free catalogue of the resolved scope (plugin.catalogue) and the
               plugin's planner prompt, and emits a structured JSON plan (which records / fields, arithmetic,
               text-search queries)
  3. execute   plugin.execute compiles the plan to SQL over the typed tables; arithmetic is evaluated in
               Python (safe AST evaluator) — numbers are computed, never estimated by the model
  4. enrich    hybrid BM25 + dense search over chunks of the resolved scope ONLY (generic)
  5. generate  answer only from the records + passages (plugin.evidence_block / ANSWER_SYS), citing record
               IDs and pages

The pipeline sees only the question text. Gold metadata is used solely afterwards for the
resolved_correct_filing diagnostic (in run.py).
"""
import inspect
import time

from poc import llm
from poc.ontology import domains
from poc.ontology.runtime import ANSWER_TAG as TAG, Usage, chat as _chat, page_view, ro as con, safe_eval  # noqa: F401


def dom():
    return domains.get()


# ------------------------------------------------------------------ 2. query planning (generic)
async def plan(question: str, scope: list[str], usage: Usage, thinking: bool = False) -> tuple[dict, dict, str]:
    d = dom()
    params = inspect.signature(d.catalogue).parameters
    cat, keymap = d.catalogue(scope, question) if len(params) > 1 else d.catalogue(scope)
    if not cat:
        cat = getattr(d, "EMPTY_CATALOGUE", "(no structured records for this scope)")
    system = d.plan_system(cat) if hasattr(d, "plan_system") else d.PLAN_SYS
    msgs = [{"role": "system", "content": system},
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
        p = getattr(d, "default_plan", lambda q: {"needs_numbers": False, "vars": {}, "calc": {}, "search": [q]})(question)
    return p, keymap, cat


# ------------------------------------------------------------------ 4. scoped text search (generic)
_EMB = None


def _embedder():
    global _EMB
    if _EMB is None:
        from sentence_transformers import SentenceTransformer
        from poc import config
        _EMB = SentenceTransformer(config.EMBED_MODEL, device=config.torch_device())
    return _EMB


def scoped_search(queries: list[str], scope: list[str], k: int = 6) -> list[dict]:
    """Hybrid dense + BM25 over chunks whose scope id (chunk.filing_id) is in `scope`; returns the top-k
    pages rendered from page text."""
    if not scope:
        return []
    c = con()
    ph = ",".join("?" * len(scope))
    scores: dict[str, float] = {}
    qv = _embedder().encode(queries, prompt_name="query", normalize_embeddings=True)
    for q, v in zip(queries, qv):
        dense = c.execute(
            f"SELECT id FROM chunk WHERE filing_id IN ({ph}) "
            f"ORDER BY array_cosine_similarity(emb, ?::FLOAT[1024]) DESC LIMIT 20",
            [*scope, v.tolist()]).fetchall()
        sparse = c.execute(
            f"SELECT id FROM (SELECT id, fts_main_chunk.match_bm25(id, ?) AS s FROM chunk WHERE filing_id IN ({ph})) "
            f"WHERE s IS NOT NULL ORDER BY s DESC LIMIT 20", [q, *scope]).fetchall()
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
        out.append({"id": f"P:{doc}:{pg}", "doc": doc, "page": pg, "text": page_view(t[0])})
    return out


# ------------------------------------------------------------------ the 5 steps
ANSWER_THINK_BUDGET = 4000


async def answer(question: str, plan_thinking: bool = True, answer_thinking: bool = True) -> dict:
    t0 = time.perf_counter()
    d = dom()
    usage = Usage()
    res = await d.resolve(question, usage)                                   # 1. resolve
    scope = res.get("scope", [])
    p, keymap, _ = await plan(question, scope, usage, thinking=plan_thinking)  # 2. plan
    ex = d.execute(p, keymap)                                                # 3. SQL + arithmetic
    queries = [q for q in (p.get("search") or []) if isinstance(q, str)][:3] + [question]
    passages = scoped_search(queries, scope, k=getattr(d, "SEARCH_K", 6))    # 4. scoped search
    ev = d.evidence_block(question, res, ex, passages)                       # 5. cited answer
    msgs = [{"role": "system", "content": d.ANSWER_SYS}, {"role": "user", "content": ev}]
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
        "domain": d.NAME,
        "resolved": {"company": res.get("entity", res.get("company")), "filings": [d.doc_of(s) for s in scope]},
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


# Backwards-compatible names (finance plugin), e.g. for notebooks / scripts that imported them from here.
def __getattr__(name: str):
    if name in ("catalogue", "execute", "resolve", "match_companies", "rule_resolve", "period_mentions",
                "evidence_block", "RESOLVE_SYS", "PLAN_SYS", "ANSWER_SYS", "KIND_CODE"):
        return getattr(domains.get("finance"), name)
    raise AttributeError(name)
