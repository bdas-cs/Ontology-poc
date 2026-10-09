"""Stratified document sample -> one Claude call per document -> candidate questions.

Sampling: eligible documents (enough prose, some digits) stratified by primary topic
(rarest selected label wins, so budget / income_tax are not swamped by schemes) and by
language group (en vs Indian-language), oversampling non-English to ~28% of documents.
Each document is assigned two target question types on a rotation so the three types
come out roughly balanced; Claude may substitute a type the document cannot support.

Long documents are truncated to a page window (page 1 + a seeded contiguous block)
that fits a per-language character budget; page numbers are the real pseudo-page
numbers from bharat.split_pages, so evidence pages line up with pages.parquet.
"""
import asyncio
import json
import random
import re

import anthropic

from poc.datasets import bharat
from poc.qgen import BUDGET_USD, CANDIDATES, MODEL, PLAN, QUESTION_TYPES, cost, ledger, spent

SEED = 7
TOPIC_ORDER = ["income_tax", "budget", "public_finance", "schemes"]  # rarest first
DOCS_PER_TOPIC = {"schemes": 40, "public_finance": 26, "budget": 20, "income_tax": 18}
NON_EN_SHARE = 0.28
MYSCHEME_MAX_SHARE = 0.6
SEND_CHARS = {"en": 20000, "other": 9000}  # Indic scripts cost ~2-3x more tokens per char
TYPE_PAIRS = [("numeric", "eligibility_fact"), ("numeric", "narrative"), ("eligibility_fact", "narrative")]
EST_OUTPUT_TOKENS = 2500  # thinking at effort=medium + JSON; used only for the pre-run estimate

SYSTEM = """You write benchmark questions for a retrieval-augmented QA evaluation over a corpus of ~14,000 Indian government documents (scheme pages, budget speeches and documents, finance-department circulars, income-tax notices, public-finance reports). A retrieval system will later have to find the right document among all of them using only the question text, then answer it.

You are given ONE document (split into numbered pages; long documents are shown as a page window). Write the requested questions about it. Every question must:
1. Be answerable only from this document, with a single short, checkable gold answer (a figure, date, name, list of up to ~4 items, or one or two sentences for narrative questions).
2. Name its subject explicitly enough to be resolved from the question alone: the scheme / programme / tax / budget head by name, plus the issuing department or ministry, the state (or "Government of India" / central), and the financial year or date when relevant. Never say "this document", "the scheme", "the notice" or similar.
3. Not be answerable from the document title alone, and must not contain its own answer.
4. Be grounded in an evidence quote copied EXACTLY, character for character, from ONE page of the document as shown (a contiguous span, typically one sentence, table row or line, at most ~300 characters, no ellipses, no paraphrase, no added or removed words; keep the original script). The quote must contain the fact that answers the question (e.g. the figure itself), not just its lead-in; a line break inside the quote may be written as a single space. Give the page number shown in the "=== PAGE n ===" header where the quote appears.
5. Have a gold answer that the evidence quote supports directly (simple unit conversion or arithmetic over the quote is fine; explain it in the justification).

Question types:
- numeric: amounts, counts, rates, percentages, budget lines, dates, deadlines, age limits.
- eligibility_fact: who is eligible, conditions, required documents, procedures, which authority / form / office, what a rule says.
- narrative: purpose or objective, explanation of a provision, or a comparison within the document (answer in one or two sentences).

Language: always write the question in English. If the document is in Hindi or another Indian language, either quote the answer in the source language or translate it to English, and record which (answer_language = ISO 639-1 code of the answer text; answer_translated = true if you translated it). The evidence quote must always be in the original language exactly as shown.

If the document cannot support a requested type, write a different type instead and label it correctly. If the document is unusable (navigation page, link list, garbled extraction, no substantive facts), return an empty questions list and say why in skip_reason."""

SCHEMA = {
    "type": "object",
    "properties": {
        "skip_reason": {"type": "string"},
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_type": {"type": "string", "enum": list(QUESTION_TYPES)},
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "answer_language": {"type": "string"},
                    "answer_translated": {"type": "boolean"},
                    "justification": {"type": "string"},
                    "evidence_quote": {"type": "string"},
                    "evidence_page": {"type": "integer"},
                    "entity_anchor": {"type": "string",
                                      "description": "the names in the question that identify the document: scheme/department/state/year"},
                },
                "required": ["question_type", "question", "answer", "answer_language", "answer_translated",
                             "justification", "evidence_quote", "evidence_page", "entity_anchor"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["skip_reason", "questions"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------- sampling

def _prose_ok(text: str) -> bool:
    t = text[:20000]
    lines = [l for l in t.split("\n") if l.strip()]
    if not lines:
        return False
    long_chars = sum(len(l) for l in lines if len(l) >= 40)
    return long_chars / max(1, sum(len(l) for l in lines)) >= 0.45 and len(re.findall(r"\d", t)) >= 5


def primary_topic(topics: list[str]) -> str | None:
    return next((t for t in TOPIC_ORDER if t in topics), None)


def sample_docs() -> list[dict]:
    info = bharat.doc_info()
    sub = bharat._subset()
    rng = random.Random(SEED)
    pools: dict[tuple[str, str], list[str]] = {}
    for d, i in info.items():
        topic = primary_topic(i["topics"])
        if topic is None or i["n_chars"] < 1500 or i["legacy_font"] or not _prose_ok(sub.loc[d, "text"]):
            continue
        group = "en" if i["language"] == "en" else "other"
        pools.setdefault((topic, group), []).append(d)
    for k in pools:
        pools[k].sort()
        rng.shuffle(pools[k])

    chosen = []
    for topic, n in DOCS_PER_TOPIC.items():
        other = pools.get((topic, "other"), [])
        n_other = min(len(other), round(n * NON_EN_SHARE))
        picks = other[:n_other]
        en = pools.get((topic, "en"), [])
        if topic == "schemes":  # cap the templated myScheme pages
            cap = int((n - n_other) * MYSCHEME_MAX_SHARE)
            my = [d for d in en if info[d]["host"].endswith("myscheme.gov.in")][:cap]
            rest = [d for d in en if not info[d]["host"].endswith("myscheme.gov.in")]
            en = my + rest
        picks += en[: n - len(picks)]
        chosen += [(topic, d) for d in picks]
    plan = []
    for k, (topic, d) in enumerate(chosen):
        i = info[d]
        plan.append({"doc_name": d, "topic": topic, "language": i["language"],
                     "types": list(TYPE_PAIRS[k % len(TYPE_PAIRS)])})
    return plan


def page_window(doc_name: str, language: str) -> list[tuple[int, str]]:
    """[(1-based page, text)] fitting the char budget: all pages, or page 1 + a seeded block."""
    pages = bharat.doc_pages(doc_name)
    budget = SEND_CHARS["en" if language == "en" else "other"]
    if sum(len(p) for p in pages) <= budget:
        return list(enumerate(pages, start=1))
    out = [(1, pages[0][:budget])]
    used = len(out[0][1])
    rng = random.Random(f"{SEED}-{doc_name}")
    start = rng.randrange(1, len(pages))
    for n in range(start, len(pages)):
        if used + len(pages[n]) > budget:
            break
        out.append((n + 1, pages[n]))
        used += len(pages[n])
    if len(out) == 1 and start > 1:  # nothing after start fit: take from page 2
        for n in range(1, len(pages)):
            if used + len(pages[n]) > budget:
                break
            out.append((n + 1, pages[n]))
            used += len(pages[n])
    return out


def build_prompt(item: dict) -> str:
    i = bharat.doc_info()[item["doc_name"]]
    window = page_window(item["doc_name"], item["language"])
    total = len(bharat.doc_pages(item["doc_name"]))
    shown = ", ".join(str(n) for n, _ in window)
    head = (f"Document title: {i['title'] or '(none)'}\nURL: {i['url']}\nTopic labels: {', '.join(i['topics'])}\n"
            f"Detected language: {i['language']}{' (bilingual)' if i['bilingual'] else ''}\n"
            f"Pages: {total} total; shown: {shown}\n\n")
    body = "\n\n".join(f"=== PAGE {n} ===\n{t}" for n, t in window)
    ask = (f"\n\nWrite exactly 2 questions: the first of type '{item['types'][0]}', the second of type "
           f"'{item['types'][1]}' (substitute another type only if the document cannot support it).")
    return head + body + ask


# ---------------------------------------------------------------- estimate / run

def estimate(plan: list[dict]) -> dict:
    client = anthropic.Anthropic()
    total_in = 0
    for item in plan:
        r = client.messages.count_tokens(model=MODEL, system=SYSTEM,
                                         messages=[{"role": "user", "content": build_prompt(item)}])
        total_in += r.input_tokens
    out = EST_OUTPUT_TOKENS * len(plan)
    usd = cost({"input_tokens": total_in, "output_tokens": out})
    return {"calls": len(plan), "input_tokens": total_in, "est_output_tokens": out, "est_usd": round(usd, 2)}


def _done() -> set[str]:
    if not CANDIDATES.exists():
        return set()
    return {json.loads(l)["doc_name"] for l in CANDIDATES.open() if l.strip()}


async def _one(client, item: dict) -> dict:
    resp = await client.beta.messages.create(
        model=MODEL,
        max_tokens=8000,
        system=SYSTEM,
        messages=[{"role": "user", "content": build_prompt(item)}],
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    u = resp.usage
    usage = {k: getattr(u, k, 0) or 0 for k in
             ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}
    c = cost(usage)
    ledger({"doc_name": item["doc_name"], "model": resp.model, "stop_reason": resp.stop_reason,
            "request_id": getattr(resp, "_request_id", None), **usage, "cost_usd": round(c, 6)})
    row = {**item, "model": resp.model, "stop_reason": resp.stop_reason, "cost_usd": c,
           "pages_shown": [n for n, _ in page_window(item["doc_name"], item["language"])]}
    if resp.stop_reason == "refusal":
        return {**row, "skip_reason": "refusal", "questions": []}
    if resp.stop_reason == "max_tokens":
        return {**row, "skip_reason": "max_tokens", "questions": []}
    text = next(b.text for b in resp.content if b.type == "text")
    return {**row, **json.loads(text)}


async def run(plan: list[dict], budget: float = BUDGET_USD, concurrency: int = 6) -> None:
    todo = [p for p in plan if p["doc_name"] not in _done()]
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(concurrency)
    state = {"spent": spent(), "reserved": 0.0, "stopped": False}
    reserve = cost({"input_tokens": 9000, "output_tokens": EST_OUTPUT_TOKENS})
    lock = asyncio.Lock()
    print(f"{len(todo)} documents to process; spent so far ${state['spent']:.2f} of ${budget:.2f}")

    async def worker(item: dict) -> None:
        async with sem:
            async with lock:
                if state["spent"] + state["reserved"] + reserve > budget:
                    state["stopped"] = True
                    return
                state["reserved"] += reserve
            try:
                row = await _one(client, item)
            except anthropic.APIStatusError as e:
                print(f"  {item['doc_name']}: API error {e.status_code}")
                async with lock:
                    state["reserved"] -= reserve
                return
            async with lock:
                state["reserved"] -= reserve
                state["spent"] += row["cost_usd"]
                with CANDIDATES.open("a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  {item['doc_name']} [{item['topic']}/{item['language']}] "
                  f"{len(row['questions'])} q  ${row['cost_usd']:.3f}  total ${state['spent']:.2f}")

    await asyncio.gather(*(worker(p) for p in todo))
    if state["stopped"]:
        print(f"STOPPED: budget ${budget:.2f} reached (spent ${state['spent']:.2f})")
    print(f"done; total qgen spend ${state['spent']:.2f}")


def save_plan(plan: list[dict]) -> None:
    with PLAN.open("w") as f:
        for p in plan:
            f.write(json.dumps(p) + "\n")


def load_plan() -> list[dict]:
    return [json.loads(l) for l in PLAN.open() if l.strip()]
