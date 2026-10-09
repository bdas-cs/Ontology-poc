"""Candidates -> automatic filters -> balanced final set in the common question schema.

Drop reasons (logged to runs/qgen-filter.jsonl):
  evidence_not_found    quote not on any shown page, even after whitespace / markdown-
                        emphasis / quote-mark normalisation (page is corrected if the
                        quote is found on a different shown page)
  ellipsis_in_quote     quote is not contiguous
  answer_not_in_evidence numeric answer whose numbers appear nowhere on the evidence page
  answer_in_question    the question leaks its own answer
  trivial_from_title    the answer is contained in the document title
  legacy_font_text      document is Hindi in a legacy 8-bit font (Latin gibberish), so the
                        quote and a translated answer cannot be verified by a reviewer
  vague_reference       "this document / the scheme" style references
  near_duplicate        question token-Jaccard >= 0.6 with a kept question, or same doc + answer
Then: 50 per question_type (shortfalls filled from other types' surplus), with ~24%
questions over non-English documents, round-robin over topics and distinct documents.
"""
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict

from poc.datasets import bharat
from poc.qgen import CANDIDATES, FILTER_LOG, QUESTION_TYPES, QUESTIONS

SEED = 11
PER_TYPE = 50
NON_EN_PER_TYPE = 12  # 36 / 150 = 24%
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_STOP = {"the", "of", "and", "for", "in", "to", "a", "an", "on", "by", "under", "scheme", "yojana",
         "government", "india", "department", "what", "is", "are", "which", "who", "how", "does", "do",
         "per", "as", "with", "from", "at", "or", "be", "that", "its", "this"}
_VAGUE = re.compile(r"\b(this|the above|the given|the said) (document|notice|circular|page|scheme|order|report)\b", re.I)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).translate(_DEV_DIGITS)
    s = s.replace(" ", " ").replace("​", "").replace("‌", "").replace("‍", "")
    s = re.sub(r"[‘’`´]", "'", s)
    s = re.sub(r"[“”]", '"', s)
    s = re.sub(r"[–—−]", "-", s)
    s = s.replace("*", "")
    return re.sub(r"\s+", " ", s).strip()


def tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"\w+", norm(s).lower()) if t not in _STOP and len(t) > 1}


def numbers(s: str) -> set[str]:
    s = norm(s).replace(",", "")
    return {n.lstrip("0") or "0" for n in re.findall(r"\d+(?:\.\d+)?", s) if len(n.replace(".", "")) >= 2}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def qlang(c: dict, k: int, q: dict, page: int, exact: bool) -> str:
    """Language of the evidence (what a retriever must match): script of the quote, else the doc's."""
    quote = q["evidence_quote"]
    if len(re.findall(r"[^\W\d_]", quote)) >= 15:
        return bharat.detect_language(quote)[0]
    return bharat.doc_info()[c["doc_name"]]["language"]


def _check(c: dict, q: dict, pages: list[str]) -> tuple[str | None, int | None, bool]:
    """-> (drop_reason, 0-based evidence page, exact_match)."""
    quote = q["evidence_quote"].strip()
    if bharat.doc_info()[c["doc_name"]]["legacy_font"]:
        return "legacy_font_text", None, False
    if not quote:
        return "evidence_not_found", None, False
    if "..." in quote or "…" in quote:
        return "ellipsis_in_quote", None, False
    shown = [p - 1 for p in c["pages_shown"] if 0 < p <= len(pages)]
    claimed = q["evidence_page"] - 1
    order = ([claimed] if claimed in shown else []) + [p for p in shown if p != claimed]
    page, exact = None, False
    for p in order:
        if quote in pages[p]:
            page, exact = p, True
            break
    if page is None:
        nq = norm(quote)
        page = next((p for p in order if nq in norm(pages[p])), None)
    if page is None:
        return "evidence_not_found", None, False
    if q["question_type"] == "numeric" and not q.get("answer_translated"):
        nums = numbers(q["answer"])
        if nums and not (nums & numbers(pages[page])):
            return "answer_not_in_evidence", page, exact
    ans = norm(q["answer"]).lower()
    if len(ans) >= 4 and ans in norm(q["question"]).lower():
        return "answer_in_question", page, exact
    title = norm(bharat.doc_info()[c["doc_name"]]["title"]).lower()
    at = tokens(q["answer"])
    if (len(ans) >= 4 and ans in title) or (at and len(at) <= 6 and at <= tokens(title)):
        return "trivial_from_title", page, exact
    if _VAGUE.search(q["question"]):
        return "vague_reference", page, exact
    return None, page, exact


def run() -> dict:
    cands = [json.loads(l) for l in CANDIDATES.open() if l.strip()]
    info = bharat.doc_info()
    drops, log, ok = Counter(), [], []
    n_cand = 0
    exact_n = 0
    for c in cands:
        pages = bharat.doc_pages(c["doc_name"])
        for k, q in enumerate(c.get("questions", [])):
            n_cand += 1
            reason, page, exact = _check(c, q, pages)
            row = {"qid": f"bh-{c['doc_name']}-{k}", "doc_name": c["doc_name"], "reason": reason,
                   "question": q["question"], "question_type": q["question_type"]}
            if reason:
                drops[reason] += 1
                log.append(row)
                continue
            exact_n += exact
            ok.append((c, k, q, page, exact))

    # near-duplicates (deterministic order: as generated)
    kept, seen_tokens, seen_ans = [], [], set()
    for c, k, q, page, exact in ok:
        qt = tokens(q["question"])
        key = (c["doc_name"], norm(q["answer"]).lower())
        if key in seen_ans or any(jaccard(qt, s) >= 0.6 for s in seen_tokens):
            drops["near_duplicate"] += 1
            log.append({"qid": f"bh-{c['doc_name']}-{k}", "doc_name": c["doc_name"], "reason": "near_duplicate",
                        "question": q["question"], "question_type": q["question_type"]})
            continue
        seen_ans.add(key)
        seen_tokens.append(qt)
        kept.append((c, k, q, page, exact))

    # balance
    rng = random.Random(SEED)
    by_type = defaultdict(list)
    for item in kept:
        by_type[item[2]["question_type"]].append(item)

    def pick(items: list, n: int, used_docs: Counter) -> list:
        items = sorted(items, key=lambda x: (x[0]["doc_name"], x[1]))
        rng.shuffle(items)
        out = []
        for max_per_doc in (1, 2, 99):  # spread over documents first
            topics = defaultdict(list)
            for it in items:
                if it not in out and used_docs[it[0]["doc_name"]] < max_per_doc:
                    topics[it[0]["topic"]].append(it)
            while len(out) < n and any(topics.values()):  # round-robin over topics
                for t in sorted(topics):
                    if topics[t] and len(out) < n:
                        it = topics[t].pop(0)
                        if used_docs[it[0]["doc_name"]] < max_per_doc:
                            out.append(it)
                            used_docs[it[0]["doc_name"]] += 1
            if len(out) >= n:
                break
        return out

    final, used_docs = [], Counter()
    for t in QUESTION_TYPES:
        pool = by_type[t]
        non_en = [x for x in pool if qlang(*x) != "en"]
        en = [x for x in pool if qlang(*x) == "en"]
        sel = pick(non_en, NON_EN_PER_TYPE, used_docs)
        sel += pick(en, PER_TYPE - len(sel), used_docs)
        if len(sel) < PER_TYPE:
            sel += pick([x for x in non_en if x not in sel], PER_TYPE - len(sel), used_docs)
        final += sel
    target = PER_TYPE * len(QUESTION_TYPES)
    if len(final) < target:  # fill shortfalls from any surplus
        rest = [x for x in kept if x not in final]
        final += pick(rest, target - len(final), used_docs)

    out = []
    for c, k, q, page, exact in final:
        i = info[c["doc_name"]]
        out.append({
            "qid": f"bh-{c['doc_name']}-{k}",
            "question": q["question"].strip(),
            "answer": q["answer"].strip(),
            "justification": q["justification"].strip(),
            "question_type": q["question_type"],
            "doc_name": c["doc_name"],
            "evidence": [{"evidence_text": q["evidence_quote"].strip(), "doc_name": c["doc_name"],
                          "evidence_page_num": page}],
            "language": qlang(c, k, q, page, exact),
            "doc_language": i["language"],
            "topic": c["topic"],
            "generated_by": c["model"],
            "answer_language": q["answer_language"],
            "answer_translated": q["answer_translated"],
            "evidence_exact": exact,
            "entity_anchor": q["entity_anchor"],
            "title": i["title"],
            "url": i["url"],
        })
    # Same evidence elsewhere in the corpus (mirrored scheme pages, re-hosted PDFs): record the
    # other doc_names so retrieval scoring can accept them; the question may still be unique.
    sub = bharat._subset()
    normed = {d: norm(t) for d, t in zip(sub.document_id, sub.text)}
    for r in out:
        nq = norm(r["evidence"][0]["evidence_text"])
        r["evidence_also_in"] = [d for d, t in normed.items() if d != r["doc_name"] and nq in t] if len(nq) >= 30 else []
    out.sort(key=lambda r: (QUESTION_TYPES.index(r["question_type"]), r["qid"]))
    with QUESTIONS.open("w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with FILTER_LOG.open("w") as f:
        for r in log:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "documents": len(cands),
        "skipped_documents": sum(1 for c in cands if not c.get("questions")),
        "skip_reasons": Counter(c.get("skip_reason", "")[:60] for c in cands if not c.get("questions")),
        "candidates": n_cand, "passed_filters": len(kept), "drops": dict(drops),
        "evidence_exact_of_passed": exact_n,
        "kept": len(out),
        "by_type": dict(Counter(r["question_type"] for r in out)),
        "by_language": dict(Counter(r["language"] for r in out)),
        "non_english_share": round(sum(r["language"] != "en" for r in out) / max(1, len(out)), 3),
        "by_topic": dict(Counter(r["topic"] for r in out)),
        "translated_answers": sum(r["answer_translated"] for r in out),
        "evidence_found_in_other_docs": sum(bool(r["evidence_also_in"]) for r in out),
        "distinct_docs": len({r["doc_name"] for r in out}),
        "candidates_by_type": dict(Counter(q["question_type"] for c in cands for q in c.get("questions", []))),
    }
    return stats
