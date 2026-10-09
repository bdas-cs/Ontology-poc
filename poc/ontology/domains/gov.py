"""Gov domain plugin (Bharat): Indian central / state government documents -> typed records.

Design: ~/ontology-poc-data/bharat/gov_ontology_proposal.md (pragmatic subset). Object types; every row keeps
provenance (doc_name + 1-based pseudo-page; facts also keep a verbatim evidence_text span):

  jurisdiction  (id, level central/state/ut, name_en, aliases[])                     -- fixed seed list
  authority     (id, key, name, name_en, kind, jurisdiction_id, source, doc_name, page) -- one row per mention
                                                                                     (doc_info regex or LLM)
  document      (id = scope id "GD:<doc_name>", doc_name, title, url, host, kind, topics[], language, bilingual,
                 legacy_font, jurisdiction_id, department, n_pages, printed_title, title_en, doc_type,
                 reference_no, issued_on, issued_iso, fiscal_year, issuer, extracted, page)
  scheme        (id, doc_name, page, name, name_en, acronym, aliases[], authority, jurisdiction_id, source,
                 same_as[])   -- cross-site copies are linked by same_as (normalised English name), not merged
  benefit       (id, scheme_id, scheme, kind, amount_raw, value, unit_raw, value_norm, unit_norm, per, recipient, ...)
  eligibility   (id, scheme_id, scheme, kind age/income/category/residence/..., include, text, ...)
  requirement   (id, scheme_id, scheme, kind step/document/form/where_to_apply/..., step, text, ...)
  date_event    (id, scheme_id, subject, kind launch/deadline/effective/issued/event/..., date_text, date_iso, ...)
  budget_line   (id, jurisdiction_id, head, head_code, fiscal_year, estimate BE/RE/Actual, amount_raw, value,
                 unit_raw, value_norm, unit_norm, ...)
  tax_provision (id, tax, section, rule, rate_raw, limit_raw, value_norm, unit_norm, assessment_year, ...)
  figure        (id, subject, value_raw, value, unit_raw, value_norm, unit_norm, period, ...)  -- other stated
                 quantities (counts, shares, indicators): the proposal's FiscalIndicator, generalised

Amounts keep the source string (`*_raw`, Indian digit grouping / Devanagari digits as written) plus a normalised
number in code: lakh = 1e5, crore = 1e7, thousand = 1e3; `unit_norm` is "INR" (value_norm in rupees), "%" or
the stated unit.

Ingest: base rows from the dataset's doc_info (no LLM). Extraction: local LLM over page windows of
<= GOV_WINDOW_CHARS (default 12k chars ~ 4 pseudo-pages, at most GOV_MAX_WINDOWS per document), cached per window
in db/extract_cache/<doc>__w<k>.json. Multilingual: values as written, English gloss for names and criteria.
Documents flagged legacy_font (Kruti Dev Hindi, unreadable as text) are skipped. Optional
GOV_EXTRACT_DOCS=<file of doc_names>: full extraction only for those; every other document gets a cheap
"card" pass (doc-level fields + scheme / authority names from its first CARD_PAGES pages, cached as
<doc>__card.json), and resolution cards only use names found on those first pages, so a capped extraction does
not make the fully extracted documents easier to resolve (GOV_CARD_PASS=0 disables the card pass).

Query path: resolve the documents named by the question (BM25 over per-document name cards: titles, English
glosses, URL slug, host, state, authority, reference no., fiscal year, scheme names + acronyms; phrase / acronym /
reference / jurisdiction rules; LLM pick among the top candidates when the rules are not decisive) -> value-free
catalogue of the typed records in those documents -> plan (record keys + arithmetic + search) -> SQL, arithmetic
in Python -> scoped text search over the resolved documents -> cited answer. Narrative questions mostly fall
through to the scoped search.
"""
import asyncio
import datetime as dt
import json
import math
import os
import re
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd

from poc import config, llm
from poc.ontology.runtime import ANSWER_TAG, INGEST_TAG, Usage, chat, ro, rw, safe_eval, slug

NAME = "gov"
TAG = INGEST_TAG
CACHE = config.DB_DIR / "extract_cache"
WINDOW_CHARS = int(os.getenv("GOV_WINDOW_CHARS", "12000"))
MAX_WINDOWS = int(os.getenv("GOV_MAX_WINDOWS", "12"))
CARD_CHARS = 6000
CARD_PAGES = 2  # pseudo-pages are ~3,000 chars: names on pages 1-2 ~ what the card pass reads
EXTRACT_MAX_TOKENS = int(os.getenv("GOV_EXTRACT_MAX_TOKENS", "6000"))
SEARCH_K = 6
MAX_SCOPE = 3

DDL = """
CREATE TABLE IF NOT EXISTS jurisdiction (
  id VARCHAR PRIMARY KEY, level VARCHAR, name_en VARCHAR, aliases VARCHAR[]
);
CREATE TABLE IF NOT EXISTS authority (
  id VARCHAR PRIMARY KEY, key VARCHAR, name VARCHAR, name_en VARCHAR, kind VARCHAR, jurisdiction_id VARCHAR,
  source VARCHAR, doc_name VARCHAR, page INTEGER
);
CREATE TABLE IF NOT EXISTS document (
  id VARCHAR PRIMARY KEY, doc_name VARCHAR, title VARCHAR, url VARCHAR, host VARCHAR, kind VARCHAR,
  topics VARCHAR[], language VARCHAR, bilingual BOOLEAN, legacy_font BOOLEAN, jurisdiction_id VARCHAR,
  department VARCHAR, n_pages INTEGER, printed_title VARCHAR, title_en VARCHAR, doc_type VARCHAR,
  reference_no VARCHAR, issued_on VARCHAR, issued_iso VARCHAR, fiscal_year VARCHAR, issuer VARCHAR,
  extracted VARCHAR, page INTEGER
);
CREATE TABLE IF NOT EXISTS scheme (
  id VARCHAR PRIMARY KEY, doc_name VARCHAR, page INTEGER, name VARCHAR, name_en VARCHAR, acronym VARCHAR,
  aliases VARCHAR[], authority VARCHAR, jurisdiction_id VARCHAR, source VARCHAR, same_as VARCHAR[]
);
CREATE TABLE IF NOT EXISTS benefit (
  id VARCHAR PRIMARY KEY, scheme_id VARCHAR, scheme VARCHAR, kind VARCHAR, amount_raw VARCHAR, value DOUBLE,
  unit_raw VARCHAR, value_norm DOUBLE, unit_norm VARCHAR, per VARCHAR, recipient VARCHAR, evidence_text VARCHAR,
  doc_name VARCHAR, page INTEGER
);
CREATE TABLE IF NOT EXISTS eligibility (
  id VARCHAR PRIMARY KEY, scheme_id VARCHAR, scheme VARCHAR, kind VARCHAR, is_inclusion BOOLEAN, text VARCHAR,
  evidence_text VARCHAR, doc_name VARCHAR, page INTEGER
);
CREATE TABLE IF NOT EXISTS requirement (
  id VARCHAR PRIMARY KEY, scheme_id VARCHAR, scheme VARCHAR, kind VARCHAR, step INTEGER, text VARCHAR,
  evidence_text VARCHAR, doc_name VARCHAR, page INTEGER
);
CREATE TABLE IF NOT EXISTS date_event (
  id VARCHAR PRIMARY KEY, scheme_id VARCHAR, subject VARCHAR, kind VARCHAR, date_text VARCHAR, date_iso VARCHAR,
  evidence_text VARCHAR, doc_name VARCHAR, page INTEGER
);
CREATE TABLE IF NOT EXISTS budget_line (
  id VARCHAR PRIMARY KEY, jurisdiction_id VARCHAR, head VARCHAR, head_code VARCHAR, fiscal_year VARCHAR,
  estimate VARCHAR, amount_raw VARCHAR, value DOUBLE, unit_raw VARCHAR, value_norm DOUBLE, unit_norm VARCHAR,
  evidence_text VARCHAR, doc_name VARCHAR, page INTEGER
);
CREATE TABLE IF NOT EXISTS tax_provision (
  id VARCHAR PRIMARY KEY, tax VARCHAR, section VARCHAR, rule VARCHAR, rate_raw VARCHAR, limit_raw VARCHAR,
  value_norm DOUBLE, unit_norm VARCHAR, assessment_year VARCHAR, evidence_text VARCHAR, doc_name VARCHAR,
  page INTEGER
);
CREATE TABLE IF NOT EXISTS figure (
  id VARCHAR PRIMARY KEY, subject VARCHAR, value_raw VARCHAR, value DOUBLE, unit_raw VARCHAR, value_norm DOUBLE,
  unit_norm VARCHAR, period VARCHAR, evidence_text VARCHAR, doc_name VARCHAR, page INTEGER
);
"""
FACT_TABLES = ["scheme", "authority", "benefit", "eligibility", "requirement", "date_event", "budget_line",
               "tax_provision", "figure"]


def scope_id(doc_name: str) -> str:
    return "GD:" + doc_name


def doc_of(scope: str) -> str:
    return scope[3:]


# ================================================================== normalisation helpers
_DIGITS = {base + i: str(i) for base in (0x966, 0x9E6, 0xA66, 0xAE6, 0xB66, 0xBE6, 0xC66, 0xCE6, 0xD66)
           for i in range(10)}
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_SCALES = [(re.compile(r"crores?\b|\bcr\b\.?|करोड़|करोड|कोटी", re.I), 1e7),
           (re.compile(r"lakhs?\b|\blacs?\b|लाख", re.I), 1e5),
           (re.compile(r"thousands?\b|हज़ार|हजार|'000", re.I), 1e3),
           (re.compile(r"millions?\b|\bmn\b", re.I), 1e6),
           (re.compile(r"billions?\b|\bbn\b", re.I), 1e9)]
_INR = re.compile(r"₹|\brs\b\.?|\brs\.|\binr\b|rupee|रु|रुपये|रुपए|रुपया", re.I)
_PCT = re.compile(r"%|per ?cent|प्रतिशत|टक्के", re.I)


def ascii_digits(s: str) -> str:
    return (s or "").translate(_DIGITS)


def parse_number(raw) -> float | None:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    m = _NUM.search(ascii_digits(str(raw or "")))
    if not m:
        return None
    try:
        return float(m.group().replace(",", ""))
    except ValueError:
        return None


def normalise(raw, unit) -> tuple[float | None, float | None, str | None]:
    """(value as written, normalised value, normalised unit). The scale word (lakh / crore / thousand ...) is taken
    from the unit field first (table headers: 'Rs. in lakh'), else from the raw string; INR values are returned in
    rupees, percentages as percent."""
    v = parse_number(raw)
    if v is None:
        return None, None, None
    u, r = str(unit or ""), str(raw or "")
    if _PCT.search(u) or _PCT.search(r):
        return v, v, "%"
    scale = next((s for rx, s in _SCALES if rx.search(u)), None) or next((s for rx, s in _SCALES if rx.search(r)), 1.0)
    if _INR.search(u) or _INR.search(r):
        return v, v * scale, "INR"
    word = re.sub(r"[^\w ]+", " ", u).lower()
    for rx, _ in _SCALES:
        word = rx.sub(" ", word)
    return v, v * scale, re.sub(r"\s+", " ", word).strip() or None


_MONTHS = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                            "september", "october", "november", "december"])}
_MONTHS.update({m[:3]: i for m, i in list(_MONTHS.items())})
_MONTHS.update({"sept": 9, "जनवरी": 1, "फरवरी": 2, "फ़रवरी": 2, "मार्च": 3, "अप्रैल": 4, "मई": 5, "जून": 6,
                "जुलाई": 7, "अगस्त": 8, "सितंबर": 9, "सितम्बर": 9, "अक्टूबर": 10, "नवंबर": 11, "नवम्बर": 11,
                "दिसंबर": 12, "दिसम्बर": 12, "जानेवारी": 1, "फेब्रुवारी": 2, "एप्रिल": 4, "ऑगस्ट": 8,
                "सप्टेंबर": 9, "ऑक्टोबर": 10, "नोव्हेंबर": 11, "डिसेंबर": 12})


def date_iso(text: str) -> str | None:
    t = ascii_digits(text or "").lower()

    def mk(y, m, d):
        y = int(y)
        y = y + 2000 if y < 100 else y
        try:
            return dt.date(y, int(m), int(d)).isoformat()
        except ValueError:
            return None
    if m := re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", t):
        return mk(m[1], m[2], m[3])
    if m := re.search(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4}|\d{2})\b", t):
        return mk(m[3], m[2], m[1])
    if m := re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([^\s\d,.]+)[,.]?\s+(\d{4})", t):
        if m[2] in _MONTHS:
            return mk(m[3], _MONTHS[m[2]], m[1])
    if m := re.search(r"(?<!\S)([^\s\d,.]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", t):
        if m[1] in _MONTHS:
            return mk(m[3], _MONTHS[m[1]], m[2])
    return None


def fiscal_year(text) -> str | None:
    m = re.search(r"\b((?:19|20)\d{2})\s*[-–/]\s*(\d{2,4})\b", ascii_digits(str(text or "")))
    if m and (int(m[2][-2:]) - int(m[1][-2:])) % 100 == 1:
        return f"{m[1]}-{m[2][-2:]}"
    return None


def _norm(s: str) -> str:
    return re.sub(r"[\W_]+", "", (s or "").lower())


def _s(x, n: int = 300) -> str | None:
    if x is None or isinstance(x, (dict, list)):
        return None
    x = str(x).strip()
    return x[:n] if x and x.lower() not in ("null", "none", "n/a", "na", "-") else None


# ================================================================== jurisdictions
_UT = {"Delhi", "Jammu and Kashmir", "Ladakh", "Puducherry", "Chandigarh", "Andaman and Nicobar Islands",
       "Lakshadweep", "Dadra and Nagar Haveli and Daman and Diu"}
_EXTRA_ALIASES = {"Jammu and Kashmir": ["j&k", "jammu & kashmir", "jammu and kashmir"], "Odisha": ["orissa"],
                  "Andhra Pradesh": ["andhra"], "Delhi": ["nct of delhi", "new delhi"], "Puducherry": ["pondicherry"],
                  "Tamil Nadu": ["tamilnadu"], "Uttarakhand": ["uttaranchal"]}


@lru_cache
def _states() -> dict[str, list[str]]:
    from poc.datasets.bharat import STATES  # the adapter's fixed list (36 states / UTs)

    return {s: sorted({s.lower(), *_EXTRA_ALIASES.get(s, [])}) for s in STATES}


def jid(state: str | None) -> str | None:
    if not state:
        return None
    if state.strip().lower() in ("central", "india", "government of india", "union", "centre", "center"):
        return "J:central"
    n = _norm(state)
    for s, al in _states().items():
        if n == _norm(s) or any(n == _norm(a) for a in al):
            return "J:" + slug(s)
    return None


def states_in(text: str) -> set[str]:
    tl = f" {text.lower()} "
    out = set()
    for s, al in _states().items():
        if any(re.search(r"(?<![\w])" + re.escape(a) + r"(?![\w])", tl) for a in al):
            out.add("J:" + slug(s))
    return out


# ================================================================== ingest: base (no LLM)
def _scheme_from_title(title: str) -> tuple[str, str | None]:
    m = re.search(r"\(([A-Z][A-Z0-9\-&.]{1,15})\)\s*$", title or "")
    return (title[: m.start()].strip(), m.group(1)) if m else ((title or "").strip(), None)


def step_base(docs: list[str], info: dict, pg: pd.DataFrame) -> dict:
    npages = pg.groupby("doc_name").size().to_dict()
    jur = [["J:central", "central", "Government of India", ["central", "union", "government of india", "centre"]]]
    jur += [["J:" + slug(s), "ut" if s in _UT else "state", s, al] for s, al in _states().items()]
    drows, arows, srows = [], [], []
    for d in docs:
        i = info[d]
        j = jid(i.get("state"))
        drows.append([scope_id(d), d, i.get("title"), i.get("url"), i.get("host"), i.get("kind"),
                      list(i.get("topics") or []), i.get("language"), bool(i.get("bilingual")),
                      bool(i.get("legacy_font")), j, i.get("department"), int(npages.get(d, 0)),
                      None, None, None, None, None, None, None, None, None, 1])
        if i.get("department"):
            arows.append([f"AU:{d}:base", _norm(i["department"]), i["department"], i["department"], None, j,
                          "doc_info", d, 1])
        if i.get("kind") == "scheme" and i.get("title"):  # myScheme pages: the title is the scheme name
            name, acr = _scheme_from_title(i["title"])
            srows.append([f"SC:{d}:base", d, 1, name, name, acr, [x for x in (acr,) if x], i.get("department"), j,
                          "doc_info", []])
    con = rw()
    con.execute("DELETE FROM jurisdiction")
    con.executemany("INSERT INTO jurisdiction VALUES (?,?,?,?)", jur)
    con.execute("DELETE FROM document")
    con.executemany(f"INSERT INTO document VALUES ({','.join('?' * 23)})", drows)
    con.execute("DELETE FROM authority WHERE source = 'doc_info'")
    if arows:
        con.executemany("INSERT INTO authority VALUES (?,?,?,?,?,?,?,?,?)", arows)
    con.execute("DELETE FROM scheme WHERE source = 'doc_info'")
    if srows:
        con.executemany("INSERT INTO scheme VALUES (?,?,?,?,?,?,?,?,?,?,?)", srows)
    con.close()
    return {"documents": len(drows), "jurisdictions": len(jur), "authority_seed": len(arows),
            "scheme_seed": len(srows), "base_llm_prompt_tokens": 0, "base_llm_completion_tokens": 0}


# ================================================================== ingest: LLM extraction
DOC_PART = """ "doc": {"printed_title": "<the document's own title as printed, in its language>",
         "title_en": "<English title / gloss>",
         "doc_type": "scheme page|budget speech|budget document|circular|order|notification|office memorandum|notice|report|policy|form|faq|tender|recruitment|other",
         "issuer": "<issuing ministry / department / office, English>", "jurisdiction": "<state / UT in English, or 'central'>",
         "reference_no": "<order / circular / F.No. / notification number as written, or null>",
         "issued_on": "<issue date as written, or null>", "fiscal_year": "<e.g. 2024-25, or null>"},
"""

EXTRACT_SYS = """You extract typed records from one part of an Indian government document (scheme page, budget
document or speech, circular, government order, notification, income-tax notice, report, form). The text may be in
English, Hindi, Marathi or another Indian language. Be exact and conservative: include only what this text
explicitly states.
Return only JSON (omit keys that have no records):
{
DOC_PART "schemes": [{"name": "<scheme / yojana / programme name as written>", "name_en": "<English name or gloss>",
              "acronym": "<or null>", "authority": "<administering body, English>", "jurisdiction": "<state / UT or 'central'>"}],
 "authorities": [{"name": "<as written>", "name_en": "<English>", "kind": "ministry|department|directorate|board|corporation|commission|office|district|university|other",
                  "jurisdiction": "<state / UT or 'central'>"}],
 "benefits": [{"scheme": "<scheme name_en or null>", "kind": "cash|subsidy|loan|interest subvention|in-kind|insurance|fee waiver|scholarship|pension|award|other",
               "amount": "<as written, e.g. 'Rs. 5 lakh', '₹2,000/-', '50%'>", "unit": "<currency / scale as written, e.g. 'Rs', 'Rs lakh', '%'>",
               "per": "<per month / per year / per instalment / one-time / ... or null>", "recipient": "<who gets it, English>", "quote": "<verbatim span>"}],
 "eligibility": [{"scheme": "<or null>", "kind": "age|income|residence|category|gender|occupation|landholding|education|disability|service|other",
                  "include": true, "text": "<the condition in English>", "quote": "<verbatim span>"}],
 "procedure": [{"scheme": "<or null>", "kind": "step|document|form|where_to_apply|mode|fee|authority|other", "step": <int or null>,
                "text": "<the step / required document / office, English>", "quote": "<verbatim span>"}],
 "dates": [{"subject": "<what the date is for, English>", "kind": "launch|deadline|effective|issued|event|period|other",
            "date": "<as written>", "quote": "<verbatim span>"}],
 "budget_lines": [{"head": "<head / item / scheme name, English>", "code": "<head code / demand no. or null>", "fy": "<e.g. 2024-25>",
                   "estimate": "BE|RE|Actual|other", "amount": "<cell as written>", "unit": "<unit from the table header or text, e.g. 'Rs lakh', '₹ thousand', '₹ crore'>",
                   "quote": "<the row as written>"}],
 "tax": [{"tax": "income tax|GST|stamp duty|excise|professional tax|other", "section": "<section / rule or null>",
          "rule": "<the provision in English>", "rate": "<as written or null>", "limit": "<threshold / limit as written or null>",
          "ay": "<assessment year or null>", "quote": "<verbatim span>"}],
 "figures": [{"subject": "<what is counted or measured, English>", "value": "<as written>", "unit": "<as written: %, Rs crore, persons, posts, items ...>",
              "period": "<year / date or null>", "quote": "<verbatim span>"}]
}
Rules:
- quote: copied verbatim from the text in its original language and script, <= 40 words; cut with "..." but never paraphrase.
- Numbers exactly as written (keep Indian digit grouping such as 1,00,000 and Devanagari digits); never convert units.
- A budget / statistical table's unit is usually in its header or caption ("Rs. in lakh", "₹ crore"): always carry it into "unit".
- Names and criteria in English (translate / gloss when the source is not English); keep official names recognisable.
- Use "figures" for any other stated quantity (counts, shares, rates, totals) not covered by the other types.
- At most 50 records in total for this part; prefer specific facts (amounts, limits, dates, conditions, steps)."""

CARD_SYS = """From the first pages of an Indian government document (any Indian language), extract its identity.
Return only JSON:
{
DOC_PART "schemes": [{"name": "<as written>", "name_en": "<English>", "acronym": "<or null>"}],
 "authorities": [{"name": "<as written>", "name_en": "<English>", "kind": "ministry|department|directorate|board|corporation|commission|office|district|university|other"}]
}
Names in English where the source is not English. Only what the text states."""


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


def _balance(t: str) -> str:
    """Close a JSON text truncated by the token limit: drop the last incomplete element, close brackets."""
    i = t.find("{")
    if i < 0:
        return t
    t = t[i:]
    stack, in_str, esc, last_ok = [], False, False, 0
    for j, ch in enumerate(t):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            if len(stack) >= 1:
                last_ok = j + 1
    if not stack:
        return t
    cut = t[:last_ok].rstrip().rstrip(",")
    depth = []
    in_str = esc = False
    for ch in cut:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth.append(ch)
        elif ch in "}]" and depth:
            depth.pop()
    return cut + "".join("}" if c == "{" else "]" for c in reversed(depth))


def _parse(text: str):
    try:
        js = llm.parse_json(text)
        if isinstance(js, dict):
            return js, False
    except Exception:
        pass
    try:
        js = json.loads(_balance(text.strip().removeprefix("```json").removeprefix("```")))
        if isinstance(js, dict):
            return js, True
    except Exception:
        pass
    return None, False


async def _call(doc: str, unit: str, msgs: list[dict], max_tokens: int, usage_acc: list, fails: list) -> dict | None:
    cpath = CACHE / f"{doc}__{unit}.json"
    if cpath.exists():
        return json.loads(cpath.read_text())
    for attempt in range(2):
        out = await chat(msgs, tag=TAG, max_tokens=max_tokens, temperature=0.0)
        usage_acc.append(out["usage"])
        js, repaired = _parse(out["content"])
        if js is not None:
            js["_repaired"] = repaired
            cpath.write_text(json.dumps(js, ensure_ascii=False))
            return js
    fails.append(f"{doc}:{unit}")
    return None


def _header(i: dict) -> str:
    return f"Document: {i.get('title') or '-'} | {i.get('url') or '-'} | language: {i.get('language')}"


async def _extract_window(doc, k, n, win, i, usage_acc, fails):
    body = "\n\n".join(f"[page {p}]\n{t}" for p, t in win)
    sys = EXTRACT_SYS.replace("DOC_PART", DOC_PART if k == 0 else "")
    msgs = [{"role": "system", "content": sys},
            {"role": "user", "content": f"{_header(i)} (part {k + 1} of {n})\n\n{body}"}]
    js = await _call(doc, f"w{k}", msgs, EXTRACT_MAX_TOKENS, usage_acc, fails)
    if js is not None:
        js["_pages"] = [p for p, _ in win]
    return js


async def _extract_card(doc, first, i, usage_acc, fails):
    """first: the document's first CARD_PAGES pages; at most CARD_CHARS characters are shown."""
    body = "\n\n".join(f"[page {p}]\n{t}" for p, t in first)[:CARD_CHARS]
    msgs = [{"role": "system", "content": CARD_SYS.replace("DOC_PART", DOC_PART)},
            {"role": "user", "content": f"{_header(i)}\n\n{body}"}]
    js = await _call(doc, "card", msgs, 1200, usage_acc, fails)
    if js is not None:
        js["_pages"] = [p for p, _ in first]
    return js


def _locate(quote: str, win: dict[int, str], default: int) -> int:
    q = _norm(ascii_digits(str(quote or "").split("...")[0]))[:60]
    if len(q) >= 10:
        for p, t in win.items():
            if q in _norm(ascii_digits(t)):
                return p
    return default


def _lst(js: dict, key: str) -> list[dict]:
    v = js.get(key) or []
    return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []


def _records(doc: str, k, js: dict, texts: dict[int, str], doc_j: str | None) -> dict[str, list]:
    pages = js.get("_pages") or [min(texts)]
    win = {p: texts.get(p, "") for p in pages}
    p0 = pages[0]
    out = {t: [] for t in FACT_TABLES}
    loc = lambda x: _locate(x.get("quote") or x.get("name") or "", win, p0)  # noqa: E731
    q = lambda x: _s(x.get("quote"), 1200)  # noqa: E731
    for n, x in enumerate(_lst(js, "schemes")):
        name = _s(x.get("name"))
        name_en = _s(x.get("name_en")) or name
        if not name_en:
            continue
        acr = _s(x.get("acronym"), 30)
        al = sorted({a for a in (name, name_en, acr) if a})
        out["scheme"].append([f"SC:{doc}:{k}:{n}", doc, loc(x), name or name_en, name_en, acr, al,
                              _s(x.get("authority")), jid(_s(x.get("jurisdiction"))) or doc_j, "llm", []])
    for n, x in enumerate(_lst(js, "authorities")):
        nm = _s(x.get("name_en")) or _s(x.get("name"))
        if nm:
            out["authority"].append([f"AU:{doc}:{k}:{n}", _norm(nm), _s(x.get("name")) or nm, nm, _s(x.get("kind"), 40),
                                     jid(_s(x.get("jurisdiction"))) or doc_j, "llm", doc, loc(x)])
    for n, x in enumerate(_lst(js, "benefits")):
        raw = _s(x.get("amount"), 200)
        v, vn, un = normalise(raw, x.get("unit"))
        if raw or q(x):
            out["benefit"].append([f"BN:{doc}:{k}:{n}", None, _s(x.get("scheme")), _s(x.get("kind"), 40), raw, v,
                                   _s(x.get("unit"), 60), vn, un, _s(x.get("per"), 80), _s(x.get("recipient")), q(x),
                                   doc, loc(x)])
    for n, x in enumerate(_lst(js, "eligibility")):
        if _s(x.get("text")):
            inc = x.get("include")
            out["eligibility"].append([f"EL:{doc}:{k}:{n}", None, _s(x.get("scheme")), _s(x.get("kind"), 40),
                                       inc if isinstance(inc, bool) else True, _s(x.get("text"), 1000), q(x), doc, loc(x)])
    for n, x in enumerate(_lst(js, "procedure")):
        if _s(x.get("text")):
            st = x.get("step")
            st = int(st) if isinstance(st, (int, float)) and not isinstance(st, bool) else None
            out["requirement"].append([f"RQ:{doc}:{k}:{n}", None, _s(x.get("scheme")), _s(x.get("kind"), 40), st,
                                       _s(x.get("text"), 1000), q(x), doc, loc(x)])
    for n, x in enumerate(_lst(js, "dates")):
        d = _s(x.get("date"), 120)
        if d:
            out["date_event"].append([f"DT:{doc}:{k}:{n}", None, _s(x.get("subject")), _s(x.get("kind"), 40), d,
                                      date_iso(d), q(x), doc, loc(x)])
    for n, x in enumerate(_lst(js, "budget_lines")):
        raw = _s(x.get("amount"), 120)
        if not raw:
            continue
        v, vn, un = normalise(raw, x.get("unit"))
        out["budget_line"].append([f"BL:{doc}:{k}:{n}", doc_j, _s(x.get("head")), _s(x.get("code"), 80),
                                   fiscal_year(x.get("fy")) or _s(x.get("fy"), 40), _s(x.get("estimate"), 20), raw, v,
                                   _s(x.get("unit"), 60), vn, un, q(x), doc, loc(x)])
    for n, x in enumerate(_lst(js, "tax")):
        if _s(x.get("rule")) or _s(x.get("rate")):
            rate, lim = _s(x.get("rate"), 120), _s(x.get("limit"), 120)
            _, vn, un = normalise(rate or lim, rate or lim)
            out["tax_provision"].append([f"TX:{doc}:{k}:{n}", _s(x.get("tax"), 60), _s(x.get("section"), 60),
                                         _s(x.get("rule"), 1000), rate, lim, vn, un,
                                         fiscal_year(x.get("ay")) or _s(x.get("ay"), 40), q(x), doc, loc(x)])
    for n, x in enumerate(_lst(js, "figures")):
        raw = _s(x.get("value"), 120)
        if not raw or not _s(x.get("subject")):
            continue
        v, vn, un = normalise(raw, x.get("unit"))
        out["figure"].append([f"FG:{doc}:{k}:{n}", _s(x.get("subject")), raw, v, _s(x.get("unit"), 60), vn, un,
                              _s(x.get("period"), 60), q(x), doc, loc(x)])
    return out


def _link_schemes(tables: dict[str, list], base_schemes: dict[str, list]) -> None:
    """Fill scheme_id on benefit / eligibility / requirement / date rows: match the fact's scheme text (or the
    date subject) against the scheme names / acronyms of the same document."""
    by_doc: dict[str, list] = {}
    for r in tables["scheme"]:
        by_doc.setdefault(r[1], []).append(r)
    for d, rows in base_schemes.items():
        by_doc.setdefault(d, []).extend(rows)
    col = {"benefit": (2, 1, 12), "eligibility": (2, 1, 7), "requirement": (2, 1, 7), "date_event": (2, 1, 7)}
    for t, (txt_i, sid_i, doc_i) in col.items():
        for r in tables[t]:
            cands = by_doc.get(r[doc_i]) or []
            if not cands:
                continue
            txt = _norm(r[txt_i] or "")
            hit = None
            if txt:
                hit = next((s for s in cands if any(_norm(a) and (_norm(a) in txt or txt in _norm(a))
                                                    for a in (s[4], s[3], s[5]) if a)), None)
            if hit is None and len({s[4] for s in cands}) == 1 and t != "date_event":
                hit = cands[0]  # single-scheme document (e.g. a myScheme page)
            if hit is not None:
                r[sid_i] = hit[0]


def _insert(con, table: str, rows: list[list]) -> None:
    """Bulk insert via a DataFrame (executemany is row-by-row in DuckDB; full Bharat runs write ~10^5-10^6 rows)."""
    if not rows:
        return
    cols = [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]
    df = pd.DataFrame(rows, columns=cols).astype(object).where(lambda x: pd.notna(x), None)
    con.register("_ins", df)
    con.execute(f"INSERT INTO {table} SELECT * FROM _ins")
    con.unregister("_ins")


def _extract_set() -> set[str] | None:
    p = os.getenv("GOV_EXTRACT_DOCS")
    if not p:
        return None
    return {l.strip() for l in Path(p).expanduser().read_text().splitlines() if l.strip()}


def step_extract(pg: pd.DataFrame, only: list[str] | None = None, cache_only: bool = False) -> dict:
    from poc import data

    t0 = time.perf_counter()
    info = data.doc_info()
    full_set = _extract_set()
    card_pass = os.getenv("GOV_CARD_PASS", "1") != "0"
    CACHE.mkdir(exist_ok=True)
    jobs, texts_of, skipped_legacy, card_docs, langs = [], {}, [], [], Counter()
    for d, g in sorted(pg.groupby("doc_name"), key=lambda kv: kv[0]):
        if only and d not in only:
            continue
        i = info.get(d, {})
        if i.get("legacy_font"):
            skipped_legacy.append(d)
            continue
        g = g.sort_values("page")
        texts_of[d] = dict(zip(g.page.astype(int), g.text))
        if full_set is not None and d not in full_set:
            if card_pass and (not cache_only or (CACHE / f"{d}__card.json").exists()):
                jobs.append(("card", d, 0, 1, list(texts_of[d].items())[:CARD_PAGES], i))
                card_docs.append(d)
            continue
        langs[i.get("language")] += 1
        wins = _windows(list(texts_of[d].items()))
        for k, win in enumerate(wins):
            if cache_only and not (CACHE / f"{d}__w{k}.json").exists():
                continue
            jobs.append(("full", d, k, len(wins), win, i))
    usage_acc, fails = [], []
    print(f"gov extract: {len(jobs)} LLM units over {len(texts_of)} documents "
          f"({len(card_docs)} card-only, {len(skipped_legacy)} legacy-font skipped)", flush=True)

    async def run():
        done = 0

        async def one(j):
            nonlocal done
            kind, d, k, n, win, i = j
            r = await (_extract_card(d, win, i, usage_acc, fails) if kind == "card"
                       else _extract_window(d, k, n, win, i, usage_acc, fails))
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(jobs)} units, {len(usage_acc)} LLM calls, "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)
            return r
        return await asyncio.gather(*(one(j) for j in jobs))

    results = asyncio.run(run())
    tables = {t: [] for t in FACT_TABLES}
    docfields: dict[str, dict] = {}
    repaired = 0
    for (kind, d, k, _, _, i), js in zip(jobs, results):
        if not js:
            continue
        repaired += bool(js.get("_repaired"))
        doc_j = jid(i.get("state"))
        dj = js.get("doc") if isinstance(js.get("doc"), dict) else None
        if dj:
            doc_j = doc_j or jid(_s(dj.get("jurisdiction")))
            docfields[d] = {**dj, "_mode": kind}
        recs = _records(d, k if kind == "full" else "card", js, texts_of[d], doc_j)
        for t in FACT_TABLES:
            tables[t] += recs[t]
        docfields.setdefault(d, {"_mode": kind})
    con = rw()
    base_schemes: dict[str, list] = {}
    for r in con.execute("SELECT * FROM scheme WHERE source = 'doc_info'").fetchall():
        base_schemes.setdefault(r[1], []).append(list(r))
    _link_schemes(tables, base_schemes)
    for t in FACT_TABLES:
        src = " AND source = 'llm'" if t in ("scheme", "authority") else ""
        if only:
            con.execute(f"DELETE FROM {t} WHERE list_contains(?, doc_name){src}", [list(only)])
        else:
            con.execute(f"DELETE FROM {t} WHERE true{src}")
        _insert(con, t, tables[t])
    upd = []
    for d, f in docfields.items():
        issued = _s(f.get("issued_on"), 80)
        upd.append([scope_id(d), _s(f.get("printed_title")), _s(f.get("title_en")), _s(f.get("doc_type"), 40),
                    _s(f.get("reference_no"), 120), issued, date_iso(issued or ""), fiscal_year(f.get("fiscal_year")),
                    _s(f.get("issuer")), f["_mode"], jid(_s(f.get("jurisdiction")))])
    upd += [[scope_id(d), *([None] * 8), "skipped_legacy_font", None] for d in skipped_legacy]
    if upd:
        u = pd.DataFrame(upd, columns=["id", "printed_title", "title_en", "doc_type", "reference_no", "issued_on",
                                       "issued_iso", "fiscal_year", "issuer", "extracted", "jur"]).astype(object)
        con.register("_upd", u.where(pd.notna(u), None))
        con.execute("UPDATE document SET printed_title = u.printed_title, title_en = u.title_en, doc_type = u.doc_type, "
                    "reference_no = u.reference_no, issued_on = u.issued_on, issued_iso = u.issued_iso, "
                    "fiscal_year = u.fiscal_year, issuer = u.issuer, extracted = u.extracted, "
                    "jurisdiction_id = coalesce(document.jurisdiction_id, u.jur) FROM _upd u WHERE document.id = u.id")
        con.unregister("_upd")
    # same_as: scheme rows in different documents with the same normalised English name (cross-site copies)
    sch = con.execute("SELECT id, doc_name, name_en FROM scheme").fetchall()
    groups: dict[str, list] = {}
    for sid, d, n in sch:
        if n and len(_norm(n)) >= 8:
            groups.setdefault(_norm(n), []).append((sid, d))
    upd = []  # (same_as ids, scheme id)
    for members in groups.values():
        if len({d for _, d in members}) > 1:
            for sid, d in members:
                upd.append([[o for o, od in members if od != d], sid])
    con.execute("UPDATE scheme SET same_as = []")
    if upd:
        con.register("_sa", pd.DataFrame(upd, columns=["same_as", "id"]))
        con.execute("UPDATE scheme SET same_as = s.same_as FROM _sa s WHERE scheme.id = s.id")
        con.unregister("_sa")
    con.close()
    return {"extract_units": len(jobs), "documents_extracted": len(texts_of) - len(card_docs),
            "documents_card_only": len(card_docs), "documents_skipped_legacy_font": len(skipped_legacy),
            "extract_languages": dict(langs),
            **{f"{t}_records": len(v) for t, v in tables.items()},
            "schemes_linked_same_as": len(upd), "units_failed": len(fails), "units_json_repaired": repaired,
            "extract_llm_calls": len(usage_acc),
            "extract_llm_prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage_acc),
            "extract_llm_completion_tokens": sum(u.get("completion_tokens", 0) for u in usage_acc),
            "extract_wall_s": round(time.perf_counter() - t0, 1)}


# ================================================================== query: 1. resolve the documents
_STOP = set("the a an of in for to and or is was were what which who whom whose how much many by on at as from with "
            "its it this that these those are be been did does do under according per into within than their there "
            "s any all".split())
_HOST_STOP = {"www", "gov", "in", "nic", "org", "com", "co", "ac", "res", "net"}


def _toks(s: str) -> list[str]:
    return [t for t in re.findall(r"[^\W_]+", ascii_digits(s or "").lower()) if t not in _STOP and len(t) > 1]


@lru_cache
def _cards() -> dict:
    """Per-document name card (document fields + scheme / authority names) and a BM25 index over it."""
    c = ro()
    cur = c.execute("SELECT doc_name, title, url, host, jurisdiction_id, department, printed_title, title_en, "
                    "doc_type, reference_no, issued_on, fiscal_year, issuer, language FROM document ORDER BY doc_name")
    cols = [x[0] for x in cur.description]
    docs = [dict(zip(cols, r)) for r in cur.fetchall()]
    names: dict[str, list[str]] = {}
    # Only names from the first pages (what the card pass also sees), so documents with full extraction do not get
    # richer cards than card-only ones when GOV_EXTRACT_DOCS caps the full extraction.
    for d, n, ne, a in c.execute("SELECT doc_name, name, name_en, acronym FROM scheme WHERE page <= ?",
                                 [CARD_PAGES]).fetchall():
        names.setdefault(d, []).extend(x for x in (n, ne, a) if x)
    for d, n, ne in c.execute("SELECT doc_name, name, name_en FROM authority WHERE page <= ?", [CARD_PAGES]).fetchall():
        names.setdefault(d, []).extend(x for x in (n, ne) if x)
    jname = dict(c.execute("SELECT id, name_en FROM jurisdiction").fetchall())
    rows, toks = [], []
    for r in docs:
        u = urlparse(r["url"] or "")
        path = re.sub(r"\.(pdf|html?|php|aspx?|docx?)$", "", unquote(u.path or ""), flags=re.I)
        host = " ".join(x for x in (r["host"] or "").split(".") if x not in _HOST_STOP)
        nm = list(dict.fromkeys(names.get(r["doc_name"], [])))[:40]
        fields = [r["title"], r["printed_title"], r["title_en"], path, host, jname.get(r["jurisdiction_id"]),
                  r["department"], r["doc_type"], r["reference_no"], r["issued_on"], r["fiscal_year"], r["issuer"], *nm]
        text = " | ".join(str(x) for x in fields if x)
        r["names"] = [x for x in [r["title_en"], r["title"], r["printed_title"], *nm] if x]
        r["card"] = text
        rows.append(r)
        toks.append(_toks(text))
    df: Counter = Counter()
    post: dict[str, list] = {}
    for i, ts in enumerate(toks):
        tf = Counter(ts)
        for t, n in tf.items():
            df[t] += 1
            post.setdefault(t, []).append((i, n))
    lens = [len(t) for t in toks]
    return {"rows": rows, "post": post, "df": df, "lens": lens, "avg": (sum(lens) / len(lens)) if lens else 1.0}


def _bm25(question: str, k1: float = 1.2, b: float = 0.75) -> dict[int, float]:
    cd = _cards()
    n = len(cd["rows"])
    sc: dict[int, float] = {}
    for t in set(_toks(question)):
        if t not in cd["post"]:
            continue
        idf = math.log(1 + (n - cd["df"][t] + 0.5) / (cd["df"][t] + 0.5))
        for i, tf in cd["post"][t]:
            sc[i] = sc.get(i, 0.0) + idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * cd["lens"][i] / cd["avg"]))
    return sc


_ACR = re.compile(r"\b[A-Z][A-Z0-9&\-]{2,}\b")


def match_documents(question: str, top: int = 12) -> list[tuple[float, dict, list[str]]]:
    """Rules over the name cards: BM25 + bonuses for a full name phrase / acronym / reference number found in the
    question and for the question's state; a document of a different named state is down-weighted."""
    cd = _cards()
    base = _bm25(question)
    if not base:
        return []
    qn = _norm(question)
    qacr = set(_ACR.findall(question))
    qstates = states_in(question)
    pool = sorted(base, key=lambda i: -base[i])[:200]
    out = []
    for i in pool:
        r = cd["rows"][i]
        s, why = base[i], []
        for nm in r["names"]:
            nn = _norm(nm)
            if len(nn) >= 12 and nn in qn:
                s += 6
                why.append(f"name:{nm[:40]}")
                break
        acrs = {a for nm in r["names"] for a in _ACR.findall(nm)} & qacr
        if acrs:
            s += 2 * len(acrs)
            why.append("acronym:" + ",".join(sorted(acrs)))
        if r["reference_no"] and len(_norm(r["reference_no"])) >= 5 and _norm(r["reference_no"]) in qn:
            s += 6
            why.append("ref_no")
        if qstates:
            if r["jurisdiction_id"] in qstates:
                s += 2
                why.append("state")
            elif r["jurisdiction_id"] and r["jurisdiction_id"] != "J:central":
                s *= 0.6
        out.append((s, r, why))
    out.sort(key=lambda x: -x[0])
    return out[:top]


RESOLVE_SYS = """You map a question about Indian government documents to the document(s) it is about. The question
names its entity explicitly (scheme, department / office, state, order or circular, fiscal year, publication).
You get candidate documents (id | title | English title | site | jurisdiction | type | ref no. | FY | language |
names). Pick the document(s) that the question is about, best first: at most 3, and more than one only when
several candidates are copies of the same document or the question spans them. Match state, year, order / notice
number and publication exactly. Return JSON {"documents": ["<id>", ...]}."""


async def resolve(question: str, usage: Usage) -> dict:
    hits = match_documents(question)
    if not hits:
        return {"entity": None, "scope": [], "how": "none"}
    top = hits[0]
    strong = any(w.startswith(("name:", "ref_no", "acronym:")) for w in top[2])
    decisive = strong and (len(hits) == 1 or top[0] >= 1.3 * hits[1][0])
    if decisive:
        chosen, how = [top[1]["doc_name"]], "rules"
    else:
        lines = []
        for s, r, why in hits:
            nm = "; ".join(dict.fromkeys(x for x in r["names"][3:8]))[:200]
            lines.append(" | ".join(str(x or "-") for x in (
                r["doc_name"], (r["title"] or "")[:120], (r["title_en"] or "")[:120], r["host"],
                r["jurisdiction_id"], r["doc_type"], r["reference_no"], r["fiscal_year"], r["language"], nm)))
        out = await chat([{"role": "system", "content": RESOLVE_SYS},
                          {"role": "user", "content": f"Question: {question}\n\nCandidates:\n" + "\n".join(lines)}],
                         tag=ANSWER_TAG, max_tokens=200, temperature=0.0)
        usage.add(out)
        ids = [h[1]["doc_name"] for h in hits]
        try:
            chosen = [d for d in llm.parse_json(out["content"]).get("documents", []) if d in ids]
        except Exception:
            chosen = [d for d in dict.fromkeys(re.findall(r"[0-9a-f]{24}", out["content"])) if d in ids]
        chosen = chosen[:MAX_SCOPE] or [top[1]["doc_name"]]
        how = "llm"
    r0 = next(h[1] for h in hits if h[1]["doc_name"] == chosen[0])
    return {"entity": r0["title_en"] or r0["title"], "scope": [scope_id(d) for d in chosen], "how": how,
            "candidates": [h[1]["doc_name"] for h in hits[:5]]}


# ================================================================== query: 2. catalogue (value-free)
EMPTY_CATALOGUE = "(no structured records extracted for the resolved document(s))"
TYPE_LIMIT = {"S": 15, "B": 25, "E": 30, "P": 30, "D": 25, "L": 40, "X": 20, "F": 40}
TYPES = {  # key prefix -> (table, label SQL, header)
    "S": ("scheme", "coalesce(name_en, name) || coalesce(' (' || acronym || ')', '')", "schemes"),
    "B": ("benefit", "coalesce(scheme, '-') || ' / ' || coalesce(kind, '-') || ' / for: ' || coalesce(recipient, '-') "
                     "|| ' / per: ' || coalesce(per, '-') || ' / unit: ' || coalesce(unit_norm, unit_raw, '-')", "benefits"),
    "E": ("eligibility", "coalesce(scheme, '-') || ' / ' || coalesce(kind, '-') || ' / ' || "
                         "CASE WHEN is_inclusion THEN 'inclusion' ELSE 'exclusion' END", "eligibility criteria"),
    "P": ("requirement", "coalesce(scheme, '-') || ' / ' || coalesce(kind, '-') || coalesce(' / step ' || CAST(step AS VARCHAR), '')",
          "procedure / requirements"),
    "D": ("date_event", "coalesce(subject, '-') || ' / ' || coalesce(kind, '-')", "dates"),
    "L": ("budget_line", "coalesce(head, '-') || coalesce(' [' || head_code || ']', '') || ' / FY ' || "
                         "coalesce(fiscal_year, '?') || ' / ' || coalesce(estimate, '-') || ' / unit: ' || "
                         "coalesce(unit_raw, '-')", "budget lines"),
    "X": ("tax_provision", "coalesce(tax, '-') || coalesce(' s.' || section, '') || coalesce(' / AY ' || assessment_year, '')",
          "tax provisions"),
    "F": ("figure", "coalesce(subject, '-') || coalesce(' / ' || period, '') || ' / unit: ' || coalesce(unit_raw, unit_norm, '-')",
          "figures (other stated quantities)"),
}


def _rank(question: str, rows: list[tuple], limit: int) -> list[tuple]:
    if len(rows) <= limit:
        return rows
    q = set(_toks(question))
    return sorted(rows, key=lambda r: (-len(q & set(_toks(r[1]))), r[2]))[:limit]


def catalogue(scope: list[str], question: str = "") -> tuple[str, dict]:
    c = ro()
    parts, keys, n = [], {}, Counter()
    for sid in scope:
        d = c.execute("SELECT doc_name, title, title_en, doc_type, jurisdiction_id, coalesce(issuer, department), "
                      "fiscal_year, reference_no, language, n_pages, extracted FROM document WHERE id = ?",
                      [sid]).fetchone()
        if not d:
            continue
        doc = d[0]
        parts.append(f"## document {sid}: {d[2] or d[1]} | type: {d[3] or '?'} | jurisdiction: {d[4] or '?'} | "
                     f"issuer: {d[5] or '?'} | FY: {d[6] or '?'} | ref: {d[7] or '-'} | language: {d[8]} | "
                     f"{d[9]} pages | extraction: {d[10] or 'none'}")
        for pre, (tbl, label, head) in TYPES.items():
            rows = c.execute(f"SELECT id, {label}, page FROM {tbl} WHERE doc_name = ? ORDER BY page, id", [doc]).fetchall()
            if not rows:
                continue
            shown = _rank(question, rows, TYPE_LIMIT[pre])
            parts.append(f"{head} ({len(rows)}):")
            for rid, lab, page in sorted(shown, key=lambda r: (r[2], r[0])):
                n[pre] += 1
                key = f"{pre}{n[pre]}"
                keys[key] = (tbl, rid)
                parts.append(f"   {key}: {lab} (p.{page})")
            if len(rows) > len(shown):
                parts.append(f"   ... {len(rows) - len(shown)} more {head} not listed (less related to the question)")
    if not keys:
        return "\n".join(parts + [EMPTY_CATALOGUE]) if parts else "", {"keys": {}, "scope": scope}
    return "\n".join(parts), {"keys": keys, "scope": scope}


PLAN_SYS = """You are the query planner of an ontology-grounded QA system over Indian government documents. You do NOT
answer the question. You get the question and a value-free catalogue of the typed records extracted from the
resolved document(s): schemes (S), benefits (B), eligibility criteria (E), procedure / requirements (P), dates (D),
budget lines (L), tax provisions (X) and other stated figures (F), each with a key, a label and a page. Values and
texts are hidden; they will be fetched by SQL from your plan.

Return JSON:
{
 "records": ["<key>", ...],          // records to fetch; "E*" fetches every record of that type
 "vars": {"<name>": "<key>"},        // numeric records used in arithmetic (value normalised: rupees for INR, % as %)
 "calc": {"<name>": "<expression over vars / earlier calc names, + - * / ( ), abs, min, max, round, sum, avg>"},
 "search": ["<query>", ...]          // 1-3 short phrases for text search inside the resolved document(s)
}
Rules:
- Fetch every record that may hold the answer (when unsure, fetch the whole type with "X*").
- Use vars / calc only when the answer needs arithmetic (sum, difference, growth, share); otherwise leave them empty.
  Never put numbers from the question into calc unless the question states them.
- Purpose / explanation / process questions, or questions no record type covers: leave records empty and rely on search.
- Search queries: words likely to appear in the document text (English, or the Hindi / regional term when the
  document is not in English). Always give at least one."""


def default_plan(question: str) -> dict:
    return {"records": [], "vars": {}, "calc": {}, "search": [question]}


# ================================================================== query: 3. SQL + arithmetic
_SELECT = {
    "scheme": ("SELECT id, 'scheme', coalesce(name_en, name), name || coalesce(' (' || acronym || ')', '') || "
               "coalesce(' | authority: ' || authority, '') || coalesce(' | jurisdiction: ' || jurisdiction_id, ''), "
               "NULL, NULL, NULL, doc_name, page FROM scheme"),
    "benefit": ("SELECT id, 'benefit', coalesce(scheme, '-') || ' / ' || coalesce(kind, '-'), amount_raw || "
                "coalesce(' (unit: ' || unit_raw || ')', '') || coalesce(' per ' || per, '') || "
                "coalesce(' | for: ' || recipient, ''), value_norm, unit_norm, evidence_text, doc_name, page FROM benefit"),
    "eligibility": ("SELECT id, 'eligibility', coalesce(scheme, '-') || ' / ' || coalesce(kind, '-') || ' / ' || "
                    "CASE WHEN is_inclusion THEN 'inclusion' ELSE 'exclusion' END, text, NULL, NULL, evidence_text, doc_name, "
                    "page FROM eligibility"),
    "requirement": ("SELECT id, 'procedure', coalesce(scheme, '-') || ' / ' || coalesce(kind, '-') || "
                    "coalesce(' / step ' || CAST(step AS VARCHAR), ''), text, NULL, NULL, evidence_text, doc_name, page FROM requirement"),
    "date_event": ("SELECT id, 'date', coalesce(subject, '-') || ' / ' || coalesce(kind, '-'), date_text || "
                   "coalesce(' (= ' || date_iso || ')', ''), NULL, NULL, evidence_text, doc_name, page FROM date_event"),
    "budget_line": ("SELECT id, 'budget line', coalesce(head, '-') || coalesce(' [' || head_code || ']', '') || ' / FY ' || "
                    "coalesce(fiscal_year, '?') || ' / ' || coalesce(estimate, '-'), amount_raw || "
                    "coalesce(' (unit: ' || unit_raw || ')', ''), value_norm, unit_norm, evidence_text, doc_name, page "
                    "FROM budget_line"),
    "tax_provision": ("SELECT id, 'tax provision', coalesce(tax, '-') || coalesce(' s.' || section, '') || "
                      "coalesce(' / AY ' || assessment_year, ''), coalesce(rule, '') || coalesce(' | rate: ' || rate_raw, '') "
                      "|| coalesce(' | limit: ' || limit_raw, ''), value_norm, unit_norm, evidence_text, doc_name, page "
                      "FROM tax_provision"),
    "figure": ("SELECT id, 'figure', coalesce(subject, '-') || coalesce(' / ' || period, ''), value_raw || "
               "coalesce(' (unit: ' || unit_raw || ')', ''), value_norm, unit_norm, evidence_text, doc_name, page "
               "FROM figure"),
}


def execute(p: dict, keymap: dict) -> dict:
    errors, records, results = [], {}, {}
    keys = (keymap or {}).get("keys") or {}
    if not keymap or not keymap.get("scope"):
        return {"sql": None, "records": {}, "results": {}, "errors": ["no document resolved"]}
    want = [k for k in (p.get("records") or []) if isinstance(k, str)]
    vars_ = {n: k for n, k in (p.get("vars") or {}).items() if isinstance(k, str)} if isinstance(p.get("vars"), dict) else {}
    sel: list[str] = []
    for k in [*want, *vars_.values()]:
        k = k.strip()
        if k.endswith("*") and k[:-1] in TYPES:
            sel += [x for x in keys if x[0] == k[:-1] and x[1:].isdigit()]
        elif k in keys:
            sel.append(k)
        else:
            errors.append(f"unknown key {k}")
    sel = list(dict.fromkeys(sel))
    by_tbl: dict[str, list[str]] = {}
    for k in sel:
        by_tbl.setdefault(keys[k][0], []).append(k)
    c = ro()
    sqls, rec_of_key = [], {}
    for tbl, ks in by_tbl.items():
        ids = [keys[k][1] for k in ks]
        sql = f"{_SELECT[tbl]} WHERE id IN ({','.join('?' * len(ids))}) ORDER BY page, id"
        sqls.append(sql.replace(f"({','.join('?' * len(ids))})", "(" + ",".join(f"'{i}'" for i in ids) + ")"))
        rows = {r[0]: r for r in c.execute(sql, ids).fetchall()}
        for k in ks:
            r = rows.get(keys[k][1])
            if not r:
                continue
            rec = {"id": r[0], "key": k, "type": r[1], "label": r[2], "value": r[3], "value_norm": r[4],
                   "unit_norm": r[5], "quote": r[6], "doc": r[7], "page": r[8]}
            records[r[0]] = rec
            rec_of_key[k] = rec
    env, units = {}, {}
    for name, k in vars_.items():
        rec = rec_of_key.get(k.strip())
        if not rec:
            continue
        if rec["value_norm"] is None:
            errors.append(f"var {name} ({k}) has no numeric value")
            continue
        env[name], units[name] = rec["value_norm"], rec["unit_norm"]
    calc = p.get("calc") if isinstance(p.get("calc"), dict) else {}
    for name, expr in calc.items():
        if not isinstance(expr, str):
            continue
        try:
            v = float(safe_eval(expr, env))
            used = {str(u) for n, u in units.items() if re.search(rf"\b{re.escape(n)}\b", expr)}
            if len(used) > 1 and not re.search(r"[*/]", expr):
                errors.append(f"calc {name} adds values with different units {sorted(used)}")
            # a sum / difference keeps the unit; a ratio or product is reported without one
            unit = next(iter(used)) if len(used) == 1 and not re.search(r"[*/]", expr) else None
            results[name] = {"expr": expr, "value": v, "unit": unit}
            units[name] = unit
            env[name] = v
        except Exception as e:
            errors.append(f"calc {name} = {expr}: {type(e).__name__}: {e}"[:200])
    return {"sql": ";\n".join(sqls) or None, "records": records, "results": results, "errors": errors}


# ================================================================== query: 5. grounded generation
def fmt_value(v: float | None, unit: str | None) -> str:
    if v is None:
        return ""
    if unit == "INR":
        s = f"₹{v:,.2f}".rstrip("0").rstrip(".")
        return f"{s} (= {v / 1e5:,.4g} lakh = {v / 1e7:,.4g} crore)"
    if unit == "%":
        return f"{v:g}%"
    return f"{v:,.6g}" + (f" {unit}" if unit else "")


ANSWER_SYS = """You answer questions about Indian government documents (scheme pages, budgets, circulars, orders,
notifications, tax notices, reports) using ONLY the evidence given: typed records extracted from the resolved
document(s) (IDs in brackets, with page and a verbatim source quote), values computed in code, and text passages
from the same document(s).
- Answer in English. Source text may be Hindi, Marathi or another Indian language: translate faithfully, and you may
  quote the source wording (in its script) as support.
- Give figures with their unit exactly as the source states them (₹ / Rs, lakh, crore, thousand, %), and the
  converted value when it helps (1 lakh = 1,00,000; 1 crore = 100 lakh). Use the computed values for any
  arithmetic; never do arithmetic yourself and never invent a value.
- Records were extracted by a model and can be mislabelled: check them against their quotes and the passages; if the
  passages show the answer more directly, use the passages.
- Check the scheme, state, department, year and order number match the question. If the evidence does not contain
  the answer, say it is not stated in the documents found.
Format: first line "Answer: <direct answer>", then 1-4 short lines of support with citations like [BN:...] (<doc> p.N)
or (<doc> p.N) for passages."""


def evidence_block(question: str, res: dict, ex: dict, passages: list[dict]) -> str:
    docs = ", ".join(doc_of(s) for s in res.get("scope", [])) or "none"
    lines = [f"Question: {question}", f"Resolved document(s): {res.get('entity') or '-'} ({docs})", ""]
    if ex["records"]:
        lines.append("Structured records (SQL over the government-document ontology):")
        for r in ex["records"].values():
            norm = fmt_value(r.get("value_norm"), r.get("unit_norm"))
            q = f' | quote: "{str(r["quote"])[:500]}"' if r.get("quote") else ""
            lines.append(f"- [{r['id']}] {r['type']} / {r['label']}: {str(r['value'] or '')[:1200]}"
                         f"{f' [normalised: {norm}]' if norm and r['type'] != 'scheme' else ''}{q} ({r['doc']} p.{r['page']})")
    else:
        lines.append("Structured records: none matched the plan.")
    if ex["results"]:
        lines.append("Computed in code:")
        for k, v in ex["results"].items():
            lines.append(f"- {k} = {v['expr']} = {fmt_value(v['value'], v.get('unit'))}")
    if ex["errors"]:
        lines.append("Notes: " + "; ".join(ex["errors"]))
    lines.append("")
    lines.append("Passages (text search scoped to the resolved document(s)):")
    for p in passages:
        lines.append(f"--- ({p['doc']} p.{p['page']})\n{p['text']}")
    return "\n".join(lines)
