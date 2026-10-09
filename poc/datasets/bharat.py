"""Bharat Guide: screened Indian government documents (HF ankitjh4/bharat-government-documents).

Research use only: the dataset has no blanket licence (rights stay with each publisher),
so raw text, pages and the generated questions live under POC_DATA_DIR/bharat/ and are
never committed.

The source has no questions; poc.qgen builds them with Claude into bharat/questions.jsonl.

Corpus = documents carrying any of BHARAT_TOPICS (comma-separated raw topic labels,
default schemes,budget,public_finance,income_tax -> ~14.4k docs). If the subset exceeds
BHARAT_MAX_DOCS (default 15000) it is sampled down to BHARAT_SAMPLE_DOCS (default 13000)
with a fixed seed, always keeping every document a question cites.

Documents are web pages / PDF text without real pagination, so build_pages() splits each
body into ~3,000-char pseudo-pages on paragraph boundaries. Very long bodies (budget
volumes, manuals) are capped at BHARAT_MAX_CHARS chars (default 150,000 ≈ 50 pages; 0 = no cap).
"""
import json
import os
import re
import unicodedata
from functools import lru_cache
from urllib.parse import urlparse

import pandas as pd

from poc import config

NAME = "bharat"
ONTOLOGY_DOMAIN = "gov"
JUDGE_CONTEXT = (
    "Questions are about Indian government documents (central and state government web pages, "
    "scheme descriptions from myScheme, budget speeches and budget documents, finance-department "
    "circulars, income-tax notices and public-finance reports). The ground-truth answer is a "
    "specific figure or fact taken from one document. A candidate is correct only if it matches the "
    "gold figure/fact: amounts must match after unit conversion (₹ lakh / crore / thousand, per cent), "
    "and the scheme, state, department and financial year must be the ones asked about. Indian number "
    "formatting (1,00,000 = 100,000 = 1 lakh) is equivalent. Many source documents are in Hindi or "
    "other Indian languages; the question is in English but the answer may be quoted in the source "
    "language or translated -- judge the meaning, not the language or script."
)

REPO_ID = "ankitjh4/bharat-government-documents"
BHARAT_DIR = config.DATA_DIR / "bharat"
RAW_DIR = BHARAT_DIR / "raw"
DB_DIR = BHARAT_DIR / "db"
PAGES_PATH = DB_DIR / "pages.parquet"
QUESTIONS_PATH = BHARAT_DIR / "questions.jsonl"

DEFAULT_TOPICS = "schemes,budget,public_finance,income_tax"
TOPICS = [t.strip() for t in os.getenv("BHARAT_TOPICS", DEFAULT_TOPICS).split(",") if t.strip()]
MAX_DOCS = int(os.getenv("BHARAT_MAX_DOCS", "15000"))
SAMPLE_DOCS = int(os.getenv("BHARAT_SAMPLE_DOCS", "13000"))
MAX_CHARS = int(os.getenv("BHARAT_MAX_CHARS", "150000"))
PAGE_CHARS = 3000
SEED = 20261009

for d in (RAW_DIR, DB_DIR):
    d.mkdir(parents=True, exist_ok=True)


def download() -> None:
    """Parquet shards only (~320 MB). Never the SQLite backups or documents.csv.gz."""
    from huggingface_hub import snapshot_download

    snapshot_download(REPO_ID, repo_type="dataset", local_dir=str(RAW_DIR),
                      allow_patterns=["data/train-*.parquet", "README.md"])
    n = len(list((RAW_DIR / "data").glob("train-*.parquet")))
    print(f"{n} parquet shards in {RAW_DIR / 'data'}")


# ---------------------------------------------------------------- language / state / dept

_SCRIPTS = {
    "deva": r"[ऀ-ॿ]", "bn": r"[ঀ-৿]", "pa": r"[਀-੿]",
    "gu": r"[઀-૿]", "or": r"[଀-୿]", "ta": r"[஀-௿]",
    "te": r"[ఀ-౿]", "kn": r"[ಀ-೿]", "ml": r"[ഀ-ൿ]",
    "ur": r"[؀-ۿ]", "en": r"[A-Za-z]",
}
_SCRIPT_RE = {k: re.compile(v) for k, v in _SCRIPTS.items()}
# Marathi vs Hindi (both Devanagari): a few high-frequency Marathi function words.
_MR = re.compile(r"(?:आहे|आहेत|च्या|साठी|किंवा|यांच्या|करण्यात)")
_HI = re.compile(r"(?:है|हैं|के लिए|किया|तथा|एवं|की)")
# Hindi typed in legacy 8-bit fonts (Kruti Dev etc.) extracts as Latin gibberish: "gS" = है,
# "esa" = में, "vkSj" = और ... Such bodies are Hindi but unreadable as text.
_LEGACY = re.compile(r"(?<![A-Za-z])(?:gS|ds|dk|esa|ls|fd|vkSj|dh|gSa|fy,|rFkk|ugha|djus)(?![A-Za-z])")


def is_legacy_font(text: str) -> bool:
    return len(_LEGACY.findall(text[:6000])) >= 15


def detect_language(text: str) -> tuple[str, bool]:
    """(language code, bilingual) from Unicode script counts on the first 6k chars.

    Cheap heuristic: Devanagari is split hi/mr on function words; legacy-font Hindi (Latin
    gibberish) is reported as hi; Latin-script Indian languages (e.g. Mizo) show as en;
    'bilingual' when the
    second script has >= 25% of the letters (common for Hindi+English notices).
    """
    t = text[:6000]
    counts = {k: len(r.findall(t)) for k, r in _SCRIPT_RE.items()}
    total = sum(counts.values()) or 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    lang = ranked[0][0] if ranked[0][1] else "en"
    if lang == "deva":
        lang = "mr" if len(_MR.findall(t)) > len(_HI.findall(t)) else "hi"
    elif lang == "en" and is_legacy_font(t):
        lang = "hi"
    return lang, ranked[1][1] / total >= 0.25


STATES = {
    "Andhra Pradesh": ["andhra", "ap"], "Arunachal Pradesh": ["arunachal"], "Assam": ["assam"],
    "Bihar": ["bihar"], "Chhattisgarh": ["chhattisgarh", "cg"], "Goa": ["goa"], "Gujarat": ["gujarat"],
    "Haryana": ["haryana", "hry"], "Himachal Pradesh": ["himachal", "hp"], "Jharkhand": ["jharkhand"],
    "Karnataka": ["karnataka", "kar"], "Kerala": ["kerala"], "Madhya Pradesh": ["mp", "madhya"],
    "Maharashtra": ["maharashtra", "mahakosh", "mah"], "Manipur": ["manipur"], "Meghalaya": ["meghalaya"],
    "Mizoram": ["mizoram"], "Nagaland": ["nagaland"], "Odisha": ["odisha", "orissa"],
    "Punjab": ["punjab"], "Rajasthan": ["rajasthan", "raj"], "Sikkim": ["sikkim"],
    "Tamil Nadu": ["tn", "tamilnadu"], "Telangana": ["telangana", "ts"], "Tripura": ["tripura"],
    "Uttar Pradesh": ["up", "uttarpradesh"], "Uttarakhand": ["uttarakhand", "uk"],
    "West Bengal": ["wb", "westbengal", "bengal"], "Delhi": ["delhi"], "Jammu and Kashmir": ["jk", "jammu"],
    "Ladakh": ["ladakh"], "Puducherry": ["puducherry", "py"], "Chandigarh": ["chandigarh"],
    "Andaman and Nicobar Islands": ["andaman"], "Lakshadweep": ["lakshadweep"],
    "Dadra and Nagar Haveli and Daman and Diu": ["dnh", "daman"],
}
_STATE_NAME_RE = re.compile(r"\b(" + "|".join(re.escape(s) for s in STATES) + r")\b", re.I)
_DEPT_RE = re.compile(r"\b((?:Ministry|Department|Directorate|Commissionerate|Office) of [A-Z][\w&,'\- ]{2,60}?)"
                      r"(?=[,.;:()|\n]| Government| Govt| in | under |$)")
_CITY_STATE = {"hyderabad": "Telangana", "mumbai": "Maharashtra", "chennai": "Tamil Nadu",
               "kolkata": "West Bengal", "bengaluru": "Karnataka", "bangalore": "Karnataka",
               "ahmedabad": "Gujarat", "lucknow": "Uttar Pradesh", "jaipur": "Rajasthan",
               "bhopal": "Madhya Pradesh", "patna": "Bihar", "guwahati": "Assam"}


def infer_state(host: str, title: str) -> str | None:
    labels = host.lower().split(".")
    for state, keys in STATES.items():
        for k in keys:
            # short codes must be a whole host label; longer keys may be a substring of one
            if any(lab == k or (len(k) > 4 and k in lab) for lab in labels):
                return state
    for city, state in _CITY_STATE.items():
        if city in host.lower():
            return state
    m = _STATE_NAME_RE.search(title or "")
    if m:
        return next(s for s in STATES if s.lower() == m.group(1).lower())
    return None


def infer_department(title: str, text: str) -> str | None:
    m = _DEPT_RE.search(f"{title}\n{text[:1500]}")
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------- loading / corpus

@lru_cache
def _all() -> pd.DataFrame:
    files = sorted((RAW_DIR / "data").glob("train-*.parquet"))
    if not files:
        raise SystemExit(f"no parquet shards in {RAW_DIR / 'data'}; run download() first")
    cols = ["document_id", "title", "text", "url", "kind", "trust", "quality", "topics"]
    df = pd.concat((pd.read_parquet(f, columns=cols) for f in files), ignore_index=True)
    df["topics"] = df.topics.apply(lambda ts: [str(t) for t in ts] if ts is not None else [])
    return df


def _question_docs() -> set[str]:
    if not QUESTIONS_PATH.exists():
        return set()
    return {json.loads(l)["doc_name"] for l in QUESTIONS_PATH.open() if l.strip()}


@lru_cache
def _subset() -> pd.DataFrame:
    df = _all()
    wanted = set(TOPICS)
    sub = df[df.topics.apply(lambda ts: bool(wanted & set(ts)))]
    if len(sub) > MAX_DOCS:
        keep = sub.document_id.isin(_question_docs())
        rest = sub[~keep].sample(n=max(0, SAMPLE_DOCS - int(keep.sum())), random_state=SEED)
        sub = pd.concat([sub[keep], rest])
    return sub.sort_values("document_id").set_index("document_id", drop=False)


def corpus_docs() -> list[str]:
    return list(_subset().document_id)


@lru_cache
def doc_info() -> dict[str, dict]:
    out = {}
    for r in _subset().itertuples(index=False):
        host = urlparse(r.url).hostname or ""
        lang, bilingual = detect_language(r.text)
        out[r.document_id] = {
            "doc_name": r.document_id, "title": (r.title or "").strip(), "url": r.url, "host": host,
            "kind": r.kind, "trust": r.trust, "quality": r.quality,
            "topics": [t for t in r.topics], "language": lang, "bilingual": bilingual,
            "legacy_font": is_legacy_font(r.text),
            "state": infer_state(host, r.title or ""), "department": infer_department(r.title or "", r.text),
            "n_chars": len(r.text),
        }
    return out


# ---------------------------------------------------------------- pseudo-pages

def _clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _pieces(text: str, limit: int) -> list[str]:
    """Paragraphs; a paragraph longer than `limit` is split on lines, then hard-split."""
    out = []
    for para in re.split(r"\n\s*\n", text):
        if len(para) <= limit:
            out.append(para)
            continue
        buf = ""
        for line in para.split("\n"):
            if len(line) > limit and buf:
                out.append(buf)
                buf = ""
            while len(line) > limit:
                cut = line.rfind(" ", 0, limit)
                cut = cut if cut > limit // 2 else limit
                out.append(line[:cut])
                line = line[cut:].lstrip()
            if len(buf) + len(line) + 1 > limit:
                out.append(buf)
                buf = line
            else:
                buf = f"{buf}\n{line}" if buf else line
        if buf:
            out.append(buf)
    return [p for p in out if p.strip()]


def split_pages(text: str, page_chars: int = PAGE_CHARS, max_chars: int = MAX_CHARS) -> list[str]:
    """Deterministic ~page_chars pseudo-pages on paragraph boundaries (qgen relies on this)."""
    text = _clean(text)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    pages, buf = [], ""
    for p in _pieces(text, page_chars):
        if buf and len(buf) + len(p) + 2 > page_chars:
            pages.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        pages.append(buf)
    return pages or [""]


def doc_pages(doc_name: str) -> list[str]:
    return split_pages(_subset().loc[doc_name, "text"])


def build_pages() -> None:
    sub = _subset()
    rows = []
    for r in sub.itertuples(index=False):
        for i, t in enumerate(split_pages(r.text), start=1):
            rows.append({"doc_name": r.document_id, "page": i, "text": t, "n_images": 0})
    df = pd.DataFrame(rows)
    df.to_parquet(PAGES_PATH)
    info = doc_info()
    langs = pd.Series([i["language"] for i in info.values()]).value_counts().to_dict()
    print(f"topics={TOPICS}: {len(sub)} docs, {len(df)} pages, {df.text.str.len().sum() / 1e6:.1f}M chars, "
          f"languages={langs} -> {PAGES_PATH}")


# ---------------------------------------------------------------- questions

@lru_cache
def questions() -> list[dict]:
    if not QUESTIONS_PATH.exists():
        raise SystemExit(f"{QUESTIONS_PATH} missing; generate it with `python -m poc.qgen.run all`")
    return [json.loads(l) for l in QUESTIONS_PATH.open() if l.strip()]
