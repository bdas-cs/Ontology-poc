"""DocFinQA (kensho/DocFinQA, MIT): FinQA questions re-posed over the FULL 10-K text.

Source: test.json (922 questions over 240 distinct filings; fields Context / Question / Program /
Answer). We sample DFQ_N (default 300) questions with a fixed seed (DFQ_SEED, default 20261009);
the corpus is the filings those questions reference (identical contexts deduplicated). The seed and
the sampled qids are written to docfinqa/prepared/sample.json.

Documents. The context is a markdown rendering of the filing (words separated by tabs, tables as
pipe tables with "\\t" inside cells, "# " headings) with NO page markers, so build_pages() splits it
into ~3,000-char pseudo-pages on paragraph boundaries: a "# " heading starts a new page once the
current page holds >= 1,500 chars, and a pipe table is never split unless it alone exceeds 6,000
chars (then it is cut at row boundaries, repeating the first row). Tabs are normalised to spaces.

doc_info. Company = the line printed above "(Exact name of registrant ...)" on the cover page,
fiscal year = year of "For the fiscal year ended <date>"; doc_type "10K". DocFinQA carries no ticker
or company metadata, so these are parsed from the filing header (fallbacks are logged in
doc_info[...]["meta_source"]).

Questions. FinQA questions are document-scoped and usually do not name the company ("what was the
percentage change in ..."), because the original task hands the model the filing. To keep the task
answerable in a multi-document corpus (for every arm alike) the question text is prefixed with the
filing identity: "In <Company>'s 10-K for fiscal <FY>: <question>". The raw text is kept in
question_raw.

question_type (arithmetic complexity of the gold program):
  lookup       0 arithmetic operations (value read off the filing)
  single-op    exactly 1 operation (difference, ratio, sum of two, share of total, ...)
  multi-op     >= 2 operations (percentage change = sub + div, averages, multi-term sums, ...)
Operations = binary + - * / on the right-hand side of program lines (unary minus and the final
"* 100" / "/ 100" percent scaling are not counted) plus calls to sum/max/min/avg/len/... (each call
counts as one, sum(x)/len(x) counts as two). 26% of the test programs are empty; for those the type
is inferred from the question wording (question_type_source = "question"): "percentage change /
growth / average" -> multi-op, "ratio / percent of / difference / change / total / sum" -> single-op,
otherwise lookup.

Evidence: the source has no evidence locations. evidence_page_num is a best-effort locator: the
pseudo-page containing the most of the program's numeric constants (needs >= 2 matches, or 1 when
the program has one constant); empty when nothing matches.
"""
import hashlib
import json
import os
import random
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pandas as pd

from poc import config

NAME = "docfinqa"
ONTOLOGY_DOMAIN = "finance"
JUDGE_CONTEXT = (
    "Questions are FinQA-style numerical questions about a single company's 10-K annual report (full "
    "filing text, including tables in the notes and MD&A). Gold answers are usually a bare number in the "
    "units of the source table (e.g. 94 meaning $94 million) or a percentage, computed by the gold program. "
    "Accept a candidate whose number matches the gold within rounding (1% relative, or the gold's own "
    "rounding), whether or not the candidate states units or the % sign; percent and fraction forms of the "
    "same value (15% vs 0.15) are equivalent. A different sign is wrong unless the question wording makes the "
    "direction explicit and the candidate states it correctly (e.g. gold -42.4 vs 'decreased by 42.4'). "
    "'expanded' may accept a defensible alternative reading of an ambiguous question (e.g. a different but "
    "reasonable base year or row) when the candidate's figures are correct for that reading."
)

HF_REPO = "kensho/DocFinQA"
SPLIT = "test"
SEED = int(os.getenv("DFQ_SEED", "20261009"))
N_QUESTIONS = int(os.getenv("DFQ_N", "300"))
PAGE_CHARS = 3000
MAX_TABLE_CHARS = 6000

BASE = config.DATA_DIR / NAME
RAW_DIR = BASE / "raw"
PREP_DIR = BASE / "prepared"
DB_DIR = BASE / "db"
PAGES_PATH = DB_DIR / "pages.parquet"
# Reviewed filing identity per context (sha1 -> company/ticker/fiscal_year), produced once by
# `python -m poc.datasets.docfinqa --llm-meta` (local LLM over the cover page + header heuristics) and
# committed, so question text is identical on every machine without an LLM call.
META_PATH = Path(__file__).with_name("docfinqa_meta.json")


def download() -> None:
    from huggingface_hub import hf_hub_download

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not (RAW_DIR / f"{SPLIT}.json").exists():
        hf_hub_download(HF_REPO, f"{SPLIT}.json", repo_type="dataset", local_dir=str(RAW_DIR))
    print(f"{HF_REPO} {SPLIT}.json -> {RAW_DIR}")


# ------------------------------------------------------------------ text normalisation + pages
def clean(text: str) -> str:
    t = text.replace("\\t", " ").replace("\t", " ").replace("\r", "")
    t = re.sub(r"[  ]{2,}", lambda m: " " if len(m.group()) < 4 else "    ", t)
    return t.strip()


def _blocks(text: str) -> list[tuple[str, str]]:
    """Paragraph blocks: ('table', ...) for consecutive pipe-table lines, ('heading', ...), ('text', ...)."""
    out: list[tuple[str, str]] = []
    for para in re.split(r"\n\s*\n", text):
        lines = para.split("\n")
        cur, cur_kind = [], None
        for ln in lines:
            kind = "table" if ln.lstrip().startswith("|") else "text"
            if cur and kind != cur_kind:
                out.append((cur_kind, "\n".join(cur)))
                cur = []
            cur.append(ln)
            cur_kind = kind
        if cur:
            blk = "\n".join(cur)
            if cur_kind == "text" and blk.lstrip().startswith("#"):
                cur_kind = "heading"
            out.append((cur_kind, blk))
    # merge adjacent table pieces split only by blank lines
    merged: list[tuple[str, str]] = []
    for kind, b in out:
        if merged and kind == "table" and merged[-1][0] == "table":
            merged[-1] = ("table", merged[-1][1] + "\n" + b)
        else:
            merged.append((kind, b))
    return merged


def _split_table(tbl: str) -> list[str]:
    if len(tbl) <= MAX_TABLE_CHARS:
        return [tbl]
    rows = tbl.split("\n")
    head = rows[:2] if len(rows) > 2 and re.match(r"^\|\s*:?-{3}", rows[1].strip()) else rows[:1]
    parts, cur = [], list(head)
    for r in rows[len(head):]:
        if sum(len(x) + 1 for x in cur) + len(r) > MAX_TABLE_CHARS and len(cur) > len(head):
            parts.append("\n".join(cur))
            cur = list(head)
        cur.append(r)
    parts.append("\n".join(cur))
    return parts


def split_pages(text: str) -> list[str]:
    """Pack blocks into ~PAGE_CHARS pages. Headings and short caption lines stay with the block that follows
    them (they move to the next page with it), so a statement title is on the same page as its table."""
    pages: list[str] = []
    cur: list[tuple[str, str]] = []  # (kind, text) on the current page

    def size() -> int:
        return sum(len(t) + 2 for _, t in cur)

    def flush(carry: bool = True):
        nonlocal cur
        keep: list[tuple[str, str]] = []
        while carry and cur and len(keep) < 3 and (cur[-1][0] == "heading" or (cur[-1][0] == "text" and len(cur[-1][1]) < 200)):
            keep.insert(0, cur.pop())
        if not cur:  # the page holds only captions: keep them with the next block on this page
            cur = keep
            return
        body = "\n\n".join(t for _, t in cur).strip()
        if body:
            pages.append(body)
        cur = keep

    for kind, blk in _blocks(text):
        pieces = _split_table(blk) if kind == "table" else [blk]
        for piece in pieces:
            if kind == "heading" and size() >= PAGE_CHARS // 2:
                flush(carry=False)
            if cur and size() + len(piece) > PAGE_CHARS:
                flush(carry=kind == "table")
            if kind == "text" and len(piece) > PAGE_CHARS:  # very long paragraph: cut on sentences
                buf = ""
                for s in re.split(r"(?<=[.;])\s+", piece):
                    if buf and size() + len(buf) + len(s) > PAGE_CHARS:
                        cur.append(("text", buf))
                        flush(carry=False)
                        buf = ""
                    buf += (" " if buf else "") + s
                if buf:
                    cur.append(("text", buf))
                continue
            cur.append((kind, piece))
    flush(carry=False)
    return pages


# ------------------------------------------------------------------ header metadata
_NAME_RE = re.compile(r"\n\s*#?\s*([^\n]{3,120}?)\s*\n+\s*\(?\s*exact\s+name\s+of\s+(the\s+)?registrant", re.I)
_FY_RE = re.compile(r"fiscal\s+year\s+ended:?\s*#?\s*([A-Za-z]+\.?\s+\d{1,2}\s*,?\s*((?:19|20)\d{2}))", re.I)
_SUFFIX = re.compile(r"\b(inc|incorporated|corp|corporation|company|co|ltd|limited|plc|llc|l\.p|lp|group|holdings|trust|n\.v|s\.a|ag|bancorp)\b\.?", re.I)
_SMALL = {"and", "of", "the", "for", "&"}


def _pretty(name: str) -> str:
    name = re.sub(r"^[#\s*]+|[\s*]+$", "", name).strip(" ,.")
    if name.isupper():
        words = []
        for w in name.split():
            lw = w.lower()
            if lw in _SMALL:
                words.append(lw)
            elif len(w) <= 3 and not _SUFFIX.fullmatch(w.strip(".,")) and "." not in w:
                words.append(w)  # acronyms like CME, US, UPS
            else:
                words.append("-".join(p[:1].upper() + p[1:].lower() for p in w.split("-")))
        name = " ".join(words)
    return name


def parse_header(text: str) -> dict:
    head = text[:20000]
    meta = {"company": None, "fiscal_year": None, "fiscal_year_end": None, "meta_source": []}
    m = _NAME_RE.search("\n" + head)
    if m and not re.search(r"commission|securities|annual report|transition|state or other", m.group(1), re.I):
        meta["company"] = _pretty(m.group(1))
        meta["meta_source"].append("company:cover")
    else:
        cands = [ln.strip("# ").strip() for ln in head.split("\n")
                 if _SUFFIX.search(ln) and 3 < len(ln.strip("# ")) < 80 and not re.search(r"\b(act|section|exchange|commission|form)\b", ln, re.I)]
        if cands:
            meta["company"] = _pretty(Counter(cands).most_common(1)[0][0])
            meta["meta_source"].append("company:header-guess")
    m = _FY_RE.search(head)
    if m:
        meta["fiscal_year"] = int(m.group(2))
        meta["fiscal_year_end"] = re.sub(r"\s+", " ", m.group(1)).strip()
        meta["meta_source"].append("fy:cover")
    elif (ty := table_year(text)) is not None:
        meta["fiscal_year"] = ty
        meta["meta_source"].append("fy:table-headers")
    return meta


def table_year(text: str) -> int | None:
    """Most common latest year in table header rows that list >= 2 years (e.g. "| | 2016 | 2015 |")."""
    c = Counter()
    for ln in text.split("\n"):
        if ln.lstrip().startswith("|"):
            ys = [int(y) for y in re.findall(r"(?<![\d.,$])((?:19[89]|20[0-2])\d)(?![\d,])", ln)]
            if len(ys) >= 2 and len(set(ys)) >= 2:
                c[max(ys)] += 1
    return c.most_common(1)[0][0] if c else None


_ORG = re.compile(r"\b((?:[A-Z][A-Za-z.&'\-]+ ){0,4}(?:Inc|Incorporated|Corporation|Corp|Company|Co|Group|Holdings|Ltd|plc|LLC|Bancorp|Trust)\b\.?)")


def org_mentions(text: str, n: int = 8) -> list[str]:
    c = Counter(m.strip() for m in _ORG.findall(text[:400000]))
    return [f"{k} ({v})" for k, v in c.most_common(n)]


# ------------------------------------------------------------------ question typing
_FUNC = re.compile(r"\b(sum|max|min|avg|average|mean|len|abs|round)\s*\(")


def count_ops(program: str) -> int:
    n = 0
    for line in program.split("\n"):
        line = line.split("#", 1)[0]
        if "=" not in line:
            continue
        rhs = line.split("=", 1)[1]
        rhs = re.sub(r"\[[^\]]*\]", "L", rhs)  # list literals are data, not operations
        rhs = re.sub(r"[*/]\s*100(?:\.0+)?\s*$", "", rhs.strip())  # percent scaling
        calls = _FUNC.findall(rhs)
        n += sum(1 for c in calls if c != "len")  # sum(x)/len(x): the "/" below counts the division
        rhs = _FUNC.sub("(", rhs)
        toks = re.findall(r"\d+\.?\d*|[A-Za-z_]\w*|[-+*/()]", rhs)
        prev = None
        for t in toks:
            if t in "+-*/" and prev is not None and prev not in "+-*/(":
                n += 1
            prev = t
    return n


def infer_type_from_question(q: str) -> str:
    ql = q.lower()
    if re.search(r"percent(age)? (change|increase|decrease|growth)|% (change|increase|decrease)|growth rate|"
                 r"rate of (change|growth)|average|cagr|annual(ized)? (growth|return)", ql):
        return "multi-op"
    if re.search(r"ratio|percent|portion|proportion|fraction|difference|change|increase|decrease|total|sum|"
                 r"combined|net (effect|change)|times|more than|less than|how much (more|less)", ql):
        return "single-op"
    return "lookup"


def question_type(program: str, question: str) -> tuple[str, str]:
    if program.strip():
        n = count_ops(program)
        return ("lookup" if n == 0 else "single-op" if n == 1 else "multi-op"), "program"
    return infer_type_from_question(question), "question"


def _program_numbers(program: str) -> list[str]:
    nums = []
    for line in program.split("\n"):
        line = line.split("#", 1)[0]
        if "=" in line:
            for x in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", line.split("=", 1)[1]):
                if x not in ("100", "1", "2", "0") and x not in nums:
                    nums.append(x)
    return nums


def _num_variants(x: str) -> list[str]:
    v = {x}
    try:
        f = float(x)
        if f >= 1000 and f == int(f):
            v.add(f"{int(f):,}")
        if "." in x:
            v.add(x.rstrip("0").rstrip("."))
    except ValueError:
        pass
    return [s for s in v if s]


def locate_evidence(pages: list[str], program: str) -> int | None:
    nums = _program_numbers(program)
    if not nums:
        return None
    best, best_n = None, 0
    for i, t in enumerate(pages):
        n = sum(1 for x in nums if any(re.search(rf"(?<![\d,.]){re.escape(v)}(?![\d])", t) for v in _num_variants(x)))
        if n > best_n:
            best, best_n = i, n
    need = 1 if len(nums) == 1 else 2
    return best if best_n >= need else None


# ------------------------------------------------------------------ prepare (sample + docs + questions)
def _slug(s: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", s.upper()).strip("_")[:40]


def prepare(force: bool = False) -> None:
    """Sample questions, dedupe contexts, split pages, write prepared/{sample,doc_info,questions}.json(l)
    and prepared/pages.parquet. Idempotent."""
    if (PREP_DIR / "questions.jsonl").exists() and not force:
        return
    src = RAW_DIR / f"{SPLIT}.json"
    if not src.exists():
        raise SystemExit(f"{src} missing; run download() (POC_DATASET=docfinqa make data)")
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    rows = json.loads(src.read_text())
    rng = random.Random(SEED)
    idx = sorted(rng.sample(range(len(rows)), min(N_QUESTIONS, len(rows))))
    reviewed = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    ctx_doc: dict[str, str] = {}
    info: dict[str, dict] = {}
    page_rows, used_names = [], set()
    doc_pages: dict[str, list[str]] = {}
    for i in idx:
        ctx = rows[i]["Context"]
        h = hashlib.sha1(ctx.encode()).hexdigest()
        if h in ctx_doc:
            continue
        text = clean(ctx)
        meta = parse_header(text)
        if (rv := reviewed.get(h)):
            meta.update({k: rv[k] for k in ("company", "fiscal_year") if rv.get(k)})
            meta["ticker"] = rv.get("ticker")
            meta["meta_source"].append("reviewed:" + rv.get("source", "llm"))
        base = f"{_slug(meta['company'] or 'UNKNOWN')}_{meta['fiscal_year'] or 'NA'}_10K"
        name, k = base, 2
        while name in used_names:
            name, k = f"{base}_{k}", k + 1
        used_names.add(name)
        ctx_doc[h] = name
        pages = split_pages(text)
        doc_pages[name] = pages
        info[name] = {"doc_name": name, "company": meta["company"] or name, "doc_type": "10k",
                      "doc_period": meta["fiscal_year"], "fiscal_year_end": meta["fiscal_year_end"],
                      "ticker": meta.get("ticker"), "gics_sector": None, "meta_source": meta["meta_source"],
                      "context_sha1": h, "n_chars": len(text), "n_pages": len(pages)}
        page_rows += [{"doc_name": name, "page": p, "text": t, "n_images": 0} for p, t in enumerate(pages, start=1)]
    qs = []
    for i in idx:
        r = rows[i]
        doc = ctx_doc[hashlib.sha1(r["Context"].encode()).hexdigest()]
        d = info[doc]
        qtype, qsrc = question_type(r["Program"] or "", r["Question"])
        ev_page = locate_evidence(doc_pages[doc], r["Program"] or "")
        raw_q = re.sub(r"\s+", " ", r["Question"]).strip()
        qs.append({
            "qid": f"dfq-{SPLIT}-{i:04d}",
            "question": f"In {d['company']}'s 10-K for fiscal {d['doc_period']}: {raw_q}",
            "answer": str(r["Answer"]).strip(),
            "justification": (r["Program"] or "").strip(),
            "question_type": qtype,
            "doc_name": doc,
            "evidence": ([{"evidence_text": "", "doc_name": doc, "evidence_page_num": ev_page}]
                         if ev_page is not None else []),
            "question_raw": raw_q,
            "question_type_source": qsrc,
            "n_ops": count_ops(r["Program"]) if (r["Program"] or "").strip() else None,
            "company": d["company"],
            "source_index": i,
        })
    (PREP_DIR / "sample.json").write_text(json.dumps(
        {"source": f"{HF_REPO}/{SPLIT}.json", "seed": SEED, "n": len(idx), "source_rows": len(rows),
         "qids": [q["qid"] for q in qs], "source_indices": idx}, indent=1))
    (PREP_DIR / "doc_info.json").write_text(json.dumps(info, indent=1))
    pd.DataFrame(page_rows).to_parquet(PREP_DIR / "pages.parquet")
    with (PREP_DIR / "questions.jsonl").open("w") as f:
        for q in qs:
            f.write(json.dumps(q) + "\n")
    print(f"docfinqa: {len(qs)} questions over {len(info)} filings, {len(page_rows)} pseudo-pages -> {PREP_DIR}")


@lru_cache
def questions() -> list[dict]:
    prepare()
    return [json.loads(l) for l in (PREP_DIR / "questions.jsonl").open()]


@lru_cache
def doc_info() -> dict[str, dict]:
    prepare()
    return json.loads((PREP_DIR / "doc_info.json").read_text())


def corpus_docs() -> list[str]:
    return sorted(doc_info())


def build_pages() -> None:
    prepare()
    DB_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PREP_DIR / "pages.parquet")
    df.to_parquet(PAGES_PATH)
    print(f"{df.doc_name.nunique()} docs, {len(df)} pages, {df.text.str.len().sum() / 1e6:.1f}M chars -> {PAGES_PATH}")


META_PROMPT = """Identify the registrant of this SEC 10-K annual report from its cover page and the most frequent
organisation names in the document. Return JSON only:
{"company": "<registrant name as it would appear in prose, e.g. 'Snap-on Incorporated', 'JPMorgan Chase & Co.'>",
 "ticker": "<primary US ticker or null>", "fiscal_year": <int, the fiscal year this report covers>}
If the filing is a combined report of a parent and its subsidiaries, give the parent."""


def llm_meta(limit: int | None = None) -> None:
    """Review company / ticker / fiscal year of every sampled filing with the local LLM; writes META_PATH."""
    import asyncio

    from poc import llm

    rows = json.loads((RAW_DIR / f"{SPLIT}.json").read_text())
    idx = sorted(random.Random(SEED).sample(range(len(rows)), min(N_QUESTIONS, len(rows))))
    ctxs = {}
    for i in idx:
        ctxs.setdefault(hashlib.sha1(rows[i]["Context"].encode()).hexdigest(), rows[i]["Context"])
    done = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    todo = [(h, c) for h, c in ctxs.items() if h not in done][: limit or None]

    async def one(h, ctx):
        text = clean(ctx)
        g = parse_header(text)
        hint = (f"Header heuristics: company={g['company']!r} ({','.join(g['meta_source'])}), "
                f"fiscal_year={g['fiscal_year']}, latest year in table headers={table_year(text)}\n"
                f"Most frequent organisation names: {'; '.join(org_mentions(text))}")
        out = await llm.chat([{"role": "system", "content": META_PROMPT},
                              {"role": "user", "content": f"{hint}\n\nCover page:\n{text[:3500]}"}],
                             tag="docfinqa-meta", max_tokens=200, temperature=0.0)
        try:
            js = llm.parse_json(out["content"])
            fy = int(js.get("fiscal_year")) if js.get("fiscal_year") else g["fiscal_year"]
            return h, {"company": js.get("company") or g["company"], "ticker": js.get("ticker"),
                       "fiscal_year": fy, "source": "llm", "heuristic_company": g["company"],
                       "heuristic_fy": g["fiscal_year"]}
        except Exception:
            return h, None

    async def run():
        return await asyncio.gather(*(one(h, c) for h, c in todo))

    for h, m in asyncio.run(run()):
        if m:
            done[h] = m
    META_PATH.write_text(json.dumps(done, indent=1, sort_keys=True, ensure_ascii=False))
    print(f"{len(done)}/{len(ctxs)} filings reviewed -> {META_PATH}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-sample and rebuild prepared/")
    ap.add_argument("--llm-meta", action="store_true", help="review filing identity with the local LLM, then rebuild")
    a = ap.parse_args()
    if a.llm_meta:
        llm_meta()
    prepare(force=a.force or a.llm_meta)
