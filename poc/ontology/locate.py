"""Keyword heuristics that locate the primary financial-statement pages in a filing.

Every page gets a score per kind (income | balance | cashflow) from:
  * a statement title near the top of the page (matched on whitespace-squashed text, since the
    PDF text layer often splits words: "Balance Shee t"),
  * anchor line items every such statement has,
  * numeric density (statements are dense tables; TOCs / auditor letters / prose are not).
The primary statements sit together in Item 8 (or the back of an earnings release), so we pick the
combination of candidate pages that maximises total score minus a page-span penalty, instead of
taking each kind independently (which picks up MD&A summary tables).
The page after a selected page is added when the statement visibly continues.

Text corpora (DocFinQA) have markdown pipe tables on ~3,000-char pseudo-pages instead of PDF pages, so a
statement title can sit anywhere on the page, directly above its table. For pages with a pipe table
(>= 5 lines starting with "|") the title window is the text immediately preceding the page's largest
table (plus its first row) instead of the page top; PDF pages never take this branch.
"""
import itertools
import re

TITLE = {
    "income": re.compile(r"statements?of(consolidated)?(net)?(income|operations|earnings)(?!tax)|incomestatements?"),
    "balance": re.compile(r"balancesheets?|statements?of(consolidated)?financial(position|condition)"),
    "cashflow": re.compile(r"statements?of(consolidated)?cashflows?|cashflows?statements?"),
}
ANCHORS = {
    "income": [r"net (income|earnings|loss)", r"(revenue|net sales|sales)", r"(per share|diluted)",
               r"(operating income|income from operations|operating (profit|loss)|income before|earnings before)"],
    "balance": [r"total assets", r"total (current )?liabilities",
                r"(stockholders|shareholders|shareowners)['’]? ?equity|total equity",
                r"(cash and (cash )?equivalents|cash and due from banks)"],
    "cashflow": [r"operating activities|net cash (from|provided by) operations", r"investing", r"financing",
                 r"(depreciation|amortization)"],
}
CLOSER = {"income": r"(per share|diluted)", "balance": r"total liabilities and", "cashflow": r"financing"}
NUM = re.compile(r"\(?\$?\s?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?|\(\d+(?:\.\d+)?\)|\b\d+\.\d+\b")


def _squash(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def _md_head(text: str) -> tuple[str, str] | None:
    """(title window, raw preceding text) for a markdown page: text right above its largest pipe table."""
    lines = text.split("\n")
    blocks, start = [], None
    for i, ln in enumerate(lines + [""]):
        is_t = ln.lstrip().startswith("|")
        if is_t and start is None:
            start = i
        elif not is_t and start is not None:
            blocks.append((start, i))
            start = None
    if sum(b - a for a, b in blocks) < 5:
        return None
    a, b = max(blocks, key=lambda x: x[1] - x[0])
    prev_end = max([e for s0, e in blocks if e <= a], default=0)
    pre = "\n".join(lines[prev_end:a])[-600:]
    return _squash(pre)[-250:] + _squash(lines[a])[:50], pre


def score_page(text: str, kind: str) -> float:
    md = _md_head(text)
    head = md[0] if md else _squash(text[:2500])[:300]
    m = TITLE[kind].search(head)
    if not m:
        return 0.0
    low = text.lower()
    if re.search(r"report of independent|we have audited|in our opinion", md[1].lower() if md else low):
        return 0.0
    if kind == "income" and "comprehensiveincome" in head and not re.search(r"(revenue|net sales|total sales)", low):
        return 0.0
    anchors = sum(1 for a in ANCHORS[kind] if re.search(a, low))
    nums = len(NUM.findall(text))
    if nums < 20 or anchors < 2:
        return 0.0
    s = anchors * 10 + min(nums, 200) / 10
    if "consolidated" in head[: m.start() + 20] or "condensed" in head[: m.start() + 20]:
        s += 15
    if m.start() < 120:
        s += 5
    return s


def locate(pages: list[tuple[int, str]]) -> dict[str, list[int]]:
    """pages: [(page_no, text)] -> {kind: [page_no, (continuation page)]}."""
    by_no = dict(pages)
    cands = {}
    for kind in TITLE:
        sc = sorted(((score_page(t, kind), p) for p, t in pages), reverse=True)
        sc = [x for x in sc if x[0] > 0][:6]
        cands[kind] = sc + [(0.0, None)]  # allow "missing"
    best, best_combo = -1e9, None
    for combo in itertools.product(*cands.values()):
        ps = [p for _, p in combo if p is not None]
        if not ps:
            continue
        span = max(ps) - min(ps)
        total = sum(s for s, _ in combo) - 2.0 * max(0, span - 4) - 40 * (3 - len(ps))
        if total > best:
            best, best_combo = total, combo
    out = {}
    if best_combo is None:
        return out
    for kind, (s, p) in zip(cands, best_combo):
        if p is None:
            continue
        sel = [p]
        nxt = by_no.get(p + 1, "")
        if not re.search(CLOSER[kind], by_no[p], re.I) and len(NUM.findall(nxt)) >= 20:
            sel.append(p + 1)
        out[kind] = sel
    return out
