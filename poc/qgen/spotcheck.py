"""30-question human spot-check pack: spotcheck.md (readable) + spotcheck.csv (verdicts).

Stratified: 10 per question_type, and within each type ~quarter over non-English documents.
Verdict column is left blank: OK / wrong answer / ambiguous / not answerable.
"""
import csv
import json
import random
from collections import defaultdict

from poc.qgen import OUT_DIR, QUESTION_TYPES, QUESTIONS

SEED = 30
PER_TYPE = 10
NON_EN = {"numeric": 3, "eligibility_fact": 3, "narrative": 2}
VERDICTS = "OK / wrong answer / ambiguous / not answerable"


def sample(qs: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    groups = defaultdict(list)
    for q in qs:
        groups[(q["question_type"], q["language"] != "en")].append(q)
    out = []
    for t in QUESTION_TYPES:
        non = groups[(t, True)]
        en = groups[(t, False)]
        k = min(NON_EN[t], len(non))
        pick = rng.sample(non, k)
        pick += rng.sample(en, min(PER_TYPE - k, len(en)))
        out += pick
    return out


def run() -> str:
    qs = [json.loads(l) for l in QUESTIONS.open() if l.strip()]
    rows = sample(qs)
    md = [f"# Bharat benchmark — spot-check pack ({len(rows)} of {len(qs)} questions)\n",
          "Research use only; source texts have no blanket licence — do not redistribute.\n",
          "For each question, check the gold answer against the evidence quote (and, if needed, the source URL), "
          f"then record a verdict in `spotcheck.csv`: **{VERDICTS}**.\n",
          "Page numbers are 1-based pseudo-pages (~3,000 chars) of the extracted text, not the PDF's own pages.\n"]
    for n, q in enumerate(rows, 1):
        ev = q["evidence"][0]
        lang = q["language"] + (f", answer {q['answer_language']}{' (translated)' if q['answer_translated'] else ''}"
                                if q["language"] != "en" else "")
        md += [f"\n## {n}. [{q['question_type']}] {q['qid']}\n",
               f"- **Topic / language:** {q['topic']} / {lang}",
               f"- **Document:** {q['title'] or '(untitled)'} — <{q['url']}>",
               f"- **Question:** {q['question']}",
               f"- **Gold answer:** {q['answer']}",
               f"- **Evidence (page {ev['evidence_page_num'] + 1}):**\n\n  > {ev['evidence_text'].replace(chr(10), ' ')}\n",
               f"- **Justification:** {q['justification']}",
               "- **Verdict:** ______  (OK / wrong answer / ambiguous / not answerable)\n"]
    md_path = OUT_DIR / "spotcheck.md"
    md_path.write_text("\n".join(md))
    with (OUT_DIR / "spotcheck.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "qid", "question_type", "topic", "language", "question", "gold_answer", "evidence_quote",
                    "page_1based", "title", "url", "verdict", "note"])
        for n, q in enumerate(rows, 1):
            ev = q["evidence"][0]
            w.writerow([n, q["qid"], q["question_type"], q["topic"], q["language"], q["question"], q["answer"],
                        ev["evidence_text"], ev["evidence_page_num"] + 1, q["title"], q["url"], "", ""])
    return str(md_path)
