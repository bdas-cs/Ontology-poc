"""CUAD v1 (theatticusproject/cuad, CC BY 4.0): 510 commercial contracts, 41 clause categories.

Source: CUAD_v1/CUAD_v1.json (SQuAD-style: one context = the full contract text, one qa per category,
answer spans with char offsets) + CUAD_v1/master_clauses.csv (normalised answers such as
"Governing Law-Answer" = "Nevada"). Contract type = the PDF folder in full_contract_pdf/ (repo
listing saved as raw/files.json). Only the 499 contracts present in both files are eligible.

Sampling (fixed seed CUAD_SEED, default 20261009, recorded with the qids in cuad/prepared/sample.json):
  1. contracts: CUAD_CONTRACTS (default 100). Contracts holding the rarest clauses are taken first
     (seeded order) until every category has >= 4 present examples where the data allows, then the
     rest is filled at random.
  2. questions: CUAD_N (default 300) spread evenly over the 41 categories (7-8 each), half
     "present" and half "absent" per category where the pools allow (a short pool is filled from the
     other side).

Questions are natural-language versions of the category prompts, naming the contract by its filing
metadata (filer, contract type, filing date — the same fields the Contract record carries), e.g.
"Which state's or country's law governs the LIMEENERGYCO Distributor Agreement (filed 1999-09-09)?"
or "Does ... contain a non-compete clause ...? If so, quote it."
Gold answer: metadata categories (dates, parties, governing law, renewal, notice) -> the normalised
master_clauses answer when there is one, followed by the annotated span(s); clause categories ->
"Yes. Clause text: ..." with the annotated span(s); absent -> "Not present in this contract."

Pages: the JSON context (exactly what the spans index into) split into ~3,000-char pseudo-pages on
paragraph boundaries; evidence_page_num (0-based) is the pseudo-page holding each span's start.

question_type = clause group:
  parties_law             Document Name, Parties, Governing Law, Third Party Beneficiary
  dates_term              Agreement / Effective / Expiration Date, Renewal Term, Notice Period To
                          Terminate Renewal
  restrictive_covenants   Most Favored Nation, Competitive Restriction Exception, Non-Compete,
                          Exclusivity, No-Solicit Of Customers / Employees, Non-Disparagement,
                          Rofr/Rofo/Rofn, Price Restrictions, Covenant Not To Sue
  termination_assignment  Termination For Convenience, Change Of Control, Anti-Assignment,
                          Post-Termination Services
  ip_license              Ip Ownership Assignment, Joint Ip Ownership, License Grant,
                          Non-Transferable License, Affiliate License-Licensor / -Licensee,
                          Unlimited/All-You-Can-Eat-License, Irrevocable Or Perpetual License,
                          Source Code Escrow
  financial_liability     Revenue/Profit Sharing, Minimum Commitment, Volume Restriction,
                          Audit Rights, Uncapped Liability, Cap On Liability, Liquidated Damages,
                          Warranty Duration, Insurance
Extra fields: category, present (bool).
"""
import json
import os
import random
import re
from collections import Counter
from functools import lru_cache

import pandas as pd

from poc import config

NAME = "cuad"
ONTOLOGY_DOMAIN = "contracts"
JUDGE_CONTEXT = (
    "Questions are about a single commercial contract (CUAD: SEC-filed exhibits such as distribution, "
    "licence, supply, development and co-branding agreements). Each question targets one clause category; "
    "the ground truth is the clause text a lawyer annotated (possibly several spans), a normalised value for "
    "dates / governing law / parties / renewal / notice questions, or 'Not present in this contract'.\n"
    "strict: true if the candidate identifies the same clause content as the ground truth (same obligation, "
    "party, duration, jurisdiction or date; paraphrase or a shorter quote of the key sentence is fine), or, "
    "when the ground truth is 'Not present in this contract', the candidate clearly says the contract has no "
    "such clause. Saying 'yes' but quoting an unrelated clause is false; saying a clause is absent when the "
    "ground truth has one is false; claiming a clause exists when the ground truth says not present is false.\n"
    "expanded: true if strict, OR the candidate's quoted clause substantially overlaps the ground-truth spans "
    "(covers the main operative sentence even if it misses secondary spans or adds neighbouring text), OR for a "
    "normalised value it gives an equivalent value (e.g. a date in another format, 'State of New York' vs "
    "'New York')."
)

HF_REPO = "theatticusproject/cuad"
SEED = int(os.getenv("CUAD_SEED", "20261009"))
N_CONTRACTS = int(os.getenv("CUAD_CONTRACTS", "100"))
N_QUESTIONS = int(os.getenv("CUAD_N", "300"))
PAGE_CHARS = 3000
MIN_PRESENT = 4

BASE = config.DATA_DIR / NAME
RAW_DIR = BASE / "raw"
PREP_DIR = BASE / "prepared"
DB_DIR = BASE / "db"
PAGES_PATH = DB_DIR / "pages.parquet"
JSON_PATH = RAW_DIR / "CUAD_v1" / "CUAD_v1.json"
MASTER_PATH = RAW_DIR / "CUAD_v1" / "master_clauses.csv"
FILES_PATH = RAW_DIR / "files.json"

GROUPS = {
    "parties_law": ["Document Name", "Parties", "Governing Law", "Third Party Beneficiary"],
    "dates_term": ["Agreement Date", "Effective Date", "Expiration Date", "Renewal Term",
                   "Notice Period To Terminate Renewal"],
    "restrictive_covenants": ["Most Favored Nation", "Competitive Restriction Exception", "Non-Compete", "Exclusivity",
                              "No-Solicit Of Customers", "No-Solicit Of Employees", "Non-Disparagement",
                              "Rofr/Rofo/Rofn", "Price Restrictions", "Covenant Not To Sue"],
    "termination_assignment": ["Termination For Convenience", "Change Of Control", "Anti-Assignment",
                               "Post-Termination Services"],
    "ip_license": ["Ip Ownership Assignment", "Joint Ip Ownership", "License Grant", "Non-Transferable License",
                   "Affiliate License-Licensor", "Affiliate License-Licensee", "Unlimited/All-You-Can-Eat-License",
                   "Irrevocable Or Perpetual License", "Source Code Escrow"],
    "financial_liability": ["Revenue/Profit Sharing", "Minimum Commitment", "Volume Restriction", "Audit Rights",
                            "Uncapped Liability", "Cap On Liability", "Liquidated Damages", "Warranty Duration",
                            "Insurance"],
}
GROUP_OF = {c: g for g, cs in GROUPS.items() for c in cs}
CATEGORIES = [c for cs in GROUPS.values() for c in cs]

# value-style categories: open questions; gold = normalised answer + spans
VALUE_Q = {
    "Document Name": "What is the name (title) of {C}?",
    "Parties": "Who are the parties to {C}?",
    "Agreement Date": "What is the agreement date of {C}?",
    "Effective Date": "What is the effective date of {C}?",
    "Expiration Date": "When does the initial term of {C} expire?",
    "Renewal Term": "What is the renewal term of {C} after its initial term expires (e.g. automatic renewal)?",
    "Notice Period To Terminate Renewal": "What notice period is required to terminate renewal of {C}?",
    "Governing Law": "Which state's or country's law governs {C}?",
    "Warranty Duration": "What is the duration of any warranty given in {C}? Quote the warranty clause.",
}
# clause categories: "Does {C} contain <desc>? If so, quote it."
CLAUSE_DESC = {
    "Most Favored Nation": "a most-favored-nation clause (a party must get terms at least as good as any third party)",
    "Competitive Restriction Exception": "an exception or carve-out to a non-compete, exclusivity or no-solicit restriction",
    "Non-Compete": "a non-compete clause restricting a party from competing with the counterparty or operating in a certain geography or line of business",
    "Exclusivity": "an exclusivity commitment (exclusive dealing, or a restriction on buying from / selling to other parties)",
    "No-Solicit Of Customers": "a restriction on soliciting or contracting with the counterparty's customers or partners",
    "No-Solicit Of Employees": "a restriction on soliciting or hiring the counterparty's employees or contractors",
    "Non-Disparagement": "a non-disparagement clause",
    "Termination For Convenience": "a right for a party to terminate without cause (for convenience)",
    "Rofr/Rofo/Rofn": "a right of first refusal, first offer or first negotiation",
    "Change Of Control": "a change-of-control clause (termination right, consent or notice when a party is acquired, merges or sells substantially all assets)",
    "Anti-Assignment": "an anti-assignment clause (consent or notice required to assign the contract)",
    "Revenue/Profit Sharing": "an obligation to share revenue or profit with the counterparty (e.g. royalties, revenue share)",
    "Price Restrictions": "a restriction on a party's ability to raise or reduce prices",
    "Minimum Commitment": "a minimum commitment (minimum order size, purchase amount or units per period)",
    "Volume Restriction": "a volume restriction (fee increase or consent needed when use exceeds a threshold)",
    "Ip Ownership Assignment": "a clause under which intellectual property created by one party becomes the property of the counterparty",
    "Joint Ip Ownership": "joint or shared ownership of intellectual property",
    "License Grant": "a license granted by one party to the other",
    "Non-Transferable License": "a limit on transferring or sublicensing a license granted under it",
    "Affiliate License-Licensor": "a license grant by the licensor's affiliates (or covering IP of the licensor's affiliates)",
    "Affiliate License-Licensee": "a license grant that extends to the licensee's affiliates",
    "Unlimited/All-You-Can-Eat-License": "an unlimited, enterprise-wide or all-you-can-eat license",
    "Irrevocable Or Perpetual License": "an irrevocable or perpetual license",
    "Source Code Escrow": "a requirement to deposit source code into escrow",
    "Post-Termination Services": "obligations that survive termination or expiration (transition services, wind-down, last-buy, payments, return of IP and similar)",
    "Audit Rights": "a right to audit a party's books, records or facilities",
    "Uncapped Liability": "uncapped liability for a party (liability excluded from any cap, e.g. for certain breaches)",
    "Cap On Liability": "a cap or limitation on liability",
    "Liquidated Damages": "a liquidated damages, break-up fee or termination fee clause",
    "Insurance": "a requirement for a party to maintain insurance",
    "Covenant Not To Sue": "a covenant not to sue or not to challenge the counterparty's IP",
    "Third Party Beneficiary": "a third-party beneficiary provision",
}
ABSENT = "Not present in this contract."
_TYPE_NORM = {"Affiliate_Agreements": "Affiliate Agreement", "Affiliate Agreement": "Affiliate Agreement",
              "License_Agreements": "License Agreement", "Co_Branding": "Co-Branding Agreement",
              "Joint Venture _ Filing": "Joint Venture Agreement", "Agency Agreements": "Agency Agreement",
              "Endorsement Agreement": "Endorsement Agreement", "Non_Compete_Non_Solicit": "Non-Compete Agreement",
              "Commercial Contracts (Part II-A)": None, "IP": "IP Agreement"}


def download() -> None:
    from huggingface_hub import HfApi, hf_hub_download

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for f in ("CUAD_v1/CUAD_v1.json", "CUAD_v1/master_clauses.csv"):
        if not (RAW_DIR / f).exists():
            hf_hub_download(HF_REPO, f, repo_type="dataset", local_dir=str(RAW_DIR))
    if not FILES_PATH.exists():
        FILES_PATH.write_text(json.dumps(HfApi().list_repo_files(HF_REPO, repo_type="dataset")))
    print(f"{HF_REPO} -> {RAW_DIR}")


# ------------------------------------------------------------------ helpers
def safe_name(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_")


def parse_title(title: str) -> dict:
    """'LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT' | 'LohaCompanyltd_20191209_F-1_EX-10.16_..._Supply Agreement'."""
    m = re.match(r"^(.+?)_(\d{2})_(\d{2})_(\d{4})", title)
    if m:
        filer, date = m.group(1), f"{m.group(4)}-{m.group(2)}-{m.group(3)}"
    else:
        m = re.match(r"^(.+?)_((?:19|20)\d{2})(\d{2})(\d{2})", title)
        filer, date = (m.group(1), f"{m.group(2)}-{m.group(3)}-{m.group(4)}") if m else (re.split(r"\s+-\s+|_", title)[0].strip(), None)
    tail = re.split(r"[-_](?=[^-_]*$)", title)[-1].strip()
    tail = re.sub(r"\d+$", "", tail).strip()
    return {"filer": filer, "filing_date": date, "title_tail": tail}


def _type_from_folder(folder: str | None, tail: str) -> str:
    if folder in _TYPE_NORM and _TYPE_NORM[folder]:
        return _TYPE_NORM[folder]
    if folder and folder in _TYPE_NORM:  # Part II-A: no type folder, use the title
        return tail.title()
    if folder:
        f = re.sub(r"(?i)agreements$", "Agreement", folder.replace("_", " ").strip())
        return f if f.lower().endswith("agreement") else f"{f} Agreement"
    return tail.title()


def label(info: dict) -> str:
    kind = info["contract_type"] or info["title_tail"].title()
    date = f" (filed {info['filing_date']})" if info.get("filing_date") else ""
    return f"the {info['filer']} {kind}{date}"


def _page_starts(text: str) -> list[int]:
    """Start offsets of ~PAGE_CHARS pseudo-pages cut at paragraph (else line) boundaries."""
    starts, i, n = [0], 0, len(text)
    while i + PAGE_CHARS < n:
        win = text[i + PAGE_CHARS // 2: i + PAGE_CHARS]
        cut = win.rfind("\n\n")
        if cut == -1:
            cut = win.rfind("\n")
        if cut == -1:
            cut = win.rfind(". ")
        j = i + PAGE_CHARS // 2 + cut + 1 if cut != -1 else i + PAGE_CHARS
        starts.append(j)
        i = j
    return starts


def _clean_page(t: str) -> str:
    t = re.sub(r"[ \t ]{2,}", " ", t)
    return re.sub(r"\n\s*\n\s*(\n\s*)+", "\n\n", t).strip()


def _page_of(starts: list[int], off: int) -> int:
    lo = 0
    for k, s in enumerate(starts):
        if s <= off:
            lo = k
        else:
            break
    return lo


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# ------------------------------------------------------------------ prepare
def prepare(force: bool = False) -> None:
    if (PREP_DIR / "questions.jsonl").exists() and not force:
        return
    if not JSON_PATH.exists():
        raise SystemExit(f"{JSON_PATH} missing; run download() (POC_DATASET=cuad)")
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(JSON_PATH.read_text())["data"]
    master = pd.read_csv(MASTER_PATH)
    master["title"] = master.Filename.str.replace(r"\.(pdf|PDF)$", "", regex=True)
    mrow = {r["title"]: r for r in master.to_dict("records")}
    folders = {}
    if FILES_PATH.exists():
        for f in json.loads(FILES_PATH.read_text()):
            if f.startswith("CUAD_v1/full_contract_pdf/"):
                parts = f.split("/")
                folders[re.sub(r"\.(pdf|PDF)$", "", parts[-1])] = parts[-2]
    contracts = {}
    for x in data:
        if x["title"] not in mrow:
            continue
        p = x["paragraphs"][0]
        ann = {}
        for qa in p["qas"]:
            cat = qa["id"].split("__", 1)[1]
            spans = sorted({(a["answer_start"], a["text"]) for a in qa["answers"]})
            ann[cat] = spans
        contracts[x["title"]] = {"context": p["context"], "ann": ann}
    titles = sorted(contracts)
    rng = random.Random(SEED)
    rng.shuffle(titles)
    # 1. contracts: cover rare categories first, then random fill
    chosen: list[str] = []
    present_n = Counter()
    for cat in sorted(CATEGORIES, key=lambda c: sum(bool(contracts[t]["ann"].get(c)) for t in titles)):
        for t in titles:
            if present_n[cat] >= MIN_PRESENT or len(chosen) >= N_CONTRACTS:
                break
            if t not in chosen and contracts[t]["ann"].get(cat):
                chosen.append(t)
                for c in CATEGORIES:
                    present_n[c] += bool(contracts[t]["ann"].get(c))
    for t in titles:
        if len(chosen) >= N_CONTRACTS:
            break
        if t not in chosen:
            chosen.append(t)
    chosen = sorted(chosen)
    # 2. questions: even over categories, half present / half absent
    per_cat = {c: N_QUESTIONS // len(CATEGORIES) for c in CATEGORIES}
    for c in rng.sample(CATEGORIES, N_QUESTIONS - sum(per_cat.values())):
        per_cat[c] += 1
    picks = []
    for cat in CATEGORIES:
        pres = [t for t in chosen if contracts[t]["ann"].get(cat)]
        absn = [t for t in chosen if not contracts[t]["ann"].get(cat)]
        rng.shuffle(pres)
        rng.shuffle(absn)
        n = per_cat[cat]
        k_p = min(len(pres), (n + 1) // 2)
        k_a = min(len(absn), n - k_p)
        k_p = min(len(pres), n - k_a)
        picks += [(t, cat, True) for t in pres[:k_p]] + [(t, cat, False) for t in absn[:k_a]]
    # pages + doc_info
    info, page_rows, starts_of = {}, [], {}
    for t in chosen:
        ctx = contracts[t]["context"]
        st = _page_starts(ctx)
        starts_of[t] = st
        pages = [_clean_page(ctx[a:b]) for a, b in zip(st, st[1:] + [len(ctx)])]
        meta = parse_title(t)
        folder = folders.get(t)
        m = mrow[t]
        d = safe_name(t)
        info[d] = {"doc_name": d, "title": t, **meta, "contract_type": _type_from_folder(folder, meta["title_tail"]),
                   "type_folder": folder, "company": meta["filer"], "doc_type": "contract",
                   "doc_period": (meta["filing_date"] or "")[:4] or None, "n_pages": len(pages), "n_chars": len(ctx)}
        info[d]["label"] = label(info[d])
        page_rows += [{"doc_name": d, "page": i, "text": tx, "n_images": 0} for i, tx in enumerate(pages, start=1)]
    dup = Counter(v["label"] for v in info.values())
    seen = Counter()
    for v in info.values():  # e.g. "... Content License Agreement1" / "...2": number them
        if dup[v["label"]] > 1:
            seen[v["label"]] += 1
            n = re.search(r"(\d+)$", v["title"])
            k = n.group(1) if n else str(seen[v["label"]])
            v["label"] = re.sub(r"( \(filed|$)", f" No. {k}\\1", v["label"], count=1)
    qs = []
    for t, cat, present in sorted(picks, key=lambda x: (x[0], CATEGORIES.index(x[1]))):
        d = safe_name(t)
        C = info[d]["label"]
        q = VALUE_Q[cat].format(C=C) if cat in VALUE_Q else f"Does {C} contain {CLAUSE_DESC[cat]}? If so, quote it."
        spans = contracts[t]["ann"].get(cat) or []
        quoted = " | ".join(f'"{_norm_ws(s)[:800]}"' for _, s in spans)
        norm = mrow[t].get(f"{cat}-Answer", mrow[t].get(f"{cat}- Answer"))
        norm = None if (norm is None or (isinstance(norm, float) and pd.isna(norm)) or str(norm).strip() in ("", "Yes", "No")) else str(norm).strip()
        if not present:
            gold = ABSENT
        elif cat in VALUE_Q and norm:
            gold = f"{norm}. Clause text: {quoted}"
        elif cat in VALUE_Q:
            gold = f"Clause text: {quoted}"
        else:
            gold = f"Yes. Clause text: {quoted}"
        qs.append({
            "qid": f"cuad-{safe_name(t)[:60]}-{re.sub(r'[^a-z0-9]+', '_', cat.lower()).strip('_')}",
            "question": q,
            "answer": gold,
            "justification": "" if present else f"No '{cat}' span was annotated in this contract.",
            "question_type": GROUP_OF[cat],
            "doc_name": d,
            "evidence": [{"evidence_text": _norm_ws(s), "doc_name": d,
                          "evidence_page_num": _page_of(starts_of[t], off)} for off, s in spans],
            "category": cat,
            "present": present,
        })
    (PREP_DIR / "sample.json").write_text(json.dumps(
        {"source": f"{HF_REPO}/CUAD_v1/CUAD_v1.json", "seed": SEED, "n_contracts": len(chosen),
         "n_questions": len(qs), "contracts": chosen, "qids": [q["qid"] for q in qs],
         "per_category": {c: [sum(1 for q in qs if q["category"] == c and q["present"]),
                              sum(1 for q in qs if q["category"] == c and not q["present"])] for c in CATEGORIES}},
        indent=1))
    (PREP_DIR / "doc_info.json").write_text(json.dumps(info, indent=1))
    pd.DataFrame(page_rows).to_parquet(PREP_DIR / "pages.parquet")
    with (PREP_DIR / "questions.jsonl").open("w") as f:
        for q in qs:
            f.write(json.dumps(q) + "\n")
    print(f"cuad: {len(qs)} questions over {len(chosen)} contracts, {len(page_rows)} pseudo-pages -> {PREP_DIR}")


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


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    prepare(force=ap.parse_args().force)
