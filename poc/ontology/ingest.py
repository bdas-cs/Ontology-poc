"""Build the ontology (DuckDB) from the parsed corpus — generic driver over the active domain plugin
(poc.ontology.domains; finance for financebench/docfinqa, contracts for cuad, gov for bharat).

Steps (each idempotent; `all` runs them in order):
  base     domain objects from the system of record (plugin.step_base: doc_info + page text, may use the
           LLM e.g. for company aliases) + generic Page rows (scope id = plugin.scope_id(doc_name))
  extract  LLM extraction of typed records (plugin.step_extract; ledger tag "ontology-ingest")
  load     re-load cached extractions only (no LLM calls)
  embed    chunk vectors for scoped dense search (reuses the Phase-1 LanceDB table when present)
  fts      BM25 index over chunks for scoped keyword search

  uv run python -m poc.ontology.ingest all
  uv run python -m poc.ontology.ingest all --only DOC_A DOC_B      # smoke test on a few documents
  uv run python -m poc.ontology.ingest all --docs docs.txt          # same, doc_names from a file (one per line)
  ONTO_DOCS=docs.txt make ontology-ingest                           # env form (a file path or comma-separated)
Stats -> runs/ingest-ontology.json (full runs) or runs/ingest-ontology-subset.json (--only / --docs runs)
"""
import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from poc import config, data
from poc.ontology import domains, schema
from poc.ontology.runtime import rw, squeeze

STATS_PATH = config.RUNS_DIR / "ingest-ontology.json"
SUBSET_STATS_PATH = config.RUNS_DIR / "ingest-ontology-subset.json"
CHUNK_TABLE = os.getenv("ONTOLOGY_CHUNK_TABLE", "chunks")


def _load_stats() -> dict:
    return json.loads(STATS_PATH.read_text()) if STATS_PATH.exists() else {}


def _save_stats(upd: dict) -> None:
    s = _load_stats()
    s.update(upd)
    STATS_PATH.write_text(json.dumps(s, indent=2))


def _pages(only: list[str] | None = None) -> pd.DataFrame:
    pg = data.pages()
    return pg[pg.doc_name.isin(only)] if only else pg


# ---------------------------------------------------------------- base objects
def step_base(only: list[str] | None = None) -> dict:
    t0 = time.perf_counter()
    dom = domains.get()
    info = data.doc_info()
    docs = [d for d in data.corpus_docs() if not only or d in only]
    pg = _pages(docs)
    st = dom.step_base(docs, info, pg)
    con = rw()
    con.execute("DELETE FROM page")
    pdf = pg.assign(filing_id=pg.doc_name.map(dom.scope_id))[["filing_id", "doc_name", "page", "text"]]
    con.execute("INSERT INTO page SELECT * FROM pdf")
    con.close()
    return {"domain": dom.NAME, **st, "pages": len(pdf), "base_wall_s": round(time.perf_counter() - t0, 1)}


# ---------------------------------------------------------------- typed extraction
def step_extract(only: list[str] | None = None, cache_only: bool = False) -> dict:
    return domains.get().step_extract(_pages(only), only=only, cache_only=cache_only)


# ---------------------------------------------------------------- chunks / embeddings / FTS
def _chunks(text: str, size: int = 1400, overlap: int = 200) -> list[str]:
    t = squeeze(text).strip()
    if not t:
        return []
    out, i = [], 0
    while i < len(t):
        out.append(t[i : i + size])
        if i + size >= len(t):
            break
        i += size - overlap
    return out


_EMB = None


def embedder():
    global _EMB
    if _EMB is None:
        from sentence_transformers import SentenceTransformer
        _EMB = SentenceTransformer(config.EMBED_MODEL, device=config.torch_device())
        _EMB.max_seq_length = 512
    return _EMB


def step_embed(only: list[str] | None = None) -> dict:
    """Chunk vectors for scoped dense search. Reuses the Phase-1 LanceDB table (same embedding model over
    the same pages) when it exists, so both arms share identical text vectors; otherwise embeds locally."""
    t0 = time.perf_counter()
    sid = domains.get().scope_id
    lpath = config.DB_DIR / "lancedb"
    try:
        import lancedb
        tbl = lancedb.connect(str(lpath)).open_table(CHUNK_TABLE)
        src = tbl.to_pandas()
        if only:
            src = src[src.doc_name.isin(only)]
        if src.empty:
            raise ValueError(f"lancedb:{CHUNK_TABLE} has no chunks for these documents")
        df = pd.DataFrame({"id": "K:" + src.doc_name + ":" + src.chunk_id.astype(str), "filing_id": src.doc_name.map(sid),
                           "doc_name": src.doc_name, "page": src.page_start.astype(int), "page_end": src.page_end.astype(int),
                           "text": src.text})
        df["emb"] = [v.astype("float32").tolist() for v in src.vector]
        origin = f"reused lancedb:{CHUNK_TABLE} (Phase-1 index)"
    except Exception as e:
        print("lancedb reuse failed, embedding locally:", e)
        pg = _pages(only)
        rows = []
        for d, p, t in zip(pg.doc_name, pg.page, pg.text):
            for j, c in enumerate(_chunks(t)):
                rows.append((f"K:{d}:{p}:{j}", sid(d), d, int(p), int(p), c))
        embs = embedder().encode([r[5] for r in rows], batch_size=32, normalize_embeddings=True, show_progress_bar=True)
        df = pd.DataFrame(rows, columns=["id", "filing_id", "doc_name", "page", "page_end", "text"])
        df["emb"] = [e.astype("float32").tolist() for e in embs]
        origin = "embedded locally"
    con = rw()
    con.execute("DROP TABLE IF EXISTS chunk")
    con.execute(schema.GENERIC_DDL)
    con.execute("INSERT INTO chunk SELECT id, filing_id, doc_name, page, page_end, text, emb::FLOAT[1024] FROM df")
    con.close()
    return {"chunks": len(df), "chunk_vectors": origin, "embed_wall_s": round(time.perf_counter() - t0, 1)}


def step_fts() -> dict:
    t0 = time.perf_counter()
    con = rw()
    con.execute("INSTALL fts; LOAD fts")
    con.execute("PRAGMA create_fts_index('chunk', 'id', 'text', overwrite=1)")
    con.close()
    return {"fts_wall_s": round(time.perf_counter() - t0, 1)}


def _doc_list(spec: str) -> list[str]:
    """A file of doc_names (one per line) or a comma-separated list."""
    p = Path(spec).expanduser()
    items = p.read_text().splitlines() if p.exists() else spec.split(",")
    return [x.strip() for x in items if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["base", "extract", "load", "embed", "fts", "all"])
    ap.add_argument("--only", nargs="*", help="restrict every step to these doc_names (smoke tests / dev)")
    ap.add_argument("--docs", default=os.getenv("ONTO_DOCS"),
                    help="restrict every step to the doc_names in this file (or comma list); env ONTO_DOCS")
    a = ap.parse_args()
    if a.docs:
        a.only = list(dict.fromkeys([*(a.only or []), *_doc_list(a.docs)]))
        print(f"restricted to {len(a.only)} documents ({a.docs})")
    steps = ["base", "extract", "embed", "fts"] if a.step == "all" else [a.step]
    for s in steps:
        print(f"== {s} (domain={domains.get().NAME}, dataset={config.DATASET})")
        st = {"base": lambda: step_base(a.only), "extract": lambda: step_extract(a.only),
              "load": lambda: step_extract(a.only, cache_only=True),
              "embed": lambda: step_embed(a.only), "fts": step_fts}[s]()
        print(json.dumps(st, indent=1))
        if not a.only and s != "load":
            _save_stats(st)
        elif a.only:
            prev = json.loads(SUBSET_STATS_PATH.read_text()) if SUBSET_STATS_PATH.exists() else {}
            SUBSET_STATS_PATH.write_text(json.dumps({**prev, **st, "subset_docs": len(a.only)}, indent=2))
    if not a.only:
        s = _load_stats()
        s["total_wall_s"] = round(sum(v for k, v in s.items() if k.endswith("_wall_s") and k != "total_wall_s"), 1)
        s["total_llm_prompt_tokens"] = s.get("base_llm_prompt_tokens", 0) + s.get("extract_llm_prompt_tokens", 0)
        s["total_llm_completion_tokens"] = s.get("base_llm_completion_tokens", 0) + s.get("extract_llm_completion_tokens", 0)
        STATS_PATH.write_text(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
