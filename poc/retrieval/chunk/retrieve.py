"""Top-k dense retrieval over the shared LanceDB chunk index (all 84 filings, no doc filter)."""
from functools import lru_cache

from poc import config

DB_PATH = config.DB_DIR / "lancedb"


@lru_cache
def _model():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(config.EMBED_MODEL, device="mps")
    m.max_seq_length = 512
    return m


@lru_cache
def _table(name: str):
    import lancedb
    return lancedb.connect(str(DB_PATH)).open_table(name)


def embed_queries(texts: list[str]):
    return _model().encode(texts, prompt_name="query", normalize_embeddings=True,
                           batch_size=32, convert_to_numpy=True, show_progress_bar=False)


def search(questions: list[str], k: int = 10, table: str = "chunks") -> list[list[dict]]:
    """Returns, per question, k hits: {doc_name, page_start, page_end, text, score (cosine sim)}."""
    tbl = _table(table)
    out = []
    for v in embed_queries(questions):
        df = (tbl.search(v.tolist()).metric("cosine").limit(k)
              .select(["doc_name", "page_start", "page_end", "text"]).to_pandas())
        df["score"] = 1.0 - df["_distance"]
        out.append(df[["doc_name", "page_start", "page_end", "text", "score"]].to_dict("records"))
    return out


def gold_pages(q: dict) -> set[tuple[str, int]]:
    """FinanceBench evidence_page_num is 0-based; pages.parquet is 1-based (verified: 160/189
    evidence snippets match page num+1, 3 match num, rest are fuzzy/OCR)."""
    return {(e.get("doc_name") or q["doc_name"], int(e["evidence_page_num"]) + 1) for e in q.get("evidence", [])}


def hit_stats(q: dict, hits: list[dict]) -> dict:
    gp = gold_pages(q)
    gdocs = {d for d, _ in gp} | {q["doc_name"]}
    doc_hit = any(h["doc_name"] in gdocs for h in hits)
    page_hit = any(h["doc_name"] == d and h["page_start"] <= p <= h["page_end"] for h in hits for d, p in gp)
    first = next((i for i, h in enumerate(hits) if h["doc_name"] in gdocs), None)
    return {"gold_doc_hit": doc_hit, "gold_page_hit": page_hit, "gold_doc_rank": first}
