"""Phase 1 ingest: pages.parquet -> token-window chunks -> Qwen3-Embedding -> LanceDB.

Chunks never cross documents; each keeps doc_name + page range and is prefixed with a
short header ("<company> | <doc type> | <period> | <doc_name> p.a-b") so the retriever
can match company / fiscal period from the question.

  uv run python -m poc.ingest.chunk                      # default 1000 tok, 150 overlap -> table "chunks"
  uv run python -m poc.ingest.chunk --size 600 --table chunks_c600
Stats -> runs/ingest-<table>.json (runs/ingest-chunk.json for the default table).
"""
import argparse
import json
import os
import time

import pandas as pd

from poc import config, data

DB_PATH = config.DB_DIR / "lancedb"
DOC_TYPE = {"10k": "10-K annual report", "10q": "10-Q quarterly report", "8k": "8-K current report",
            "earnings": "earnings release"}


def header(doc_name: str, p0: int, p1: int) -> str:
    info = data.doc_info().get(doc_name, {})
    company = info.get("company", doc_name.split("_")[0])
    dtype = DOC_TYPE.get(str(info.get("doc_type", "")).lower(), info.get("doc_type", ""))
    period = info.get("doc_period", "")
    pages = f"p.{p0}" if p0 == p1 else f"p.{p0}-{p1}"
    return f"{company} | {dtype} | FY{period} | {doc_name} {pages}"


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(config.EMBED_MODEL)


def make_chunks(pages: pd.DataFrame, tok, size: int = 1000, overlap: int = 150) -> list[dict]:
    """Line-packing token windows per document, carrying ~`overlap` tokens of trailing lines."""
    out: list[dict] = []
    for doc_name, g in pages.sort_values(["doc_name", "page"]).groupby("doc_name", sort=False):
        units: list[tuple[int, str]] = []  # (page, line)
        for page, text in zip(g.page, g.text):
            for line in (text or "").splitlines():
                line = " ".join(line.split())  # collapse layout whitespace from sort=True
                if line:
                    units.append((int(page), line))
        if not units:
            continue
        counts = [len(ids) + 1 for ids in tok([u[1] for u in units], add_special_tokens=False)["input_ids"]]
        # split pathological long lines by characters
        expanded: list[tuple[int, str, int]] = []
        for (pg, line), n in zip(units, counts):
            if n <= size // 2:
                expanded.append((pg, line, n))
            else:
                k = -(-n // (size // 2))
                step = -(-len(line) // k)
                for i in range(0, len(line), step):
                    expanded.append((pg, line[i:i + step], -(-n // k)))
        cur: list[tuple[int, str, int]] = []
        cur_n = 0

        def emit():
            p0, p1 = cur[0][0], cur[-1][0]
            body = "\n".join(u[1] for u in cur)
            out.append({"doc_name": doc_name, "page_start": p0, "page_end": p1,
                        "text": f"[{header(doc_name, p0, p1)}]\n{body}", "n_tokens": cur_n})

        for u in expanded:
            if cur and cur_n + u[2] > size:
                emit()
                tail, tn = [], 0
                for v in reversed(cur):
                    if tn + v[2] > overlap:
                        break
                    tail.insert(0, v)
                    tn += v[2]
                cur, cur_n = tail, tn
            cur.append(u)
            cur_n += u[2]
        if cur and (cur_n > overlap or len(out) == 0 or out[-1]["doc_name"] != doc_name):
            emit()
    for i, c in enumerate(out):
        c["chunk_id"] = i
    return out


def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL, device=config.torch_device())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--table", default="chunks")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    import lancedb
    import torch

    t0 = time.perf_counter()
    tok = load_tokenizer()
    pages = data.pages()
    pages = pages[pages.doc_name.isin(data.corpus_docs())]
    chunks = make_chunks(pages, tok, a.size, a.overlap)
    t_chunk = time.perf_counter() - t0
    print(f"{len(chunks)} chunks from {pages.doc_name.nunique()} docs in {t_chunk:.1f}s", flush=True)

    model = load_embedder()
    model.max_seq_length = a.size + 256
    texts = [c["text"] for c in chunks]
    t1 = time.perf_counter()
    # length-sorted batches minimise padding
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    vecs = [None] * len(texts)
    B = a.batch * 64
    for s in range(0, len(order), B):
        idx = order[s:s + B]
        emb = model.encode([texts[i] for i in idx], batch_size=a.batch, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
        for i, e in zip(idx, emb):
            vecs[i] = e
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        el = time.perf_counter() - t1
        print(f"  embedded {min(s + B, len(order))}/{len(order)}  {el:.0f}s", flush=True)
    t_embed = time.perf_counter() - t1
    embed_tokens = sum(len(x) for x in tok(texts, add_special_tokens=False)["input_ids"])

    db = lancedb.connect(str(DB_PATH))
    rows = [{**c, "vector": v.tolist()} for c, v in zip(chunks, vecs)]
    tbl = db.create_table(a.table, data=rows, mode="overwrite")
    wall = time.perf_counter() - t0
    stats = {
        "table": a.table, "embed_model": config.EMBED_MODEL, "chunk_size_tokens": a.size,
        "overlap_tokens": a.overlap, "n_docs": int(pages.doc_name.nunique()), "n_pages": len(pages),
        "n_chunks": len(chunks), "embedding_tokens": int(embed_tokens),
        "mean_chunk_tokens": round(embed_tokens / len(chunks), 1),
        "chunk_s": round(t_chunk, 1), "embed_s": round(t_embed, 1),
        "embed_tok_per_s": round(embed_tokens / t_embed, 1), "wall_s": round(wall, 1),
        "device": config.torch_device(), "batch_size": a.batch, "dim": len(vecs[0]), "ts": time.time(),
    }
    name = "ingest-chunk.json" if a.table == "chunks" else f"ingest-{a.table}.json"
    (config.RUNS_DIR / name).write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), f"rows={tbl.count_rows()}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    main()
