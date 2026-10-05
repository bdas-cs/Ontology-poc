"""Parse corpus PDFs to page text with PyMuPDF (process pool) → db/pages.parquet."""
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pymupdf

from poc import data


def _parse(doc_name: str) -> list[dict]:
    rows = []
    with pymupdf.open(data.PDF_DIR / f"{doc_name}.pdf") as pdf:
        for i, page in enumerate(pdf, start=1):
            rows.append({
                "doc_name": doc_name,
                "page": i,
                "text": page.get_text("text", sort=True),
                "n_images": len(page.get_images()),
            })
    return rows


def main() -> None:
    docs = data.corpus_docs()
    t0 = time.perf_counter()
    with ProcessPoolExecutor() as ex:
        rows = [r for doc_rows in ex.map(_parse, docs) for r in doc_rows]
    df = pd.DataFrame(rows)
    df.to_parquet(data.PAGES_PATH)
    empty = (df.text.str.strip().str.len() < 20).sum()
    print(f"{len(docs)} docs, {len(df)} pages, {empty} near-empty pages, "
          f"{df.text.str.len().sum() / 1e6:.1f}M chars in {time.perf_counter() - t0:.1f}s → {data.PAGES_PATH}")


if __name__ == "__main__":
    main()
