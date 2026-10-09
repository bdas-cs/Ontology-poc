"""Phase 1 answering: shared-index top-k chunks -> local Qwen3.6 (thinking off) -> answers-chunk.jsonl.

Retrieval is over all 84 filings; the question's gold doc_name is used only for the
gold_doc_hit / gold_page_hit diagnostics, never to filter.

  LLM_MAX_INFLIGHT=8 uv run python -m poc.retrieval.chunk.answer --k 10
  LLM_MAX_INFLIGHT=8 uv run python -m poc.retrieval.chunk.answer --k 20 --out answers-chunk-k20.jsonl
"""
import argparse
import asyncio
import json
import time

from poc import config, data, llm
from poc.retrieval.chunk.retrieve import hit_stats, search

SYSTEM = """You are a financial analyst answering questions about SEC filings and earnings reports.
Use ONLY the numbered excerpts provided; do not use outside knowledge. Each excerpt starts with a
header naming the company, document and page(s). Make sure you use the excerpts from the company
and fiscal period the question asks about.

Reply in exactly this format, and keep it short:
Answer: <the final answer first - the figure with units and period, or yes/no plus a one-line conclusion>
Working: <at most 5 short lines: the line items/inputs you used and the calculation>
Sources: [doc_name p.N], ...

Follow the question's requested units, rounding and formula if it gives one. Do not think aloud,
second-guess or re-check in the reply; pick the most relevant table and commit. If the excerpts
genuinely do not contain the information, say so in the Answer line and give your best estimate
from what is there."""

FINALIZE = ("Your previous reply was cut off. Based on it, state only the final answer in the required "
            "format (Answer / Working in at most 3 lines / Sources). No further deliberation.")


def build_prompt(question: str, hits: list[dict]) -> str:
    ctx = "\n\n".join(f"### Excerpt {i + 1}\n{h['text']}" for i, h in enumerate(hits))
    return f"{ctx}\n\n### Question\n{question}"


THINKING = False  # --thinking: control run matching the ontology arm's answer step


async def answer_one(q: dict, hits: list[dict], max_tokens: int) -> dict:
    t0 = time.perf_counter()
    out = await llm.chat(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(q["question"], hits)}],
        tag="chunk-answer", max_tokens=max_tokens, thinking=THINKING, temperature=0.0,
    )
    u = dict(out["usage"])
    content, retried = out["content"], False
    if u.get("completion_tokens", 0) >= max_tokens:  # ran out of budget mid-reasoning -> force a final answer
        retried = True
        out2 = await llm.chat(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(q["question"], hits)},
             {"role": "assistant", "content": content}, {"role": "user", "content": FINALIZE}],
            tag="chunk-answer", max_tokens=256, thinking=False, temperature=0.0,
        )
        content = out2["content"]
        for key in ("prompt_tokens", "completion_tokens"):
            u[key] = u.get(key, 0) + out2["usage"].get(key, 0)
        out = {**out, "wall_s": out["wall_s"] + out2["wall_s"]}
    return {
        "qid": q["qid"], "arm": "chunk", "answer": content, "truncated_retry": retried,
        "thinking": THINKING,
        "retrieved": [{"doc_name": h["doc_name"], "page": h["page_start"], "page_end": h["page_end"],
                       "score": round(float(h["score"]), 4)} for h in hits],
        **hit_stats(q, hits),
        "wall_s": round(time.perf_counter() - t0, 2), "llm_wall_s": round(out["wall_s"], 2),
        "prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0),
    }


async def run(qs: list[dict], hits: list[list[dict]], max_tokens: int, partial) -> list[dict]:
    """Each finished row is appended to `partial` immediately so a crash loses nothing (resume skips it)."""
    async def safe(q, h):
        for attempt in range(4):
            try:
                r = await answer_one(q, h, max_tokens)
                partial.write(json.dumps(r) + "\n")
                partial.flush()
                return r
            except Exception as e:  # server hiccup: back off and retry, then give up on this question
                print(f"  {q['qid']} attempt {attempt + 1} failed: {e!r}", flush=True)
                await asyncio.sleep(30 * (attempt + 1))
        return None
    return [r for r in await asyncio.gather(*(safe(q, h) for q, h in zip(qs, hits))) if r]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--table", default="chunks")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (smoke test)")
    ap.add_argument("--out", default="answers-chunk.jsonl")
    ap.add_argument("--fresh", action="store_true", help="ignore a previous partial run")
    ap.add_argument("--thinking", action="store_true", help="enable reasoning (use with --max-tokens 4000)")
    a = ap.parse_args()
    global THINKING
    THINKING = a.thinking
    all_qs = data.questions()[: a.limit or None]
    path = config.RUNS_DIR / a.out
    part = path.with_suffix(".partial.jsonl")
    done = {}
    if part.exists() and not a.fresh:
        done = {r["qid"]: r for r in map(json.loads, part.open())}
    elif part.exists():
        part.unlink()
    qs = [q for q in all_qs if q["qid"] not in done]
    print(f"{len(done)} already answered, {len(qs)} to go", flush=True)
    t0 = time.perf_counter()
    hits = search([q["question"] for q in qs], k=a.k, table=a.table) if qs else []
    t_ret = time.perf_counter() - t0
    with part.open("a") as pf:
        new = asyncio.run(run(qs, hits, a.max_tokens, pf)) if qs else []
    wall = time.perf_counter() - t0
    done.update({r["qid"]: r for r in new})
    rows = [done[q["qid"]] for q in all_qs if q["qid"] in done]
    missing = len(all_qs) - len(rows)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = len(rows)
    summary = {
        "out": str(path), "n": n, "missing": missing, "resumed_rows": n - len(new),
        "note": "wall_s/retrieval_s cover this invocation only; sum llm_wall_s over rows for per-question time", "k": a.k, "table": a.table, "retrieval_s": round(t_ret, 1),
        "wall_s": round(wall, 1), "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "completion_tokens": sum(r["completion_tokens"] for r in rows),
        "gold_doc_hit": round(sum(r["gold_doc_hit"] for r in rows) / n, 3),
        "gold_page_hit": round(sum(r["gold_page_hit"] for r in rows) / n, 3),
        "max_inflight": llm.MAX_INFLIGHT,
        "truncated_retries": sum(r["truncated_retry"] for r in rows),
    }
    (config.RUNS_DIR / (a.out.replace(".jsonl", "") + ".summary.json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
