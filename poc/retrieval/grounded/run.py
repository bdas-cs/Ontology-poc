"""Run the ontology-grounded arm over the active dataset (POC_DATASET; ontology domain from the adapter).

  uv run python -m poc.retrieval.grounded.run --dev        # 20-question dev slice -> answers-ontology-dev.jsonl
  uv run python -m poc.retrieval.grounded.run              # all questions (frozen pipeline) -> answers-ontology.jsonl
  uv run python -m poc.retrieval.grounded.run --limit 3    # smoke test: first 3 questions
  uv run python -m poc.retrieval.grounded.run --out answers-ontology-r2.jsonl --seed 2   # repeat run
  uv run python -m poc.eval.judge ontology --local         # provisional local judge

Only the question text is passed to the pipeline. Gold doc_name is read AFTER answering, solely to
compute the resolved_correct_filing diagnostic.

Concurrency: ONTO_QUESTION_CONCURRENCY questions in flight (default 8; each runs its LLM steps
sequentially); LLM_MAX_INFLIGHT still caps concurrent LLM requests (poc.llm).
--seed N: recorded on every row and used to shuffle the order questions are submitted in (so repeat runs
interleave differently on a batching server). Sampling itself stays at temperature 0.
"""
import argparse
import asyncio
import json
import os
import random
import time

from poc import config, data
from poc.retrieval.grounded import pipeline

DEV_IDS_PATH = config.RUNS_DIR / "ontology-dev-ids.json"
QUESTION_CONCURRENCY = int(os.getenv("ONTO_QUESTION_CONCURRENCY", "8"))


def dev_ids(n: int = 20, seed: int = 7) -> list[str]:
    """Stratified dev slice (fixed seed), cached per dataset in runs/ontology-dev-ids.json.
    FinanceBench (3 types): 7 + 7 + 6, as originally drawn."""
    if DEV_IDS_PATH.exists():
        return json.loads(DEV_IDS_PATH.read_text())
    rng = random.Random(seed)
    types = sorted({q["question_type"] for q in data.questions()})
    if len(types) == 3 and n == 20:
        quota = [7, 7, 6]
    else:
        quota = [n // len(types) + (1 if i < n % len(types) else 0) for i in range(len(types))]
    ids = []
    for t, k in zip(types, quota):
        pool = [q["qid"] for q in data.questions() if q["question_type"] == t]
        ids += rng.sample(pool, min(k, len(pool)))
    DEV_IDS_PATH.write_text(json.dumps(ids, indent=1))
    return ids


async def run(qs: list[dict], arm: str, out_path, plan_thinking: bool, seed: int | None = None) -> None:
    done = set()
    if out_path.exists():
        done = {json.loads(l)["qid"] for l in out_path.open()}
    todo = [q for q in qs if q["qid"] not in done]
    if seed is not None:
        random.Random(seed).shuffle(todo)
    print(f"{len(todo)} to answer ({len(done)} already in {out_path.name}); dataset={config.DATASET} "
          f"domain={pipeline.dom().NAME} concurrency={QUESTION_CONCURRENCY}")
    t0 = time.perf_counter()

    qsem = asyncio.Semaphore(QUESTION_CONCURRENCY)

    async def one(q):
        async with qsem:
            await _one(q)

    async def _one(q):
        for attempt in range(4):  # infra retries: the supervised server restarts after a Metal crash (~1 min)
            try:
                r = await pipeline.answer(q["question"], plan_thinking=plan_thinking)  # question text only
                break
            except Exception as e:
                err = e
                await asyncio.sleep(90)
        else:
            e = err
            r = {"answer": f"(pipeline error: {type(e).__name__}: {e})", "resolved": {"company": None, "filings": []},
                 "sql": None, "records_used": [], "pages_used": [], "wall_s": None, "prompt_tokens": 0,
                 "completion_tokens": 0}
        # diagnostic only, computed after answering:
        r["resolved_correct_filing"] = q["doc_name"] in r["resolved"]["filings"]
        r["resolved_primary_correct"] = bool(r["resolved"]["filings"]) and r["resolved"]["filings"][0] == q["doc_name"]
        row = {"qid": q["qid"], "arm": arm, "dataset": config.DATASET, **({"seed": seed} if seed is not None else {}), **r}
        with out_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{q['qid']}] {r['wall_s']}s ok={r['resolved_correct_filing']} {r['answer'][:110]!r}")

    await asyncio.gather(*(one(q) for q in todo))
    print(f"wall {time.perf_counter() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (smoke test)")
    ap.add_argument("--arm", default=None, help="arm label (default: from --out, else ontology / ontology-dev)")
    ap.add_argument("--out", default=None, help="answers file in runs/ (default answers-<arm>.jsonl)")
    ap.add_argument("--seed", type=int, default=None, help="repeat-run seed (recorded; shuffles submission order)")
    ap.add_argument("--no-plan-thinking", dest="plan_thinking", action="store_false")
    ap.add_argument("--fresh", action="store_true", help="overwrite the answers file")
    a = ap.parse_args()
    qs = data.questions()
    if a.dev:
        ids = set(dev_ids())
        qs = [q for q in qs if q["qid"] in ids]
    if a.ids:
        qs = [q for q in qs if q["qid"] in set(a.ids)]
    if a.limit:
        qs = qs[: a.limit]
    if a.out:
        out = config.RUNS_DIR / a.out
        arm = a.arm or (out.stem[len("answers-"):] if out.stem.startswith("answers-") else out.stem)
    else:
        arm = a.arm or ("ontology-dev" if a.dev else "ontology")
        out = config.RUNS_DIR / f"answers-{arm}.jsonl"
    if a.fresh and out.exists():
        out.unlink()
    asyncio.run(run(qs, arm, out, a.plan_thinking, a.seed))


if __name__ == "__main__":
    main()
