"""Run the ontology-grounded arm over FinanceBench.

  uv run python -m poc.retrieval.grounded.run --dev        # 20-question dev slice -> answers-ontology-dev.jsonl
  uv run python -m poc.retrieval.grounded.run              # all 150 (frozen pipeline) -> answers-ontology.jsonl
  uv run python -m poc.eval.judge ontology --local         # provisional local judge

Only the question text is passed to the pipeline. Gold doc_name is read AFTER answering, solely to
compute the resolved_correct_filing diagnostic.
"""
import argparse
import asyncio
import json
import random
import time

from poc import config, data
from poc.retrieval.grounded import pipeline

DEV_IDS_PATH = config.RUNS_DIR / "ontology-dev-ids.json"
QUESTION_CONCURRENCY = 8  # questions in flight; each runs its LLM steps sequentially


def dev_ids(n_per_type: int = 7, seed: int = 7) -> list[str]:
    """Stratified dev slice: 7 + 7 + 6 questions across the three question types (fixed seed)."""
    if DEV_IDS_PATH.exists():
        return json.loads(DEV_IDS_PATH.read_text())
    rng = random.Random(seed)
    ids = []
    for i, t in enumerate(sorted({q["question_type"] for q in data.questions()})):
        pool = [q["financebench_id"] for q in data.questions() if q["question_type"] == t]
        ids += rng.sample(pool, n_per_type - (1 if i == 2 else 0))
    DEV_IDS_PATH.write_text(json.dumps(ids, indent=1))
    return ids


async def run(qs: list[dict], arm: str, out_path, plan_thinking: bool) -> None:
    done = set()
    if out_path.exists():
        done = {json.loads(l)["financebench_id"] for l in out_path.open()}
    todo = [q for q in qs if q["financebench_id"] not in done]
    print(f"{len(todo)} to answer ({len(done)} already in {out_path.name})")
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
        row = {"financebench_id": q["financebench_id"], "arm": arm, **r}
        with out_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{q['financebench_id']}] {r['wall_s']}s ok={r['resolved_correct_filing']} {r['answer'][:110]!r}")

    await asyncio.gather(*(one(q) for q in todo))
    print(f"wall {time.perf_counter() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--arm", default=None)
    ap.add_argument("--no-plan-thinking", dest="plan_thinking", action="store_false")
    ap.add_argument("--fresh", action="store_true", help="overwrite the answers file")
    a = ap.parse_args()
    qs = data.questions()
    if a.dev:
        ids = set(dev_ids())
        qs = [q for q in qs if q["financebench_id"] in ids]
    if a.ids:
        qs = [q for q in qs if q["financebench_id"] in set(a.ids)]
    arm = a.arm or ("ontology-dev" if a.dev else "ontology")
    out = config.RUNS_DIR / f"answers-{arm}.jsonl"
    if a.fresh and out.exists():
        out.unlink()
    asyncio.run(run(qs, arm, out, a.plan_thinking))


if __name__ == "__main__":
    main()
