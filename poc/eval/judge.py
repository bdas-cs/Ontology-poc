"""Judge answers against FinanceBench ground truth.

Input:  runs/answers-<arm>.jsonl   rows: {financebench_id, arm, answer, ...}
Output: runs/judged-<arm>.jsonl    rows: input + {strict, expanded, rationale, judge}
Prints strict / expanded accuracy overall and by question_type.

Judge = Claude (JUDGE_MODEL, default claude-opus-5-5) when Anthropic credentials are
available; otherwise the local model with `--local` (labelled, provisional).

  uv run python -m poc.eval.judge chunk ontology
"""
import argparse
import asyncio
import json

import anthropic
import pandas as pd

from poc import config, data, llm

SYSTEM = """You grade answers to financial questions about SEC filings against a ground-truth answer.

Return two verdicts:
- strict: true only if the candidate reaches the same answer as the ground truth. For numbers,
  accept differences due to rounding or units (e.g. $1,577M vs $1.58B) and within 1% relative;
  a correct number with a wrong unit or period is false. For yes/no or qualitative questions the
  conclusion must match.
- expanded: true if strict is true, OR the candidate gives a different but correctly reasoned
  answer that the filing evidence supports (e.g. a defensible alternative definition of a metric,
  or a qualitative judgement backed by the right figures).
A refusal, "not found", or hedged non-answer is false on both.
Keep the rationale to one or two sentences."""

SCHEMA = {
    "type": "object",
    "properties": {
        "strict": {"type": "boolean"},
        "expanded": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": ["strict", "expanded", "rationale"],
    "additionalProperties": False,
}


def _prompt(q: dict, answer: str) -> str:
    evidence = "\n---\n".join(e.get("evidence_text", "")[:3000] for e in q.get("evidence", []))
    return (
        f"Question:\n{q['question']}\n\n"
        f"Ground-truth answer:\n{q['answer']}\n\n"
        f"Ground-truth justification:\n{q.get('justification') or '(none)'}\n\n"
        f"Filing evidence:\n{evidence}\n\n"
        f"Candidate answer:\n{answer}"
    )


async def _judge_claude(client: anthropic.AsyncAnthropic, q: dict, answer: str) -> dict:
    resp = await client.beta.messages.create(
        model=config.JUDGE_MODEL,
        max_tokens=4000,
        system=SYSTEM,
        messages=[{"role": "user", "content": _prompt(q, answer)}],
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if resp.stop_reason == "refusal":
        return {"strict": False, "expanded": False, "rationale": "judge refused", "judge": config.JUDGE_MODEL}
    text = next(b.text for b in resp.content if b.type == "text")
    return {**json.loads(text), "judge": resp.model}


async def _judge_local(q: dict, answer: str) -> dict:
    out = await llm.chat(
        [{"role": "system", "content": SYSTEM + "\nReply with JSON: {\"strict\": bool, \"expanded\": bool, \"rationale\": str}"},
         {"role": "user", "content": _prompt(q, answer)}],
        tag="judge-local", max_tokens=400, temperature=0.0, json_mode=True,
    )
    return {**llm.parse_json(out["content"]), "judge": f"local:{config.LLM_MODEL}"}


async def judge_arm(arm: str, local: bool, concurrency: int = 8) -> pd.DataFrame:
    qs = {q["financebench_id"]: q for q in data.questions()}
    src = config.RUNS_DIR / f"answers-{arm}.jsonl"
    rows = [json.loads(l) for l in src.open()]
    client = None if local else anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(concurrency)

    async def one(row: dict) -> dict:
        q = qs[row["financebench_id"]]
        async with sem:
            v = await (_judge_local(q, row["answer"]) if local else _judge_claude(client, q, row["answer"]))
        return {**row, **v, "question_type": q["question_type"], "gold": q["answer"]}

    judged = await asyncio.gather(*(one(r) for r in rows))
    out = config.RUNS_DIR / f"judged-{arm}{'-localjudge' if local else ''}.jsonl"
    with out.open("w") as f:
        for r in judged:
            f.write(json.dumps(r) + "\n")
    return pd.DataFrame(judged)


def report(df: pd.DataFrame, arm: str) -> None:
    n = len(df)
    print(f"\n== {arm}  (n={n}, judge={df.judge.iloc[0]})")
    print(f"strict   {df.strict.mean():.1%}\nexpanded {df.expanded.mean():.1%}")
    print(df.groupby("question_type")[["strict", "expanded"]].mean().round(3).to_string())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("arms", nargs="+")
    p.add_argument("--local", action="store_true", help="use the local model as a provisional judge")
    a = p.parse_args()
    for arm in a.arms:
        report(asyncio.run(judge_arm(arm, a.local)), arm)


if __name__ == "__main__":
    main()
