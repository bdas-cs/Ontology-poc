"""Phase 0 / Phase 3 benchmark against the local OpenAI-compatible endpoint.

Default: one request with reasoning on, one with reasoning off, reporting
latency, tokens/sec and how much of the output was reasoning.
--sweep 1,2,4,8: N concurrent requests per level, reporting aggregate tokens/sec.
Every call is written to a run ledger: runs/bench-<kind>-<ts>.jsonl, or
runs/bench-<kind>-<tag>.jsonl with --tag (stable name, used by scripts/gpu_run.sh to resume).
"""
import argparse
import asyncio
import json
import time

import httpx
from rich.console import Console
from rich.table import Table

from poc import config

PROMPT = (
    "3M's FY2018 capital expenditures were $1,577M and revenue was $32,765M. "
    "What was capex as a percentage of revenue? Answer in one sentence."
)

# Sweep uses a long-output prompt so decode throughput dominates.
SWEEP_PROMPT = (
    "Explain, in about 400 words, how a company's capital expenditures, depreciation "
    "and free cash flow relate, using a manufacturing company as the example."
)

console = Console()


async def call(client: httpx.AsyncClient, thinking: bool, max_tokens: int, prompt: str = PROMPT) -> dict:
    body = {
        "model": config.LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    t0 = time.perf_counter()
    r = await client.post(f"{config.LLM_BASE_URL}/chat/completions", json=body)
    r.raise_for_status()
    wall = time.perf_counter() - t0
    data = r.json()
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    # Some servers inline reasoning as <think>...</think> in content.
    if "</think>" in content:
        reasoning, content = content.split("</think>", 1)
    usage = data.get("usage", {})
    out_tok = usage.get("completion_tokens", 0)
    return {
        "thinking": thinking,
        "wall_s": round(wall, 3),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": out_tok,
        "tok_per_s": round(out_tok / wall, 1) if wall else 0,
        "reasoning_chars": len(reasoning),
        "answer_chars": len(content.strip()),
        "answer": content.strip()[:200],
    }


RUN_TAG = ""


def ledger(rows: list[dict], tag: str) -> None:
    path = config.RUNS_DIR / f"bench-{tag}-{RUN_TAG or int(time.time())}.jsonl"
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    console.print(f"[dim]ledger → {path}[/dim]")


async def single(max_tokens: int) -> None:
    async with httpx.AsyncClient(timeout=600) as client:
        rows = [await call(client, t, max_tokens) for t in (True, False)]
    table = Table("thinking", "wall s", "out tok", "tok/s", "reasoning share", "answer")
    for r in rows:
        total = r["reasoning_chars"] + r["answer_chars"] or 1
        table.add_row(
            str(r["thinking"]), str(r["wall_s"]), str(r["completion_tokens"]),
            str(r["tok_per_s"]), f"{r['reasoning_chars'] / total:.0%}", r["answer"][:80],
        )
    console.print(table)
    ledger(rows, "smoke")


async def sweep(levels: list[int], thinking: bool, max_tokens: int) -> None:
    rows, table = [], Table("concurrency", "wall s", "total out tok", "aggregate tok/s", "per-req tok/s")
    conns = max(64, max(levels))
    limits = httpx.Limits(max_connections=conns, max_keepalive_connections=conns)
    async with httpx.AsyncClient(timeout=1800, limits=limits) as client:
        await call(client, False, 16)  # warm-up
        for n in levels:
            t0 = time.perf_counter()
            res = await asyncio.gather(*(call(client, thinking, max_tokens, SWEEP_PROMPT) for _ in range(n)))
            wall = time.perf_counter() - t0
            out = sum(r["completion_tokens"] for r in res)
            per_req = sum(r["tok_per_s"] for r in res) / n
            table.add_row(str(n), f"{wall:.1f}", str(out), f"{out / wall:.1f}", f"{per_req:.1f}")
            rows += [{**r, "concurrency": n, "level_wall_s": round(wall, 3)} for r in res]
    console.print(table)
    ledger(rows, "sweep")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", help="comma-separated concurrency levels, e.g. 1,2,4,8")
    p.add_argument("--thinking", action="store_true", help="enable reasoning in sweep")
    p.add_argument("--max-tokens", type=int, default=600)
    p.add_argument("--tag", default="", help="ledger name suffix instead of a timestamp")
    a = p.parse_args()
    global RUN_TAG
    RUN_TAG = a.tag
    if a.sweep:
        asyncio.run(sweep([int(x) for x in a.sweep.split(",")], a.thinking, a.max_tokens))
    else:
        asyncio.run(single(a.max_tokens))


if __name__ == "__main__":
    main()
