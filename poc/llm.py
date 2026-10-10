"""Shared async client for the local OpenAI-compatible LLM server.

Every call goes through `chat()`, which enforces a global admission limit
(semaphore) and appends a row to the run ledger, so token counts, latency
and concurrency can be analysed per phase afterwards (paper §4–5).
"""
import asyncio
import json
import os
import random
import time
from pathlib import Path

import httpx

from poc import config

MAX_INFLIGHT = int(os.getenv("LLM_MAX_INFLIGHT", "16"))
_sem: asyncio.Semaphore | None = None
_client: httpx.AsyncClient | None = None
_loop: asyncio.AbstractEventLoop | None = None


def _get() -> tuple[asyncio.Semaphore, httpx.AsyncClient]:
    """Per-event-loop semaphore and client: a process that calls asyncio.run() more than once
    (e.g. `ontology.ingest all`) must not reuse connections bound to a closed loop."""
    global _sem, _client, _loop
    loop = asyncio.get_running_loop()
    if _sem is None or _loop is not loop:
        _loop = loop
        _sem = asyncio.Semaphore(MAX_INFLIGHT)
        _client = httpx.AsyncClient(timeout=1800, limits=httpx.Limits(max_connections=MAX_INFLIGHT * 2))
    return _sem, _client


def _ledger_path(tag: str) -> Path:
    return config.RUNS_DIR / f"llm-{tag}.jsonl"


def split_reasoning(msg: dict) -> tuple[str, str]:
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if "</think>" in content:
        r, content = content.split("</think>", 1)
        reasoning = reasoning or r.replace("<think>", "")
    return reasoning.strip(), content.strip()


async def chat(
    messages: list[dict],
    *,
    tag: str,
    max_tokens: int = 1024,
    thinking: bool = False,
    temperature: float = 0.2,
    json_mode: bool = False,
    retries: int = 3,
) -> dict:
    """Returns {"content", "reasoning", "usage", "wall_s"} and logs to runs/llm-<tag>.jsonl."""
    sem, client = _get()
    body = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    for attempt in range(retries + 1):
        try:
            async with sem:
                t0 = time.perf_counter()
                r = await client.post(f"{config.LLM_BASE_URL}/chat/completions", json=body)
                r.raise_for_status()
                wall = time.perf_counter() - t0
            break
        except (httpx.HTTPError,) as e:
            if attempt == retries:
                raise
            # Jittered backoff so workers don't retry in lockstep (paper §4).
            await asyncio.sleep((2**attempt) + random.random() * 2)
    data = r.json()
    reasoning, content = split_reasoning(data["choices"][0]["message"])
    usage = data.get("usage", {})
    row = {
        "ts": time.time(),
        "tag": tag,
        "wall_s": round(wall, 3),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "thinking": thinking,
        "reasoning_chars": len(reasoning),
        "finish_reason": data["choices"][0].get("finish_reason"),
    }
    with _ledger_path(tag).open("a") as f:
        f.write(json.dumps(row) + "\n")
    return {"content": content, "reasoning": reasoning, "usage": usage, "wall_s": wall}


def parse_json(text: str):
    """Lenient JSON extraction from model output (strips code fences / prose)."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1].rsplit("```", 1)[0]
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = t.find(opener), t.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(t[i : j + 1])
            except json.JSONDecodeError:
                continue
    return json.loads(t)
