"""Domain-independent helpers shared by the ontology ingest driver, the grounded pipeline and the domain
plugins (poc/ontology/domains/*): LLM call wrapper with infra retries, token accounting, DuckDB
connections, a safe arithmetic evaluator and small text utilities."""
import ast
import asyncio
import math
import operator
import re
import time

import duckdb

from poc import llm
from poc.ontology import schema

INGEST_TAG = "ontology-ingest"
ANSWER_TAG = "ontology-answer"


class Usage:
    def __init__(self):
        self.prompt = self.completion = self.calls = 0

    def add(self, out: dict):
        u = out.get("usage") or {}
        self.prompt += u.get("prompt_tokens", 0)
        self.completion += u.get("completion_tokens", 0)
        self.calls += 1


async def chat(messages, **kw) -> dict:
    """poc.llm.chat with longer infra retries: the supervised server needs ~1 min to come back after a
    Metal crash, longer than poc.llm's own 3 quick retries. Pipeline logic is unchanged."""
    for attempt in range(8):
        try:
            return await llm.chat(messages, **kw)
        except Exception:
            if attempt == 7:
                raise
            await asyncio.sleep(30)


# ------------------------------------------------------------------ DuckDB
_RO = None


def ro() -> duckdb.DuckDBPyConnection:
    """Shared read-only connection for the query path (FTS loaded)."""
    global _RO
    if _RO is None:
        _RO = duckdb.connect(str(schema.DB_PATH), read_only=True)
        _RO.execute("LOAD fts")
    return _RO


def rw() -> duckdb.DuckDBPyConnection:
    """Write connection for ingest steps; waits while another ingest step holds the lock."""
    for _ in range(60):
        try:
            return schema.connect()
        except Exception as e:  # duckdb.IOException on lock conflict
            if "lock" not in str(e).lower():
                raise
            time.sleep(5)
    return schema.connect()


def has_table(c: duckdb.DuckDBPyConnection, name: str) -> bool:
    return bool(c.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]).fetchone()[0])


# ------------------------------------------------------------------ arithmetic
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"abs": abs, "min": min, "max": max, "round": round, "sum": lambda *a: sum(a), "avg": lambda *a: sum(a) / len(a),
          "sqrt": math.sqrt}


def safe_eval(expr: str, env: dict) -> float:
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.Name):
            return env[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ValueError(f"unsupported expression: {ast.dump(n)[:80]}")
    return ev(ast.parse(expr, mode="eval"))


# ------------------------------------------------------------------ text
def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def squeeze(t: str) -> str:
    t = re.sub(r"[ \t]{3,}", "  ", t)
    return re.sub(r"\n\s*\n+", "\n", t)


def page_view(text: str, limit: int = 6000) -> str:
    """Page text as shown to the answer model: column gaps kept as ' | ' so table rows stay aligned."""
    txt = re.sub(r"[ \t]{3,}", " | ", text)
    return re.sub(r"\n\s*\n+", "\n", txt)[:limit]
