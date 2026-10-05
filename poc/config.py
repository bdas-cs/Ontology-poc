"""Central config, read from .env / environment."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8091/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "mlx-community/Qwen3.6-35B-A3B-8bit")
EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-opus-5-5")

DATA_DIR = Path(os.path.expanduser(os.getenv("POC_DATA_DIR", "~/ontology-poc-data")))
FINANCEBENCH_DIR = DATA_DIR / "financebench"
RUNS_DIR = DATA_DIR / "runs"
DB_DIR = DATA_DIR / "db"

for d in (FINANCEBENCH_DIR, RUNS_DIR, DB_DIR):
    d.mkdir(parents=True, exist_ok=True)
