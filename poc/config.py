"""Central config, read from .env / environment.

POC_DATASET selects the benchmark (financebench, docfinqa, cuad, bharat). Each
dataset keeps its own db/ and runs/ under POC_DATA_DIR/<dataset>/ so results never
mix; telemetry is machine-level and lives in POC_DATA_DIR/telemetry/.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8091/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "mlx-community/Qwen3.6-35B-A3B-8bit")
EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-opus-5-5")

DATASET = os.getenv("POC_DATASET", "financebench")
DATA_DIR = Path(os.path.expanduser(os.getenv("POC_DATA_DIR", "~/ontology-poc-data")))
DATASET_DIR = DATA_DIR / DATASET
DB_DIR = DATASET_DIR / "db"
RUNS_DIR = DATASET_DIR / "runs"
TELEMETRY_DIR = DATA_DIR / "telemetry"
# Kept for the FinanceBench adapter (raw repo checkout lives here).
FINANCEBENCH_DIR = DATA_DIR / "financebench"

for d in (DB_DIR, RUNS_DIR, TELEMETRY_DIR):
    d.mkdir(parents=True, exist_ok=True)


def torch_device() -> str:
    """cuda on a GPU server, mps on Apple silicon, else cpu (override with EMBED_DEVICE)."""
    if os.getenv("EMBED_DEVICE"):
        return os.environ["EMBED_DEVICE"]
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
