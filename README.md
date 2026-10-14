# Ontology-grounded retrieval for a self-hosted LLM

Code for a study of where a self-hosted open model's answer accuracy comes from: ontology-grounded
retrieval versus chunk RAG, with the model's reasoning mode matched, on four document benchmarks.
The answering model is the open Qwen3.6-35B-A3B; Claude is used only as the judge (and to generate
the Bharat questions).

**Study 1 (1× NVIDIA B300, vLLM, FP8), strict accuracy, Claude judge:**

| Benchmark | Chunk RAG | Chunk RAG + reasoning | Ontology-grounded | Ontology − reasoning-matched chunk (95% CI) |
|---|---|---|---|---|
| FinanceBench (150) | 58.7 | 70.0 | **82.0** | +12.0 [+4.7, +20.0], p = 0.004 |
| DocFinQA (300) | 34.0 | 42.3 | **52.3** | +10.0 [+4.7, +15.0], p < 0.001 |
| CUAD (300) | 77.3 | 77.3 | **82.7** | +5.3 [+1.7, +9.0], p = 0.007 |
| Bharat (150) | 76.0 | 80.0 | **84.7** | +4.7 [−4.0, +13.3], p = 0.35 |

Paired exact McNemar tests and bootstrap intervals over questions (`poc.eval.stats`). Study 2 (Apple M5
Max, MLX 8-bit) replicates FinanceBench: 60.7 / 71.3 / 75.3. Full results, repeats, the chunk-size
ablation, throughput, cost and a hand audit of the judge: [docs/RESULTS_GPU.md](docs/RESULTS_GPU.md).
Figures: `scripts/make_figures.py` → [docs/figures/](docs/figures/).

## Reproduce

1. **Install and configure** (Quickstart below): `cp .env.example .env`, add `ANTHROPIC_API_KEY`
   for the judge, `make setup`.
2. **Answer** on a GPU server (see [docs/GPU.md](docs/GPU.md)): `make setup model`, `make serve-gpu`,
   then `scripts/gpu_run.sh financebench docfinqa cuad bharat` (downloads data, builds the indexes and
   ontologies, runs the three arms, repeats and the throughput sweep; resumable). On Apple silicon use
   `make serve` and the per-step make targets instead.
3. **Judge and test** (any machine with the API key):
   `make judge DATASET=<name> ARMS="chunk chunk-think ontology"`, then
   `POC_DATASET=<name> uv run python -m poc.eval.stats chunk chunk-think ontology`.
   For results copied back from a server, prefix both with `POC_RUNS_SUBDIR=runs-gpu`.
4. **Cost and figures:** `make cost`, `uv run --no-project --with matplotlib python scripts/make_figures.py`.

Bharat has no questions of its own; `make qgen-bharat` generates them with Claude (about $4). They are
not distributed with this repository because the source documents carry no blanket licence.

## Stack
- **LLM:** `mlx-community/Qwen3.6-35B-A3B-8bit` via `mlx_lm.server` (OpenAI-compatible, :8091 — 8080 is taken by Docker);
  on a Linux GPU box `Qwen/Qwen3.6-35B-A3B-FP8` via vLLM on the same port (`make serve-gpu`).
  Compare later with llama.cpp `ggml-org/Qwen3.6-35B-A3B-GGUF` Q8_0 for parallel-slot batching.
- **Embeddings:** `Qwen/Qwen3-Embedding-0.6B` (sentence-transformers on MPS).
- **Ontology store:** DuckDB (typed object tables + links table). **Vectors:** LanceDB.
- **Judge:** Claude (`JUDGE_MODEL`), strict + expanded scoring, human spot-check of ~20.
- Data, venv, DBs and run ledgers live in `~/ontology-poc-data` (outside OneDrive).

## Datasets
Select with `DATASET=<name>` on any make target (exported as `POC_DATASET`); each keeps its own
`~/ontology-poc-data/<name>/{db,runs}`. Run `make download pages DATASET=<name>` first.

| name | source | licence |
|---|---|---|
| `financebench` | [patronus-ai/financebench](https://github.com/patronus-ai/financebench) (HF [PatronusAI/financebench](https://huggingface.co/datasets/PatronusAI/financebench)) | CC BY-NC 4.0 (per HF card) |
| `docfinqa` | [kensho/DocFinQA](https://huggingface.co/datasets/kensho/DocFinQA) | MIT |
| `cuad` | [theatticusproject/cuad](https://huggingface.co/datasets/theatticusproject/cuad) | CC BY 4.0 |
| `bharat` | [ankitjh4/bharat-government-documents](https://huggingface.co/datasets/ankitjh4/bharat-government-documents) | no blanket licence ("other"): research use only, do not redistribute document text |

Raw data, parsed pages and run outputs stay outside the repo and are never committed.

## Run on a GPU
The same pipelines run on a rented NVIDIA GPU (H100 or B300) with vLLM (`Qwen/Qwen3.6-35B-A3B-FP8`):
`make serve-gpu` + `scripts/gpu_run.sh` on the server, `make pull-results HOST=<ip>` back on
the Mac (into `runs-gpu/`, never over Mac results), judging stays on the Mac.
Step-by-step DigitalOcean runbook, costs and caveats: [docs/GPU.md](docs/GPU.md).

## Phases
0. **Setup + smoke** (½ d): `make setup data model`, `make serve`, `make smoke`.
1. **Chunk-RAG baseline** (1 d): parse → chunk → embed → top-k → answer → judge. Target ≈ 65% strict.
2. **Ontology-grounded** (2–3 d): `Company`, `Filing`, `Statement`, `LineItem`, `Segment`
   with provenance to page; LLM extracts statement tables at ingest; 5-step query path
   (resolve → plan as SQL → execute → scoped text search → cited answer). Score by question type.
3. **Throughput** (1–2 d): concurrency sweep, reasoning on/off share, admission limits,
   occupancy; re-run ingestion with best settings. Energy cost (W × h) vs API list rates.
4. **Inference avoidance** (stretch): `powermetrics`/`macmon` → DuckDB rolling windows →
   ontology view (`thermal_state`, `capacity_pressure`, `anomaly`); LLM only on exceptions.

## Quickstart
```bash
cp .env.example .env      # add ANTHROPIC_API_KEY for the judge
make setup                # uv venv in ~/ontology-poc-data/.venv
make data                 # FinanceBench questions + PDFs
make model                # ~37 GB
make serve                # terminal 1
make smoke                # terminal 2
make sweep
```

## Licence
Code: [MIT](LICENSE). The licence covers this repository's code only. The datasets keep their own
licences (table above; FinanceBench is non-commercial, Bharat documents are research use only), and
no dataset content is included here.
