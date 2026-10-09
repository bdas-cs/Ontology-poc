# Ontology POC — "Grounded on a laptop"

Local test-and-learn companion to *More Cognition per Dollar*. Runs on an
M5 Max (128 GB) with a self-hosted open model; Claude is used only as the judge.

**Headline question (paper §9):** does ontology-grounded retrieval recover the
8-point accuracy gap when the answering model is the self-hosted
Qwen3.6-35B-A3B instead of a hosted model?

| Retrieval | Model | Strict (paper) | Strict (this POC) |
|---|---|---|---|
| Ontology-grounded | hosted | 93 | — |
| Chunk RAG | hosted | 73 | — |
| Chunk RAG | Qwen3.6-35B (self-hosted) | 65 | **60.7** (70.7 expanded; k=32: 62.7) |
| Chunk RAG + reasoning on (control) | Qwen3.6-35B (self-hosted) | — | **71.3** (77.3 expanded) |
| Ontology-grounded (reasoning on) | Qwen3.6-35B (self-hosted) | not measured | **75.3** (84.0 expanded; held-out 130: 75.4) |

Claude judge (`claude-opus-5-5`), n=150, same 84-filing corpus for both arms.
Paired exact McNemar (strict): ontology vs chunk (reasoning off) 40/18, p=0.005; chunk reasoning on vs off 21/5, p=0.002;
**ontology vs chunk, both reasoning on 23/17, p=0.43**; metric questions only 9/1, p=0.021.
By type (strict, chunk off → chunk on → ontology): metrics 56→80→96, domain 60→66→72, novel 66→68→58.
Reading: ~10.6 of the 14.6-point gain is reasoning; structure adds ~4 overall (n.s.) and 16 on metric questions (significant).
Reproduce the control with `make chunk-thinking && make judge ARMS=chunk-think`.

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
The same pipelines run on a rented NVIDIA H100 with vLLM (`Qwen/Qwen3.6-35B-A3B-FP8`):
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
