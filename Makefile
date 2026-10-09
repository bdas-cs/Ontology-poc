# Keep the virtualenv outside OneDrive so it isn't synced.
export UV_PROJECT_ENVIRONMENT := $(HOME)/ontology-poc-data/.venv
DATA := $(HOME)/ontology-poc-data
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
MODEL ?= mlx-community/Qwen3.6-35B-A3B-8bit
else
MODEL ?= Qwen/Qwen3.6-35B-A3B-FP8
endif
PORT ?= 8091

# Benchmark for every pipeline target: financebench | docfinqa | cuad | bharat
#   make chunk-ingest DATASET=cuad      (POC_DATASET in the environment works too)
DATASET ?= $(or $(POC_DATASET),financebench)
export POC_DATASET := $(DATASET)

# LLM_MAX_INFLIGHT override; empty = per-target Mac default (scripts/gpu_run.sh uses 64).
INFLIGHT ?=
CHUNK_TABLE ?= chunks_c600
CHUNK_K ?= 16
# Repeat targets write runs r2..r$(REPEAT) (r1 is the main run). REPEAT_TEMP > 0 samples;
# at 0 the repeats only measure serving nondeterminism.
REPEAT ?= 3
REPEAT_TEMP ?= 0.0
SWEEP_LEVELS ?= 1,2,4,8,16,32,64,128,256
DATASETS ?= financebench docfinqa cuad bharat
# Droplet IP for pull-results.
HOST ?=

CHUNK_ANSWER = uv run python -m poc.retrieval.chunk.answer --table $(CHUNK_TABLE) --k $(CHUNK_K)

.PHONY: help setup data download model serve serve-gpu smoke sweep sweep-gpu pages judge cost \
	telemetry-start telemetry-stop telemetry telemetry-watch telemetry-report \
	ontology-ingest ontology-extract ontology-dev ontology ontology-repeat ontology-report \
	chunk-ingest chunk-dev chunk-answer chunk-thinking chunk-answer-repeat chunk-thinking-repeat \
	gpu-run pull-results

help:             ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

setup:            ## create venv + install deps (mlx-lm on macOS, vLLM on Linux x86_64)
	uv sync

data:             ## FinanceBench questions + PDFs (sparse clone, no vectorstores)
	./scripts/get_financebench.sh $(DATA)/financebench

download:         ## fetch raw data for DATASET (idempotent)
	uv run python -c 'from poc import data; data.download()'

model:            ## download model weights (Mac: MLX 8-bit ~37 GB; Linux: FP8 ~37 GB)
	uv run hf download $(MODEL)

serve:            ## mlx_lm OpenAI-compatible server on :$(PORT) (Apple silicon)
	MODEL=$(MODEL) PORT=$(PORT) ./scripts/serve.sh

serve-gpu:        ## vLLM OpenAI-compatible server on 127.0.0.1:$(PORT) (NVIDIA, Linux; run in tmux)
	MODEL=$(MODEL) PORT=$(PORT) ./scripts/serve_vllm.sh

smoke:            ## single-request tokens/sec, reasoning on vs off
	uv run python -m poc.bench.smoke

sweep:            ## concurrency sweep 1..16
	uv run python -m poc.bench.smoke --sweep 1,2,4,8,16

sweep-gpu:        ## concurrency sweep $(SWEEP_LEVELS), reasoning off then on -> runs/bench-sweep-{off,on}.jsonl
	uv run python -m poc.bench.smoke --sweep $(SWEEP_LEVELS) --tag off
	uv run python -m poc.bench.smoke --sweep $(SWEEP_LEVELS) --thinking --tag on

pages:            ## parse DATASET's corpus to db/pages.parquet
	uv run python -m poc.ingest.parse

judge:            ## judge answers: make judge ARMS="chunk ontology"
	uv run python -m poc.eval.judge $(ARMS)

cost:             ## tokens per phase: API-equivalent vs laptop vs rented-GPU cost (COST_ARGS="--runs DIR")
	uv run python -m poc.bench.cost $(COST_ARGS)

telemetry-start:  ## detached collector (macmon on Mac, nvidia-smi on GPU servers; pidfile)
	uv run python -m poc.telemetry.collector --daemon

telemetry-stop:
	uv run python -m poc.telemetry.collector --stop

telemetry:        ## aggregate -> ontology view -> explain exceptions (one pass)
	LLM_MAX_INFLIGHT=2 uv run python -m poc.telemetry.run

telemetry-watch:  ## continuous, every 60 s
	LLM_MAX_INFLIGHT=2 uv run python -m poc.telemetry.run --watch 60

telemetry-report: ## summary JSON + SQL Q&A over the view
	uv run python -m poc.telemetry.report

ontology-ingest:  ## build ontology (needs `make pages`; reuses chunk LanceDB vectors)
	LLM_MAX_INFLIGHT=$(or $(INFLIGHT),16) uv run python -m poc.ontology.ingest all

ontology-extract: ## LLM statement extraction only (skips cached statements)
	LLM_MAX_INFLIGHT=$(or $(INFLIGHT),16) uv run python -m poc.ontology.ingest extract

ontology-dev:     ## grounded pipeline on the 20-question dev slice
	uv run python -m poc.retrieval.grounded.run --dev --fresh

ontology:         ## grounded pipeline on all questions (resumes; add FRESH=--fresh to restart)
	LLM_MAX_INFLIGHT=$(or $(INFLIGHT),16) uv run python -m poc.retrieval.grounded.run $(FRESH)

# Needs `--out` / `--seed` in poc.retrieval.grounded.run (owned by the ontology work stream).
ontology-repeat:  ## ontology repeats -> answers-ontology-r2..r$(REPEAT).jsonl
	@for i in $$(seq 2 $(REPEAT)); do \
	  echo "== ontology repeat r$$i"; \
	  LLM_MAX_INFLIGHT=$(or $(INFLIGHT),16) uv run python -m poc.retrieval.grounded.run \
	    --out answers-ontology-r$$i.jsonl --seed $$i || exit 1; \
	done

ontology-report:  ## resolution accuracy, tokens, judged dev vs held-out
	uv run python -m poc.retrieval.grounded.report --judged judged-ontology.jsonl

chunk-ingest:     ## chunk (600 tok, 100 overlap) + embed into LanceDB
	uv run python -m poc.ingest.chunk --size 600 --overlap 100 --table $(CHUNK_TABLE)

chunk-dev:        ## dev-slice retrieval sweep (chunk size x k)
	uv run python -m poc.retrieval.chunk.dev --tables chunks $(CHUNK_TABLE) --ks 5 10 16 20 32

# Restarts from scratch by default; FRESH= resumes a partial run (scripts/gpu_run.sh does this).
chunk-answer: FRESH ?= --fresh
chunk-answer:     ## chunk-RAG answers for all questions (k=16, reasoning off)
	LLM_MAX_INFLIGHT=$(or $(INFLIGHT),8) $(CHUNK_ANSWER) $(FRESH)

chunk-thinking:   ## control: chunk-RAG with thinking on (matches ontology answer budget)
	LLM_MAX_INFLIGHT=$(or $(INFLIGHT),8) $(CHUNK_ANSWER) --thinking --max-tokens 4000 --out answers-chunk-think.jsonl $(FRESH)

chunk-answer-repeat:   ## repeats of chunk-answer -> answers-chunk-r2..r$(REPEAT).jsonl (resumable)
	@for i in $$(seq 2 $(REPEAT)); do \
	  echo "== chunk repeat r$$i"; \
	  LLM_MAX_INFLIGHT=$(or $(INFLIGHT),8) $(CHUNK_ANSWER) --temperature $(REPEAT_TEMP) \
	    --out answers-chunk-r$$i.jsonl || exit 1; \
	done

chunk-thinking-repeat: ## repeats of chunk-thinking -> answers-chunk-think-r2..r$(REPEAT).jsonl (resumable)
	@for i in $$(seq 2 $(REPEAT)); do \
	  echo "== chunk-thinking repeat r$$i"; \
	  LLM_MAX_INFLIGHT=$(or $(INFLIGHT),8) $(CHUNK_ANSWER) --thinking --max-tokens 4000 \
	    --temperature $(REPEAT_TEMP) --out answers-chunk-think-r$$i.jsonl || exit 1; \
	done

gpu-run:          ## GPU droplet: full pipeline for DATASETS (see docs/GPU.md)
	./scripts/gpu_run.sh $(DATASETS)

pull-results:     ## on the Mac: make pull-results HOST=<droplet-ip>
	./scripts/pull_results.sh $(HOST) $(PULL_ARGS)

qgen-estimate:    ## Bharat: estimate Claude cost of question generation
	POC_DATASET=bharat uv run python -m poc.qgen.run estimate

qgen-bharat:      ## Bharat: generate + filter questions and the spot-check pack (Claude, ~$4)
	POC_DATASET=bharat uv run python -m poc.qgen.run all
