# Keep the virtualenv outside OneDrive so it isn't synced.
export UV_PROJECT_ENVIRONMENT := $(HOME)/ontology-poc-data/.venv
DATA := $(HOME)/ontology-poc-data
MODEL ?= mlx-community/Qwen3.6-35B-A3B-8bit
PORT ?= 8091

.PHONY: setup data model serve smoke sweep

setup:            ## create venv + install deps
	uv sync

data:             ## FinanceBench questions + PDFs (sparse clone, no vectorstores)
	./scripts/get_financebench.sh $(DATA)/financebench

model:            ## download the MLX model weights (~37 GB for 8-bit)
	uv run hf download $(MODEL)

serve:            ## OpenAI-compatible server on :$(PORT)
	MODEL=$(MODEL) PORT=$(PORT) ./scripts/serve.sh

smoke:            ## single-request tokens/sec, reasoning on vs off
	uv run python -m poc.bench.smoke

sweep:            ## concurrency sweep 1..16
	uv run python -m poc.bench.smoke --sweep 1,2,4,8,16

pages:            ## parse the 84 corpus filings to db/pages.parquet
	uv run python -m poc.ingest.parse

judge:            ## judge answers: make judge ARMS="chunk ontology"
	uv run python -m poc.eval.judge $(ARMS)

cost:             ## tokens per phase, API-equivalent vs laptop cost
	uv run python -m poc.bench.cost

telemetry-start:  ## detached macmon collector (pidfile)
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
	uv run python -m poc.ontology.ingest all

ontology-extract: ## LLM statement extraction only (skips cached statements)
	LLM_MAX_INFLIGHT=16 uv run python -m poc.ontology.ingest extract

ontology-dev:     ## grounded pipeline on the 20-question dev slice
	uv run python -m poc.retrieval.grounded.run --dev --fresh

ontology:         ## grounded pipeline on all 150 questions (resumes; add FRESH=--fresh to restart)
	LLM_MAX_INFLIGHT=16 uv run python -m poc.retrieval.grounded.run $(FRESH)

ontology-report:  ## resolution accuracy, tokens, judged dev vs held-out
	uv run python -m poc.retrieval.grounded.report --judged judged-ontology.jsonl

chunk-ingest:     ## chunk (600 tok, 100 overlap) + embed into LanceDB
	uv run python -m poc.ingest.chunk --size 600 --overlap 100 --table chunks_c600

chunk-dev:        ## dev-slice retrieval sweep (chunk size x k)
	uv run python -m poc.retrieval.chunk.dev --tables chunks chunks_c600 --ks 5 10 16 20 32

chunk-answer:     ## chunk-RAG answers for all 150 (k=16)
	LLM_MAX_INFLIGHT=8 uv run python -m poc.retrieval.chunk.answer --table chunks_c600 --k 16 --fresh

chunk-thinking:   ## control: chunk-RAG with thinking on (matches ontology answer budget)
	LLM_MAX_INFLIGHT=8 uv run python -m poc.retrieval.chunk.answer --table chunks_c600 --k 16 --thinking --max-tokens 4000 --out answers-chunk-think.jsonl
