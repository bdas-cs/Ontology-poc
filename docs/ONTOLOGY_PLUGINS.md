# Ontology domain plugins

The ontology-grounded arm is split into a **generic driver** and a **domain plugin**:

| Generic (do not fork per domain) | Plugin (`poc/ontology/domains/<name>.py`) |
|---|---|
| `poc/ontology/ingest.py`: step order, Page table, chunk vectors, BM25 index, `--only` smoke runs, stats | typed object tables (DDL), base objects from `doc_info`, LLM extraction + cache |
| `poc/retrieval/grounded/pipeline.py`: the 5 steps, plan LLM call, scoped hybrid search, answer LLM call, output row | entity resolution, value-free planner catalogue + planner prompt, plan -> SQL execution, evidence rendering + answer prompt |
| `poc/ontology/runtime.py`: `Usage`, `chat` (LLM with infra retries), `ro()`/`rw()` DuckDB connections, `safe_eval`, `slug`, `squeeze`, `page_view` | |

Existing plugins: `finance` (FinanceBench, DocFinQA), `contracts` (CUAD) and `gov` (Bharat; see
`poc/ontology/domains/gov.py` and the section at the end).

## Selecting the domain

`poc.ontology.domains.get()` returns the active plugin module:

1. `ONTOLOGY_DOMAIN` environment variable (override), else
2. the active dataset adapter's `ONTOLOGY_DOMAIN` attribute (`POC_DATASET` → `poc/datasets/<name>.py`), else
3. `finance`.

Register a new plugin in `DOMAINS` in `poc/ontology/domains/__init__.py` (`gov` is already there).
Each dataset keeps its own DB under `POC_DATA_DIR/<dataset>/db/ontology.duckdb`.

## The interface

A plugin is a Python module with these attributes. `get()` checks that all the required ones exist.

### Identity and storage

| Name | Kind | Contract |
|---|---|---|
| `NAME` | `str` | Domain name, written to every answer row as `domain`. |
| `DDL` | `str` | `CREATE TABLE IF NOT EXISTS ...` for the domain's typed tables. Run on every write connection. Every record needs provenance columns `doc_name` and `page` (1-based pages from `pages.parquet`). |
| `scope_id(doc_name) -> str` | function | The id of the record that owns a document, e.g. finance `"F:"+doc` (Filing.id), contracts `"CT:"+doc` (Contract.id). It is stored in the generic `page.filing_id` / `chunk.filing_id` columns, and scoped search filters on it. (The column keeps its historical name so existing FinanceBench databases stay valid.) |
| `doc_of(scope) -> str` | function | Inverse of `scope_id`. It fills `resolved.filings` in the answer row, which drives the `resolved_correct_filing` diagnostic. |

The generic tables are `page(filing_id, doc_name, page, text)` and
`chunk(id, filing_id, doc_name, page, page_end, text, emb FLOAT[1024])`, defined in `poc/ontology/schema.py`.

### Ingest

| Name | Signature | Contract |
|---|---|---|
| `step_base` | `(docs: list[str], info: dict, pg: DataFrame) -> dict` | Build entity / document objects from the system of record (`doc_info` + page text). Open your own write connection with `runtime.rw()` **after** any LLM work, so the DB lock is not held during inference. Replace rows (DELETE + INSERT); `docs` may be a subset (`--only`). Return stats; include `base_llm_prompt_tokens` and `base_llm_completion_tokens` if you call the LLM. The driver inserts Page rows itself. |
| `step_extract` | `(pg: DataFrame, only: list[str] \| None = None, cache_only: bool = False) -> dict` | LLM extraction of typed facts. Use tag `runtime.INGEST_TAG` (`"ontology-ingest"`) for the ledger. Cache every LLM result under `config.DB_DIR / "extract_cache"` keyed by document and unit (statement, window, page), so re-runs are free; `cache_only=True` means load the cache without LLM calls. With `only`, delete and rewrite only those documents' rows. Return stats including `extract_llm_calls`, `extract_llm_prompt_tokens`, `extract_llm_completion_tokens` and `extract_wall_s`. Keep prompts bounded (page windows), and run calls with `asyncio.gather`; `poc.llm` enforces `LLM_MAX_INFLIGHT`. |

The driver runs `base → extract → embed → fts`:

```
POC_DATASET=<ds> uv run python -m poc.ontology.ingest all [--only DOC ...]
```

`embed` reuses the chunk arm's LanceDB table (`ONTOLOGY_CHUNK_TABLE`, default `chunks`) when it exists, and
otherwise embeds pages locally.

### Query (the 5 steps)

`pipeline.answer(question)` calls these, in order:

1. **`async resolve(question, usage: Usage) -> dict`**
   Maps the question text, and nothing else, to records. It must return `scope: list[str]` (scope ids, most
   relevant first; it may be empty) and `entity: str | None` (a display name). Extra keys are passed through to
   `evidence_block`. Use deterministic matching (aliases, codes, dates) first and an LLM fallback second, via
   `runtime.chat(..., tag=runtime.ANSWER_TAG)`. Call `usage.add(out)` for every LLM call.
2. **`catalogue(scope, question="") -> (text, keymap)`**
   A **value-free** description of the structured records in scope: keys, labels, periods or categories, and
   counts, but never the values. `keymap` is anything `execute` needs in order to turn plan references into SQL.
   The `question` argument is optional; finance uses it to pick relevant text tables. Optional:
   `EMPTY_CATALOGUE` (text used when the catalogue is empty).
3. **`PLAN_SYS`** (str), or optionally **`plan_system(catalogue_text) -> str`**
   The planner system prompt. The generic code sends `Question: …\n\nCatalogue:\n…` and parses JSON, using a
   5000-token budget with thinking on and 900 tokens with it off. The plan schema is yours. It must include
   `search: [queries]`, which the generic step 4 uses. Optional: `default_plan(question) -> dict`, used when the
   planner output does not parse.
4. **`execute(plan, keymap) -> {"sql", "records", "results", "errors"}`**
   Compile the plan to SQL over your tables and run it with `runtime.ro()`.
   - `records` maps a name to a dict, and each dict must contain `id`, `doc` and `page` (used for `records_used`
     and `pages_used`).
   - Put any arithmetic in `results` as `{name: {"expr", "value"}}`, computed with `runtime.safe_eval`. The model
     never does the arithmetic.
   - `errors` is a list of strings: bad references, and requested items that have no records.
5. **`evidence_block(question, res, ex, passages) -> str`** and **`ANSWER_SYS`** (str)
   Render the records (with IDs and pages), the computed results, the plan issues and the scoped passages. The
   answer prompt must require citations and forbid invented values.

Optional: `SEARCH_K` (int, default 6) is the number of passages from the scoped search.

Generic output row keys: `answer`, `domain`, `resolved{company, filings}`, `plan`, `sql`, `records_used`,
`computed`, `plan_errors`, `pages_used`, `wall_s`, `prompt_tokens`, `completion_tokens` and `llm_calls`.
`run.py` adds `qid`, `arm`, `dataset`, `seed`, `resolved_correct_filing` and `resolved_primary_correct`.

### Dataset adapter side

In `poc/datasets/<name>.py`, set:

- `ONTOLOGY_DOMAIN = "<plugin name>"`.
- `JUDGE_CONTEXT = "<domain description and rubric notes>"`. The judge appends it to its generic grading
  instructions.

`doc_info()` is the plugin's system of record. Put every entity or metadata field that resolution needs into
it: names, codes, dates, types and jurisdictions.

## Reference implementations

- **finance**: `company → filing → statement → line_item` and `section`. The extract step is the LLM statement
  extraction, run on pages found by `poc/ontology/locate.py`. On text corpora it also parses every pipe table
  deterministically into `fin_table` / `table_cell`.
  - The catalogue uses statement row keys `I3`, `B12` and `C7`, plus `T<n>r<i>` for text tables, and column codes
    `c<j>`.
  - The plan is `{vars: {name: "B12.c0"}, calc: {name: expr}, search}`.
  - `execute` runs SQL on `line_item` and `table_cell`, then `safe_eval`.
- **contracts**: `contract`, `party`, `clause(category)`, `key_date(kind)`, `amount` and `governing_law`.
  - Extraction runs per contract in windows of 16k characters or fewer, with at most 12 windows per contract,
    cached per window.
  - The catalogue lists fields and per-category clause counts and pages.
  - The plan is `{fields, categories, search}`.
  - `execute` runs SQL selects by category or field.

## What a `gov` plugin (Bharat) must implement

The proposal is in `~/ontology-poc-data/bharat/gov_ontology_proposal.md`. Mapped onto the interface:

| Interface | gov |
|---|---|
| `NAME` | `"gov"` |
| `DDL` | `jurisdiction`, `authority`, `document` (doc_type, reference_no, issued_on, language, issuer_id, jurisdiction_id, fiscal_year_id), `scheme` (+ aliases, `same_as`), `fiscal_period`, plus typed fact tables such as `budget_line` / `allocation` (amount, unit ₹ lakh/crore, FY, head), `eligibility_rule`, `benefit`, `tax_rate` / `notification`. Every fact needs `doc_name`, `page` and `evidence_text`. |
| `scope_id` / `doc_of` | Scope = document: `"GD:"+doc_name` (or a scheme id if you scope by scheme, but then `doc_of` must map a scope back to doc names; the simplest option is document scopes, possibly several per question). |
| `step_base` | Jurisdiction, authority and document rows from `doc_info` (state, department, host, title, language), plus scheme and alias seeding. Use the LLM only for alias expansion, batched like finance `_company_aliases`. |
| `step_extract` | Per-page or per-window LLM extraction of typed facts, with a JSON schema per type; cache under `extract_cache/`. Normalise amounts (lakh/crore → a base unit) and keep the raw string. |
| `resolve` | Question → scheme / authority / jurisdiction / fiscal year using alias match first, then LLM → candidate document scopes (several allowed, best first). |
| `catalogue` / `PLAN_SYS` | Value-free list of the fact records in scope (type, label, FY, unit, page) with keys. The plan picks fact keys plus `calc` (reuse the finance `vars`/`calc` pattern and `runtime.safe_eval` for numeric questions) and `search` (essential for narrative questions). |
| `execute` | SQL over the fact tables, then arithmetic via `safe_eval`; records need `id`, `doc` and `page`. |
| `evidence_block` / `ANSWER_SYS` | Records with units (₹ lakh/crore), FY and jurisdiction, plus passages. The answer prompt must allow source-language quotes. |

Implemented in `poc/ontology/domains/gov.py` (pragmatic subset: `jurisdiction`, `authority`, `document`,
`scheme` with `same_as`, `benefit`, `eligibility`, `requirement`, `date_event`, `budget_line`, `tax_provision`,
`figure`). Resolution is BM25 + rules over per-document name cards (titles, English glosses, URL slug, host, state,
issuer, reference no., FY, first-page scheme / authority names), with an LLM pick among the top 12 when the rules are
not decisive. Plan keys are `S/B/E/P/D/L/X/F<n>` (or `E*` for a whole type) plus `vars` / `calc` / `search`.

Ingest knobs: `--docs FILE` / `ONTO_DOCS=FILE` restricts every step to a doc subset (any domain);
`GOV_EXTRACT_DOCS=FILE` restricts the full extraction and gives every other document a cheap first-pages "card" pass
(`GOV_CARD_PASS=0` disables it); `GOV_WINDOW_CHARS` (12000), `GOV_MAX_WINDOWS` (12). Legacy-font documents are skipped.

```
POC_DATASET=bharat uv run python -m poc.ontology.ingest base                    # all documents, no LLM
POC_DATASET=bharat uv run python -m poc.ontology.ingest extract --docs docs.txt
POC_DATASET=bharat uv run python -m poc.ontology.ingest embed --docs docs.txt
POC_DATASET=bharat uv run python -m poc.ontology.ingest fts
POC_DATASET=bharat uv run python -m poc.retrieval.grounded.run --ids Q1 Q2 Q3 --arm ontology-smoke --fresh
```

The report's resolution diagnostic counts a question's `evidence_also_in` documents (mirrored copies) as correct.

## Regression guarantee for finance

The finance plugin is the pre-plugin code moved verbatim. Additions that only trigger on markdown pipe tables,
or on an explicit "10-K for fiscal YYYY" phrase, never fire on FinanceBench. This was verified on three
FinanceBench dev questions:

- Before and after the refactor, these were identical: resolved filings, plan, SQL, records, computed values,
  pages, answer text and token counts.
- Catalogues, key maps, prompts, locator output, rule resolution, `execute` and scoped search outputs were
  byte-identical across the corpus.
