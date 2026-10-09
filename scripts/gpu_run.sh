#!/usr/bin/env bash
# One-shot orchestration for the GPU droplet (see docs/GPU.md). For each dataset, in order:
#
#   download -> pages -> chunk-ingest -> ontology-ingest -> chunk-answer (reasoning off)
#   -> chunk-thinking -> ontology -> repeats (REPEAT_DATASETS only: chunk, chunk-thinking,
#   ontology x REPEAT)
#
# then, once, the throughput sweep (concurrency 1..256, reasoning off and on).
#
# Resumable: a step that finished writes runs/.gpu-done-<step> and is skipped next time; the
# answer runners also resume their own partial files, so just re-run the script after a failure.
# Each step logs to ~/ontology-poc-data/<dataset>/runs/gpu-<step>.log. Stops at the first error.
# Never runs the Claude judge (no API key on the server; judge on the Mac after pull_results.sh).
#
#   scripts/gpu_run.sh                                # all four datasets
#   scripts/gpu_run.sh financebench cuad              # a subset, in that order
#   scripts/gpu_run.sh --only pages,chunk-ingest cuad # just these steps
#   scripts/gpu_run.sh --redo chunk-answer financebench  # ignore that step's done-marker
#   scripts/gpu_run.sh --no-sweep --no-repeats
#
# Env: INFLIGHT (LLM_MAX_INFLIGHT, default 64), REPEAT (3), REPEAT_DATASETS ("financebench"),
#      NO_TELEMETRY=1 to skip starting the collector, LLM_WAIT_S (wait for the server, 1800).
set -euo pipefail

cd "$(dirname "$0")/.."
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/ontology-poc-data/.venv}"
DATA="${POC_DATA_DIR:-$HOME/ontology-poc-data}"
DATA="${DATA/#\~/$HOME}"
INFLIGHT="${INFLIGHT:-64}"
REPEAT="${REPEAT:-3}"
REPEAT_DATASETS="${REPEAT_DATASETS:-financebench}"
LLM_WAIT_S="${LLM_WAIT_S:-1800}"

STEPS=(download pages chunk-ingest ontology-ingest chunk-answer chunk-thinking ontology repeats)
LLM_STEPS=" ontology-ingest chunk-answer chunk-thinking ontology repeats sweep "

ONLY="" REDO="" SWEEP=1 REPEATS=1 DATASETS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY=",$2,"; shift 2 ;;
    --redo) REDO=",$2,"; shift 2 ;;
    --no-sweep) SWEEP=0; shift ;;
    --no-repeats) REPEATS=0; shift ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) DATASETS+=("$1"); shift ;;
  esac
done
[ ${#DATASETS[@]} -gt 0 ] || DATASETS=(financebench docfinqa cuad bharat)

say() { echo "$(date '+%F %T') [gpu_run] $*"; }

fail() {  # fail <dataset> <step> <log> <exit code>
  say "FAILED: dataset=$1 step=$2 (exit $4). Last lines of $3:"
  tail -n 30 "$3" | sed 's/^/    /'
  say "Fix the cause and re-run the same command; finished steps are skipped."
  exit 1
}

has_target() { make -n "$1" DATASET=x >/dev/null 2>&1; }

LLM_URL="" LLM_MODEL=""
wait_llm() {  # block until a 1-token chat completion succeeds
  if [ -z "$LLM_URL" ]; then
    read -r LLM_URL LLM_MODEL < <(uv run python -c 'from poc import config; print(config.LLM_BASE_URL, config.LLM_MODEL)')
  fi
  local t0 code
  t0=$(date +%s)
  while true; do
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 120 "$LLM_URL/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$LLM_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":1,\"chat_template_kwargs\":{\"enable_thinking\":false}}" || true)
    [ "$code" = "200" ] && return 0
    if [ $(( $(date +%s) - t0 )) -ge "$LLM_WAIT_S" ]; then
      say "LLM server at $LLM_URL (model $LLM_MODEL) not answering (last HTTP $code) after ${LLM_WAIT_S}s."
      say "Start it with 'make serve-gpu' (in tmux) and check ~/ontology-poc-data/server.log."
      exit 1
    fi
    say "waiting for LLM server ($LLM_URL, last HTTP $code)..."
    sleep 20
  done
}

check_complete() {  # check_complete <runs dir> <answers file>...: summary must report missing == 0
  local runs=$1; shift
  uv run python - "$runs" "$@" <<'PY'
import json, sys
from pathlib import Path
runs, bad = Path(sys.argv[1]), []
for name in sys.argv[2:]:
    p = runs / (name.removesuffix(".jsonl") + ".summary.json")
    if not p.exists():
        bad.append(f"{p.name} missing")
    elif json.loads(p.read_text()).get("missing", 0):
        bad.append(f"{name}: {json.loads(p.read_text())['missing']} questions unanswered")
if bad:
    print("incomplete: " + "; ".join(bad) + " (re-run to resume)")
    sys.exit(1)
PY
}

run_make() { make "$@" DATASET="$ds" INFLIGHT="$INFLIGHT"; }

step_cmd() {  # step_cmd <step> -- runs the step for $ds
  local runs="$DATA/$ds/runs"
  case "$1" in
    download)        run_make download ;;
    pages)           run_make pages ;;
    chunk-ingest)    run_make chunk-ingest ;;
    ontology-ingest) run_make ontology-ingest ;;
    chunk-answer)    run_make chunk-answer FRESH= && check_complete "$runs" answers-chunk.jsonl ;;
    chunk-thinking)  run_make chunk-thinking FRESH= && check_complete "$runs" answers-chunk-think.jsonl ;;
    ontology)        run_make ontology ;;
    repeats)
      local outs=() i
      for i in $(seq 2 "$REPEAT"); do outs+=("answers-chunk-r$i.jsonl" "answers-chunk-think-r$i.jsonl"); done
      run_make chunk-answer-repeat REPEAT="$REPEAT" &&
        run_make chunk-thinking-repeat REPEAT="$REPEAT" &&
        check_complete "$runs" "${outs[@]}" || return 1
      if uv run python -m poc.retrieval.grounded.run --help 2>/dev/null | grep -q -- '--seed'; then
        run_make ontology-repeat REPEAT="$REPEAT"
      else
        echo "NOTE: poc.retrieval.grounded.run has no --out/--seed yet; ontology repeats skipped." \
             "Re-run with: scripts/gpu_run.sh --only repeats --redo repeats $ds"
        return 3
      fi ;;
    sweep)           run_make sweep-gpu ;;   # -> runs/bench-sweep-off.jsonl, bench-sweep-on.jsonl
    *) echo "unknown step $1"; return 2 ;;
  esac
}

run_step() {  # run_step <step>
  local step=$1 runs="$DATA/$ds/runs"
  local marker="$runs/.gpu-done-$step" log="$runs/gpu-$step.log" rc=0
  mkdir -p "$runs"
  if [ -n "$ONLY" ] && [[ "$ONLY" != *",$step,"* ]]; then return 0; fi
  if [[ "$REDO" == *",$step,"* ]]; then rm -f "$marker"; fi
  if [ -e "$marker" ]; then say "$ds/$step: done earlier, skipping"; return 0; fi
  case "$step" in ontology-ingest|ontology)
    has_target "$step" || { say "make target '$step' is missing"; exit 1; } ;;
  esac
  [[ "$LLM_STEPS" == *" $step "* ]] && wait_llm
  say "$ds/$step: start (log $log)"
  local t0; t0=$(date +%s)
  { echo "=== $(date '+%F %T') $ds/$step (INFLIGHT=$INFLIGHT)"; step_cmd "$step"; } >> "$log" 2>&1 || rc=$?
  if [ "$rc" = 3 ]; then
    say "$ds/$step: partly done (ontology repeats skipped, see $log); not marking done"; return 0
  fi
  [ "$rc" = 0 ] || fail "$ds" "$step" "$log" "$rc"
  date '+%F %T' > "$marker"
  say "$ds/$step: ok in $(( $(date +%s) - t0 ))s"
}

say "datasets: ${DATASETS[*]}; INFLIGHT=$INFLIGHT; repeats x$REPEAT for: $REPEAT_DATASETS"
if [ "${NO_TELEMETRY:-0}" != 1 ]; then
  make telemetry-start >/dev/null && say "telemetry collector running (make telemetry-stop to end)"
fi

for ds in "${DATASETS[@]}"; do
  for step in "${STEPS[@]}"; do
    if [ "$step" = repeats ]; then
      [ "$REPEATS" = 1 ] && [[ " $REPEAT_DATASETS " == *" $ds "* ]] || continue
    fi
    run_step "$step"
  done
done

if [ "$SWEEP" = 1 ]; then
  ds="${DATASETS[0]}"  # sweep ledgers land in this dataset's runs/ (machine-level benchmark)
  run_step sweep
fi
say "all requested steps finished. Results: $DATA/<dataset>/runs/ — pull them with scripts/pull_results.sh"
