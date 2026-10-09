#!/usr/bin/env bash
# Supervised vLLM OpenAI-compatible server for a single NVIDIA GPU (Linux), the GPU
# counterpart of scripts/serve.sh: auto-restart loop + a watchdog that sends a real
# 1-token chat completion every 30 s and restarts the server after two consecutive
# failures (a wedged engine can keep answering /health and /v1/models).
#
#   make serve-gpu                       # foreground; run it inside tmux
#   MAX_NUM_SEQS=128 make serve-gpu      # every knob below can be overridden from the env
#   EXTRA_ARGS='--speculative-config {"method":"qwen3_next_mtp","num_speculative_tokens":2}' make serve-gpu
#
# Thinking: requests pass chat_template_kwargs.enable_thinking (poc/llm.py). With
# --reasoning-parser qwen3 the thinking text is returned in message.reasoning (vLLM >= 0.2x
# renamed reasoning_content -> reasoning); poc/llm.py reads either field.
set -uo pipefail

MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
PORT="${PORT:-8091}"
HOST="${HOST:-127.0.0.1}"              # loopback only; use an SSH tunnel to reach it from the Mac
LOG="${LOG:-$HOME/ontology-poc-data/server.log}"   # the telemetry collector tails this file
# ~20k-token prompts + up to 6k generated tokens (ontology extraction / thinking answers), with headroom.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
# Leave ~12 GB of the 80 GB for the in-process Qwen3-Embedding model (chunk ingest, query
# embedding) that shares the GPU. Weights ~35 GB; the rest is KV cache + CUDA graphs.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"               # throughput sweep goes up to 256 concurrent
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-16384}" # chunked-prefill budget per step (H100 default is 8192)
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-1800}"    # first start compiles + captures CUDA graphs
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$(dirname "$0")/.." || exit 1
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/ontology-poc-data/.venv}"
mkdir -p "$(dirname "$LOG")"

log() { echo "$(date '+%F %T') [serve_vllm.sh] $*" | tee -a "$LOG"; }

PID=""
stop_server() {
  [ -n "$PID" ] || return 0
  # vLLM runs the engine in child processes; signal the whole process group.
  kill -TERM -- "-$PID" 2>/dev/null
  for _ in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$PID" 2>/dev/null
  wait "$PID" 2>/dev/null
  PID=""
}
trap 'log "supervisor stopping"; stop_server; exit 0' INT TERM

probe() {  # prints the HTTP status of a 1-token, thinking-off chat completion
  curl -s -o /dev/null -w '%{http_code}' -m 300 "http://$HOST:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":1,\"chat_template_kwargs\":{\"enable_thinking\":false}}"
}

while true; do
  log "starting vllm serve $MODEL on $HOST:$PORT (max_model_len=$MAX_MODEL_LEN gpu_mem=$GPU_MEM_UTIL max_num_seqs=$MAX_NUM_SEQS)"
  # shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
  setsid uv run vllm serve "$MODEL" \
    --host "$HOST" --port "$PORT" \
    --reasoning-parser qwen3 \
    --language-model-only \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --generation-config vllm \
    $EXTRA_ARGS >> "$LOG" 2>&1 &
  PID=$!

  # Wait until the first generation succeeds (or the process dies / startup times out).
  t0=$(date +%s); up=0
  while kill -0 "$PID" 2>/dev/null; do
    if [ "$(probe)" = "200" ]; then up=1; break; fi
    if [ $(( $(date +%s) - t0 )) -ge "$STARTUP_TIMEOUT_S" ]; then
      log "no successful generation after ${STARTUP_TIMEOUT_S}s, restarting"; break
    fi
    sleep 10
  done
  [ "$up" = 1 ] && log "ready after $(( $(date +%s) - t0 ))s"

  # Watchdog: probe generation every 30 s; restart after two consecutive failures.
  fails=0
  while [ "$up" = 1 ] && kill -0 "$PID" 2>/dev/null; do
    sleep 30
    code=$(probe)
    if [ "$code" = "200" ]; then fails=0; else fails=$((fails + 1)); fi
    if [ "$fails" -ge 2 ]; then
      log "generation probe failed twice (last $code), restarting"
      break
    fi
  done
  stop_server
  log "server exited, restarting in 10 s"
  sleep 10
done
