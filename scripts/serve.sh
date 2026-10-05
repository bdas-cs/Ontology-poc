#!/usr/bin/env bash
# Supervised mlx_lm.server: bounded prompt cache (the unbounded default exhausted
# Metal's buffer-allocation limit under concurrent long prompts) + auto-restart,
# and a health watchdog that restarts the server if generation stops working
# (after a generation-thread crash it keeps listening but answers 404).
set -uo pipefail
MODEL="${MODEL:-mlx-community/Qwen3.6-35B-A3B-8bit}"
PORT="${PORT:-8091}"
LOG="${LOG:-$HOME/ontology-poc-data/server.log}"
while true; do
  echo "$(date '+%F %T') [serve.sh] starting server" >> "$LOG"
  uv run mlx_lm.server --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --decode-concurrency 16 --prompt-concurrency 8 \
    --prompt-cache-size 16 --prompt-cache-bytes $((8 * 1024 * 1024 * 1024)) >> "$LOG" 2>&1 &
  PID=$!
  # watchdog: after warm-up, probe generation every 30 s; restart on failure
  sleep 90
  fails=0
  while kill -0 "$PID" 2>/dev/null; do
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":1,\"chat_template_kwargs\":{\"enable_thinking\":false}}")
    if [ "$code" = "200" ]; then fails=0; else fails=$((fails + 1)); fi
    if [ "$fails" -ge 2 ]; then
      echo "$(date '+%F %T') [serve.sh] health check failed twice (last $code), restarting" >> "$LOG"
      kill "$PID"; sleep 5; kill -9 "$PID" 2>/dev/null
      break
    fi
    sleep 30
  done
  wait "$PID" 2>/dev/null
  echo "$(date '+%F %T') [serve.sh] server exited ($?), restarting in 5 s" >> "$LOG"
  sleep 5
done
