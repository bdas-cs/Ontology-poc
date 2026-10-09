#!/usr/bin/env bash
# First ~10 minutes on a new GPU droplet: prove the stack works before the long run.
# Fails fast with a clear message; nothing here is billed beyond the droplet itself.
#   scripts/gpu_preflight.sh            (run on the droplet, from the repo root)
set -euo pipefail
PORT="${PORT:-8091}"
step() { echo; echo "== $*"; }

step "GPU and driver"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "$drv" -ge 580 ] || { echo "FAIL: driver $drv < 580 (CUDA 13 wheels need >= 580). Use the AI/ML Ready image."; exit 1; }

step "Python env sees the GPU"
uv run python - <<'PY'
import torch, vllm
assert torch.cuda.is_available(), "torch cannot see CUDA"
cap = torch.cuda.get_device_capability()
print("torch", torch.__version__, "cuda", torch.version.cuda, "device", torch.cuda.get_device_name(), "sm", cap)
print("vllm", vllm.__version__)
PY

step "Model server answers (waits up to 30 min for first load)"
curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null || {
  echo "server not running — start it in another shell: make serve-gpu"; exit 1; }
for i in $(seq 1 180); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 60 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"'"${LLM_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"'","messages":[{"role":"user","content":"Say OK"}],"max_tokens":4,"chat_template_kwargs":{"enable_thinking":false}}')
  [ "$code" = 200 ] && break; sleep 10
done
[ "$code" = 200 ] || { echo "FAIL: no successful generation after 30 min (see server.log)"; exit 1; }
echo "generation OK"

step "Reasoning on/off + short concurrency sweep"
uv run python -m poc.bench.smoke
uv run python -m poc.bench.smoke --sweep 1,8,32 --tag preflight

echo; echo "PREFLIGHT PASSED — safe to start: scripts/gpu_run.sh"
