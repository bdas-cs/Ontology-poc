# Running the benchmarks on a rented H100 (DigitalOcean)

What this gets you: the same pipelines as on the Mac (chunk RAG with reasoning off and on,
ontology-grounded, repeats, throughput sweep) for FinanceBench, DocFinQA, CUAD and Bharat. They
run on one NVIDIA H100 80 GB with vLLM serving `Qwen/Qwen3.6-35B-A3B-FP8`. Answers are judged
afterwards **on the Mac**; the server never gets an Anthropic key.

Items marked **(unverified)** could not be checked from the Mac. Check them on the droplet.

---

## A. What you do on DigitalOcean (about 15 min)

> **October 2026 availability note.** H100/H200 single-GPU plans were out of capacity; the plan
> used is **1× B300 (Spot, Liquid Cooled), 288 GB, $8.00/GPU-hour, Public Preview**, with the
> **AI/ML Ready** image. Set `GPU_HOURLY_USD=8.00` for `make cost`. Spot droplets can be
> reclaimed (DO aims for ~2 h notice, sometimes less or none), so run
> `scripts/pull_loop.sh <IP> 20` on the Mac for the whole session; `gpu_run.sh` resumes from its
> last completed step on a new droplet. Blackwell needs driver ≥ 580 and a Blackwell-capable vLLM
> build — `scripts/gpu_preflight.sh` checks both in the first minutes.

1. **Account and billing.** Sign up at <https://cloud.digitalocean.com> and add a payment method.
2. **GPU access.** New accounts often have low Droplet limits. In the control panel, open
   *Create → Droplets → GPU*. If H100 is greyed out or creation fails with a limit error, ask
   for an increase: *Settings → Profile → Droplet limit → Increase*, or open a support ticket
   saying "1× H100 GPU Droplet for ~1 day of LLM benchmarking". Approval can take hours, so
   request it early.
3. **SSH key.** On the Mac, run `cat ~/.ssh/id_ed25519.pub`. If that file doesn't exist, create
   it with `ssh-keygen -t ed25519`. Paste the key into *Settings → Security → SSH keys → Add SSH Key*.
4. **Create the droplet.** Choose *Create → Droplets* and set:
   - **Region:** one that has H100: NYC2, TOR1 or AMS3 (from DO's availability page as of Oct 2026).
   - **Type:** *GPU Droplet*, **NVIDIA H100, 1 GPU**. Size slug `gpu-h100x1-80gb` and spec of
     about 20 vCPU, 240 GiB RAM, 720 GiB boot disk and a scratch disk **(unverified)**.
   - **Image:** the **AI/ML-ready** NVIDIA image. It ships Ubuntu, the NVIDIA driver (580/590
     series) and CUDA 13, according to DO docs.
   - **Authentication:** your SSH key, not a password.
   - Give it a name such as `ontology-h100` and create it.
5. **Billing warning.** The droplet bills every hour while it exists, **even when powered
   off**, because DO keeps the GPU reserved. The only way to stop billing is to **destroy** it
   (section D).

## B. What to send back

Send **the droplet's public IPv4 address** (shown on the droplet page). Nothing else is needed.

GitHub access for the private repo: the main session will create a key on the droplet and send
you a **public** key. Add it at *github.com/bdas-cs/Ontology-poc → Settings → Deploy keys → Add
deploy key* and leave "Allow write access" **unchecked**. This read-only key works for this one
repo only. Delete it after the run. A fine-grained personal access token with *Contents:
read-only* on this repo also works. Don't paste tokens into chat. If you use one, type it on the
droplet yourself.

## C. What the main session runs over SSH

```bash
ssh root@<IP>                      # AI/ML image logs in as root (unverified)
nvidia-smi                         # expect 1x H100 80GB; driver >= 580 is needed for the CUDA 13 wheels
apt-get update && apt-get install -y git tmux rsync build-essential
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env

# repo via read-only deploy key
ssh-keygen -t ed25519 -N '' -f ~/.ssh/ontology_deploy && cat ~/.ssh/ontology_deploy.pub   # -> user adds it
cat >> ~/.ssh/config <<'EOF'
Host github.com
  IdentityFile ~/.ssh/ontology_deploy
  StrictHostKeyChecking accept-new
EOF
git clone git@github.com:bdas-cs/Ontology-poc.git && cd Ontology-poc
# from the Mac: Bharat questions are kept out of git (no blanket licence on the source text)
#   scripts/push_private_data.sh <IP>

cat > .env <<'EOF'
LLM_BASE_URL=http://127.0.0.1:8091/v1
LLM_MODEL=Qwen/Qwen3.6-35B-A3B-FP8
EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B
POC_DATA_DIR=~/ontology-poc-data
EOF
# no ANTHROPIC_API_KEY on the server; add HF_TOKEN=... only if a dataset turns out to be gated

make setup                         # uv sync: installs vLLM + CUDA torch (Linux x86_64 marker), ~5-10 min
make model                         # hf download Qwen/Qwen3.6-35B-A3B-FP8 (~37.5 GB)

tmux new -d -s llm 'make serve-gpu'          # supervised vLLM on 127.0.0.1:8091, log ~/ontology-poc-data/server.log
tail -f ~/ontology-poc-data/server.log       # wait for "[serve_vllm.sh] ready" (first start: compile + CUDA graphs, ~5-15 min)
scripts/gpu_preflight.sh                     # ~10 min go/no-go: driver, torch/vLLM see the GPU, generation, short sweep

tmux new -d -s run 'scripts/gpu_run.sh financebench docfinqa cuad bharat 2>&1 | tee ~/ontology-poc-data/gpu_run.log'
```

`scripts/gpu_run.sh` starts the telemetry collector (`nvidia-smi` backend, 1 s samples). It
then runs these steps for each dataset:
`download → pages → chunk-ingest → ontology-ingest → chunk-answer → chunk-thinking → ontology`.
After that come the FinanceBench repeats (r2, r3 for chunk, chunk-thinking and ontology). Last
is `make sweep-gpu`, which runs concurrency 1…256 with reasoning off and on. Each step logs to
`~/ontology-poc-data/<dataset>/runs/gpu-<step>.log` and writes a `.gpu-done-<step>` marker. On
an error, the script prints the log tail and stops. Running the same command again resumes from
the failed step. Useful flags: `--only`, `--redo`, `--no-sweep` and `--no-repeats`. Concurrency
is `INFLIGHT=64` by default.

vLLM settings (`scripts/serve_vllm.sh`; each can be overridden from the environment):

| flag | value | why |
|---|---|---|
| `--reasoning-parser qwen3` | | thinking is returned in `message.reasoning`, the answer in `content`; `enable_thinking=false` gives content only |
| `--language-model-only` | | text only; skips the vision encoder |
| `--max-model-len` | 40960 | prompts up to ~20k tokens plus up to 6k generated, with headroom |
| `--gpu-memory-utilization` | 0.85 | leaves ~12 GB for the in-process embedding model |
| `--max-num-seqs` | 256 | the sweep tops out at 256 concurrent requests |
| `--max-num-batched-tokens` | 16384 | larger prefill chunks for long RAG prompts |
| `--generation-config vllm` | | neutral sampling defaults (top_p 1, no top_k), like `mlx_lm.server` |

Before pulling results, on the droplet:

```bash
make telemetry-stop
make telemetry && make telemetry-report    # windows + view + summary, uses the local model for exceptions
```

## D. Pull results back, then destroy the droplet

On the Mac:

```bash
make pull-results HOST=<IP>        # or: scripts/pull_results.sh <IP> [--tag gpu] [--with-lancedb]
```

Files land in their own folders, so Mac results are never overwritten:
`~/ontology-poc-data/<dataset>/runs-gpu/`, `<dataset>/db-gpu/` (pages, ontology.duckdb;
LanceDB only with `--with-lancedb`) and `~/ontology-poc-data/telemetry-gpu/`. Then:

```bash
make cost COST_ARGS="--runs ~/ontology-poc-data/financebench/runs-gpu --telemetry ~/ontology-poc-data/telemetry-gpu/telemetry.duckdb"
```

Judge GPU answers on the Mac by pointing the judge at the pulled folder:

```bash
POC_RUNS_SUBDIR=runs-gpu make judge DATASET=financebench ARMS="chunk chunk-think ontology"
```

**Destroy the droplet:** open the droplet page, choose *Destroy → Destroy this Droplet*, then
confirm it is gone from the Droplets list. Also remove the GitHub deploy key.

## E. Time and cost estimate

Rate: **$4.41/h** on demand for 1× H100, per DO's pricing page on 2026-10-09 (the repo default
for `GPU_HOURLY_USD`). Set `GPU_HOURLY_USD` to your actual rate for `make cost` if it differs.

| phase | estimate |
|---|---|
| setup (packages, uv sync, 37 GB model, first vLLM start) | 0.5–1 h |
| FinanceBench (all steps + 2 repeats of 3 arms) | ~1 h |
| DocFinQA, CUAD, Bharat | 1–2 h each; depends on question count and corpus size, not known yet |
| throughput sweep (2 × 9 levels) | ~0.5 h |
| slack for failures / reruns | 1–2 h |
| **total** | **~6–10 h ≈ $26–45** |

These are estimates and haven't been measured. On the Mac, FinanceBench chunk prompts are
~10k tokens each. The H100's parallel prefill and decode should make each 150-question arm take
minutes rather than an hour. The ontology arm runs 8 questions at a time, and that limits its
speed (see `QUESTION_CONCURRENCY` in `poc/retrieval/grounded/run.py`).

## Unverified / risks

- vLLM, CUDA and the server flags were checked against vLLM 0.31.0 source (`arg_utils.py`,
  reasoning parser registry) and the model card. They have **not** been run: vLLM cannot run on
  the Mac.
- The lock pins torch 2.13.0 with CUDA 13.0 wheels on Linux, which needs NVIDIA driver ≥ 580.
  Check `nvidia-smi` before `make setup`.
- KV-cache headroom at 0.85: weights are ~35 GB, leaving roughly 25–30 GB of KV at ~20 KiB per
  token (10 full-attention layers) plus per-sequence linear-attention state. That covers about
  100 concurrent 12k-token RAG prompts. Past that, vLLM queues requests. If startup fails with
  an OOM, lower `MAX_NUM_SEQS` or `GPU_MEM_UTIL`.
- The model is FP8 on the GPU and 8-bit MLX on the Mac. The weights are close but not
  identical, so expect small accuracy differences between machines.
