# GPU study results (1× NVIDIA B300, 2026-10-10)

Run: DigitalOcean B300 Spot droplet (sm_103, driver 580.173, CUDA 13.1), vLLM 0.31.0 serving
`Qwen/Qwen3.6-35B-A3B-FP8`, temperature 0, `scripts/gpu_run.sh financebench docfinqa cuad bharat`.
Judge: `claude-opus-5-5` on the Mac (`poc.eval.judge`). Statistics: `poc.eval.stats` (percentile
bootstrap, 10,000 resamples over questions, seed 20261010; exact two-sided McNemar on paired verdicts).

Arms: **chunk** = chunk RAG (600-token chunks, k=16), reasoning off; **chunk-think** = same with
reasoning on; **ontology** = ontology-grounded pipeline (reasoning on in planning), scoped search
over the same 600-token chunk index.

## Accuracy (strict), 95% bootstrap CI

| dataset | n | chunk | chunk-think | ontology | reasoning effect (think − chunk) | ontology effect (ontology − think) |
|---|---|---|---|---|---|---|
| FinanceBench | 150 | 58.7 [50.7, 66.7] | 70.0 [62.7, 77.3] | 82.0 [75.3, 88.0] | +11.3 [+5.3, +18.0] p=0.0009 | +12.0 [+4.7, +20.0] p=0.004 |
| FinanceBench held-out | 130 | 59.2 [50.8, 67.7] | 69.2 [61.5, 76.9] | 81.5 [74.6, 87.7] | +10.0 [+3.1, +16.9] p=0.007 | +12.3 [+4.6, +20.8] p=0.007 |
| DocFinQA | 300 | 34.0 [28.7, 39.3] | 42.3 [36.7, 48.0] | 52.3 [46.7, 58.0] | +8.3 [+4.7, +12.3] p<0.001 | +10.0 [+4.7, +15.0] p<0.001 |
| CUAD | 300 | 77.3 [72.7, 82.0] | 77.3 [72.7, 82.0] | 82.7 [78.3, 87.0] | +0.0 [−2.7, +2.7] p=1 | +5.3 [+1.7, +9.0] p=0.007 |
| Bharat | 150 | 76.0 [69.3, 82.7] | 80.0 [73.3, 86.0] | 84.7 [78.7, 90.0] | +4.0 [+0.7, +8.0] p=0.07 | +4.7 [−4.0, +13.3] p=0.35 |

Holm correction over the four primary (ontology − think) tests leaves FinanceBench, DocFinQA and CUAD
significant at 0.05 (adjusted p 0.012, 0.0012, 0.014); Bharat is not.

Expanded-metric results follow the same pattern (ontology − think: FinanceBench +11.3, DocFinQA
+16.0, CUAD +5.0, Bharat +6.0 n.s.). Full numbers: `~/ontology-poc-data/<dataset>/runs-gpu/stats-*.json`.

Held-out: FinanceBench excludes the 20-question dev slice used while building the pipeline. The
finance plugin was never tuned on DocFinQA, and CUAD/Bharat had no dev-slice tuning of answers, so
those sets are held-out as a whole.

## Robustness

- **Repeats (FinanceBench, three runs each, temperature 0):** chunk 58.7 / 58.0 / 61.3; chunk-think
  70.0 / 71.3 / 72.0; ontology 82.0 / 80.0 / 80.0. Server nondeterminism moves arms by 1–3 points.
- **Chunk-size ablation (FinanceBench):** ontology with the 1,000-token / 150-overlap scoped index
  (the Mac study's setting) scores 79.3 strict; 600-token scores 82.0 (diff +2.7 [−0.7, +6.0],
  p=0.22). Even with the 1,000-token index the ontology beats chunk-think by +9.3 [+1.3, +17.3], p=0.03.
- **Cross-platform (FinanceBench):** Mac (MLX 8-bit) chunk 60.7, chunk-think 71.3, ontology 75.3
  (ontology − think +4.0 [−4.0, +12.0], p=0.43). The baselines replicate within 2 points; the
  ontology arm is 4–7 points higher on the GPU. Chunk size explains ~3 points of that; the rest is
  attributed to quantisation and the Mac server's crash/retry behaviour, not isolated.
- **Resolution:** the ontology resolver found the gold document for 300/300 CUAD questions and
  121/150 Bharat questions (101/150 from metadata alone before the card pass).

## Where the ontology helps (by question type, strict)

- FinanceBench gains Mac→GPU were all in qualitative types; GPU ontology vs chunk-think by type in
  `judged-*.jsonl`.
- Bharat: eligibility_fact 86 vs 70 (chunk-think); narrative 88 vs 86; numeric 80 vs 84.
- CUAD: parties/law 93%, IP/licence 86%, termination/assignment 69% (weakest).

## Throughput and cost

Sweep (FinanceBench prompts, 600 output tokens): 286 tok/s at concurrency 1, 4,599 at 32,
11,262 at 128, 13,265 at 256; reasoning on/off makes no difference to raw decode throughput.
Mean B300 board power while serving: ~519 W (548 W busy average in the cost basis);
5.26 output tokens/J when serving, 3.26 when saturated.

GPU busy-hours at $8.00/h vs hosted-API equivalent (paper's reference prices):

| dataset | API equivalent | B300 busy-hours |
|---|---|---|
| FinanceBench (incl. 2 repeats) | $6.39 | $3.37 |
| DocFinQA | $5.50 | $2.88 |
| CUAD | $4.13 | $2.28 |
| Bharat (14,432-doc ingest) | $21.25 | $8.40 |

Droplet time from creation (~06:00 UTC) to the end of the run (~10:10 UTC) ≈ 4.2 h ≈ $34, plus idle time until destroyed.

## Incidents

1. `ontology.ingest all` reused an httpx client across `asyncio.run()` calls → fixed (050ab2e).
2. CUAD resolver: NULL contract fields became NaN via pandas → TypeError on every question, then
   4×90 s infra retries per question (~52 min lost). Fixed and retries limited to HTTP/OS errors
   (38379ac); the CUAD ontology arm was rerun from scratch.
3. The telemetry explainer's system prompt and summary label were hard-coded for the Apple M5 Max,
   so the 18 model-written explanations on the B300 were produced with the wrong platform described.
   Avoidance counts and rule-derived fields are unaffected; the prompt now follows the detected
   backend.

## Judge audit (2026-10-14)

First author hand-audited 80 Study 1 verdicts (20 per benchmark, seeded; 10 judged correct and 10
judged wrong per benchmark, at most 4 of each per arm). Agreement with the Claude judge: 80/80 on
strict and 80/80 on expanded (exact 95% CI 95.5–100% overall; 83.2–100% per benchmark). With the 12
Study 2 verdicts audited earlier: 92/92. The audit was not blind (the judge's verdict was shown).
Answers: `~/ontology-poc-data/judge-audit/judge-audit-answers.csv` (Mac only; contains Bharat ids).
