"""Phase 4 step 5: measure inference avoidance and answer judgment questions in SQL.

Writes runs/telemetry-summary.json with:
  collection span + sample counts, windows evaluated per size, exceptions,
  model calls vs reuse (carry_forward / rollup), % inference avoided,
  actual tokens spent vs naive baselines, and the SQL Q&A (queries + rows).

Naive baseline = every window's raw samples serialised as JSON (and, as a
thriftier lower bound, as CSV for 1-min windows only) and sent to the model (tokens/sample measured with the served model's own tokenizer) plus the
same system prompt and the same average completion length.
"""
from __future__ import annotations

import json
import time

from huggingface_hub import try_to_load_from_cache
from tokenizers import Tokenizer

from poc import config
from poc.telemetry import collector
from poc.telemetry.explain import SYSTEM

QUESTIONS = {
    "Q1 When was the GPU under high/critical capacity pressure, and why?": """
SELECT strftime(window_start, '%H:%M') AS start, window_s, capacity_pressure, operating_state,
       round(gpu_busy_frac, 2) AS gpu_busy, round(concurrency_avg, 2) AS in_flight,
       round(gen_tok_s, 1) AS tok_s, explained_by, likely_cause
FROM device_state_view
WHERE window_s = 300 AND capacity_pressure IN ('high', 'critical')
ORDER BY ws""",
    "Q2 Which windows had reduced/degraded inference efficiency -- thermal or load?": """
SELECT strftime(window_start, '%H:%M') AS start, window_s, inference_efficiency,
       round(tok_per_joule, 2) AS tok_per_j, round(tok_per_joule_vs_baseline, 2) AS vs_bl,
       thermal_state, round(gpu_temp_max, 1) AS gpu_c_max, round(concurrency_avg, 2) AS in_flight,
       CASE WHEN thermal_state IN ('hot', 'critical') AND concurrency_avg < 1 THEN 'thermal'
            WHEN thermal_state IN ('hot', 'critical') THEN 'thermal+load'
            WHEN concurrency_avg < 0.5 THEN 'low load (fixed power, few tokens)'
            ELSE 'load / contention' END AS attribution,
       likely_cause
FROM device_state_view
WHERE window_s = 60 AND inference_efficiency IN ('reduced', 'degraded')
ORDER BY ws""",
    "Q3 Was the GPU busy with work that was not LLM serving? What was it?": """
SELECT likely_cause, window_s, count(*) AS windows,
       strftime(min(window_start), '%H:%M') AS first, strftime(max(window_end), '%H:%M') AS last,
       round(avg(gpu_active_avg), 2) AS gpu_avg, round(avg(soc_power_avg), 1) AS soc_w,
       list(DISTINCT explained_by) AS how
FROM device_state_view
WHERE list_contains(anomaly_flags, 'gpu_busy_no_llm_tokens')
GROUP BY ALL ORDER BY window_s, first""",
    "Q4 How did time, power and serving efficiency split by operating state (1-min windows)?": """
SELECT operating_state, count(*) AS minutes, round(avg(soc_power_avg), 1) AS soc_w,
       round(avg(gpu_temp_avg), 1) AS gpu_c, round(avg(gen_tok_s), 1) AS tok_s,
       round(sum(gen_tok_s * window_s) / nullif(sum(soc_power_avg * window_s), 0), 3) AS tok_per_j,
       round(sum(soc_power_avg * window_s) / 3600, 2) AS soc_wh
FROM device_state_view WHERE window_s = 60
GROUP BY ALL ORDER BY minutes DESC""",
    "Q5 What were the warning/critical events and the persisted explanation for each (15-min view + model-explained 1-min)?": """
SELECT strftime(window_start, '%H:%M') AS start, window_s, severity, anomaly_type, explained_by,
       round(confidence, 2) AS conf, likely_cause, explanation
FROM device_state_view
WHERE severity IN ('warning', 'critical') AND (window_s = 900 OR explained_by = 'model')
ORDER BY ws, window_s""",
}


def tokenizer() -> Tokenizer:
    return Tokenizer.from_file(try_to_load_from_cache(config.LLM_MODEL, "tokenizer.json"))


def summarize() -> dict:
    con = collector.connect(read_only=True)
    tok = tokenizer()
    # tokens per raw sample, measured on 200 real rows serialised as JSON
    cols = [c[0] for c in con.execute("SELECT * FROM samples LIMIT 0").description]
    sample_rows = con.execute("SELECT * REPLACE (ts::VARCHAR AS ts) FROM samples USING SAMPLE 200 ROWS").fetchall()
    per_sample = sum(len(tok.encode(json.dumps(dict(zip(cols, r)), default=str)).ids) for r in sample_rows) / max(len(sample_rows), 1)
    # a thriftier naive encoding: one CSV header + rows (lower bound for "send raw samples")
    csv_text = ",".join(cols) + "\n" + "\n".join(",".join("" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))
                                                         for v in r) for r in sample_rows)
    per_sample_csv = len(tok.encode(csv_text).ids) / max(len(sample_rows), 1)
    sys_tokens = len(tok.encode(SYSTEM).ids) + 20  # + chat template overhead

    col = con.execute("""SELECT count(*), strftime(min(ts), '%Y-%m-%d %H:%M:%S%z'), strftime(max(ts), '%Y-%m-%d %H:%M:%S%z'), (max(epoch) - min(epoch)) / 60,
                                count(DISTINCT device_id) FROM samples""").fetchone()
    reqs = con.execute("SELECT tag, count(*), sum(prompt_tokens), sum(completion_tokens) FROM llm_requests "
                       "WHERE epoch >= (SELECT min(epoch) FROM samples) GROUP BY 1 ORDER BY 2 DESC").fetchall()
    per_size = con.execute("""
        SELECT window_s, count(*) AS windows, count(*) FILTER (WHERE is_exception) AS exceptions,
               count(*) FILTER (WHERE explained_by = 'model') AS model_calls,
               count(*) FILTER (WHERE explained_by = 'carry_forward') AS carry_forward,
               count(*) FILTER (WHERE explained_by = 'rollup') AS rollup,
               count(*) FILTER (WHERE explained_by = 'pending') AS pending,
               sum(n_samples) AS samples
        FROM device_state_view WHERE operating_state <> 'partial' GROUP BY 1 ORDER BY 1""").fetchall()
    actual = con.execute("""SELECT count(*), coalesce(sum(prompt_tokens),0), coalesce(sum(completion_tokens),0),
                                   coalesce(sum(wall_s),0)
                            FROM window_explanations WHERE explained_by = 'model'""").fetchone()
    n_windows = sum(r[1] for r in per_size)
    n_exc = sum(r[2] for r in per_size)
    n_model = sum(r[3] for r in per_size)
    avg_completion = actual[2] / actual[0] if actual[0] else 100
    avg_call = (actual[1] + actual[2]) / actual[0] if actual[0] else 0
    naive_by_size = {r[0]: round(r[7] * per_sample + r[1] * (sys_tokens + avg_completion)) for r in per_size}
    naive_all = sum(naive_by_size.values())
    naive_csv_1m = next((round(r[7] * per_sample_csv + r[1] * (sys_tokens + avg_completion)) for r in per_size if r[0] == 60), 0)
    actual_tokens = actual[1] + actual[2]

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": f"{collector.DEVICE_ID} ({'Apple M5 Max' if collector.BACKEND == 'macmon' else 'NVIDIA'}, 1 GPU) serving {config.LLM_MODEL}",
        "collection": {"samples": col[0], "first": str(col[1]), "last": str(col[2]),
                       "minutes": round(col[3], 1), "sample_period_s": 1.0},
        "llm_requests_during_collection": [{"tag": t, "requests": n, "prompt_tokens": int(p), "completion_tokens": int(c)}
                                           for t, n, p, c in reqs],
        "windows": [dict(zip(["window_s", "windows", "exceptions", "model_calls", "carry_forward", "rollup",
                              "pending", "samples"], r)) for r in per_size],
        "windows_evaluated": n_windows,
        "exception_windows": n_exc,
        "model_calls": n_model,
        "pct_inference_avoided": round(100 * (1 - n_model / n_windows), 1) if n_windows else None,
        "pct_exceptions_reused_without_call": round(100 * (1 - n_model / n_exc), 1) if n_exc else None,
        "tokens": {
            "actual_model_tokens": actual_tokens,
            "actual_prompt_tokens": actual[1], "actual_completion_tokens": actual[2],
            "actual_model_wall_s": round(actual[3], 1),
            "avg_tokens_per_exception_call": round(avg_call, 1),
            "measured_tokens_per_raw_sample": round(per_sample, 1),
            "naive_raw_every_window_tokens": naive_all,
            "naive_raw_by_window_s": naive_by_size,
            "naive_raw_1min_only_tokens": naive_by_size.get(60),
            "measured_tokens_per_raw_sample_csv": round(per_sample_csv, 1),
            "naive_csv_1min_only_tokens": naive_csv_1m,
            "compact_context_every_window_tokens": round(n_windows * avg_call),
            "reduction_vs_naive_csv_1min_only_x": round(naive_csv_1m / actual_tokens, 1) if actual_tokens else None,
            "reduction_vs_naive_raw_x": round(naive_all / actual_tokens, 1) if actual_tokens else None,
            "reduction_vs_naive_1min_only_x": round(naive_by_size.get(60, 0) / actual_tokens, 1) if actual_tokens else None,
        },
        "qa": {},
    }
    for q, sql in QUESTIONS.items():
        res = con.execute(sql)
        names = [d[0] for d in res.description]
        summary["qa"][q] = {"sql": " ".join(sql.split()), "columns": names,
                            "rows": [[str(v) if not isinstance(v, (int, float, type(None))) else v for v in r]
                                     for r in res.fetchall()]}
    con.close()
    return summary


def main() -> None:
    s = summarize()
    out = config.TELEMETRY_DIR / "telemetry-summary.json"
    out.write_text(json.dumps(s, indent=2, default=str))
    qa = s.pop("qa")
    print(json.dumps(s, indent=2, default=str))
    con = collector.connect(read_only=True)
    for q, sql in QUESTIONS.items():
        print(f"\n## {q}\n{sql.strip()}")
        print(con.sql(sql).fetchdf().to_string(index=False, max_colwidth=90))
    con.close()
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
