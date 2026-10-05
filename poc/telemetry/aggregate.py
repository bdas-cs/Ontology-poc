"""Phase 4 steps 2-3: deterministic SQL aggregation -> consolidated state.

`window_state` holds one row per device per closed window, for three window
sizes (1 min, 5 min, 15 min -- scaled down from the paper's 10m/1h/8h/24h):
min/max/avg/variance of hardware metrics, temperature trend (°C/min, linear
regression), LLM serving throughput and in-flight concurrency (request
intervals from the ledgers are pro-rated into the windows they overlap),
energy per token, and a trailing baseline (previous K windows of the same
size) with change-from-baseline z-scores.

Everything here is plain SQL and idempotent: the table is rebuilt from
`samples` / `llm_requests` / `log_events`, and only *closed* windows (window
end <= last sample - SETTLE_S) are emitted. Baselines look only backwards, so a window's
row never changes once it is closed -- incremental and batch runs agree.
"""
from __future__ import annotations

import os

import duckdb

WINDOWS = {60: 15, 300: 6, 900: 3}  # window_s -> trailing baseline length K
SAMPLE_PERIOD_S = 1.0
# LLM ledger rows are written when a request *completes*, and requests can run for minutes,
# so a window is only emitted once it has been closed for SETTLE_S seconds (late rows settled).
SETTLE_S = int(os.getenv("TELEMETRY_SETTLE_S", "720"))

WINDOW_SQL = """
CREATE OR REPLACE TEMP TABLE _win AS
WITH sizes(window_s, k) AS (VALUES {sizes}),
s AS (
  SELECT sz.window_s, sz.k, floor(x.epoch / sz.window_s) * sz.window_s AS ws, x.*
  FROM samples x CROSS JOIN sizes sz
),
last AS (SELECT max(epoch) AS last_epoch FROM samples),
hw AS (
  SELECT device_id, window_s, any_value(k) AS k, ws, ws + window_s AS we,
    count(*)                       AS n_samples,
    avg(gpu_active)                AS gpu_active_avg,
    max(gpu_active)                AS gpu_active_max,
    min(gpu_active)                AS gpu_active_min,
    var_samp(gpu_active)           AS gpu_active_var,
    avg((gpu_active > 0.5)::INT)         AS gpu_busy_frac,
    avg(gpu_freq_mhz)              AS gpu_freq_avg,
    avg(gpu_power_w)               AS gpu_power_avg,
    max(gpu_power_w)               AS gpu_power_max,
    avg(all_power_w)               AS soc_power_avg,
    max(all_power_w)               AS soc_power_max,
    var_samp(all_power_w)          AS soc_power_var,
    avg(sys_power_w)               AS sys_power_avg,
    max(sys_power_w)               AS sys_power_max,
    avg(cpu_usage)                 AS cpu_usage_avg,
    avg(cpu_power_w)               AS cpu_power_avg,
    avg(gpu_temp_c)                AS gpu_temp_avg,
    min(gpu_temp_c)                AS gpu_temp_min,
    max(gpu_temp_c)                AS gpu_temp_max,
    max(cpu_temp_c)                AS cpu_temp_max,
    regr_slope(gpu_temp_c, epoch) * 60 AS gpu_temp_slope_c_per_min,
    avg(fan_rpm)                   AS fan_rpm_avg,
    max(fan_rpm / nullif(fan_max_rpm, 0)) AS fan_frac_max,
    avg(ram_used_gb)               AS ram_used_avg_gb,
    max(ram_used_gb)               AS ram_used_max_gb,
    max(swap_used_gb)              AS swap_max_gb,
    max(swap_used_gb) - min(swap_used_gb) AS swap_delta_gb,
    sum(log_posts)                 AS log_posts,
    sum(log_warnings)              AS log_warnings,
    max(cache_gb)                  AS prompt_cache_gb
  FROM s GROUP BY device_id, window_s, ws
),
ro AS (  -- pro-rate each request's [start, end] interval into the windows it overlaps
  SELECT h.device_id, h.window_s, h.ws, h.we, r.tag, r.epoch, r.wall_s, r.prompt_tokens,
         r.completion_tokens, r.finish_reason,
         least(r.epoch, h.we) - greatest(r.epoch - r.wall_s, h.ws) AS ov,
         r.epoch >= h.ws AND r.epoch < h.we AS done_here
  FROM hw h JOIN llm_requests r
    ON r.device_id = h.device_id AND r.epoch - r.wall_s < h.we AND r.epoch > h.ws
),
req AS (
  SELECT device_id, window_s, ws,
    sum(ov)                                         AS busy_req_s,
    sum(completion_tokens * ov / nullif(wall_s, 0)) AS gen_tokens,
    sum(prompt_tokens * ov / nullif(wall_s, 0))     AS prompt_tokens,
    count(*) FILTER (WHERE done_here)               AS reqs_completed,
    count(*) FILTER (WHERE done_here AND tag = 'telemetry-explain') AS explain_reqs,
    sum(completion_tokens) FILTER (WHERE done_here) / nullif(sum(wall_s) FILTER (WHERE done_here), 0) AS per_req_tok_s,
    count(*) FILTER (WHERE done_here AND finish_reason = 'length') AS reqs_truncated,
    string_agg(DISTINCT tag, ',')                   AS tags
  FROM ro GROUP BY device_id, window_s, ws
),
errs AS (  -- one count per exception (final traceback line) or ERROR/5xx line; framing lines excluded
  SELECT h.device_id, h.window_s, h.ws, count(*) AS log_errors
  FROM hw h JOIN log_events e ON e.device_id = h.device_id AND e.epoch >= h.ws AND e.epoch < h.we
  WHERE e.kind IN ('exception', 'error') AND e.line NOT LIKE 'Traceback%' AND e.line NOT LIKE 'Exception occurred%'
  GROUP BY ALL
)
SELECT hw.*,
  coalesce(errs.log_errors, 0)                       AS log_errors,
  coalesce(req.busy_req_s, 0) / hw.window_s          AS concurrency_avg,
  coalesce(req.gen_tokens, 0) / hw.window_s          AS gen_tok_s,
  coalesce(req.prompt_tokens, 0) / hw.window_s       AS prompt_tok_s,
  coalesce(req.reqs_completed, 0)                    AS reqs_completed,
  coalesce(req.explain_reqs, 0)                      AS explain_reqs,
  coalesce(req.reqs_truncated, 0)                    AS reqs_truncated,
  req.per_req_tok_s,
  req.tags,
  -- energy efficiency: generated tokens per joule of SoC energy
  coalesce(req.gen_tokens, 0) / nullif(hw.soc_power_avg * hw.window_s, 0) AS tok_per_joule,
  hw.n_samples >= 0.6 * hw.window_s / {period} AS complete
FROM hw LEFT JOIN req USING (device_id, window_s, ws) LEFT JOIN errs USING (device_id, window_s, ws), last
WHERE hw.we <= last.last_epoch - {settle};
"""

BASELINE_SQL = """
CREATE OR REPLACE TABLE window_state AS
WITH b AS (
  SELECT w.*,
    -- trailing baseline over the previous K closed windows of the same size (k differs per size,
    -- so compute with the max K and filter by row distance)
    row_number() OVER d AS rn
  FROM _win w
  WINDOW d AS (PARTITION BY device_id, window_s ORDER BY ws)
),
j AS (
  SELECT cur.device_id, cur.window_s, cur.ws,
    count(p.ws)                       AS baseline_n,
    avg(p.gpu_temp_avg)               AS bl_gpu_temp,
    stddev_samp(p.gpu_temp_avg)       AS bl_gpu_temp_sd,
    avg(p.soc_power_avg)              AS bl_soc_power,
    stddev_samp(p.soc_power_avg)      AS bl_soc_power_sd,
    avg(p.gpu_active_avg)             AS bl_gpu_active,
    -- power not explained by utilisation: W per unit of GPU activity (busy windows only)
    avg(p.soc_power_avg / p.gpu_active_avg) FILTER (WHERE p.gpu_active_avg > 0.3)        AS bl_w_per_util,
    stddev_samp(p.soc_power_avg / p.gpu_active_avg) FILTER (WHERE p.gpu_active_avg > 0.3) AS bl_w_per_util_sd,
    -- throughput baseline only from windows where the GPU was busy serving
    median(p.gen_tok_s) FILTER (WHERE p.gpu_active_avg > 0.5 AND p.gen_tok_s > 1)       AS bl_busy_tok_s,
    median(p.tok_per_joule) FILTER (WHERE p.gen_tok_s > 1)                              AS bl_tok_per_joule,
    avg(p.log_errors)                 AS bl_errors
  FROM b cur
  LEFT JOIN b p ON p.device_id = cur.device_id AND p.window_s = cur.window_s
              AND p.rn < cur.rn AND p.rn >= cur.rn - cur.k AND p.complete
  GROUP BY cur.device_id, cur.window_s, cur.ws
)
SELECT b.* EXCLUDE (rn, k),
  to_timestamp(b.ws) AS window_start, to_timestamp(b.we) AS window_end,
  j.* EXCLUDE (device_id, window_s, ws),
  (b.gpu_temp_avg - j.bl_gpu_temp) / greatest(coalesce(j.bl_gpu_temp_sd, 0), 1.0)   AS z_gpu_temp,
  (b.soc_power_avg - j.bl_soc_power) / greatest(coalesce(j.bl_soc_power_sd, 0), 2.0) AS z_soc_power,
  CASE WHEN b.gpu_active_avg > 0.3 AND j.bl_w_per_util IS NOT NULL
       THEN (b.soc_power_avg / b.gpu_active_avg - j.bl_w_per_util)
            / greatest(coalesce(j.bl_w_per_util_sd, 0), 0.1 * j.bl_w_per_util) END   AS z_w_per_util,
  b.gen_tok_s / nullif(j.bl_busy_tok_s, 0)     AS tok_s_vs_baseline,
  b.tok_per_joule / nullif(j.bl_tok_per_joule, 0) AS tok_per_joule_vs_baseline
FROM b JOIN j USING (device_id, window_s, ws)
ORDER BY window_s, ws;
"""


def build(con: duckdb.DuckDBPyConnection) -> int:
    sizes = ",".join(f"({w},{k})" for w, k in WINDOWS.items())
    con.execute(WINDOW_SQL.format(sizes=sizes, period=SAMPLE_PERIOD_S, settle=SETTLE_S))
    con.execute(BASELINE_SQL)
    return con.execute("SELECT count(*) FROM window_state").fetchone()[0]
