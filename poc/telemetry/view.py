"""Phase 4 step 4: the ontology view -- persistent semantic fields per window.

`device_state_view` is a materialised table, one row per device per window,
with the semantic fields of paper §7:

  operating_state, thermal_state, capacity_pressure, severity,
  inference_efficiency, anomaly_type (+ anomaly_flags), likely_cause,
  explanation, confidence, explained_by

Normal windows are interpreted entirely by the deterministic rules below (no
model call). Windows whose rules raise an anomaly flag are *exceptions*; their
likely_cause / explanation / confidence come from `window_explanations`, which
is filled by explain.py (model call, carry-forward or roll-up) and persists
across rebuilds -- the interpretation is written once and read many times.
"""
from __future__ import annotations

import os
import shutil

import duckdb


def _ram_total_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return 128.0


# Host RAM of the device being interpreted (override when analysing another machine's telemetry).
RAM_TOTAL_GB = float(os.getenv("TELEMETRY_RAM_TOTAL_GB") or _ram_total_gb())
# Temperature thresholds differ by hardware: Apple GPU die vs NVIDIA data-centre GPU
# (H100 slows down around 85-90 °C). TELEMETRY_PROFILE=apple|nvidia, default from nvidia-smi presence.
PROFILE = os.getenv("TELEMETRY_PROFILE") or ("nvidia" if shutil.which("nvidia-smi") else "apple")

# Thresholds (documented so the rules are auditable).
T = dict(
    idle_gpu=0.10,          # gpu_active below this and nothing in flight => idle
    busy_gpu=0.50,          # gpu_active above this => GPU busy
    saturated_gpu=0.85,
    serving_conc=0.05,      # avg in-flight LLM requests that counts as serving
    temp_warm=75.0, temp_hot=95.0, temp_crit=102.0,   # °C, Apple GPU die avg (nvidia below)
    z_temp=3.0, temp_slope=3.0,                       # °C/min rise with z>=3 => excursion
    z_w_per_util=3.0,       # power per unit GPU activity vs baseline
    tok_drop=0.5,           # throughput < 50% of busy baseline while GPU busy
    err_burst=2,            # error lines in one window
    swap_jump_gb=0.5, ram_frac_crit=0.92,
)
if PROFILE == "nvidia":
    T.update(temp_warm=70.0, temp_hot=83.0, temp_crit=88.0)

SCHEMA = """
CREATE TABLE IF NOT EXISTS window_explanations (
  device_id VARCHAR, window_s INTEGER, ws DOUBLE, signature VARCHAR,
  likely_cause VARCHAR, explanation VARCHAR, confidence DOUBLE,
  explained_by VARCHAR,          -- model | carry_forward | rollup
  source_ws DOUBLE, source_window_s INTEGER,
  prompt_tokens INTEGER, completion_tokens INTEGER, wall_s DOUBLE,
  created_at TIMESTAMPTZ DEFAULT current_timestamp
);
"""

RULES_SQL = """
CREATE OR REPLACE TEMP TABLE _rules AS
WITH c AS (
  SELECT w.*,
    CASE
      WHEN NOT complete THEN 'partial'
      WHEN concurrency_avg >= {serving_conc} OR gen_tok_s >= 1 THEN
        CASE WHEN gpu_active_avg >= {saturated_gpu} THEN 'serving_saturated' ELSE 'serving' END
      WHEN gpu_active_avg >= {busy_gpu} THEN 'gpu_busy_non_llm'
      WHEN gpu_active_avg < {idle_gpu} THEN 'idle'
      ELSE 'light_background'
    END AS operating_state,
    CASE
      WHEN gpu_temp_max >= {temp_crit} THEN 'critical'
      WHEN gpu_temp_max >= {temp_hot} THEN 'hot'
      WHEN gpu_temp_avg >= {temp_warm} THEN 'warm'
      ELSE 'cool'
    END AS thermal_state,
    CASE
      WHEN gpu_busy_frac >= 0.95 AND (concurrency_avg >= 4 OR swap_delta_gb >= {swap_jump_gb}
           OR ram_used_max_gb >= {ram_frac_crit} * {ram_total}) THEN 'critical'
      WHEN gpu_busy_frac >= 0.80 OR concurrency_avg >= 2 THEN 'high'
      WHEN gpu_busy_frac >= 0.30 OR concurrency_avg >= 0.5 THEN 'moderate'
      ELSE 'low'
    END AS capacity_pressure,
    CASE
      WHEN gen_tok_s < 1 THEN 'n/a'
      WHEN coalesce(tok_per_joule_vs_baseline, 1) < 0.5
        OR (gpu_active_avg >= {busy_gpu} AND coalesce(tok_s_vs_baseline, 1) < {tok_drop}) THEN 'degraded'
      WHEN coalesce(tok_per_joule_vs_baseline, 1) < 0.8 THEN 'reduced'
      ELSE 'nominal'
    END AS inference_efficiency,
    list_filter([
      CASE WHEN gpu_temp_max >= {temp_crit}
             OR (z_gpu_temp >= {z_temp} AND gpu_temp_slope_c_per_min >= {temp_slope} AND baseline_n >= 3)
           THEN 'thermal_excursion' END,
      CASE WHEN z_w_per_util >= {z_w_per_util} AND baseline_n >= 3 THEN 'power_excursion' END,
      CASE WHEN gpu_active_avg >= {busy_gpu} AND concurrency_avg >= 0.5
             AND tok_s_vs_baseline < {tok_drop} THEN 'throughput_drop_gpu_busy' END,
      CASE WHEN gpu_active_avg >= {busy_gpu} AND gen_tok_s < 1 AND concurrency_avg < {serving_conc}
           THEN 'gpu_busy_no_llm_tokens' END,
      CASE WHEN log_errors >= {err_burst} THEN 'error_burst' END,
      CASE WHEN swap_delta_gb >= {swap_jump_gb} OR ram_used_max_gb >= {ram_frac_crit} * {ram_total}
           THEN 'memory_pressure' END
    ], x -> x IS NOT NULL) AS anomaly_flags
  FROM window_state w
)
SELECT c.*,
  complete AND len(anomaly_flags) > 0 AS is_exception,
  CASE WHEN complete AND len(anomaly_flags) > 0 THEN array_to_string(list_sort(anomaly_flags), '+') END AS signature,
  CASE WHEN len(anomaly_flags) > 0 THEN anomaly_flags[1] ELSE 'none' END AS anomaly_type,
  -- deterministic interpretation for normal windows (no model call)
  CASE operating_state
    WHEN 'idle' THEN 'idle: no LLM requests in flight, GPU idle'
    WHEN 'light_background' THEN 'light background activity (desktop / other processes), no LLM serving'
    WHEN 'serving' THEN 'LLM serving load from ' || coalesce(tags, '?')
    WHEN 'serving_saturated' THEN 'LLM serving load from ' || coalesce(tags, '?') || ' saturating the GPU'
    WHEN 'gpu_busy_non_llm' THEN 'GPU busy outside the LLM server'
    ELSE 'incomplete window (collector gap)'
  END AS rule_cause,
  printf('%s; GPU %.0f%% busy, %.1f tok/s at %.2f avg in-flight, SoC %.1f W, GPU %.1f°C (max %.1f, %+.2f°C/min)%s',
         operating_state, 100 * gpu_active_avg, gen_tok_s, concurrency_avg, soc_power_avg,
         gpu_temp_avg, gpu_temp_max, coalesce(gpu_temp_slope_c_per_min, 0),
         CASE WHEN tok_per_joule > 0 THEN printf(', %.2f tok/J', tok_per_joule) ELSE '' END) AS rule_explanation
FROM c;
"""

VIEW_SQL = """
CREATE OR REPLACE TABLE device_state_view AS
SELECT
  r.device_id, r.window_s, r.window_start, r.window_end, r.ws,
  r.operating_state, r.thermal_state, r.capacity_pressure,
  CASE
    WHEN r.thermal_state = 'critical' OR r.log_errors >= 10
         OR (r.is_exception AND list_contains(r.anomaly_flags, 'memory_pressure')) THEN 'critical'
    WHEN r.is_exception THEN 'warning'
    WHEN r.capacity_pressure IN ('high', 'critical') OR r.thermal_state = 'hot' THEN 'notice'
    ELSE 'info'
  END AS severity,
  r.inference_efficiency, r.anomaly_type, r.anomaly_flags, r.is_exception,
  CASE WHEN r.is_exception THEN coalesce(e.likely_cause, 'PENDING: ' || r.signature) ELSE r.rule_cause END AS likely_cause,
  CASE WHEN r.is_exception THEN coalesce(e.explanation, r.rule_explanation) ELSE r.rule_explanation END AS explanation,
  CASE WHEN r.is_exception THEN e.confidence ELSE 0.9 END AS confidence,
  CASE WHEN r.is_exception THEN coalesce(e.explained_by, 'pending') ELSE 'rule' END AS explained_by,
  -- key consolidated metrics carried on the view for querying
  r.n_samples, r.gpu_active_avg, r.gpu_busy_frac, r.soc_power_avg, r.soc_power_max, r.sys_power_avg,
  r.gpu_temp_avg, r.gpu_temp_max, r.gpu_temp_slope_c_per_min, r.fan_frac_max,
  r.gen_tok_s, r.prompt_tok_s, r.concurrency_avg, r.reqs_completed, r.per_req_tok_s, r.tok_per_joule,
  r.tok_s_vs_baseline, r.tok_per_joule_vs_baseline, r.z_gpu_temp, r.z_soc_power, r.z_w_per_util,
  r.log_errors, r.ram_used_max_gb, r.swap_max_gb, r.tags, r.baseline_n
FROM _rules r
LEFT JOIN window_explanations e
  ON e.device_id = r.device_id AND e.window_s = r.window_s AND e.ws = r.ws AND e.signature = r.signature
ORDER BY r.window_s, r.ws;
"""


def build(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(SCHEMA)
    con.execute(RULES_SQL.format(**T, ram_total=RAM_TOTAL_GB))
    con.execute(VIEW_SQL)
