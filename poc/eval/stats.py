"""Paired statistics over judged arms of the active dataset (POC_DATASET, POC_RUNS_SUBDIR).

For each arm: accuracy with a percentile bootstrap 95% CI over questions. For each pair of arms:
the paired accuracy difference with a bootstrap CI (questions resampled jointly) and an exact
McNemar test on the discordant pairs. Reported on all questions and on the held-out set
(all minus the ontology dev slice in runs/ontology-dev-ids.json, when present).

  POC_RUNS_SUBDIR=runs-gpu POC_DATASET=cuad uv run python -m poc.eval.stats chunk chunk-think ontology
  ... --metric expanded --json out.json
"""
import argparse
import itertools
import json
from math import comb

import numpy as np

from poc import config

B = 10_000
SEED = 20261010


def load(arm: str, metric: str) -> dict[str, bool]:
    rows = [json.loads(l) for l in (config.RUNS_DIR / f"judged-{arm}.jsonl").open()]
    return {r["qid"]: bool(r[metric]) for r in rows}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p (binomial on the b + c discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def ci(x: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    boots = x[idx].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def analyse(arms: list[str], metric: str, ids: list[str]) -> dict:
    data = {a: load(a, metric) for a in arms}
    ids = [q for q in ids if all(q in data[a] for a in arms)]
    n = len(ids)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(B, n))
    mat = {a: np.array([data[a][q] for q in ids], dtype=float) for a in arms}
    out = {"n": n, "arms": {}, "pairs": {}}
    for a in arms:
        lo, hi = ci(mat[a], idx)
        out["arms"][a] = {"acc": mat[a].mean(), "ci95": [lo, hi]}
    for a, b in itertools.combinations(arms, 2):
        d = mat[b] - mat[a]
        lo, hi = ci(d, idx)
        nb = int(((mat[a] == 1) & (mat[b] == 0)).sum())
        nc = int(((mat[a] == 0) & (mat[b] == 1)).sum())
        out["pairs"][f"{b} - {a}"] = {"diff": d.mean(), "ci95": [lo, hi], "a_only": nb, "b_only": nc,
                                      "mcnemar_p": mcnemar_exact(nb, nc)}
    return out


def show(title: str, res: dict) -> None:
    print(f"\n== {title} (n={res['n']})")
    for a, r in res["arms"].items():
        print(f"  {a:<16} {r['acc']:6.1%}  [{r['ci95'][0]:.1%}, {r['ci95'][1]:.1%}]")
    for p, r in res["pairs"].items():
        print(f"  {p:<30} {r['diff']:+6.1%}  [{r['ci95'][0]:+.1%}, {r['ci95'][1]:+.1%}]"
              f"  discordant {r['a_only']}/{r['b_only']}  McNemar p={r['mcnemar_p']:.3g}")


def main() -> None:
    from poc.retrieval.grounded.run import DEV_IDS_PATH

    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--metric", default="strict", choices=["strict", "expanded"])
    ap.add_argument("--json", default=None, help="also write results to this path")
    a = ap.parse_args()
    all_ids = sorted(load(a.arms[0], a.metric))
    dev = set(json.loads(DEV_IDS_PATH.read_text())) if DEV_IDS_PATH.exists() else set()
    res = {"dataset": config.DATASET, "runs": str(config.RUNS_DIR), "metric": a.metric,
           "all": analyse(a.arms, a.metric, all_ids)}
    show(f"{config.DATASET} / {a.metric} / all", res["all"])
    if dev:
        res["held_out"] = analyse(a.arms, a.metric, [q for q in all_ids if q not in dev])
        show(f"{config.DATASET} / {a.metric} / held-out (minus {len(dev)} dev)", res["held_out"])
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
