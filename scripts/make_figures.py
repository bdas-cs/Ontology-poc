"""Manuscript figures (DTA submission) from the Study 1 stats and the throughput sweep.

  uv run --no-project --with matplotlib python scripts/make_figures.py

Reads ~/ontology-poc-data/<dataset>/runs-gpu/stats-strict.json (poc.eval.stats output); the sweep
numbers are the B300 'off' sweep (financebench/runs-gpu/bench-sweep-off.jsonl) and the M5 Max
smoke/sweep in the Mac study. Writes docs/figures/fig{1,2}.{png,pdf}: 600 dpi PNG and vector PDF,
light surface, readable in greyscale (marker shape + direct labels carry identity, not colour alone).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

DATA = Path.home() / "ontology-poc-data"
OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1-2 (validated, light)
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.linewidth": 0.8, "pdf.fonttype": 42,
})


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0, labelcolor=INK2)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig1():
    """Paired effects with 95% bootstrap CIs: reasoning effect and reasoning-matched ontology effect."""
    names = [("financebench", "FinanceBench (n = 150)"), ("docfinqa", "DocFinQA (n = 300)"),
             ("cuad", "CUAD (n = 300)"), ("bharat", "Bharat (n = 150)")]
    rows = []
    for key, label in names:
        s = json.load(open(DATA / key / "runs-gpu" / "stats-strict.json"))["all"]["pairs"]
        rows.append((label, s["chunk-think - chunk"], s["ontology - chunk-think"]))

    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    off = 0.17
    for i, (label, r, o) in enumerate(rows):
        y = len(rows) - 1 - i
        for d, yy, c, m in ((r, y + off, BLUE, "o"), (o, y - off, ORANGE, "s")):
            lo, hi = (v * 100 for v in d["ci95"])
            ax.plot([lo, hi], [yy, yy], color=c, linewidth=2, solid_capstyle="round", zorder=2)
            ax.plot(d["diff"] * 100, yy, marker=m, markersize=7, color=c, markeredgecolor="white",
                    markeredgewidth=1.2, zorder=3)
        p = o["mcnemar_p"]
        ax.text(21.5, y - off, f"{o['diff'] * 100:+.1f}  (p {'< 0.001' if p < 0.001 else f'= {p:.2g}'})",
                va="center", ha="left", fontsize=8, color=INK2)
    ax.axvline(0, color=AXIS, linewidth=1, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], color=INK)
    ax.set_xlim(-6, 21)
    ax.set_xlabel("Difference in strict accuracy, percentage points (95% bootstrap CI)")
    clean(ax)
    ax.spines["left"].set_visible(False)
    h1 = ax.plot([], [], marker="o", color=BLUE, linewidth=2, label="Reasoning effect (chunk RAG, reasoning on − off)")[0]
    h2 = ax.plot([], [], marker="s", color=ORANGE, linewidth=2,
                 label="Ontology effect, reasoning matched (ontology − chunk RAG with reasoning)")[0]
    ax.legend(handles=[h1, h2], loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=1, frameon=False,
              fontsize=8)
    ax.text(21.5, len(rows) - 0.55, "Ontology effect\n(McNemar p, unadjusted)", fontsize=8, color=MUTED,
            ha="left", va="bottom")
    ax.set_ylim(-0.5, len(rows) - 0.5)
    fig.subplots_adjust(left=0.24, right=0.78, top=0.88, bottom=0.33)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig1.{ext}", dpi=600, facecolor="white")
    plt.close(fig)


def fig2():
    """Aggregate decode throughput by concurrency, log-log, both platforms."""
    b300 = {1: 286.4, 2: 493.9, 4: 860.8, 8: 1544.9, 16: 2682.0, 32: 4599.2, 64: 7563.1, 128: 11261.6, 256: 13265.0}
    m5 = {1: 81, 16: 365}
    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    ax.plot(list(b300), list(b300.values()), color=BLUE, linewidth=2, marker="o", markersize=6,
            markeredgecolor="white", markeredgewidth=1.0, zorder=3)
    # two measurements only: markers, no connecting line (a line would imply the points between)
    ax.plot(list(m5), list(m5.values()), color=ORANGE, linestyle="none", marker="s", markersize=7,
            markeredgecolor="white", markeredgewidth=1.0, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(list(b300)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator([100, 300, 1000, 3000, 10000]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(50, 25000)
    ax.set_xlim(0.8, 400)
    ax.set_xlabel("Concurrent requests")
    ax.set_ylabel("Aggregate output tokens per second")
    clean(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.annotate("NVIDIA B300, vLLM, FP8", (256, 13265), xytext=(-6, 8), textcoords="offset points",
                ha="right", fontsize=8, color=INK)
    ax.annotate("13,265", (256, 13265), xytext=(0, -14), textcoords="offset points", ha="center",
                fontsize=8, color=INK2)
    ax.annotate("286", (1, 286.4), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax.annotate("Apple M5 Max, mlx-lm, 8-bit", (16, 365), xytext=(8, -4), textcoords="offset points",
                ha="left", fontsize=8, color=INK)
    ax.annotate("81", (1, 81), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax.annotate("365", (16, 365), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.16)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig2.{ext}", dpi=600, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    print("wrote", sorted(p.name for p in OUT.iterdir()))
