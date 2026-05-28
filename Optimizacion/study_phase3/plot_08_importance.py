"""
Optimizacion/plot_08_importance.py
====================================
Parameter importance plot from ``SweepResult.importance_ranking()``.

SOURCE TABLE:  SweepResult.importance_ranking()
  Columns: var_ratio, n_unique_values, score_range
  Index:   parameter

FIGURES (1):
  imp_01_ranking.png  — Horizontal bars of var_ratio per parameter (left axis)
                         + score_range as a secondary axis (right), sorted
                         by var_ratio descending.

Usage
-----
    python plot_08_importance.py [--pkl PATH]
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from plot_style import PALETTE, apply_research_style

SHOW_FIGS   = False
SAVE_FIGS   = True
TABLE_LABEL = "[importance_ranking()]"


def _out(plots_dir: str, name: str) -> str:
    return os.path.join(plots_dir, name)


def _save(fig: plt.Figure, path: str) -> None:
    if SAVE_FIGS:
        fig.savefig(path)
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Plot imp_01 — Importance ranking (var_ratio + score_range)
# ══════════════════════════════════════════════════════════════════════════════
def plot_imp01_ranking(rank_df, plots_dir: str) -> None:
    rank_df = rank_df.sort_values("var_ratio", ascending=True)   # ascending for barh
    params  = rank_df.index.tolist()
    vr      = rank_df["var_ratio"].values
    sr      = rank_df["score_range"].values

    y_pos = np.arange(len(params))

    # Display-name mapping for parameter axis labels
    _PARAM_DISPLAY = {
        "step_cyc": "$\\Delta T_{step}$",
        "N_cyc":    "$N_{cycles}$",
        "N_fen":    "$N_{fen}$",
    }
    display_params = [_PARAM_DISPLAY.get(p, p) for p in params]

    fig, ax1 = plt.subplots(figsize=(8, max(4, len(params) * 1.4)))

    # Left axis — var_ratio
    bars = ax1.barh(y_pos, vr * 100, color=PALETTE[0], alpha=0.80,
                    height=0.55, label="var ratio  (%)")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(display_params, fontsize=15)
    ax1.set_xlabel("Variance explained  (%)", color=PALETTE[0])
    ax1.tick_params(axis="x", labelcolor=PALETTE[0])

    # Annotate var_ratio value
    for bar, v in zip(bars, vr):
        ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{v * 100:.1f} %", va="center", fontsize=13, color=PALETTE[0])

    # Right axis — score_range
    ax2 = ax1.twiny()
    ax2.plot(sr * 1e3, y_pos, "D--", color=PALETTE[1], linewidth=1.8,
             markersize=8, label="score range  (ms)")
    ax2.set_xlabel("Score range  [ms]  (max − min group mean)", color=PALETTE[1])
    ax2.tick_params(axis="x", labelcolor=PALETTE[1])

    ax1.set_title(f"{TABLE_LABEL}  Parameter importance ranking", pad=22)
    ax1.xaxis.grid(True, alpha=0.4)
    ax1.set_axisbelow(True)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    fig.tight_layout()
    _save(fig, _out(plots_dir, "imp_01_ranking.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", default=os.path.join(_HERE, "sweep_output", "sweep_result.pkl"))
    args = ap.parse_args()

    apply_research_style()

    print(f"Loading sweep from: {args.pkl}")
    with open(args.pkl, "rb") as fh:
        sweep = pickle.load(fh)

    rank_df = sweep.importance_ranking()
    if rank_df.empty:
        print("importance_ranking() returned empty — nothing to plot.")
        return

    plots_dir = os.path.join(_HERE, "sweep_output", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Saving plots to:   {plots_dir}")

    plot_imp01_ranking(rank_df, plots_dir)
    print("  [1/1] imp_01_ranking.png")
    print("plot_08 done.")


if __name__ == "__main__":
    main()
