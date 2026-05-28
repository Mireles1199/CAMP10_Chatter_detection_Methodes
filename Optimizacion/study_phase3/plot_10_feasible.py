"""
Optimizacion/plot_10_feasible.py
==================================
Feasible-space quality plots from ``SweepResult.feasible_space_quality()``.

SOURCE TABLE:  SweepResult.feasible_space_quality(score_threshold=<t>)
  Columns: n_total, n_valid, n_good, pct_good, score_threshold
  Index:   (indicator, K_total)

FIGURES (2):
  feas_01_heatmap_K_thresh.png  — Heatmap K × score_threshold → pct_good (%)
                                    for thresholds [1.3, 1.4, 1.5, 2.0]
  feas_02_line_pctgood_K.png    — Line pct_good vs K for each threshold,
                                    per indicator

Usage
-----
    python plot_10_feasible.py [--pkl PATH]
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from plot_style import PALETTE, apply_research_style

SHOW_FIGS         = False
SAVE_FIGS         = True
TABLE_LABEL       = "[feasible_space_quality()]"
SCORE_THRESHOLDS  = [1.3, 1.4, 1.5, 2.0]


def _out(plots_dir: str, name: str) -> str:
    return os.path.join(plots_dir, name)


def _save(fig: plt.Figure, path: str) -> None:
    if SAVE_FIGS:
        fig.savefig(path)
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)


def _build_combined(sweep) -> pd.DataFrame:
    """Call feasible_space_quality for each threshold and concatenate."""
    frames = []
    for t in SCORE_THRESHOLDS:
        df_t = sweep.feasible_space_quality(score_threshold=t)
        frames.append(df_t)
    return pd.concat(frames)


# ══════════════════════════════════════════════════════════════════════════════
#  Plot feas_01 — Heatmap K × threshold → pct_good
# ══════════════════════════════════════════════════════════════════════════════
def plot_feas01_heatmap(combined: pd.DataFrame, plots_dir: str) -> None:
    indicators = combined.index.get_level_values("indicator").unique()
    n_ind      = len(indicators)
    fig, axes  = plt.subplots(1, n_ind, figsize=(7 * n_ind, 5), squeeze=False)

    for ax, ind in zip(axes[0], sorted(indicators)):
        rows = []
        for t in SCORE_THRESHOLDS:
            sub_t = combined[combined["score_threshold"] == t].loc[ind]
            rows.append(sub_t["pct_good"].rename(t))
        heatmap_df = pd.DataFrame(rows)  # shape: thresholds × K
        heatmap_df.index.name   = "threshold"
        heatmap_df.columns.name = "$N_{cycles,total}$"

        im = ax.imshow(heatmap_df.values, aspect="auto", origin="upper",
                       cmap="YlGn", vmin=0, vmax=100)
        cb = fig.colorbar(im, ax=ax, label="% configs with score < threshold")
        cb.ax.tick_params(labelsize=13)
        ax.set_xticks(range(len(heatmap_df.columns)))
        ax.set_xticklabels(heatmap_df.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(SCORE_THRESHOLDS)))
        ax.set_yticklabels([f"{t:.2f} s" for t in SCORE_THRESHOLDS])
        ax.set_xlabel("$N_{cycles,total}$  [cycles]")
        ax.set_ylabel("score threshold  [s]")
        ax.set_title(f"{TABLE_LABEL}  Feasible-space width — {ind.upper()}")

        # Annotate cells
        for r_idx in range(heatmap_df.shape[0]):
            for c_idx in range(heatmap_df.shape[1]):
                val = heatmap_df.iloc[r_idx, c_idx]
                ax.text(c_idx, r_idx, f"{val:.0f}%",
                        ha="center", va="center",
                        fontsize=11,
                        color="white" if val > 50 else "black")

    fig.tight_layout()
    _save(fig, _out(plots_dir, "feas_01_heatmap_K_thresh.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  Plot feas_02 — Line pct_good vs K per threshold
# ══════════════════════════════════════════════════════════════════════════════
def plot_feas02_line(combined: pd.DataFrame, plots_dir: str) -> None:
    indicators = combined.index.get_level_values("indicator").unique()
    n_ind      = len(indicators)
    fig, axes  = plt.subplots(1, n_ind, figsize=(9 * n_ind, 5), squeeze=False)

    markers = ["o", "s", "^", "D"]

    for ax, ind in zip(axes[0], sorted(indicators)):
        for i, t in enumerate(SCORE_THRESHOLDS):
            sub_t = combined[combined["score_threshold"] == t].loc[ind].sort_index()
            Ks    = sub_t.index.values
            pct   = sub_t["pct_good"].values
            ax.plot(Ks, pct, marker=markers[i % 4], color=PALETTE[i % len(PALETTE)],
                    linewidth=2, markersize=8, label=f"threshold = {t:.2f} s")

        ax.set_xlabel("$N_{cycles,total}$  [cycles]")
        ax.set_ylabel("% configs with score < threshold")
        ax.set_title(f"{TABLE_LABEL}  Feasible-space evolution — {ind.upper()}")
        ax.legend(fontsize=12)
        ax.set_ylim(-5, 105)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, _out(plots_dir, "feas_02_line_pctgood_K.png"))


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

    combined = _build_combined(sweep)

    plots_dir = os.path.join(_HERE, "sweep_output", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Saving plots to:   {plots_dir}")

    plot_feas01_heatmap(combined, plots_dir)
    print("  [1/2] feas_01_heatmap_K_thresh.png")
    plot_feas02_line(combined, plots_dir)
    print("  [2/2] feas_02_line_pctgood_K.png")
    print("plot_10 done.")


if __name__ == "__main__":
    main()
