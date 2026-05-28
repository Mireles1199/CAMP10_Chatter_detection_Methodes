"""
Optimizacion/plot_06_convergence.py
=====================================
Convergence analysis plots from ``SweepResult.convergence_vs_k()``.

SOURCE TABLE:  SweepResult.convergence_vs_k()
  Columns: best_score, marginal_gain, best_Dtd_ms, best_Nfa,
           n_combos, n_detected, P_det_rate, pct_good_05
  Index:   (indicator, K_total)

FIGURES (2):
  conv_01_curve_marginal.png  — best_score line (left axis) +
                                 marginal_gain colour-coded bars (right axis)
  conv_02_ndetected_K.png     — n_detected vs n_combos per K (stacked bars)

Usage
-----
    python plot_06_convergence.py [--pkl PATH]
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from plot_style import PALETTE, apply_research_style

SHOW_FIGS   = False
SAVE_FIGS   = True
TABLE_LABEL = "[convergence_vs_k()]"


def _out(plots_dir: str, name: str) -> str:
    return os.path.join(plots_dir, name)


def _save(fig: plt.Figure, path: str) -> None:
    if SAVE_FIGS:
        fig.savefig(path)
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Plot conv_01 — best_score line + marginal_gain bars (dual axis)
# ══════════════════════════════════════════════════════════════════════════════
def plot_conv01_curve_marginal(conv_df, plots_dir: str) -> None:
    indicators = conv_df.index.get_level_values("indicator").unique()
    n_ind      = len(indicators)
    fig, axes  = plt.subplots(n_ind, 1, figsize=(10, 5 * n_ind), squeeze=False)

    for row, ind in enumerate(sorted(indicators)):
        ax1 = axes[row][0]
        sub = conv_df.loc[ind].sort_index().dropna(subset=["best_score"])
        Ks  = sub.index.values

        best   = sub["best_score"].values
        marg   = sub["marginal_gain"].values   # NaN for K=min
        mean   = sub["mean_score"].values       if "mean_score" in sub.columns else None
        std    = sub["std_score"].fillna(0).values if "std_score" in sub.columns else None

        # ── left axis: best / mean score ─────────────────────────────────────
        ax1.plot(Ks, best, "o-", color=PALETTE[0], linewidth=2.5,
                 markersize=8, label="best score (min)", zorder=5)
        if mean is not None:
            ax1.fill_between(Ks,
                             np.where(np.isfinite(mean - std), mean - std, np.nan),
                             np.where(np.isfinite(mean + std), mean + std, np.nan),
                             color=PALETTE[4], alpha=0.18)
            ax1.plot(Ks, mean, "s--", color=PALETTE[4], linewidth=1.8,
                     markersize=7, label="mean score ± 1\u03c3", zorder=4)
        ax1.set_ylabel("score  [s]", color="#333333")
        ax1.tick_params(axis="y", labelcolor="#333333")

        # ── right axis: marginal gain bars ───────────────────────────────────
        ax2 = ax1.twinx()
        colors_bar = [PALETTE[2] if (m > 0 or np.isnan(m)) else PALETTE[1]
                      for m in marg]
        ax2.bar(Ks, marg, color=colors_bar, alpha=0.55, width=0.6,
                label="marginal gain")
        ax2.axhline(0, color="black", linewidth=0.7, linestyle=":")
        ax2.set_ylabel("marginal gain  [s]  (score$_{K-1}$ − score$_K$)",
                       color="#555555")
        ax2.tick_params(axis="y", labelcolor="#555555")

        ax1.set_xlabel("$N_{cycles,total}$  [cycles]")
        ax1.set_title(f"{TABLE_LABEL}  Convergence curve — {ind.upper()}")
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax1.yaxis.grid(True, alpha=0.4)
        ax1.set_axisbelow(True)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    _save(fig, _out(plots_dir, "conv_01_curve_marginal.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  Plot conv_02 — n_detected / n_combos per K (stacked bars)
# ══════════════════════════════════════════════════════════════════════════════
def plot_conv02_ndetected(conv_df, plots_dir: str) -> None:
    indicators = conv_df.index.get_level_values("indicator").unique()
    n_ind      = len(indicators)
    fig, axes  = plt.subplots(1, n_ind, figsize=(8 * n_ind, 5), squeeze=False)

    for ax, ind in zip(axes[0], sorted(indicators)):
        sub      = conv_df.loc[ind].sort_index()
        Ks       = sub.index.values
        n_det    = sub["n_detected"].values.astype(float)
        n_combo  = sub["n_combos"].values.astype(float)
        n_notdet = np.maximum(n_combo - n_det, 0)

        ax.bar(Ks, n_det, color=PALETTE[2], alpha=0.8,
               label="detected", width=0.6)
        ax.bar(Ks, n_notdet, bottom=n_det, color=PALETTE[7] if len(PALETTE) > 7 else "#aaaaaa",
               alpha=0.5, label="not detected", width=0.6)
        ax.set_xlabel("$N_{cycles,total}$  [cycles]")
        ax.set_ylabel("Count of runs")
        ax.set_title(f"{TABLE_LABEL}  Detected vs total configs — {ind.upper()}")
        ax.legend()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, _out(plots_dir, "conv_02_ndetected_K.png"))


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

    conv_df = sweep.convergence_vs_k()
    if conv_df.empty:
        print("convergence_vs_k() returned empty — nothing to plot.")
        return

    plots_dir = os.path.join(_HERE, "sweep_output", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Saving plots to:   {plots_dir}")

    plot_conv01_curve_marginal(conv_df, plots_dir)
    print("  [1/2] conv_01_curve_marginal.png")
    plot_conv02_ndetected(conv_df, plots_dir)
    print("  [2/2] conv_02_ndetected_K.png")
    print("plot_06 done.")


if __name__ == "__main__":
    main()
