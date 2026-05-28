"""
Optimizacion/plot_04_gap_curve.py
==================================
Discretisation gap plots from ``SweepResult.gap_curve()``.

SOURCE TABLE:  SweepResult.gap_curve()
  Columns: mean_delta_T, max_delta_T, mean_delta_K
  Index:   (indicator, K_total)

FIGURES (1):
  gap_01_deltaT_K.png  — mean_delta_T (ms) ± band up to max_delta_T vs K,
                          per indicator. Shows how quantisation error grows
                          with K and whether it could affect score.

Usage
-----
    python plot_04_gap_curve.py [--pkl PATH]
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from plot_style import PALETTE, apply_research_style

SHOW_FIGS   = False
SAVE_FIGS   = True
TABLE_LABEL = "[gap_curve()]"


def _out(plots_dir: str, name: str) -> str:
    return os.path.join(plots_dir, name)


def _save(fig: plt.Figure, path: str) -> None:
    if SAVE_FIGS:
        fig.savefig(path)
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Plot gap_01 — mean & max delta_T vs K
# ══════════════════════════════════════════════════════════════════════════════
def plot_gap01_deltaT(gap_df, plots_dir: str) -> None:
    indicators = gap_df.index.get_level_values("indicator").unique()
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, ind in enumerate(sorted(indicators)):
        sub  = gap_df.loc[ind].sort_index()
        Ks   = sub.index.values
        mean = sub["mean_delta_T"].values * 1e3   # → ms
        mx   = sub["max_delta_T"].values * 1e3
        col  = PALETTE[i % len(PALETTE)]
        ax.fill_between(Ks, mean, mx, color=col, alpha=0.20,
                        label=f"{ind.upper()} range [mean, max]")
        ax.plot(Ks, mean, "o-", color=col, linewidth=2,
                markersize=7, label=f"{ind.upper()} mean $\\Delta T$")

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("$N_{cycles,total}$  [cycles]")
    ax.set_ylabel("$\\Delta T_{total}$  [ms]  (actual − desired)")
    ax.set_title(f"{TABLE_LABEL}  Discretisation gap $\\Delta T_{{total}}$ vs $N_{{cycles,total}}$")
    ax.legend(fontsize=12)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    _save(fig, _out(plots_dir, "gap_01_deltaT_K.png"))


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

    gap_df = sweep.gap_curve()
    if gap_df.empty:
        print("gap_curve() returned empty DataFrame — nothing to plot.")
        return

    plots_dir = os.path.join(_HERE, "sweep_output", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Saving plots to:   {plots_dir}")

    plot_gap01_deltaT(gap_df, plots_dir)
    print("  [1/1] gap_01_deltaT_K.png")
    print("plot_04 done.")


if __name__ == "__main__":
    main()
