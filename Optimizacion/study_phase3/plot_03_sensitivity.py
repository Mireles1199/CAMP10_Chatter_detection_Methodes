"""
Optimizacion/plot_03_sensitivity.py
====================================
Sensitivity analysis plots from ``SweepResult.sensitivity()``.

SOURCE TABLE:  SweepResult.sensitivity()
  Columns: mean_delta_t_d, std_delta_t_d, min_delta_t_d,
           mean_N_fa, P_det_rate, n_valid
  Index:   (indicator, K_total)

FIGURES (2):
  sens_01_mean_dtd_K.png   — mean Δt_d ± 1σ vs K, with min_Δt_d as lower envelope
  sens_02_mean_Nfa_K.png   — mean N_fa vs K, per indicator

Usage
-----
    python plot_03_sensitivity.py [--pkl PATH]
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
TABLE_LABEL = "[sensitivity()]"


def _out(plots_dir: str, name: str) -> str:
    return os.path.join(plots_dir, name)


def _save(fig: plt.Figure, path: str) -> None:
    if SAVE_FIGS:
        fig.savefig(path)
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Plot sens_01 — mean Δt_d ± σ vs K
# ══════════════════════════════════════════════════════════════════════════════
def plot_sens01_dtd(sens_df, plots_dir: str) -> None:
    indicators = sens_df.index.get_level_values("indicator").unique()
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, ind in enumerate(sorted(indicators)):
        sub  = sens_df.loc[ind].sort_index()
        Ks   = sub.index.values
        mean = sub["mean_delta_t_d"].values * 1e3     # → ms
        std  = sub["std_delta_t_d"].fillna(0).values * 1e3
        mn   = sub["min_delta_t_d"].values * 1e3
        col  = PALETTE[i % len(PALETTE)]
        ax.fill_between(Ks, mean - std, mean + std,
                        color=col, alpha=0.20)
        ax.plot(Ks, mean, "o-", color=col, linewidth=2,
                markersize=7, label=f"{ind.upper()} mean ± 1σ")
        ax.plot(Ks, mn, "--", color=col, linewidth=1.2,
                alpha=0.7, label=f"{ind.upper()} min")

    ax.set_xlabel("$N_{cycles,total}$  [cycles]")
    ax.set_ylabel("$\\Delta t_d$  [ms]")
    ax.set_title(f"{TABLE_LABEL}  Mean detection latency $\\Delta t_d$ vs $N_{{cycles,total}}$")
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    _save(fig, _out(plots_dir, "sens_01_mean_dtd_K.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  Plot sens_02 — mean N_fa vs K
# ══════════════════════════════════════════════════════════════════════════════
def plot_sens02_nfa(sens_df, plots_dir: str) -> None:
    indicators = sens_df.index.get_level_values("indicator").unique()
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, ind in enumerate(sorted(indicators)):
        sub = sens_df.loc[ind].sort_index()
        Ks  = sub.index.values
        nfa = sub["mean_N_fa"].values
        col = PALETTE[i % len(PALETTE)]
        ax.plot(Ks, nfa, "s-", color=col, linewidth=2,
                markersize=7, label=ind.upper())

    ax.set_xlabel("$N_{cycles,total}$  [cycles]")
    ax.set_ylabel("mean $N_{fa}$")
    ax.set_title(f"{TABLE_LABEL}  Mean false-alarm count vs $N_{{cycles,total}}$")
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    _save(fig, _out(plots_dir, "sens_02_mean_Nfa_K.png"))


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

    sens_df = sweep.sensitivity()
    if sens_df.empty:
        print("sensitivity() returned empty DataFrame — nothing to plot.")
        return

    plots_dir = os.path.join(_HERE, "sweep_output", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Saving plots to:   {plots_dir}")

    plot_sens01_dtd(sens_df, plots_dir)
    print("  [1/2] sens_01_mean_dtd_K.png")
    plot_sens02_nfa(sens_df, plots_dir)
    print("  [2/2] sens_02_mean_Nfa_K.png")
    print("plot_03 done.")


if __name__ == "__main__":
    main()
