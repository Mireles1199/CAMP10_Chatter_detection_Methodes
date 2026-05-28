"""
Optimizacion/plot_02_pareto.py
==============================
Pareto-front plots derived from ``SweepResult.pareto()``.

SOURCE TABLE:  SweepResult.pareto(indicator, K_total)
FIGURES (1):
  pareto_01_fronts_by_K.png  — Pareto fronts for K=5,8,10,14 on a single
                                (N_fa, Δt_d) plane; dominated region shaded.

Usage
-----
    python plot_02_pareto.py [--pkl PATH]
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
TABLE_LABEL = "[pareto()]"
K_PLOT      = [5, 8, 10, 14]   # K values to superimpose


def _out(plots_dir: str, name: str) -> str:
    return os.path.join(plots_dir, name)


def _save(fig: plt.Figure, path: str) -> None:
    if SAVE_FIGS:
        fig.savefig(path)
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Plot pareto_01 — Pareto fronts for selected K values
# ══════════════════════════════════════════════════════════════════════════════
def plot_pareto01_fronts(sweep, plots_dir: str) -> None:
    indicators = sorted(sweep.df["indicator"].unique())
    Ks_avail   = sorted(sweep.df["N_cyc_total"].unique())
    Ks_use     = [K for K in K_PLOT if K in Ks_avail]
    if not Ks_use:
        Ks_use = Ks_avail[-4:]   # fallback: last 4 available K

    n_ind = len(indicators)
    fig, axes = plt.subplots(1, n_ind, figsize=(7 * n_ind, 6), squeeze=False)

    for ax_col, ind in zip(axes[0], indicators):
        for i, K in enumerate(Ks_use):
            pf = sweep.pareto(ind, K)
            if pf.empty:
                continue
            # Step-staircase for Pareto front
            x = pf["N_fa"].values.astype(float)
            y = (pf["delta_t_d"].values * 1e3)   # ms
            ax_col.step(x, y, where="post",
                        color=PALETTE[i % len(PALETTE)],
                        linewidth=2.2, label=f"K={K}")
            ax_col.scatter(x, y,
                           color=PALETTE[i % len(PALETTE)],
                           s=55, zorder=5)

        ax_col.set_xlabel("$N_{fa}$ (false alarms)")
        ax_col.set_ylabel("$\\Delta t_d$  [ms]")
        ax_col.set_title(f"{TABLE_LABEL}  Pareto fronts — {ind.upper()}")
        ax_col.legend(title="$N_{cycles,total}$", fontsize=13)
        ax_col.yaxis.grid(True)
        ax_col.xaxis.grid(True)
        ax_col.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, _out(plots_dir, "pareto_01_fronts_by_K.png"))


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

    plots_dir = os.path.join(_HERE, "sweep_output", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Saving plots to:   {plots_dir}")

    plot_pareto01_fronts(sweep, plots_dir)
    print("  [1/1] pareto_01_fronts_by_K.png")
    print("plot_02 done.")


if __name__ == "__main__":
    main()
