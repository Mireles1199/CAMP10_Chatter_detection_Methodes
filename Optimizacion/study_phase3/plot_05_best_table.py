"""
Optimizacion/plot_05_best_table.py
====================================
Plots derived from ``SweepResult.best_table()``.

SOURCE TABLE:  SweepResult.best_table()
  Columns: N_win, step, N_acc, t_d[ms], t_d_true[ms], Dt_d[ms], N_fa, score
  Index:   (indicator, K)

FIGURES (2):
  best_01_bars_score_K.png  — Horizontal bars of score per K, coloured by
                               step; annotated with Δt_d and N_fa.
  best_02_line_Dtd_K.png    — Detection latency Δt_d [ms] of the best config
                               vs K per indicator.

Usage
-----
    python plot_05_best_table.py [--pkl PATH]
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
TABLE_LABEL = "[best_table()]"


def _out(plots_dir: str, name: str) -> str:
    return os.path.join(plots_dir, name)


def _save(fig: plt.Figure, path: str) -> None:
    if SAVE_FIGS:
        fig.savefig(path)
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Plot best_01 — Horizontal bars: score per K, colour = step
# ══════════════════════════════════════════════════════════════════════════════
def plot_best01_bars(bt: "pd.DataFrame", plots_dir: str) -> None:
    import pandas as pd
    indicators = bt.index.get_level_values("indicator").unique()
    n_ind      = len(indicators)
    fig, axes  = plt.subplots(1, n_ind, figsize=(8 * n_ind, 7), squeeze=False)

    for ax, ind in zip(axes[0], sorted(indicators)):
        sub    = bt.loc[ind].dropna(subset=["score"]).sort_index()
        Ks     = sub.index.values
        scores = sub["score"].values
        steps  = sub["step_cyc"].values
        dtds   = sub["Dt_d [ms]"].values
        nfas   = sub["N_fa"].values

        unique_steps = sorted(set(steps[~pd.isna(steps)]))
        step_col     = {s: PALETTE[i % len(PALETTE)]
                        for i, s in enumerate(unique_steps)}
        colors       = [step_col.get(s, PALETTE[-1]) for s in steps]

        y_pos = np.arange(len(Ks))
        ax.barh(y_pos, scores, color=colors, edgecolor="white",
                linewidth=0.6, height=0.7)

        # Annotation
        for j, (sc, dt, nf) in enumerate(zip(scores, dtds, nfas)):
            if not np.isnan(sc):
                ax.text(sc + max(scores) * 0.01, j,
                        f"Δt={dt:.0f}ms  N_fa={int(nf)}",
                        va="center", fontsize=11, color="#333333")

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"K={k}" for k in Ks])
        ax.set_xlabel("Best score  [s]")
        ax.set_title(f"{TABLE_LABEL}  Best score per K — {ind.upper()}")
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)

        # Legend for step colours
        handles = [plt.Rectangle((0, 0), 1, 1, color=step_col[s])
                   for s in unique_steps]
        ax.legend(handles, [f"step={s}" for s in unique_steps],
                  title="$\\Delta T_{step}$", loc="lower right", fontsize=12)

    fig.tight_layout()
    _save(fig, _out(plots_dir, "best_01_bars_score_K.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  Plot best_02 — Δt_d [ms] of best config vs K
# ══════════════════════════════════════════════════════════════════════════════
def plot_best02_dtd(bt: "pd.DataFrame", plots_dir: str) -> None:
    indicators = bt.index.get_level_values("indicator").unique()
    fig, ax    = plt.subplots(figsize=(9, 5))

    for i, ind in enumerate(sorted(indicators)):
        sub = bt.loc[ind].dropna(subset=["Dt_d [ms]"]).sort_index()
        Ks  = sub.index.values
        dtd = sub["Dt_d [ms]"].values
        ax.plot(Ks, dtd, "o-", color=PALETTE[i % len(PALETTE)],
                linewidth=2, markersize=7, label=ind.upper())

    ax.set_xlabel("$N_{cycles,total}$  [cycles]")
    ax.set_ylabel("$\\Delta t_d$  [ms]  (best config)")
    ax.set_title(f"{TABLE_LABEL}  Best-config detection latency vs $N_{{cycles,total}}$")
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    _save(fig, _out(plots_dir, "best_02_line_Dtd_K.png"))


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

    bt = sweep.best_table()
    if bt.empty:
        print("best_table() returned empty — nothing to plot.")
        return

    plots_dir = os.path.join(_HERE, "sweep_output", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Saving plots to:   {plots_dir}")

    plot_best01_bars(bt, plots_dir)
    print("  [1/2] best_01_bars_score_K.png")
    plot_best02_dtd(bt, plots_dir)
    print("  [2/2] best_02_line_Dtd_K.png")
    print("plot_05 done.")


if __name__ == "__main__":
    main()
