"""
Optimizacion/plot_09_tradeoff.py
=================================
Trade-off scatter plot from ``SweepResult.tradeoff_table()``.

SOURCE TABLE:  SweepResult.tradeoff_table(param='step')
  Columns: mean_Dtd_ms, mean_Nfa, min_Dtd_ms, min_Nfa, n_runs
  Index:   (indicator, step)

FIGURES (1):
  trade_01_pareto_agregado.png  — Scatter mean_Nfa vs mean_Dtd_ms,
                                    colour = step value,
                                    marker size ∝ n_runs,
                                    per indicator (subplot columns).

Usage
-----
    python plot_09_tradeoff.py [--pkl PATH]
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
TABLE_LABEL = "[tradeoff_table()]"


def _out(plots_dir: str, name: str) -> str:
    return os.path.join(plots_dir, name)


def _save(fig: plt.Figure, path: str) -> None:
    if SAVE_FIGS:
        fig.savefig(path)
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Plot trade_01 — Scatter N_fa vs Δt_d coloured by step
# ══════════════════════════════════════════════════════════════════════════════
def plot_trade01_scatter(tt: "pd.DataFrame", plots_dir: str) -> None:
    indicators = tt.index.get_level_values("indicator").unique()
    n_ind      = len(indicators)
    fig, axes  = plt.subplots(1, n_ind, figsize=(7 * n_ind, 6), squeeze=False)

    # Global step → colour mapping
    all_steps  = sorted(tt.index.get_level_values("step_cyc").unique())
    step_col   = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(all_steps)}

    for ax, ind in zip(axes[0], sorted(indicators)):
        sub = tt.loc[ind]
        for step_val, row in sub.iterrows():
            size = max(40, row["n_runs"] * 15)
            ax.scatter(row["mean_Nfa"], row["mean_Dtd_ms"],
                       color=step_col[step_val], s=size,
                       edgecolors="white", linewidths=0.8,
                       zorder=5, label=f"step={step_val}")
            ax.annotate(f"  s={step_val}",
                        (row["mean_Nfa"], row["mean_Dtd_ms"]),
                        fontsize=11, color="#444444")

        ax.set_xlabel("mean $N_{fa}$ (false alarms)")
        ax.set_ylabel("mean $\\Delta t_d$  [ms]")
        ax.set_title(f"{TABLE_LABEL}  $\\Delta t_d$ vs $N_{{fa}}$ trade-off — {ind.upper()}")
        # Deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = h
        ax.legend(seen.values(), seen.keys(), title="$\\Delta T_{step}$", fontsize=12)
        ax.yaxis.grid(True)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, _out(plots_dir, "trade_01_pareto_agregado.png"))


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

    tt = sweep.tradeoff_table(param="step_cyc")
    if tt.empty:
        print("tradeoff_table() returned empty — nothing to plot.")
        return

    plots_dir = os.path.join(_HERE, "sweep_output", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Saving plots to:   {plots_dir}")

    plot_trade01_scatter(tt, plots_dir)
    print("  [1/1] trade_01_pareto_agregado.png")
    print("plot_09 done.")


if __name__ == "__main__":
    main()
