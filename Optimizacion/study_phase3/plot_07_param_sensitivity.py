"""
Optimizacion/plot_07_param_sensitivity.py
==========================================
Parameter sensitivity plots from ``SweepResult.param_sensitivity()``.

SOURCE TABLE:  SweepResult.param_sensitivity()
  Returns dict  keyed by 'step', 'N_win', 'n_accum'.
  Each value: DataFrame indexed by param value.
  Columns: n_runs, P_det_rate, mean_score, std_score, min_score,
           mean_Dtd_ms, std_Dtd_ms, min_Dtd_ms, mean_Nfa, var_ratio.

FIGURES (3):
  psens_01_step_score.png    — mean_score ± σ vs step value + min_score line
  psens_02_Nwin_score.png    — mean_score ± σ vs N_win value + min_score line
  psens_03_step_dual.png     — Dual-axis: mean_Dtd_ms (left) & mean_Nfa (right)
                                vs step value

Usage
-----
    python plot_07_param_sensitivity.py [--pkl PATH]
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
TABLE_LABEL = "[param_sensitivity()]"


def _out(plots_dir: str, name: str) -> str:
    return os.path.join(plots_dir, name)


def _save(fig: plt.Figure, path: str) -> None:
    if SAVE_FIGS:
        fig.savefig(path)
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)


def _score_band_plot(ax, df_p, param_label: str,
                     col_mean: str, title: str) -> None:
    """Shared helper: mean ± σ + min line for a sensitivity DataFrame."""
    x    = df_p.index.values.astype(float)
    mean = df_p["mean_score"].values
    std  = df_p["std_score"].fillna(0).values
    mn   = df_p["min_score"].values

    ax.fill_between(x, mean - std, mean + std,
                    color=col_mean, alpha=0.22, label="mean ± 1σ")
    ax.plot(x, mean, "o-", color=col_mean, linewidth=2.2, markersize=8,
            label="mean score")
    ax.plot(x, mn, "s--", color=PALETTE[1], linewidth=1.6, markersize=7,
            label="min score")
    ax.set_xlabel(param_label)
    ax.set_ylabel("score  [s]")
    ax.set_title(title)
    ax.legend()
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)


# ══════════════════════════════════════════════════════════════════════════════
#  Plot psens_01 — mean score ± σ vs step
# ══════════════════════════════════════════════════════════════════════════════
def plot_psens01_step(ps_dict: dict, plots_dir: str) -> None:
    if "step_cyc" not in ps_dict:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    _score_band_plot(ax, ps_dict["step_cyc"], "$\\Delta T_{step}$  [cycles]",
                     PALETTE[0], f"{TABLE_LABEL}  Score sensitivity to $\\Delta T_{{step}}$")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    _save(fig, _out(plots_dir, "psens_01_step_score.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  Plot psens_02 — mean score ± σ vs N_win
# ══════════════════════════════════════════════════════════════════════════════
def plot_psens02_nwin(ps_dict: dict, plots_dir: str) -> None:
    if "N_cyc" not in ps_dict:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    _score_band_plot(ax, ps_dict["N_cyc"], "$N_{cycles}$  [cycles]",
                     PALETTE[4], f"{TABLE_LABEL}  Score sensitivity to $N_{{cycles}}$")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    _save(fig, _out(plots_dir, "psens_02_Nwin_score.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  Plot psens_03 — Dual axis: Δt_d & N_fa vs step
# ══════════════════════════════════════════════════════════════════════════════
def plot_psens03_step_dual(ps_dict: dict, plots_dir: str) -> None:
    if "step_cyc" not in ps_dict:
        return
    df_p = ps_dict["step_cyc"]
    x    = df_p.index.values.astype(float)
    dtd  = df_p["mean_Dtd_ms"].values
    nfa  = df_p["mean_Nfa"].values

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(x, dtd, "o-", color=PALETTE[0], linewidth=2.2,
             markersize=8, label="mean $\\Delta t_d$")
    ax1.set_xlabel("$\\Delta T_{step}$  [cycles]")
    ax1.set_ylabel("mean $\\Delta t_d$  [ms]", color=PALETTE[0])
    ax1.tick_params(axis="y", labelcolor=PALETTE[0])

    ax2 = ax1.twinx()
    ax2.plot(x, nfa, "s--", color=PALETTE[1], linewidth=2.2,
             markersize=8, label="mean $N_{fa}$")
    ax2.set_ylabel("mean $N_{fa}$", color=PALETTE[1])
    ax2.tick_params(axis="y", labelcolor=PALETTE[1])

    ax1.set_title(f"{TABLE_LABEL}  $\\Delta t_d$ and $N_{{fa}}$ trade-off vs $\\Delta T_{{step}}$")
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.yaxis.grid(True, alpha=0.4)
    ax1.set_axisbelow(True)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)
    _save(fig, _out(plots_dir, "psens_03_step_dual.png"))


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

    ps_dict = sweep.param_sensitivity()
    if not ps_dict:
        print("param_sensitivity() returned empty — nothing to plot.")
        return

    plots_dir = os.path.join(_HERE, "sweep_output", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Saving plots to:   {plots_dir}")

    plot_psens01_step(ps_dict, plots_dir)
    print("  [1/3] psens_01_step_score.png")
    plot_psens02_nwin(ps_dict, plots_dir)
    print("  [2/3] psens_02_Nwin_score.png")
    plot_psens03_step_dual(ps_dict, plots_dir)
    print("  [3/3] psens_03_step_dual.png")
    print("plot_07 done.")


if __name__ == "__main__":
    main()
