"""
Optimizacion/plot_01_df_raw.py
==============================
Plots derived directly from ``sweep.df`` (raw DataFrame — 1 row per run).

SOURCE TABLE:  SweepResult.df
FIGURES (9):
  df_01_boxplot_score_K.png     — Score distribution per K (boxplot)
  df_02_scatter_td_K.png        — Detection time vs K coloured by step
  df_03_heatmap_K_step.png      — Heatmap K × step → min(score)
  df_04_scatter_score_lb.png    — score vs score_lb (lower-bound tightness)
  df_05_heatmap_Nwin_step.png   — Heatmap N_win × step → mean(score)
  df_06_hist_deltaT.png         — Histogram of delta_T_total_vs_des by K
  df_07_violin_Nfa_K.png        — Violin plot of N_fa per K
  df_08_scatter_overlap_score.png — overlap_frac vs mean score (aggregated)
  df_09_line_Pdet_K.png         — P_det rate vs K (fraction that detects)

Usage
-----
    python plot_01_df_raw.py [--pkl PATH]
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

# ── Resolve paths ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from plot_style import PALETTE, apply_research_style

# ── Constants ─────────────────────────────────────────────────────────────────
T_GT        = 5.365770208787228   # ground-truth chatter onset [s]
SHOW_FIGS   = False
SAVE_FIGS   = True
TABLE_LABEL = "[sweep.df]"        # prefix in every figure title


def _out(plots_dir: str, name: str) -> str:
    return os.path.join(plots_dir, name)


def _save(fig: plt.Figure, path: str) -> None:
    if SAVE_FIGS:
        fig.savefig(path)
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Plot df_01 — Boxplot of score distribution per K
# ══════════════════════════════════════════════════════════════════════════════
def plot_df01_boxplot_score(df: pd.DataFrame, plots_dir: str) -> None:
    valid = df[df["run_ok"] & df["score"].notna() & (df["score"] >= 0)]
    Ks    = sorted(valid["N_cyc_total"].unique())
    data  = [valid[valid["N_cyc_total"] == K]["score"].values for K in Ks]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color=PALETTE[1], linewidth=2))
    for patch in bp["boxes"]:
        patch.set_facecolor(PALETTE[0])
        patch.set_alpha(0.6)
    ax.set_xticks(range(1, len(Ks) + 1))
    ax.set_xticklabels(Ks)
    ax.set_xlabel("$N_{cycles,total}$  [cycles]")
    ax.set_ylabel("Score  $\\delta t_d + \\lambda N_{fa} T_u$  [s]")
    ax.set_title(f"{TABLE_LABEL}  Score distribution per $N_{{cycles,total}}$")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    _save(fig, _out(plots_dir, "df_01_boxplot_score_K.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  Plot df_02 — Scatter t_d_first_true vs K, coloured by step
# ══════════════════════════════════════════════════════════════════════════════
def plot_df02_scatter_td(df: pd.DataFrame, plots_dir: str) -> None:
    valid = df[df["run_ok"] & df["P_det"].astype(bool) & df["t_d_first_true"].notna()]
    steps = sorted(valid["step_cyc"].unique())
    step_colors = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(steps)}

    fig, ax = plt.subplots(figsize=(10, 5))
    for step in steps:
        sub = valid[valid["step_cyc"] == step]
        ax.scatter(sub["N_cyc_total"] + np.random.uniform(-0.15, 0.15, len(sub)),
                   sub["t_d_first_true"],
                   color=step_colors[step], alpha=0.55, s=40,
                   label=f"step={step}")
    ax.axhline(T_GT, color=PALETTE[1], linewidth=2, linestyle="--",
               label=f"$T_{{GT}}$ = {T_GT:.3f} s")
    ax.set_xlabel("$N_{cycles,total}$  [cycles]")
    ax.set_ylabel("$t_d^{true}$  [s]")
    ax.set_title(f"{TABLE_LABEL}  First true detection time vs $N_{{cycles,total}}$")
    ax.legend(ncol=min(len(steps), 5), fontsize=12)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    _save(fig, _out(plots_dir, "df_02_scatter_td_K.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  Plot df_03 — Heatmap K × step → min(score)
# ══════════════════════════════════════════════════════════════════════════════
def plot_df03_heatmap_K_step(df: pd.DataFrame, plots_dir: str) -> None:
    valid = df[df["run_ok"] & df["score"].notna() & (df["score"] >= 0)]
    pivot = valid.groupby(["N_cyc_total", "step_cyc"])["score"].min().unstack(fill_value=np.nan)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.T, aspect="auto", origin="lower",
                   cmap="viridis_r",
                   extent=[pivot.index.min() - 0.5, pivot.index.max() + 0.5,
                            -0.5, len(pivot.columns) - 0.5])
    cb = fig.colorbar(im, ax=ax, label="min score [s]")
    cb.ax.tick_params(labelsize=13)
    ax.set_xlabel("$N_{cycles,total}$  [cycles]")
    ax.set_ylabel("$\\Delta T_{step}$  [cycles]")
    ax.set_yticks(range(len(pivot.columns)))
    ax.set_yticklabels(pivot.columns)
    ax.set_title(f"{TABLE_LABEL}  Heatmap $N_{{cycles,total}} \\times \\Delta T_{{step}}$ → min score")
    _save(fig, _out(plots_dir, "df_03_heatmap_K_step.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  Plot df_04 — Scatter score vs score_lb
# ══════════════════════════════════════════════════════════════════════════════
def plot_df04_scatter_score_lb(df: pd.DataFrame, plots_dir: str) -> None:
    valid = df[df["run_ok"] & df["score"].notna() & df["score_lb"].notna()
               & (df["score"] >= 0) & (df["score_lb"] >= 0)]
    if valid.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(valid["score_lb"], valid["score"],
                    c=valid["N_cyc_total"], cmap="plasma", alpha=0.6, s=30)
    lims = [min(valid["score_lb"].min(), valid["score"].min()) * 0.98,
            max(valid["score_lb"].max(), valid["score"].max()) * 1.02]
    ax.plot(lims, lims, "k--", linewidth=1.2, label="score = score$_{lb}$")
    cb = fig.colorbar(sc, ax=ax, label="$N_{cycles,total}$  [cycles]")
    cb.ax.tick_params(labelsize=13)
    ax.set_xlabel("score$_{lb}$ (lower bound)  [s]")
    ax.set_ylabel("score  [s]")
    ax.set_title(f"{TABLE_LABEL}  Score vs lower-bound tightness")
    ax.legend()
    _save(fig, _out(plots_dir, "df_04_scatter_score_lb.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  Plot df_05 — Heatmap N_win × step → mean(score)
# ══════════════════════════════════════════════════════════════════════════════
def plot_df05_heatmap_Nwin_step(df: pd.DataFrame, plots_dir: str) -> None:
    valid = df[df["run_ok"] & df["score"].notna() & (df["score"] >= 0)
               & df["N_cyc"].notna()]
    pivot = valid.groupby(["N_cyc", "step_cyc"])["score"].mean().unstack(fill_value=np.nan)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.T, aspect="auto", origin="lower",
                   cmap="viridis_r",
                   extent=[-0.5, len(pivot.index) - 0.5,
                            -0.5, len(pivot.columns) - 0.5])
    cb = fig.colorbar(im, ax=ax, label="mean score [s]")
    cb.ax.tick_params(labelsize=13)
    ax.set_xlabel("$N_{cycles}$  [cycles]")
    ax.set_ylabel("$\\Delta T_{step}$  [cycles]")
    ax.set_xticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.columns)))
    ax.set_yticklabels(pivot.columns)
    ax.set_title(f"{TABLE_LABEL}  Heatmap $N_{{cycles}} \\times \\Delta T_{{step}}$ → mean score")
    _save(fig, _out(plots_dir, "df_05_heatmap_Nwin_step.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  Plot df_06 — Histogram of delta_T_total_vs_des by K
# ══════════════════════════════════════════════════════════════════════════════
def plot_df06_hist_deltaT(df: pd.DataFrame, plots_dir: str) -> None:
    valid = df[df["run_ok"] & df["delta_T_total_vs_des"].notna()]
    Ks    = sorted(valid["N_cyc_total"].unique())

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, K in enumerate(Ks):
        data = valid[valid["N_cyc_total"] == K]["delta_T_total_vs_des"].values * 1e3
        ax.hist(data, bins=20, alpha=0.45, label=f"K={K}",
                color=PALETTE[i % len(PALETTE)], edgecolor="none")
    ax.set_xlabel("$\\Delta T_{total}$  [ms]  (actual − desired)")
    ax.set_ylabel("Count")
    ax.set_title(f"{TABLE_LABEL}  Discretisation gap $\\Delta T_{{total}}$ distribution")
    ax.legend(ncol=4, fontsize=11)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    _save(fig, _out(plots_dir, "df_06_hist_deltaT.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  Plot df_07 — Violin N_fa per K
# ══════════════════════════════════════════════════════════════════════════════
def plot_df07_violin_Nfa(df: pd.DataFrame, plots_dir: str) -> None:
    valid = df[df["run_ok"] & df["P_det"].astype(bool)]
    Ks    = sorted(valid["N_cyc_total"].unique())
    data  = [valid[valid["N_cyc_total"] == K]["N_fa"].values for K in Ks]
    # Filter out empty arrays
    pairs = [(K, d) for K, d in zip(Ks, data) if len(d) > 0]
    if not pairs:
        return
    Ks_plot, data_plot = zip(*pairs)

    fig, ax = plt.subplots(figsize=(10, 5))
    parts = ax.violinplot(data_plot, positions=range(len(Ks_plot)),
                          showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(PALETTE[0])
        pc.set_alpha(0.6)
    parts["cmedians"].set_color(PALETTE[1])
    parts["cmedians"].set_linewidth(2)
    ax.set_xticks(range(len(Ks_plot)))
    ax.set_xticklabels(Ks_plot)
    ax.set_xlabel("$N_{cycles,total}$  [cycles]")
    ax.set_ylabel("$N_{fa}$ (false alarms)")
    ax.set_title(f"{TABLE_LABEL}  False-alarm count distribution per $N_{{cycles,total}}$")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    _save(fig, _out(plots_dir, "df_07_violin_Nfa_K.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  Plot df_08 — Scatter overlap_frac vs mean score (aggregated by overlap)
# ══════════════════════════════════════════════════════════════════════════════
def plot_df08_scatter_overlap_score(df: pd.DataFrame, plots_dir: str) -> None:
    valid = df[df["run_ok"] & df["score"].notna() & (df["score"] >= 0)
               & df["overlap_frac"].notna()]
    agg   = valid.groupby("overlap_frac")["score"].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["overlap_frac", "mean_score", "std_score", "n"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(agg["overlap_frac"], agg["mean_score"],
                yerr=agg["std_score"].fillna(0),
                fmt="o", color=PALETTE[0], ecolor=PALETTE[0],
                elinewidth=1.2, capsize=4, markersize=7)
    ax.set_xlabel("overlap fraction  $(1 - step/N_{win})$")
    ax.set_ylabel("mean score  [s]")
    ax.set_title(f"{TABLE_LABEL}  Mean score vs overlap fraction")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    _save(fig, _out(plots_dir, "df_08_scatter_overlap_score.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  Plot df_09 — P_det rate vs K
# ══════════════════════════════════════════════════════════════════════════════
def plot_df09_pdet_K(df: pd.DataFrame, plots_dir: str) -> None:
    ok   = df[df["run_ok"]]
    Ks   = sorted(ok["N_cyc_total"].unique())
    rate = []
    for K in Ks:
        sub = ok[ok["N_cyc_total"] == K]
        rate.append(sub["P_det"].astype(bool).sum() / max(len(sub), 1))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(Ks, [r * 100 for r in rate], "o-",
            color=PALETTE[0], linewidth=2, markersize=7)
    ax.axhline(100, color=PALETTE[2], linewidth=1.2, linestyle="--",
               label="100 %")
    ax.set_xlabel("$N_{cycles,total}$  [cycles]")
    ax.set_ylabel("Detection rate  [%]")
    ax.set_title(f"{TABLE_LABEL}  Detection rate $P_{{det}}$ vs $N_{{cycles,total}}$")
    ax.set_ylim(-5, 115)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    _save(fig, _out(plots_dir, "df_09_line_Pdet_K.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", default=os.path.join(_HERE, "sweep_output", "sweep_result.pkl"),
                    help="Path to SweepResult pickle file")
    args = ap.parse_args()

    apply_research_style()

    print(f"Loading sweep from: {args.pkl}")
    with open(args.pkl, "rb") as fh:
        sweep = pickle.load(fh)
    df = sweep.df

    plots_dir = os.path.join(_HERE, "sweep_output", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Saving plots to:   {plots_dir}")

    plot_df01_boxplot_score(df, plots_dir)
    print("  [1/9] df_01_boxplot_score_K.png")
    plot_df02_scatter_td(df, plots_dir)
    print("  [2/9] df_02_scatter_td_K.png")
    plot_df03_heatmap_K_step(df, plots_dir)
    print("  [3/9] df_03_heatmap_K_step.png")
    plot_df04_scatter_score_lb(df, plots_dir)
    print("  [4/9] df_04_scatter_score_lb.png")
    plot_df05_heatmap_Nwin_step(df, plots_dir)
    print("  [5/9] df_05_heatmap_Nwin_step.png")
    plot_df06_hist_deltaT(df, plots_dir)
    print("  [6/9] df_06_hist_deltaT.png")
    plot_df07_violin_Nfa(df, plots_dir)
    print("  [7/9] df_07_violin_Nfa_K.png")
    plot_df08_scatter_overlap_score(df, plots_dir)
    print("  [8/9] df_08_scatter_overlap_score.png")
    plot_df09_pdet_K(df, plots_dir)
    print("  [9/9] df_09_line_Pdet_K.png")

    print("plot_01 done.")


if __name__ == "__main__":
    main()
