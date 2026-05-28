"""
Optimizacion/plot_11_score_hist.py
====================================
Score distribution analysis — histograms and violin plots of the composite
score per K, separated into detecting and non-detecting runs.

SOURCE TABLE:  SweepResult.df  (raw rows, one per run)

FIGURES (2):
  hist_01_score_kde_K.png    — KDE curves of score per K (detecting runs).
                                One coloured curve per K, superimposed on a
                                single axis per indicator.  Vertical reference
                                line at score=0.05.
  hist_02_score_violin_K.png — Violin + strip of score per K (detecting runs)
                                with mean_score marker (star) and best_score
                                marker (diamond) overlaid.

Rationale
---------
The ``best_score`` (min) selected by the sweep is the lucky optimum over the
full factorial grid.  In practice an operator picks "reasonable" parameters,
so the *distribution* and *mean* of score are more indicative of a K's true
performance.  Comparing both tells us whether K improves the optimum, the
average, or both.

Usage
-----
    python plot_11_score_hist.py [--pkl PATH]
"""
from __future__ import annotations

import argparse
import math
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
TABLE_LABEL = "[sweep.df  score dist.]"

# Score reference line — runs below this are "good"
SCORE_REF = 0.05


def _out(plots_dir: str, name: str) -> str:
    return os.path.join(plots_dir, name)


def _save(fig: plt.Figure, path: str) -> None:
    if SAVE_FIGS:
        fig.savefig(path)
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)


# ── KDE helper ────────────────────────────────────────────────────────────────

def _kde_curve(data: np.ndarray, x_grid: np.ndarray) -> np.ndarray | None:
    """Gaussian KDE evaluated on x_grid.  Returns None if not enough points."""
    data = data[np.isfinite(data)]
    if len(data) < 3:
        return None
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data, bw_method="scott")
        return kde(x_grid)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Plot hist_01 — KDE curves per K, per indicator
# ══════════════════════════════════════════════════════════════════════════════
def plot_hist01_kde(df, plots_dir: str) -> None:
    """
    One subplot per indicator.  Each K gets one KDE curve.
    Colour encodes K using a continuous colormap so many Ks remain readable.
    Only detecting runs with valid score are included.
    """
    valid = df[
        df["run_ok"] & df["P_det"].astype(bool)
        & df["score"].notna() & (df["score"] >= 0)
    ]
    if valid.empty:
        return

    indicators = sorted(valid["indicator"].unique())
    n_ind      = len(indicators)
    fig, axes  = plt.subplots(1, n_ind, figsize=(9 * n_ind, 6), squeeze=False)

    # Global x range (shared across indicators for comparability)
    x_max  = float(valid["score"].quantile(0.98)) * 1.1
    x_grid = np.linspace(0, x_max, 400)

    cmap = plt.get_cmap("plasma")

    for ax, ind in zip(axes[0], indicators):
        sub  = valid[valid["indicator"] == ind]
        Ks   = sorted(sub["N_cyc_total"].unique())
        n_K  = len(Ks)

        for i, K in enumerate(Ks):
            scores = sub[sub["N_cyc_total"] == K]["score"].values
            color  = cmap(i / max(n_K - 1, 1))
            kde_y  = _kde_curve(scores, x_grid)
            if kde_y is not None:
                ax.plot(x_grid, kde_y, linewidth=1.8, color=color,
                        alpha=0.85, label=f"K={K}")
            else:
                # Fallback: just a rug
                ax.plot(scores, np.zeros_like(scores), "|",
                        color=color, markersize=12, alpha=0.8, label=f"K={K}")

        # Reference vertical line
        ax.axvline(SCORE_REF, color="black", linewidth=1.2, linestyle="--",
                   alpha=0.7, label=f"ref = {SCORE_REF}")

        ax.set_xlabel("score  [s]")
        ax.set_ylabel("density")
        ax.set_title(
            f"{TABLE_LABEL}  Score distribution per $N_{{cycles,total}}$ — {ind.upper()}"
        )
        ax.set_xlim(left=0, right=x_max)
        ax.set_ylim(bottom=0)

        # Compact legend (many K values): put outside if > 8
        if n_K <= 8:
            ax.legend(title="$N_{cycles,total}$", fontsize=12, ncol=2)
        else:
            # Colourbar instead of legend
            sm = plt.cm.ScalarMappable(
                cmap=cmap,
                norm=plt.Normalize(vmin=min(Ks), vmax=max(Ks))
            )
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax)
            cb.set_label("$N_{cycles,total}$  [cycles]", fontsize=13)

        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, _out(plots_dir, "hist_01_score_kde_K.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  Plot hist_02 — Violin + mean/best markers per K
# ══════════════════════════════════════════════════════════════════════════════
def plot_hist02_violin(df, plots_dir: str) -> None:
    """
    Violin plot of score (detecting runs) per K, per indicator.
    Overlaid:
      ★ mean_score  (filled star, orange)
      ◆ best_score  (diamond, red)
    Non-detecting runs shown as a rug on the right edge at y = score_penalty
    (visualised as a separate horizontal strip).
    """
    valid = df[
        df["run_ok"] & df["P_det"].astype(bool)
        & df["score"].notna() & (df["score"] >= 0)
    ]
    if valid.empty:
        return

    indicators = sorted(valid["indicator"].unique())
    n_ind      = len(indicators)
    fig, axes  = plt.subplots(1, n_ind, figsize=(12 * n_ind, 6), squeeze=False)

    for ax, ind in zip(axes[0], indicators):
        sub  = valid[valid["indicator"] == ind]
        Ks   = sorted(sub["N_cyc_total"].unique())

        data_list   = []
        mean_scores = []
        best_scores = []
        valid_Ks    = []

        for K in Ks:
            scores = sub[sub["N_cyc_total"] == K]["score"].values
            if len(scores) == 0:
                continue
            data_list.append(scores)
            mean_scores.append(float(scores.mean()))
            best_scores.append(float(scores.min()))
            valid_Ks.append(K)

        if not data_list:
            ax.set_visible(False)
            continue

        positions = np.arange(len(valid_Ks))

        # Violin
        parts = ax.violinplot(
            data_list, positions=positions,
            showmedians=True, showextrema=True, widths=0.7,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(PALETTE[0])
            pc.set_alpha(0.45)
        parts["cmedians"].set_color(PALETTE[0])
        parts["cmedians"].set_linewidth(2)

        # Individual points (strip / jitter)
        rng = np.random.default_rng(42)
        for i, scores in enumerate(data_list):
            jitter = rng.uniform(-0.12, 0.12, len(scores))
            ax.scatter(positions[i] + jitter, scores,
                       color=PALETTE[0], alpha=0.30, s=18, zorder=3)

        # mean_score marker (★)
        ax.scatter(positions, mean_scores,
                   marker="*", s=220, color=PALETTE[3],
                   zorder=6, label="mean score  ★", edgecolors="white",
                   linewidths=0.5)

        # best_score marker (◆)
        ax.scatter(positions, best_scores,
                   marker="D", s=80, color=PALETTE[1],
                   zorder=7, label="best score  ◆", edgecolors="white",
                   linewidths=0.5)

        # Reference line
        ax.axhline(SCORE_REF, color="black", linewidth=1.2,
                   linestyle="--", alpha=0.6,
                   label=f"ref = {SCORE_REF}")

        ax.set_xticks(positions)
        ax.set_xticklabels([f"{K}" for K in valid_Ks], rotation=45, ha="right")
        ax.set_xlabel("$N_{cycles,total}$  [cycles]")
        ax.set_ylabel("score  [s]")
        ax.set_title(
            f"{TABLE_LABEL}  Score distribution (violin) per $N_{{cycles,total}}$ — {ind.upper()}"
        )
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=13)
        ax.yaxis.grid(True, alpha=0.35)
        ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, _out(plots_dir, "hist_02_score_violin_K.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--pkl",
        default=os.path.join(_HERE, "sweep_output", "sweep_result.pkl"),
        help="Path to SweepResult pickle file",
    )
    args = ap.parse_args()

    apply_research_style()

    print(f"Loading sweep from: {args.pkl}")
    with open(args.pkl, "rb") as fh:
        sweep = pickle.load(fh)
    df = sweep.df

    plots_dir = os.path.join(_HERE, "sweep_output", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Saving plots to:   {plots_dir}")

    plot_hist01_kde(df, plots_dir)
    print("  [1/2] hist_01_score_kde_K.png")
    plot_hist02_violin(df, plots_dir)
    print("  [2/2] hist_02_score_violin_K.png")
    print("plot_11 done.")


if __name__ == "__main__":
    main()
