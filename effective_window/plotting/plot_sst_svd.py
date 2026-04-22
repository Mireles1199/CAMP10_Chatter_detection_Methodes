"""
plotting/plot_sst_svd.py
========================
Production and debug plots for the SST-SVD (ssq_chatter) indicator.

Theory window formula:  T_w = w + (n_A - 1) * h   [ms]
  n_A     — number of analysis frames  (Ai_length)
  w       — frame width  [ms]          (win_length_ms)
  h       — hop size  [ms]             (hop_ms = h_ratio * w)

Production plots:
  - plot_sst_svd_overview  : signal v(t) + chatter index I(t) with T_des/t_d
  - plot_sst_svd_geometry  : n_A overlapping frames composing T_w

Debug plots (debug_level >= 3):
  - plot_sst_svd_resolution_steps : raw → rounded for solved_var, T_w vs T_des
  - plot_sst_svd_constraint_report: semáforo per constraint level
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from ..runner import IndicatorReport
    from ..signal_data import SignalData

_C_SIGNAL = "#2b7bb9"
_C_IND    = "#9467bd"
_C_TDES   = "#f5a623"
_C_TD     = "#d62728"
_C_FRAME  = "#1f77b4"
_C_OK     = "#2ca02c"


# ──────────────────────────────────────────────────────────────────────────────
# Production plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_sst_svd_overview(signal: "SignalData", report: "IndicatorReport") -> None:
    """
    Two-panel overview:
      Top    — signal v(t) with detection markers
      Bottom — SST-SVD chatter index I(t) with T_des annotation and t_d markers
    """
    if report.result is None:
        return
    res = report.result
    t_I = np.asarray(res.t)
    I_t = np.asarray(res.I_t)
    t_d = np.asarray(res.t_d) if res.t_d is not None else np.array([])

    fig, (ax_sig, ax_ind) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.6]},
    )
    fig.suptitle("SST-SVD — Indicator Overview", fontweight="bold")

    ax_sig.plot(signal.t_analysis, signal.signal_analysis, color=_C_SIGNAL,
                lw=0.9, label="$v(t)$")
    for td in t_d:
        ax_sig.axvline(td, color=_C_TD, lw=1.2, ls="--", alpha=0.8)
    ax_sig.set_ylabel("Velocity [m/s]")
    ax_sig.legend(loc="upper left")
    ax_sig.grid(True, lw=0.4)

    ax_ind.plot(t_I, I_t, color=_C_IND, lw=1.0, label="SST-SVD $I(t)$")
    for i, td in enumerate(t_d):
        ax_ind.axvline(td, color=_C_TD, lw=1.2, ls="--",
                       label="$t_d$" if i == 0 else "")
    ax_ind.set_ylabel("Chatter Index $I(t)$")
    ax_ind.set_xlabel("Time [s]")
    ax_ind.legend(loc="upper left")
    ax_ind.grid(True, lw=0.4)

    if report.T_des is not None:
        rp = report.resolved_params or {}
        w_ms = rp.get("w", rp.get("win_length_ms", None))
        h_ms = rp.get("hop_ms", None)
        ann = f"$T_{{des}}={report.T_des*1000:.1f}\\,\\mathrm{{ms}}$"
        if w_ms:
            ann += f"  $w={w_ms:.1f}\\,\\mathrm{{ms}}$"
        if h_ms:
            ann += f"  $h={h_ms:.1f}\\,\\mathrm{{ms}}$"
        ax_ind.annotate(
            ann, xy=(t_I[0] if len(t_I) else 0, I_t.max() if len(I_t) else 0),
            fontsize=8, color=_C_TDES,
        )

    plt.tight_layout()
    plt.show()


def plot_sst_svd_geometry(report: "IndicatorReport") -> None:
    """
    Overlapping frame diagram:
      n_A horizontal bars of width w_ms, each shifted by h_ms,
      composing T_w = w + (n_A-1)*h.
    """
    rr = report.resolver_result
    if rr is None:
        return
    rp = rr.resolved_params
    n_A   = int(rp.get("n_A", rp.get("Ai_length", 1)))
    w_ms  = rp.get("w", rp.get("win_length_ms", 0.0))
    h_ms  = rp.get("hop_ms", w_ms)   # fallback = full hop
    T_w   = (report.T_w_actual or 0.0) * 1000   # ms
    T_des = (report.T_des or 0.0) * 1000         # ms

    fig, ax = plt.subplots(figsize=(11, 2.5))
    fig.suptitle("SST-SVD — Window Geometry", fontweight="bold")

    for k in range(n_A):
        start = k * h_ms
        alpha = 0.35 + 0.05 * (k % 3)
        ax.barh(0, w_ms, left=start, height=0.35,
                color=_C_FRAME, alpha=alpha, edgecolor="k", linewidth=0.5)
        ax.text(start + w_ms / 2, 0, f"{k+1}", ha="center", va="center",
                fontsize=7, color="white")

    ax.axvline(T_w, color=_C_TDES, lw=2, ls="--",
               label=f"$T_w$ = {T_w:.1f} ms")
    if T_des:
        ax.axvline(T_des, color="gray", lw=1.5, ls=":",
                   label=f"$T_{{des}}$ = {T_des:.1f} ms")

    ax.set_xlim(0, max(T_w, T_des) * 1.10)
    ax.set_xlabel("Time [ms]")
    ax.set_yticks([])
    ax.legend(loc="lower right")
    ax.grid(axis="x", lw=0.4)

    info = f"$n_A={n_A}$,  $w={w_ms:.1f}\\,\\mathrm{{ms}}$,  $h={h_ms:.1f}\\,\\mathrm{{ms}}$"
    ax.set_title(info, fontsize=9)

    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Debug plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_sst_svd_resolution_steps(report: "IndicatorReport") -> None:
    """Raw → rounded for solved_var; T_w vs T_des; ΔT_w."""
    rr = report.resolver_result
    if rr is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    fig.suptitle("SST-SVD — Resolution Steps", fontweight="bold")

    label = rr.solved_var
    axes[0].bar(["raw", "rounded"], [rr.raw_value, rr.rounded_value],
                color=[_C_IND, _C_OK], alpha=0.75, edgecolor="k")
    axes[0].set_title(f"Solved: ${label}$")
    axes[0].set_ylabel("Value")
    axes[0].grid(axis="y", lw=0.4)

    axes[1].bar(["$T_w$", "$T_{des}$"],
                [rr.T_w_actual * 1000, rr.T_des * 1000],
                color=[_C_IND, _C_TDES], alpha=0.75, edgecolor="k")
    axes[1].set_title("Window comparison")
    axes[1].set_ylabel("Time [ms]")
    axes[1].grid(axis="y", lw=0.4)

    dT = rr.delta_T_w * 1000
    color = _C_TD if abs(dT) > 0.5 else _C_OK
    axes[2].bar(["$\\Delta T_w$"], [dT], color=color, alpha=0.75, edgecolor="k")
    axes[2].axhline(0, color="k", lw=0.8)
    axes[2].set_title(f"$\\Delta T_w = {dT:+.3f}$ ms")
    axes[2].set_ylabel("ΔT_w [ms]")
    axes[2].grid(axis="y", lw=0.4)

    plt.tight_layout()
    plt.show()


def plot_sst_svd_constraint_report(report: "IndicatorReport") -> None:
    """Semáforo per constraint level with h_ratio advisory note."""
    cr = report.constraint_report
    if cr is None:
        return
    levels = ["basic", "feasibility", "degenerate"]
    _draw_semaphore(cr, levels, "SST-SVD — Constraint Report")

    # Display hop advisory if present
    if cr.passed and cr.details and cr.details.get("_runner_hop_advisory"):
        note = cr.details["_runner_hop_advisory"]
        fig, ax = plt.subplots(figsize=(6, 0.8))
        fig.patch.set_facecolor("#fff3cd")
        ax.axis("off")
        ax.text(0.5, 0.5, f"Advisory: {note}", ha="center", va="center",
                fontsize=9, color="#856404", wrap=True,
                transform=ax.transAxes)
        plt.tight_layout()
        plt.show()


def _draw_semaphore(cr, levels, title):
    colors = []
    for lv in levels:
        if not cr.passed and cr.level_failed == lv:
            colors.append("#d62728")
        elif cr.passed or (cr.level_failed and levels.index(lv) > levels.index(cr.level_failed)):
            colors.append("#d3d3d3")
        else:
            colors.append(_C_OK)

    fig, ax = plt.subplots(figsize=(6, 1.5))
    fig.suptitle(title, fontweight="bold")
    ax.set_xlim(0, len(levels))
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, (lv, c) in enumerate(zip(levels, colors)):
        rect = mpatches.FancyBboxPatch(
            (i + 0.05, 0.1), 0.85, 0.8,
            boxstyle="round,pad=0.05", color=c, alpha=0.8,
        )
        ax.add_patch(rect)
        ax.text(i + 0.5, 0.5, lv, ha="center", va="center", fontsize=11, color="white")
    if not cr.passed:
        ax.set_title(f"FAIL: {cr.message}", fontsize=9, color="#d62728")
    else:
        ax.set_title("All constraints PASSED", fontsize=9, color=_C_OK)
    plt.tight_layout()
    plt.show()
