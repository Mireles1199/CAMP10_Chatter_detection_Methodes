"""
plotting/plot_maxent.py
=======================
Production and debug plots for the MaxEnt-SPRT indicator.

Production plots:
  - plot_maxent_overview  : signal v(t) + SPRT statistic S(t) with thresholds
  - plot_maxent_geometry  : N_seg revolution segments visualised as bars

Debug plots (debug_level >= 3):
  - plot_maxent_resolution_steps : raw → rounded N_seg, T_w vs T_des
  - plot_maxent_constraint_report: semáforo per constraint level
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
_C_IND    = "#8c564b"
_C_TDES   = "#f5a623"
_C_TD     = "#d62728"
_C_SEG    = "#17becf"
_C_OK     = "#2ca02c"


# ──────────────────────────────────────────────────────────────────────────────
# Production plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_maxent_overview(signal: "SignalData", report: "IndicatorReport") -> None:
    """
    Two-panel overview:
      Top   — signal v(t) with detection markers
      Bottom— SPRT statistic S(t) with upper threshold b and detection markers
    """
    if report.result is None:
        return
    res = report.result
    t_I = np.asarray(res.t)
    I_t = np.asarray(res.I_t)
    t_d = np.asarray(res.t_d) if res.t_d is not None else np.array([])
    b   = res.meta.get("b", None)   # upper SPRT threshold

    fig, (ax_sig, ax_ind) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.6]},
    )
    fig.suptitle("MaxEnt-SPRT — Indicator Overview", fontweight="bold")

    ax_sig.plot(signal.t_analysis, signal.signal_analysis, color=_C_SIGNAL,
                lw=0.9, label="$v(t)$")
    for td in t_d:
        ax_sig.axvline(td, color=_C_TD, lw=1.2, ls="--", alpha=0.8)
    ax_sig.set_ylabel("Velocity [m/s]")
    ax_sig.legend(loc="upper left")
    ax_sig.grid(True, lw=0.4)

    ax_ind.plot(t_I, I_t, color=_C_IND, lw=1.0, label="SPRT $S(t)$")
    if b is not None:
        ax_ind.axhline(b, color="k", lw=1.2, ls="--", label=f"threshold $b={b:.2f}$")
    for i, td in enumerate(t_d):
        ax_ind.axvline(td, color=_C_TD, lw=1.2, ls="--",
                       label="$t_d$" if i == 0 else "")
    ax_ind.set_ylabel("SPRT statistic $S(t)$")
    ax_ind.set_xlabel("Time [s]")
    ax_ind.legend(loc="upper left")
    ax_ind.grid(True, lw=0.4)

    if report.T_des is not None:
        ax_ind.annotate(
            f"$T_{{des}}={report.T_des*1000:.1f}\\,\\mathrm{{ms}}$",
            xy=(t_I[0] if len(t_I) else 0, I_t.max() if len(I_t) else 0),
            fontsize=9, color=_C_TDES,
        )

    plt.tight_layout()
    plt.show()


def plot_maxent_geometry(report: "IndicatorReport") -> None:
    """
    Horizontal bars: N_seg revolution periods that compose T_w.

    Each bar = one revolution T_rev.  The total width = T_w_actual.
    A dashed line marks T_des.
    """
    rr = report.resolver_result
    if rr is None:
        return
    N_seg = int(rr.rounded_value)
    T_w   = report.T_w_actual or 0.0
    T_des = report.T_des or 0.0
    T_rev = T_w / N_seg if N_seg > 0 else T_w

    fig, ax = plt.subplots(figsize=(10, 2))
    fig.suptitle("MaxEnt-SPRT — Window Geometry", fontweight="bold")

    for k in range(N_seg):
        ax.barh(0, T_rev * 1000, left=k * T_rev * 1000,
                height=0.4, color=_C_SEG, alpha=0.55 + 0.02 * k,
                edgecolor="k", linewidth=0.6)

    ax.axvline(T_w * 1000, color=_C_TDES, lw=2, ls="--",
               label=f"$T_w$ = {T_w*1000:.1f} ms  ($N_{{seg}}={N_seg}$)")
    if T_des:
        ax.axvline(T_des * 1000, color="gray", lw=1.5, ls=":",
                   label=f"$T_{{des}}$ = {T_des*1000:.1f} ms")

    ax.set_xlim(0, max(T_w, T_des) * 1000 * 1.12)
    ax.set_xlabel("Time [ms]")
    ax.set_yticks([])
    ax.legend(loc="lower right")
    ax.grid(axis="x", lw=0.4)
    ax.text(T_rev * 1000 / 2, 0, "$T_{rev}$", ha="center", va="center",
            fontsize=9, color="k")

    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Debug plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_maxent_resolution_steps(report: "IndicatorReport") -> None:
    """Raw → rounded N_seg and T_w vs T_des comparison."""
    rr = report.resolver_result
    if rr is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    fig.suptitle("MaxEnt-SPRT — Resolution Steps", fontweight="bold")

    axes[0].bar(["raw", "rounded"], [rr.raw_value, rr.rounded_value],
                color=[_C_IND, _C_OK], alpha=0.75, edgecolor="k")
    axes[0].set_title("$N_{seg}$")
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


def plot_maxent_constraint_report(report: "IndicatorReport") -> None:
    """Semáforo per constraint level."""
    cr = report.constraint_report
    if cr is None:
        return
    levels = ["basic", "feasibility", "degenerate"]
    _draw_semaphore(cr, levels, "MaxEnt-SPRT — Constraint Report")


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
