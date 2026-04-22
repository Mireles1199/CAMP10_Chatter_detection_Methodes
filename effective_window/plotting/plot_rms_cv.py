"""
plotting/plot_rms_cv.py
=======================
Production and debug plots for the RMS-CV indicator.

Production plots (show_plots=True):
  - plot_rms_cv_overview  : signal v(t) + CV indicator I(t) with T_des line and t_d markers
  - plot_rms_cv_geometry  : visual bar of the T_w composition (T_rms + overlapping hops)

Debug plots (debug_level >= 3):
  - plot_rms_cv_resolution_steps : raw → rounded → T_w_actual vs T_des
  - plot_rms_cv_constraint_report: semáforo per constraint level
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from ..runner import IndicatorReport
    from ..signal_data import SignalData


# ── colours ──────────────────────────────────────────────────────────────────
_C_SIGNAL  = "#2b7bb9"
_C_IND     = "#e05c2a"
_C_TDES    = "#f5a623"
_C_TD      = "#d62728"
_C_TRMS    = "#2ca02c"
_C_HOP     = "#9467bd"


# ──────────────────────────────────────────────────────────────────────────────
# Production plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_rms_cv_overview(signal: "SignalData", report: "IndicatorReport") -> None:
    """
    Two-panel overview:
      Top   — raw signal v(t) with t_d markers
      Bottom— CV indicator I(t) with T_des horizontal annotation and t_d markers
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
    fig.suptitle("RMS-CV — Indicator Overview", fontweight="bold")

    # Signal
    ax_sig.plot(signal.t_analysis, signal.signal_analysis, color=_C_SIGNAL,
                lw=0.9, label="$v(t)$")
    for td in t_d:
        ax_sig.axvline(td, color=_C_TD, lw=1.2, ls="--", alpha=0.8)
    ax_sig.set_ylabel("Velocity [m/s]")
    ax_sig.legend(loc="upper left")
    ax_sig.grid(True, lw=0.4)

    # Indicator
    ax_ind.plot(t_I, I_t, color=_C_IND, lw=1.1, label="CV $= I(t)$")
    for td in t_d:
        ax_ind.axvline(td, color=_C_TD, lw=1.2, ls="--",
                       label="$t_d$" if td == t_d[0] else "")
    if report.T_des is not None:
        ax_ind.annotate(
            f"$T_{{des}}={report.T_des*1000:.1f}\\,\\mathrm{{ms}}$",
            xy=(t_I[0], ax_ind.get_ylim()[1] if len(t_I) else 0),
            fontsize=9, color=_C_TDES,
        )
    ax_ind.set_ylabel("CV (dimensionless)")
    ax_ind.set_xlabel("Time [s]")
    ax_ind.legend(loc="upper left")
    ax_ind.grid(True, lw=0.4)

    plt.tight_layout()
    plt.show()


def plot_rms_cv_geometry(report: "IndicatorReport") -> None:
    """
    Window-geometry bar showing how T_rms windows and hop shifts compose T_w.

      ┌──────────────────────────────────────────┐
      │  T_rms  |hop|  T_rms  |hop|  T_rms …    │
      └──────────────────────────────────────────┘
                             ← T_w_actual →
    """
    p = report.resolved_params
    N     = p.get("N") or p.get("samples_per_window", 1)
    rho   = p.get("rho") or p.get("overlap_pct", 0.0)
    n_max = int(p.get("n_max", 1))
    fs    = report.resolver_result.resolved_params.get("_fs", 1.0) if report.resolver_result else 1.0
    # fs may not carry into resolved_params; retrieve from T_w
    T_rms = float(N) / (fs if fs and fs > 1 else 1.0)

    # Try to retrieve fs from T_w_actual and formula
    if report.T_w_actual and n_max > 1 and rho < 1:
        # T_w = T_rms * [1 + (n_max-1)(1-rho)]  → T_rms = T_w / K
        K = 1.0 + (n_max - 1) * (1.0 - rho)
        T_rms = report.T_w_actual / K if K > 0 else report.T_w_actual
    elif report.T_w_actual:
        T_rms = report.T_w_actual

    dt = T_rms * (1.0 - rho)   # hop shift
    T_w = report.T_w_actual or T_rms

    fig, ax = plt.subplots(figsize=(10, 2))
    fig.suptitle("RMS-CV — Window Geometry", fontweight="bold")

    for k in range(n_max):
        start = k * dt
        ax.barh(0, T_rms, left=start * 1000, height=0.4,
                color=_C_TRMS, alpha=0.55, edgecolor="k", linewidth=0.6)
        if k < n_max - 1:
            ax.barh(0, dt * 1000, left=(start + T_rms) * 1000 - dt * 1000,
                    height=0.15, color=_C_HOP, alpha=0.8)

    ax.axvline(T_w * 1000, color=_C_TDES, lw=2, ls="--", label=f"$T_w$ = {T_w*1000:.1f} ms")
    if report.T_des:
        ax.axvline(report.T_des * 1000, color="gray", lw=1.5, ls=":",
                   label=f"$T_{{des}}$ = {report.T_des*1000:.1f} ms")

    ax.set_xlim(0, T_w * 1000 * 1.1)
    ax.set_xlabel("Time [ms]")
    ax.set_yticks([])
    ax.legend(loc="lower right")
    ax.grid(axis="x", lw=0.4)

    rms_patch = mpatches.Patch(color=_C_TRMS, alpha=0.55, label=f"$T_{{rms}}$ window (×{n_max})")
    hop_patch = mpatches.Patch(color=_C_HOP, alpha=0.8, label=f"Hop shift (ρ={rho:.2f})")
    ax.legend(handles=[rms_patch, hop_patch], loc="lower right")

    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Debug plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_rms_cv_resolution_steps(report: "IndicatorReport") -> None:
    """
    Bar chart: raw solved value → rounded value → T_w_actual vs T_des.
    Visualises the discretisation error ΔT_w.
    """
    rr = report.resolver_result
    if rr is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    fig.suptitle(f"RMS-CV — Resolution Steps  (solve_{rr.solved_var})", fontweight="bold")

    # Panel 1: solved variable raw vs rounded
    ax = axes[0]
    ax.bar(["raw", "rounded"], [rr.raw_value, rr.rounded_value],
           color=[_C_IND, _C_TRMS], alpha=0.75, edgecolor="k")
    ax.set_title(f"${rr.solved_var}$")
    ax.set_ylabel("Value")
    ax.grid(axis="y", lw=0.4)

    # Panel 2: T_w_actual vs T_des
    ax = axes[1]
    ax.bar(["$T_w$", "$T_{des}$"],
           [rr.T_w_actual * 1000, rr.T_des * 1000],
           color=[_C_IND, _C_TDES], alpha=0.75, edgecolor="k")
    ax.set_title("Window comparison")
    ax.set_ylabel("Time [ms]")
    ax.grid(axis="y", lw=0.4)

    # Panel 3: ΔT_w
    ax = axes[2]
    dT = rr.delta_T_w * 1000
    color = _C_TD if abs(dT) > 0.5 else _C_TRMS
    ax.bar(["$\\Delta T_w$"], [dT], color=color, alpha=0.75, edgecolor="k")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"$\\Delta T_w = {dT:+.3f}$ ms")
    ax.set_ylabel("ΔT_w [ms]")
    ax.grid(axis="y", lw=0.4)

    plt.tight_layout()
    plt.show()


def plot_rms_cv_constraint_report(report: "IndicatorReport") -> None:
    """
    Semáforo: one cell per constraint level, green=pass / red=fail.
    """
    cr = report.constraint_report
    if cr is None:
        return

    levels = ["basic", "feasibility", "degenerate"]
    colors = []
    for lv in levels:
        if not cr.passed and cr.level_failed == lv:
            colors.append("#d62728")
        elif cr.passed or (cr.level_failed and levels.index(lv) > levels.index(cr.level_failed)):
            colors.append("#d3d3d3")  # not reached
        else:
            colors.append("#2ca02c")

    fig, ax = plt.subplots(figsize=(6, 1.5))
    fig.suptitle("RMS-CV — Constraint Report", fontweight="bold")
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
        ax.set_title(f"FAIL: {cr.message}", fontsize=9, color=_C_TD)
    else:
        ax.set_title("All constraints PASSED", fontsize=9, color="#2ca02c")
    plt.tight_layout()
    plt.show()
