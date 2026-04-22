"""
plotting/plot_combined.py
=========================
Combined and cross-indicator plots called by WindowRunner.

Functions called by runner.py (production, show_plots=True):
  - plot_indicator_overview(signal, report)     — dispatches to per-indicator module
  - plot_window_geometry(report)               — dispatches to per-indicator module
  - plot_all_indicators(signal, result)        — stacked I(t) for all indicators
  - plot_parameter_table(result)               — matplotlib table: params + T_w + ΔT_w

Functions called by runner.py (debug, debug_level >= 3):
  - plot_resolution_steps(report)              — dispatches to per-indicator module
  - plot_constraint_report(report)             — dispatches to per-indicator module

Extra summary plots (can be called manually):
  - plot_delta_Tw_comparison(result)           — bar chart |ΔT_w| per indicator
  - plot_feasibility_summary(result)           — heatmap: constraint levels per indicator
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from ..runner import IndicatorReport, WindowResult
    from ..signal_data import SignalData

_C_OK      = "#2ca02c"
_C_WARN    = "#ff7f0e"
_C_FAIL    = "#d62728"
_C_NEUTRAL = "#aec7e8"

# ── Per-indicator dispatch maps ───────────────────────────────────────────────

def _get_overview_fn(indicator_id: str):
    iid = indicator_id.lower()
    if "rms" in iid or "rms_cv" in iid:
        from .plot_rms_cv import plot_rms_cv_overview
        return plot_rms_cv_overview
    elif "maxent" in iid or "sprt" in iid:
        from .plot_maxent import plot_maxent_overview
        return plot_maxent_overview
    elif "sst" in iid or "ssq" in iid or "svd" in iid:
        from .plot_sst_svd import plot_sst_svd_overview
        return plot_sst_svd_overview
    return None


def _get_geometry_fn(indicator_id: str):
    iid = indicator_id.lower()
    if "rms" in iid or "rms_cv" in iid:
        from .plot_rms_cv import plot_rms_cv_geometry
        return plot_rms_cv_geometry
    elif "maxent" in iid or "sprt" in iid:
        from .plot_maxent import plot_maxent_geometry
        return plot_maxent_geometry
    elif "sst" in iid or "ssq" in iid or "svd" in iid:
        from .plot_sst_svd import plot_sst_svd_geometry
        return plot_sst_svd_geometry
    return None


def _get_resolution_fn(indicator_id: str):
    iid = indicator_id.lower()
    if "rms" in iid or "rms_cv" in iid:
        from .plot_rms_cv import plot_rms_cv_resolution_steps
        return plot_rms_cv_resolution_steps
    elif "maxent" in iid or "sprt" in iid:
        from .plot_maxent import plot_maxent_resolution_steps
        return plot_maxent_resolution_steps
    elif "sst" in iid or "ssq" in iid or "svd" in iid:
        from .plot_sst_svd import plot_sst_svd_resolution_steps
        return plot_sst_svd_resolution_steps
    return None


def _get_constraint_fn(indicator_id: str):
    iid = indicator_id.lower()
    if "rms" in iid or "rms_cv" in iid:
        from .plot_rms_cv import plot_rms_cv_constraint_report
        return plot_rms_cv_constraint_report
    elif "maxent" in iid or "sprt" in iid:
        from .plot_maxent import plot_maxent_constraint_report
        return plot_maxent_constraint_report
    elif "sst" in iid or "ssq" in iid or "svd" in iid:
        from .plot_sst_svd import plot_sst_svd_constraint_report
        return plot_sst_svd_constraint_report
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Dispatch wrappers (called by runner.py)
# ──────────────────────────────────────────────────────────────────────────────

def plot_indicator_overview(signal: "SignalData", report: "IndicatorReport") -> None:
    """Dispatch to the per-indicator overview plot."""
    fn = _get_overview_fn(report.indicator_id)
    if fn is not None:
        fn(signal, report)


def plot_window_geometry(report: "IndicatorReport") -> None:
    """Dispatch to the per-indicator window geometry plot."""
    fn = _get_geometry_fn(report.indicator_id)
    if fn is not None:
        fn(report)


def plot_resolution_steps(report: "IndicatorReport") -> None:
    """Dispatch to the per-indicator resolution-steps debug plot."""
    fn = _get_resolution_fn(report.indicator_id)
    if fn is not None:
        fn(report)


def plot_constraint_report(report: "IndicatorReport") -> None:
    """Dispatch to the per-indicator constraint-report debug plot."""
    fn = _get_constraint_fn(report.indicator_id)
    if fn is not None:
        fn(report)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-indicator production plots (runner.py uses these when show_plots=True)
# ──────────────────────────────────────────────────────────────────────────────

def plot_all_indicators(signal: "SignalData", result: "WindowResult") -> None:
    """
    Stacked panel plot: one row per successful indicator.
    Each row: I(t) with t_d markers and T_des annotation.
    """
    reports = result.successful()
    if not reports:
        return

    n = len(reports)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f"All Indicators  —  $T_{{des}} = {result.T_des * 1000:.1f}\\,\\mathrm{{ms}}$",
        fontweight="bold",
    )

    colors = ["#1f77b4", "#d62728", "#9467bd", "#2ca02c", "#ff7f0e"]
    for ax, rep, col in zip(axes, reports, colors):
        if rep.result is None:
            ax.text(0.5, 0.5, "No result", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_ylabel(rep.indicator_id, fontsize=9)
            continue
        res = rep.result
        t_I = np.asarray(res.t)
        I_t = np.asarray(res.I_t)
        t_d = np.asarray(res.t_d) if res.t_d is not None else np.array([])

        ax.plot(t_I, I_t, color=col, lw=1.0)
        for i, td in enumerate(t_d):
            ax.axvline(td, color="#d62728", lw=1.0, ls="--",
                       label="$t_d$" if i == 0 else "")
        ax.set_ylabel(rep.indicator_id, fontsize=9)
        ax.grid(True, lw=0.4)
        info = f"$T_w={rep.T_w_actual*1000:.1f}$ms, $\\Delta T_w={rep.delta_T_w*1000:+.2f}$ms"
        ax.annotate(info, xy=(0.02, 0.88), xycoords="axes fraction",
                    fontsize=8, color=col)

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()


def plot_parameter_table(result: "WindowResult") -> None:
    """
    Matplotlib table showing resolved parameters + T_w + ΔT_w for every indicator.
    Rows = indicators, columns = parameters.
    """
    reports = [r for r in result.reports if not r.skipped]
    if not reports:
        return

    # Gather all parameter keys
    all_keys: list[str] = []
    for rep in reports:
        for k in (rep.resolved_params or {}):
            if k not in all_keys:
                all_keys.append(k)
    col_headers = all_keys + ["T_w (ms)", "ΔT_w (ms)", "Status"]
    row_headers = [r.indicator_id for r in reports]

    cell_text = []
    for rep in reports:
        rp = rep.resolved_params or {}
        row = []
        for k in all_keys:
            v = rp.get(k, "—")
            if isinstance(v, float):
                row.append(f"{v:.4g}")
            else:
                row.append(str(v))
        row.append(f"{rep.T_w_actual * 1000:.2f}" if rep.T_w_actual else "—")
        row.append(f"{rep.delta_T_w * 1000:+.2f}" if rep.delta_T_w is not None else "—")
        status = "OK" if (rep.constraint_report and rep.constraint_report.passed) else "FAIL"
        row.append(status)
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(max(8, len(col_headers) * 1.4), len(reports) * 0.7 + 1.5))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_text,
        rowLabels=row_headers,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    # Colour status column
    n_cols = len(col_headers)
    for i, rep in enumerate(reports):
        status = cell_text[i][-1]
        c = _C_OK if status == "OK" else _C_FAIL
        tbl[i + 1, n_cols - 1].set_facecolor(c)
        tbl[i + 1, n_cols - 1].set_text_props(color="white")

    fig.suptitle("Resolved Parameters Summary", fontweight="bold")
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Optional summary plots (call manually for analysis)
# ──────────────────────────────────────────────────────────────────────────────

def plot_delta_Tw_comparison(result: "WindowResult") -> None:
    """
    Bar chart: |ΔT_w| [ms] per indicator.

    Bars coloured by magnitude:
      green  → |ΔT_w| < 0.5 ms
      orange → 0.5 ≤ |ΔT_w| < 2.0 ms
      red    → |ΔT_w| ≥ 2.0 ms
    """
    reports = [r for r in result.reports if not r.skipped and r.delta_T_w is not None]
    if not reports:
        return

    ids  = [r.indicator_id for r in reports]
    vals = [abs(r.delta_T_w) * 1000 for r in reports]

    def _color(v):
        if v < 0.5:
            return _C_OK
        if v < 2.0:
            return _C_WARN
        return _C_FAIL

    colors = [_color(v) for v in vals]

    fig, ax = plt.subplots(figsize=(max(5, len(ids) * 1.5), 4))
    ax.bar(ids, vals, color=colors, edgecolor="k", alpha=0.8)
    ax.axhline(0.5, color=_C_WARN, lw=1, ls="--", label="0.5 ms threshold")
    ax.axhline(2.0, color=_C_FAIL, lw=1, ls="--", label="2.0 ms threshold")
    ax.set_ylabel("|ΔT_w|  [ms]")
    ax.set_title(
        f"Window Mismatch |$\\Delta T_w$|  —  $T_{{des}}={result.T_des*1000:.1f}$ ms",
        fontweight="bold",
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", lw=0.4)
    plt.tight_layout()
    plt.show()


def plot_feasibility_summary(result: "WindowResult") -> None:
    """
    Heatmap: constraint status per indicator × constraint level.

    Colour scale:
      green  = passed
      red    = failed at this level
      grey   = not reached (failed earlier)
    """
    reports = [r for r in result.reports if not r.skipped]
    if not reports:
        return

    levels = ["basic", "feasibility", "degenerate"]
    ids = [r.indicator_id for r in reports]
    matrix = np.zeros((len(reports), len(levels)))

    for i, rep in enumerate(reports):
        cr = rep.constraint_report
        if cr is None:
            matrix[i, :] = 0.5  # unknown
            continue
        if cr.passed:
            matrix[i, :] = 1.0  # all green
        else:
            lf_idx = levels.index(cr.level_failed) if cr.level_failed in levels else -1
            for j in range(len(levels)):
                if j < lf_idx:
                    matrix[i, j] = 1.0   # passed
                elif j == lf_idx:
                    matrix[i, j] = 0.0   # failed here
                else:
                    matrix[i, j] = 0.5   # not reached

    # Build custom colormap: 0 = red, 0.5 = grey, 1 = green
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "semaphore", [
            (0.0, _C_FAIL),
            (0.5, "#d3d3d3"),
            (1.0, _C_OK),
        ]
    )

    fig, ax = plt.subplots(figsize=(max(5, len(levels) * 1.5), max(3, len(reports) * 0.8)))
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels, fontsize=10)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(ids, fontsize=10)
    ax.set_title("Constraint Feasibility Summary", fontweight="bold")

    labels = {0.0: "FAIL", 0.5: "N/A", 1.0: "PASS"}
    for i in range(len(reports)):
        for j in range(len(levels)):
            v = matrix[i, j]
            txt = labels.get(round(v * 2) / 2, "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if v != 0.5 else "k")

    plt.tight_layout()
    plt.show()
