"""
Publication-quality composite figure generator for the RMS-CV indicator.

Use :func:`plots_rms_cv` to produce a three-panel figure that shows:

1. The raw tool-velocity signal with optional RMS-window boundaries.
2. The windowed RMS sequence with optional CV-block boundaries.
3. The online CV sequence with the detection threshold.

All sub-plots share the same x-axis limits and styling defined by the
:func:`configurar_estilo_global` helper.  The helper function
:func:`fig_size` provides IEEE/Elsevier compatible figure dimensions.
"""

#%%
# ========= Imports =========
from __future__ import annotations
# import matplotlib
# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

import colorsys

from typing import Dict, Any, Sequence, Optional

import numpy as np
from scipy.stats import norm as _scipy_norm

from ..utils.types import IndicatorResult, SignalData

import colorsys

r, g, b = colorsys.hls_to_rgb(346/360, 0.45, 0.99)
color_red    = (r, g, b)   # alarm / upper threshold

r, g, b = colorsys.hls_to_rgb(36/360, 0.45, 0.99)
color_orange = (r, g, b)   # chatter signal / detection td / CV scatter

r, g, b = colorsys.hls_to_rgb(279/360, 0.36, 0.99)
color_purple = (r, g, b)   # auxiliary curves

r, g, b = colorsys.hls_to_rgb(98/360, 0.36, 0.99)
color_verde  = (r, g, b)   # stable threshold / mu_stable

r, g, b = colorsys.hls_to_rgb(206.957/360, 0.40941, 0.55603)
color_azul   = (r, g, b)   # stable signal / RMS sequence



def fig_size(scale=1.0, ncols=1, base_width=3.4):
    """Return a Matplotlib-compatible figure size tuple.

    Computes width and height so that figures fit the standard column widths
    used by IEEE and Elsevier journals.  The height is always 70 % of the
    computed width.

    Args:
        scale (float, optional): Global scaling factor applied to both
            dimensions.  ``1.0`` gives the nominal journal column width.
            Defaults to ``1.0``.
        ncols (int, optional): Number of journal columns the figure should
            span (``1`` = single-column, ``2`` = double-column).  Defaults
            to ``1``.
        base_width (float, optional): Width [inches] of a single journal
            column.  Defaults to ``3.4`` (IEEE single-column).

    Returns:
        tuple[float, float]: ``(width, height)`` in inches.

    Example:
        >>> fig_size(scale=1.5, ncols=2)
        (10.2, 7.140000000000001)
    """
    width = base_width * ncols * scale
    height = width * 0.8   # relación agradable
    return (width, height)

def configurar_estilo_global() -> None:
    """Configura el estilo global de los gráficos."""
    # plt.style.use('dark_background')

    local_style = {
        # Tipografía general
        'font.family': 'serif',
        'font.size': 9,

        # Tamaños de títulos y etiquetas
        'axes.titlesize': 25,
        'axes.labelsize': 25,
        'xtick.labelsize': 23,
        'ytick.labelsize': 23,
        'legend.fontsize': 23,

        # Estética de líneas
        'lines.linewidth': 1.25,
        'lines.markersize': 6,

        # Bordes y ejes
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,

        # Ticks
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2.5,
        'ytick.minor.size': 2.5,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,

        # Texto matemático
        'mathtext.fontset': 'stix',
        'axes.formatter.use_mathtext': True,

        # Leyenda
        'legend.frameon': False,
        'legend.loc': 'best',
        'legend.handlelength': 2.0,
        'legend.borderaxespad': 0.5,

        # Exportación
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'savefig.transparent': True,

        # Fondo
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        }

    plt.rcParams.update(local_style)

# %%
# ========= Configuración global de estilo de gráficos =========



#%%
# ===========

def plots_rms_cv(
    signal: Optional[SignalData],
    result: IndicatorResult,
    show_signal: bool = True,
    show: bool = True,
    zoom_x: Optional[tuple[float, float]] = None,
    zoom_y: Optional[tuple[float, float]] = None,
    vlines: Optional[Sequence[float]] = None,
    hlines: Optional[Sequence[float]] = None,
    t_gt: Optional[float] = None,
) -> plt.Figure:
    """Generate the three-panel RMS-CV diagnostic figure.

    Produces three independent
    :class:`~matplotlib.figure.Figure` objects that are each displayed or
    returned:

    1. **Tool velocity** — ``signal.signal_analysis`` vs ``signal.t_analysis``
       with optional RMS-window start-markers (*vlines* from the RMS indices).
    2. **RMS sequence** — windowed RMS values with optional CV-block
       boundaries drawn every *n_max* RMS frames.
    3. **CV sequence** — online CV values with the threshold line and
       optional user annotations.

    Args:
        signal (Optional[SignalData]): Container for the raw signal.  When
            supplied its ``signal_analysis`` and ``t_analysis`` arrays are
            used for panel 1.  Pass ``None`` to skip the signal panel.
        result (IndicatorResult): Result object returned by
            :func:`~rms_cv.lib.runner.rms_cv_pipeline`.  The ``meta``
            dictionary must contain at least the keys ``"t_rms"``,
            ``"rms_values"``, ``"cv_time"``, ``"cv_values"``, and
            ``"cv_threshold"``.
        show_signal (bool, optional): Whether to render panel 1 (tool
            velocity).  Defaults to ``True``.
        show (bool, optional): Call :func:`matplotlib.pyplot.show` after
            creating all figures.  Defaults to ``True``.
        zoom_x (Optional[tuple[float, float]], optional): Horizontal
            x-axis limits ``(x_min, x_max)`` applied to all panels.
            ``None`` = auto.
        zoom_y (Optional[tuple[float, float]], optional): Vertical y-axis
            limits applied to the CV panel.  ``None`` = auto.
        vlines (Optional[Sequence[float]], optional): Additional vertical
            lines [s] drawn across all panels (e.g., known chatter onset
            times from a reference measurement).
        hlines (Optional[Sequence[float]], optional): Horizontal reference
            lines drawn on the CV panel only.

    Returns:
        plt.Figure: The last figure created (CV panel).  The signal and RMS
        figures are accessible via the standard Matplotlib figure manager.

    Example:
        >>> from rms_cv import run_rms_cv, SignalData
        >>> from rms_cv.viz.rms_cv_plots import plots_rms_cv
        >>> fig = plots_rms_cv(signal_data, result, zoom_x=(0.5, 2.0))
    """

    def _draw_vlines(ax, vlines, default_color="black", default_ls="--"):
        """Draw vertical event lines with optional rotated text labels (indicator-plot-style)."""
        if vlines is None:
            return
        for entry in vlines:
            if isinstance(entry, (list, tuple)):
                vx    = float(entry[0])
                label = str(entry[1]) if len(entry) > 1 else None
                color = entry[2]      if len(entry) > 2 else default_color
            else:
                vx, label, color = float(entry), None, default_color
            ax.axvline(x=vx, color=color, linestyle=default_ls, lw=1.2)
            if label:
                ax.text(
                    vx, 0.97, f"  {label}",
                    rotation=90, va="top", ha="right", fontsize=16,
                    color=color, transform=ax.get_xaxis_transform(),
                )

    def _plot_rms(times: "np.ndarray", rms: "np.ndarray",
                  zoom_x: Optional[tuple[float, float]] = None,
                  zoom_y: Optional[tuple[float, float]] = None, *,
                  title: str = "RMS", scale: float = 1.0,
                  vlines: Optional[Sequence[float]] = None,
                  hlines: Optional[Sequence[float]] = None,
                  **kargs) -> tuple:
        fig, axes = plt.subplots(figsize=fig_size(scale=scale, ncols=1))
        axes.plot(times, rms, marker="o", color=color_azul)
        axes.set_xlabel("Time (s)")
        axes.set_ylabel("RMS")
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        if zoom_x is not None:
            axes.set_xlim(zoom_x)
        _draw_vlines(axes, vlines)
        if hlines is not None:
            for yv in hlines:
                axes.axhline(y=yv, color='gray', linestyle='--', lw=1, alpha=0.7)
        axes.set_title(title)
        axes.grid(False)
        plt.tight_layout()
        return fig, axes

    def _plot_cv(time_seq: Sequence[float], cv_seq: Sequence[float],
                 cv_threshold: Optional[float],
                 zoom_x: Optional[tuple[float, float]] = None,
                 zoom_y: Optional[tuple[float, float]] = None,
                 *, title: str = "CV", scale: float = 1.0,
                 cv_threshold_method: str = "fixed",
                 cv_threshold_low: Optional[float] = None,
                 cv_mu_stable: Optional[float] = None,
                 vlines: Optional[Sequence[float]] = None,
                 hlines: Optional[Sequence[float]] = None) -> tuple:
        fig, axes = plt.subplots(figsize=fig_size(scale=scale, ncols=1))
        axes.scatter(time_seq, cv_seq, color=color_orange, marker="o", s=30)
        if cv_threshold is not None:
            _is_adaptive = cv_threshold_method == "stable_region"
            axes.axhline(y=cv_threshold, color=color_red, linestyle="--", linewidth=1.4)
            axes.text(0.99, cv_threshold,
                      rf"$\mu + 3\sigma = {cv_threshold:.4g}$",
                      transform=axes.get_yaxis_transform(),
                      color=color_red, ha='right', va='bottom', fontsize=16)
            if _is_adaptive and cv_threshold_low is not None and cv_threshold_low > 0:
                axes.axhline(y=cv_threshold_low, color=color_red, linestyle=":", linewidth=1.2)
                axes.text(0.99, cv_threshold_low,
                          rf"$\mu - 3\sigma = {cv_threshold_low:.4g}$",
                          transform=axes.get_yaxis_transform(),
                          color=color_red, ha='right', va='top', fontsize=16)
            if cv_mu_stable is not None:
                axes.axhline(y=cv_mu_stable, color=color_verde, linestyle="-", linewidth=1.0)
                axes.text(0.99, cv_mu_stable,
                          rf"$\mu_{{stable}} = {cv_mu_stable:.4g}$",
                          transform=axes.get_yaxis_transform(),
                          color=color_verde, ha='right', va='bottom', fontsize=16)
        axes.set_xlabel("Time (s)")
        axes.set_ylabel("CV")
        axes.set_title(title)
        axes.grid(False)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        if zoom_x is not None:
            axes.set_xlim(zoom_x)
        if zoom_y is not None:
            axes.set_ylim(zoom_y)
        _draw_vlines(axes, vlines)
        if hlines is not None:
            for yv in hlines:
                axes.axhline(y=yv, color='gray', linestyle='--', lw=1, alpha=0.7)
        plt.tight_layout()
        return fig, axes

    def _plot_signal(t: "np.ndarray", x: "np.ndarray", *,
                     zoom_x: Optional[tuple[float, float]] = None,
                     zoom_y: Optional[tuple[float, float]] = None,
                     title: str = "Signal",
                     scale: float = 1.0,
                     vlines: Optional[Sequence[float]] = None,
                     hlines: Optional[Sequence[float]] = None,
                     **kargs) -> tuple:
        fig, axes = plt.subplots(figsize=fig_size(scale=scale, ncols=1))
        axes.plot(t, x, color=color_azul)
        axes.set_xlabel("Time (s)")
        axes.set_ylabel(r"Velocity $v(t)$ [m/s]")
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        if zoom_x is not None:
            axes.set_xlim(zoom_x)
        _draw_vlines(axes, vlines)
        if hlines is not None:
            for yv in hlines:
                axes.axhline(y=yv, color='gray', linestyle='--', lw=1, alpha=0.7)
        axes.set_title(title)
        axes.grid(False)
        plt.tight_layout()
        return fig, axes

    # ── C1: Signal split by region ──────────────────────────────────────────
    def _plot_signal_split(
        t_s: np.ndarray, x_s: np.ndarray, t_gt_val: float,
        zoom_x=None, zoom_y=None, scale: float = 1.0,
        vlines=None, fig_label: Optional[str] = None,
        **kargs,
    ) -> tuple:
        """Signal colored by region: stable (azul) before t_gt, chatter (orange) after."""
        fig, ax = plt.subplots(figsize=fig_size(scale=scale, ncols=1), num=fig_label)
        mask_s = t_s < t_gt_val
        mask_c = t_s >= t_gt_val
        if np.any(mask_s):
            ax.plot(t_s[mask_s], x_s[mask_s], color=color_azul, label="Stable")
        if np.any(mask_c):
            ax.plot(t_s[mask_c], x_s[mask_c], color=color_orange, label="Chatter")
        if zoom_x is not None:
            ax.set_xlim(zoom_x)
        if zoom_y is not None:
            ax.set_ylim(zoom_y)
        _draw_vlines(ax, vlines)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"Velocity $v(t)$ [m/s]")
        ax.set_title("Tool Velocity — Split by Region")
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.legend()
        ax.grid(False)
        plt.tight_layout()
        return fig, ax

    # ── C2: RMS colored by region ───────────────────────────────────────────
    def _plot_rms_colored(
        t_rms_arr: np.ndarray, rms_arr: np.ndarray, t_gt_val: float,
        cv_num_data: Optional[int] = None,
        zoom_x=None, scale: float = 1.0,
        vlines=None, fig_label: Optional[str] = None,
        **kargs,
    ) -> tuple:
        """RMS sequence colored by region + vertical CV-block boundaries every n_max frames."""
        fig, ax = plt.subplots(figsize=fig_size(scale=scale, ncols=1), num=fig_label)
        mask_s = t_rms_arr < t_gt_val
        mask_c = t_rms_arr >= t_gt_val
        if np.any(mask_s):
            ax.plot(t_rms_arr[mask_s], rms_arr[mask_s], marker="o", markersize=3,
                    color=color_azul, label="Stable RMS")
        if np.any(mask_c):
            ax.plot(t_rms_arr[mask_c], rms_arr[mask_c], marker="o", markersize=3,
                    color=color_orange, label="Chatter RMS")
        # if cv_num_data is not None and cv_num_data > 0:
        #     for i in range(0, len(t_rms_arr), cv_num_data):
        #         ax.axvline(t_rms_arr[i], color='gray', ls=':', lw=0.8, alpha=0.5)
        if zoom_x is not None:
            ax.set_xlim(zoom_x)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        _draw_vlines(ax, vlines)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("RMS")
        ax.set_title("RMS Sequence — Colored by Region")
        ax.legend()
        ax.grid(False)
        plt.tight_layout()
        return fig, ax

    # ── C3: CV histogram stable vs chatter ──────────────────────────────────
    def _plot_cv_hist(
        cv_time_arr: np.ndarray, cv_arr: np.ndarray, t_gt_val: float,
        cv_threshold: Optional[float] = None,
        cv_threshold_low: Optional[float] = None,
        mu_stable: Optional[float] = None,
        scale: float = 1.0,
        fig_label: Optional[str] = None,
        **kargs,
    ) -> tuple:
        """Histogram of CV values: stable (blue) vs chatter (orange) + threshold annotations."""
        fig, ax = plt.subplots(figsize=fig_size(scale=scale, ncols=1), num=fig_label)
        cv_np  = np.asarray(cv_arr)
        t_np   = np.asarray(cv_time_arr)
        mask_s = t_np < t_gt_val
        mask_c = t_np >= t_gt_val
        if np.any(mask_s):
            ax.hist(cv_np[mask_s], bins=40, density=True, alpha=0.55,
                    color=color_azul, label=f"Stable  (n={int(mask_s.sum())})")
            mu_s, std_s = np.mean(cv_np[mask_s]), np.std(cv_np[mask_s])
            if std_s > 0:
                xs = np.linspace(mu_s - 4 * std_s, mu_s + 4 * std_s, 300)
                ax.plot(xs, _scipy_norm.pdf(xs, mu_s, std_s),
                        color=color_azul, lw=1.8, ls="-")
                ax.axvline(mu_s, color=color_verde, ls="-", lw=1.4)
                ax.text(mu_s, 0.97, rf"  $\mu={mu_s:.3g}$",
                        rotation=90, va="top", ha="right", fontsize=14,
                        color=color_verde, transform=ax.get_xaxis_transform())
        if cv_threshold is not None:
            ax.axvline(cv_threshold, color=color_red, ls="--", lw=1.4)
            ax.text(cv_threshold, 0.97, rf"  $\mu+3\sigma={cv_threshold:.4g}$",
                    rotation=90, va="top", ha="right", fontsize=16,
                    color=color_red, transform=ax.get_xaxis_transform())
        if cv_threshold_low is not None and cv_threshold_low > 0:
            ax.axvline(cv_threshold_low, color=color_red, ls=":", lw=1.2)
            ax.text(cv_threshold_low, 0.97, rf"  $\mu-3\sigma={cv_threshold_low:.4g}$",
                    rotation=90, va="top", ha="right", fontsize=16,
                    color=color_red, transform=ax.get_xaxis_transform())
        ax.set_xlabel("CV")
        ax.set_ylabel("Density")
        ax.set_title("CV Distribution — Stable vs Chatter")
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.legend()
        ax.grid(False)
        plt.tight_layout()
        return fig, ax

    # ── C5: μ and σ per CV window evolution ────────────────────────────────
    def _plot_mu_sigma_evolution(
        cv_time_arr: np.ndarray, mu_arr: np.ndarray, sigma_arr: np.ndarray,
        zoom_x=None, scale: float = 1.0,
        vlines=None,
        fig_label_mu: Optional[str] = None,
        fig_label_sigma: Optional[str] = None,
        **kargs,
    ) -> tuple:
        """Two separate figures: \u03bc(t) (azul) and \u03c3(t) (purple) per CV window."""
        # — Figure \u03bc(t) —
        fig_mu, ax_mu = plt.subplots(figsize=fig_size(scale=scale, ncols=1), num=fig_label_mu)
        ax_mu.plot(cv_time_arr, mu_arr, color=color_azul,
                   marker="o", markersize=3, linestyle="-")
        ax_mu.set_xlabel("Time (s)")
        ax_mu.set_ylabel(r"$\mu$ (RMS mean)")
        ax_mu.set_title(r"Per-Window Mean $\mu(t)$")
        ax_mu.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax_mu.grid(False)
        _draw_vlines(ax_mu, vlines)
        if zoom_x is not None:
            ax_mu.set_xlim(zoom_x)
        fig_mu.tight_layout()
        # — Figure \u03c3(t) —
        fig_sig, ax_sig = plt.subplots(figsize=fig_size(scale=scale, ncols=1), num=fig_label_sigma)
        ax_sig.plot(cv_time_arr, sigma_arr, color=color_purple,
                    marker="o", markersize=3, linestyle="-")
        ax_sig.set_xlabel("Time (s)")
        ax_sig.set_ylabel(r"$\sigma$ (RMS std)")
        ax_sig.set_title(r"Per-Window Std $\sigma(t)$")
        ax_sig.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax_sig.grid(False)
        _draw_vlines(ax_sig, vlines)
        if zoom_x is not None:
            ax_sig.set_xlim(zoom_x)
        fig_sig.tight_layout()
        return (fig_mu, ax_mu), (fig_sig, ax_sig)

    # ── C4: Signal + CV joint (2 stacked subplots) ──────────────────────────
    def _plot_signal_cv_joint(
        t_sig: np.ndarray, x_sig: np.ndarray,
        cv_time_arr: np.ndarray, cv_arr: np.ndarray,
        t_gt_val: Optional[float] = None,
        cv_threshold: Optional[float] = None,
        cv_mu_stable: Optional[float] = None,
        cv_threshold_method: str = "fixed",
        cv_threshold_low: Optional[float] = None,
        zoom_x=None, scale: float = 1.0,
        vlines=None, fig_label: Optional[str] = None,
        **kargs,
    ) -> tuple:
        """Two stacked subplots (shared x-axis): signal (top) + CV scatter (bottom)."""
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=fig_size(scale=scale, ncols=1),
            sharex=True, constrained_layout=True, num=fig_label,
        )
        fig.suptitle("Signal + CV Joint Diagnostic")
        # Top: signal colored by region
        if t_gt_val is not None:
            mask_s = t_sig < t_gt_val
            mask_c = t_sig >= t_gt_val
            if np.any(mask_s):
                ax_top.plot(t_sig[mask_s], x_sig[mask_s], color=color_azul, label="Stable")
            if np.any(mask_c):
                ax_top.plot(t_sig[mask_c], x_sig[mask_c], color=color_orange, label="Chatter")
        else:
            ax_top.plot(t_sig, x_sig, color=color_azul)
        ax_top.set_ylabel(r"Velocity $v(t)$ [m/s]")
        ax_top.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax_top.legend()
        ax_top.grid(False)
        _draw_vlines(ax_top, vlines)
        # Bottom: CV scatter + threshold labels
        ax_bot.scatter(cv_time_arr, cv_arr, color=color_orange, marker="o", s=20)
        if cv_threshold is not None:
            ax_bot.axhline(cv_threshold, color=color_red, ls="--", lw=1.4)
            ax_bot.text(0.99, cv_threshold, rf"$\mu+3\sigma={cv_threshold:.4g}$",
                        transform=ax_bot.get_yaxis_transform(),
                        color=color_red, ha='right', va='bottom', fontsize=16)
            _is_adaptive = cv_threshold_method == "stable_region"
            if _is_adaptive and cv_threshold_low is not None and cv_threshold_low > 0:
                ax_bot.axhline(cv_threshold_low, color=color_red, ls=":", lw=1.2)
                ax_bot.text(0.99, cv_threshold_low,
                            rf"$\mu-3\sigma={cv_threshold_low:.4g}$",
                            transform=ax_bot.get_yaxis_transform(),
                            color=color_red, ha='right', va='top', fontsize=16)
            if cv_mu_stable is not None:
                ax_bot.axhline(cv_mu_stable, color=color_verde, ls="-", lw=1.0)
                ax_bot.text(0.99, cv_mu_stable,
                            rf"$\mu_{{stable}}={cv_mu_stable:.4g}$",
                            transform=ax_bot.get_yaxis_transform(),
                            color=color_verde, ha='right', va='bottom', fontsize=16)
        ax_bot.set_xlabel("Time (s)")
        ax_bot.set_ylabel("CV")
        ax_bot.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax_bot.grid(False)
        _draw_vlines(ax_bot, vlines)
        if zoom_x is not None:
            ax_bot.set_xlim(zoom_x)
        return fig, (ax_top, ax_bot)

    # ────────────────────────────────────────────────────────────────────────
    configurar_estilo_global()

    meta = result.meta or {}
    t = signal.t_analysis
    signal_analysis = signal.signal_analysis

    t_rms = meta.get("t_rms", None)
    rms_values = meta.get("rms_values", None)
    cv_time = meta.get("cv_time", None)
    cv_values = meta.get("cv_values", None)
    # prefer the computed threshold (adaptive or fixed) over the raw config value
    cv_threshold = meta.get("cv_threshold_used") or meta.get("cv_threshold", None)
    cv_threshold_method = meta.get("cv_threshold_method", "fixed")
    # lower bound and mean (only meaningful for adaptive threshold)
    _mu_stable  = meta.get("mu_stable", None)
    _cv_thr_low = (2.0 * _mu_stable - cv_threshold) if (_mu_stable is not None and cv_threshold is not None) else None
    rms_threshold = meta.get("rms_threshold", None)
    window_sec    = meta.get("window_sec", None)
    idx_rms_windows = meta.get("idx_rms_windows", None)
    times_rms_windows = None

    if idx_rms_windows is not None and t is not None:
        times_rms_windows = t[idx_rms_windows[:, 0]]

    cv_num_data = meta.get("n_max", None)

    # ── auto vlines — labeled tuples (value, label, color) ──────────────────
    _t_d = np.asarray(result.t_d) if result.t_d is not None and len(result.t_d) > 0 else np.array([])
    _t_first_det       = float(_t_d[0])              if _t_d.size > 0 else None
    _t_first_det_after = float(_t_d[_t_d > t_gt][0]) if (t_gt is not None and _t_d.size > 0 and np.any(_t_d > t_gt)) else None
    _avl = []
    if t_gt is not None:
        _avl.append((t_gt,              f"$t_{{gt}}={t_gt:.3f}$ s",            "black"))
    if _t_first_det is not None:
        _avl.append((_t_first_det,      f"$t_d={_t_first_det:.3f}$ s",         color_orange))
    if _t_first_det_after is not None and _t_first_det_after != _t_first_det:
        _avl.append((_t_first_det_after, f"$t_d^+={_t_first_det_after:.3f}$ s", color_orange))
    auto_vlines = _avl if _avl else None

    scale = 3.0

    # ── Original 3 figures ───────────────────────────────────────────────────
    fig_signal, axes_signal = _plot_signal(
        t, signal_analysis, zoom_x=zoom_x, zoom_y=zoom_y,
        title="Tool Velocity", scale=scale, vlines=auto_vlines,
    )
    fig_rms, axes_rms = _plot_rms(
        t_rms, rms_values, zoom_x=zoom_x, zoom_y=zoom_y,
        title="RMS Sequence", scale=scale, vlines=auto_vlines,
    )
    fig_cv, axes_cv = _plot_cv(
        cv_time, cv_values, cv_threshold,
        zoom_x=zoom_x, zoom_y=zoom_y, title="CV Sequence", scale=scale,
        cv_threshold_method=cv_threshold_method,
        cv_threshold_low=_cv_thr_low, cv_mu_stable=_mu_stable,
        vlines=auto_vlines, hlines=hlines,
    )

    # ── New figures C1–C4 ────────────────────────────────────────────────────
    if t_gt is not None:
        _plot_signal_split(
            t, signal_analysis, t_gt_val=t_gt,
            zoom_x=zoom_x, scale=scale, vlines=auto_vlines,
            fig_label="C1 — Signal Split by Region",
        )
        if t_rms is not None and rms_values is not None:
            _plot_rms_colored(
                t_rms, rms_values, t_gt_val=t_gt,
                cv_num_data=cv_num_data, zoom_x=zoom_x, scale=scale,
                vlines=auto_vlines, fig_label="C2 — RMS Colored by Region",
            )
        if cv_time is not None and cv_values is not None:
            _plot_cv_hist(
                cv_time, cv_values, t_gt_val=t_gt,
                cv_threshold=cv_threshold, cv_threshold_low=_cv_thr_low,
                mu_stable=_mu_stable, scale=scale,
                fig_label="C3 — CV Histogram",
            )
    if cv_time is not None and cv_values is not None:
        _plot_signal_cv_joint(
            t, signal_analysis, cv_time, cv_values,
            t_gt_val=t_gt, cv_threshold=cv_threshold,
            cv_mu_stable=_mu_stable,
            cv_threshold_method=cv_threshold_method,
            cv_threshold_low=_cv_thr_low,
            zoom_x=zoom_x, scale=scale, vlines=auto_vlines,
            fig_label="C4 — Signal + CV Joint",
        )

    _mu_arr    = np.asarray(meta.get("mu",    []))
    _sigma_arr = np.asarray(meta.get("sigma", []))
    if cv_time is not None and _mu_arr.size > 0 and _sigma_arr.size > 0:
        _plot_mu_sigma_evolution(
            np.asarray(cv_time), _mu_arr, _sigma_arr,
            zoom_x=zoom_x, scale=scale, vlines=auto_vlines,
            fig_label_mu="C5 \u2014 \u03bc per Window",
            fig_label_sigma="C6 \u2014 \u03c3 per Window",
        )

    plt.show(block=True)





