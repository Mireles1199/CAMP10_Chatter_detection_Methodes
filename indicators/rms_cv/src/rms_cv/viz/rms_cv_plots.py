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
                    color=color, transform=ax.get_xaxis_transform(), clip_on=True,
                )

    def _draw_block_boundaries(ax, times_arr, block_size, cap: int = 200):
        """Thin vertical markers every `block_size`-th frame (CV-window boundaries).
        ponytail: capped at `cap` lines -- skips drawing on long signals instead of
        rendering thousands of overlapping vlines; raise `cap` if you need them anyway.
        """
        if times_arr is None or not block_size or block_size <= 0:
            return
        idx = np.arange(0, len(times_arr), int(block_size))
        if idx.size == 0 or idx.size > cap:
            return
        for i in idx:
            ax.axvline(times_arr[i], color='gray', ls=':', lw=0.8, alpha=0.5)

    def _plot_rms(times: "np.ndarray", rms: "np.ndarray",
                  zoom_x: Optional[tuple[float, float]] = None,
                  zoom_y: Optional[tuple[float, float]] = None, *,
                  title: str = "RMS", scale: float = 1.0,
                  vlines: Optional[Sequence[float]] = None,
                  hlines: Optional[Sequence[float]] = None,
                  rms_threshold: Optional[float] = None,
                  block_times: Optional[np.ndarray] = None,
                  block_size: Optional[int] = None,
                  **kargs) -> tuple:
        fig, axes = plt.subplots(figsize=fig_size(scale=scale, ncols=1))
        axes.plot(times, rms, marker="o", color=color_azul)
        axes.set_xlabel("Time (s)")
        axes.set_ylabel("RMS")
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        if zoom_x is not None:
            axes.set_xlim(zoom_x)
        if zoom_y is not None:
            axes.set_ylim(zoom_y)
        _draw_block_boundaries(axes, block_times, block_size)
        _draw_vlines(axes, vlines)
        if rms_threshold is not None:
            axes.axhline(y=rms_threshold, color=color_red, linestyle="--", linewidth=1.2)
            axes.text(0.99, rms_threshold, rf"$RMS_{{thr}}={rms_threshold:.4g}$",
                      transform=axes.get_yaxis_transform(), clip_on=True,
                      color=color_red, ha='right', va='bottom', fontsize=14)
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
                 cv_threshold_low: Optional[float] = None,
                 cv_mu_stable: Optional[float] = None,
                 vlines: Optional[Sequence[float]] = None,
                 hlines: Optional[Sequence[float]] = None) -> tuple:
        fig, axes = plt.subplots(figsize=fig_size(scale=scale, ncols=1))
        axes.scatter(time_seq, cv_seq, color=color_orange, marker="o", s=30)
        if cv_threshold is not None:
            axes.axhline(y=cv_threshold, color=color_red, linestyle="--", linewidth=1.4)
            axes.text(0.99, cv_threshold,
                      rf"$\mu + 3\sigma = {cv_threshold:.4g}$",
                      transform=axes.get_yaxis_transform(), clip_on=True,
                      color=color_red, ha='right', va='bottom', fontsize=16)
            if cv_threshold_low is not None and cv_threshold_low > 0:
                axes.axhline(y=cv_threshold_low, color=color_red, linestyle=":", linewidth=1.2)
                axes.text(0.99, cv_threshold_low,
                          rf"$\mu - 3\sigma = {cv_threshold_low:.4g}$",
                          transform=axes.get_yaxis_transform(), clip_on=True,
                          color=color_red, ha='right', va='top', fontsize=16)
            if cv_mu_stable is not None:
                axes.axhline(y=cv_mu_stable, color=color_verde, linestyle="-", linewidth=1.0)
                axes.text(0.99, cv_mu_stable,
                          rf"$\mu_{{stable}} = {cv_mu_stable:.4g}$",
                          transform=axes.get_yaxis_transform(), clip_on=True,
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
                     block_times: Optional[np.ndarray] = None,
                     block_size: Optional[int] = None,
                     **kargs) -> tuple:
        fig, axes = plt.subplots(figsize=fig_size(scale=scale, ncols=1))
        axes.plot(t, x, color=color_azul)
        axes.set_xlabel("Time (s)")
        axes.set_ylabel(r"Velocity $v(t)$ [m/s]")
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        if zoom_x is not None:
            axes.set_xlim(zoom_x)
        if zoom_y is not None:
            axes.set_ylim(zoom_y)
        _draw_block_boundaries(axes, block_times, block_size)
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
        t_s: np.ndarray, x_s: np.ndarray, t_gt_val: Optional[float] = None,
        zoom_x=None, zoom_y=None, scale: float = 1.0,
        vlines=None, fig_label: Optional[str] = None,
        **kargs,
    ) -> tuple:
        """Signal colored by region: stable (azul) before t_gt, chatter (orange) after.
        Without t_gt_val there is no region to split by, so it plots as a single
        series with no color forced -- matplotlib's own default cycle, not a
        hardcoded "stable" color for data we don't actually know is stable."""
        fig, ax = plt.subplots(figsize=fig_size(scale=scale, ncols=1), num=fig_label)
        if t_gt_val is not None:
            mask_s = t_s < t_gt_val
            mask_c = t_s >= t_gt_val
            if np.any(mask_s):
                ax.plot(t_s[mask_s], x_s[mask_s], color=color_azul, label="Stable")
            if np.any(mask_c):
                ax.plot(t_s[mask_c], x_s[mask_c], color=color_orange, label="Chatter")
        else:
            ax.plot(t_s, x_s)
        if zoom_x is not None:
            ax.set_xlim(zoom_x)
        if zoom_y is not None:
            ax.set_ylim(zoom_y)
        _draw_vlines(ax, vlines)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"Velocity $v(t)$ [m/s]")
        ax.set_title("Tool Velocity — Split by Region")
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        if t_gt_val is not None:
            ax.legend()
        ax.grid(False)
        plt.tight_layout()
        return fig, ax

    # ── C2: RMS colored by region ───────────────────────────────────────────
    def _plot_rms_colored(
        t_rms_arr: np.ndarray, rms_arr: np.ndarray, t_gt_val: Optional[float] = None,
        cv_num_data: Optional[int] = None,
        zoom_x=None, scale: float = 1.0,
        vlines=None, rms_threshold: Optional[float] = None,
        fig_label: Optional[str] = None,
        **kargs,
    ) -> tuple:
        """RMS sequence colored by region + vertical CV-block boundaries every n_max frames.
        Without t_gt_val there is no region to split by, so it plots as a single
        series with no color forced -- matplotlib's own default cycle, not a
        hardcoded "stable" color for data we don't actually know is stable."""
        fig, ax = plt.subplots(figsize=fig_size(scale=scale, ncols=1), num=fig_label)
        if t_gt_val is not None:
            mask_s = t_rms_arr < t_gt_val
            mask_c = t_rms_arr >= t_gt_val
            if np.any(mask_s):
                ax.plot(t_rms_arr[mask_s], rms_arr[mask_s], marker="o", markersize=3,
                        color=color_azul, label="Stable RMS")
            if np.any(mask_c):
                ax.plot(t_rms_arr[mask_c], rms_arr[mask_c], marker="o", markersize=3,
                        color=color_orange, label="Chatter RMS")
        else:
            ax.plot(t_rms_arr, rms_arr, marker="o", markersize=3)
        _draw_block_boundaries(ax, t_rms_arr, cv_num_data)
        if zoom_x is not None:
            ax.set_xlim(zoom_x)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        _draw_vlines(ax, vlines)
        if rms_threshold is not None:
            ax.axhline(y=rms_threshold, color=color_red, ls="--", lw=1.2)
            ax.text(0.99, rms_threshold, rf"$RMS_{{thr}}={rms_threshold:.4g}$",
                    transform=ax.get_yaxis_transform(), clip_on=True,
                    color=color_red, ha='right', va='bottom', fontsize=14)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("RMS")
        ax.set_title("RMS Sequence — Colored by Region")
        if t_gt_val is not None:
            ax.legend()
        ax.grid(False)
        plt.tight_layout()
        return fig, ax

    # ── C3: CV histogram of the actual training population (internal or external
    #        reference, whichever was used to fit mu/sigma) ─────────────────────
    def _plot_cv_hist(
        training_values: Optional[np.ndarray],
        mu_stable: Optional[float] = None,
        sigma_stable: Optional[float] = None,
        cv_threshold: Optional[float] = None,
        cv_threshold_low: Optional[float] = None,
        training_source: str = "internal",
        cv_value_range: Optional[tuple[float, float]] = None,
        scale: float = 1.0,
        fig_label: Optional[str] = None,
        **kargs,
    ) -> tuple:
        """Histogram + fitted normal/MAD curve of the CV population that was actually
        used to compute mu_stable/sigma_stable (internal stable-region crop, or the
        full external reference_signal CV series) -- never recomputed from t_gt."""
        fig, ax = plt.subplots(figsize=fig_size(scale=scale, ncols=1), num=fig_label)
        if training_values is not None and training_values.size > 0:
            ax.hist(training_values, bins=40, density=True, alpha=0.55,
                    color=color_azul, label=f"Training ({training_source})")
            if mu_stable is not None and sigma_stable is not None and sigma_stable > 0:
                xs = np.linspace(mu_stable - 4 * sigma_stable, mu_stable + 4 * sigma_stable, 300)
                ax.plot(xs, _scipy_norm.pdf(xs, mu_stable, sigma_stable),
                        color=color_azul, lw=1.8, ls="-")
                ax.axvline(mu_stable, color=color_verde, ls="-", lw=1.4)
                ax.text(mu_stable, 0.97, rf"  $\mu={mu_stable:.3g}$",
                        rotation=90, va="top", ha="right", fontsize=14,
                        color=color_verde, transform=ax.get_xaxis_transform(), clip_on=True)
        if cv_threshold is not None:
            ax.axvline(cv_threshold, color=color_red, ls="--", lw=1.4)
            ax.text(cv_threshold, 0.97, rf"  $\mu+3\sigma={cv_threshold:.4g}$",
                    rotation=90, va="top", ha="right", fontsize=16,
                    color=color_red, transform=ax.get_xaxis_transform(), clip_on=True)
        if cv_threshold_low is not None:
            ax.axvline(cv_threshold_low, color=color_red, ls=":", lw=1.2)
            ax.text(cv_threshold_low, 0.97, rf"  $\mu-3\sigma={cv_threshold_low:.4g}$",
                    rotation=90, va="top", ha="right", fontsize=16,
                    color=color_red, transform=ax.get_xaxis_transform(), clip_on=True)

        ax.set_xlabel("CV")
        ax.set_ylabel("Density")
        ax.set_title("CV Distribution — Training Population")
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        if cv_value_range is not None:
            ax.set_xlim(cv_value_range)
        ax.legend()
        ax.grid(False)
        plt.tight_layout()
        return fig, ax

    # ── C8: raw training curve (verification of what fed the normal/MAD fit) ──
    def _plot_cv_training_curve(
        train_time: Optional[np.ndarray], train_values: np.ndarray,
        mu_stable: Optional[float] = None,
        cv_threshold: Optional[float] = None,
        zoom_x: Optional[tuple[float, float]] = None,
        zoom_y: Optional[tuple[float, float]] = None,
        scale: float = 1.0,
        fig_label: Optional[str] = None,
        **kargs,
    ) -> tuple:
        """Plot the exact CV sequence handed to CVStableRegionDetector -- lets you
        eyeball the population before trusting the mu/sigma (or median/MAD) fit.

        `zoom_x` only makes sense when `train_time` shares the analyzed signal's
        own clock (training_source == "internal"); the external-reference
        recording has its own unrelated timeline, so callers must not pass
        zoom_x for that case (this function does not guess -- it just applies
        whatever range it is given).
        """
        fig, ax = plt.subplots(figsize=fig_size(scale=scale, ncols=1), num=fig_label)
        x = train_time if train_time is not None else np.arange(train_values.size)
        ax.plot(x, train_values, marker="o", markersize=3, linestyle="-", color=color_azul)
        if mu_stable is not None:
            ax.axhline(mu_stable, color=color_verde, ls="-", lw=1.4)
            ax.text(0.99, mu_stable, rf"$\mu={mu_stable:.4g}$",
                    transform=ax.get_yaxis_transform(), clip_on=True,
                    color=color_verde, ha='right', va='bottom', fontsize=16)
        if cv_threshold is not None:
            ax.axhline(cv_threshold, color=color_red, ls="--", lw=1.4)
            ax.text(0.99, cv_threshold, rf"$\mu+3\sigma={cv_threshold:.4g}$",
                    transform=ax.get_yaxis_transform(), clip_on=True,
                    color=color_red, ha='right', va='bottom', fontsize=16)
        ax.set_xlabel("Time (s)" if train_time is not None else "Sample index (training population)")
        ax.set_ylabel("CV")
        ax.set_title("CV Training Curve")
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        if zoom_x is not None:
            ax.set_xlim(zoom_x)
        if zoom_y is not None:
            ax.set_ylim(zoom_y)
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
        ax_mu.set_title(r"Per-Window RMS Mean $\mu(t)$ — Online CV Monitor")
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
        ax_sig.set_title(r"Per-Window RMS Std $\sigma(t)$ — Online CV Monitor")
        ax_sig.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax_sig.grid(False)
        _draw_vlines(ax_sig, vlines)
        if zoom_x is not None:
            ax_sig.set_xlim(zoom_x)
        fig_sig.tight_layout()
        return (fig_mu, ax_mu), (fig_sig, ax_sig)

    # ── C7: μ and σ per window — combined single figure ─────────────────────
    def _plot_mu_sigma_combined(
        cv_time_arr: np.ndarray, mu_arr: np.ndarray, sigma_arr: np.ndarray,
        zoom_x=None, scale: float = 1.0,
        vlines=None, fig_label: Optional[str] = None,
        **kargs,
    ) -> tuple:
        """Single figure, shared Y axis: μ(t) (azul) and σ(t) (purple) per CV window."""
        fig, ax = plt.subplots(figsize=fig_size(scale=scale, ncols=1), num=fig_label)
        ax.plot(cv_time_arr, mu_arr, color=color_azul,
                marker="o", markersize=3, linestyle="-", label=r"$\mu(t)$")
        ax.plot(cv_time_arr, sigma_arr, color=color_purple,
                marker="s", markersize=3, linestyle="--", label=r"$\sigma(t)$")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"RMS statistics")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.set_title(r"Per-Window RMS Mean $\mu(t)$ and Std $\sigma(t)$ — Online CV Monitor")
        ax.legend(loc="upper left")
        ax.grid(False)
        _draw_vlines(ax, vlines)
        if zoom_x is not None:
            ax.set_xlim(zoom_x)
        fig.tight_layout()
        return fig, ax

    # ── C4: Signal + CV joint (2 stacked subplots) ──────────────────────────
    def _plot_signal_cv_joint(
        t_sig: np.ndarray, x_sig: np.ndarray,
        cv_time_arr: np.ndarray, cv_arr: np.ndarray,
        t_gt_val: Optional[float] = None,
        cv_threshold: Optional[float] = None,
        cv_mu_stable: Optional[float] = None,
        cv_threshold_low: Optional[float] = None,
        zoom_x=None, zoom_y=None, scale: float = 1.0,
        vlines=None, hlines=None, fig_label: Optional[str] = None,
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
            # No t_gt -> no known stable/chatter split; plot as a single series
            # with no color forced, instead of hardcoding the "stable" color for
            # data we don't actually know is stable.
            ax_top.plot(t_sig, x_sig)
        ax_top.set_ylabel(r"Velocity $v(t)$ [m/s]")
        ax_top.set_title("Signal — colored by region" if t_gt_val is not None else "Signal")
        ax_top.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        if t_gt_val is not None:
            ax_top.legend()
        ax_top.grid(False)
        _draw_vlines(ax_top, vlines)
        # Bottom: CV scatter + threshold labels
        ax_bot.set_title("CV")
        ax_bot.scatter(cv_time_arr, cv_arr, color=color_orange, marker="o", s=20)
        if cv_threshold is not None:
            ax_bot.axhline(cv_threshold, color=color_red, ls="--", lw=1.4)
            ax_bot.text(0.99, cv_threshold, rf"$\mu+3\sigma={cv_threshold:.4g}$",
                        transform=ax_bot.get_yaxis_transform(), clip_on=True,
                        color=color_red, ha='right', va='bottom', fontsize=16)
            if cv_threshold_low is not None and cv_threshold_low > 0:
                ax_bot.axhline(cv_threshold_low, color=color_red, ls=":", lw=1.2)
                ax_bot.text(0.99, cv_threshold_low,
                            rf"$\mu-3\sigma={cv_threshold_low:.4g}$",
                            transform=ax_bot.get_yaxis_transform(), clip_on=True,
                            color=color_red, ha='right', va='top', fontsize=16)
            if cv_mu_stable is not None:
                ax_bot.axhline(cv_mu_stable, color=color_verde, ls="-", lw=1.0)
                ax_bot.text(0.99, cv_mu_stable,
                            rf"$\mu_{{stable}}={cv_mu_stable:.4g}$",
                            transform=ax_bot.get_yaxis_transform(), clip_on=True,
                            color=color_verde, ha='right', va='bottom', fontsize=16)
        ax_bot.set_xlabel("Time (s)")
        ax_bot.set_ylabel("CV")
        ax_bot.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax_bot.grid(False)
        _draw_vlines(ax_bot, vlines)
        if hlines is not None:
            for yv in hlines:
                ax_bot.axhline(y=yv, color='gray', linestyle='--', lw=1, alpha=0.7)
        if zoom_x is not None:
            ax_bot.set_xlim(zoom_x)
        if zoom_y is not None:
            ax_bot.set_ylim(zoom_y)
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
    # Only two vertical lines are ever allowed on a detection-over-time panel:
    # (a) t_gt/t_theorical, and (b) the single first detection -- always the
    # raw t_d[0] (never t_d_no_FAR[0], which by construction always falls
    # after t_gt and would make this line redundant with it).
    # Never one line per detection, and never both t_d[0] AND t_d_no_FAR[0].
    _t_d = np.asarray(result.t_d) if result.t_d is not None and len(result.t_d) > 0 else np.array([])
    _t_first_detection = float(_t_d[0]) if _t_d.size > 0 else None
    _avl = []
    if t_gt is not None:
        _avl.append((t_gt,               f"$t_{{gt}}={t_gt:.3f}$ s", "black"))
    if _t_first_detection is not None:
        _avl.append((_t_first_detection, f"$t_d={_t_first_detection:.3f}$ s", color_orange))
    auto_vlines = _avl if _avl else None

    scale = 3.0

    # ── Original 3 figures ───────────────────────────────────────────────────
    # NB: zoom_y is documented/scoped to the CV panels only (values there share a
    # unit; velocity/RMS panels are on unrelated scales) -- only fed to _plot_cv
    # and the CV half of C4 below, never to _plot_signal/_plot_rms.
    fig_signal, axes_signal = _plot_signal(
        t, signal_analysis, zoom_x=zoom_x,
        title="Tool Velocity", scale=scale, vlines=auto_vlines,
        block_times=times_rms_windows, block_size=cv_num_data,
    )
    fig_rms, axes_rms = _plot_rms(
        t_rms, rms_values, zoom_x=zoom_x,
        title="RMS Sequence", scale=scale, vlines=auto_vlines,
        rms_threshold=rms_threshold, block_times=t_rms, block_size=cv_num_data,
    )
    fig_cv, axes_cv = _plot_cv(
        cv_time, cv_values, cv_threshold,
        zoom_x=zoom_x, zoom_y=zoom_y,
        title="CV Sequence", scale=scale,
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
                vlines=auto_vlines, rms_threshold=rms_threshold,
                fig_label="C2 — RMS Colored by Region",
            )
    _training_source = meta.get("training_source", "internal")
    _training_values = meta.get("cv_training_values")
    if _training_values is not None and np.asarray(_training_values).size > 0:
        _plot_cv_hist(
            np.asarray(_training_values),
            mu_stable=_mu_stable, sigma_stable=meta.get("sigma_stable"),
            cv_threshold=cv_threshold, cv_threshold_low=_cv_thr_low,
            training_source=_training_source, cv_value_range=zoom_y, scale=scale,
            fig_label="C3 — CV Histogram (Training Population)",
        )
    if cv_time is not None and cv_values is not None:
        _plot_signal_cv_joint(
            t, signal_analysis, cv_time, cv_values,
            t_gt_val=t_gt, cv_threshold=cv_threshold,
            cv_mu_stable=_mu_stable,
            cv_threshold_low=_cv_thr_low,
            zoom_x=zoom_x, zoom_y=zoom_y, scale=scale,
            vlines=auto_vlines, hlines=hlines,
            fig_label="C4 — Signal + CV Joint",
        )

    _mu_arr    = np.asarray(meta.get("mu",    []))
    _sigma_arr = np.asarray(meta.get("sigma", []))
    if cv_time is not None and _mu_arr.size > 0 and _sigma_arr.size > 0:
        _plot_mu_sigma_evolution(
            np.asarray(cv_time), _mu_arr, _sigma_arr,
            zoom_x=zoom_x, scale=scale, vlines=auto_vlines,
            fig_label_mu="C5 — μ per Window",
            fig_label_sigma="C6 — σ per Window",
        )
        _plot_mu_sigma_combined(
            np.asarray(cv_time), _mu_arr, _sigma_arr,
            zoom_x=zoom_x, scale=scale, vlines=auto_vlines,
            fig_label="C7 — μ and σ per Window (combined)",
        )

    if _training_values is not None and np.asarray(_training_values).size > 0:
        # zoom_x only makes sense here when the training curve shares the analyzed
        # signal's own clock (internal mode) -- the external reference recording
        # has an unrelated timeline, so its own zoom_x would silently clip it wrong.
        _c8_zoom_x = zoom_x if _training_source == "internal" else None
        _plot_cv_training_curve(
            meta.get("cv_training_time"), np.asarray(_training_values),
            mu_stable=_mu_stable, cv_threshold=cv_threshold,
            zoom_x=_c8_zoom_x, zoom_y=zoom_y,
            scale=scale, fig_label="C8 — CV Training Curve",
        )

    plt.show(block=True)





