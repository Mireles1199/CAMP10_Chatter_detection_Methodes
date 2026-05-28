#%%
# ========= Imports =========
from __future__ import annotations
import matplotlib.pyplot as plt
import colorsys

from typing import Dict, Any, Sequence, Optional

import numpy as np
from scipy.stats import norm as _scipy_norm

from ..utils.types import IndicatorResult, SignalData

r, g, b = colorsys.hls_to_rgb(346/360, 0.45, 0.99)
color_red    = (r, g, b)   # alarm / upper threshold

r, g, b = colorsys.hls_to_rgb(36/360, 0.45, 0.99)
color_orange = (r, g, b)   # chatter signal / detection td

r, g, b = colorsys.hls_to_rgb(279/360, 0.36, 0.99)
color_purple = (r, g, b)   # SVD curve / auxiliary

r, g, b = colorsys.hls_to_rgb(98/360, 0.36, 0.99)
color_verde  = (r, g, b)   # stable threshold / mu_stable

r, g, b = colorsys.hls_to_rgb(206.957/360, 0.40941, 0.55603)
color_azul   = (r, g, b)   # stable signal / raw time series


def fig_size(scale=1.0, ncols=1, base_width=3.4):
    """
    scale: factor de escala (1 = tamaño normal)
    ncols: 1=single, 2=double, 3=triple
    base_width: ancho de una columna típica
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

def plots_sst_svd(
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

    def _plot_S(Sx: "np.ndarray", f: "float", t: float,
                  zoom_x: Optional[tuple[float, float]] = None,
                  zoom_y: Optional[tuple[float, float]] = None,
                  title: str = "STFT - Short Time Fourier Transform", scale: float = 1.0,
                  vlines: Optional[Sequence[float]] = None,
                  hlines: Optional[Sequence[float]] = None,
                  **kargs) -> tuple:
        t = t / 1000  # convertir ms a s
        Sx = abs(Sx)
        fig, axes = plt.subplots(figsize=fig_size(scale=scale, ncols=1))
        axes.pcolormesh(t, f, Sx, shading='gouraud', cmap='viridis')
        axes.set_title(title)
        axes.set_ylabel("Frequency (Hz)")
        axes.set_xlabel("Time (s)")
        axes.set_ylim(0, 250)
        if zoom_x is not None:
            axes.set_xlim(zoom_x)
        _draw_vlines(axes, vlines)
        if hlines is not None:
            for yv in hlines:
                if yv is not None:
                    axes.axhline(y=yv, color='gray', linestyle='--', lw=1, alpha=0.7)
        plt.tight_layout()
        return fig, axes

    def _plot_svd(times: "np.ndarray", d1: "np.ndarray",
                  zoom_x: Optional[tuple[float, float]] = None,
                  zoom_y: Optional[tuple[float, float]] = None, *,
                  title: str = "SVD 1st Component", scale: float = 1.0,
                  lim_sup: Optional[float] = None,
                  lim_inf: Optional[float] = None,
                  vlines: Optional[Sequence[float]] = None,
                  hlines: Optional[Sequence[float]] = None,
                  **kargs) -> tuple:
        fig, axes = plt.subplots(figsize=fig_size(scale=scale, ncols=1))
        axes.plot(times, d1, marker="o", markersize=4, linestyle="-", color=color_purple)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        axes.set_title(title)
        axes.set_xlabel("Time (s)")
        axes.set_ylabel("1st SVD Component")
        if lim_sup is not None:
            axes.axhline(y=lim_sup, color=color_red, linestyle="--", linewidth=1.4)
            axes.text(0.99, lim_sup, rf"$\mu + 3\sigma = {lim_sup:.4g}$",
                      transform=axes.get_yaxis_transform(),
                      color=color_red, ha='right', va='bottom', fontsize=16)
        if lim_inf is not None:
            axes.axhline(y=lim_inf, color=color_red, linestyle=":", linewidth=1.2)
            axes.text(0.99, lim_inf, rf"$\mu - 3\sigma = {lim_inf:.4g}$",
                      transform=axes.get_yaxis_transform(),
                      color=color_red, ha='right', va='top', fontsize=16)
        if zoom_x is not None:
            axes.set_xlim(zoom_x)
        _draw_vlines(axes, vlines)
        if hlines is not None:
            for yv in hlines:
                if yv is not None:
                    axes.axhline(y=yv, color='gray', linestyle='--', lw=1, alpha=0.7)
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

    # ── C2: SVD colored by region ────────────────────────────────────────────
    def _plot_svd_colored(
        t_svd: np.ndarray, d1_arr: np.ndarray, t_gt_val: float,
        lim_sup: Optional[float] = None,
        lim_inf: Optional[float] = None,
        zoom_x=None, scale: float = 1.0,
        vlines=None, fig_label: Optional[str] = None,
        **kargs,
    ) -> tuple:
        """SVD 1st component colored by region (stable=azul / chatter=orange) + threshold labels."""
        fig, ax = plt.subplots(figsize=fig_size(scale=scale, ncols=1), num=fig_label)
        mask_s = t_svd < t_gt_val
        mask_c = t_svd >= t_gt_val
        if np.any(mask_s):
            ax.plot(t_svd[mask_s], d1_arr[mask_s], color=color_azul, marker="o", markersize=4, linestyle="-", label="Stable")
        if np.any(mask_c):
            ax.plot(t_svd[mask_c], d1_arr[mask_c], color=color_orange, marker="o", markersize=4, linestyle="-", label="Chatter")
        if lim_sup is not None:
            ax.axhline(lim_sup, color=color_red, ls="--", lw=1.4)
            ax.text(0.99, lim_sup, rf"$\mu + 3\sigma = {lim_sup:.4g}$",
                    transform=ax.get_yaxis_transform(),
                    color=color_red, ha='right', va='bottom', fontsize=16)
        if lim_inf is not None:
            ax.axhline(lim_inf, color=color_red, ls=":", lw=1.2)
            ax.text(0.99, lim_inf, rf"$\mu - 3\sigma = {lim_inf:.4g}$",
                    transform=ax.get_yaxis_transform(),
                    color=color_red, ha='right', va='top', fontsize=16)
        if zoom_x is not None:
            ax.set_xlim(zoom_x)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        _draw_vlines(ax, vlines)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("1st SVD Component")
        ax.set_title("SVD 1st Component — Colored by Region")
        ax.legend()
        ax.grid(False)
        plt.tight_layout()
        return fig, ax

    # ── C3: SVD histogram stable vs chatter (log₁₀ scale) ──────────────────
    def _plot_svd_hist(
        t_svd: np.ndarray, d1_arr: np.ndarray, t_gt_val: float,
        lim_sup: Optional[float] = None,
        lim_inf: Optional[float] = None,
        scale: float = 1.0,
        fig_label: Optional[str] = None,
        **kargs,
    ) -> tuple:
        """Histogram of log10(SVD) values: stable (blue) vs chatter (orange) + Gaussian curves."""
        fig, ax = plt.subplots(figsize=fig_size(scale=scale, ncols=1), num=fig_label)
        d1_np = np.asarray(d1_arr, dtype=float)
        t_np  = np.asarray(t_svd,  dtype=float)
        # SVD singular values span many orders of magnitude → use log10 scale
        pos  = d1_np > 0
        d1_log = np.where(pos, np.log10(np.where(pos, d1_np, 1.0)), np.nan)
        mask_s = (t_np < t_gt_val) & pos
        mask_c = (t_np >= t_gt_val) & pos
        if np.any(mask_s):
            ax.hist(d1_log[mask_s], bins=40, density=True, alpha=0.55,
                    color=color_azul, label=f"Stable  (n={int(mask_s.sum())})")
            mu_s, std_s = np.mean(d1_log[mask_s]), np.std(d1_log[mask_s])
            if std_s > 0:
                xs = np.linspace(mu_s - 4 * std_s, mu_s + 4 * std_s, 300)
                ax.plot(xs, _scipy_norm.pdf(xs, mu_s, std_s),
                        color=color_azul, lw=1.8, ls="-")
                # mu line
                ax.axvline(mu_s, color=color_verde, ls="-", lw=1.4)
                ax.text(mu_s, 0.97, rf"  $\mu={mu_s:.3g}$",
                        rotation=90, va="top", ha="right", fontsize=14,
                        color=color_verde, transform=ax.get_xaxis_transform())
                # mu ± sigma lines
                ax.axvline(mu_s + std_s, color=color_verde, ls="--", lw=1.2)
                ax.text(mu_s + std_s, 0.97, rf"  $\mu+\sigma={mu_s+std_s:.3g}$",
                        rotation=90, va="top", ha="right", fontsize=14,
                        color=color_verde, transform=ax.get_xaxis_transform())
                ax.axvline(mu_s - std_s, color=color_verde, ls="--", lw=1.2)
                ax.text(mu_s - std_s, 0.97, rf"  $\mu-\sigma={mu_s-std_s:.3g}$",
                        rotation=90, va="top", ha="right", fontsize=14,
                        color=color_verde, transform=ax.get_xaxis_transform())
        if lim_sup is not None and lim_sup > 0:
            log_sup = np.log10(lim_sup)
            ax.axvline(log_sup, color=color_red, ls="--", lw=1.4)
            ax.text(log_sup, 0.97, rf"  $\mu+3\sigma={lim_sup:.4g}$",
                    rotation=90, va="top", ha="right", fontsize=16,
                    color=color_red, transform=ax.get_xaxis_transform())
        if lim_inf is not None and lim_inf > 0:
            log_inf = np.log10(lim_inf)
            ax.axvline(log_inf, color=color_red, ls=":", lw=1.2)
            ax.text(log_inf, 0.97, rf"  $\mu-3\sigma={lim_inf:.4g}$",
                    rotation=90, va="top", ha="right", fontsize=16,
                    color=color_red, transform=ax.get_xaxis_transform())
        ax.set_xlabel(r"$\log_{10}$(1st SVD Component)")
        ax.set_ylabel("Density")
        ax.set_title("SVD Distribution — Stable vs Chatter")
        ax.legend()
        ax.grid(False)
        fig.tight_layout()
        return fig, ax

    # ── C4: Signal + SVD joint (2 stacked subplots) ──────────────────────────
    def _plot_signal_svd_joint(
        t_sig: np.ndarray, x_sig: np.ndarray,
        t_svd: np.ndarray, d1_arr: np.ndarray,
        t_gt_val: Optional[float] = None,
        lim_sup: Optional[float] = None,
        lim_inf: Optional[float] = None,
        zoom_x=None, scale: float = 1.0,
        vlines=None, fig_label: Optional[str] = None,
        **kargs,
    ) -> tuple:
        """Two stacked subplots (shared x-axis): signal (top) + SVD line (bottom)."""
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=fig_size(scale=scale, ncols=1),
            sharex=True, constrained_layout=True, num=fig_label,
        )
        fig.suptitle("Signal + SVD Joint Diagnostic")
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
        # Bottom: SVD line + threshold labels
        ax_bot.plot(t_svd, d1_arr, color=color_purple, marker="o", markersize=4, linestyle="-")
        if lim_sup is not None:
            ax_bot.axhline(lim_sup, color=color_red, ls="--", lw=1.4)
            ax_bot.text(0.99, lim_sup, rf"$\mu+3\sigma={lim_sup:.4g}$",
                        transform=ax_bot.get_yaxis_transform(),
                        color=color_red, ha='right', va='bottom', fontsize=16)
        if lim_inf is not None:
            ax_bot.axhline(lim_inf, color=color_red, ls=":", lw=1.2)
            ax_bot.text(0.99, lim_inf, rf"$\mu-3\sigma={lim_inf:.4g}$",
                        transform=ax_bot.get_yaxis_transform(),
                        color=color_red, ha='right', va='top', fontsize=16)
        ax_bot.set_xlabel("Time (s)")
        ax_bot.set_ylabel("1st SVD Component")
        ax_bot.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax_bot.grid(False)
        _draw_vlines(ax_bot, vlines)
        if zoom_x is not None:
            ax_bot.set_xlim(zoom_x)
        return fig, (ax_top, ax_bot)

    # ────────────────────────────────────────────────────────────────────────
    meta = result.meta or {}
    t_sig_arr = signal.t_analysis
    sig_arr   = signal.signal_analysis
    fs        = signal.fs

    t_i = result.t
    d1  = result.I_t

    scale = 3.0

    Sx  = meta.get("Sx", None)
    Tsx = meta.get("Tsx", None)
    lim_sup = meta.get("lim_sup", None)
    lim_inf = meta.get("lim_inf", None)

    f   = np.linspace(0, fs / 2, Sx.shape[0])
    t_s = np.arange(Sx.shape[1]) * meta.get("hop_ms", 10e-3)

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

    configurar_estilo_global()

    # ── Original 3 figures ───────────────────────────────────────────────────
    fig_Sx, axes_Sx = _plot_S(
        Sx, f, t_s, zoom_x=zoom_x, zoom_y=zoom_y,
        title="STFT — Short Time Fourier Transform",
        scale=scale, vlines=auto_vlines,
    )
    fig_Tsx, axes_Tsx = _plot_S(
        Tsx, f, t_s, zoom_x=zoom_x, zoom_y=zoom_y,
        title="SST — Synchrosqueezing Transform",
        scale=scale, vlines=auto_vlines,
    )
    fig_svd, axes_svd = _plot_svd(
        t_i, d1, zoom_x=zoom_x, zoom_y=zoom_y,
        title="SVD — 1st Singular Value Component",
        scale=scale, vlines=auto_vlines,
        lim_sup=lim_sup, lim_inf=lim_inf,
    )

    # ── New figures C1–C4 ────────────────────────────────────────────────────
    if t_gt is not None:
        _plot_signal_split(
            t_sig_arr, sig_arr, t_gt_val=t_gt,
            zoom_x=zoom_x, scale=scale, vlines=auto_vlines,
            fig_label="C1 — Signal Split by Region",
        )
        if t_i is not None and d1 is not None:
            _plot_svd_colored(
                t_i, d1, t_gt_val=t_gt,
                lim_sup=lim_sup, lim_inf=lim_inf,
                zoom_x=zoom_x, scale=scale, vlines=auto_vlines,
                fig_label="C2 — SVD Colored by Region",
            )
            _plot_svd_hist(
                t_i, d1, t_gt_val=t_gt,
                lim_sup=lim_sup, lim_inf=lim_inf,
                scale=scale, fig_label="C3 — SVD Histogram",
            )
    if t_i is not None and d1 is not None:
        _plot_signal_svd_joint(
            t_sig_arr, sig_arr, t_i, d1,
            t_gt_val=t_gt, lim_sup=lim_sup, lim_inf=lim_inf,
            zoom_x=zoom_x, scale=scale, vlines=auto_vlines,
            fig_label="C4 — Signal + SVD Joint",
        )

    plt.show(block=True)