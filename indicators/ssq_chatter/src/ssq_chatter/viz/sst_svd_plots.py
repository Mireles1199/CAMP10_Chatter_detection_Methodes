#%%
# ========= Imports =========
from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.colors as _mcolors
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
    waterfall_lines: str = "time",   # "time" | "freq" | "both"
    training_intervals=None,

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
                  title: str = "STFT — Short Time Fourier Transform", scale: float = 1.0,
                  vlines: Optional[Sequence[float]] = None,
                  hlines: Optional[Sequence[float]] = None,
                  fig_label: Optional[str] = None,
                  **kargs) -> tuple:
        t = t / 1000  # convertir ms a s
        Sabs = np.abs(Sx)
        fig, axes = plt.subplots(figsize=fig_size(scale=scale, ncols=1), num=fig_label)
        mesh = axes.pcolormesh(t, f, Sabs, shading='gouraud', cmap='viridis')
        fig.colorbar(mesh, ax=axes, label="Amplitude")
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

    def _plot_freq_slice(
            Sx: "np.ndarray", f: "np.ndarray", t: "np.ndarray",
            freq_hz: float = 150.0,
            zoom_x: Optional[tuple] = None,
            title: Optional[str] = None,
            scale: float = 1.0,
            vlines: Optional[Sequence] = None,
            fig_label: Optional[str] = None,
            **kargs) -> tuple:
        """Corte horizontal del espectrograma a una frecuencia dada."""
        t_s = t / 1000  # ms → s
        Sabs = np.abs(Sx)                           # (n_freq, n_time)
        # índice de la fila más cercana a freq_hz
        f_arr = np.asarray(f)
        idx_f = int(np.argmin(np.abs(f_arr - freq_hz)))
        f_real = float(f_arr[idx_f])
        slice_amp = Sabs[idx_f, :]                  # (n_time,)
        _title = title or f"STFT — Slice at {f_real:.0f} Hz"
        fig, ax = plt.subplots(figsize=fig_size(scale=scale, ncols=1), num=fig_label)
        ax.plot(t_s, slice_amp, color=color_azul, linewidth=0.9)
        ax.set_title(_title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Amplitude  @ {f_real:.0f} Hz")
        ax.set_yscale("log")
        # ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.grid(False)
        if zoom_x is not None:
            ax.set_xlim(zoom_x)
        _draw_vlines(ax, vlines)
        fig.tight_layout()
        return fig, ax

    def _plot_waterfall_3d(
            Sx: "np.ndarray", f: "np.ndarray", t: "np.ndarray",
            f_max: float = 250.0,
            n_freq_pts: int = 150,
            n_time_pts: int = 120,
            lines: str = "surface",  # "surface" | "time" | "freq" | "both" | "wire"
            zoom_x: Optional[tuple] = None,
            title: str = "STFT \u2014 Cascade (Waterfall)",
            scale: float = 1.0,
            vlines: Optional[Sequence] = None,
            fig_label: Optional[str] = None,
            **kargs) -> tuple:
        """Cascade 3D del espectrograma STFT.

        Modo por defecto: lines="surface"  → superficie viridis + contorno al piso.
        Otros modos: "time", "freq", "both", "wire"
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        import matplotlib.cm as _cm
        t_s   = t / 1000
        f_arr = np.asarray(f)
        Sabs  = np.abs(Sx)

        # ── recorte de frecuencia ─────────────────────────────────────────
        f_mask = f_arr <= f_max
        f_plot = f_arr[f_mask]
        S_plot = Sabs[f_mask, :]

        # ── diezmado ──────────────────────────────────────────────────────
        nf, nt = S_plot.shape
        fi = np.linspace(0, nf - 1, min(n_freq_pts, nf), dtype=int)
        if zoom_x is not None:
            t_lo, t_hi = zoom_x
            valid = np.where((t_s >= t_lo) & (t_s <= t_hi))[0]
            ti = (valid[np.linspace(0, len(valid) - 1, min(n_time_pts, len(valid)),
                                    dtype=int)] if valid.size > 1
                  else np.linspace(0, nt - 1, min(n_time_pts, nt), dtype=int))
        else:
            ti = np.linspace(0, nt - 1, min(n_time_pts, nt), dtype=int)

        f_d = f_plot[fi]
        t_d = t_s[ti]
        S_d = S_plot[np.ix_(fi, ti)]          # (n_freq_pts, n_time_pts)
        F_mesh, T_mesh = np.meshgrid(f_d, t_d, indexing='ij')
        s_min, s_max = float(S_d.min()), float(S_d.max())

        # figura — 3D necesita más espacio que los plots 2D
        _w, _h = fig_size(scale=scale, ncols=1)
        fig = plt.figure(figsize=(_w * 1.0, _h * 1.0), num=fig_label)
        ax  = fig.add_subplot(111, projection='3d')

        # ══ SURFACE (modo principal) ══════════════════════════════════════
        if lines == "surface":
            norm_s = _mcolors.Normalize(vmin=s_min, vmax=s_max)
            surf = ax.plot_surface(
                F_mesh, T_mesh, S_d,
                cmap='viridis', norm=norm_s,
                rcount=n_freq_pts, ccount=n_time_pts,
                linewidth=0, antialiased=True, alpha=1.0)
            # contorno proyectado al piso (profundidad visual)
            ax.contourf(F_mesh, T_mesh, S_d,
                        zdir='z', offset=0,
                        cmap='viridis', norm=norm_s,
                        levels=20, alpha=0.28)
            cb = fig.colorbar(surf, ax=ax, shrink=0.50, pad=0.10,
                              label="Amplitude")
            cb.ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        # ══ LÍNEAS POR TIEMPO ════════════════════════════════════════════
        elif lines in ("time", "both"):
            cmap_v = _cm.get_cmap('viridis')
            amp_t  = S_d.max(axis=0)
            lo, hi = float(amp_t.min()), float(amp_t.max())
            amp_tn = (amp_t - lo) / max(hi - lo, 1e-30)
            lw = 1.1 if lines == "time" else 0.60
            al = 0.93 if lines == "time" else 0.78
            for k in np.argsort(amp_t):
                ax.plot(f_d, np.full_like(f_d, t_d[k]), S_d[:, k],
                        color=cmap_v(amp_tn[k]), linewidth=lw, alpha=al,
                        solid_capstyle='round', solid_joinstyle='round')
            if lines == "both":
                amp_f  = S_d.max(axis=1)
                lof, hif = float(amp_f.min()), float(amp_f.max())
                amp_fn = (amp_f - lof) / max(hif - lof, 1e-30)
                for i in np.argsort(amp_f):
                    ax.plot(np.full_like(t_d, f_d[i]), t_d, S_d[i, :],
                            color=cmap_v(amp_fn[i]), linewidth=0.48,
                            alpha=0.65, solid_capstyle='round')
            sm = plt.cm.ScalarMappable(cmap='viridis',
                                       norm=_mcolors.Normalize(vmin=lo, vmax=hi))
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, shrink=0.50, pad=0.10, label="Amplitude")
            cb.ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        # ══ LÍNEAS POR FRECUENCIA ════════════════════════════════════════
        elif lines == "freq":
            cmap_v = _cm.get_cmap('viridis')
            amp_f  = S_d.max(axis=1)
            lo, hi = float(amp_f.min()), float(amp_f.max())
            amp_fn = (amp_f - lo) / max(hi - lo, 1e-30)
            for i in np.argsort(amp_f):
                ax.plot(np.full_like(t_d, f_d[i]), t_d, S_d[i, :],
                        color=cmap_v(amp_fn[i]), linewidth=1.1, alpha=0.93,
                        solid_capstyle='round', solid_joinstyle='round')
            sm = plt.cm.ScalarMappable(cmap='viridis',
                                       norm=_mcolors.Normalize(vmin=lo, vmax=hi))
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, shrink=0.50, pad=0.10, label="Amplitude")
            cb.ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        # ══ WIREFRAME ════════════════════════════════════════════════════
        elif lines == "wire":
            ax.plot_wireframe(F_mesh, T_mesh, S_d,
                              rcount=50, ccount=50,
                              color=color_azul, linewidth=0.45, alpha=0.80)
            sm = plt.cm.ScalarMappable(cmap='viridis',
                                       norm=_mcolors.Normalize(vmin=s_min, vmax=s_max))
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, shrink=0.50, pad=0.10, label="Amplitude")
            cb.ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        # ── vlines (plano al nivel del suelo) ─────────────────────────────
        if vlines is not None:
            for v in vlines:
                tv = float(v[0]) if hasattr(v, '__len__') else float(v)
                ax.plot([f_d[0], f_d[-1]], [tv, tv], [0, 0],
                        color='dimgray', linewidth=1.2, alpha=0.85,
                        linestyle='--', zorder=5)

        # ── cosmética 3D ──────────────────────────────────────────────────
        ax.grid(False)
        for _pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            _pane.fill = False
            _pane.set_edgecolor('none')
        ax.xaxis.pane.set_edgecolor('#cccccc')
        ax.yaxis.pane.set_edgecolor('#cccccc')
        ax.xaxis.set_rotate_label(True)
        ax.yaxis.set_rotate_label(True)
        ax.view_init(elev=28, azim=-50)

        ax.set_xlabel("Frequency (Hz)", labelpad=16)
        ax.set_ylabel("Time (s)",        labelpad=16)
        ax.set_zlabel("")                            # cubierto por la colorbar
        ax.ticklabel_format(style="sci", axis="z", scilimits=(0, 0))
        fig.suptitle(title, y=1.0)                  # título fuera del área 3D
        fig.tight_layout()
        return fig, ax

    def _plot_svd(times: "np.ndarray", d1: "np.ndarray",
                  zoom_x: Optional[tuple[float, float]] = None,
                  zoom_y: Optional[tuple[float, float]] = None, *,
                  title: str = "SVD 1st Component", scale: float = 1.0,
                  lim_sup: Optional[float] = None,
                  lim_inf: Optional[float] = None,
                  vlines: Optional[Sequence[float]] = None,
                  hlines: Optional[Sequence[float]] = None,
                  fig_label: Optional[str] = None,
                  **kargs) -> tuple:
        fig, axes = plt.subplots(figsize=fig_size(scale=scale, ncols=1), num=fig_label)
        axes.plot(times, d1, marker="o", markersize=4, linestyle="-", color=color_purple)
        axes.set_yscale('log')
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
        ax.set_yscale('log')
        _draw_vlines(ax, vlines)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("1st SVD Component")
        ax.set_title("SVD 1st Component — Colored by Region")
        ax.legend()
        ax.grid(False)
        plt.tight_layout()
        return fig, ax

    # ── C2b: SVD time-series — one stable label highlighted (log Y) ─────────
    def _plot_svd_colored_one_segment(
        t_svd: np.ndarray, d1_arr: np.ndarray,
        seg_ranges: list, seg_label: str,
        all_stable_ranges: list,
        lim_sup: Optional[float] = None,
        lim_inf: Optional[float] = None,
        zoom_x=None, scale: float = 1.0,
        vlines=None, fig_label: Optional[str] = None,
        **kargs,
    ) -> tuple:
        """d1(t): all-stable faded (azul) + one label highlighted. Log Y + threshold lines."""
        fig, ax = plt.subplots(figsize=fig_size(scale=scale, ncols=1), num=fig_label)
        t_np  = np.asarray(t_svd,  dtype=float)
        d1_np = np.asarray(d1_arr, dtype=float)
        # Background: each stable interval plotted separately → no connecting line across gaps
        for _bi, (_t0, _t1) in enumerate(all_stable_ranges):
            _m = (t_np >= _t0) & (t_np <= _t1)
            if _m.any():
                ax.plot(t_np[_m], d1_np[_m],
                        color=color_azul, alpha=0.25, lw=0.8,
                        marker="o", markersize=2, linestyle="-",
                        label="Stable" if _bi == 0 else "_nolegend_")
        # Highlighted: each interval of the label plotted separately
        for _si, (_t0, _t1) in enumerate(seg_ranges):
            _m = (t_np >= _t0) & (t_np <= _t1)
            if _m.any():
                ax.plot(t_np[_m], d1_np[_m],
                        color=color_azul, alpha=1.0, lw=1.6,
                        marker="o", markersize=3, linestyle="-",
                        label=seg_label if _si == 0 else "_nolegend_")
        # Threshold lines
        if lim_sup is not None:
            ax.axhline(lim_sup, color=color_red, ls="--", lw=1.4)
            ax.text(0.99, lim_sup, rf"$\mu+3\sigma={lim_sup:.4g}$",
                    transform=ax.get_yaxis_transform(),
                    color=color_red, ha='right', va='bottom', fontsize=16)
        if lim_inf is not None:
            ax.axhline(lim_inf, color=color_red, ls=":", lw=1.2)
            ax.text(0.99, lim_inf, rf"$\mu-3\sigma={lim_inf:.4g}$",
                    transform=ax.get_yaxis_transform(),
                    color=color_red, ha='right', va='top', fontsize=16)
        ax.set_yscale('log')
        if zoom_x is not None:
            ax.set_xlim(zoom_x)
        # _draw_vlines(ax, vlines)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("1st SVD Component")
        ax.set_title(rf"SVD — Stable | {seg_label}")
        ax.legend(loc="lower left")
        ax.grid(False)
        plt.tight_layout()
        return fig, ax

    # ── C3: SVD histogram stable vs chatter (log₁₀ scale) ──────────────────
    def _plot_svd_hist(
        t_svd: np.ndarray, d1_arr: np.ndarray, t_gt_val: float,
        lim_sup: Optional[float] = None,
        lim_inf: Optional[float] = None,
        all_stable_ranges=None,
        scale: float = 1.0,
        fig_label: Optional[str] = None,
        **kargs,
    ) -> tuple:
        """Histogram of log10(SVD) values: stable (blue) + Gaussian curves."""
        fig, ax = plt.subplots(figsize=fig_size(scale=scale, ncols=1), num=fig_label)
        d1_np = np.asarray(d1_arr, dtype=float)
        t_np  = np.asarray(t_svd,  dtype=float)
        # SVD singular values span many orders of magnitude → use log10 scale
        pos  = d1_np > 0
        d1_log = np.where(pos, np.log10(np.where(pos, d1_np, 1.0)), np.nan)
        # Stable mask: union of training intervals labeled stable*, or fallback t < t_gt
        if all_stable_ranges is not None:
            mask_s = np.zeros(len(t_np), dtype=bool)
            for _t0, _t1 in all_stable_ranges:
                mask_s |= (t_np >= _t0) & (t_np <= _t1)
            mask_s &= pos
        else:
            mask_s = (t_np < t_gt_val) & pos
        mask_c = (t_np >= t_gt_val) & pos
        if np.any(mask_s):
            ax.hist(d1_log[mask_s], bins=40, density=True, alpha=0.55,
                    color=color_azul, label=f"Stable")
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
        if lim_sup is not None and lim_sup > 0:
            log_sup = np.log10(lim_sup)
            ax.axvline(log_sup, color=color_red, ls="--", lw=1.4)
            ax.text(log_sup, 0.97, rf"  $\mu+3\sigma={lim_sup:.4g}$",
                    rotation=90, va="top", ha="right", fontsize=16,
                    color=color_red, transform=ax.get_xaxis_transform())
        if lim_inf is not None:
            log_inf = np.log10(lim_inf)
            ax.axvline(log_inf, color=color_red, ls=":", lw=1.2)
            ax.text(log_inf, 0.97, rf"  $\mu-3\sigma={lim_inf:.4g}$",
                    rotation=90, va="top", ha="right", fontsize=16,
                    color=color_red, transform=ax.get_xaxis_transform())
        ax.set_xlabel(r"$\log_{10}$(1st SVD Component)")
        ax.set_ylabel("Density")
        ax.set_title("SVD Distribution — Stable")
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

    # ── F3b: d1 time series — one stable label highlighted ──────────────────
    def _plot_d1_time_one_segment(
        t_svd: np.ndarray, d1_arr: np.ndarray,
        seg_ranges: list, seg_label: str,
        zoom_x=None, scale: float = 1.0,
        vlines=None, fig_label: Optional[str] = None,
        **kargs,
    ) -> tuple:
        """d1(t): full trace faded, one stable label group highlighted."""
        fig, ax = plt.subplots(figsize=fig_size(scale=scale, ncols=1), num=fig_label)
        t_np  = np.asarray(t_svd, dtype=float)
        d1_np = np.asarray(d1_arr, dtype=float)
        ax.plot(t_np, d1_np, color=color_azul, alpha=0.25, lw=0.8, label="All d1")
        _ranges_str = ", ".join(
            rf"$[{_t0:.2f},\,{_t1:.2f}]$" for _t0, _t1 in seg_ranges
        )
        for _ri, (_t0, _t1) in enumerate(seg_ranges):
            _m = (t_np >= _t0) & (t_np <= _t1)
            if not _m.any():
                continue
            _lbl_entry = rf"{seg_label}  {_ranges_str} s" if _ri == 0 else "_nolegend_"
            ax.plot(t_np[_m], d1_np[_m], color=color_azul, alpha=1.0, lw=1.6,
                    marker='o', markersize=2, label=_lbl_entry)
        if zoom_x is not None:
            ax.set_xlim(zoom_x)
        _draw_vlines(ax, vlines)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("1st SVD Component")
        ax.set_title(rf"Training d1 — Stable Segments | {seg_label}")
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.legend()
        ax.grid(False)
        plt.tight_layout()
        return fig, ax

    # ── C3b: log10(d1) histogram — one stable label highlighted + Gaussian ──
    def _plot_d1_pdf_one_segment(
        t_svd: np.ndarray, d1_arr: np.ndarray,
        seg_ranges: list, seg_label: str,
        all_stable_ranges: list,
        lim_sup: Optional[float] = None,
        lim_inf: Optional[float] = None,
        scale: float = 1.0,
        fig_label: Optional[str] = None,
        **kargs,
    ) -> tuple:
        """log10(d1): C3 stable background (faded) + label contribution highlighted + μ/3σ lines."""
        fig, ax = plt.subplots(figsize=fig_size(scale=scale, ncols=1), num=fig_label)
        d1_np  = np.asarray(d1_arr, dtype=float)
        t_np   = np.asarray(t_svd,  dtype=float)
        pos    = d1_np > 0
        d1_log = np.where(pos, np.log10(np.where(pos, d1_np, 1.0)), np.nan)
        # stable mask — same as C3: union of all stable training intervals
        mask_s = np.zeros(len(t_np), dtype=bool)
        for _t0, _t1 in all_stable_ranges:
            mask_s |= (t_np >= _t0) & (t_np <= _t1)
        mask_s &= pos & np.isfinite(d1_log)
        if not mask_s.any():
            fig.tight_layout()
            return fig, ax
        # shared bin edges from all stable data (same as C3 auto-bins on mask_s)
        n_bins = 40
        _, bin_edges = np.histogram(d1_log[mask_s], bins=n_bins)
        n_stable = int(mask_s.sum())
        widths   = np.diff(bin_edges)
        # ── Background: all stable histogram (identical to C3) ──────────────────────
        cnt_all, _ = np.histogram(d1_log[mask_s], bins=bin_edges)
        ax.bar(bin_edges[:-1], cnt_all / (n_stable * widths), width=widths,
               color=color_azul, alpha=0.35, align="edge", label="Stable")
        # ── Stable Gaussian (same as C3) ────────────────────────────────
        mu_s  = float(np.mean(d1_log[mask_s]))
        std_s = float(np.std(d1_log[mask_s]))
        if std_s > 0:
            xs = np.linspace(mu_s - 4 * std_s, mu_s + 4 * std_s, 300)
            ax.plot(xs, _scipy_norm.pdf(xs, mu_s, std_s),
                    color=color_azul, lw=1.8, ls="-")
        # ── Highlighted label contribution ─────────────────────────────────
        seg_mask = np.zeros(len(t_np), dtype=bool)
        for _t0, _t1 in seg_ranges:
            seg_mask |= (t_np >= _t0) & (t_np <= _t1)
        seg_valid = seg_mask & pos & np.isfinite(d1_log)
        if seg_valid.sum() >= 2:
            _ranges_str = ", ".join(
                rf"$[{_t0:.2f},\,{_t1:.2f}]$" for _t0, _t1 in seg_ranges
            )
            cnt_seg, _ = np.histogram(d1_log[seg_valid], bins=bin_edges)
            ax.bar(bin_edges[:-1], cnt_seg / (n_stable * widths), width=widths,
                   color=color_azul, alpha=0.85, align="edge",
                   label=rf"{seg_label}  {_ranges_str} s")
        # ── μ and μ±3σ lines ─────────────────────────────────────────────
        ax.axvline(mu_s, color=color_verde, ls="-", lw=1.4)
        ax.text(mu_s, 0.97, rf"  $\mu={mu_s:.3g}$",
                rotation=90, va="top", ha="right",
                color=color_verde, transform=ax.get_xaxis_transform(), fontsize=14)
        # μ+3σ: use runner threshold (log-converted) if available, else histogram stats
        if lim_sup is not None and lim_sup > 0:
            _sup_pos = np.log10(lim_sup)
            _sup_lbl = rf"  $\mu+3\sigma={lim_sup:.4g}$"
        elif std_s > 0:
            _sup_pos = mu_s + 3 * std_s
            _sup_lbl = rf"  $\mu+3\sigma={_sup_pos:.3g}$"
        else:
            _sup_pos = None
        if _sup_pos is not None:
            ax.axvline(_sup_pos, color=color_red, ls="--", lw=1.4)
            ax.text(_sup_pos, 0.97, _sup_lbl,
                    rotation=90, va="top", ha="right",
                    color=color_red, transform=ax.get_xaxis_transform(), fontsize=16)
        # μ-3σ: use runner threshold (log-converted) if available, else histogram stats
        if lim_inf is not None and lim_inf > 0:
            _inf_pos = np.log10(lim_inf)
            _inf_lbl = rf"  $\mu-3\sigma={lim_inf:.4g}$"
        elif std_s > 0:
            _inf_pos = mu_s - 3 * std_s
            _inf_lbl = rf"  $\mu-3\sigma={_inf_pos:.3g}$"
        else:
            _inf_pos = None
        if _inf_pos is not None:
            ax.axvline(_inf_pos, color=color_red, ls=":", lw=1.2)
            ax.text(_inf_pos, 0.97, _inf_lbl,
                    rotation=90, va="top", ha="right",
                    color=color_red, transform=ax.get_xaxis_transform(), fontsize=16)
        ax.set_xlabel(r"$\log_{10}$(1st SVD Component)")
        ax.set_ylabel("Density")
        ax.set_title(rf"Training PDF — Stable d1 | {seg_label}")
        ax.legend(loc='lower left')
        ax.grid(False)
        fig.tight_layout()
        return fig, ax

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

    _ti_meta = training_intervals if training_intervals is not None else meta.get("training_intervals", None)

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
        fig_label="F1 — STFT Spectrogram",
    )
    _plot_freq_slice(
        Sx, f, t_s, freq_hz=150.0,
        zoom_x=zoom_x, scale=scale, vlines=auto_vlines,
        fig_label="F1b — Slice at 150 Hz",
    )
    _plot_waterfall_3d(
        Sx, f, t_s, f_max=250.0,
        lines=waterfall_lines,
        zoom_x=zoom_x, scale=scale, vlines=auto_vlines,
        fig_label="F1c — Waterfall 3D",
    )
    fig_Tsx, axes_Tsx = _plot_S(
        Tsx, f, t_s, zoom_x=zoom_x, zoom_y=zoom_y,
        title="SST — Synchrosqueezing Transform",
        scale=scale, vlines=auto_vlines,
        fig_label="F2 — SST Spectrogram",
    )
    _plot_freq_slice(
        Tsx, f, t_s, freq_hz=150.0,
        title="SST — Slice at 150 Hz",
        zoom_x=zoom_x, scale=scale, vlines=auto_vlines,
        fig_label="F2b — SST Slice at 150 Hz",
    )
    _plot_waterfall_3d(
        Tsx, f, t_s, f_max=250.0,
        title="SST — Cascade (Waterfall)",
        lines=waterfall_lines,
        zoom_x=zoom_x, scale=scale, vlines=auto_vlines,
        fig_label="F2c — SST Waterfall 3D",
    )
    fig_svd, axes_svd = _plot_svd(
        t_i, d1, zoom_x=zoom_x, zoom_y=zoom_y,
        title="SVD — 1st Singular Value Component",
        scale=scale, vlines=auto_vlines,
        lim_sup=lim_sup, lim_inf=lim_inf,
        fig_label="F3 — SVD 1st Component",
    )

    # F3b — one figure per distinct stable label (only if ≥2 distinct stable labels)
    if _ti_meta is not None and d1 is not None:
        _stable_grps_t: dict = {}
        for _t0, _t1, _lbl in _ti_meta:
            _lbl_lo = str(_lbl).lower().strip()
            if _lbl_lo.startswith("stable"):
                _stable_grps_t.setdefault(_lbl_lo, []).append((_t0, _t1))
        if len(_stable_grps_t) >= 2:
            _t_i_np = np.asarray(t_i, dtype=float)
            _d1_np  = np.asarray(d1,  dtype=float)
            for _gi, (_lbl_name, _ranges) in enumerate(_stable_grps_t.items()):
                _plot_d1_time_one_segment(
                    _t_i_np, _d1_np,
                    seg_ranges=_ranges,
                    seg_label=_lbl_name,
                    zoom_x=zoom_x, scale=scale, vlines=auto_vlines,
                    fig_label=f"F3b.{_gi} — {_lbl_name}",
                )

    # ── New figures C1–C4 ────────────────────────────────────────────────────
    if t_gt is not None:
        _plot_signal_split(
            t_sig_arr, sig_arr, t_gt_val=t_gt,
            zoom_x=zoom_x, scale=scale, vlines=auto_vlines,
            fig_label="C1 — Signal Split by Region",
        )
        if t_i is not None and d1 is not None:
            # precompute stable ranges (shared by C2b, C3, C3b)
            _all_stable_ranges: list = []
            _stable_grps: dict = {}
            if _ti_meta is not None:
                for _t0, _t1, _lbl in _ti_meta:
                    _lbl_lo = str(_lbl).lower().strip()
                    if _lbl_lo.startswith("stable"):
                        _all_stable_ranges.append((_t0, _t1))
                        _stable_grps.setdefault(_lbl_lo, []).append((_t0, _t1))
            _plot_svd_colored(
                t_i, d1, t_gt_val=t_gt,
                lim_sup=lim_sup, lim_inf=lim_inf,
                zoom_x=zoom_x, scale=scale, vlines=auto_vlines,
                fig_label="C2 — SVD Colored by Region",
            )
            # C2b — one figure per distinct stable label (same philosophy as C3b)
            if _all_stable_ranges and len(_stable_grps) >= 2:
                for _gi, (_lbl_name, _ranges) in enumerate(_stable_grps.items()):
                    _plot_svd_colored_one_segment(
                        np.asarray(t_i, dtype=float), np.asarray(d1, dtype=float),
                        seg_ranges=_ranges,
                        seg_label=_lbl_name,
                        all_stable_ranges=_all_stable_ranges,
                        lim_sup=lim_sup, lim_inf=lim_inf,
                        zoom_x=zoom_x, scale=scale, vlines=auto_vlines,
                        fig_label=f"C2b.{_gi} — {_lbl_name}",
                    )
            _plot_svd_hist(
                t_i, d1, t_gt_val=t_gt,
                lim_sup=lim_sup, lim_inf=lim_inf,
                all_stable_ranges=_all_stable_ranges if _all_stable_ranges else None,
                scale=scale, fig_label="C3 — SVD Histogram",
            )
            # C3b — one figure per distinct stable label
            if _all_stable_ranges and len(_stable_grps) >= 2:
                for _gi, (_lbl_name, _ranges) in enumerate(_stable_grps.items()):
                    _plot_d1_pdf_one_segment(
                        np.asarray(t_i, dtype=float), np.asarray(d1, dtype=float),
                        seg_ranges=_ranges,
                        seg_label=_lbl_name,
                        all_stable_ranges=_all_stable_ranges,
                        lim_sup=lim_sup, lim_inf=lim_inf,
                        scale=scale,
                        fig_label=f"C3b.{_gi} — {_lbl_name}",
                    )
    if t_i is not None and d1 is not None:
        _plot_signal_svd_joint(
            t_sig_arr, sig_arr, t_i, d1,
            t_gt_val=t_gt, lim_sup=lim_sup, lim_inf=lim_inf,
            zoom_x=zoom_x, scale=scale, vlines=auto_vlines,
            fig_label="C4 — Signal + SVD Joint",
        )

    plt.show(block=True)