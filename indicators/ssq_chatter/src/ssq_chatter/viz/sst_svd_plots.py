#%%
# ========= Imports =========
from __future__ import annotations
import matplotlib.pyplot as plt
import colorsys

from typing import Dict, Any, Sequence, Optional

import numpy as np

from ..utils.types import IndicatorResult, SignalData

import colorsys

#Red
H = 346/360
S = 0.99
L = 0.45

r, g, b = colorsys.hls_to_rgb(H, L, S)
color_rms = (r, g, b)

# Orange
H = 36/360   # de grados a [0,1]
S = 0.99
L = 0.45

r, g, b =  colorsys.hls_to_rgb(H, L, S)
color_CV = (r, g, b)

# Purple
H = 279/360   # de grados a [0,1]
S = 0.99
L = 0.36

r, g, b =  colorsys.hls_to_rgb(H, L, S)
color_purple = (r, g, b)


# Verde
H = 98/360   # de grados a [0,1]
S = 0.99
L = 0.36

r, g, b =  colorsys.hls_to_rgb(H, L, S)
color_verde = (r, g, b)


def fig_size(scale=1.0, ncols=1, base_width=3.4):
    """
    scale: factor de escala (1 = tamaño normal)
    ncols: 1=single, 2=double, 3=triple
    base_width: ancho de una columna típica
    """
    width = base_width * ncols * scale
    height = width * 0.7   # relación agradable
    return (width, height)

def configurar_estilo_global() -> None:
    """Configura el estilo global de los gráficos."""
    # plt.style.use('dark_background')

    local_style = {
        # Tipografía general
        'font.family': 'serif',
        'font.size': 16,

        # Tamaños de títulos y etiquetas
        'axes.titlesize': 18,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,

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

) -> plt.Figure:

    def _plot_S(Sx: "np.ndarray", f: "float", t: float,
                  zoom_x: Optional[tuple[float, float]] = None,
                  zoom_y: Optional[tuple[float, float]] = None,
                  title: str = "STFT - Short Time Fourier Transform", scale: float = 1.0,
                  vlines: Optional[Sequence[float]] = None,
                  hlines: Optional[Sequence[float]] = None,
                  **kargs) -> None:
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

        y_low, y_high = axes.get_ylim()
        x_low, x_high = axes.get_xlim()

        vlines_2 = kargs.get( "vlines_2", None)

        if vlines is not None:
            for xv in vlines:
                axes.axvline(x=xv, ymin=y_low, ymax=y_high, color="black", linestyle=":", linewidth=1.2)
        if hlines is not None:
            for yv in hlines:
                axes.axhline(y=yv, xmin=x_low, xmax=x_high, color="black", linestyle="--", linewidth=1.5)
        if vlines_2 is not None:
            color = ["orange", "red"]
            for i, xv in enumerate(vlines_2):
                axes.axvline(x=xv, ymin=y_low, ymax=y_high, color=color[i % len(color)], linestyle="--", linewidth=1.5,
                                alpha=1.0  )


        plt.tight_layout()

        return fig, axes

    def _plot_svd(times: "np.ndarray", d1: "np.ndarray",
                  zoom_x: Optional[tuple[float, float]] = None,
                  zoom_y: Optional[tuple[float, float]] = None,*,
                  title: str = "CV", scale: float = 1.0,
                  vlines: Optional[Sequence[float]] = None,
                  hlines: Optional[Sequence[float]] = None,
                  **kargs) -> None:

        fig, axes = plt.subplots(figsize=fig_size(scale=scale, ncols=1))
        axes.plot(times, d1, marker="o", color=color_purple)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        axes.set_title(title)
        axes.set_xlabel("Time (s)")
        axes.set_ylabel("1st SVD Component")

        if zoom_x is not None:
            axes.set_xlim(zoom_x)

        y_low, y_high = axes.get_ylim()
        x_low, x_high = axes.get_xlim()

        vlines_2 = kargs.get( "vlines_2", None)

        if vlines is not None:
            for xv in vlines:
                axes.axvline(x=xv, ymin=y_low, ymax=y_high, color="black", linestyle=":", linewidth=1.2)
        if hlines is not None:
            for yv in hlines:
                axes.axhline(y=yv, xmin=x_low, xmax=x_high, color="red", linestyle="--", linewidth=1.5)
        if vlines_2 is not None:
            color = ["orange", "red"]
            for i, xv in enumerate(vlines_2):
                axes.axvline(x=xv, ymin=y_low, ymax=y_high, color=color[i % len(color)], linestyle="--", linewidth=1.5,
                                alpha=1.0  )


        plt.tight_layout()

        return fig, axes

    meta = result.meta or {}
    t = signal.t_analysis
    signal_analysis = signal.signal_analysis
    fs = signal.fs

    t_i = result.t
    d1 = result.I_t

    scale= 3.0

    Sx = meta.get("Sx", None)
    Tsx = meta.get("Tsx", None)

    f = np.linspace(0, fs/2, Sx.shape[0])
    t = np.arange(Sx.shape[1]) * meta.get("hop_ms", 10e-3)
    kargs = {
        "vlines_2": vlines
    }

    configurar_estilo_global()

    fig_Sx, axes_Sx = _plot_S(Sx, f, t, zoom_x=zoom_x,
                              zoom_y=zoom_y, title="STFT - Short Time Fourier Transform", 
                              scale=scale,
                              vlines=vlines, hlines=None,
                              **kargs)

    fig_Sx, axes_Sx = _plot_S(Tsx, f, t, zoom_x=zoom_x,
                            zoom_y=zoom_y, title="SST - Synchrosqueezing Transform", 
                            scale=scale,
                            vlines=vlines, hlines=None,
                            **kargs)

    hilines = [meta.get('lim_sup', None), meta.get('lim_inf', None)]
    fig_svd, axes_svd =  _plot_svd(t_i, d1, zoom_x=zoom_x,
                                  zoom_y=zoom_y, title="SVD - Singular Value Decomposition - 1st Component",
                                  scale=scale,
                                  vlines=vlines, hlines=hilines,
                                  **kargs)


    plt.tight_layout()
    plt.show()