
#%%
# ========= Imports =========
from __future__ import annotations
# import matplotlib
# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

import colorsys

from typing import Dict, Any, Sequence, Optional

import numpy as np

from ..utils.types import IndicatorResult, SignalData

import colorsys

H = 346/360
S = 0.99
L = 0.45

r, g, b = colorsys.hls_to_rgb(H, L, S)
color_rms = (r, g, b)


H = 36/360   # de grados a [0,1]
S = 0.99
L = 0.45

r, g, b =  colorsys.hls_to_rgb(H, L, S)
color_CV = (r, g, b)



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
        'font.size': 9,

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

) -> plt.Figure:

    def _plot_rms(times: "np.ndarray", rms: "np.ndarray",
                  zoom_x: Optional[tuple[float, float]] = None,
                  zoom_y: Optional[tuple[float, float]] = None,*,
                  title: str = "RMS", scale: float = 1.0,
                  vlines: Optional[Sequence[float]] = None,
                  hlines: Optional[Sequence[float]] = None,
                  **kargs) -> None:
        # Comentario: traza secuencia RMS
        fig, axes = plt.subplots(figsize=fig_size(scale=scale, ncols=1))


        axes.plot(times, rms, marker="o", color=color_rms)   
        axes.set_xlabel("Time (s)")
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        if zoom_x is not None:
            axes.set_xlim(zoom_x)
            # axes.relim()              # recomputa límites basados en datos visibles
            # axes.autoscale(axis="y")  # autoscale solo en Y

        y_low, y_high = axes.get_ylim()
        x_low, x_high = axes.get_xlim()
        y_low, y_high = 0.0, 1.0
        x_low, x_high =0.0, 1.0

        vlines_2 = kargs.get( "vlines_2", None)

        if vlines is not None:
            for xv in vlines:
                axes.axvline(x=xv, ymin=y_low, ymax=y_high, color="black", linestyle=":", linewidth=1.2)
        if hlines is not None:
            for yv in hlines:
                axes.axhline(y=yv, xmin=x_low, xmax=x_high, color="black", linestyle="--", linewidth=1.5)
        if vlines_2 is not None:
            for xv in vlines_2:
                axes.axvline(x=xv, ymin=y_low, ymax=y_high, color="black", linestyle="--", linewidth=1.5,
                                alpha=1.0  )

        axes.set_ylabel("RMS")
        axes.set_title(title)
        axes.grid(False)
        plt.tight_layout()

        return fig, axes

    def _plot_cv(time_seq: Sequence[float], cv_seq: Sequence[float],
                 cv_threshold: float, zoom_x: Optional[tuple[float, float]] = None,
                 zoom_y: Optional[tuple[float, float]] = None,
                 *, title: str = "CV", scale: float = 1.0,
                 vlines: Optional[Sequence[float]] = None,
                 hlines: Optional[Sequence[float]] = None) -> None:
        # Comentario: traza CV
        fig, axes =  plt.subplots(figsize=fig_size(scale=scale, ncols=1))
        axes.scatter(time_seq, cv_seq, color=color_CV, marker="o", s=30)
        axes.axhline(y=cv_threshold, color="r", linestyle="--", label="CV threshold")
        axes.set_xlabel("Time (s)")
        axes.set_ylabel("CV")
        axes.set_title(title)
        axes.grid(False)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))


        y_low, y_high = axes.get_ylim()
        x_low, x_high = axes.get_xlim()
        if zoom_x is not None:
            axes.set_xlim(zoom_x)
            # axes.relim()              # recomputa límites basados en datos visibles
            # axes.autoscale(axis="y")
            #
        if vlines is not None:
            for xv in vlines:
                axes.axvline(x=xv, ymin=y_low, ymax=y_high, color="black", linestyle="--", linewidth=1.5)
        if hlines is not None:
            for yv in hlines:
                axes.axhline(y=yv, xmin=x_low, xmax=x_high, color="black", linestyle="--", linewidth=1.5)
        plt.tight_layout()

        return fig, axes

    def _plot_signal(t: "np.ndarray", x: "np.ndarray", *,
                     zoom_x: Optional[tuple[float, float]] = None,
                     zoom_y: Optional[tuple[float, float]] = None,
                     title: str = "Signal", 
                     scale: float = 1.0,
                     vlines: Optional[Sequence[float]] = None,
                     hlines: Optional[Sequence[float]] = None,
                     **kargs) -> None:
        # Comentario: traza señal temporal
        fig, axes =  plt.subplots(figsize=fig_size(scale=scale, ncols=1))
        axes.plot(t, x)
        axes.set_xlabel("Time (s)")
        axes.set_ylabel(r"Velocity $v(t)$ [m/s]")
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        if zoom_x is not None:
            axes.set_xlim(zoom_x)
            # axes.relim()              # recomputa límites basados en datos visibles
            # axes.autoscale(axis="y") 
        y_low, y_high = 0.0, 1.0
        x_low, x_high = 0.0, 1.0
        vlines_2 = kargs.get( "vlines_2", None)

        if vlines is not None:
            for xv in vlines:
                axes.axvline(x=xv, ymin=y_low, ymax=y_high, color="gray", linestyle="--", linewidth=1,
                             alpha=0.7  )
        if vlines_2 is not None:
            for xv in vlines_2:
                axes.axvline(x=xv, ymin=y_low, ymax=y_high, color="black", linestyle="--", linewidth=1.5,
                                alpha=1.0  )
        if hlines is not None:
            for yv in hlines:
                axes.axhline(y=yv, xmin=x_low, xmax=x_high, color="gray", linestyle="--", linewidth=1,
                             alpha=0.7  )
        axes.set_title(title)
        axes.grid(False)

        plt.tight_layout()

        return fig, axes

    configurar_estilo_global()

    meta = result.meta or {}
    t = signal.t_analysis
    signal_analysis = signal.signal_analysis

    t_rms = meta.get("t_rms", None)
    rms_values = meta.get("rms_values", None)
    cv_time = meta.get("cv_time", None)
    cv_values = meta.get("cv_values", None)
    cv_threshold = meta.get("cv_threshold", None)
    rms_threshold = meta.get("rms_threshold", None)
    window_sec = meta.get("window_sec", None)
    idx_rms_windows = meta.get("idx_rms_windows", None)
    times_rms_windows = None

    if idx_rms_windows is not None and t is not None:
        times_rms_windows = t[idx_rms_windows[:, 0]]

    cv_num_data = meta.get("n_max", None)
    cv_vlines = []
    if cv_num_data is not None:
        cv_vlines = t_rms[:: cv_num_data]

    scale = 3.0
    kargs = {
        "vlines_2": vlines
    }



    fig_signal, axes_signal = _plot_signal(t, signal_analysis, zoom_x=zoom_x, zoom_y=zoom_y, 
                                           title="Tool Velocity ", scale=scale,
                                           vlines=times_rms_windows, hlines=None,
                                           **kargs) 

    fig_rms, axes_rms = _plot_rms(t_rms, rms_values, zoom_x=zoom_x, 
                                  zoom_y=zoom_y, title="RMS Sequence", scale=scale,
                                  vlines=cv_vlines, hlines=None,
                                  **kargs)
    fig_cv, axes_cv = _plot_cv(cv_time, cv_values, cv_threshold, 
                               zoom_x=zoom_x, zoom_y=zoom_y, title="CV Sequence",
                               scale=scale,
                               vlines=vlines, hlines=hlines)   
    plt.tight_layout()
    plt.show()





