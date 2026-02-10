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
color_red = (r, g, b)

# Orange
H = 36/360   # de grados a [0,1]
S = 0.99
L = 0.45

r, g, b =  colorsys.hls_to_rgb(H, L, S)
color_orange = (r, g, b)

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

# Azul
H = 206.957/360   # de grados a [0,1]
S = 0.55603
L = 0.40941

r, g, b =  colorsys.hls_to_rgb(H, L, S)
color_azul = (r, g, b)



def fig_size(scale=1.0, ncols=1, base_width=3.4):
    """
    scale: factor de escala (1 = tamaño normal)
    ncols: 1=single, 2=double, 3=triple
    base_width: ancho de una columna típica
    """
    width = base_width * ncols * scale
    height = width * 0.40   # relación agradable
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


def plots_maxent_sprt(
    signal: Optional[SignalData],
    result: IndicatorResult,
    show_signal: bool = True,
    show: bool = True,
    zoom_x: Optional[tuple[float, float]] = None,
    zoom_y: Optional[tuple[float, float]] = None,
    vlines: Optional[Sequence[float]] = None,
    hlines: Optional[Sequence[float]] = None,

) -> plt.Figure:

    def _plot_signal(
        t_stable: np.ndarray,
        signal_analysis_stable: np.ndarray,
        t_chatter: np.ndarray,
        signal_analysis_chatter: np.ndarray,
        scale: float = 1.0,
        title: str = "Tool Velocity",
        zoom_x: Optional[tuple[float, float]] = None,
        zoom_y: Optional[tuple[float, float]] = None,
        vlines: Optional[Sequence[float]] = None,
        hlines: Optional[Sequence[float]] = None,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(figsize=fig_size(scale=scale, ncols=1))

        ax.plot(t_stable, signal_analysis_stable, label="Stable Signal", color='blue', alpha=0.7)
        ax.plot(t_chatter, signal_analysis_chatter, label="Chatter Signal", color=color_orange, alpha=0.7)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        if zoom_x is not None:
            ax.set_xlim(zoom_x)
        if zoom_y is not None:
            ax.set_ylim(zoom_y)

        if vlines is not None:
            for vx in vlines:
                ax.axvline(x=vx, color=color_orange, linestyle='--', alpha=0.7, linewidth=1.0)

        if hlines is not None:
            for hy in hlines:
                ax.axhline(y=hy, color='gray', linestyle='--', alpha=0.7)

        ax.set_title("Tool Velocity")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"Velocity $v(t)$ [m/s]")
        ax.legend()

        return fig, ax

    def _plot_opr(
        t_opr: np.ndarray,
        v_opr: np.ndarray,
        t: np.ndarray,
        v: np.ndarray,
        scale: float = 1.0,
        title: str = "Tool Velocity - OPR Sampled",
        zoom_x: Optional[tuple[float, float]] = None,
        zoom_y: Optional[tuple[float, float]] = None,
        vlines: Optional[Sequence[float]] = None,
        hlines: Optional[Sequence[float]] = None,
        size: Optional[tuple[float, float]] = None,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        if size is None and scale is not None:
            fig, ax = plt.subplots(figsize=fig_size(scale=scale, ncols=1))
        else:
            fig, ax = plt.subplots(figsize=size)

        ax.plot(t, v, label="Stable Signal", color=color_azul, alpha=0.99)
        ax.scatter(t_opr, v_opr, label="OPR Sampled", color=color_red, 
                   alpha=0.99, s=7,
                   zorder = 5)

        if zoom_x is not None:
            ax.set_xlim(zoom_x)
        if zoom_y is not None:
            ax.set_ylim(zoom_y)

        if vlines is not None:
            for vx in vlines:
                ax.axvline(x=vx, color=color_orange, linestyle='--', alpha=0.99, linewidth=1.5)

        if hlines is not None:
            for hy in hlines:
                ax.axhline(y=hy, color='gray', linestyle='--', alpha=0.7)

        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"Velocity $v(t)$ [m/s]")
        ax.legend()

        return fig, ax

    def _plot_pdf(
        mu: float,
        sigma: float,
        n_points: int = 1000,

        scale: float = 1.0,
        title: str = "Tool Velocity - OPR Sampled",
        zoom_x: Optional[tuple[float, float]] = None,
        zoom_y: Optional[tuple[float, float]] = None,
        vlines: Optional[Sequence[float]] = None,
        hlines: Optional[Sequence[float]] = None,
        size: Optional[tuple[float, float]] = None,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:

        # rango de graficación ~ 4 sigmas a cada lado
        v = np.linspace(mu - 4*sigma, mu + 4*sigma, n_points)

        # pdf
        f = 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-(v-mu)**2 / (2*sigma**2))

        plt.figure(figsize=fig_size(scale=scale, ncols=1))
        plt.plot(v, f, color=color_verde, lw=20)

        # líneas verticales
        xs = [mu,
            mu + sigma, mu - sigma,
            mu + 2*sigma, mu - 2*sigma,
            mu + 3*sigma, mu - 3*sigma]

        # estilos de línea
        styles = ['--', '--', '--', '--', '--', '--', '--']

        # for x, s in zip(xs, styles):
        #     plt.axvline(x, color='gray', ls=s, lw=1.2)

        # anotaciones
        # plt.axvline(mu, color='red', lw=2, label='media')

        plt.xlabel(r'$v [m/s]$')
        plt.ylabel(r'$f_{\mathrm{seg},n}(v)$')
        plt.title('Normal PDF Estimated from  1 Segement (Stable Signal)')

        plt.grid(False)
        plt.xticks([])            # mostrar números del eje X
        plt.yticks([])
        plt.tight_layout()

    def _plot_PDF_model(
        P0, P1,
        H_free, H_chat,
        scale: float = 1.0,
        title: str = "Tool Velocity - OPR Sampled",
        zoom_x: Optional[tuple[float, float]] = None,
        zoom_y: Optional[tuple[float, float]] = None,
        vlines: Optional[Sequence[float]] = None,
        hlines: Optional[Sequence[float]] = None,
        size: Optional[tuple[float, float]] = None,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        if size is None and scale is not None:
            fig, ax = plt.subplots(figsize=fig_size(scale=scale, ncols=1))
        else:
            fig, ax = plt.subplots(figsize=size)

        xs = np.linspace(
            min(H_free.min(), H_chat.min()) - 0.1,
            max(H_free.max(), H_chat.max()) + 0.1,
            200,
        )

        pdf0 = np.exp([P0.logpdf(x) for x in xs])
        pdf1 = np.exp([P1.logpdf(x) for x in xs])

        ax.plot(xs, pdf0, label=r"PDF $P_0(H)$ Stable", 
                color=color_azul, alpha=0.99, linewidth=5.0)
        ax.plot(xs, pdf1, label=r"PDF $P_1(H)$ Chatter", 
                color=color_orange, alpha=0.99, linewidth=5.0)

        if zoom_x is not None:
            ax.set_xlim(zoom_x)
        if zoom_y is not None:
            ax.set_ylim(zoom_y)

        if vlines is not None:
            for vx in vlines:
                ax.axvline(x=vx, color=color_orange, linestyle='--', alpha=0.99, linewidth=1.5)

        if hlines is not None:
            for hy in hlines:
                ax.axhline(y=hy, color='gray', linestyle='--', alpha=0.7)

        ax.set_title(title)
        ax.set_xlabel("Entropy H")
        ax.set_ylabel(r"Probability Density Function $f_H(H)$")

        return fig, ax

    def _plot_H(
        H_free, H_chat,
        scale: float = 1.0,
        title: str = "Tool Velocity - OPR Sampled",
        zoom_x: Optional[tuple[float, float]] = None,
        zoom_y: Optional[tuple[float, float]] = None,
        vlines: Optional[Sequence[float]] = None,
        hlines: Optional[Sequence[float]] = None,
        size: Optional[tuple[float, float]] = None,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        if size is None and scale is not None:
            fig, ax = plt.subplots(figsize=fig_size(scale=scale, ncols=1))
        else:
            fig, ax = plt.subplots(figsize=size)

        xs = np.linspace(
            min(H_free.min(), H_chat.min()) - 0.1,
            max(H_free.max(), H_chat.max()) + 0.1,
            200,
        )

        segment = np.arange(1, len(H_free)+1)
        ax.plot(segment, H_free, label="H Stable", marker='o', 
                color=color_azul, alpha=0.99, linewidth=5.0,
                markersize=10)

        if zoom_x is not None:
            ax.set_xlim(zoom_x)
        if zoom_y is not None:
            ax.set_ylim(zoom_y)

        if vlines is not None:
            for vx in vlines:
                ax.axvline(x=vx, color=color_orange, linestyle='--', alpha=0.99, linewidth=1.5)

        if hlines is not None:
            for hy in hlines:
                ax.axhline(y=hy, color='gray', linestyle='--', alpha=0.7)

        ax.set_title(title)
        ax.set_xlabel("Number of Segments")
        ax.set_ylabel(r"Entropy H")
        ax.legend()

        return fig, ax

    def _plot_S_n(
        t_i, I, lim_sup: float, lim_inf: float,
        scale: float = 1.0,
        title: str = "Tool Velocity - OPR Sampled",
        zoom_x: Optional[tuple[float, float]] = None,
        zoom_y: Optional[tuple[float, float]] = None,
        vlines: Optional[Sequence[float]] = None,
        hlines: Optional[Sequence[float]] = None,
        size: Optional[tuple[float, float]] = None,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        if size is None and scale is not None:
            fig, ax = plt.subplots(figsize=fig_size(scale=scale, ncols=1))
        else:
            fig, ax = plt.subplots(figsize=size)

        ax.plot(t_i, I, label="Indicator I_n", marker='o',
                color=color_purple, alpha=0.99, linewidth=5.0,
                markersize=10)
        ax.axhline(y=lim_sup, color=color_red, linestyle='--', alpha=0.99, linewidth=3, label="Upper SPRT Limit")

        if zoom_x is not None:
            ax.set_xlim(zoom_x)
        if zoom_y is not None:
            ax.set_ylim(zoom_y)

        if vlines is not None:
            for vx in vlines:
                ax.axvline(x=vx, color=color_orange, linestyle='--', alpha=0.99, linewidth=1.5)

        if hlines is not None:
            for hy in hlines:
                ax.axhline(y=hy, color='gray', linestyle='--', alpha=0.7)

        ax.set_title(title)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(r"$I_{SPRT}$")
        # ax.legend()

        return fig, ax




    configurar_estilo_global()

    meta = result.meta or {}
    t = signal.t_analysis
    x = signal.signal_analysis
    fs = signal.fs

    t_stable = meta.get("t_stable", None)
    signal_analysis_stable = meta.get("signal_analysis_stable", None)
    t_chatter = meta.get("t_chatter", None)
    signal_analysis_chatter = meta.get("signal_analysis_chatter", None)

    t_opr_free = meta.get("t_opr_free", None)
    opr_free = meta.get("opr_free", None)
    t_opr_chat = meta.get("t_opr_chat", None)
    opr_chat = meta.get("opr_chat", None)

    t_i = result.t
    I = result.I_t

    lim_sup = meta.get("sprt_result", None).b
    lim_inf = meta.get("sprt_result", None).a

    scale = 5.0
    kargs = {
    "vlines_2": vlines
    }



    fig_sigal, axes_signal = _plot_signal(t_stable, signal_analysis_stable,
                                  t_chatter, signal_analysis_chatter, scale=scale,
                                  zoom_x=zoom_x, zoom_y=zoom_y,
                                  vlines=None, hlines=hlines,
                                  **kargs)

    fig_opr_free, axes_opr_free = _plot_opr(t_opr_free, opr_free,
                                  t_stable, signal_analysis_stable, scale=scale,
                                    zoom_x=zoom_x, zoom_y=zoom_y,
                                    vlines=vlines, hlines=hlines,
                                    title="Tool Velocity - OPR Sampled (Stable)",
                                    # fig_size=,
                                    **kargs)

    fig_opr_chat, axes_opr_chat = _plot_opr(t_opr_chat, opr_chat,
                                  t_chatter, signal_analysis_chatter, scale=scale,
                                    zoom_x=zoom_x, zoom_y=zoom_y,
                                    vlines=vlines, hlines=hlines,
                                    title="Tool Velocity - OPR Sampled (Chatter)",
                                    # fig_size=(6,4),
                                    **kargs)

    _plot_pdf(
        mu=meta.get("P0_mu", 0.0),
        sigma=meta.get("P0_sigma", 1.0),
        scale=scale,
        title="PDF Estimated - Stable Signal",
        zoom_x=zoom_x,
        zoom_y=zoom_y,
        vlines=vlines,
        hlines=hlines,
        size=(7,4),
    )

    p0 = meta.get("models_trained", None).p0
    p1 = meta.get("models_trained", None).p1
    H_free = meta.get("detector", None).H_free
    H_chat = meta.get("detector", None).H_chat

    fig_pdf_models, axes_pdf_models = _plot_PDF_model(
        P0=p0,
        P1=p1,
        H_free=H_free,
        H_chat=H_chat,
        scale=scale,
        title="MaxEnt PDF Models Estimated - Stable vs Chatter",
        zoom_x=zoom_x,
        zoom_y=zoom_y,
        vlines=None,
        hlines=hlines,
        # size=(7,4),
    )

    fig_H, axes_H = _plot_H(
        H_free=H_free,
        H_chat=H_chat,
        scale=scale,
        title="Entropy H Sequence - Stable",
        zoom_x=zoom_x,
        zoom_y=zoom_y,
        vlines=None,
        hlines=hlines,
        # size=(7,4),
    )

    fig_S_n, axes_S_n = _plot_S_n(
        t_i=t_i,
        I=I,
        lim_sup=lim_sup,
        lim_inf=lim_inf,
        scale=scale,
        title=R"MaxEnt SPRT  $I_{SPRT}$ Sequence",
        zoom_x=zoom_x,
        zoom_y=zoom_y,
        vlines=None,
        hlines=hlines,
        # size=(7,4),
    )

    plt.tight_layout()
    plt.show()


