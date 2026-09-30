# ========= article-plot-style — shared style for MaxEnt-SPRT figures =========
"""Single source of truth for figure size/rcParams/palette in this package's viz/.

Copied per-package on purpose (repo convention: no cross-package imports —
see indicators/COMMON_TEMPLATE.md). Based on the `article-plot-style` skill.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ARTICLE_RCPARAMS = {
    'font.family': 'serif', 'font.size': 12,
    'axes.titlesize': 16, 'axes.labelsize': 16,
    'xtick.labelsize': 14, 'ytick.labelsize': 14,
    'legend.fontsize': 10,
    'lines.linewidth': 1.2, 'lines.markersize': 3,  # small data-point markers; event/highlight markers use an explicit larger s= per article-plot-style sec. 4
    'axes.linewidth': 0.8, 'grid.linewidth': 0.5,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 4, 'ytick.major.size': 4,
    'xtick.minor.size': 2.5, 'ytick.minor.size': 2.5,
    'xtick.minor.width': 0.6, 'ytick.minor.width': 0.6,
    'mathtext.fontset': 'stix', 'axes.formatter.use_mathtext': True,
    'legend.frameon': False, 'legend.loc': 'best',
    'legend.handlelength': 2.0, 'legend.borderaxespad': 0.5,
    'figure.dpi': 100, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02, 'savefig.transparent': True,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
}
plt.rcParams.update(ARTICLE_RCPARAMS)

FIGSIZE_SIMPLE = (3.5, 2.6)   # 1 columna
FIGSIZE_WIDE   = (7.16, 2.6)  # ancho de pagina completa / 2 paneles lado a lado

FIGSCALE_SIMPLE = 2  # default multiplier for this indicator batch (overrides skill's 1.5 suggestion)
SCALE = FIGSCALE_SIMPLE  # variable editable: 1 = tamano articulo, 1.5, 2, ...


def figsize_grid(ncols: int, nrows: int = 1, base: tuple[float, float] | None = None) -> tuple[float, float]:
    w, h = base if base is not None else FIGSIZE_SIMPLE
    return (w * ncols, h * nrows)


def figsize_from_scale(base_figsize: tuple[float, float], scale: float) -> tuple[float, float]:
    """Escala un figsize base preservando su relacion de aspecto."""
    w, h = base_figsize
    return (w * scale, h * scale)


# Paleta Okabe-Ito (accesible a daltonismo) mapeada a los roles semanticos que
# ya usa este paquete (antes: paleta HLS ad hoc con 5 colores arbitrarios).
COLORS = {
    "stable":        "#0072B2",  # azul — P0 / stable
    "chatter":       "#E69F00",  # naranja — P1 / chatter
    "threshold":     "crimson",  # limites de decision / alarma
    "reference":     "navy",     # lineas de referencia
    "zero_line":     "black",    # linea cero
    "accent_purple": "#CC79A7",  # estadistico Sk
    "accent_green":  "#009E73",  # curva secundaria / region beta
}


def apply_sci_yaxis(ax) -> None:
    fmt = mticker.ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(fmt)
    off = ax.yaxis.get_offset_text()
    off.set_size(14)
    off.set_x(-0.12)
    off.set_y(-0.05)
