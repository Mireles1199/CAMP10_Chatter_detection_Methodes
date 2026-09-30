"""Article-ready plotting style (article-plot-style skill) for emd_hht.

Own copy, not shared with the other indicator packages (repo convention:
no cross-package imports — see indicators/COMMON_TEMPLATE.md).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ARTICLE_RCPARAMS = {
    'font.family': 'serif', 'font.size': 12,
    'axes.titlesize': 16, 'axes.labelsize': 16,
    'xtick.labelsize': 14, 'ytick.labelsize': 14,
    'legend.fontsize': 10,
    'lines.linewidth': 1.2, 'lines.markersize': 3,
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

FIGSIZE_SIMPLE = (3.5, 2.6)   # 1 column, single panel
FIGSIZE_WIDE   = (7.16, 2.6)  # full page width, single panel or 2 side-by-side

FIGSCALE_SIMPLE = 2  # default multiplier for this package (overrides skill default of 1.5)
SCALE = FIGSCALE_SIMPLE  # editable multiplier applied to every figure size in this package


def figsize_grid(ncols: int, nrows: int = 1, base: tuple[float, float] | None = None) -> tuple[float, float]:
    w, h = base if base is not None else FIGSIZE_SIMPLE
    return (w * ncols, h * nrows)


def figsize_from_scale(base_figsize: tuple[float, float], scale: float) -> tuple[float, float]:
    """Scale a base figsize preserving its aspect ratio."""
    w, h = base_figsize
    return (w * scale, h * scale)


def apply_sci_yaxis(ax) -> None:
    fmt = mticker.ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(fmt)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    off = ax.yaxis.get_offset_text()
    off.set_size(14)
    off.set_x(-0.12)
    off.set_y(-0.05)
