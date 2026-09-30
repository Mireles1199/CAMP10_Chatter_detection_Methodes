"""Shared article-figure style for ssq_chatter/viz (article-plot-style skill).

Own copy per package on purpose (see COMMON_TEMPLATE.md: no cross-package imports).
"""
from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ARTICLE_RCPARAMS = {
    # Typography
    'font.family': 'serif', 'font.size': 12,
    'axes.titlesize': 16, 'axes.labelsize': 16,
    'xtick.labelsize': 14, 'ytick.labelsize': 14,
    'legend.fontsize': 10,
    # Lines & markers
    'lines.linewidth': 1.2, 'lines.markersize': 10,
    # Axes borders
    'axes.linewidth': 0.8, 'grid.linewidth': 0.5,
    # Ticks — inward, with minor ticks
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 4, 'ytick.major.size': 4,
    'xtick.minor.size': 2.5, 'ytick.minor.size': 2.5,
    'xtick.minor.width': 0.6, 'ytick.minor.width': 0.6,
    # Math text — STIX font
    'mathtext.fontset': 'stix', 'axes.formatter.use_mathtext': True,
    # Legend — no frame
    'legend.frameon': False, 'legend.loc': 'best',
    'legend.handlelength': 2.0, 'legend.borderaxespad': 0.5,
    # Export
    'figure.dpi': 100, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02, 'savefig.transparent': True,
    # Background — white
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
}
plt.rcParams.update(ARTICLE_RCPARAMS)

FIGSIZE_SIMPLE = (3.5, 2.6)   # 1 column — single panel
FIGSIZE_WIDE   = (7.16, 2.6)  # full page width — wide single panel or 2 side-by-side


def figsize_grid(ncols: int, nrows: int = 1, base: Tuple[float, float] = None) -> Tuple[float, float]:
    w, h = base if base is not None else FIGSIZE_SIMPLE
    return (w * ncols, h * nrows)


def figsize_from_scale(base_figsize: Tuple[float, float], scale: float) -> Tuple[float, float]:
    """Scale a base figsize preserving its aspect ratio (1 = original size)."""
    w, h = base_figsize
    return (w * scale, h * scale)


FIGSCALE_SIMPLE = 2  # default multiplier (article-plot-style skill sec. 1)
SCALE = FIGSCALE_SIMPLE  # editable: 1 = article size, >1 to enlarge (e.g. for a poster)

# Semantic palette (Okabe-Ito derived, colorblind-safe) replacing the old ad hoc
# HLS palette. "decision_variable"/"mean" cover roles this package needs beyond
# the base stable/chatter/threshold/reference/zero_line set.
COLORS = {
    "stable": "#0072B2",             # was color_azul
    "chatter": "#E69F00",            # was color_orange
    "threshold": "crimson",          # was color_red
    "reference": "navy",
    "zero_line": "black",
    "decision_variable": "#CC79A7",  # was color_purple (SVD 1st component curve)
    "mean": "#009E73",               # was color_verde (mu line)
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
