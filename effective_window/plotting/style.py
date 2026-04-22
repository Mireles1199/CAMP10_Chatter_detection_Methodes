"""
plotting/style.py
=================
Global matplotlib style for the effective-window framework.

Ported verbatim from Areas_Indicator_V1.py :: configure_global_style().
Call ``configure_global_style()`` once at import time of the plotting package,
or call it manually before creating any figure.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


def configure_global_style() -> None:
    """Configure global matplotlib style for all effective-window plots."""
    local_style = {
        # Typography
        'font.family':               'serif',
        'font.size':                 9,
        # Title / label sizes
        'axes.titlesize':            15,
        'axes.labelsize':            15,
        'xtick.labelsize':           12,
        'ytick.labelsize':           12,
        'legend.fontsize':           12,
        # Line aesthetics
        'lines.linewidth':           1.2,
        'lines.markersize':          4,
        # Axes / grid borders
        'axes.linewidth':            0.8,
        'grid.linewidth':            0.5,
        # Ticks
        'xtick.major.width':         0.8,
        'ytick.major.width':         0.8,
        'xtick.direction':           'in',
        'ytick.direction':           'in',
        'xtick.major.size':          4,
        'ytick.major.size':          4,
        'xtick.minor.size':          2.5,
        'ytick.minor.size':          2.5,
        'xtick.minor.width':         0.6,
        'ytick.minor.width':         0.6,
        # Math text
        'mathtext.fontset':          'stix',
        'axes.formatter.use_mathtext': True,
        # Legend
        'legend.frameon':            False,
        'legend.loc':                'best',
        'legend.handlelength':       2.0,
        'legend.borderaxespad':      0.5,
        # Export
        'figure.dpi':                100,
        'savefig.dpi':               300,
        'savefig.bbox':              'tight',
        'savefig.pad_inches':        0.02,
        'savefig.transparent':       True,
        # Background
        'figure.facecolor':          'white',
        'axes.facecolor':            'white',
    }
    plt.rcParams.update(local_style)
