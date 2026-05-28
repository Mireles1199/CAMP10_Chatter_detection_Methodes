"""
Optimizacion/plot_style.py
==========================
Local copy of the research plotting style used across all sweep plot scripts.
Font sizes bumped to 16 pt for presentation-quality figures.

Usage
-----
    from plot_style import apply_research_style, PALETTE
    apply_research_style()
"""

from __future__ import annotations

import matplotlib.pyplot as plt


def apply_research_style() -> None:
    """Apply a publication-quality Matplotlib style globally (font size 16)."""
    plt.rcParams.update(
        {
            # Typography
            "font.family":              "serif",
            "font.size":                14,

            # Titles and labels
            "axes.titlesize":           18,
            "axes.labelsize":           16,
            "xtick.labelsize":          14,
            "ytick.labelsize":          14,
            "legend.fontsize":          14,

            # Lines and markers
            "lines.linewidth":          1.8,
            "lines.markersize":         6,

            # Axes borders
            "axes.linewidth":           0.9,
            "grid.linewidth":           0.5,
            "grid.alpha":               0.4,

            # Ticks
            "xtick.major.width":        0.9,
            "ytick.major.width":        0.9,
            "xtick.direction":          "in",
            "ytick.direction":          "in",
            "xtick.major.size":         5,
            "ytick.major.size":         5,
            "xtick.minor.size":         3,
            "ytick.minor.size":         3,
            "xtick.minor.width":        0.6,
            "ytick.minor.width":        0.6,

            # Math text
            "mathtext.fontset":         "stix",
            "axes.formatter.use_mathtext": True,

            # Legend
            "legend.frameon":           False,
            "legend.loc":               "best",
            "legend.handlelength":      2.0,
            "legend.borderaxespad":     0.5,

            # Export / save
            "figure.dpi":               100,
            "savefig.dpi":              300,
            "savefig.bbox":             "tight",
            "savefig.pad_inches":       0.05,
            "savefig.transparent":      False,

            # Background
            "figure.facecolor":         "white",
            "axes.facecolor":           "white",
        }
    )


# Colour palette — up to 8 distinct series
PALETTE: list[str] = [
    "#1f77b4",  # steel blue
    "#d62728",  # brick red
    "#2ca02c",  # forest green
    "#ff7f0e",  # burnt orange
    "#9467bd",  # muted purple
    "#8c564b",  # brown
    "#17becf",  # teal
    "#7f7f7f",  # grey
]

# ── Nomenclature helpers ──────────────────────────────────────────────────────

# Short display name per indicator (lower-case key → display string)
IND_SHORT: dict[str, str] = {
    "maxent":       "MaxEnt",
    "maxent_sprt":  "MaxEnt",
    "rms_cv":       "RMS-CV",
    "ssq_chatter":  "SSQ",
    "sst_svd":      "SST-SVD",
}


def make_label(col: str, ind: str = "", units: bool = True) -> str:
    """Return a LaTeX axis label for a DataFrame column.

    Parameters
    ----------
    col : str
        DataFrame column name (``"N_cyc"``, ``"step_cyc"``, ``"N_fen"``,
        ``"N_cyc_total"``, etc.).
    ind : str, optional
        Indicator name (lower-case, e.g. ``"maxent"``).  When provided, a
        superscript ``^{(short)}`` is appended to the symbol.
    units : bool
        Whether to append ``[cycles]`` for cycle-count columns (default True).

    Returns
    -------
    str
        LaTeX string suitable for ``ax.set_xlabel()`` / ``ax.set_ylabel()``.
    """
    sup_raw = IND_SHORT.get(ind.lower(), ind) if ind else ""
    sup_str = f"^{{({sup_raw})}}" if sup_raw else ""
    u = "  [cycles]" if units else ""

    _MAP = {
        "N_cyc":              f"$N_{{cycles}}{sup_str}${u}",
        "step_cyc":           f"$\\Delta T_{{step}}{sup_str}${u}",
        "N_fen":              f"$N_{{fen}}{sup_str}${u}",
        "N_cyc_total":        f"$N_{{cycles,total}}{sup_str}${u}",
        "N_cyc_total_actual": f"$N_{{cycles,total,act}}{sup_str}${u}",
        "overlap_frac":       "$1 - step / N_{win}$",
        "delta_t_d":          "$\\Delta t_d$  [s]",
        "N_fa":               "$N_{fa}$",
        "score":              "score  [s]",
        "P_det":              "$P_{det}$",
        "T_des_s":            "$T_{des}$  [s]",
    }
    return _MAP.get(col, col)
