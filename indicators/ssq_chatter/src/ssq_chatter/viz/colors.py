"""Colours of the curves of this package (semantic roles). The style itself (rcParams, sizes) is plot_style.py, an
identical copy of DOE_utils/DOE_plots/plot_style.py (check_plot_style.py --sync): the palette lives here, not there."""

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
