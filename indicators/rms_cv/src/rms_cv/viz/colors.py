"""Colours of the curves of this package (semantic roles). The style itself (rcParams, sizes) is plot_style.py, an
identical copy of DOE_utils/DOE_plots/plot_style.py (check_plot_style.py --sync): the palette lives here, not there."""

COLORS = {
    "stable": "#0072B2",     # main/default trace color (Okabe-Ito blue)
    "chatter": "#E69F00",    # chatter/detection marker (Okabe-Ito orange)
    "threshold": "crimson",  # alarm / threshold lines
    "reference": "navy",     # reference/mu marker
    "zero_line": "black",
}
