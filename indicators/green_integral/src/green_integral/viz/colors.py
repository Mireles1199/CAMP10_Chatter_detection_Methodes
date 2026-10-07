"""Colours of the curves of this package (semantic roles). The style itself (rcParams, sizes) is plot_style.py, an
identical copy of DOE_utils/DOE_plots/plot_style.py (check_plot_style.py --sync): the palette lives here, not there."""

COLORS = {
    "stable": "#0072B2",       # main trace / training-population color
    "chatter": "#E69F00",      # instability / detection event markers (t_gt, t_d)
    "threshold": "crimson",    # mu +- z*sigma threshold lines
    "mean": "#009E73",         # mu / central-tendency line (Okabe-Ito bluish green)
    "secondary": "#CC79A7",    # extra trace when more than 2 series needed (Okabe-Ito purple)
    "reference": "navy",
    "zero_line": "black",
}
