"""Colours of the curves of this package (semantic roles). The style itself (rcParams, sizes) is plot_style.py, an
identical copy of DOE_utils/DOE_plots/plot_style.py (check_plot_style.py --sync): the palette lives here, not there."""

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
