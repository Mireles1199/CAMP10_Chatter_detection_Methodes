#!/usr/bin/env python
# coding: utf-8
"""plot_style.py — Fuente única de verdad del estilo article-plot-style (skill).

rcParams, tamaños de figura y helper de texto bilingüe para cualquier figura de
artículo/tesis de este repo. Nunca copiar `ARTICLE_RCPARAMS` en otro archivo —
importar de acá y `plt.rcParams.update(ARTICLE_RCPARAMS)` (o, para no ensuciar el
estilo global de una app con estado propio, `matplotlib.rc_context(ARTICLE_RCPARAMS)`).
"""

from __future__ import annotations

ARTICLE_RCPARAMS = {
    # Typography
    "font.family": "serif", "font.size": 12,
    "axes.titlesize": 16, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 10,
    # Lines & markers
    "lines.linewidth": 1.2, "lines.markersize": 10,
    # Axes borders
    "axes.linewidth": 0.8, "grid.linewidth": 0.5,
    # Ticks — inward, with minor ticks
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 4, "ytick.major.size": 4,
    "xtick.minor.size": 2.5, "ytick.minor.size": 2.5,
    "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
    # Math text — STIX font
    "mathtext.fontset": "stix", "axes.formatter.use_mathtext": True,
    # Legend — no frame
    "legend.frameon": False, "legend.loc": "best",
    "legend.handlelength": 2.0, "legend.borderaxespad": 0.5,
    # Export
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "savefig.transparent": True,
    # Background — white
    "figure.facecolor": "white", "axes.facecolor": "white",
}

# Plantilla: artículo a dos columnas estilo Elsevier (col ≈ 3.5 in, página ≈ 7.16 in).
FIGSIZE_SIMPLE = (3.5, 2.6)   # 1 columna — panel único
FIGSIZE_WIDE   = (7.16, 2.6)  # ancho de página completa — misma altura que SIMPLE

# Paleta estable/inestable Okabe-Ito (segura para deuteranopía/protanopía), con
# hatch como redundancia de forma además de color.
COLOR_STABLE   = "#0072B2"   # azul
COLOR_UNSTABLE = "#E69F00"   # naranja
HATCH_UNSTABLE = "//"
COLOR_GRAY     = "#7F7F7F"   # gris neutro -- zona gris (reference_dataset.py --strategy amplitude)
HATCH_GRAY     = ".."


def figsize_grid(ncols: int, nrows: int = 1) -> tuple[float, float]:
    """Generaliza FIGSIZE_SIMPLE a un grid de N paneles (small multiples)."""
    w, h = FIGSIZE_SIMPLE
    return (w * ncols, h * nrows)


def figsize_from_scale(base_figsize: tuple[float, float], scale: float) -> tuple[float, float]:
    """Escala un figsize base preservando su relación de aspecto.

    `scale` es un multiplicador simple (1 = tamaño original, 1.5, 2, 0.5, ...)
    aplicado por igual a ancho y alto, así la proporción del preset original
    se mantiene siempre.
    """
    w, h = base_figsize
    return (w * scale, h * scale)


def lang_text(en: str, fr: str, language: str, sep: str = "\n") -> str:
    """Arma el texto de la figura según el idioma elegido.

    language: "EN" (solo inglés) | "FR" (solo francés) | "both" (bilingüe, con prefijo).
    `sep` controla cómo se unen ambos idiomas en modo "both": "\\n" para
    título/ejes (una línea por idioma), " / " para una leyenda de una sola línea.
    """
    if language == "EN":
        return en
    if language == "FR":
        return fr
    if language == "both":
        return f"[EN] {en}{sep}[FR] {fr}"
    raise ValueError(f"language debe ser 'EN', 'FR' o 'both', recibido: {language!r}")


if __name__ == "__main__":
    assert figsize_grid(3) == (FIGSIZE_SIMPLE[0] * 3, FIGSIZE_SIMPLE[1])
    assert figsize_from_scale(FIGSIZE_WIDE, 2) == (FIGSIZE_WIDE[0] * 2, FIGSIZE_WIDE[1] * 2)
    assert lang_text("Stable", "Stable", "EN") == "Stable"
    assert lang_text("Stable", "Stable", "both", sep=" / ") == "[EN] Stable / [FR] Stable"
    try:
        lang_text("a", "b", "XX")
        raise AssertionError("debía fallar con language inválido")
    except ValueError:
        pass
    print("self-test OK")
