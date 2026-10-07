#!/usr/bin/env python
# coding: utf-8
"""plot_style.py — Fuente única de verdad del estilo article-plot-style (skill).

rcParams, tamaños de figura, escala del texto y helper de texto bilingüe para cualquier figura de artículo/tesis
de este repo (PLAN_plot_style.md). Los 5 paquetes de indicadores (instalables, no importan de DOE_utils) llevan una
copia BYTE A BYTE de este archivo en su `viz/`: no editarlas; `check_plot_style.py --sync` falla si divergen.

Sin efectos al importar y solo depende de matplotlib: nada cambia el estilo global hasta que se pide.
    with plt.rc_context(ps.rc_scaled(scale)): ...      # lo normal (no ensucia el estilo global de una app)
    ps.apply(scale)                                    # scripts sueltos: plt.rcParams.update(rc_scaled(scale))
Los tamaños de letra a mano se escriben RELATIVOS ('small', 'x-small', 'large'...): siguen a `font.size` y por tanto
a la escala. `ARTICLE_RCPARAMS` es exactamente el del skill (lines.markersize 3; un marcador de evento lleva s= propio).
"""

from __future__ import annotations

ARTICLE_RCPARAMS = {
    # Typography
    "font.family": "serif", "font.size": 12,
    "axes.titlesize": 16, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 10,
    # Lines & markers -- small by default; a highlighted single event point overrides with an explicit larger s=
    "lines.linewidth": 1.2, "lines.markersize": 3,
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
FIGSCALE_SIMPLE = 1.5         # multiplicador por defecto (skill §1); 1 = tamaño de artículo impreso
SCALE = FIGSCALE_SIMPLE       # alias editable que usan los módulos de figura de los paquetes

# Paleta estable/inestable Okabe-Ito (segura para deuteranopía/protanopía), con hatch como redundancia de forma.
COLOR_STABLE   = "#0072B2"   # azul
COLOR_UNSTABLE = "#E69F00"   # naranja
HATCH_UNSTABLE = "//"
COLOR_GRAY     = "#7F7F7F"   # gris neutro -- zona gris (reference_dataset.py --strategy amplitude)
HATCH_GRAY     = ".."

# Claves de ARTICLE_RCPARAMS que son una talla (pt) y escalan con el lienzo en rc_scaled: letra, líneas, marcadores, ticks.
_SIZE_KEYS = ("font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize", "ytick.labelsize", "legend.fontsize",
              "lines.linewidth", "lines.markersize", "axes.linewidth", "grid.linewidth",
              "xtick.major.width", "ytick.major.width", "xtick.major.size", "ytick.major.size",
              "xtick.minor.size", "ytick.minor.size", "xtick.minor.width", "ytick.minor.width")


def figsize_grid(ncols: int, nrows: int = 1, base: tuple[float, float] | None = None) -> tuple[float, float]:
    """Generaliza FIGSIZE_SIMPLE (o `base`) a un grid de N paneles (small multiples)."""
    w, h = base if base is not None else FIGSIZE_SIMPLE
    return (w * ncols, h * nrows)


def figsize_from_scale(base_figsize: tuple[float, float], scale: float) -> tuple[float, float]:
    """Escala un figsize base preservando su relación de aspecto.

    `scale` es un multiplicador simple (1 = tamaño original, 1.5, 2, 0.5, ...)
    aplicado por igual a ancho y alto, así la proporción del preset original
    se mantiene siempre.
    """
    w, h = base_figsize
    return (w * scale, h * scale)


def rc_scaled(scale: float = FIGSCALE_SIMPLE, follow_text: bool = True) -> dict:
    """ARTICLE_RCPARAMS para una figura de tamaño `figsize_from_scale(preset, scale)`.

    Los puntos del skill están pensados para su escala por defecto, FIGSCALE_SIMPLE (1.5): a esa escala ambos modos
    devuelven ARTICLE_RCPARAMS tal cual. follow_text=True (por defecto): a otra escala, letra, grosores, marcadores
    y ticks se multiplican por scale / FIGSCALE_SIMPLE, como un zoom (la figura es la misma, más grande o más
    pequeña). follow_text=False: puntos fijos del skill a cualquier escala (el texto parece más pequeño al ampliar).
    """
    rc = dict(ARTICLE_RCPARAMS)
    f = zoom(scale, follow_text)
    if f != 1:
        rc.update({k: ARTICLE_RCPARAMS[k] * f for k in _SIZE_KEYS})
    return rc


def zoom(scale: float, follow_text: bool = True) -> float:
    """Factor de rc_scaled(scale, follow_text); también el de un tamaño a mano (marcador, grosor): 1 en FIGSCALE_SIMPLE."""
    return scale / FIGSCALE_SIMPLE if follow_text else 1.0


def apply(scale: float = FIGSCALE_SIMPLE, follow_text: bool = True) -> None:
    """Estilo global del proceso (scripts sueltos). Una app con estado propio usa `plt.rc_context(rc_scaled(...))`."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(rc_scaled(scale, follow_text))


def apply_sci_yaxis(ax) -> None:
    """Eje Y en notación científica (skill §6); el texto ×10^n sigue a `ytick.labelsize` (y por tanto a la escala)."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    fmt = mticker.ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(fmt)
    off = ax.yaxis.get_offset_text()
    off.set_size(plt.rcParams["ytick.labelsize"])
    off.set_x(-0.12)
    off.set_y(-0.05)


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


_lang_text = lang_text   # nombre que usa el skill


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    assert figsize_grid(3) == (FIGSIZE_SIMPLE[0] * 3, FIGSIZE_SIMPLE[1])
    assert figsize_grid(2, 1, base=(1.0, 2.0)) == (2.0, 2.0)
    assert figsize_from_scale(FIGSIZE_WIDE, 2) == (FIGSIZE_WIDE[0] * 2, FIGSIZE_WIDE[1] * 2)
    assert lang_text("Stable", "Stable", "EN") == "Stable" and _lang_text is lang_text
    assert lang_text("Stable", "Stable", "both", sep=" / ") == "[EN] Stable / [FR] Stable"
    try:
        lang_text("a", "b", "XX")
        raise AssertionError("debía fallar con language inválido")
    except ValueError:
        pass
    assert zoom(1.5) == 1.0 and zoom(3) == 2.0 and zoom(3, False) == 1.0
    assert ARTICLE_RCPARAMS["lines.markersize"] == 3 and FIGSCALE_SIMPLE == 1.5
    assert rc_scaled(1.5) == ARTICLE_RCPARAMS == rc_scaled(2, follow_text=False) == rc_scaled()
    r2 = rc_scaled(3)   # el doble de la escala por defecto: el doble de todo lo que es talla
    assert r2["font.size"] == 24 and r2["axes.labelsize"] == 32 and r2["lines.linewidth"] == 2.4
    assert rc_scaled(0.75)["font.size"] == 6
    assert r2["xtick.direction"] == "in" and r2["savefig.dpi"] == 300 and r2["legend.frameon"] is False   # lo que no es talla no se toca
    assert all(k in ARTICLE_RCPARAMS for k in _SIZE_KEYS)
    with plt.rc_context(rc_scaled(3)):   # un tamaño relativo sigue a font.size y por tanto a la escala
        assert abs(FontProperties(size="small").get_size_in_points() - 0.833 * 24) < 0.05
        fig, ax = plt.subplots()
        apply_sci_yaxis(ax)
        assert ax.yaxis.get_offset_text().get_size() == 28
    plt.rcdefaults()
    apply(1.5)
    assert plt.rcParams["font.size"] == 12
    plt.rcdefaults()
    print("self-test OK")
