---
name: article-plot-style
description: >
  Estilo canonico de matplotlib para figuras de articulo/tesis (basado en ChatterPlotter,
  time.py del proyecto Articulo_Manuf21_2026), con paleta accesible a daltonismo, dos
  tamanos de figura por defecto (columna simple / ancho de pagina completa) escalables
  con un multiplicador simple, y texto bilingue EN/FR seleccionable por variable. Usar
  cuando: se cree o modifique cualquier figura de estabilidad tiempo-congelado; se
  agreguen marcadores de cruce de estabilidad (t_E, a_E); se grafiquen perfiles de
  profundidad de corte con marcadores R; se grafiquen barridos de chatter visible (a_vis,
  t_vis vs R); se grafique crecimiento acumulado G(t) o amplitud relativa e^G(t); se
  pregunte por "estilo de figura para articulo", "tamano de figura de una columna",
  "figura de pagina completa", "escalar figura", "rcParams de la tesis", "paleta de
  colores para graficas", "eje Y notacion cientifica", "eje X secundario ap/t", "texto
  bilingue EN/FR", "idioma de la figura". Aplica SIEMPRE todas las convenciones (rcParams,
  paleta, tamanos, escala, idioma, formatters, markers) al crear o modificar cualquier
  figura de articulo/tesis.
---

# Estilo de figuras para articulo — article-plot-style

Basado en `ChatterPlotter` (`Articulo_Manuf21_2026/Comparation/time.py` — el nombre de
carpeta real en disco puede aparecer con el typo histórico `Artiuclo_...`; si existe,
úsalo tal cual para rutas, pero no repitas el typo en texto nuevo), adaptado como fuente
única de verdad para cualquier figura de artículo o de la tesis en este repositorio.

Mejoras respecto a la versión original de este estilo (`.github/skills/article-plot-style`):
paleta estable/inestable accesible a daltonismo, una sola definición de `rcParams` (sin
copias que puedan divergir), y dos tamaños de figura estándar por defecto.

---

## 0. Principio: una sola fuente de verdad para rcParams

Definir `ARTICLE_RCPARAMS` **una vez** (por ejemplo en un módulo `plot_style.py` compartido)
y reutilizarlo con `plt.rcParams.update(ARTICLE_RCPARAMS)` en cualquier script o notebook.
Nunca copiar el diccionario en más de un lugar del código.

```python
import matplotlib.pyplot as plt

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
```

---

## 1. Tamanos de figura por defecto

Plantilla de referencia: articulo a dos columnas estilo Elsevier
(ancho de columna ≈ 3.5 in, ancho de pagina completa ≈ 7.16 in).
**Toda figura nueva parte de uno de estos dos presets**, nunca de un `figsize` ad-hoc.

```python
FIGSIZE_SIMPLE = (3.5, 2.6)   # 1 columna — panel unico
FIGSIZE_WIDE   = (7.16, 2.6)  # ancho de pagina completa — misma altura que SIMPLE
```

- **`FIGSIZE_SIMPLE`**: default para cualquier figura de un solo panel pensada para
  ocupar una columna del artículo.
- **`FIGSIZE_WIDE`**: default para (a) un solo panel que necesita el ancho completo de
  la página (p. ej. una serie temporal larga u otra que no cabe legible en una columna),
  o (b) **exactamente dos paneles lado a lado**:
  `plt.subplots(1, 2, figsize=FIGSIZE_WIDE, constrained_layout=True)` — es literalmente
  `FIGSIZE_SIMPLE` con el ancho duplicado y la misma altura, así que cada uno de los dos
  paneles resultantes conserva la proporción de una figura simple.

Para más de 2 paneles, generalizar a partir de la unidad simple en vez de inventar
tamaños nuevos:

```python
def figsize_grid(ncols, nrows=1):
    w, h = FIGSIZE_SIMPLE
    return (w * ncols, h * nrows)
```

### Escalar un preset preservando la proporción — `figsize_from_scale`

Cuando se necesita una figura más grande o más chica que `FIGSIZE_SIMPLE`/`FIGSIZE_WIDE`
(p. ej. una versión ampliada para un póster, o una miniatura), **no** recalcular ancho y
alto a mano. Usar `figsize_from_scale`, que toma el preset base y un multiplicador
simple (`1` = tamaño original, `1.5`, `2`, `0.5`, ...) aplicado por igual a ancho y alto,
de modo que la proporción del preset se mantiene siempre.

```python
def figsize_from_scale(base_figsize: tuple[float, float], scale: float) -> tuple[float, float]:
    """Escala un figsize base preservando su relacion de aspecto.

    `scale` es un multiplicador simple (1 = tamano original, 1.5, 2, 0.5, ...)
    aplicado por igual a ancho y alto, asi la proporcion del preset original
    se mantiene siempre.
    """
    w, h = base_figsize
    return (w * scale, h * scale)
```

Ejemplo: `figsize_from_scale(FIGSIZE_WIDE, 1)` reproduce `FIGSIZE_WIDE` sin cambios;
`figsize_from_scale(FIGSIZE_WIDE, 2)` la duplica manteniendo la proporción 7.16:2.6.

Usar siempre `constrained_layout=True` al crear la figura. **No** usar `tight_layout()`.

---

## 2. Texto bilingüe EN/FR — `_lang_text`

Para figuras que van tanto al artículo (inglés) como a la tesis (francés), el texto
(título, ejes, leyenda) debe poder generarse en inglés, en francés, o en ambos a la vez
con un prefijo `[EN]`/`[FR]` que identifique cada idioma. Usar un único parámetro
`language` (`"EN"` | `"FR"` | `"both"`) en la función de la figura, resuelto con
`_lang_text`:

```python
def _lang_text(en: str, fr: str, language: str, sep: str = "\n") -> str:
    """Arma el texto de la figura segun el idioma elegido.

    language: "EN" (solo ingles) | "FR" (solo frances) | "both" (bilingue, con prefijo).
    `sep` controla como se unen ambos idiomas en modo "both": "\n" para
    titulo/ejes (una linea por idioma), " / " para una leyenda de una sola linea.
    """
    if language == "EN":
        return en
    if language == "FR":
        return fr
    if language == "both":
        return f"[EN] {en}{sep}[FR] {fr}"
    raise ValueError(f"language debe ser 'EN', 'FR' o 'both', recibido: {language!r}")
```

Uso típico dentro de la función de la figura:
```python
ax.set_xlabel(_lang_text("Depth $a_p$ [mm]", "Profondeur $a_p$ [mm]", language))
ax.set_title(_lang_text("Frozen-time stability", "Stabilité à temps figé", language))
ax.plot(..., label=_lang_text("Stable", "Stable", language, sep=" / "))
```

Exponer siempre `language` como variable editable al inicio del script que llama a la
figura (p. ej. `FIGURE_LANGUAGE = "both"` en `main()`), **nunca** cablear el texto bilingüe
dentro de la función de ploteo.

---

## 3. Colores

Paleta basada en Okabe-Ito (segura para las formas más comunes de daltonismo:
deuteranopía/protanopía). El par rojo/verde del estilo original para
estable/inestable se reemplaza por azul/naranja, y además se añade `hatch` como
codificación redundante por forma (no solo por color).

| Elemento | Color | Notas |
|---|---|---|
| Curva principal de estabilidad Re{λ_max} | `'steelblue'` | |
| Relleno **estable** (σ < 0) | `'#0072B2'` (azul), `alpha=0.15` | sin hatch |
| Relleno **inestable** (σ ≥ 0) | `'#E69F00'` (naranja), `alpha=0.15`, `hatch='//'` | hatch como redundancia de forma |
| Marcador de cruce de estabilidad (t_E, a_E, axvline) | `'crimson'` | |
| Línea de perfil de profundidad de corte | `'navy'` | |
| Crecimiento acumulado G(t) | `'darkorange'` | |
| Curva a_vis (eje twin) | `'crimson'` | |
| Curva t_vis (eje twin) | `'steelblue'` | |
| Gradiente de barrido R | `plt.cm.plasma(np.linspace(0.1, 0.85, n_R))` | plasma es perceptualmente uniforme y colorblind-safe |

```python
ax.fill_between(t, re, 0, where=(re < 0),  alpha=0.15, color='#0072B2', label='Stable')
ax.fill_between(t, re, 0, where=(re >= 0), alpha=0.15, color='#E69F00',
                hatch='//', label='Unstable')
```

---

## 4. Marcadores de eventos clave

```python
# Cruce de estabilidad en t_E / a_E
ax.scatter([x_mark], [0.0],
           color='crimson', zorder=5, s=75,
           edgecolor='k', linewidths=0.6)

# Cruces de chatter visible (por R)
ax.scatter([x_mark], [target],
           color=col, zorder=5, s=75,
           edgecolor='k', linewidths=0.6)
```

---

## 5. Estilos de línea para eventos

```python
ax.axhline(0.0, color='k', linewidth=0.8, linestyle='--')        # línea cero
ax.axvline(t_E, color='crimson', linewidth=1.2, linestyle=':')   # marcador t_E
ax.axhline(a_E, color='crimson', linewidth=1.0, linestyle='-.')  # marcador a_E
ax.axvline(t_vis, color=col, linewidth=1.0, linestyle='--')      # líneas de barrido R
```

---

## 6. Formateador científico del eje Y (aplicar a cada `ax`)

```python
import matplotlib.ticker as mticker

fmt = mticker.ScalarFormatter(useMathText=True)
fmt.set_scientific(True)
fmt.set_powerlimits((-2, 2))
ax.yaxis.set_major_formatter(fmt)
ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

# Reposicionar el texto de offset (etiqueta ×10^n)
off = ax.yaxis.get_offset_text()
off.set_size(14)
off.set_x(-0.12)
off.set_y(-0.05)
```

---

## 7. Eje X superior sincronizado — ap [mm] ↔ t [s]

Usar cuando se necesite un eje secundario no lineal. **No** usar `secondary_xaxis` para
perfiles no lineales — usar `twiny()` con un formateador manual.

```python
def add_top_xaxis_ap(ax, t, a_mm, mode="auto", tick_step=1):
    """
    Agrega eje X superior sincronizado: ap [mm] <- t [s].
    mode: 'linear' | 'step' | 'auto' (detecta segmentos constantes)
    """
    import numpy as np

    secax = ax.twiny()
    secax.set_xlabel(r"Depth $a_p$ [mm]")
    secax.set_navigate(False)

    t     = np.asarray(t, dtype=float)
    a_mm  = np.asarray(a_mm, dtype=float)

    def detect_mode(a):
        da = np.diff(a)
        if np.all(np.isclose(da, 0.0, rtol=1e-9, atol=1e-12)):
            return "step"
        if np.any(np.isclose(da, 0.0, rtol=1e-9, atol=1e-12)):
            return "step"
        return "linear"

    if mode == "auto":
        mode = detect_mode(a_mm)

    def map_t_to_ap(x):
        x = np.asarray(x, dtype=float)
        if mode == "step":
            idx = np.searchsorted(t, x, side="right") - 1
            idx = np.clip(idx, 0, len(a_mm) - 1)
            return a_mm[idx]
        return np.interp(x, t, a_mm)

    # Copiar ticks inferiores y reetiquetar con valores de a_p
    ax.figure.canvas.draw()
    lower_ticks = ax.get_xticks()
    xmin, xmax  = ax.get_xlim()
    visible     = lower_ticks[(lower_ticks >= xmin) & (lower_ticks <= xmax)]
    if tick_step > 1:
        visible = visible[::tick_step]

    secax.set_xlim(ax.get_xlim())
    secax.set_xticks(visible)
    ap_labels = map_t_to_ap(visible)
    secax.set_xticklabels([f"{v:.1f}" for v in ap_labels])
    return secax
```

---

## 8. Convenciones de leyenda

- Sin marco: `legend.frameon = False` (ya seteado vía rcParams)
- Gráfico de estabilidad: `loc='upper left'`
- Gráficos con eje twin: combinar handles de ambos ejes:
  ```python
  lines1, labs1 = ax.get_legend_handles_labels()
  lines2, labs2 = ax2.get_legend_handles_labels()
  ax.legend(lines1 + lines2, labs1 + labs2, loc='lower right')
  ```
- Gráficos de crecimiento/amplitud: `loc='upper left'`

---

## 9. Grid

Grid apagado por defecto:
```python
# ax.grid(False)   <- default, no requiere llamada explícita
```

---

## 10. Exportación

```python
fig.savefig(str(out_path), dpi=200, bbox_inches='tight')
```
- HTML interactivo: `fig.write_html(str(html_path), include_plotlyjs='cdn')` (solo Plotly)
- `dpi=300` para figuras finales de publicación, `dpi=200` para exportaciones intermedias.

---

## 11. Catálogo de gráficos (ChatterPlotter)

| Método | Eje X | Eje Y | Elementos clave | Preset de tamaño típico |
|--------|--------|--------|-------------|-------------------------|
| `plot_stability` | t [s] o ap [mm] | Re{λ_max} [s⁻¹] | fill_between azul/naranja+hatch, axvline t_E, scatter en el cruce | `FIGSIZE_SIMPLE` |
| `plot_depth_profile` | t [s] | ap [mm] | línea navy, axvline t_E, axhline a_E, marcadores R plasma | `FIGSIZE_SIMPLE` |
| `plot_visible_sweep` | Factor R | a_vis [mm] (izq.) / t_vis [s] (der.) | eje twin, 'o-' crimson + 's--' steelblue | `FIGSIZE_SIMPLE` |
| `plot_growth` | t [s] | G(t) [—] | línea darkorange, axhline G=0, scatter R plasma | `FIGSIZE_SIMPLE` o `FIGSIZE_WIDE` si la serie es larga |
| `plot_amplitude` | t [s] | e^G(t) [—] | darkorange, axhline 1.0, inset de zoom por R | `FIGSIZE_SIMPLE` |

Todos los métodos aceptan `ax=None` (crean sus propios ejes) o un `Axes` existente.

---

## 12. Plantilla de inicio rápido

```python
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── 1. Estilo (una sola fuente de verdad) ───────────────
from plot_style import ARTICLE_RCPARAMS, FIGSIZE_SIMPLE, FIGSIZE_WIDE, figsize_from_scale, _lang_text
plt.rcParams.update(ARTICLE_RCPARAMS)

# ── 2. Idioma y escala: variables editables por el usuario ──
LANGUAGE = "both"    # "EN" | "FR" | "both"
SCALE    = 1.0       # 1 = tamano original, 1.5, 2, 0.5, ...

# ── 3. Figura: elegir el preset segun destino ───────────
fig, ax = plt.subplots(figsize=figsize_from_scale(FIGSIZE_SIMPLE, SCALE), constrained_layout=True)
# o, para dos paneles a ancho de pagina completa:
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_from_scale(FIGSIZE_WIDE, SCALE), constrained_layout=True)

# ── 4. Datos ─────────────────────────────────────────────
ax.fill_between(t, re, 0, where=(re < 0),  alpha=0.15, color='#0072B2',
                label=_lang_text('Stable', 'Stable', LANGUAGE, sep=' / '))
ax.fill_between(t, re, 0, where=(re >= 0), alpha=0.15, color='#E69F00', hatch='//',
                label=_lang_text('Unstable', 'Instable', LANGUAGE, sep=' / '))
ax.plot(t, re, color='steelblue', linewidth=1.5, label=r'$\mathrm{Re}\{\lambda_{\max}\}$')
ax.axhline(0.0, color='k', linewidth=0.8, linestyle='--')
ax.axvline(t_E, color='crimson', linewidth=1.2, linestyle=':')
ax.scatter([t_E], [0.0], color='crimson', s=75, edgecolor='k', linewidths=0.6, zorder=5)

# ── 5. Formateador cientifico ────────────────────────────
fmt = mticker.ScalarFormatter(useMathText=True)
fmt.set_scientific(True); fmt.set_powerlimits((-2, 2))
ax.yaxis.set_major_formatter(fmt)
off = ax.yaxis.get_offset_text()
off.set_size(14); off.set_x(-0.12); off.set_y(-0.05)

# ── 6. Etiquetas y leyenda ────────────────────────────────
ax.set_xlabel(_lang_text('Time [s]', 'Temps [s]', LANGUAGE))
ax.set_ylabel(r'$\mathrm{Re}\{\lambda_{\max}\}$ [s$^{-1}$]')
ax.set_title(_lang_text('Frozen-time stability', 'Stabilité à temps figé', LANGUAGE))
ax.legend(loc='upper left')

fig.savefig('fig.png', dpi=300, bbox_inches='tight')
```
