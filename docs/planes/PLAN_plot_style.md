# PLAN — Estilo de figuras unificado (skill `article-plot-style`)

**Estado (2026-10-07):** diagnóstico y propuesta escritos (T2). El usuario dio el OK a T6 (vía el manager) con las opciones recomendadas de §6 como defecto; **implementación en curso, ver §9**.
**Instrucción del usuario:** unificar el estilo; el estilo de las gráficas debe ser el del skill `article-plot-style`; borrar todo rastro de estilo antiguo o heredado; si el skill está mal, decirlo.
**Skill de referencia:** `C:\Users\quiqu\.claude\skills\article-plot-style\SKILL.md` (el que se aplica). La copia del repo `.github/skills/article-plot-style/SKILL.md` es OTRA cosa (§1.4).

---

## 1. Inventario de fuentes de estilo (estado real, 2026-10-06)

### 1.1 Copias de `plot_style.py` (contenido de `ARTICLE_RCPARAMS` comparado con el skill §0 por programa)

| Copia | Diferencia con el skill | Extras propios |
|---|---|---|
| `DOE_utils/DOE_plots/plot_style.py` (canónica de hecho) | `lines.markersize` 10 (skill: 3) | `COLOR_STABLE/UNSTABLE/GRAY`, `HATCH_*`, `lang_text` (el skill la llama `_lang_text`); **sin** `FIGSCALE_SIMPLE`/`SCALE` ni `apply_sci_yaxis` |
| `indicators/maxent_sprt/.../viz/plot_style.py` | ninguna | `FIGSCALE_SIMPLE = 2`, `SCALE`, `COLORS`, `apply_sci_yaxis`; `plt.rcParams.update` al importar |
| `indicators/emd_hht/.../viz/plot_style.py` | ninguna | idem (sin `COLORS`) |
| `indicators/green_integral/.../viz/plot_style.py` | `markersize` 10 | idem, `COLORS` propio |
| `indicators/rms_cv/.../viz/plot_style.py` | `markersize` 10 | idem, `COLORS` propio |
| `indicators/ssq_chatter/.../viz/plot_style.py` | `markersize` 10 | idem, `COLORS` propio |
| `Optimizacion/study_phase3/plot_style.py` | **otro estilo entero**: `apply_research_style()`, fuentes 14/16/18, marcador 6, `PALETTE`, `make_label` | 11 scripts `plot_NN_*.py` lo importan |

Resumen: 6 copias del mismo concepto con 4 versiones distintas (3 vs 10 de `markersize`, `FIGSCALE` 2 vs 1.5 vs ausente, nombres `lang_text`/`_lang_text`/ninguno, `COLORS` distintos por paquete) y una séptima de otro estilo. El skill pide "una sola definición, sin copias que puedan divergir": hoy no se cumple.

### 1.2 Estilos locales heredados (definen sus propios `rcParams`)

| Fichero | Qué define | Origen |
|---|---|---|
| `DOE_utils/DOE_plots/doe_indicator_plotter.py`, `doe_noise_plotter.py`, `doe_model_snr_plotter.py`, `doe_plotter.py` | `configurar_estilo_global()` con `font.size` 9, títulos/etiquetas **25**, ticks/leyenda **23**, `linewidth` 2.0, `path.simplify` | skill antiguo `indicator-plot-style` (§1.4) |
| `effective_window/plotting/style.py` | `configure_global_style()` (9 / 15 / 12) | "portado de `Areas_Indicator_V1`" |
| `Convergency_Simulation/Etapa4_bis.py` (+ `Etapa_0..3`) | `_configurar_estilo_global()` y `rcParams` sueltos | scripts antiguos |
| `TDA/tda_chatter_h5.py`, `HMM/example_hmm_cone.py`, `DOE_utils/Lobes.py` | `configurar_estilo_global()` / `rc_context` propios | idem |
| `indicators/green_integral/examples/{DDE_signal_sources,Green_Integral_Lyapunov_Tutorial,augmented_trajectory_exploration,phase_area_indicator}.py` | `configurar_estilo_global()` / `configure_global_style()` copiados | ejemplos |
| `Optimizacion/study_phase1/study_phase1.py` | 8 usos de `rcParams` | script antiguo |
| `indicators/ssq_chatter/test_viz.py` | `rcParams` suelto (script de prueba) | script de prueba |

Los que ya usan el estilo canónico: `sld_model.py`, `validation_figures.py`, `label_grid.py` y `doe_unified_selector.py` (`rc_context(ARTICLE_RCPARAMS)`), `figures_window.py` (presets × escala), `indicator_plots.py` (`--scale 1.5`), y las figuras de los 5 paquetes de indicadores (con `FIGSCALE` 2).

### 1.3 Tamaños de letra y de lienzo a mano

- **Numéricos absolutos** (`fontsize=`/`labelsize=`/`size=`): `maxent_sprt_plots.py` 81 apariciones (16 ×44, 11 ×17, 8, 9, 10, 14…), `doe_unified_selector.py` 62 (paneles del visor en pantalla, no figuras de artículo), `doe_indicator_plotter.py` 21, `green_integral_plots.py` 20, `validation_figures.py` 17 (7, 8, 9, 14), `sst_svd_plots.py` 17, `rms_cv_plots.py` 14, `doe_selector.py` 12, `doe_noise_plotter.py` 12, `sld_model.py`, `label_grid.py`, `doe_model_snr_plotter.py` 9, `doe_plotter.py` 9, `effective_window/plotting/*` ~24, `Convergency_Simulation/*` ~33, ejemplos y `legacy/` de los paquetes.
- **`figsize=(…)` literales** fuera de los presets: más de 100 usos, sobre todo en `Convergency_Simulation`, `runner_lyapunov.py` (13), `legacy/`, `examples/`, `effective_window` y 5 en `doe_unified_selector.py`.
- Efecto: con `rc_context(ARTICLE_RCPARAMS)` el texto "del estilo" (12/16/14/10) y el texto "a mano" (7–16) conviven; el segundo no responde a ningún control.

### 1.4 Skills y textos que describen estilos viejos

- `.github/skills/article-plot-style/SKILL.md` en el repo **no es** el skill actual: su `name:` es `time-plot-style`, no tiene escala/idioma/paleta daltónica y define sus propios `rcParams` (`local_style`). Es la versión "original" a la que se refiere el skill actual.
- `.github/skills/indicator-plot-style/SKILL.md`: manda copiar `configurar_estilo_global()` **tal cual** a cada módulo (letras 25/23). Contradice a `article-plot-style`; es el origen de los 4 `doe_*_plotter.py` y de los ejemplos.

### 1.5 Texto: `figure_texts.yaml` / `fig_lang.py`

Traducen EN/FR sobre una **copia exportada** (`fig_lang.translate_figure`), con la tabla `figure_texts.yaml`; "both" sigue `plot_style.lang_text`. No definen estilo (tamaños) y no hay conflicto; solo hay que mantener `lang_text` como único formato "both" (y el alias `_lang_text` del skill, §2).

---

## 2. Diferencias con el skill y observaciones sobre el skill

Lo que está **mal en el código** respecto al skill:

1. `markersize` 10 en 4 copias (skill: 3, con marcadores de evento explícitos `s=75`).
2. `FIGSCALE_SIMPLE` 2 en los paquetes (skill: 1.5) y ausente en la canónica (cada módulo define el suyo: `FIGSCALE = 1.5` en `sld_model`/`validation_figures`/`indicator_plots`).
3. Nombre del helper: `lang_text` (canónica) frente a `_lang_text` (skill).
4. Estilos 25/23, 9/15/12 y 14/16/18 (§1.2) no son del skill.
5. `plt.rcParams.update(ARTICLE_RCPARAMS)` **al importar** en los paquetes: cambia el estilo global de cualquier proceso que importe el paquete (la app usa `rc_context` justamente para evitarlo).
6. `COLORS` distintos por paquete (el skill solo fija estable/inestable y colores por tipo de curva).

Lo que el **skill** dice y conviene señalar (para que el usuario decida; no se edita):

- A. **El texto no tiene regla de escala.** El skill escala solo el lienzo (`figsize_from_scale`); las fuentes son puntos absolutos. A escala 1.5 (su propio valor por defecto) el texto resulta 1/1.5 de lo "natural"; a 2, la mitad (§3).
- B. §6/§12 fijan `offset.set_size(14)` a mano (absoluto) → mismo problema que A.
- C. Su §10 exporta con `dpi=200` mientras su §0 pone `savefig.dpi` 300; y `savefig.transparent: True` da PNG con fondo transparente (se ven negros en algunos visores). No son errores, pero son dos reglas para lo mismo.
- D. Cita `_lang_text` (privado) como API compartida: un nombre con guion bajo no debería importarse entre módulos. Propuesta: público `lang_text` y alias `_lang_text`.
- E. §1 dice "sin copias que puedan divergir" pero no dice qué hacer cuando un paquete instalable no puede importar de otro (caso de `indicators/*`): §4.
- F. No define `apply_sci_yaxis`, `COLORS` ni gris (zona gris de las etiquetas): los paquetes y la canónica lo inventaron por separado.

---

## 3. Texto y escala

Hoy: `figsize_from_scale(base, s)` multiplica el lienzo; las fuentes siguen en 12/16/14/10 pt → al exportar a escala mayor el texto parece más pequeño. El usuario espera que el texto acompañe a la talla.

**Propuesta (no toca el skill):**

1. En la canónica, `rc_scaled(scale, follow_text=True) -> dict`: devuelve `ARTICLE_RCPARAMS` con **todas las claves de tamaño** multiplicadas por `scale / FIGSCALE_SIMPLE` (la referencia es la escala por defecto del skill, 1.5: sus puntos están pensados para ese lienzo; con referencia 1 a escala 1.5 el texto salía 1.5× mayor y se solapaba, comprobado con renders) (`font.size`, `axes.titlesize`, `axes.labelsize`, `xtick/ytick.labelsize`, `legend.fontsize`; y, para que la proporción se mantenga, `lines.linewidth`, `lines.markersize`, `axes.linewidth`, grosores y largos de ticks). Con `follow_text=False` devuelve los puntos del skill sin tocar ("fijo"). Uso: `with plt.rc_context(ps.rc_scaled(scale)):`.
2. Con tamaños **relativos** (§5) todo texto "a mano" sigue a `font.size` y por tanto a la escala sin más cambios.
3. Figuras que no son nativas (visores del selector, `doe_*_plotter` en pantalla): en la copia exportada, `fig_lang`/`figures_window` escalan los objetos `Text` de la copia por `scale` (`for t in fig.findobj(Text): t.set_fontsize(t.get_fontsize()*scale)`), de modo que "sigue a la escala" también funciona donde no hay `rc_scaled`.
4. Ventana de exportar (`figures_window.py`, wt-interfaz): control **"texto"**: *sigue a la escala* (por defecto, como un zoom) | *fijo* (puntos del skill). Por defecto cambia el aspecto de lo exportado a escala ≠ 1: es lo que el usuario pidió, pero hay que decirlo en `TUTORIAL.md`.
5. Equivalencia de referencia: a escala 1.5 (`FIGSCALE_SIMPLE`) el resultado es idéntico al del skill y al de hoy; a escala 3 el texto de 12 pt pasa a 24 pt sobre un lienzo 2× mayor, es decir, **mismo aspecto relativo**; a escala 0.75 pasa a 6 pt. `follow_text=False` deja los puntos del skill a cualquier escala. Los tamaños a mano (marcadores, grosores) usan `ps.zoom(scale, follow)` con el mismo factor.

Si el usuario quiere que esta regla viva en el skill (decisión 4 de `PLAN_ronda3.md` §4), sería una sección nueva "escala del texto" con exactamente §3.1–3.5; el skill no se toca hasta entonces.

---

## 4. Propuesta de arquitectura

- **Fuente canónica única:** `DOE_utils/DOE_plots/plot_style.py`. Contenido: `ARTICLE_RCPARAMS` **exactamente el del skill** (`markersize` 3), `FIGSIZE_SIMPLE/WIDE`, `FIGSCALE_SIMPLE = 1.5`, `figsize_grid`, `figsize_from_scale`, `rc_scaled`, `lang_text` (+ alias `_lang_text`), paleta Okabe-Ito (`COLOR_STABLE/UNSTABLE/GRAY`, `HATCH_*`), `apply_sci_yaxis`. **Sin efectos al importar** (nada de `rcParams.update` a nivel de módulo; una función `apply()` explícita) y solo `matplotlib` como dependencia (para que pueda copiarse tal cual).
- **Paquetes de indicadores** (instalables; no deben importar de `DOE_utils`): copia **byte a byte idéntica** a la canónica en `…/viz/plot_style.py` de los 5, escrita por un script `tools/sync_plot_style.py` (copia canónica → 5 destinos; `--check` solo compara hashes) y **selftest** `check_plot_style_sync.py` que falla si alguna copia difiere. Los módulos de figura de cada paquete llaman a `ps.apply()` (o `rc_context`) donde hoy dependen del efecto al importar.
- `COLORS` por paquete: se conservan si son colores de **curva** de ese indicador, pero salen de `plot_style` y pasan a un `viz/colors.py` propio del paquete (el estilo no los define).
- `Optimizacion/study_phase3/plot_style.py` (otro estilo): decisión del usuario (§6.3): migrar a la canónica o conservarlo fuera de la unificación si no alimenta la tesis.
- **Estilos heredados (§1.2):** se borra `configurar_estilo_global()`/`configure_global_style()` y el `plt.rcParams.update({...})` local; el fichero pasa a `with plt.rc_context(ps.rc_scaled(SCALE))` o `ps.apply()`. No se borran los scripts, solo su definición de estilo.
- **Skills del repo:** `.github/skills/indicator-plot-style` y `.github/skills/article-plot-style` (la vieja, `time-plot-style`) se sustituyen por un `README` de una línea que apunta al skill `article-plot-style` (con OK del usuario; no se borran sin él).

## 5. Tamaños de letra relativos

`font.size` = 12 pt es la base; los nombres relativos de matplotlib se calculan sobre ella (`xx-small` 0.58, `x-small` 0.69, `small` 0.83, `medium` 1, `large` 1.2, `x-large` 1.44). Tabla de conversión propuesta para los números a mano:

| Hoy (pt) | Pasa a | Valor a escala 1 |
|---|---|---|
| 7 | `'xx-small'` | 6.9 |
| 8, 9 | `'x-small'` / `'small'` | 8.3 / 10 |
| 10, 11 | `'small'` | 10 |
| 12 | `'medium'` (o se quita el argumento) | 12 |
| 13, 14 | `'large'` | 14.4 |
| 16 | `'x-large'` | 17.3 |
| etiquetas/títulos a 16 | quitar el argumento (lo da `axes.labelsize`/`titlesize`) | 16 |

Regla: si el número coincide con un tamaño que ya da `ARTICLE_RCPARAMS`, se **quita** el argumento; si no, nombre relativo. Los paneles del visor en pantalla (`doe_unified_selector.py`, `doe_selector.py`) **no** se tocan: no son figuras de artículo; sus copias exportadas pasan por §3.3.

## 6. Decisiones del usuario antes de implementar (T6)

1. **`markersize`**: pasar a 3 (el del skill) en toda la canónica y las copias → revisar que los marcadores de línea (`'o-'`) sigan visibles; los de evento ya llevan `s=` explícito. ¿OK?
2. **Escala por defecto**: 1.5 (skill) en todo, en lugar de 2 en los paquetes. ¿OK? (cambia el tamaño de las figuras de los paquetes de indicadores).
3. **`Optimizacion/study_phase3`** (y `effective_window`, `Convergency_Simulation`, `TDA`, `HMM`, `Lobes`, ejemplos y `legacy/`): ¿cuáles alimentan la tesis/artículo (se migran) y cuáles son archivo (se dejan, marcados "estilo heredado, sin migrar")?
4. **Texto y escala** (§3): ¿"sigue a la escala" por defecto? ¿Y se amplía el skill con esa regla (decisión 4 de la ronda 3)?
5. **Skills del repo** (§4): ¿sustituir `.github/skills/{indicator,article}-plot-style` por un puntero?
6. **Propiedad** (preguntar al manager): `doe_plotter.py` y `doe_model_snr_plotter.py` no figuran en el reparto de §0 de `PLAN_ronda3.md`; `doe_indicator_plotter.py` y `doe_noise_plotter.py` son de wt-interfaz.

## 7. Reparto de ficheros y fases (cada fase: hash antes/después + selftests)

| Fase | Qué | Quién | Ficheros | Comprobación |
|---|---|---|---|---|
| F0 | Congelar línea base: renders de referencia + hash de datos de las figuras (ver abajo) | wt-validacion | solo lectura | `validacion_figs\style_baseline\` |
| F1 | Canónica nueva (`rc_scaled`, `FIGSCALE_SIMPLE`, `lang_text`+alias, `apply_sci_yaxis`, sin efectos al importar) + `sync_plot_style.py` + `check_plot_style_sync.py` | wt-validacion | `plot_style.py`, `tools/` | `python plot_style.py` selftest; sync `--check` |
| F2 | Copias idénticas en los 5 paquetes y limpieza de `COLORS` (a `viz/colors.py`) + `ps.apply()` donde hacía falta | wt-validacion (los paquetes no están en el reparto de nadie; ver 6.6) | `indicators/*/viz/plot_style.py`, `colors.py`, módulos `viz/*` | selftests de cada paquete; `indicator_plots.py` genera las figuras de cada paquete y se compara con F0 |
| F3 | Tamaños relativos en las figuras de validación y el SLD | wt-validacion | `validation_figures.py`, `sld_model.py`, `label_grid.py` (de wt-interfaz: solo si lo pide) | selftests propios; renders vs F0 |
| F4 | Borrar estilos heredados de los `doe_*_plotter` y `configurar_estilo_global` | wt-interfaz (plotters suyos) / wt-validacion (resto de `DOE_plots`) | ver §1.2 | `check_app_dialogs.py`; renders |
| F5 | Control "texto" en la ventana de exportar + escalado de `Text` en la copia + `TUTORIAL.md` | wt-interfaz | `figures_window.py`, `fig_lang.py` (de wt-validacion, función `scale_text`) | selftest de `figures_window`; export a escala 1/1.5/2 con "sigue"/"fijo" |
| F6 | Migrar o marcar legados (§6.3); punteros de skills (§6.5); README | ambas, tras decisiones | `effective_window`, `Optimizacion`, … | grep de control (abajo) |

**Verificaciones transversales**

- **Ningún resultado numérico cambia:** F0 guarda, por figura, un hash de los datos dibujados (`ax.lines[i].get_xydata()`, barras, imágenes) y F1–F5 lo recomparan: solo cambian estilos, no datos.
- **Render real:** `indicator_plots.py` (genera las figuras de cada paquete) y `validation_figures.py` sobre un conjunto pequeño; inspección visual en `validacion_figs\style_after\` vs `style_baseline\`. Cuidado con las figuras de MaxEnt/SST (piden varios GB al dibujar).
- **Lint (selftest):** falla si fuera de la canónica y de los legados marcados aparece `rcParams.update(`, `def configurar_estilo_global`, `def configure_global_style` o `fontsize=<número>` en los ficheros migrados.
- **Sincronía:** `check_plot_style_sync.py` (hash de las 5 copias == canónica).

## 8. Fuera de alcance

Colores de las curvas de cada indicador, contenido de `figure_texts.yaml`, formato de ejes de los visores en pantalla y cualquier cambio de datos o de algoritmos.

## 9. Estado de la implementación (T6, 2026-10-07)

Decisiones de §6 tomadas con las opciones recomendadas (markersize 3, escala por defecto 1.5, texto "sigue a la escala", etc.). `DOE_utils/DOE_plots/check_plot_style.py` es el selftest de todas las fases: `--render DIR` (dibuja 33 figuras reales: 15 de validación limpia n12000, 15 de un nivel de ruido, 3 de resumen), `--compare A B` (huella SHA-1 de los DATOS dibujados: líneas, barras, colecciones, imágenes), `--sync`, `--lint`, `--selftest`.

| Fase | Estado | Verificación |
|---|---|---|
| F0 línea base | hecha: `validacion_figs\style_baseline\` (33 figuras + `digests.json`, determinista: dos dibujos dan la misma huella) | |
| F1 canónica | hecha: `plot_style.py` (rcParams del skill con markersize 3, `FIGSCALE_SIMPLE`/`SCALE` 1.5, `rc_scaled`, `zoom`, `apply`, `apply_sci_yaxis`, `lang_text` + alias `_lang_text`, sin efectos al importar) | selftest; 33/33 huellas iguales; PNG idénticos |
| F3 validación + SLD | hecha: `validation_figures.py`, `sld_model.py` (letra relativa, marcadores/grosores con `_k()`, `rc_scaled`; ganchos `TEXT_FOLLOWS` y `set_style(language, scale, follow_text=True)`) | 33/33 huellas iguales; 12/33 PNG idénticos y el resto con diferencia media < 0.05 (nombres relativos: 9 pt → `small` 10 pt); selftests de ambos; `--lint` 0 |
| F5 (parte de wt-validacion) | hecha: `fig_lang.scale_text(fig, k)` para figuras no nativas | selftest |
| F2 copias en los 5 paquetes | **hecha (2026-10-07, tras T1)**: `viz/plot_style.py` de los 5 = copia byte a byte de la canónica (`check_plot_style.py --sync-write` / `--sync`); la paleta de cada paquete pasa a `viz/colors.py`; el estilo se aplica con `apply()` desde los módulos de figura (antes era efecto secundario al importar `plot_style`; MaxEnt perdió su `configurar_estilo_global`). Cambia a propósito: escala por defecto 2 → 1.5 y `markersize` 10 → 3 en Green/RMS-CV/SSQ | 46/46 huellas de datos iguales (`indicator_plots.py --pickle-dir` sobre case_007 de n12000: rms_cv 9, green 9, maxent 21, ssq 7 figuras); tests de estilo de rms_cv, green y ssq pasan; el de MaxEnt **ya fallaba antes** (figura D3: ancho 14.32 en vez de 14.0, mismo factor 1.023 en las otras anchas) y sigue igual; import de los 5 OK |
| F2b tamaños de letra a mano en los paquetes | **hecha (opción A, aprobada por el usuario 2026-10-07)**: de 124 tamaños numéricos en los 7 módulos `viz`, 16 eran exactamente un tamaño del estilo y pasaron a `'small'`/`'medium'`; **los otros 108 quedan marcados** `# tamaño a mano: anotación densa` (el lint los admite). Ninguno se pudo quitar: los 16 y 14 de los paquetes están en anotaciones de texto (`text`, líneas verticales), no en etiquetas ni títulos, y el tamaño de un texto por defecto es 12; mi estimación de §10.1 ("más de la mitad se quitan") era optimista. Mejora posible (A+, no hecha): sustituir esos 16/14 por `plt.rcParams["axes.labelsize"]` / `["xtick.labelsize"]` (exacto y sigue a la escala) | 46/46 huellas de datos y 46/46 PNG idénticos píxel a píxel; tests de estilo como antes; `--lint` 0 en 11 ficheros |
| F4 `doe_*_plotter` | pendiente (wt-interfaz; `doe_plotter.py` y `doe_model_snr_plotter.py` también) | |
| F5 (parte de wt-interfaz) | pendiente: control "texto" en `figures_window.py` y `TUTORIAL.md` | contrato abajo |
| F6 legados | **hecho a la manera del usuario (2026-10-07): no se migra nada y la organización de carpetas queda como está**; 16 ficheros que definen su propio estilo llevan `# estilo heredado, sin migrar (PLAN_plot_style.md F6)` sobre su definición (`Convergency_Simulation/Etapa_0..3, Etapa4_bis`, `Lobes.py`, `HMM`, `TDA`, `Optimizacion/study_phase1` y `study_phase3/plot_style.py`, `effective_window/plotting/style.py`, 4 ejemplos de Green, `ssq_chatter/test_viz.py`); `grep "estilo heredado"` lista lo que queda | `ast.parse` de los 16 |

**Contrato para wt-interfaz (F5):** antes de dibujar, la ventana de exportar llama `validation_figures.set_style(language, scale, follow_text)` y `sld_model.set_style(...)` (en vez de asignar `LANGUAGE`/`FIGSCALE`; asignar sigue funcionando y deja `follow_text=True`). Para una figura NO nativa (paneles del visor, `doe_*_plotter`): tras `fig_lang.translate_figure`, `fig_lang.scale_text(fig, plot_style.zoom(scale, follow_text))`. Defecto del control "texto": "sigue a la escala". A la escala por defecto 1.5 el resultado es el de hoy.

## 10. Recomendación para las dos decisiones pendientes (F2b y F6) — para decidir rápido

### 10.1 F2b — tamaños de letra a mano dentro de los paquetes de indicadores

**Hechos medidos** (figuras reales de `case_007` de n12000, tamaños de todos los textos de cada figura; 46 figuras: MaxEnt 21, Green 9, RMS-CV 9, SSQ 7):

- Los números a mano que **ya son un tamaño de `ARTICLE_RCPARAMS`** en su papel (16 en etiquetas/títulos, 14 en ticks, 10 en leyendas, 12) no cambian nada si se quitan o se pasan a su nombre exacto (`'small'` = 10.0 pt, `'medium'` = 12.0 pt). Son más de la mitad de las apariciones.
- Los demás sí cambian al pasar a un nombre relativo: 7 → `'xx-small'` (6.9, −1 %), 8 → `'x-small'` (8.3, +4 %), 9 → `'small'` (10, +11 %), 11 → `'small'` (10, −9 %), 13 → `'large'` (14.4, +11 %), 14 → `'large'` (14.4, +3 %).
- **Figuras que se verían distintas con la opción B** (las que usan 7, 8, 9, 11 o 13; con la A ninguna cambia): MaxEnt **F0a, F0b, F1, F2, F6a, F6b** (solo el 8 → +4 %: una anotación), **D3** (11 y 13), **D6** (8, 9, 11, 13), **D7, D8** (8 y 11), **D9** (7, 7.5, 8, 11, 13: la más cargada) y **D_JOINT** (8 y 11); SSQ **C3** (un texto de 9 pt); Green la FFT de `plots_signal_diagnostics` (9 pt; no sale en `case_007`). El resto (MaxEnt O3–O5, F3–F5, F7, S1, D4; Green C1–C3, C6, C7, Ĝ; RMS-CV todas; SSQ C1, C2, C4–C7) **no cambia** con la opción A.

| Opción | Qué hace | Pros | Contras |
|---|---|---|---|
| **A (recomendada)** | Quitar los tamaños que ya son los del estilo y pasar a nombre exacto los que lo son (10, 12); **dejar** los 7/8/9/11/13 de las anotaciones densas de MaxEnt D3–D9/D_JOINT, SSQ C3 y Green FFT, cada uno marcado con `# tamaño a mano: anotación densa` y permitidos en el lint de los paquetes | Cero cambio visual en las 46 figuras de `case_007` (comprobable: PNG idénticos píxel a píxel, no solo datos); retira el grueso del rastro; no retoca las figuras de diagnóstico ya afinadas a mano (las 13 de la lista de arriba conservan su aspecto) | Quedan ≈ 40–50 números a mano en los paquetes (ver la lista); esos textos no siguen a la escala *dentro* del paquete (sí al exportar, con `fig_lang.scale_text`) |
| B | Convertir TODO a nombres relativos (la regla de §5) | Sin números a mano; el lint estricto también en los paquetes | Cambia el aspecto de 12 figuras de MaxEnt, SSQ C3 y Green FFT (±11 %), con anotaciones densas que pueden solaparse; hay que revisar cada una a ojo |
| C | No hacer F2b; solo marcar | Cero riesgo y cero trabajo | El texto de los paquetes no sigue a la escala dentro del paquete; ≈ 200 números a mano siguen ahí; contradice "borrar todo rastro" |

Esfuerzo: A ≈ 1–1.5 h (semi-manual: hay que mirar cada llamada: `legend(fontsize=16)` no equivale a una etiqueta de 16), con renders de las 4 variantes y comparación de PNG; B ≈ 3 h más la revisión visual por el usuario; C 0.
**Recomendación: A**, y B solo si el usuario quiere cero números a mano y acepta revisar las figuras D de MaxEnt.

### 10.2 F6 — scripts heredados que definen su propio estilo

Hechos: `Convergency_Simulation` (6 scripts) guarda `.jpg` **y** `.pdf` a 300 dpi (calidad de publicación) y es el más activo (último commit 2026-10-05); `HMM/example_hmm_cone.py` (2026-09-30) y `DOE_utils/Lobes.py` (2026-10-01, no lo importa ningún otro fichero del repo) son recientes; `TDA`, `Optimizacion/study_phase1`, `Optimizacion/study_phase3` (22 ficheros, 11 que importan su `plot_style` propio) y `effective_window` (14 ficheros) guardan a 150 dpi o no guardan y llevan sin tocarse desde 2026-09-25; `indicators/ssq_chatter/legacy` (22), `indicators/emd_hht/legacy` (3, abril) y los `examples/` de los paquetes son archivo o demostraciones.

| Opción | Qué hace | Pros | Contras |
|---|---|---|---|
| **A (recomendada)** | Migrar solo lo que el usuario confirme que alimenta una figura de la tesis o del artículo (candidatos por actividad y dpi: `Convergency_Simulation`, `Lobes.py`, `HMM`); el resto se marca con una línea `# estilo heredado, sin migrar (PLAN_plot_style.md F6)` y el lint lo lista como "marcado" | Retoca solo lo que importa; cada migrado se verifica rehaciendo la figura; un `grep "estilo heredado"` da la lista honesta de lo que queda | El rastro de estilo antiguo sigue (marcado) en ≈ 50 ficheros de archivo |
| B | Migrar todo (≈ 60 ficheros: `effective_window`, `Optimizacion`, `TDA`, `legacy`, ejemplos) | Ningún estilo antiguo en el repo | Muchos scripts no se pueden rehacer (sin datos o dependencias antiguas), así que no se puede verificar que no cambian; riesgo de romper cosas que nadie usa; ≈ 1–2 días |
| C | Borrar los estilos antiguos de los scripts de archivo sin migrarlos | Código más limpio | Rompe scripts de archivo y es irreversible fuera de git; **no recomendado** |

Esfuerzo: A ≈ 30 min por script migrado (si hay datos para rehacerlo) + 15 min de marcado de los demás; B ≈ 1–2 días; C no.
**Recomendación: A.** Pregunta concreta para el usuario: ¿qué figuras de la tesis salen de `Convergency_Simulation`, `HMM` y `Lobes.py`, y de `Optimizacion/study_phase3` (sus 11 figuras)?

### 10.3 Revisión de "estilo heredado sin marcar"

Se buscó por programa (`rcParams`, `rc_context`, `style.use`, `def configurar_estilo_global | configure_global_style | apply_research_style`) en todo el repo. Todo lo que define estilo propio está en uno de estos grupos: canónica y migrados (`plot_style.py`, `validation_figures.py`, `sld_model.py`, `label_grid.py`, `doe_unified_selector.py`, que usan `rc_context(ARTICLE_RCPARAMS)`), F4 (los 4 `doe_*_plotter.py`: wt-interfaz), F6 (`Convergency_Simulation`, `Lobes.py`, `HMM`, `TDA`, `Optimizacion`, `effective_window`, 4 ejemplos de Green, `ssq_chatter/test_viz.py`) y los dos tests de estilo de los paquetes (`test_article_plot_style.py`, que usan `rc_context` para comprobar y no definen estilo). **Único hueco encontrado y corregido: `ssq_chatter/test_viz.py` no estaba en el inventario (§1.2).** Los `.github/skills/*` de §1.4 quedan para la decisión 5 de §6 (puntero al skill).

### 10.4 Decisiones del usuario (2026-10-07) y su ejecución

F2b = A (hecha, ver la tabla de §9: 108 de 124 tamaños quedan marcados); F6 = no migrar, solo marcar (hecho); skill `article-plot-style` ampliado con la escala del texto (`rc_scaled` / `zoom`, "sigue a la escala" con referencia `FIGSCALE_SIMPLE` = 1.5 y opción "fijo"; las dos copias son hoy el mismo fichero; la copia del repo era el skill antiguo `time-plot-style`). **Pendiente de confirmar: el `kappa` de `augmented_trajectory_exploration.py` NO es η**: es la curvatura de Frenet-Serret κ_n = ||r' × r''|| / ||r'||³ de la trayectoria r(t) = [x, v, ap] (con la torsión τ al lado), no Ap/AP_REF; convertirlo a η haría que η signifique dos cosas. Recomendación: dejarlo como κ. No se tocó.
