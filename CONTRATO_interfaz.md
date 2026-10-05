# CONTRATO_interfaz — cambios de wt-interfaz que afectan al merge

Rama `wt-interfaz` (nacida de a868948). Solo se anota lo que toca scripts de etapa / `DOE_plots/` o cambia lo que
`experiment.py` pasa a los scripts.

## 1. Visor de indicadores (`DOE_plots/`) — commit `fix(viewer)`
Excepción acordada con el manager a la regla "no tocar DOE_plots/". Solo estos dos archivos, cambios mínimos:

| Archivo | Cambio | Por qué |
|---|---|---|
| `doe_indicator_plotter.py` `load_indicator_results` | `Axial_acc` en `_SIGNAL_NAMES`; se saltan los grupos sin `I_t` | `Axial_acc` se leía como una corrida de indicador vacía → checkbox falso "Axial" |
| `doe_unified_selector.py` `_build_tree` | columnas `t_d` = las corridas que tienen `t_d` (si ninguna, todas), máx. 4 | antes eran las 4 primeras por nombre y SST (`ssq`) quedaba fuera |
| `doe_unified_selector.py` `_it_plot_yscale` | `"ssq"` en `log_prefixes` | las variantes se llaman `ssq_*`, no `sst_svd`; con `I(t)` de 0.01 a 1000 se veía plano |

Merge: `wt-validacion` solo añadió `DOE_plots/validation_figures.py` (archivo nuevo); no deberían chocar.

## 2. `early_tol_s` eliminado (lado de la app) — commit `feat(experiments)`
Contrapartida de `wt-validacion` (de35d5d: `validate_indicators.py` sin tolerancia). Esta rama no tiene aún ese commit,
así que los cambios son tolerantes a ambos lados.

| Qué | Antes | Ahora |
|---|---|---|
| Llamada a `validate_indicators.py` (`experiment.py`, `_all_stages`) | `... --channel C --early-tol T` | `... --channel C` (sin `--early-tol`; el script lo sigue aceptando e ignorando) |
| Sección `validate` del YAML | `{channel, early_tol_s}` | `{channel}`; un `early_tol_s` que traiga un YAML antiguo se ignora (no entra en la huella ni en el comando) y el formulario de Validate lo borra al guardar |
| `ValidateForm` (launcher) | campo `early_tol_s` | solo el canal |
| `EARLY_TOL_S` (experiment.py) | constante | eliminada |
| `METRIC_COLUMNS` (Compare) | con `early_alarm_rate`, `ramp_anticipated_rate` | sin ellas; Compare y el panel leen con `.get`, así que `.h5` viejos (con esas métricas) o nuevos (sin ellas) abren igual |
| Texto del panel de Validate | regla con tolerancia, "anticipated" | regla nueva: constante inestable = TP con cualquier alarma, FN si ninguna; rampa que cruza = alarma antes del inicio → FA, después → TP |

Efecto esperado: la huella de Validate cambia, así que las validaciones ya hechas quedan en naranja (*stale*); es lo
correcto, hay que re-correrlas con la regla nueva. No se tocó `doe_indicators.py` (su texto de tolerancia lo cambia wt-validacion).
Docs actualizados: `TUTORIAL.md`, `PLAN_ramps.md` (nota al inicio), ayuda del launcher.

## 3. Label grid (rejilla de datos etiquetados) — commit `feat(experiments)`
Archivo nuevo `DOE_utils/label_grid.py` + botón **Label grid** en la etapa Label build del launcher. Solo **lee** el
`reference_dataset*.h5` de Label build (`stable|gray|unstable/<case>/<canal>__NNN/{t,y}` y sus attrs `kappa`,
`labeling_*`); no cambia ningún argumento ni script de etapa, ni escribe nada. Lanzado como el visor
(`launch("gui", "label_grid.py", ["--h5", <label.out>])`). Los límites ±lim_inf/±lim_sup salen de los mismos attrs
`labeling_*` que usa `doe_unified_selector._amp_limits` (misma regla; se reimplementa en 5 líneas, no se importa).
Merge: archivo nuevo, no choca con wt-validacion.

## 4. Rampas
Por indicación del usuario no se trabaja en rampas por ahora. Las columnas `ramp_*` (Compare, panel de Validate)
llevan la marca *provisional* (el criterio de rampa es consecuencia de quitar `early_tol`). La rejilla solo muestra
las rampas (título `ramp Ap0 -> Ap1 mm`), sin lógica propia.

## 5. Nota de edición
Los scripts de parche con Python en Windows reescriben a CRLF; el repo está en LF. Hay un commit `chore` que lo
revierte; el diff neto contra a868948 es solo contenido.

## 6. Figuras de validación en el visor y en Compare — commit `feat(experiments)`
Tras sincronizar con REPO-Utils y wt-validacion (merges sin conflicto). Se usa la API de
`DOE_plots/validation_figures.py` (de wt-validacion, no se toca): `FIGURES = {nombre: fn(h5_path, out_dir=None) -> Figure}`
con `fig._keep_size`, y `fig_compare(h5_a, h5_b, out_dir=None)`.

| Archivo | Cambio |
|---|---|
| `DOE_plots/doe_unified_selector.py` (excepción acordada) | `_is_validation_h5(path)` (hay grupo `/ranking`); `_make_summary_entries` añade una entrada `Validation — <nombre>` por cada figura de `FIGURES` si el `.h5` es de validación; `_save_summary` guarda en `<carpeta del .h5>/figs_validation/` para esos archivos (y en `figs_indicators/` para el resto, como antes). Este archivo está en CRLF en el repo y se mantiene así |
| `launcher.py` pestaña Compare | botón **Plot**: `validation_figures.fig_compare(h5_A, h5_B, out_dir=<carpeta de A>/figs_validation)`, se muestra en una ventana y queda guardado como `compare_<A>_vs_<B>.png`. Si falta el resultado de alguno, aviso (no error); si la figura lanza excepción, se muestra el error |
| `check_app_dialogs.py` | comprueba el aviso de Plot sin resultados |

Una figura sin datos (p. ej. `training_coverage` sin `/training`) lanza excepción y el visor la muestra como error (ya lo hacía `_refresh_summary`). Hay 12 figuras en `FIGURES` + `fig_compare`. Solo Ap constante.
Verificado sobre un `.h5` de validación sintético (el del selftest de validation_figures) en el scratchpad: 12/12 figuras se
generan en el visor, el guardado va a `figs_validation/`, y Plot abre la ventana y guarda `compare_*.png`.

### 6b. Corrección (visor)
Primera versión de §6: las entradas `Validation — …` no salían en el desplegable de "Summary plots", porque el visor solo
lista las que empiezan por `SLD` cuando existen (`_build_right_panel`). Ahora el filtro admite `Validation` y `SLD`; en un
archivo de validación las figuras `Validation — …` van **primero** (y "ranking" queda preseleccionada).
Además, las entradas SLD (todos los modos / modo / outcome por indicador) solo se generan para el/los modelo(s) con que se
simularon los casos (`sim_model` de los grupos del `.h5`, helper `_sim_models`); si el archivo no lo dice, se mantienen todos
los presets como antes. Verificado abriendo el visor (Tk) sobre `ramp_check/doe_validation_results.h5` real, solo lectura:
12 entradas Validation + 5 SLD de `1DOF_150`.

## 7. Ventana "Figures…" del visor (reemplaza lo de §6/6b para las figuras no-referencia)
Por indicación del usuario: el panel derecho del visor ("Summary plots") conserva solo las curvas de referencia (SLD y
outcome por indicador, del modelo usado); todo lo demás va a una ventana nueva con más espacio y control.
- `DOE_plots/figures_window.py` (archivo nuevo, mío): `FiguresWindow(parent, title, labels, render, style, out_dir, language, scale)`.
  Lista de figuras, lienzo grande con la barra de matplotlib, controles de idioma EN/FR/both, escala (FIGSCALE), proporciones
  de artículo (mantiene `_keep_size`), dpi (200/300/600), formato (png/pdf/svg), Save, Save all, Open folder. No sabe nada de las
  figuras: recibe `render(label) -> Figure` (lanza si no hay datos; también se detectan los errores que solo salen al dibujar) y
  `style(language, scale)`.
- `DOE_plots/doe_unified_selector.py` (excepción acordada, CRLF): botón **Figures…** en el panel derecho;
  `_make_summary_figure` (lógica sacada de `_refresh_summary`, la comparten el panel y la ventana); `_open_figures_window`
  (todas las entradas que no empiezan por `SLD`: las `Validation — …` de §6 y las de t_d / I(t)); `_fig_style` / `_apply_fig_style`
  fijan `LANGUAGE` y `FIGSCALE` de `validation_figures` y `sld_model` (efecto colateral: el SLD del panel también cambia de idioma).
  El filtro del desplegable vuelve a ser solo `SLD` (§6b queda superado en eso; el filtro por modelo usado, `_sim_models`, se mantiene).
- Carpeta de guardado: `figs_validation/` si el `.h5` tiene `/ranking`, si no `figs_indicators/`.

### Auditoría de `validation_figures.py` contra article-plot-style (no se modificó; observaciones para wt-validacion)
Cumple: `ARTICLE_RCPARAMS` única (vía `plot_style`, en `rc_context`), tamaños `FIGSIZE_SIMPLE`/`FIGSIZE_WIDE` × `FIGSCALE`
(`figsize_from_scale`, `figsize_grid`), `constrained_layout=True` (sin `tight_layout`), texto EN/FR/both con `lang_text`,
paleta Okabe-Ito (`COLOR_STABLE` azul `#0072B2`, `COLOR_UNSTABLE` naranja `#E69F00`, gris), leyenda sin marco, grid apagado,
`dpi=300` al guardar, `_keep_size`. Comprobado a ojo en `tpr_tnr` con idioma `both`.
Desviaciones menores: (1) no hay `hatch` como codificación redundante estable/inestable en las figuras que usan solo color
(barras/puntos; la skill lo pide para rellenos y `plot_style` ya tiene `HATCH_UNSTABLE`); (2) `FIGSCALE = 1.5` está duplicado en
`validation_figures.py` y `sld_model.py`, la skill lo define como `FIGSCALE_SIMPLE` en `plot_style.py` y ahí no existe;
(3) `legend(fontsize=8)` y `fontsize=14` explícitos en vez del `legend.fontsize` de los rcParams; (4) la skill llama
`_lang_text` a lo que el proyecto llama `lang_text` (no afecta); (5) varias figuras fallan en un `.h5` que solo tiene rampas
(`score_vs_kappa`, `score_dist`: array vacío; `detection_time`: escala log sin datos positivos) — el visor lo muestra como
error; sería mejor que lancen un mensaje claro.

## 8. Exportación configurable de toda figura + eje κ / Ap en el SLD (PLAN_figuras.md)
Autorizado por el manager para `doe_unified_selector.py`, `sld_model.py`, `doe_planner.py`, `doe_val_planner.py`. Supera §7 en
la parte del guardado: ya no hay `Save PNG`/`_save_summary` ni `_export_figure`/`_save_figure_to_reference_dir`; todo pasa por
la ventana de exportación.

| Archivo | Cambio | Compatibilidad |
|---|---|---|
| `DOE_plots/figures_window.py` (mío) | ventana única: `Item(name, render, native, live)`; tamaño own / SIMPLE / WIDE / grid × escala, idioma, dpi, formato, carpeta, nombre; los paneles vivos se exportan como copia (`pickle`) | la llamada antigua `(parent, title, labels, render, style, out_dir, lang, scale)` sigue valiendo |
| `DOE_plots/fig_lang.py` + `figure_texts.yaml` (míos, nuevos) | EN / FR / both para las figuras de texto fijo: se traduce la copia exportada frase a frase; los plotters no se tocan | — |
| `DOE_plots/doe_unified_selector.py` (CRLF) | botón **💾 Export…** en la barra; `Save…`/`Figures…` del panel derecho y `💾 Export figure` de la referencia abren la misma ventana; los exportadores de la referencia ahora devuelven la figura (`_tramos_article_figure`, `_combinado_article_figure(page)`, `_combinado_pages`); casilla **κ axis**; los 4 `FuncFormatter(lambda …)` → `FormatStrFormatter("%.3g")` (mismo texto; la lambda impedía copiar el panel) | archivos `figs_*` en las mismas carpetas de antes |
| `DOE_plots/sld_model.py` (CRLF en el repo, se mantiene) | `plot_sld(..., y_axis=None)`: `None` → global `Y_AXIS = "Ap"`; `"kappa"` **solo añade un eje derecho** κ = Ap / `ap_lim(rpm mediana de los casos)` (lóbulos y casos siguen en Ap; sin casos o en un hueco → contra `ap_crit`). Corregido a petición del usuario: la primera versión dividía los lóbulos por el límite. `_scaled` (módulo) para poder copiar la figura | **llamadores revisados con grep**: el visor (3 entradas) y `doe_planner` (`plot_sld(cases, preset)`); `validation_figures` solo importa `OUTCOMES`. Con `y_axis` por defecto todo dibuja igual que antes |
| `sld_model.py` — **texto visible** | eje Ap rotulado EN « Width of cut $a_p$ [mm] » / FR « Largeur de coupe $a_p$ [mm] » (antes *Depth of cut* / *Profondeur de passe*). Decisión del usuario; cambia el rótulo de **todas** las figuras con SLD (visor, planner, exportaciones) | — |
| `DOE_simulacion/doe_planner.py` (LF) | **solo** botón **💾 Export…** + `export()` que abre la ventana con su figura | ninguna otra lógica |
| `DOE_simulacion/doe_val_planner.py` (CRLF en el repo, se mantiene) | **solo** botón **💾 Export…** + `export()` | ninguna otra lógica |
| `label_grid.py`, `launcher.py` (míos) | Export… en Label grid (cada página en estilo artículo); Compare › Plot pasa por la ventana (ya no guarda solo); Label grid con la paleta de `plot_style` (azul / gris / naranja + hatch) y `constrained_layout` | — |

Verificado: selftests (`figures_window`, `fig_lang`, `sld_model`, `doe_unified_selector`, `validation_figures`, `label_grid`,
`experiment`, `launcher`) y `check_app_dialogs.py`; Export… en los visores sobre archivos reales en solo lectura (resultados,
indicadores, validación, referencia de `ramp_check`; rejilla de `n5189`; Compare con `n12000`; los dos planners), guardando en el
scratchpad, en EN y FR: sin textos sin traducir en esos archivos. Finales de línea comprobados por bytes contra HEAD.

## 9. Casos grises en la validación (lado de la app) — tras `git merge wt-validacion` (c42d796)
Solo lectura de claves nuevas y aditivas de `/metrics/<run>` (schema sigue en `doe_validation_results/4`); todo con `.get`, así
que los `.h5` anteriores (sin esas claves) se ven igual que antes.
- Tarjeta de Validate (`experiment._stage_summary`): bajo cada indicador, si `n_gray`: « gray cases (not scored): N, K alarm |
  if all stable: TNR x, bal.acc y | if all unstable: TPR x, bal.acc y » (mismo formato que `validate_indicators.print_summary`;
  son cotas, no veredicto).
- Compare (`METRIC_COLUMNS`): `n_gray`, `gray_alarm_rate`, `gray_as_stable_TNR`, `gray_as_unstable_TPR` (celdas vacías si faltan).
- `check_app_dialogs.py`: columnas grises con A que las tiene y B que no, y la línea de la tarjeta.

## 10. Modos de casos grises (`--gray ignore|stable|unstable`) — tras `git merge wt-validacion` (970764d)
| Qué | Cambio | Compatibilidad |
|---|---|---|
| Argumentos a `validate_indicators.py` (`experiment._all_stages`) | **nuevo**: `--gray <modo>` (siempre; `ignore` = el comportamiento de antes). Además `--out` ahora trae el sufijo del modo | un YAML sin `gray` = `ignore` |
| Sección `validate` del YAML | clave nueva `gray: stable\|unstable` (ausente = `ignore`; el formulario la borra al volver a `ignore`). Un valor desconocido es un error de `check` y se trata como `ignore` | — |
| Archivo de resultados | `doe_validation_results[_gray-<modo>].h5` (`experiment.validation_path`); un `validate.out` explícito se respeta tal cual | mismo nombre de siempre en `ignore` |
| Huella de Validate | el modo entra **solo si no es `ignore`**: las validaciones ya hechas siguen al día; cambiar de modo marca Validate como desactualizada | — |
| Consumidores del archivo | `validation_metrics`, tarjeta, Compare, botón Viewer y Plot usan `validation_path` (el modo del experimento), no el nombre fijo | — |
| Carpeta de figuras | `validation_figures.figs_dir(h5)` → `figs_validation[_gray-<modo>]`: el Export del visor y Compare › Plot guardan ahí | — |
| `experiment.gray_suffix` | copia de la regla de `validate_indicators.gray_suffix` (no se importa el script: carga numpy/scipy); `check_app_dialogs` y la prueba de extremo a extremo comprueban que coinciden | — |
| Visor (`_case_legend`) | un caso con `$gray$`=1 muestra "(gray)" en su leyenda aunque su `truth` sea stable/unstable | ninguna otra lógica |
UI: `ValidateForm` gana el campo **gray cases** (ignore / stable (pessimistic) / unstable (optimistic)) con una nota; la tarjeta de
Validate empieza con "gray cases: <modo>" y, con `n_gray`, la línea de conteo dice "not scored" o "scored as <modo>". Un modo por
corrida: para tener los tres hay que correr Validate tres veces (cada una con su archivo y su carpeta de figuras).

### 8b. Corrección del eje κ y leyenda de I(t) (a petición del usuario)
- SLD: ver la fila de `sld_model.py` arriba. `envelope`/`kappa_of` ya no existen.
- Visor, gráfica de `I(t)`: la leyenda explica las líneas verticales: trazo discontinuo + punto = `t_d` (primera detección del
  indicador, color de su curva); línea de puntos = `t_onset` (la verdad de una rampa pasa a inestable; solo rampas, y con varios
  casos; con un solo caso esa línea ya lleva su propia etiqueta). Con más de 10 curvas, la leyenda muestra solo esas dos líneas.

### 8c. Gráfica de I(t) del visor: t_onset de casos constantes, retrasos, límites de los indicadores y barra de color
Solo lectura de datos que ya guardan los `.h5`; ningún script de etapa cambia.
- `t_onset` (`_case_onset`): se dibuja también en casos **constantes** de un archivo de validación (attr `$t_onset$` = primera
  muestra sobre el límite de amplitud del etiquetado), no solo en rampas; vale también para las pestañas de señales.
- Retraso Δ = `delay_onset_s` (attr de la corrida en el archivo de validación) en la leyenda de cada curva; con proxy que explica
  Δ = t_d − t_onset (negativo = detectó antes de la amplitud).
- Límites de decisión (`_indicator_limits`, línea horizontal trazo-punto): SST `meta_lim_sup`/`meta_lim_inf`; RMS-CV
  `meta_cv_threshold_used`; MaxEnt `ln((1−β)/α)` y `ln(β/(1−α))` con `pp_alpha`/`pp_beta` (la fórmula es la de
  `MaxEnt_SPRT.lib.sprt`). **Comprobado con datos reales**: en `ramp_check`, `I(t)` está bajo el límite justo antes de `t_d` y sobre él
  en `t_d` (SST 11.23, MaxEnt 6.61). Green SÍ calcula su umbral (mu + z·sigma de log10(area), `upper_log`), pero lo devuelve dentro de `meta["raw_result"]` y `doe_indicators.py` descarta esa clave al escribir el `.h5` (línea `k not in ("raw_result", "signal")` de `run_indicator`): no se dibuja hasta que se guarde. El visor ya lo dibuja si existe `meta_upper_log` (con `meta_I_t_meaning = areas_Ak`, límite = 10**upper_log). El archivo de validación solo trae unos pocos attrs por
  corrida; los `meta_*` se leen (solo attrs) del archivo de indicadores vecino (`indicator_results_file` del atributo raíz).
  Petición abierta a quien mantenga `doe_indicators.py`/Green: guardar el umbral de Green y, en el archivo de validación, copiar los
  umbrales para no depender del archivo vecino.
- Barra de color de κ en la pestaña `I(t)` (como Signals/Forces/Deflex) cuando las curvas van coloreadas por caso (un indicador), con
  marca por caso; el título de arranque "Select cases and press Plot" se sustituye por "κ — N case(s)" como en las señales; se quitó
  `tight_layout()` (la figura ya usa `constrained_layout`).
