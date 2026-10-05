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
