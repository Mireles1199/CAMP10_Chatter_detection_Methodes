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
