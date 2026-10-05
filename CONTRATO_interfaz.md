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
