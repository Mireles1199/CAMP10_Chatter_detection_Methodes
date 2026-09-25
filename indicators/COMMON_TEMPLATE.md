# Plantilla común de indicadores CAMP10

## Alcance

Aplica a `maxent_sprt`, `rms_cv`, `ssq_chatter` (SST) y `green_integral`. `rale` y `emd_hht` están obsoletos y no se cubren.

Esto es una **convención para copiar y adaptar** en los archivos propios de cada paquete — **no hay módulo base compartido**. Está prohibido importar tipos/funciones de un paquete de indicador desde otro (anti-patrón existente hoy: `rms_cv/src/rms_cv/lib/runner.py:35` hace `from MaxEnt_SPRT.logging_setup import _section` sin declarar la dependencia en su `pyproject.toml` — corregir en la sesión de RMS).

`maxent_sprt` es la implementación de referencia de esta plantilla.

## 1. `SignalData` (entrada)

```python
@dataclass
class SignalData:
    t_analysis: np.ndarray
    signal_analysis: np.ndarray
    path: str
    fs: float
    meta: Dict[str, Any] = field(default_factory=dict)
```
Referencia: `maxent_sprt/src/MaxEnt_SPRT/utils/types.py`. Ya es uniforme en los 4 paquetes.

## 2. `IndicatorResult` (salida)

```python
@dataclass
class IndicatorResult:
    name: str
    t: np.ndarray
    I_t: np.ndarray
    t_d: np.ndarray = field(default_factory=lambda: np.array([]))
    t_d_no_FAR: np.ndarray = field(default_factory=lambda: np.array([]))
    meta: Dict[str, Any] = field(default_factory=dict)
```
`t_d`/`t_d_no_FAR` son **siempre `np.ndarray`** de timestamps de detección en segundos. Array vacío = sin detección, **nunca `None`** ni un escalar.

**Patrón de bug recurrente confirmado en los 4 indicadores** (cada uno lo tenía en su propia variante): código que loguea o indexa `t_d_no_FAR[0]` asumiendo que no está vacío, sin guardar contra el caso "hubo detecciones en `t_d` pero ninguna pasó el filtro de `t_d_no_FAR`" (`IndexError` en maxent_sprt/rms_cv/ssq_chatter) o directamente nunca lo define en alguna rama (`UnboundLocalError` en green_integral). Guardar siempre con `if result.t_d_no_FAR.size > 0: ... else: ...` antes de indexar.

## 3. Contrato de `INDICATOR_CONFIG`

Consumido por `run_<indicador>(signal: SignalData, INDICATOR_CONFIG: dict) -> IndicatorResult`.

Claves de primer nivel: `id` (opcional, label de logging), `func` (`"Default"` | otro valor propio del indicador, ver punto 4), `param_mode`, `params` (modo `native`) o `params_physical` (modos físicos).

### Tabla de modos

| `param_mode` | obligatorias en `params_physical` | opcionales |
|---|---|---|
| `native` | parámetros nativos propios del indicador (van en `params`, no en `params_physical`) | — |
| `by_revolution` | `T_rev`, `N_rev_window`, `step_rev` | `segmentation` |
| `by_modal` | `T_modal`, `N_modal_window`, `step_modal` | `T_rev` (informativo, no afecta la ventana), `segmentation` |

`param_mode` es **obligatorio** en los 4 indicadores (Green lo adopta en su propia sesión — ver checklist §10).

### Reglas comunes de validación

- `T_rev > 0`, `T_modal > 0`.
- `N_*_window >= 1`.
- `0 < step_* <= N_*_window`.
- `f_cycle` se deriva **según `param_mode`**, nunca según qué claves estén presentes:
  - `by_revolution`: `f_cycle = 1 / T_rev`.
  - `by_modal`: `f_cycle = 1 / T_modal`.
- `segmentation` (`"opr"` por defecto | `"raw"`):
  - `"opr"`: `N_*_window` y `step_*` deben ser **enteros exactos** (p. ej. `2.0` es válido, `2.5` no) → `ValueError` si no lo son. Nunca truncar en silencio.
  - `"raw"`: se aceptan valores decimales; se convierten a cantidad de muestras con `ceil`.
  - **Excepción documentada (Green Integral)**: `segmentation` no aplica a indicadores cuyas ventanas se dimensionan en segundos continuos sin decimación OPR (ventana = tiempo real, no cantidad de muestras). Green Integral es el primer caso: no adopta esta clave, y en su lugar `N_*_window` exige enteros exactos siempre (nunca truncar) mientras que `step_*` acepta fraccionarios siempre (no hay modo "raw" que lo restrinja). Si un futuro indicador tampoco decima a OPR, documentarlo igual en su propia guía en vez de forzar `segmentation`.
- Los parámetros propios de cada indicador (p. ej. `alpha`/`beta`/`reset_on_H0` en MaxEnt, `cv_threshold` en RMS, `Ai_length_mode` en SST) pasan sin tocar por un `frozenset _<IND>_PASS_THROUGH_PARAMS` definido en cada paquete — nunca forman parte de este contrato común.

### `func` como punto de extensión

`"Default"` es la base. Un indicador puede definir valores adicionales (p. ej. `"FixedWindow"` en Green) solo cuando ofrece variantes algorítmicas reales bajo la misma `SignalData`/`IndicatorResult`/`param_mode`. Se documenta en la guía propia de ese indicador, no es obligatorio para todos.

Las dataclasses algorítmicas internas (`MaxEntSPRTConfig`/`SPRTConfig`, `CVOnlineConfig`, `PipelineConfig`, `GreenIntegralConfig`) quedan **internas**, construidas dentro del pipeline — nunca parte del dict público.

## 4. Esqueleto canónico de `run_<indicador>`

```python
def run_<ind>(signal: SignalData, INDICATOR_CONFIG: dict) -> IndicatorResult:
    param_mode: str = INDICATOR_CONFIG.get("param_mode", "native")
    func = INDICATOR_CONFIG["func"]
    if func == "Default":
        func = _<ind>_pipeline

    if param_mode == "native":
        params: Dict[str, Any] = INDICATOR_CONFIG.get("params", {})
        params_physical: Dict[str, Any] = {}
        unit_name, T_unit, N_cycles, step_cycles = "native", float("nan"), None, None
    else:
        params_physical = INDICATOR_CONFIG["params_physical"]
        params, trace = _resolve_physical_params_<ind>(param_mode, params_physical, signal.fs)
        unit_name  = trace["unit_name"]
        T_unit     = trace["T_unit"]
        N_cycles   = trace["N_win"]
        step_cycles = trace["step"]

    result: IndicatorResult = func(signal, **params)

    result.meta["param_mode"]   = param_mode
    result.meta["unit_name"]    = unit_name
    result.meta["f_cycle"]      = (1.0 / T_unit) if T_unit and T_unit == T_unit else <nativo propio>
    result.meta["N_cycles"]     = N_cycles
    result.meta["step_cycles"]  = step_cycles
    result.meta["Total_window"] = <ventanas totales, propio de cada indicador>
    return result
```
Nótese que `params_physical` queda **siempre definido** (dict vacío en modo `native`), y el diagnóstico se ramifica explícitamente por `param_mode` en vez de asumir que `params_physical` tiene ciertas claves. Esto corrige un `NameError` idéntico presente hoy en los tres runners existentes (ver checklist §10).

## 5. Claves estándar de `result.meta`

`param_mode`, `f_cycle` [Hz], `unit_name` (`"rev"`|`"modal"`|`"native"`), `T_unit` [s], `N_cycles`, `step_cycles`, `Total_window` (en ciclos/ventanas). En modos físicos, también `physical_params_input` y `native_params_resolved` (trazabilidad). Cada indicador puede agregar claves propias.

## 6. `HDF5Reader` y `logging_setup.py`

Referencia: `maxent_sprt/src/MaxEnt_SPRT/utils/hdf5_utils.py` y `maxent_sprt/src/MaxEnt_SPRT/logging_setup.py` (`INFO_PLUS_LEVEL=15`, `LOGGING_LEVELS`, `configure_logging()`, `_section()`). Se copian localmente en cada paquete — nunca se importan entre paquetes.

## 7. Convención de nombres

`run_<indicador>(signal, INDICATOR_CONFIG) -> IndicatorResult` y `plots_<indicador>(...)` como únicos dos puntos de entrada públicos, reexportados desde el `__init__.py` de cada paquete.

## 8. Import local vs. instalación editable (evitar código fantasma)

**Hallazgo real en la sesión de MaxEnt**: cada indicador está instalado en modo editable (`pip install -e`) apuntando a una ruta **fija y absoluta** (en este repo, el checkout `CAMP10_Chatter_detection_Methodes`, no los worktrees `wt-*`). Un `.pth` en el entorno registra esa ruta al momento de instalar y no cambia solo — no importa desde qué worktree/rama corras un script, `import MaxEnt_SPRT` (o `rms_cv`, `ssq_chatter`, `green_integral`) siempre resuelve ahí, salvo que el propio script le diga lo contrario.

Consecuencia: mientras un worktree tiene cambios sin mergear (renombres de claves, fixes de bugs, etc.), **cualquier script que no fuerce el import local corre silenciosamente contra el código viejo** — o peor, contra un config nuevo que el resolver viejo no entiende (`ValueError: ... requires 'N_rev_per_seg' ...` fue exactamente este caso).

**Regla obligatoria**: todo script bajo `examples/` de cualquier indicador debe insertar su propio `src/` al principio de `sys.path`, **antes** de importar el paquete:

```python
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from <Paquete> import ...
```

Es relativo a `__file__`, así que sigue resolviendo bien después de mergear (converge con la instalación editable en vez de competir con ella — ver checklist §10 para el estado real de cada paquete).

Esto no aplica solo a scripts de un solo indicador: cualquier script que combine varios (`pareto_stage1.py`, `Noise_SNR.py`, `effective_window/`, `Optimizacion/*`) necesita insertar el `src/` de **cada** indicador que use, antes del primer `import`. Los que ya lo hacen bien en este repo (patrón a copiar): `Optimizacion/study_phase3/study_phase3.py` y `Optimizacion/study_phase1/study_phase1.py` (calculan la raíz del repo vía `__file__`, no hardcodean el nombre del checkout).

## 9. `pyproject.toml`: dependencias completas

Cada paquete debe declarar en `dependencies` **todo** paquete de terceros que se importa en algún archivo bajo `src/<paquete>/` — no solo los que se usaron para probar en un entorno que ya los traía instalados por otra vía. Verificar con:

```bash
grep -rhoE "^(import|from) [a-zA-Z0-9_]+" src/<paquete>/**/*.py | sort -u
```
y comparar contra `dependencies` en `pyproject.toml`. Un `pip install .` en un entorno limpio (p. ej. el de un director/colaborador que nunca tocó este repo) debe funcionar sin `ModuleNotFoundError`.

## 10. Checklist de migración por indicador

Trazabilidad para que cada sesión de indicador sepa qué corregir en sus propios archivos. Esta sesión (MaxEnt) no los edita.

### RMS-CV (`rms_cv`) — ✅ hecho (sesión RMS-CV, en `wt-RMS-CV`, sin commitear/mergear)
Reportado: import cruzado reemplazado por su propio `logging_setup._section` local; `run_rms_cv` reescrito con el esqueleto de §4 (con fallback seguro en `native`, que además tenía un `KeyError` más abajo no cubierto por el checklist original); `N_rev_window`/`step_rev`/`T_rev` ya cumplían; `t_d`/`t_d_no_FAR` → `np.ndarray`; `ScenarioMetadata` borrado (cero usos confirmados); agregó a `pyproject.toml` `h5py` + **`scipy`** + **`statsmodels`** (no detectados en el grep original de §9 — usados en `viz/rms_cv_plots.py` y `lib/cv_monitor.py`); encontró y arregló el mismo `IndexError` en `t_d_no_FAR[0]` que apareció en maxent_sprt y ssq_chatter (ver nota al final de esta sección). Verificado con corridas reales (native/by_revolution/by_modal).

Hallazgos originales (referencia, ya resueltos):
- `NameError` en modo `native`: `src/rms_cv/lib/runner.py:314` usa `params_physical` sin definirlo en esa rama. Aplicar el esqueleto de §4.
- `f_cycle` (`:314-317`) elegido según qué clave está presente (`T_rev` vs `T_modal`) en vez de según `param_mode` — mismo bug que tenía MaxEnt.
- Renombrar claves de ventana: `N_rev_window`/`N_modal_window` ya coinciden con la plantilla (RMS ya las usa así); falta hacer `step_rev`/`step_modal` consistentemente obligatorios (ya lo son) y quitar la obligatoriedad de `T_rev` en `by_modal` si aplica el mismo criterio que MaxEnt.
- Import cruzado no declarado: `src/rms_cv/lib/runner.py:35` (`from MaxEnt_SPRT.logging_setup import _section`) — copiar `logging_setup.py` localmente en vez de importar de `MaxEnt_SPRT`.
- `t_d: Optional[float]` en `utils/types.py:101` (aprox.) — pasar a `np.ndarray` con `default_factory`, como en §2.
- `ScenarioMetadata` en `utils/types.py` parece no usarse por ningún runner/ejemplo — candidato a borrar si se confirma.
- **`examples/RMS_CV_Chatter_Detection_NEW.py` y `RMS_CV_Chatter_Detection_old.py` NO insertan `src/` local al `sys.path`** (verificado) — van a correr contra el paquete instalado (ruta fija, ver §8) en vez del código editado, hasta que se les agregue el bloque de §8.
- **`h5py` no está declarado en `pyproject.toml`** (usado en `src/rms_cv/utils/hdf5_utils.py`) — agregar a `dependencies` (ver §9).

### SST-SVD (`ssq_chatter`) — ✅ hecho (sesión SST, en `wt-SST`, sin commitear/mergear)
Reportado: `NameError` y `f_cycle`/`unit_name`/`N_cycles`/`step_cycles` corregidos siguiendo §4; además arregló un bug propio no listado abajo: el logging leía `N_cycles`/`step_cycles`/`Total_window` incondicionalmente aunque solo se poblaban en modo físico (en `native` quedan `None`, logging los saltea con guard explícito — evaluó usar `nan` pero lo descartó porque `N_cycles` se loguea con `%d`, que también rompe con `nan` vía `OverflowError`; `None` + guard es la opción correcta). `t_d`/`t_d_no_FAR` → `np.ndarray`. Agregó a `pyproject.toml` `h5py` + **`statsmodels`** (usado en `detection_strategies.py` vía `lilliefors`, no detectado en el grep original de §9) y descomentó `matplotlib`. Compilado sin errores; no corrido end-to-end (sin datos de prueba disponibles en su sesión).

Hallazgos originales (referencia, ya resueltos):
- Mismo `NameError` en modo `native`: `src/ssq_chatter/lib/runner.py:275`.
- Mismo bug de `f_cycle` elegido por presencia de clave en vez de por `param_mode` (`:275-278`).
- Tipo de `t_d` en `utils/types.py` — pasar a `np.ndarray`.
- Ya usa `N_rev_window`/`N_modal_window`/`step_rev`/`step_modal` — coincide con la plantilla.
- **`examples/SSQ_STFT_Chatter_Detection_NEW.py` y `SSQ_STFT_Chatter_Detection_old.py` NO insertan `src/` local al `sys.path`** (verificado) — mismo riesgo que RMS-CV, agregar el bloque de §8.
- **`h5py` no está declarado en `pyproject.toml`** (usado en `src/ssq_chatter/utils/hdf5_utils.py`), y **`matplotlib` está comentado** en `dependencies` aunque `viz/plotting.py` y `viz/sst_svd_plots.py` lo importan directo — descomentar y agregar `h5py` (ver §9).

### Green Integral (`green_integral`) — ✅ hecho (sesión Green-Area, en `wt-Green-Area`, sin commitear/mergear)
Reportado: `_resolve_physical_params_green` reescrito con el esqueleto de §4; adoptó `param_mode`/`params_physical` con `T_rev`/`N_rev_window`/`step_rev` o `T_modal`/`N_modal_window`/`step_modal` (`T_rev` opcional/informativo en `by_modal`, igual que MaxEnt); eliminó `f_modal`/`f_cycle`/`N_cycles_per_seg`/`step_cycles` del contrato público (el `f_modal` viejo no era un filtro bandpass real pese a su docstring, solo definía la duración del ciclo de ventaneo). **Decisión documentada**: Green NO adopta `segmentation` (`opr`/`raw`) de §3 — sus ventanas se dimensionan en segundos continuos, sin decimación OPR a muestras, así que esa dualidad no aplica; ver nota agregada a §3. `N_*_window` sí exige enteros exactos (antes truncaba en silencio con `int(...)`), `step_*` acepta fraccionarios (paso continuo, no hay modo raw que lo restrinja). Encontró y arregló un `UnboundLocalError` real: la rama `func=="Default"` nunca definía `t_d_no_FAR`. `t_d`/`t_d_no_FAR` → `np.ndarray`. Decisión: mantiene `StdSignalData` (no es redundante, la `SignalData` nativa separa displacement/velocity). Actualizó `Green_Integral_Detection_NEW.py` y `Green_Integral_FixedWindow_Tutorial.py` al nuevo `param_mode`; no tocó los demos legacy. Verificado con un self-check ad-hoc de 7 asserts (no commiteado, solo para validar el refactor), incluye corridas end-to-end reales para `Default` y `FixedWindow`.

Hallazgos originales (referencia, ya resueltos):
- No tiene `param_mode`. Adoptar `param_mode` (`native`/`by_revolution`/`by_modal`) + `params_physical` con `T_rev`/`T_modal`/`N_rev_window`/`N_modal_window`/`step_rev`/`step_modal`, en vez del esquema actual `f_cycle`/`N_cycles_per_seg`/`step_cycles` en `lib/runner_std.py`.
- El `f_cycle` actual (línea ~119 de `lib/runner_std.py`) se calcula a partir de un valor recibido directamente, no derivado de `T_rev`/`T_modal` según el modo — alinear con la regla de §3.
- `StdSignalData` (en `utils/types.py`) existe solo para imitar `SignalData` de este contrato — una vez que `run_green_std` adopte el contrato estándar directamente, evaluar si `StdSignalData` sigue siendo necesaria o se puede unificar con la `SignalData` nativa de Green.
- Tipo de `t_d`/`t_d_no_FAR` — confirmar que sean siempre `np.ndarray`.
- `"func": "Default" | "FixedWindow"` ya es un punto de extensión legítimo (documentado en `runner_std.py`), se mantiene tal cual.
- `examples/`: la mayoría de los scripts principales (`Green_Integral_Detection_NEW.py`, `Green_Integral_FixedWindow_Tutorial.py`, `phase_area_indicator.py`, `test_synthetic_signal.py`, `augmented_trajectory_exploration.py`, `DDE_signal_sources.py`) **sí** insertan `src/` local — bien. Los demos (`Demo_Spirale*.py`, `Demo_Trayectoria_8.py`, `Green_Area_*.py`) no lo hacen; revisar si valen la pena o son candidatos a legacy/borrado antes de arreglarlos.
- `pyproject.toml` ya declara `h5py`, `scipy`, `matplotlib`, `scikit-learn` — sin hallazgos de dependencias faltantes por ahora; re-verificar con el comando de §9 si se agregan imports nuevos.

**Fase 3 (dataset de referencia externo) — ✅ hecho.** Nueva clave opcional de primer nivel en el config de `run_green_std`: `"reference_signal": Optional[StdSignalData] = None` (se eligió `StdSignalData`, no la `SignalData` nativa, porque ese es el tipo que ya cruza la frontera pública de `run_green_std`; se convierte internamente con el mismo helper que la señal analizada, `_to_internal_signal`). Cuando está presente, `_green_integral_pipeline`/`_lyapunov_pipeline` corren el mismo pipeline de ventaneo sobre la señal de referencia (llamada recursiva con `reference_signal=None, use_area_threshold=False` para evitar recursión infinita) y usan TODAS sus ventanas como población de entrenamiento del umbral mu±zσ, en vez de `training_intervals`/`stable_time`/`frac_stable`. Si ambos se pasan a la vez, `reference_signal` gana y se loguea un `warning`. Sin `reference_signal`, comportamiento 100% idéntico al previo (verificado en el self-check). Se agregó `result.meta["training_source"]`: `"external_reference"` | `"internal"` (también en `global_data` de ambos resultados internos). Self-check con asserts: `examples/test_reference_signal.py`. De paso se encontró y arregló un bug preexistente no relacionado: `_lyapunov_pipeline` crasheaba con `TypeError` al comparar `t_d_detected >= config.t_theorical` cuando había detección pero `t_theorical` era `None` (nunca antes ejercitado porque los usos reales siempre pasan `t_theorical`).

**Corrección post-Fase 3 (histogramas de entrenamiento).** El usuario detectó que las figuras de "distribución de áreas estables" (C4/D1/D1b/D4b en `plots_lyapunov`) seguían re-derivando la población de entrenamiento cortando `training_intervals` de la señal analizada, ignorando por completo `reference_signal` — con reference externa mostraban la curva/histograma de la señal equivocada (o ninguna, si no se pasaba también `training_intervals`). Arreglado en los 3 archivos:
- `runner.py`/`runner_lyapunov.py`: ambos pipelines ahora guardan en `global_data["training_areas"]`/`["training_t_wins"]` la población EXACTA (áreas lineales + su propio eje temporal) que entrenó el umbral, sea cual sea la fuente.
- `viz/plots.py`: nueva función compartida `plot_training_distribution(global_data, name, log_transform)` — histograma + PDF gaussiana ajustada con el mu/sigma REAL del umbral (nunca recalculado) + una curva nueva ("Training Curve — normal-law input") con los mismos valores en el orden en que entraron al ajuste (eje temporal si viene de `training_t_wins`, si no índice de muestra), con líneas μ/μ±zσ — para verificar visualmente el supuesto de normalidad sea cual sea la fuente. Título siempre indica `training_source` ("internal training_intervals" | "external reference").
- `viz/green_integral_plots.py`: `plots_green_integral` (Default) ahora llama a esta función (antes no tenía ningún histograma). `plots_lyapunov` reemplazó su C4/D1 buggy por la misma función; D1b/D4b (desglose por label) se mantienen pero solo corren cuando `training_source == "internal"` y con μ/σ tomados de `area_mu_3sigma` en vez de recalculados. También se corrigió el gate de las líneas μ±zσ en C2/C2-b (`explicit_training_provided` → `thr`), que antes las ocultaba si no había `training_intervals` explícito aunque el umbral sí se hubiera entrenado vía `reference_signal`.
- Self-check ampliado (`test_reference_signal.py`, §5) verifica con backend `Agg` que ambas figuras nuevas aparecen para las 4 combinaciones (Default/Lyapunov × internal/external) y que `global_data["training_areas"]`/`["training_t_wins"]` quedan poblados.

**Corrección posterior (pedido del usuario vía Manager):** el título de `plot_training_distribution()` mostraba textualmente `training_source` ("internal training_intervals" | "external reference") — el usuario pidió sacar esa mención del título (no quiere la fuente ahí), pero mantener la verificación real de que el indicador efectivamente usa la población de referencia. Se sacó el texto de ambos títulos y en su lugar se agregó un `assert` dentro de la propia función: recalcula μ/σ desde `training_areas` y exige que reproduzcan exactamente los valores de `area_mu_3sigma` — si algún cambio futuro desincronizara ambos (el bug original que esto reemplaza), el `assert` lo detecta en el momento de graficar, no solo en un test aparte.

**Pasada de confiabilidad de gráficas (pedido del usuario vía Manager, aplica a los 4 indicadores).** Revisados todos los paneles de `viz/plots.py` y `viz/green_integral_plots.py` contra 5 puntos:
1. Títulos vagos ("Local Data"/"Local Indicator" en `plot_windows_local`/`plot_indicator_local`) → renombrados a algo concreto ("Mean Area per Cycle"/"Delta_n per Window").
2. Bug de zoom real encontrado: el eje secundario "Index of data_window" (twiny) solo resincronizaba su `xlim` cuando había >1 punto visible — al hacer zoom muy profundo quedaba con ticks obsoletos, desalineado del eje primario. Corregido moviendo `ax2.set_xlim(...)` fuera del guard, en ambas funciones.
3. `plots_signal_diagnostics` dibujaba marcadores de frecuencia (150/200/50 Hz) y un beat de 20 ms **hardcodeados**, sin relación con el caso real analizado. Ahora son parámetros opcionales (`freq_markers: Dict[str,float]`, `t_beat_ms: float`), `None` por defecto → no se dibuja nada fabricado. También se corrigió el gate de las líneas μ±zσ en C2/C2-b de `plots_lyapunov` (`explicit_training_provided` → `bool(thr)`), que las ocultaba si no había `training_intervals` explícito aunque el umbral sí viniera de `reference_signal`.
4. Población de entrenamiento real: cubierto por el punto anterior de Fase 3 (`training_areas`/`training_t_wins`); no se encontraron otros paneles con el mismo problema.
5. **Regla nueva más importante — a lo sumo 2 líneas verticales de evento en cualquier panel de detección a lo largo del tiempo** (`t_gt`/`t_theorical` + primera detección): revisado el mecanismo existente (`auto_vlines` en `plots_lyapunov`) y ya cumplía — solo usa `t_d[0]`, nunca itera sobre todas las detecciones. Verificado empíricamente (no solo lectura de código) contra un caso sintético con cientos de detecciones y contra el caso real "cono" con 1324 detecciones: 0 violaciones en C1/C2/C2-b/C3/Ĝ/Ĝs.
- Self-check nuevo: `examples/test_plot_reliability.py` (zoom-sync + no-fabricación de marcadores + conteo de vlines contra un caso sintético con muchas detecciones).

**Corrección adicional (pedido directo del usuario): C1 a un solo color + paleta homogénea con MaxEnt.** Confirmado con el usuario (vía AskUserQuestion, ambigüedad real: "Figura 1" podía referirse a cualquiera de ~7 figuras) que "Figura 1" = C1 (panel de señal, `fig_01.png` de la corrida real) y que "un solo color para todos" significa sacar el split azul=estable/naranja=chatter de la traza — el corte ya lo marcan las vlines de `t_gt`/primera detección, no hace falta duplicarlo en el color de la línea. C1 pasó de ~65 líneas de lógica de máscaras por intervalo a 2 líneas (`ax.plot(t_arr, q_arr/v_arr, color=color_azul, ...)`). De paso alineé los colores sueltos que quedaban en `plots_signal_diagnostics` ("steelblue"/"darkorange"/"forestgreen"/"red"/"crimson") a la paleta compartida `color_azul/orange/verde/red` — que YA es idéntica a la de MaxEnt (mismos valores `colorsys.hls_to_rgb`, mismo `fig_size()`, mismo `configurar_estilo_global()` — la homogeneidad de paleta/tipografía entre Green y MaxEnt ya estaba dada de antes, no hizo falta tocarla). Chequeo agregado a `test_plot_reliability.py` (§5).
- Confirmado contra caso real (`cono` + `reference_combined.h5` actual, 8,158,276 muestras): `t_d=8.390 s` (`t_gt=5.36577 s`), 1324 detecciones totales, 0 paneles con más de 2 líneas de evento, título/parámetros correctos. Figuras guardadas a PNG para inspección visual durante la auditoría (no commiteadas).

**Corrección adicional (pedido del usuario): estabilidad del área graficable al hacer zoom.** Varias figuras de `plots_lyapunov` (C1, C3, D1b, D4b, Ĝ, Ĝs) usaban un motor de layout persistente (`layout='tight'` en `plt.subplots(...)`, o `constrained_layout=True` en C1) — este tipo de motor se re-ejecuta en CADA render/`savefig`, no solo al crear la figura. Como las anotaciones de texto pegadas a las vlines/hlines (`_add_vline_label`/`_add_hline_label`) se ocultan/muestran según si su valor cae dentro del rango visible actual, el motor de layout reaccionaba a eso y encogía/agrandaba `ax.get_position()` dependiendo de qué tanto texto hubiera visible en ese momento — confirmado empíricamente: mismo zoom (mismo ancho), corrido a una posición donde la anotación cae dentro vs. fuera del rango → `ax.get_position()` cambiaba (ej. de `(0.125, 0.110, 0.775, 0.770)` a `(0.130, 0.125, 0.857, 0.804)`). Arreglado sacando el motor persistente y dejando un solo `fig.tight_layout()` (llamada única, no un motor que se re-adapta) — verificado que con eso `ax.get_position()` queda idéntico sea cual sea el estado de zoom/anotaciones visibles, en los 7 paneles de `plots_lyapunov`. Chequeo agregado a `test_plot_reliability.py` (§4).

### Consumidores externos (repo Repo-DOE)
- MaxEnt renombra `N_rev_per_seg`→`N_rev_window`, `N_modal_per_seg`→`N_modal_window`; hace el paso obligatorio; vuelve `t_d` siempre `np.ndarray`. Avisado por SendMessage; confirmar antes de mergear.
