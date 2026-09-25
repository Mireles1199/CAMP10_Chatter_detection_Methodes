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

`param_mode` es **obligatorio** en los 4 indicadores (Green lo adopta en su propia sesión — ver checklist §12).

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
Nótese que `params_physical` queda **siempre definido** (dict vacío en modo `native`), y el diagnóstico se ramifica explícitamente por `param_mode` en vez de asumir que `params_physical` tiene ciertas claves. Esto corrige un `NameError` idéntico presente hoy en los tres runners existentes (ver checklist §12).

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

Es relativo a `__file__`, así que sigue resolviendo bien después de mergear (converge con la instalación editable en vez de competir con ella — ver checklist §12 para el estado real de cada paquete).

Esto no aplica solo a scripts de un solo indicador: cualquier script que combine varios (`pareto_stage1.py`, `Noise_SNR.py`, `effective_window/`, `Optimizacion/*`) necesita insertar el `src/` de **cada** indicador que use, antes del primer `import`. Los que ya lo hacen bien en este repo (patrón a copiar): `Optimizacion/study_phase3/study_phase3.py` y `Optimizacion/study_phase1/study_phase1.py` (calculan la raíz del repo vía `__file__`, no hardcodean el nombre del checkout).

## 9. `pyproject.toml`: dependencias completas

Cada paquete debe declarar en `dependencies` **todo** paquete de terceros que se importa en algún archivo bajo `src/<paquete>/` — no solo los que se usaron para probar en un entorno que ya los traía instalados por otra vía. Verificar con:

```bash
grep -rhoE "^(import|from) [a-zA-Z0-9_]+" src/<paquete>/**/*.py | sort -u
```
y comparar contra `dependencies` en `pyproject.toml`. Un `pip install .` en un entorno limpio (p. ej. el de un director/colaborador que nunca tocó este repo) debe funcionar sin `ModuleNotFoundError`.

## 10. `reference_signal` como punto de extensión (calibración contra referencia externa)

Clave opcional de primer nivel en `INDICATOR_CONFIG` (hermana de `func`/`param_mode`/`params`/`params_physical`, no va dentro de ninguna de esas):

```python
INDICATOR_CONFIG = {
    "func": "Default",
    "param_mode": "native",  # o "by_revolution" / "by_modal"
    "params": {...},          # o "params_physical": {...}
    "reference_signal": [SignalData(...), SignalData(...), ...],  # opcional -- List[SignalData], un SignalData suelto también vale
}
```

- **Tipo**: `SignalData | List[SignalData]`. Cada elemento es un tramo aislado, físicamente continuo (p. ej. un caso DOE), etiquetado por un pipeline externo. Un `SignalData` suelto se trata como lista de un elemento.
- **Fuente recomendada**: `reference_dataset.h5` (salida de `reference_dataset.py build` en Repo-DOE), layout `/<label>/<case>/<canal>__NNN/{t, y}` con attrs `signal_id`/`t0`/`t1`/`fs`/`channel` — **no** `reference_combined.h5` (ese archivo concatena todos los tramos de todos los casos en una sola señal continua, y sigue existiendo solo como entrada del visor `doe_unified_selector.py`).
- **Regla de oro (evita contaminación de costura)**: ventanear cada tramo por separado con el mismo pipeline de ventaneo de siempre, calcular su entropía, y recién ahí juntar los RESULTADOS (entropías + tiempos) de todos los tramos — nunca concatenar la señal cruda entre tramos discontinuos antes de ventanear. Concatenar antes de ventanear deja que una ventana de longitud fija mezcle muestras de dos tramos físicamente no relacionados (p. ej. el final de un caso DOE pegado al inicio de otro), contaminando esa entropía y con ella el ajuste de `p0`/`p1`.
- **Si está presente**: el indicador usa los tramos de `reference_signal` como su región de entrenamiento/referencia completa, en reemplazo de lo que hoy carva internamente de `training_intervals`/`cut_start_time`+`t_stable_total` (legacy cut) de la señal analizada. Son arrays crudos ya etiquetados por un pipeline externo — no hace falta re-correr ningún pipeline de ventaneo propio sobre ellos, solo aplicar el ventaneo tramo por tramo.
- **Si NO está presente (default)**: comportamiento 100% idéntico al actual, cero impacto en nada existente. La firma de `func(signal, **params)` no cambia — `reference_signal` solo se agrega como kwarg cuando está presente, para no romper `func` personalizados que no lo declaren.
- Si además vienen `training_intervals`/legacy cut, `reference_signal` tiene prioridad para la región de referencia/entrenamiento; qué hacer con la contraparte que cada indicador necesite además de esa referencia (p. ej. una contraparte "chatter" para un modelo de dos hipótesis) queda a criterio de cada indicador — documentarlo en su propia sección. Loguear un aviso cuando ambos estén presentes es recomendable, no obligatorio. Nota: `training_intervals` con varias entradas del mismo label tiene el mismo riesgo de costura que `reference_signal` — cada indicador debe ventanear cada intervalo por separado también ahí, no solo en el camino de `reference_signal` (ver referencia de implementación de `maxent_sprt` más abajo).
- Agregar a `result.meta`: `"training_source": "external_reference" | "internal"`, `"reference_n_pieces"` (cantidad de tramos de `reference_signal` efectivamente usados, `0` si no se usó).

**Referencia de implementación**: `rms_cv` fue el primer paquete en implementar este patrón (`_load_reference_pieces`, lee `reference_dataset.h5` directo en vez del combinado, para evitar mezclar tramos de distintos casos). `maxent_sprt` (`lib/runner.py`, `run_maxent_sprt` + `_maxent_sprt_pipeline`, sección "Training signal split") lo implementa también: normaliza `reference_signal`/`training_intervals`/el cut legacy en una lista de "pieces" `(t, x, piece_id)`, ventanea cada piece por separado con `segment_opr`/`segment_signal_raw` (sin tocar esas funciones), calcula su entropía, y solo concatena las entropías + tiempos resultantes antes de ajustar `p0`/`p1` (`lib/offline.py::offline_train_maxent_sprt`, acepta listas de piezas). Ahí `reference_signal` reemplaza `t_stable`/`signal_analysis_stable` (P0/stable). MaxEnt además define su propia extensión (no forma parte del contrato común) para la contraparte chatter/P1: `reference_signal_chatter`, mismo tipo `SignalData | List[SignalData]`, misma clave de primer nivel en `INDICATOR_CONFIG`. Si `reference_signal_chatter` viene, reemplaza `t_chatter`/`signal_analysis_chatter`; si no, esa contraparte sigue derivándose de `training_intervals` (solo las entradas `"chatter"`) o del `cut_end_time` legacy si vienen, si no queda vacía. Cada lado (`reference_signal`/`reference_signal_chatter`) es independiente — se puede dar uno solo o los dos. `result.meta` agrega también `"chatter_source": "external_reference" | "internal"`, `"reference_n_pieces_chatter"`, y `"n_windows_per_piece_free"`/`"n_windows_per_piece_chat"` (ventanas por tramo, útil para diagnosticar tramos demasiado cortos) (todos propios de MaxEnt). Self-check: `maxent_sprt/examples/test_reference_signal.py` (incluye un test sintético de dos tramos con entropías muy distintas que falla si alguna ventana mezcla ambos tramos). Ejemplo end-to-end: `maxent_sprt/examples/MaxEnt_Detection_NEW.py` (flags `USE_EXTERNAL_REFERENCE`/`USE_EXTERNAL_REFERENCE_CHATTER`, lee `reference_dataset.h5` con h5py directo vía `_load_reference_pieces`, grupos `/<label>/<case>/<canal>__NNN`).

## 11. `load_signal` y `SIGNAL_SOURCE` — lectura uniforme de la señal a analizar (layout crudo o DOE)

Punto de extensión común para que los 4 indicadores lean la señal a analizar de la misma forma, sea que venga del layout crudo del simulador (`sens_out.hdf5`/`out.hdf5`) o del layout repackagado de DOE (`doe_results.h5`/`doe_noise_results.h5`, con grupos por caso).

### Función `load_signal`

```python
def load_signal(
    reader: "HDF5Reader",
    signal_name: str,
    case_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Lee (t, y) de un HDF5Reader ya cargado.

    - case_name=None  -> layout crudo del simulador (sens_out.hdf5/out.hdf5):
      "<signal_name>/data" como array (N,2): col 0 = tiempo, col 1 = valor.
    - case_name="<grupo>" -> layout DOE (doe_results.h5/doe_noise_results.h5):
      "<case_name>/<signal_name>/time" + "<case_name>/<signal_name>/values"
      como datasets separados.
    """
    if case_name:
        t = np.asarray(reader.get_element(f"{case_name}/{signal_name}/time"), dtype=float)
        y = np.asarray(reader.get_element(f"{case_name}/{signal_name}/values"), dtype=float)
    else:
        arr = np.asarray(reader.get_element(f"{signal_name}/data"), dtype=float)
        t, y = arr[:, 0], arr[:, 1]
    return t, y
```

Cada indicador implementa esta MISMA función en su propio paquete (`utils/hdf5_utils.py`, junto a su propio `HDF5Reader`) — sin cross-import, por la misma convención de §8.

### Convención `SIGNAL_SOURCE`

Forma estándar de declarar el origen de la señal a analizar en cualquier script de `examples/`, en vez de variables sueltas (`data_dir`/`CASE_NAME`, etc.):

```python
SIGNAL_SOURCE = {
    "hdf5_path": r"...\sens_out.hdf5",   # o doe_results.h5 / doe_noise_results.h5
    "case_name": None,                    # None (layout crudo) | "case_003" / "snr_005.00" (layout DOE)
    "disp_name": "Axial_disp",
    "vel_name":  "Axial_vel",
    "force_name": "force_N",              # opcional -- mantener el nombre de canal que ya usa cada paquete
}
```

Uso:
```python
data    = HDF5Reader(SIGNAL_SOURCE["hdf5_path"])
t, disp = load_signal(data, SIGNAL_SOURCE["disp_name"], SIGNAL_SOURCE["case_name"])
_, vel  = load_signal(data, SIGNAL_SOURCE["vel_name"],  SIGNAL_SOURCE["case_name"])
```

Las claves `disp_name`/`vel_name`/`force_name` son las que ya usa cada paquete (p. ej. distinto nombre de canal en Green Integral si aplica) — no se fuerza un nombre común, solo la forma del dict.

### Convención `CASES` / `ACTIVE_CASE` (forma estándar final)

En cualquier script de `examples/` que arma el `INDICATOR_CONFIG` completo y llama a `run_<indicador>` (no solo lee la señal), `signal_source` e `indicator_config` van **juntos**, anidados dentro de un `CASES` nombrado — misma estructura en los 4 paquetes, el contenido de `indicator_config` sigue siendo propio de cada uno:

```python
CASES: dict[str, dict] = {
    "default": {
        "signal_source": {
            "hdf5_path": ..., "case_name": None,
            "disp_name": "Axial_disp", "vel_name": "Axial_vel", "force_name": "force_N",
        },
        "indicator_config": {
            # claves propias del indicador (lo que hoy arma tu INDICATOR_CONFIG)
            "id": "...", "func": "Default", "param_mode": "...", "params_physical": {...},
        },
    },
    # ... otras entradas nombradas (presets alternativos)
}

ACTIVE_CASE      = "default"          # <- cambiar solo esta línea para elegir señal + config
SIGNAL_SOURCE    = CASES[ACTIVE_CASE]["signal_source"]
INDICATOR_CONFIG = CASES[ACTIVE_CASE]["indicator_config"]   # se pasa directo a run_<indicador>(signal, INDICATOR_CONFIG)
```

`signal_source` tiene la misma forma que la convención `SIGNAL_SOURCE` de arriba, solo que anidada dentro de cada caso en vez de suelta a nivel de módulo — así una sola variable (`ACTIVE_CASE`) alcanza para elegir señal + config juntos. Un script puede seguir exponiendo un flag de CLI para elegir entre varios `CASES` en tiempo de ejecución (conveniencia adicional) — `CASES`/`ACTIVE_CASE` sigue siendo la fuente de verdad, el flag no reemplaza la convención.

**Referencia de implementación**: `maxent_sprt` (`utils/hdf5_utils.py::load_signal`, exportada desde `MaxEnt_SPRT/__init__.py`). `examples/read_hdf_signals.py` (solo lee la señal: usa `SIGNAL_SOURCE` + `load_signal` para leer disp/vel/force, con fallback a ceros si `force_name` no está presente). `examples/MaxEnt_Detection_NEW.py` (arma el `INDICATOR_CONFIG` completo y corre el indicador: usa `CASES`/`ACTIVE_CASE`, con 5 entradas — una por variante de `param_mode`/segmentación — que hoy comparten el mismo `signal_source`; mantiene además `--case NAME` como flag de conveniencia sobre `CASES`). Self-check: `examples/test_load_signal.py` (fixture `.h5` sintético con ambos layouts). Verificado además contra datos reales: mismo resultado exacto leyendo un caso de `2DOF_Cone_DOE` vía `sens_out.hdf5` (layout crudo) y vía `doe_results.h5` (layout DOE, mismo caso), y mismo resultado en `MaxEnt_Detection_NEW.py` antes/después de migrar a `CASES`.

## 12. Checklist de migración por indicador

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

### Consumidores externos (repo Repo-DOE)
- MaxEnt renombra `N_rev_per_seg`→`N_rev_window`, `N_modal_per_seg`→`N_modal_window`; hace el paso obligatorio; vuelve `t_d` siempre `np.ndarray`. Avisado por SendMessage; confirmar antes de mergear.
