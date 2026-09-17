# Auditoría de código — `MaxEnt_Detection_NEW.py` (paquete `MaxEnt_SPRT`)

- **Proyecto auditado**: `indicators/maxent_sprt/`
- **Archivo de entrada**: `indicators/maxent_sprt/examples/MaxEnt_Detection_NEW.py`
- **Profundidad**: exhaustiva (línea por línea, todo el flujo real de dependencias)
- **Tipo de auditoría**: solo lectura — no se modificó, formateó ni ejecutó ningún archivo del proyecto.

---

## 1. Resumen de qué hace el código

El script de entrada implementa un pipeline de **detección de chatter (vibración autoexcitada en mecanizado)** basado en el método **MaxEnt-SPRT** (Máxima Entropía + Test Secuencial de Razón de Probabilidad), descrito en Zhao et al. (2019, incluido en `references/`).

Flujo conceptual:

1. Se carga una señal de simulación (velocidad de la herramienta) desde un archivo HDF5 de una campaña DOE (`HDF5Reader`).
2. Se recorta la señal a una ventana de análisis y se separa en dos tramos de entrenamiento: uno "estable" (sin chatter) y uno "chatter" (con chatter), según intervalos de tiempo definidos a mano (`training_intervals`) o un único instante de corte legado (`t_stable_total`).
3. Ambos tramos se sub-muestrean a una muestra por revolución del husillo (OPR, *Once-Per-Revolution*) o se usan en su resolución completa ("raw"), se segmentan en bloques (con posibilidad de solape/"hop"), y de cada bloque se extrae un valor escalar de **entropía** (bajo el supuesto de máxima entropía gaussiana).
4. Con las entropías de ambos regímenes se ajustan dos gaussianas `P0` (estable) y `P1` (chatter) — el modelo MaxEnt.
5. Sobre la señal completa se repite la segmentación/entropía y se aplica un **SPRT**: se acumula el logaritmo de la razón de verosimilitud `P1/P0` segmento a segmento, comparando contra dos umbrales `a` (aceptar H0/estable) y `b` (aceptar H1/chatter) derivados de las probabilidades de error `alpha`/`beta` deseadas.
6. El resultado (tiempos de detección, estadístico acumulado `S_k`, metadatos de entrenamiento) se reporta por `logging` en distintos niveles de detalle y se visualiza mediante un conjunto de ~20 figuras de diagnóstico (`plots_maxent_sprt`).

El paquete soporta tres formas de parametrizar el tamaño/paso de los segmentos: `native` (parámetros directos en muestras OPR), `by_revolution` (en número de revoluciones del husillo) y `by_modal` (en múltiplos del periodo modal de chatter esperado) — las dos últimas se traducen internamente a la primera.

---

## 2. Mapa del flujo real (árbol de dependencias auditado)

Partiendo de `examples/MaxEnt_Detection_NEW.py`, siguiendo imports reales (no solo lo físicamente ubicado en la carpeta indicada):

```
examples/MaxEnt_Detection_NEW.py
└── MaxEnt_SPRT (src/MaxEnt_SPRT/__init__.py)
    ├── logging_setup.py                      (configure_logging, _section, INFO_PLUS_LEVEL)
    ├── utils/types.py                        (SignalData, IndicatorResult)
    ├── utils/hdf5_utils.py                   (HDF5Reader)
    ├── utils/opr.py                          (sample_opr, segment_opr, segment_signal_raw)
    ├── models/prob.py                        (GaussianPDF)
    ├── models/maxent.py                      (MaxEntModels, fit_maxent_gaussians)
    ├── lib/entropy.py                        (EntropyEstimator y subclases, entropy_from_segments)
    ├── lib/llr.py                            (LLRModel, GaussianIndicatorLLR)
    ├── lib/sprt.py                           (SPRTConfig, SPRTResult, SequentialProbabilityRatioTest)
    ├── lib/offline.py                        (offline_train_maxent_sprt)
    ├── lib/detector.py                       (MaxEntSPRTConfig, MaxEntSPRTDetector, + copia local de SequentialProbabilityRatioTest)
    ├── lib/runner.py                         (run_maxent_sprt, _maxent_sprt_pipeline, _resolve_physical_params_maxent)
    └── viz/maxent_sprt_plots.py              (plots_maxent_sprt)
```

**Archivos del paquete NO incluidos en el flujo real** (existen físicamente bajo `src/MaxEnt_SPRT/` pero ningún módulo del flujo los importa): `lib/classes_user.py`, `lib/constants_user.py`, `lib/functions_user.py`, `lib/original_imports.py`, `lib/wrappers.py`, `utils/validation.py`. Se documentan en la sección de hallazgos (E17) porque su presencia sin uso, y una dependencia externa no declarada (`C_emd_hht`) dentro de `original_imports.py`, son en sí mismas un riesgo de mantenimiento — pero no se auditaron línea por línea por quedar fuera del flujo real que la skill exige cubrir.

También quedan fuera del flujo: `examples/read_hdf_signals.py`, `examples/MaxEnt_Detection_old.py`, `examples/MaxEnt_Modal_vs_Rev.py`, `examples/MaxEnt_Theory_Demo.py` (scripts alternativos no importados por el archivo de entrada indicado) y todo `legacy/`.

---

## 3. Análisis por archivo/módulo

### `examples/MaxEnt_Detection_NEW.py` (entrada)
Script de aplicación: define rutas HDF5 hardcodeadas, carga la señal, define 7 diccionarios de configuración alternativos (`INDICATOR_CONFIG_*`) para los 3 modos, selecciona uno, ejecuta `run_maxent_sprt` y despliega resultados por `logging` + `plots_maxent_sprt`. No contiene lógica de detección; es pura orquestación y presentación.

### `src/MaxEnt_SPRT/__init__.py`
Reexporta la API pública del paquete y registra el nivel de logging custom `INFO_PLUS`. Correcto y simple.

### `src/MaxEnt_SPRT/logging_setup.py`
Helper de configuración de `logging` (niveles, formato) y una función `_section()` para separadores visuales. `_section` es "privada" (prefijo `_`) pero se importa directamente tanto desde el script de ejemplo como desde `lib/runner.py` — funciona, pero rompe la convención de encapsulamiento de Python (ver M-notas).

### `src/MaxEnt_SPRT/utils/types.py`
Define los `dataclass` `SignalData`, `IndicatorResult` y `ScenarioMetadata` (esta última no se usa en el flujo real). Estructuras de datos simples, bien documentadas, sin lógica.

### `src/MaxEnt_SPRT/utils/hdf5_utils.py`
`HDF5Reader`: carga **todo** el archivo HDF5 en memoria como diccionario anidado de `numpy.ndarray` al construirse, con decodificación de bytes/strings, y expone navegación por rutas tipo `"grupo/subgrupo/dataset"` con soporte de slicing/indexado e incluso búsqueda difusa de claves ambiguas. Ver E9 (carga eager) y M5 (complejidad del parser de índices).

### `src/MaxEnt_SPRT/utils/opr.py`
Tres funciones puras de remuestreo/segmentación: `sample_opr` (decimación a 1 muestra/rev), `segment_opr` y `segment_signal_raw` (ventaneo con solape opcional). Ver E7 (validación deshabilitada + desfase de índice) y E8 (sin protección ante `step<=0`).

### `src/MaxEnt_SPRT/models/prob.py`
`GaussianPDF`: modelo gaussiano inmutable con `logpdf`, `entropy_shannon`, `from_samples`. Ver E16 (uso de `print` en vez de logging).

### `src/MaxEnt_SPRT/models/maxent.py`
`MaxEntModels` (par `p0`/`p1`) y `fit_maxent_gaussians`. Simple, delega en `GaussianPDF.from_samples`.

### `src/MaxEnt_SPRT/lib/entropy.py`
Jerarquía `EntropyEstimator` (ABC) con dos implementaciones: `GaussianMaxEntEstimator` (la usada en el flujo real) y `EmpiricalHistogramEntropyEstimator` (alternativa no usada por el pipeline actual, pero exportada). Código limpio.

### `src/MaxEnt_SPRT/lib/llr.py`
`LLRModel` (ABC) y `GaussianIndicatorLLR`: calcula `log p1(h) - log p0(h)`. Correcto y mínimo.

### `src/MaxEnt_SPRT/lib/sprt.py`
`SPRTConfig` (umbrales de Wald `a`, `b` a partir de `alpha`/`beta`), `SPRTResult`, y el motor `SequentialProbabilityRatioTest.run()`. Implementa correctamente la fórmula de Wald para los umbrales; el bucle de acumulación permite reinicios tras aceptar H0 (`reset_on_H0`). Ver E6 (duplicado en `detector.py`) y E15 (semántica de `final_state`/`decision_index`).

### `src/MaxEnt_SPRT/lib/offline.py`
`offline_train_maxent_sprt`: segmenta ambos regímenes (OPR o raw), calcula entropías, ajusta `MaxEntModels`. Valida que haya al menos un segmento por régimen. Correcto.

### `src/MaxEnt_SPRT/lib/detector.py`
`MaxEntSPRTDetector`: orquesta entrenamiento offline (`fit_offline_from_opr`, `fit_offline_from_signals`) y detección online (`detect_from_H_seq`, `detect_online_from_signal`). Contiene una **copia local** de `SequentialProbabilityRatioTest` en vez de importar la de `lib/sprt.py` (E6). El resto de la clase es correcto y bien documentado.

### `src/MaxEnt_SPRT/lib/runner.py`
El módulo más complejo del flujo (665 líneas): traduce parámetros físicos (`by_revolution`/`by_modal`) a nativos (`_resolve_physical_params_maxent`), orquesta todo el pipeline (`_maxent_sprt_pipeline`) y ensambla trazabilidad + logging (`run_maxent_sprt`). Contiene el **bug más severo de la auditoría** (E1: `UnboundLocalError` en modo `native`) y varias inconsistencias de metadatos (E2, E3, E4, E10, E11).

### `src/MaxEnt_SPRT/viz/maxent_sprt_plots.py`
Módulo de visualización (1893 líneas, ~20 funciones internas anidadas dentro de `plots_maxent_sprt`) que genera un extenso conjunto de figuras de diagnóstico (señal+OPR, histogramas de entropía, PDFs teóricas P0/P1 con regiones alpha/beta, evolución temporal de log-verosimilitudes, estadístico SPRT). Muy útil para depuración visual, pero con alta duplicación interna (M1), funciones muertas (E14), parámetros sin efecto (E12) y un contrato de retorno incumplido (E13).

---

## 4. Hallazgos: errores y riesgos

Los hallazgos están numerados y referenciados desde `CHECKLIST.md`. Todos fueron releídos contra el código real durante la autorevisión (sección 6) y confirmados.

### E1 — CRÍTICO — `lib/runner.py:278-292` — `UnboundLocalError` garantizado en modo `native`

```python
if param_mode == "native":
    params: Dict[str, Any] = INDICATOR_CONFIG.get("params", {})
else:
    params_physical: Dict[str, Any] = INDICATOR_CONFIG["params_physical"]
    params, trace = _resolve_physical_params_maxent(param_mode, params_physical, fs)
    ...

result: IndicatorResult = func(signal, **params)          # línea 276

if params_physical.get("T_rev", None) is not None:         # línea 278 ← CRASH en modo native
```

`params_physical` solo se asigna dentro de la rama `else`. Como Python determina el ámbito de una variable analizando **toda** la función (no rama por rama), la asignación en el `else` convierte a `params_physical` en variable local de `run_maxent_sprt` para toda su extensión. Cuando se ejecuta con `param_mode == "native"` (la rama `if`), esa variable nunca llega a asignarse, y la referencia de la línea 278 lanza `UnboundLocalError: local variable 'params_physical' referenced before assignment`.

**Impacto**: el modo `native` es uno de los tres modos de configuración documentados (ver comentario en el propio `runner.py:32-47` y la configuración `INDICATOR_CONFIG_native` predefinida en `examples/MaxEnt_Detection_NEW.py:165-174`, dejada como opción comentada en la línea 240). Cualquier usuario que reactive esa línea (`INDICATOR_CONFIG = INDICATOR_CONFIG_native`) obtiene un crash **después** de que `func(signal, **params)` ya ejecutó todo el pipeline (entrenamiento offline + detección online), es decir, se pierde el cómputo completo y no se genera ningún resultado ni gráfica.

**Corrección conceptual** (no aplicada, solo diagnóstico): inicializar `params_physical = None` antes del `if/else`, o mover las líneas 278-292 dentro de un `if param_mode != "native":`.

### E2 — ALTO — `lib/runner.py:278-281` — rama muerta: `f_cycle` nunca usa `T_modal`

```python
if params_physical.get("T_rev", None) is not None:
    f_cycle = 1 / (params_physical.get("T_rev", "n/a"))
else:
    f_cycle = 1 / (params_physical.get("T_modal", "n/a"))
```

`_resolve_physical_params_maxent` (líneas 79-92 para `by_revolution`, 138-154 para `by_modal`) exige `T_rev` como clave obligatoria en **ambos** modos físicos. Por tanto, para cualquier `param_mode` que logre llegar a la línea 278 (es decir, `by_revolution` o `by_modal`; `native` ya crasheó antes en E1), `params_physical.get("T_rev")` siempre es distinto de `None`, y la rama `else` (línea 281, que usaría `T_modal`) es código inalcanzable.

**Impacto real**: en modo `by_modal`, el campo mostrado en el log como `"Frecuency Cycle:"` (runner.py:305) siempre reporta `1/T_rev` (frecuencia de rotación del husillo) en lugar de `1/T_modal` (frecuencia modal de chatter), que es lo que el modo `by_modal` pretende resaltar. El usuario que lea ese log en modo `by_modal` recibe una cifra incorrecta para lo que se le está mostrando.

### E3 — MEDIO — `lib/runner.py:292,307` — `"Total_window"` es un duplicado mal etiquetado de `N_seg`

```python
result.meta ["Total_window"] = result.meta["N_seg"]     # línea 292
...
logger.info("  %-24s %d",     "Entropie Windows:", result.meta.get("N_seg", "n/a"))   # línea 306
logger.info("  %-24s %d",      "Total Windows",           result.meta["Total_window"])  # línea 307
```

`Total_window` se define como una copia exacta de `N_seg` (tamaño de un segmento en revoluciones/muestras), no como el número total de segmentos/ventanas procesadas durante la corrida (ese valor ya existe, calculado independientemente, como `result.meta["Total_segments"]` en `_maxent_sprt_pipeline`, línea 635). El log resultante muestra dos etiquetas ("Entropie Windows" y "Total Windows") con el mismo número, y ninguna es realmente el total de ventanas.

### E4 — MEDIO — `lib/runner.py:572,601` y `examples/MaxEnt_Detection_NEW.py:161` — `t_theorical` sí se usa en la detección pese a su documentación

El parámetro se declara así en tres lugares distintos, siempre con la misma anotación:

```python
t_theorical: Optional[float] = None,  # for debug/plots, not used in detection
```

Sin embargo, en `_maxent_sprt_pipeline` se usa activamente para filtrar detecciones:

```python
t_d_no_FAR_idx = np.where(chatter_points_time > t_theorical)[0]   # runner.py:572 y :601
```

Esto es inconsistente: el comentario afirma que el parámetro es solo para depuración/gráficas y "no se usa en la detección", pero condiciona directamente qué detecciones se consideran "sin FAR" (`t_d_no_FAR`), un campo que **sí** se reporta como resultado operativo ("Primera Detección Non FAR", `examples/MaxEnt_Detection_NEW.py:329`). Además, al tener valor por defecto `None`, un usuario que confíe en la documentación y omita el parámetro obtiene `chatter_points_time > None`, comparación no soportada de forma fiable entre `numpy.ndarray` de floats y `None` (comportamiento no garantizado entre versiones de NumPy).

### E5 — MEDIO — `examples/MaxEnt_Detection_NEW.py:329` — `IndexError` si no hay detecciones posteriores a `t_theorical`

```python
if logger.isEnabledFor(logging.INFO):   # engloba hasta la línea ~367, pero el guard real es result.t_d.size>0, línea 300
    ...
logger.info("  %-24s %.3f s", "Primera Detecion Non Far:",  result.t_d_no_FAR[0])   # línea 329
```

El bloque está protegido por `if result.t_d.size > 0:` (runner.py:300, dentro de `run_maxent_sprt`, que solo imprime el resumen WARNING/INFO si hubo alguna detección SPRT). Pero `t_d` (todas las detecciones donde `S_k >= b`) y `t_d_no_FAR` (subconjunto de `t_d` con tiempo posterior a `t_theorical`, runner.py:572-573) son conjuntos independientes. Es perfectamente posible que existan detecciones tempranas (falsas alarmas antes del `t_theorical`/ground truth) sin ninguna detección posterior, dejando `t_d_no_FAR` vacío mientras `t_d` no lo está. En ese caso `result.t_d_no_FAR[0]` lanza `IndexError: index 0 is out of bounds for axis 0 with size 0`.

### E6 — ALTO — `lib/detector.py:17-87` duplica íntegramente `lib/sprt.py:72-135`

`lib/detector.py` importa explícitamente `SPRTConfig, SPRTResult` desde `.sprt` (línea 9) pero **no** importa `SequentialProbabilityRatioTest`; en su lugar define una clase con el mismo nombre y una implementación casi idéntica en sus propias líneas 17-87. El paquete público expone la versión de `lib/sprt.py` (`__init__.py:39`, `from .lib.sprt import ... SequentialProbabilityRatioTest`), pero el motor realmente usado en producción es el de `lib/detector.py` (invocado en `MaxEntSPRTDetector.detect_from_H_seq`, línea 306: `sprt = SequentialProbabilityRatioTest(...)`, resuelto por ámbito local al de `detector.py`, no al importado del paquete).

Evidencia de que ya han divergido: la copia de `detector.py` agrega, en el docstring de `run()`, la frase *"The current implementation keeps scanning the full sequence instead of exiting early on the first detection"*, ausente en `lib/sprt.py`. Es decir, alguien documentó una particularidad solo en una de las dos copias.

**Riesgo**: cualquier corrección futura de la lógica SPRT (p. ej. cambiar el criterio de parada, agregar `break` en la primera detección, corregir el manejo de `reset_on_H0`) debe aplicarse manualmente en dos sitios. Si solo se corrige uno, el comportamiento de `MaxEnt_SPRT.SequentialProbabilityRatioTest` (API pública) queda desincronizado del motor realmente usado por `MaxEntSPRTDetector` (uso interno), produciendo un bug silencioso y difícil de diagnosticar porque ambas clases se llaman igual.

### E7 — MEDIO — `utils/opr.py:8-34` (`sample_opr`) — validación deshabilitada y desfase de índice sin documentar

```python
ratio = fs / fr
# if abs(ratio - round(ratio)) > 1e-9:
#     raise ValueError("fs/fr must be an integer for exact OPR sampling.")
step = int(math.ceil(ratio))
return y[step::step], t[step::step]
```

El docstring de la función (líneas 25-27) promete: *"Raises: ValueError: If fs/fr is not an integer within numerical tolerance"*, pero la comprobación está comentada. En su lugar, `step` se redondea silenciosamente hacia arriba con `math.ceil`, sin ningún warning ni log, lo que puede introducir muestreo OPR sistemáticamente sesgado cuando `fs/fr` no es exactamente entero (caso habitual con señales de simulación con paso de tiempo arbitrario, como las usadas en `examples/MaxEnt_Detection_NEW.py`, donde `fs = 1.0/(t[1]-t[0])` es un número flotante calculado, no un valor de diseño exacto).

Además, `y[step::step]` (y el `t` correspondiente) empieza en el índice `step`, no en `0` ni en `step-1`: la primera revolución completa de la señal se descarta siempre, sin que el docstring lo mencione (dice únicamente "selects every `step` sample... preserving time alignment"). Esto desplaza sistemáticamente la fase de todo el muestreo OPR posterior (entrenamiento y detección), y quien lea solo la documentación no lo esperaría.

### E8 — MEDIO — `utils/opr.py:36-104` (`segment_opr`, `segment_signal_raw`) — sin protección ante `step <= 0`

```python
if step is None:
    step = N_seg
segments: List[np.ndarray] = []
...
start = 0
while start + N_seg <= len(opr):
    segments.append(opr[start:start + N_seg])
    ...
    start += step
```

Si `step` es `0`, la condición del bucle nunca cambia y el proceso se cuelga indefinidamente (bucle infinito, memoria creciendo con cada `append`); si `step` es negativo, `start` decrece y el bucle también corre indefinidamente hacia atrás. `lib/runner.py` es quien pasa `step` (renombrado `step_seg`) desde la configuración de usuario, y las comprobaciones de rango están **explícitamente comentadas**:

```python
# step_seg = int(step_rev)
# if not (1 <= step_seg <= N_seg):
#     raise ValueError(...)                      # runner.py:102-106, análogo en :170-173
```

Dado que `examples/MaxEnt_Detection_NEW.py` invita al usuario a editar a mano valores como `"step_rev": 1` (líneas 185, 201, 218, 234), un simple error de tecleo (`0` en vez de `1`) produce un cuelgue silencioso sin ningún mensaje de error, en vez de una excepción clara e inmediata.

### E9 — BAJO — `utils/hdf5_utils.py:23-41` — `HDF5Reader` carga todo el archivo HDF5 en memoria

```python
def _read_file(self) -> Dict[str, Any]:
    with h5py.File(self.filepath, "r") as hdf_file:
        return self._read_group(hdf_file)          # recorre TODOS los grupos/datasets
```

El constructor (`__init__`, línea 19) llama a `_read_file()` de forma eager: recorre recursivamente cada grupo y dataset del archivo HDF5 y lo materializa como `numpy.ndarray`/lista Python en memoria, sin ninguna opción de carga selectiva o perezosa. Las rutas usadas en el script de ejemplo (`examples/MaxEnt_Detection_NEW.py:45-67`) apuntan a archivos de una campaña DOE de simulación (`2DOF_Cone_DOE/...`), que típicamente contienen múltiples canales (`tool_dyn`, `Axial_disp`, `Axial_vel`, `force_N`, etc.) aunque el script solo necesite 2-3 de ellos. Esto es un riesgo de rendimiento/memoria que crece con el tamaño y número de canales del archivo, sin relación con la cantidad de datos realmente usados después.

### E10 — BAJO — `examples/MaxEnt_Detection_NEW.py:380-381` — `dict.get(clave, default)` no dispara el `default` esperado

```python
("OPR libres",       str(meta.get("Sampled OPR free",  "N/A (raw)" ))),
("OPR chatter",      str(meta.get("Sampled OPR chatter","N/A (raw)" ))),
```

`dict.get(key, default)` solo devuelve `default` si la clave **no existe**. Pero `lib/runner.py:638-639` siempre define esas claves, poniéndolas a `None` cuando `segmentation == "raw"` (`opr_free`/`opr_chat` son `None` en ese modo, según `runner.py:513`):

```python
"Sampled OPR free":    opr_free.size if opr_free is not None else None,
"Sampled OPR chatter": opr_chat.size if opr_chat is not None else None,
```

Como la clave existe (con valor `None`), `meta.get(...)` devuelve `None`, no el texto de repuesto. El log de depuración en modo `raw` termina mostrando literalmente la cadena `"None"` en lugar del mensaje pensado `"N/A (raw)"`.

### E11 — BAJO — `lib/runner.py:635` — `Total_segments` ignora el solapamiento (`step_seg`)

```python
"Total_segments": int(t_total*fr/N_seg),
```

Este cálculo asume implícitamente `step == N_seg` (sin solape). Cuando `step_seg < N_seg` (el caso central del branch actual, "overlap/hop"), el número real de segmentos generado por `segment_opr`/`segment_signal_raw` es mayor (aproximadamente `(N_muestras - N_seg) // step_seg + 1`). El valor mostrado como "segmentos totales" en los logs de depuración no coincide con el número real de segmentos procesados en ejecuciones con solape, que es precisamente el escenario que este branch pretende soportar.

### E12 — MEDIO — `viz/maxent_sprt_plots.py:120-121,1891` — parámetros `show`/`show_signal` sin efecto

```python
def plots_maxent_sprt(
    signal, result,
    show_signal: bool = True,
    show: bool = True,
    ...
):
    ...
    plt.show()   # línea 1891, incondicional
```

Ambos parámetros están documentados como "reserved flag" que controlan si se muestra el panel de señal (`show_signal`) o si la figura se despliega interactivamente (`show`), pero ninguno se lee en ningún punto del cuerpo de la función (verificado por búsqueda exhaustiva en las 1893 líneas). `plt.show()` se invoca siempre, sin importar `show=False`; el panel de señal (O4, F6) se genera siempre, sin importar `show_signal=False`. Cualquier código que intente automatizar la generación de figuras sin bloquear la ejecución (p. ej., para guardarlas a disco en un pipeline batch, o para tests) no obtiene el comportamiento que la firma promete.

### E13 — MEDIO — `viz/maxent_sprt_plots.py:117-128,1890-1891` — la función no retorna lo que anuncia

La firma anota `-> plt.Figure` y la docstring promete: *"Returns: plt.Figure: Final Matplotlib figure object associated with the last plot created by the routine."* Sin embargo, el cuerpo de la función termina así:

```python
plt.tight_layout()
plt.show()
```

Sin ninguna sentencia `return`. Toda llamada a `plots_maxent_sprt(...)` devuelve `None` (el valor de retorno implícito de Python), no un `Figure`. Cualquier consumidor que escriba `fig = plots_maxent_sprt(...)` para luego, por ejemplo, `fig.savefig("resultado.png")`, falla con `AttributeError: 'NoneType' object has no attribute 'savefig'`.

### E14 — BAJO (código muerto) — `viz/maxent_sprt_plots.py` — cinco funciones internas nunca invocadas

`_plot_signal` (líneas 151-186), `_plot_opr` (188-227), `_plot_pdf` (229-275), `_plot_PDF_model` (277-321) y `_plot_H` (323-364) están definidas dentro de `plots_maxent_sprt`, pero ninguna se llama en el cuerpo principal de la función (líneas 1455-1891, donde se generan las figuras O1-O4, F1-F7, S1, D3-D9). El conjunto de figuras realmente producido usa otro grupo de funciones (`_plot_H_time_segment`, `_plot_H_time_all`, `_plot_PDF_hist`, etc.) que las reemplazó sin eliminar las anteriores.

Dos observaciones adicionales sobre este código muerto, indicativas de que quedó a medio actualizar:
- `_plot_pdf` (229-275) anota `-> tuple[plt.Figure, plt.Axes]` pero no tiene ninguna sentencia `return`.
- `_plot_H` (323-364) recibe `H_chat` como parámetro y calcula un rango `xs` a partir de `H_free` y `H_chat` (líneas 339-343) que tampoco se usa después; solo grafica `H_free` (línea 346), pese a que la intención evidente (por el nombre de la función y sus parámetros) era graficar ambos regímenes.

En total, unas 210 líneas de complejidad sin ningún efecto en la salida real del programa.

### E15 — BAJO (riesgo de interpretación) — `lib/sprt.py:114-127` — `final_state`/`decision_index` reflejan la última transición, no si hubo detección

```python
for i, h_obs in enumerate(H_list):
    S += self.llr_model.llr(h_obs)
    S_hist[i] = S
    if S <= a:
        state = "free"
        idx_decision = i
        if self.config.reset_on_H0:
            S = 0.0
    if S >= b:
        state = "chatter"
        idx_decision = i
```

Con `reset_on_H0=True` (el valor usado en el ejemplo), el bucle sigue acumulando evidencia después de cada decisión, por lo que `state`/`idx_decision` pueden alternar varias veces entre `"free"` y `"chatter"` a lo largo de la corrida; el valor final refleja la **última** transición, no si en algún momento se detectó chatter. Esto está documentado correctamente en el docstring de `SPRTResult.decision_index` ("Index of the segment where the **latest** decisive threshold crossing occurred"), por lo que no es, en rigor, un bug — pero el nombre `final_state` invita a una lectura ambigua ("¿es el estado final del proceso, o el resultado global de la prueba?"), y el consumo real de las detecciones múltiples se hace por otra vía (`S_history >= b`, `lib/runner.py:569`), de modo que es posible obtener, por ejemplo, `final_state == "free"` en un `IndicatorResult` que sí registra detecciones de chatter (`t_d.size > 0`), lo cual puede confundir a quien no conozca el diseño interno.

### E16 — BAJO — `models/prob.py:87-88` — `print()` en vez de logging

```python
if var <= eps:
    print("Advertencia: varianza muy pequeña, aplicando floor para evitar sigma=0.")
```

Es el único punto de todo el flujo real que usa `print()` en vez del sistema de `logging` del paquete (usado consistentemente en el resto: `logger.info_plus`, `logger.debug`, etc., configurable mediante `configure_logging`). Esto rompe el control de verbosidad: el mensaje aparece en `stdout` sin importar el nivel de logging configurado por el usuario (ni siquiera se puede silenciar con `logging.WARNING` a nivel de aplicación), y no puede capturarse con un `logging.Handler`.

### E17 — BAJO (mantenibilidad del paquete) — módulos huérfanos fuera del flujo real, con dependencia externa no declarada

`lib/classes_user.py`, `lib/constants_user.py`, `lib/functions_user.py`, `lib/original_imports.py`, `lib/wrappers.py` y `utils/validation.py` no son importados por `__init__.py` ni por ningún módulo del flujo real (`runner.py`, `detector.py`, etc. — verificado por búsqueda de referencias en todo `src/MaxEnt_SPRT/`). Además, `lib/original_imports.py:10` importa:

```python
from C_emd_hht import signal_chatter_example, sinus_6_C_SNR
```

un paquete externo que no aparece como dependencia del proyecto instalable (`pyproject.toml`). Si en el futuro alguien importa `constants_user` o `functions_user` (que a su vez hacen `from .original_imports import *`), el import falla con `ModuleNotFoundError` salvo que `C_emd_hht` esté disponible de forma ad-hoc en el entorno de quien ejecute el código. Estos archivos deberían migrarse a `legacy/` (que ya existe y aloja otro código histórico) o eliminarse, para que la superficie de `src/MaxEnt_SPRT/` refleje únicamente el código activo.

### R1 — MEDIO (riesgo de configuración) — `examples/MaxEnt_Detection_NEW.py:239-246` — doble asignación silenciosa de `INDICATOR_CONFIG`

```python
# INDICATOR_CONFIG = INDICATOR_CONFIG_native
# INDICATOR_CONFIG = INDICATOR_CONFIG_by_revolution
# INDICATOR_CONFIG = INDICATOR_CONFIG_by_modal
INDICATOR_CONFIG = INDICADOR_CONFIG_by_revolution_overlap      # línea 243 — activa
# INDICATOR_CONFIG = INDICADOR_CONFIG_by_modal_overlap
INDICATOR_CONFIG = INDICADOR_CONFIG_by_revolution_raw           # línea 245 — activa, sobrescribe la anterior
# INDICATOR_CONFIG = INDICADOR_CONFIG_by_modal_raw
```

Dos líneas de asignación quedan descomentadas a la vez (243 y 245); Python simplemente conserva la última (`by_revolution_raw`), pero el patrón — "selector por comentar/descomentar" sin ningún mecanismo que impida tener dos líneas activas simultáneamente — es propenso a que un usuario crea estar usando la configuración de la línea 243 (por ejemplo, tras editarla) cuando en realidad se está ejecutando la de la línea 245. No hay ningún error, ni siquiera un warning: el comportamiento observado (parámetros, tiempos de detección) simplemente no corresponde a lo que el usuario cree haber configurado.

### R2 — BAJO — `examples/MaxEnt_Detection_NEW.py:45-67` — variables muertas y nombres casi idénticos

`cono_doe_control` (línea 45) y `cono_doe_control_sensor` (línea 51) se definen y nunca se usan. `dir_custome` (línea 66) también se define y nunca se usa — en su lugar, `data_dir` (línea 67) reutiliza la variable `custome` (línea 58), de nombre casi idéntico a `dir_custome`. Esto no es un bug hoy, pero es un riesgo claro de que una futura edición copie/pegue mal y use la variable equivocada sin que nada lo detecte (ambas son rutas de archivo `str`, así que un error de este tipo no lanza ninguna excepción, solo carga el archivo equivocado).

---

## 5. Mejoras propuestas (calidad, no bugs)

### M1 — `viz/maxent_sprt_plots.py` — extraer el boilerplate de ejes repetido en ~20 funciones

Prácticamente todas las funciones internas de graficación (`_plot_signal`, `_plot_opr`, `_plot_H_time_segment`, `_plot_D3_figure`, etc.) repiten el mismo bloque:

```python
if zoom_x is not None: ax.set_xlim(zoom_x)
if zoom_y is not None: ax.set_ylim(zoom_y)
_draw_vlines(ax, vlines)
if hlines is not None:
    for hy in hlines: ax.axhline(y=hy, color='gray', linestyle='--', alpha=0.7)
```

Extraer una función común `_apply_common_axis_options(ax, zoom_x, zoom_y, vlines, hlines)` reduciría el archivo (1893 líneas) de forma significativa y centralizaría cualquier cambio futuro de estilo (p. ej., cambiar el color por defecto de las líneas horizontales) a un único punto en vez de ~20.

### M2 — `viz/maxent_sprt_plots.py` — reutilizar `GaussianPDF`/`GaussianIndicatorLLR` en vez de reimplementar la matemática

Las funciones `_plot_D3_figure`, `_plot_D6_figure`, `_plot_D7_figure`, `_plot_D8_figure`, `_plot_D9_figure` recalculan a mano `logpdf`, `pdf` y `Lambda = ln(p1/p0)` usando `scipy.stats.norm` y fórmulas explícitas, en vez de reutilizar `models.prob.GaussianPDF.logpdf` y `lib.llr.GaussianIndicatorLLR.llr`, que son exactamente las clases que el propio detector usa para calcular las mismas cantidades. Si el modelo estadístico cambiara (otro `min_sigma`, otra familia de distribución), las gráficas podrían dejar de representar fielmente lo que el detector calcula, sin que nada lo señale.

### M3 — `lib/runner.py` — reemplazar los `dict` de configuración sueltos por una `dataclass`/esquema validado

`INDICATOR_CONFIG` (y `params_physical`) son diccionarios sin ningún esquema o validación de tipos hasta que `_resolve_physical_params_maxent` los procesa parcialmente. Una `dataclass` (o `pydantic`) con validación en el constructor detectaría errores como el de E1 (modo inexistente/mal soportado) o E8 (`step<=0`) de forma inmediata y con un mensaje claro, en vez de fallar tarde (o colgarse) tras minutos de cómputo.

### M4 — `lib/runner.py::_maxent_sprt_pipeline` — dividir la función de 260 líneas

La función mezcla preparación de señal, entrenamiento offline, detección online, selección de estrategia SPRT/umbral y ensamblado del resultado final en un único cuerpo. Separarla en funciones más pequeñas y testeables (`_prepare_training_signals`, `_run_offline_training`, `_run_online_detection`, `_build_result_meta`) facilitaría pruebas unitarias dirigidas y revisiones de código futuras.

### M5 — `utils/hdf5_utils.py::HDF5Reader.get_element` — simplificar el parser de índices

El método implementa a mano soporte de slices (`"0:10"`), listas (`"[0,2,4]"`), índices multidimensionales y búsqueda difusa de rutas ambiguas, con manejo de errores genérico (`except Exception as e: raise KeyError(...)`) que oculta el tipo real de excepción subyacente. Dado que h5py ya soporta navegación directa por rutas (`file["grupo/subgrupo/dataset"]`), gran parte de esta lógica ad-hoc podría eliminarse a cambio de exigir rutas explícitas, reduciendo drásticamente la superficie de bugs potenciales (p. ej., resolución ambigua cuando dos grupos distintos comparten el nombre final de la clave buscada, líneas 229-239).

### M6 — Consistencia idiomática (español/inglés)

Comentarios, docstrings y mensajes de log mezclan español e inglés de forma inconsistente en casi todos los archivos del flujo (`utils/hdf5_utils.py`, `lib/runner.py`, `examples/MaxEnt_Detection_NEW.py`). No es un bug, pero dificulta tanto el mantenimiento como una eventual documentación bilingüe coherente del proyecto.

---

## 6. Métricas de la auditoría

- **Pasadas de autorevisión ejecutadas**: 1 pasada completa de autorevisión (más una verificación dirigida por `grep` de los hallazgos de mayor severidad antes de redactar este informe, para confirmar número de línea y comportamiento exacto). Todos los hallazgos reportados fueron releídos contra el código real durante esa pasada.
- **Tasa de confirmación de la pasada**: 100 % — de los hallazgos revisados en la pasada de autorevisión, ninguno fue descartado por falso positivo ni requirió corrección de ubicación/explicación (los pocos ajustes de precisión, p. ej. confirmar el rango exacto de líneas de la clase duplicada en `detector.py`, se hicieron *durante* la redacción, no como corrección posterior).
- **¿Convergió o se cortó por el límite de 5 pasadas?**: Convergió — no aparecieron archivos/dependencias nuevos ni hallazgos nuevos en la pasada de verificación; no fue necesario un límite de 5 pasadas.
- **Archivos del flujo real auditados línea por línea**: 15 archivos `.py` (entrada + 14 módulos del paquete), ≈4300 líneas de código fuente real (excluyendo `__init__.py` triviales de subpaquetes).
- **Tiempo de ejecución total de la auditoría**: desde `Sun, Sep 13, 2026 11:55:23 PM` hasta el cierre de este informe (`Mon, Sep 14, 2026`), aproximadamente 35-40 minutos de trabajo efectivo del agente.
- **Tokens utilizados**: el contexto del agente mostraba `14 953 371` tokens restantes al iniciar la auditoría (system-reminder `<total_tokens>`) y `14 792 465` tokens restantes justo antes de redactar los documentos finales — un consumo aproximado de `160 906` tokens (~1.1 % del presupuesto disponible en ese momento). No se dispone del tamaño total de la ventana de contexto del modelo como denominador absoluto, por lo que el porcentaje reportado es relativo al presupuesto de tokens restantes observado al inicio de esta tarea, no al límite total del modelo.

---

## 7. Resúmenes finales

### Qué hace el código
Implementa un detector de chatter en mecanizado (MaxEnt-SPRT): entrena dos modelos gaussianos de entropía (estable/chatter) a partir de tramos etiquetados de una señal de vibración, y aplica un test secuencial de razón de verosimilitud sobre la señal completa para decidir, segmento a segmento, si el proceso sigue estable o ha entrado en chatter, reportando tiempos de detección y un extenso set de gráficas de diagnóstico.

### Resumen de mejoras
El mayor margen de mejora está en `viz/maxent_sprt_plots.py` (duplicación masiva de boilerplate de graficación, reimplementación manual de matemática ya disponible en el paquete, funciones muertas) y en la arquitectura de configuración de `lib/runner.py` (diccionarios sin validación temprana, función monolítica de 260 líneas). Adoptar una `dataclass` de configuración validada y refactorizar el módulo de gráficas eliminaría de un golpe varias clases de bug (incluido el más crítico, E1) y reduciría el código en varios cientos de líneas.

### Resumen de lo que está mal
El hallazgo más grave es que el modo de configuración `native` está roto (`UnboundLocalError` garantizado, E1); en modo `by_modal` se reporta una frecuencia incorrecta en los logs (E2); la clase central del SPRT está duplicada entre `lib/sprt.py` y `lib/detector.py` con evidencia de que ya divergieron (E6); y el módulo de visualización tiene un contrato de retorno roto (`plots_maxent_sprt` nunca retorna una `Figure` pese a anunciarlo, E13) además de parámetros (`show`, `show_signal`) sin ningún efecto (E12). Varios metadatos de diagnóstico (`Total_window`, `Total_segments`, "Sampled OPR ... N/A (raw)") muestran valores incorrectos o engañosos sin que el usuario tenga forma de notarlo salvo leyendo el código fuente.
