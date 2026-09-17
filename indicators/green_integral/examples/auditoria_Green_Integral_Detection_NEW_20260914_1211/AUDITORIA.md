
# AUDITORÍA — `Green_Integral_Detection_NEW.py`

Auditoría técnica de solo lectura, exhaustiva, del indicador **Green Integral**
(detección de chatter en usinado por área de fase desplazamiento–velocidad).
No se ha modificado ni ejecutado ningún archivo.

- **Carpeta base**: `indicators/green_integral/`
- **Archivo de entrada**: `indicators/green_integral/examples/Green_Integral_Detection_NEW.py`
- **Profundidad**: exhaustivo (línea por línea, todo el árbol de dependencias real)

---

## 1. Resumen del código

`Green_Integral_Detection_NEW.py` es un script de ejemplo que:

1. Selecciona un caso de estudio (`_ACTIVE_CASE`: `"cono"`, `"stable_5mm"`,
   `"chatter_15mm"` o `"custom_case"`), cada uno apuntando a un archivo HDF5
   con resultados de simulación dinámica de fresado (desplazamiento y
   velocidad de la herramienta, a veces fuerza de corte).
2. Carga la señal con `HDF5Reader` y recorta la ventana temporal de interés.
3. Construye un `StdSignalData` (interfaz estándar CAMP10, compartida con
   MaxEnt-SPRT, RMS-CV y SSQ) y un diccionario de configuración.
4. Ejecuta el indicador Green Integral en una de dos variantes, controladas
   por el flag `USE_FIXED_WINDOW`:
   - **`Default`** (clustering, cruces por cero, `run_green_integral`):
     agrupa ciclos de fase por ventana usando detección de cruces por cero
     de la velocidad (o de Hilbert), calcula el área de cada ciclo por el
     teorema de Green (fórmula del shoelace) y deriva `delta_n`, la tasa de
     crecimiento logarítmico del área entre ciclos consecutivos. `delta_n<0`
     se interpreta como inestabilidad (chatter).
   - **`FixedWindow`** (sin clustering, `run_fixed_window`): ventanas de
     duración fija, área de la órbita completa por ventana (shoelace
     orientado, con contribución "abierta" C y "cierre" K), y estimación del
     exponente de Lyapunov instantáneo σ̂ a partir de la razón de áreas
     logarítmicas consecutivas (o ajuste lineal local "frozen time").
     Opcionalmente se suaviza con EWMA y se acumula en un indicador Ĝ.
5. En ambas variantes hay un umbral opcional μ±zσ sobre las áreas de una
   región de entrenamiento ("estable"), que produce un tiempo de detección
   `t_d`.
6. Imprime un resumen por consola y genera figuras de diagnóstico
   (`plots_fixed_window` o `plots_green_integral`).

Por defecto `USE_FIXED_WINDOW = True`, así que la ejecución real del script
tal como está configurado usa **únicamente** la ruta `FixedWindow`. La ruta
`Default` (clustering) es alcanzable simplemente cambiando esa constante a
`False`, y por tanto forma parte del flujo real auditado — y contiene el
hallazgo más grave de esta auditoría (ver §3.1).

---

## 2. Análisis por archivo/módulo

Árbol de dependencias real (ver `FLUJO_TECNICO.md` para el diagrama):

### `examples/Green_Integral_Detection_NEW.py`
Script de entrada. Selecciona caso, carga HDF5, arma `StdSignalData` y los
dos diccionarios de configuración (`config_std`, `config_std_fixed`), llama
a `run_green_std`, imprime resultados y llama a `plots_fixed_window` /
`plots_green_integral`. Contiene rutas HDF5 absolutas de Windows
hardcodeadas específicas de la máquina del autor.

### `src/green_integral/__init__.py`
Registra el nivel de logging personalizado `INFO_PLUS_LEVEL` (15) mediante
un monkey-patch de `logging.Logger` y re-exporta la API pública (`types`,
`HDF5Reader`, `run_green_integral`, `run_fixed_window`, `run_green_std`,
funciones de `viz`).

### `src/green_integral/logging_setup.py`
`configure_logging()` — configura el logging raíz una sola vez (o reconfigura
el nivel de los handlers existentes). `LOGGING_LEVELS` mapea nombres a
niveles.

### `src/green_integral/utils/types.py`
Dataclasses de entrada/salida: `SignalData`, `GreenIntegralConfig`,
`GreenIntegralResult`, `FixedWindowConfig`, `FixedWindowResult`,
`StdSignalData`, `IndicatorResult`. Es la fuente de la inconsistencia de
tipos entre variantes (ver §3.1): `GreenIntegralResult.t_d` es
`Optional[float]` (escalar) mientras que `FixedWindowResult.t_d` es
`np.ndarray` (ver también `t_d_no_FAR`, que solo existe en
`FixedWindowResult`).

### `src/green_integral/utils/hdf5_utils.py`
`HDF5Reader`: carga eager de todo el árbol HDF5 en memoria (dict anidado +
arrays NumPy), con navegación por rutas tipo `"grupo/subgrupo/dataset"` y
soporte de slicing/índices tipo NumPy sobre esas rutas. Sin problemas
funcionales detectados; el parser de índices (`_parse_index`) es complejo
pero está bien acotado con manejo de excepciones.

### `src/green_integral/lib/runner_std.py`
`run_green_std`: adaptador a la interfaz estándar CAMP10. Resuelve
`f_cycle`/`N_cycles_per_seg`/`step_cycles` a los parámetros nativos
(`f_modal`, `num_T`, `dt`) del núcleo, arma `SignalData` interno, despacha a
`run_green_integral` o `run_fixed_window` según `config["func"]`, y empaqueta
el resultado en `IndicatorResult`. **Contiene el bug crítico de la
auditoría** (§3.1): la rama `func == "Default"` nunca asigna `t_d_no_FAR`, y
el bloque de logging que sigue asume incondicionalmente que `raw_result`
tiene atributos array (`t_d[0]`, `.t_d.size`, `.t_wins`) que solo existen en
`FixedWindowResult`.

### `src/green_integral/lib/runner.py`
`run_green_integral` — despachador y pipeline de la variante `Default`.
Resuelve la config (merge con `_DEFAULT_PARAMS`), llama a
`process_windows_serial`, agrupa resultados con `build_cycle_groups`, y
opcionalmente calcula el umbral μ±zσ sobre las áreas (`center_area_value` /
`median_area`) de las ventanas de entrenamiento. Sin filtro de piso de
ruido (a diferencia de `runner_fixed.py`, ver §3.2.3).

### `src/green_integral/lib/runner_fixed.py`
El archivo más grande (1099 líneas) y el que ejecuta el script por defecto.
Contiene: geometría del área (shoelace, shoelace orientado, contribuciones
"abiertas"/"cierre"), extracción de ciclos completos por cruces
interpolados de v=0 (`extract_complete_cycles`), estimación de σ̂ (razón de
logaritmos o ajuste "frozen time" con Theil-Sen/`polyfit`), suavizado EWMA,
acumulación Ĝ (total y deslizante), umbral μ±zσ en log10, y un bloque de
diagnóstico muy extenso (activo solo con `debug_level>=2`) que genera
~10 figuras de matplotlib por ventana, incluido un mapa de número de
enrollamiento (winding number) calculado con doble bucle Python 500×500.
Contiene dos bugs (§3.1.3, §3.1.4) y varios riesgos de rendimiento/memoria
(§3.2.2, §3.2.5).

### `src/green_integral/lib/window_processor.py`
`process_windows_serial` / `_process_one_window` — pipeline de ventaneo de
la variante `Default`: filtra (Savitzky–Golay), detecta cruces por cero,
calcula áreas por ciclo vía `Contour_Line_Area`, calcula `delta_n` vía
`compute_delta_n`, arma el diccionario de resultado por ventana. Captura
`Exception` de forma amplia por ventana (§3.2.4) y no protege la llamada al
filtro Savitzky–Golay contra ventanas cortas (§3.1.2 / relacionado).
Importa `matplotlib.pyplot` y `statistics` sin usarlos (§3.3.1).

### `src/green_integral/lib/delta_n.py`
`compute_delta_n` — indicador `delta_n` por ventana: razón de logaritmos
consecutivos de área (modo por defecto) o pendiente Theil–Sen. No valida
que las áreas sean estrictamente positivas en el modo "ratio" (confía en
que el llamador ya filtró): ver §3.2.3.

### `src/green_integral/lib/cycle_groups.py`
`build_cycle_groups` — agrega resultados por ventana en un diccionario
indexado por "cycle key" (agrupación temporal discretizada) y calcula la
mediana global de `delta_n`. Sin problemas funcionales.

### `src/green_integral/lib/diagnostics.py`
Utilidades de diagnóstico de fase (centro lento, trayectoria centrada, fase
local, razón de deriva) usadas solo dentro del bloque de debug de
`runner_fixed.py` (`debug_level>=2`). Código correcto y autocontenido.

### `src/green_integral/utils/zero_crossing.py`
Jerarquía de detectores de cruce por cero: `Simple_ZeroCrossing` (cambio de
signo, usada en la config por defecto del script) y `ZeroCrossing_Hilbert`
(fase de Hilbert, usada solo si `config.hilbert=True`, que no es el caso en
el script de entrada). Contiene un bug numérico confirmado en
`_estimate_f0_autocorr` (§3.1.5), inactivo con la configuración actual
porque `f0_estimada` siempre se pasa explícitamente desde
`window_processor.py`.

### `src/green_integral/utils/cycle_grouper.py`
`CrossingGrouper` — agrupa índices de cruce por proximidad (usado solo por
la variante `Default` cuando `cycles_cluster_points` no es `None`, que es
el caso en `config_std` del script: `35`). Sin problemas funcionales.

### `src/green_integral/utils/contour_area.py`
`Contour_Line_Area` — calcula el área de cada ciclo (variante `Default`) por
el teorema de Green, optimizando el "centro" de la órbita con
`scipy.optimize.minimize` (Nelder–Mead). Contiene el hallazgo de
rendimiento más significativo de la auditoría (§3.2.1: el óptimo tiene
solución analítica trivial, ya calculada como `initial_center`) y un patrón
frágil para fijar la velocidad en los extremos del ciclo a cero (§3.2.6).

### `src/green_integral/utils/signal_filter.py`
`savgol_filter_window` / `moving_average` / `filter_window_signals` —
envoltorios de `scipy.signal.savgol_filter` / `uniform_filter1d`. Sin
protección propia contra señales cortas (la responsabilidad de verificar
`len(signal)>=7` recae en el llamador, y solo uno de los dos llamadores la
implementa — ver §3.1.2 relacionado / §3.2.4).

### `src/green_integral/utils/debug.py`
`DebugManager` — gestiona niveles de debug y ventanas de interés para
graficar. Sin problemas funcionales; API limpia y bien acotada.

### `src/green_integral/viz/plots.py` y `src/green_integral/viz/green_integral_plots.py`
Funciones de graficado (`plot_windows_local`, `plot_windows_duration`,
`plot_indicator_local`, `plots_green_integral`, `plots_fixed_window`,
`plots_signal_diagnostics`). Ambos módulos mutan el estado global de
`matplotlib.rcParams` como efecto secundario de la simple importación
(§3.2.7), con valores distintos entre sí — uno de los dos gana siempre por
orden de import, dejando al otro sin efecto real. Sin bugs funcionales en
la lógica de graficado en sí (multitud de guardas `if x.any()` /
`if valid.sum()>=N` bien puestas).

---

## 3. Hallazgos

### 3.1 Errores (bugs)

#### 3.1.1 — `run_green_std` con `func="Default"` siempre falla (CRÍTICO)
**Ubicación**: `src/green_integral/lib/runner_std.py:240-262` (rama
`Default`, nunca asigna `t_d_no_FAR`) y `runner_std.py:327`
(`t_d_no_FAR=t_d_no_FAR` en el `return`), junto con el bloque de logging
`runner_std.py:301-313`.

**Qué pasa**: en la rama `func == "Default"` (líneas 240-262) la variable
local `t_d_no_FAR` nunca se asigna — solo se asigna en la rama `else`
(`FixedWindow`, línea 277: `t_d_no_FAR = raw_result.t_d_no_FAR`). Sin
embargo, el `return IndicatorResult(...)` final (línea 327) referencia
`t_d_no_FAR` de forma incondicional, sin importar qué rama se ejecutó.

Esto produce uno de dos fallos, dependiendo de si hubo detección:

- Si `t_d` (el de la variante Default, que es un **escalar `float`**, ver
  `runner.py:167` y `types.py:132`) **no es `None`** (es decir, si el
  umbral μ±zσ detectó chatter), se entra al bloque de logging
  (`runner_std.py:281`) y falla antes de llegar siquiera al `return`:
  en la línea 309, `raw_result.t_d[0]` intenta indexar un `float` como si
  fuera un array → `TypeError: 'float' object is not subscriptable`.
  Si ese error no ocurriera, la línea 312
  (`raw_result.t_wins[0]*1000`) fallaría igualmente con
  `AttributeError`, porque `GreenIntegralResult` (la clase de resultado de
  la variante Default, `types.py:116-132`) **no tiene** el atributo
  `t_wins` — ese campo solo existe en `FixedWindowResult`
  (`types.py:265`).
- Si `t_d` es `None` (sin detección, o `use_area_threshold=False`), el
  bloque de logging se salta (guardado por `if t_d is not None...` en la
  línea 281), pero entonces el `return` en la línea 327 falla con
  `UnboundLocalError: local variable 't_d_no_FAR' referenced before
  assignment`.

**Por qué importa**: esto significa que la variante `Default`/clustering
del indicador Green Integral **no se puede ejecutar en absoluto** a través
de la interfaz estándar `run_green_std` — que es precisamente la interfaz
que usa el script de entrada y, por diseño, la que comparte con
MaxEnt-SPRT/RMS-CV/SSQ para comparaciones homogéneas en CAMP10. Solo se
evita hoy porque el script trae `USE_FIXED_WINDOW = True` por defecto
(línea 52 del script de entrada); basta con poner ese flag en `False`
(la propia cabecera del script invita a hacerlo: "Toggle... Set
`USE_FIXED_WINDOW = False` to run the original clustering-based
indicator") para que el script termine en excepción no capturada.

**Causa raíz**: `GreenIntegralResult` y `FixedWindowResult`
(`utils/types.py`) exponen campos con el mismo nombre (`t_d`) pero de
**tipos incompatibles** (escalar vs. array), y solo una de las dos clases
define `t_d_no_FAR` / `t_wins`. `runner_std.py` fue escrito asumiendo la
forma de `FixedWindowResult` para ambas ramas.

#### 3.1.2 — Filtro Savitzky–Golay sin protección de longitud mínima en la variante Default
**Ubicación**: `src/green_integral/lib/window_processor.py:45-46`
(llamada a `filter_window_signals(q_win, q_o_win)` sin condición de
longitud), comparado con la guarda explícita en
`src/green_integral/lib/runner_fixed.py:537`
(`if config.data_filtrated and len(q_win) >= 7:`).

**Qué pasa**: `savgol_filter_window` (`utils/signal_filter.py:22-35`) exige
implícitamente `len(signal) >= 7` (mínimo impuesto por
`compute_window_length`, línea 19: `max(wl, 7)`); si la ventana recibida
tiene menos de 7 muestras, `scipy.signal.savgol_filter` lanza
`ValueError` ("window_length must be less than or equal to the size of
x."). La variante `FixedWindow` protege explícitamente esta llamada con
`len(q_win) >= 7`; la variante `Default`, en `_process_one_window`
(`window_processor.py:45`), llama a `filter_window_signals` de forma
incondicional siempre que `config.data_filtrated` sea `True` (que es el
valor por defecto y el que usa el script de entrada), sin verificar la
longitud de `q_win`/`q_o_win`.

**Por qué importa**: en ventanas muy cortas (posibles cerca de los bordes
de la señal, o con configuraciones de `f_cycle`/`N_cycles_per_seg` que
produzcan ventanas pequeñas respecto al muestreo), esto lanza una
excepción. El daño está parcialmente mitigado porque
`process_windows_serial` envuelve el procesamiento de cada ventana en un
`try/except Exception` (línea 295, ver §3.2.4) que la captura y continúa,
así que el script no se cae — pero la ventana se descarta silenciosamente
y solo queda un `logger.error` como rastro, fácil de pasar por alto.

#### 3.1.3 — `t_d_no_FAR` en `runner_fixed.py` puede fallar según `t_theorical`
**Ubicación**: `src/green_integral/lib/runner_fixed.py:1009`.

```python
t_d_detected_no_FAR_idx = np.where(t_d_detected >= config.t_theorical)[0] if t_d_detected is not None else np.array([], dtype=int)
```

**Qué pasa**: `config.t_theorical` tiene como valor por defecto `None`
(`utils/types.py:240`, documentado explícitamente como "for debug/plots,
not used in detection"). Si un usuario deja `t_theorical` sin especificar
(su valor por defecto) y hay una detección (`t_d_detected` no es `None`,
es decir `use_area_threshold=True` y `training_intervals` provisto), la
comparación `t_d_detected >= None` con un array NumPy de `float` lanza
`TypeError: '>=' not supported between instances of 'numpy.float64' and
'NoneType'`.

**Por qué importa**: contradice directamente el propio docstring del
campo, que afirma que `t_theorical` "no afecta la detección" — en
realidad si se omite y hay detección, rompe la ejecución. El script de
entrada evita el problema porque explícitamente fija
`"t_theorical": _T_GT` (línea 285 del script), pero cualquier otro caso de
uso de `run_fixed_window`/`run_green_std` con `func="FixedWindow"`,
`use_area_threshold=True` y `t_theorical` no especificado (su valor por
defecto documentado) está expuesto a este fallo.

#### 3.1.4 — `Contour_Line_Area.window_duration` no se actualiza si no hay ciclos completos
**Ubicación**: `src/green_integral/utils/contour_area.py:59` (valor por
defecto `[0.0, 0.0]`) y `contour_area.py:184-189` (bucle que solo actualiza
`window_duration` cuando `min(max_iterations, points_cicles_t.shape[0]-1)
>= 1`).

**Qué pasa**: si una ventana produce menos de 2 cruces por cero (0 o 1
ciclo detectable), el bucle de `analyze_contour_area_interpolate` no
itera ni una vez, y `self.window_duration` permanece en `[0.0, 0.0]`
(el valor puesto en `__init__`). `get_window_times()` devolvería entonces
`(0.0, 0.0)` en lugar de los límites reales de la ventana, y en
`window_processor.py:219` la máscara `w_mask = (t >= start_w) & (t <
end_w)` quedaría vacía (`(t>=0.0)&(t<0.0)` nunca es verdad para `t>0`).

**Por qué importa hoy vs. en general**: con la configuración actual del
script (`N_cycles_per_seg=4`, `while_loop_extend=False`), este caso está
en la práctica cubierto por la guarda de `window_processor.py:194-200`
(`if not config.while_loop_extend and len(t_areas) < config.num_T:
continue`), que descarta la ventana **antes** de llegar a usar
`start_w/end_w`. El bug es real pero **latente**: se activaría si algún
día se usa `num_T<=1` o `while_loop_extend=True` con ventanas pobres en
cruces, produciendo silenciosamente arrays de señal vacíos para esa
ventana en el resultado exportado (`window_q_signal`,
`window_q_o_signal`, `window_times` todos `[]`).

#### 3.1.5 — Estimación de frecuencia con doble multiplicación por `dt`
**Ubicación**: `src/green_integral/utils/zero_crossing.py:171-182`
(`_estimate_f0_autocorr`), concretamente la línea 181:

```python
lags = np.arange(len(corr)) * dt          # ya está en segundos
...
T0 = lags[mask][idx_peak] * dt            # BUG: multiplica por dt otra vez
return 1.0 / T0
```

**Qué pasa**: `lags` ya se construye multiplicando por `dt` (línea 178),
por lo que `lags[mask][idx_peak]` ya es un tiempo en segundos (el periodo
estimado por autocorrelación). La línea 181 lo vuelve a multiplicar por
`dt`, lo que produce un "periodo" con unidades de `s²` en vez de `s`
(típicamente un número minúsculo, del orden de `dt≈1e-5..1e-4`), y por
tanto una frecuencia `1/T0` absurdamente alta.

**Por qué importa hoy vs. en general**: es un bug real y reproducible en
aislamiento (bastaría con llamar `ZeroCrossing_Hilbert.calculate_zero_
crossings()` sin pasar `f0_estimada`). Sin embargo, está **inactivo** en
el flujo real auditado: `window_processor.py:66` siempre invoca
`zc.calculate_zero_crossings(f0_estimada=config.f_modal, ...)`, pasando
explícitamente la frecuencia modal, por lo que `_estimate_f0_autocorr`
nunca se ejecuta con la configuración del script de entrada (que además
usa `hilbert=False`, así que ni siquiera se instancia
`ZeroCrossing_Hilbert`). Se reporta por ser parte del árbol de
dependencias real y por el riesgo de reactivarse silenciosamente si en el
futuro se usa `hilbert=True` sin fijar `f_modal`/`f0_estimada`.

### 3.2 Riesgos y código frágil

#### 3.2.1 — Optimización innecesaria del "centro" de cada ciclo (rendimiento)
**Ubicación**: `src/green_integral/utils/contour_area.py:216-225`.

```python
initial_center = [float(np.mean(x_window)), float(np.mean(v_window))]
result = minimize(self._cost_function, initial_center,
                   args=(x_window, v_window), method="Nelder-Mead")
```

El costo minimizado (`_cost_function`, línea 156-164) es
`Σ (x_i-cx)² + (v_i-cv)²`. El mínimo de esta suma de cuadrados tiene
**solución analítica exacta y trivial**: `cx = mean(x)`, `cv = mean(v)` —
exactamente el `initial_center` que ya se calcula en la línea anterior.
Llamar a `scipy.optimize.minimize` con Nelder–Mead (un optimizador
iterativo sin derivadas) para "encontrar" un valor que ya se tiene es
computacionalmente innecesario: se ejecuta **una vez por ciclo, por
ventana**, es decir potencialmente miles de veces por señal completa.
Sustituirlo por la media directa eliminaría esa sobrecarga sin cambiar el
resultado (salvo el ruido numérico propio de la convergencia del
optimizador, que en el mejor caso solo introduce imprecisión adicional
frente a la fórmula cerrada).

#### 3.2.2 — Posible explosión de memoria/tiempo en el bloque de debug de `runner_fixed.py`
**Ubicación**: `src/green_integral/lib/runner_fixed.py:478-866`
(`if _do_debug: ...`), en particular el mapa de número de enrollamiento
(líneas 761-771, doble bucle `500×500` = 250.000 llamadas a
`winding_number_point` por ventana) y la creación de ~10 figuras de
matplotlib por ventana sin ningún `plt.close()` posterior (solo la última,
`fig_multi`/`fig_bar`, llega a `plt.show()`).

**Por qué importa**: con la configuración por defecto del script
(`debug_level: 1` para `config_std_fixed`, línea 283), este bloque nunca
se activa (`is_do_debug` exige `debug_level>=2`). Es, no obstante, una
funcionalidad de primera clase expuesta por `FixedWindowConfig.debug_level`
y documentada en el propio docstring del módulo — cualquier usuario que
suba `debug_level` a 2 para depurar una ventana concreta (uso previsto,
dado que `debug_window_range` permite acotar el rango) generará, por cada
ventana dentro de ese rango, ~10 figuras abiertas simultáneamente en
memoria (sin `plt.close()`) más un cálculo O(250.000) en Python puro —
lento incluso para una sola ventana, y agotador de memoria/handles gráficos
si el rango de debug cubre varias ventanas.

#### 3.2.3 — Sin piso de ruido para las áreas en la variante Default
**Ubicación**: `src/green_integral/lib/delta_n.py:87-96` (modo "ratio":
`np.log(A[1:]) - np.log(A[:-1])` sin filtrar `A<=0`) comparado con
`src/green_integral/lib/runner_fixed.py:900-901`
(`below_floor = ~np.isfinite(areas) | (areas <= config.area_noise_eps);
areas[below_floor] = np.nan`).

**Por qué importa**: la variante `FixedWindow` explícitamente convierte a
`NaN` cualquier área por debajo de `area_noise_eps` antes de estimar σ̂,
evitando que un área nula o negativa (numéricamente posible en ventanas
degeneradas) contamine el resultado con `-inf`/`NaN` no detectados. La
variante `Default` (`compute_delta_n`, modo ratio) no hace ese filtrado:
si `Contour_Line_Area` devolviera un área exactamente `0.0` para algún
ciclo (geométricamente posible si los puntos de un "ciclo" quedan
colineales o degenerados), `np.log(0)` produce `-inf`, que se propaga sin
aviso al `delta_n` de esa ventana (y potencialmente a la mediana global,
`Mediana_delta_n`, interpretada como "más inestable" de lo real). No se ha
confirmado que esto ocurra con los datos de ejemplo actuales, pero no hay
ninguna salvaguarda que lo impida.

#### 3.2.4 — `except Exception` amplio en el bucle de ventanas de la variante Default
**Ubicación**: `src/green_integral/lib/window_processor.py:158-302`
(`try: ... except Exception as exc: logger.error(...); continue`).

Cualquier excepción dentro del procesamiento de una ventana —incluidos
errores de programación reales, no solo casos de borde de datos, como el
`ValueError` de Savitzky–Golay (§3.1.2) o un futuro `IndexError`— se
captura, se registra como `logger.error` y la ventana se descarta
silenciosamente, dejando que el resto del pipeline (mediana de `delta_n`,
umbral μ±zσ, gráficos) siga con menos ventanas de las esperadas sin que
el usuario lo note salvo que revise los logs línea por línea. Preferible
sería capturar excepciones específicas y esperadas (o registrar un
resumen agregado — "N ventanas descartadas de M" ya existe en
`window_processor.py:304-305`, pero no distingue error genuino de caso de
borde esperado).

#### 3.2.5 — Suposición de muestreo uniforme basada solo en las dos primeras muestras
**Ubicación**: `examples/Green_Integral_Detection_NEW.py:213`
(`fs = 1.0 / (t[1] - t[0])`), `src/green_integral/lib/runner_fixed.py:425`
(`dt_sig = float(t[1] - t[0])`), y
`src/green_integral/viz/green_integral_plots.py:651`
(`dt = float(t[1] - t[0])`, en `plots_signal_diagnostics`).

Toda la lógica de tamaño de ventana (`N_win`, `step`) y de conversión
tiempo↔frecuencia deriva de un único intervalo `t[1]-t[0]`, sin verificar
que el resto de la señal esté efectivamente muestreada de forma uniforme.
Si el HDF5 proviene de un integrador de paso variable (posible en
simulaciones dinámicas explícitas), esta suposición silenciosa podría
desalinear ventanas/áreas sin ningún error visible.

#### 3.2.6 — Extremos de velocidad de ciclo fijados vía una columna "casualmente cero"
**Ubicación**: `src/green_integral/utils/contour_area.py:201` y `:205`.

```python
v_window = np.append(v_window, self.points_cicles_t[i + 1, 1])   # línea 201
...
v_window = np.insert(v_window, 0, self.points_cicles_t[i, 1])    # línea 205
```

Estos extremos SÍ deben valer 0 (son cruces por cero de la velocidad, por
construcción del propio algoritmo de detección), pero el código obtiene
ese `0.0` leyendo la columna 1 de `points_cicles_t` — que vale 0 solo
porque así se construyó `crossings = np.column_stack([all_zero_x,
np.zeros_like(all_zero_x)])` en `zero_crossing.py:110`, no porque el
nombre o la intención de esa columna sea "velocidad". Es un acoplamiento
implícito y no documentado entre dos módulos: si en el futuro alguien
cambia qué guarda la columna 1 de `crossing_0_t` (p. ej. para almacenar
algo útil ahí), este código seguiría ejecutando sin error pero produciría
áreas incorrectas de forma silenciosa. Usar un literal `0.0` explícito (o
una constante nombrada) sería más robusto y legible.

#### 3.2.7 — Mutación global de `matplotlib.rcParams` como efecto secundario del import
**Ubicación**: `src/green_integral/viz/plots.py:28-49` (`_configurar_estilo`,
invocada a nivel de módulo en la línea 49) y
`src/green_integral/viz/green_integral_plots.py:30-62`
(`configurar_estilo_global`, invocada a nivel de módulo en la línea 62).

Ambos módulos cambian `matplotlib.rcParams` con solo importarlos, con
**valores distintos** entre sí (p. ej. `font.size: 18` vs. `9`,
`axes.titlesize: 18` vs. `25`). Como `viz/__init__.py` importa primero
`plots.py` y después `green_integral_plots.py`, la configuración de este
último siempre sobrescribe a la del primero — por lo que la llamada a
`_configurar_estilo()` en `plots.py` no tiene ningún efecto observable en
la práctica (código muerto en efecto, aunque no en ejecución). Más
relevante: importar `green_integral` reconfigura el estilo global de
matplotlib para **todo el proceso Python**, lo que puede interferir con
otros indicadores del mismo repositorio (`maxent_sprt`, `rms_cv`,
`ssq_chatter`) si se importan varios en la misma sesión/notebook — un
patrón conocido de "acción a distancia" no deseada.

### 3.3 Mejoras propuestas

#### 3.3.1 — Imports sin usar en `window_processor.py`
**Ubicación**: `src/green_integral/lib/window_processor.py:6`
(`import statistics`) y `:10` (`import matplotlib.pyplot as plt`).
Ninguno de los dos se usa en el archivo (confirmado por búsqueda textual).
Además de ser ruido, la importación de `matplotlib.pyplot` en un módulo
que no grafica nada sugiere (incorrectamente) al lector que este archivo
tiene efectos de graficado.

#### 3.3.2 — Etiquetas de logging duplicadas/engañosas en `run_green_std`
**Ubicación**: `src/green_integral/lib/runner_std.py:288-289`.

```python
logger.info("  %-24s %d", "Area Windows:", params_physical.get("N_cycles_per_seg", "n/a"))
logger.info("  %-24s %d", "Total Windows", params_physical.get("N_cycles_per_seg", "n/a"))
```

Ambas líneas imprimen el mismo valor (`N_cycles_per_seg`, ciclos por
segmento/ventana) bajo dos etiquetas distintas ("Area Windows" y "Total
Windows"), cuando "Total Windows" sugiere naturalmente el número total de
ventanas analizadas en toda la señal (`len(result_std.t)`, ya disponible).
Es coherente con el patrón repetido en `meta` (`runner_std.py:338-339`:
`"N_cycles_per_seg"` y `"Total_window"` con el mismo valor). Confuso para
quien lea los logs esperando dos magnitudes distintas.

#### 3.3.3 — Acceso directo a diccionario en vez de `.get()` (riesgo de `KeyError`)
**Ubicación**: `src/green_integral/lib/runner_std.py:294`
(`if params_physical["use_area_threshold"] == False:`), inconsistente con
el resto de la misma función, que usa sistemáticamente
`params_physical.get(..., "n/a")`. Si un usuario omite
`use_area_threshold` de `params_physical` (no está en
`_RESOLVER_KEYS` y por tanto no tiene valor por defecto forzado en este
punto del código), esta línea lanza `KeyError` en vez de degradar
graciosamente. Con la configuración del script de entrada no ocurre
(siempre se define explícitamente), pero es una inconsistencia de estilo
que además es una trampa para otros llamadores de `run_green_std`.

#### 3.3.4 — Inconsistencia de signo en la normalización `"median"` de `cycle_area_norm`
**Ubicación**: `src/green_integral/lib/runner_fixed.py:868-884`.

En las ramas `"none"` y `"mean"`, `_K_out` conserva el signo de
`K_beta` (`_K_out = K_beta` / `K_beta / _n_cyc`); en la rama `"median"`,
`_K_out` se calcula como la mediana de **valores absolutos**
(`abs(_closure_contribution(...))`). El mismo campo (`trayectory_K` en el
resultado) cambia de convención de signo según el modo de normalización
elegido, lo que puede confundir a quien compare resultados entre modos o
grafique `K_k` esperando un signo consistente (ver
`viz/green_integral_plots.py:336`, donde además se le vuelve a aplicar
`abs()` al graficar, ocultando el problema en ese caso concreto pero no en
el análisis programático de `raw.trayectory_K`).

#### 3.3.5 — Código muerto/comentado dejado en `runner_fixed.py`
**Ubicación**: `src/green_integral/lib/runner_fixed.py:454-464`. Bloque
comentado que duplica (en versión antigua, sin filtrado) la lógica de
extracción de ventana que sigue justo después, sin comentario explicativo
de por qué se conserva. Debería eliminarse o, si tiene valor histórico,
moverse a control de versiones (que ya lo conserva) en vez de al cuerpo
del archivo.

#### 3.3.6 — Unificar el tipo de `t_d`/`t_d_no_FAR` entre variantes
**Ubicación**: `src/green_integral/utils/types.py:132` (`GreenIntegralResult.t_d:
Optional[float]`) vs. `types.py:275-276`
(`FixedWindowResult.t_d: ... ` / `t_d_no_FAR: ...`, ambos
`Optional[np.ndarray]` en la práctica). Esta es la causa raíz del bug
crítico §3.1.1. Se recomienda unificar la forma (por ejemplo, ambas como
array de detecciones, o ambas como "primer tiempo de detección" escalar
más un campo aparte para la lista completa) para que el código que
consume ambos resultados de forma genérica (como `runner_std.py`) no
tenga que hacer suposiciones distintas según la variante.

#### 3.3.7 — Vectorizar `_integrate_G_sliding`
**Ubicación**: `src/green_integral/lib/runner_fixed.py:249-282`. La
integral deslizante recalcula la suma trapezoidal completa desde `i0`
hasta `k` en un bucle Python anidado para cada ventana `k`
(complejidad `O(n × tamaño_memoria)`), en vez de aprovechar que
`_integrate_G` ya calcula la integral acumulada total (`G`) y que
`G_slide[k] ≈ G[k] - G[i0]` (con una pequeña corrección en el extremo
`i0`) se puede obtener por resta con `np.searchsorted`, evitando el bucle
interno por completo.

#### 3.3.8 — Rutas HDF5 y `_BASE` hardcodeados
**Ubicación**: `examples/Green_Integral_Detection_NEW.py:74-96`. Todas las
rutas de los casos de estudio son absolutas y específicas de la máquina
Windows del autor (`D:\Thesis\...`). Es un script de ejemplo, así que es
aceptable como está, pero conviene documentar (o mover a un archivo de
configuración/variable de entorno) para que el ejemplo sea reproducible
en otra máquina sin editar el código fuente del script.

---

## 4. Métricas de la auditoría

- **Pasadas de autorevisión**: 1 pasada completa de autorevisión (Paso 3),
  además de la verificación inicial línea por línea (Paso 2). En la pasada
  de autorevisión se releyeron contra el código real las ubicaciones
  exactas de los 5 hallazgos de tipo "Error" (usando `Grep` dirigido a las
  líneas citadas) y se confirmaron sin cambios de ubicación ni de
  explicación. No aparecieron archivos nuevos en el árbol de dependencias
  durante la pasada, ni hallazgos nuevos que exigieran una segunda vuelta
  completa.
- **Tasa de confirmación de la última pasada**: 100 % de los hallazgos
  revisados en la pasada de autorevisión (todos los 20 hallazgos listados)
  se clasificaron como **Confirmados** (ninguno **Corregido** ni
  **Descartado**) tras releer el código real.
- **Convergencia**: la auditoría convergió en 1 pasada de autorevisión
  (dentro del límite de 5), cumpliendo las tres condiciones de parada:
  100 % de confirmación, cero archivos nuevos y cero hallazgos nuevos en
  esa pasada.
- **Tiempo de ejecución total**: de 12:04:38 a 12:10:49 (hora local del
  sistema, `date`), aproximadamente 6 minutos de trabajo activo de lectura
  y análisis (no incluye el tiempo de redacción de los cuatro archivos de
  salida).
- **Tokens utilizados**: el contexto expuso `<total_tokens>N tokens left</total_tokens>`
  a lo largo de la sesión; el valor pasó de aproximadamente 14 949 601
  tokens restantes (al iniciar el Paso 1) a aproximadamente 14 808 910
  tokens restantes (justo antes de escribir este archivo), es decir,
  un consumo aproximado de **~141 000 tokens** para todo el análisis. No
  se dispone del tamaño total de la ventana de contexto asignada a este
  agente, por lo que no se puede calcular un porcentaje exacto usado sobre
  el total; solo se reporta el consumo absoluto observado.

---

## 5. Resúmenes finales

**Qué hace el código**: implementa el indicador de chatter "Green
Integral" (área de fase desplazamiento–velocidad, teorema de Green) en dos
variantes intercambiables mediante la interfaz estándar CAMP10
(`run_green_std`): una basada en agrupación de ciclos por cruces por cero
con umbral de área μ±zσ (`Default`), y otra de ventana fija que estima un
exponente de Lyapunov instantáneo y lo acumula opcionalmente en el tiempo
(`FixedWindow`). El script de ejemplo carga una señal real desde HDF5,
ejecuta la variante activa (`FixedWindow` por defecto) y grafica los
resultados de diagnóstico.

**Resumen de mejoras**: eliminar el optimizador Nelder–Mead redundante en
`Contour_Line_Area` (tiene solución cerrada trivial); cerrar las figuras de
matplotlib en el bloque de debug de `runner_fixed.py` y acotar el coste del
mapa de winding number; añadir un piso de ruido a las áreas también en la
variante `Default`; unificar el tipo de `t_d`/`t_d_no_FAR` entre las dos
variantes de resultado; limpiar imports sin usar y código muerto/comentado;
evitar que dos módulos de `viz` compitan por configurar
`matplotlib.rcParams` globalmente al importarse; sustituir el
`except Exception` amplio del bucle de ventanas por manejo de errores más
específico; vectorizar `_integrate_G_sliding`.

**Resumen de lo que está mal**: la variante `Default` del indicador es
**inejecutable** a través de la interfaz estándar `run_green_std`
(`UnboundLocalError` o `TypeError`/`AttributeError` según haya detección o
no) — el único motivo por el que el script de ejemplo funciona hoy es que
usa la variante `FixedWindow` por defecto. Además, la variante `Default`
puede lanzar `ValueError` en el filtro Savitzky–Golay para ventanas cortas
(sin protección, a diferencia de `FixedWindow`), la variante `FixedWindow`
puede fallar si se deja `t_theorical` en su valor por defecto documentado
(`None`) y hay detección, y hay un bug numérico confirmado (aunque
inactivo con la configuración actual) en la estimación de frecuencia por
autocorrelación de `ZeroCrossing_Hilbert`.
