# PLAN — rampas de Ap en la app de experimentos

## §0. Para retomar (leer primero)

- **Estado (2026-10-05):** **F0–F8 implementados**, un commit por fase (`3d0c314`..; ver `git log ddfe63d..`), sin push,
  selftests y `check_app_dialogs.py` en verde. E2E real con el experimento `ramp_check` (ver la bitácora al final).
  Pendiente para el usuario: (1) decidir el valor de `early_tol_s` (0.5 s convierte en "alarma temprana" las
  detecciones durante el crecimiento del chatter, ver F5 en la bitácora); (2) re-correr Label template/build del cono
  y Validate de `DOE_Test_1DOF_150_n12000` (quedan en naranja a propósito); (3) copiar la `db_def` de dos sentidos a
  otras carpetas de caso si quiere rampas decrecientes allí. Informe final de la noche: `INFORME_ramps_2026-10-05.md`.
- **Python:** `D:/Thesis/03-Code_Storage/02-Altintlas_Nessy2m_Storage/Env/entorno_CAMP10/Scripts/python.exe`.
- **Restricciones:** no correr indicadores de verdad salvo el E2E corto acordado (F0 / Verificación 3); no tocar los
  experimentos del usuario ni sus borrados sin commitear; sin ramas ni worktrees nuevos; `check_app_dialogs.py` en
  segundo plano (> 2 min); los scripts de parche se escriben con la herramienta Write (heredocs rompen `\`).
- **Decisiones cerradas** (no volver a preguntar): verdad de rampa por ventanas como los indicadores (sin envolventes,
  alternancia tal cual, `t_onset` = 1ª ventana inestable); **ningún tiempo teórico de cruce κ=1** en ningún sitio;
  tolerancia `early_tol_s`; alarma temprana en un inestable = FA y cuenta como **FN**; onset de constantes sigue por
  muestra (`t_onset_amp`); `delay_start_s` igual; sin excluir transitorio; métricas globales/ranking/AUC sin las
  rampas que cruzan, métricas `ramp_*` aparte; mezcla constante + rampa por fila; entrenamiento siempre constante.


## Contexto

Hoy cada caso del DOE tiene **una profundidad**: la app rechaza `Ap_end ≠ Ap_start`, κ es un escalar por caso, la
etiqueta es una clase por caso y la validación cuenta un acierto o fallo por caso. El usuario quiere **validar
indicadores con casos en rampa de Ap** (la profundidad crece o decrece durante el corte, con `n` fija). El
**entrenamiento sigue siendo de Ap constante**. Todo el flujo debe funcionar con rampas: crear, simular, extraer,
etiquetar, indicadores, validar contra un entrenamiento, visores y herramientas (SLD, planner…), sin cambiar el
comportamiento actual de los casos constantes.

Decisiones del usuario (2026-10-04):
1. **Verdad de una rampa = la regla de amplitud de siempre, ventana a ventana** (sin envolventes): la señal se
   recorre con ventanas **configuradas como las de un indicador** (`mode` by_revolution / by_modal, `N_*_window`,
   `step_*`, resueltas por caso con T_rev = 60/n o T_modal); en cada ventana `max|y|` frente a `lim_inf` / `lim_sup`
   da estable / gray / inestable; ventanas seguidas iguales forman los intervalos. Una sola configuración por
   experimento (heredada del entrenamiento); por defecto, la ventana de las variantes del experimento (la más larga
   si difieren), escrita explícita en el YAML. Si las ventanas alternan cerca del cruce se dejan tal cual;
   `t_onset` = inicio de la primera ventana inestable. **No** se usa ningún tiempo teórico de cruce (κ(t)=1 del
   SLD): ni se calcula, ni se guarda, ni se muestra, ni entra en métricas.
2. **Detección antes del cruce**: ventana de tolerancia configurable (`early_tol_s`). Si la detección cae dentro de
   ella es acierto anticipado; antes es falsa alarma. Métricas: acierto por caso y retraso con signo respecto al cruce.
3. **Mezcla por caso**: un experimento puede tener filas constantes y filas en rampa.
4. **Simulación real corta permitida**: 1-2 casos de rampa para verificar de punta a punta.

## Lo que ya existe y se reutiliza

- **Nessy2m** ya hace rampas: en `in/mdl_data.py` del caso, `R_ext_1 = R_int + Ap_start` y `R_ext_2 = R_int + Ap_end`
  sobre `L_cylindre = 150 mm` con avance constante (`Vf = n·f_tooth`). A 12098 rpm y 0.05 mm/diente, 150 mm son
  unos 15 s, igual que la duración de la señal. Hipótesis: **Ap(t) lineal en el tiempo de corte** (se verifica en F0).
- **`doe_runner` extract** ya escribe `kappa_start` / `kappa_end` cuando `Ap_start ≠ Ap_end`
  (`DOE_simulacion/doe_runner.py`, extract_doe_results).
- **Etiquetas por intervalos**: el YAML de etiquetas, `reference_dataset.py build` (`sig.intervals`) y
  `validate_indicators.read_intervals` / `truth_windows` / `case_truth` (que ya devuelve `mixed`) aceptan varios
  intervalos por caso. Solo faltan una estrategia que los genere y la regla de puntuación de un caso mixto.
- **`validate_indicators`**: ya calcula `amplitude_onset`, `delay_onset_s`, `delay_start_s`, ventana a ventana.
- **SLD**: `sld_model.plot_sld` / `case_points` ya dibujan una rampa como segmento vertical (lo usan el visor y
  `doe_planner`); `doe_planner.zone()` ya clasifica "cruza el límite" y `kappa_txt()` muestra "κ inicio → κ fin".
- **App**: `SimulationFrame`, `SldPicker` (clic, propuesta, aceptar, zoom), `case_rows`, `check_simulation`,
  `simulation_of_run`, `inspect_h5` / `add_case_attrs`, `link_truth`, `kappa_overlap` en `DOE_utils/experiment.py` y
  `DOE_utils/launcher.py`. Se amplían, no se sustituyen.

## Definiciones comunes (un solo sitio: `experiment.py`)

- **Caso en rampa**: `|Ap_end − Ap_start| > 1e-9 m`. Constante en otro caso (todo igual que hoy).
- **Ap(t)** = `Ap_start + (Ap_end − Ap_start)·(t − t0)/(t1 − t0)` sobre el intervalo de la señal `[t0, t1]` (hipótesis
  lineal de F0, función `ap_of_t(attrs, t)`).
- Atributos nuevos por caso (escritos por la app, sin tocar Nessy2m): `ramp` (bool), `kappa_start`, `kappa_end` (ya de
  extract), y tras el etiquetado `t_onset` (cruce de la verdad) y `true_label = "mixed"` cuando hay
  estable e inestable.

## Fases (un commit por fase, selftests en verde antes de pasar a la siguiente)

### F0. Verificar la física de la rampa (datos reales)
- **Primero con lo que ya existe**: `Data/1DOF_150_Cone_New/Cono_dexel_20e-5_dt_200/Cono_dexel_20e-5_dt_200_5mm_15mm`
  (rampa 5 → 15 mm, κ 0.58 → 1.74, con `doe_results.h5` y resultados de indicadores; solo lectura). Comprobar la
  duración de la señal, el crecimiento del `max|Axial_disp|` por ventana y de la fuerza `res_R_p` (proporcional a Ap).
- Después, una simulación real corta (`ramp_check`, 1-2 casos, ~1 h): una rampa **decreciente** (12 → 5 mm), que no
  existe en los datos, y una creciente como control.
- Resultado: confirmar o corregir `ap_of_t` (lineal sobre la señal, o sobre `L/Vf` del caso) y anotarlo en la bitácora.
- Hallazgo ya visto en el cono: su `doe_results.h5` tiene a la vez `kappa` (= κ de inicio) y `kappa_start` /
  `kappa_end`. En una rampa la app debe ignorar `kappa` (ver F1).

### F1. Núcleo de datos y simulación (`experiment.py`)
- `build_simulation(..., depths_end=None)`: lista de fin por fila (en Ap mm o κ; vacía = constante). Con κ, el fin
  se convierte con el mismo `ap_ref` a la `n` de la fila.
- `simulation_form`: ya no rechaza rampas; devuelve `depths` y `depths_end`.
- `case_rows`: `kappa_start`, `kappa_end`, `ramp`, `kappa` = None en las rampas.
- En todo lo que lee κ de un `.h5` (`h5_info`, `_label_cases`, visor, `validate_indicators`, `doe_val_planner`): si el
  caso es rampa (`Ap_start ≠ Ap_end`), se usan `kappa_start` / `kappa_end` y se **ignora** `kappa` aunque exista.
- `sim_problems`: comprueba también `Ap_end` (finito, > 0, unidades); aviso informativo si una rampa no cruza κ = 1.
- `h5_info` / `summary` / tarjeta y lista: rango de κ con las rampas (mínimo de inicios, máximo de fines) y
  "rampas k de N".
- `dry_run`, `check_simulation`, vista previa de Import, resumen de Extract: "Ap a→b mm, κ a→b".
- `kappa_overlap`: las rampas no se comparan punto a punto (cubren un rango); nota "ramp cases: no repeat check".
- `inspect_h5` / Standardize (`add_case_attrs`): en una rampa pide y calcula `kappa_start` y `kappa_end`, no `kappa`.
- `simulation_from_folder` / `simulation_of_run`: ya leen `$Ap_end$`, solo se verifica con rampas.

### F2. Interfaz de simulación (`launcher.py`)
- `SimulationFrame`: campo **"depths end"** junto a "depths" (vacío = constante; por fila = rampa; misma longitud).
  La vista previa añade columnas `Ap fin`, `κ fin` y el tipo (const / ramp).
- `SldPicker`: modo **"ramps"** (botón de opción junto a "kappa / Ap"):
  - dos clics = una rampa (inicio, fin), dibujada como segmento vertical (con flecha según la dirección);
  - clic derecho quita la rampa más cercana; la lista muestra "Ap a→b, κ a→b", con los grupos
    [proposed] / [accepted] / [yours] como hoy;
  - **Propose** en modo rampa: N rampas con el inicio estratificado en [desde, hasta] y un "span" fijo
    (campo nuevo, en κ o mm), con azar y semilla como ahora; sin margen respecto al entrenamiento (una rampa cubre
    un rango);
  - **Use these Ap** rellena "depths" y "depths end".
- El zoom se conserva igual que ahora; las leyendas distinguen puntos y rampas.

### F3. Etiquetado (`DOE_simulacion/reference_dataset.py`)
- `_label_by_amplitude`: casos constantes **sin cambios** (una ventana = la señal entera). En una rampa, la misma
  regla **ventana a ventana**: ventanas de `window_N` × T (T = T_rev = 60/n del caso, o T_modal = 1/f_modal) con paso
  `window_step` × T, desde `t0 + warmup`; en cada una `max|y|` del canal de etiquetado frente a `lim_inf` / `lim_sup`
  → estable / gray / inestable; ventanas seguidas iguales se unen en un intervalo `[inicio 1ª ventana, fin última]`
  (los solapes por el paso se resuelven asignando a cada instante la etiqueta de la ventana que empieza en él). Sin
  suavizado ni fusión: si alternan, quedan tal cual. Funciona en rampas crecientes y decrecientes.
- Parámetros nuevos en `labeling_*` (y en `LABEL_PARAMS` / `LabelForm` de la app), con los mismos nombres y la misma
  resolución que en los indicadores: `window_mode` (by_revolution / by_modal), `window_N`, `window_step` (+ `f_modal`
  si by_modal). Por defecto, al crear el experimento o activar el etiquetado, se escriben los de las variantes del
  experimento (la ventana más larga si difieren; hoy by_revolution, 7, paso 1). Con referencia, se heredan del
  entrenamiento como el resto.
- `resolve_physics` / `indicator_config` de `experiment.py` se reutilizan para pasar de vueltas a segundos por caso.
- `build`: cada pieza guarda su `Ap_start_mm` / `Ap_end_mm` (Ap en t0 y t1, con `ap_of_t`).
- App: `_label_cases` / `_yaml_labels` pasan a dar por caso `stable` / `gray` / `unstable` / `mixed` + `t_onset`. El
  resumen de Label template y Label build cuenta las rampas aparte ("ramps: crosses at t ≈ …") y la frontera
  "último estable / primer inestable" solo usa los casos constantes.

### F4. Indicadores (`DOE_analisis/doe_indicators.py`)
- El cálculo no cambia (ya trabaja sobre la señal en el tiempo).
- Resumen por caso: en una rampa muestra el cruce de la verdad, la detección y el retraso con signo, y aplica la
  misma regla de tolerancia que Validate (`OK` / `ANTICIPATED` / `MAL: false alarm` / `MAL: missed`).
- `write_results` y `link_truth`: `true_label = mixed`, `t_onset`, `kappa_start` / `kappa_end`.

### F5. Validación (`DOE_analisis/validate_indicators.py`)
**Revisión de las métricas de los casos constantes (antes de heredarlas).** Comprobado con datos reales
(`DOE_Test_1DOF_150_n12000` y el cono): `t_d` es el estado de alarma ventana a ventana en las 4 variantes (97-100 %
de detecciones consecutivas, paso de 1 vuelta), así que `alarm_fraction` y `persistence` son correctas; Wilson,
Hanley-McNeil y MCC también. Lo que se corrige:
- **Alarma temprana ya no es acierto** (decisión del usuario). Hoy un caso inestable es TP con *cualquier* alarma; en
  los datos, RMS-CV alarma a los 0.07 s en todos los casos y eso cuenta como TP (infla la TPR). Regla nueva, **la
  misma para constantes y rampas**, con la **primera detección** `t_det` del caso:
  - `t_det ≥ t_onset` → **TP**;
  - `t_onset − early_tol ≤ t_det < t_onset` → **TP anticipado** (acierto, retraso negativo);
  - `t_det < t_onset − early_tol` → **FA** (falsa alarma temprana);
  - sin detección → **FN**.
  - En la matriz 2×2, **FA cuenta como FN** (decisión del usuario: el positivo no se detectó a tiempo). TPR = TP /
    todos los inestables; TNR solo con estables, como hoy. Además se informa `early_alarm_rate` (FA / inestables).
  - Casos estables: igual que hoy (FP con cualquier alarma, TN sin alarmas).
- **Onset de los constantes: se queda como hoy** (decisión del usuario): `t_onset_amp` = primera **muestra** con
  `|y|` sobre `lim_sup`. En las rampas, `t_onset` = inicio de la primera **ventana** inestable del etiquetado.
- `delay_start_s` se mantiene como hoy.
- **Sin exclusión del transitorio de entrada** (decisión del usuario): las alarmas del arranque cuentan.
- El selftest de `validate_indicators` se amplía con un caso inestable de alarma temprana (FA → FN), uno anticipado
  dentro de la tolerancia (TP) y uno tardío (TP con retraso positivo).

**Validación de rampas (hereda la regla anterior):**
- Parámetro **`early_tol_s`**: sección `validate` del experimento (`ValidateForm`) y CLI `--early-tol` (por defecto
  0.5 s, editable). Lo usan constantes y rampas.
- Rampas que cruzan (`mixed`: estable e inestable): la misma regla, con `t_onset` de la ventana (ningún tiempo
  teórico). Rampa decreciente (inestable → estable): `t_onset` = inicio de la señal, se puntúa como inestable.
- Una rampa que no cruza (entera estable o inestable) se puntúa como un caso constante (grupo global).
- **Métricas globales, ranking y ROC/AUC: solo casos constantes y rampas que no cruzan** (mismo significado que hoy;
  decisión del usuario: las rampas que cruzan no entran).
- **Métricas de rampa aparte** (por variante, en `/metrics/<run>` con prefijo `ramp_`; la misma contabilidad: FA es
  un fallo del positivo): `ramp_n`,
  `ramp_detection_rate` = (TP + anticipados)/n, `ramp_anticipated_rate`, `ramp_early_alarm_rate`, `ramp_miss_rate`
  (las tasas con intervalo de Wilson), `ramp_median_delay_s` con signo + cuartiles (p25, p75),
  `ramp_alarm_fraction_stable` (ventanas con alarma en el tramo estable de las rampas) y `ramp_persistence` (como hoy,
  en el tramo inestable). Gray no cuenta en ningún grupo.
- Resumen por caso con `t_onset`, `Ap_start`, `Ap_end`, `kappa_start`, `kappa_end`, resultado y retraso.
- App: el resumen de Validate muestra las métricas de rampa si las hay; la pestaña **Compare** añade esas columnas.

### F6. Visores (`DOE_plots/doe_unified_selector.py`, `doe_indicator_plotter.py`)
- Tablas: columnas `kappa_start`, `kappa_end`, `t_onset`; las rampas se ordenan por κ de inicio.
- Señal e I(t) de un caso en rampa: línea vertical en `t_onset` (verdad) y sombreado de los intervalos
  estable / gray / inestable.
- Visor de datasets: las piezas de una rampa muestran su κ y su Ap en los extremos.
- La banda superior de "qué es este archivo" indica "N ramp cases".

### F7. Etapas opcionales y herramientas externas
- `static_deflection.py`: en una rampa, deflexión con Ap(t) muestra a muestra (hoy toma `Ap_start` para todo el caso).
- `doe_noise.py`, `doe_model_snr.py`: genéricos; se verifica que una rampa como caso de control no falla.
- `doe_planner.py`: ya dibuja segmentos y clasifica "cruza el límite"; solo se verifica.
- `doe_val_planner.py`: lee un entrenamiento constante y genera casos constantes; se mantiene así. En la app, el
  botón del planificador de validación lo dice ("constant cases"); las rampas se proponen con el SLD.

### F8. Documentación
- `TUTORIAL.md`: sección "Rampas de Ap" (crear, SLD en modo rampa, qué es la verdad, tolerancia, métricas, visor).
- `HELP` de la app, `STAGE_INFO`, y la bitácora en `PLAN_app_v2.md` (ronda "rampas").

## Matriz de uso (todo lo que debe funcionar con rampas)

| Uso | Dónde | Cómo se prueba |
|---|---|---|
| Crear con Ap mm, con κ (`model_at_spin`), una sola rampa, mezcla constante + rampa | New experiment | check_app_dialogs |
| SLD: rampas por clics, quitar, Propose / Accept, zoom, Use | SldPicker | check_app_dialogs |
| Load values / Copy / Settings / Edit config de una rampa | formularios | check_app_dialogs |
| Vista previa, Dry-run y errores (Ap_end inf, unidades) | experiment.py | selftest |
| Import (con .h5 y sin extraer), Standardize de un .h5 con rampas | Import / Standardize | selftest + check_app_dialogs |
| Run to goal: validación en rampa contra entrenamiento constante | experiment.chain | selftest (run_stage simulado) |
| Label template / build en intervalos, resumen | reference_dataset + app | selftest reference_dataset |
| Resumen por caso de Indicators, link_truth | doe_indicators | --selftest |
| Validate con tolerancia, métricas de rampa, Compare | validate_indicators + app | selftest validate_indicators |
| Visores con rampas | doe_unified_selector | funciones de carga (sin ventana) |
| Static deflection, Noise, Model SNR con rampas | scripts opcionales | sus selftests |
| De punta a punta con datos reales | F0 + E2E | ver Verificación |

## Verificación

1. **Selftests** (Python de `entorno_CAMP10`): `experiment.py selftest`, `launcher.py --selftest`,
   `DOE_simulacion/reference_dataset.py selftest`, `DOE_analisis/validate_indicators.py --selftest`,
   `DOE_analisis/doe_indicators.py --selftest`, `DOE_simulacion/static_deflection.py --selftest`; cada uno con casos de
   rampa sintéticos (amplitud que crece, que decrece, que no cruza y que alterna cerca del umbral) además de los
   actuales; los casos constantes deben dar exactamente las mismas etiquetas que hoy.
2. **`check_app_dialogs.py`**: sección nueva "ramps" sobre carpetas temporales sintéticas, con capturas
   (`%TEMP%/app_dialog_shots`). Se corre en segundo plano porque tarda más de 2 min.
3. **Real (F0 y E2E)**: `ramp_check` (2 rampas, ~1 h de simulación) → Extract → Label template / build (revisión de
   los intervalos) → Indicators sobre esos 2 casos con las variantes del entrenamiento (~10 min) → Validate contra el
   entrenamiento constante `DOE_Training_Tube_dxl_20e-5_RUN_10_0.5-2.0` → visor. Comprobar t_onset, el
   resultado por variante y las métricas de rampa.
4. Los experimentos constantes del usuario no deben cambiar de estado: `experiment.py status` antes y después.

## Riesgos y límites

- **Las validaciones constantes ya hechas cambian.** La regla de la alarma temprana (FA = FN) cambia TPR, exactitud
  balanceada, MCC y ranking. Validate queda en naranja (cambia su configuración: `early_tol_s`) y se vuelve a correr
  en segundos, sin tocar los indicadores. Se anota en la bitácora con las métricas antes y después en
  `DOE_Test_1DOF_150_n12000`.

- Ap(t) lineal es una hipótesis hasta F0 (entrada o salida de herramienta, o rampa definida en la longitud y no en
  el tiempo, cambiarían `ap_of_t`).
- La verdad de una rampa depende de la ventana de etiquetado; cerca del umbral puede alternar (se deja tal cual y se
  revisa en el YAML de Label template).
- `n` variable sigue fuera de alcance; el diseño no lo impide (Ap(t) y T_rev pasarían a depender de n(t)).

## Bitácora de la implementación

### F0 (2026-10-04)
- **Cono 5 → 15 mm** (solo lectura): la señal dura 14.888 s y L/Vf = 150 mm / 10.082 mm/s = 14.878 s; la fuerza
  media |res_R_p| por ventana crece en línea recta (≈ 33.6 N/s) y F/Ap = 50.2 N/mm a 6 mm y 50.1 N/mm a 14 mm.
  **Ap(t) es lineal sobre el intervalo de la señal**: `ap_of_t` queda como en el plan.
  `max|Axial_disp|` por ventana: < 1.5e-6 m hasta ~9 s, primera ventana de 7 vueltas sobre lim_sup (2e-5 m) en ~10.7 s.
- **Rampas decrecientes: la plantilla del caso no las simula.** `in/1DOF_150Hz-db_def.py` (`h_max_truncado`) solo
  construye un cono que hace CRECER Ap; con Ap_end < Ap_start la pieza queda como un tubo constante de Ap_end (y la
  alfombra de dexels se dimensiona con R_ext_2). Decisión: no se tocan las carpetas de caso del usuario; se creó
  `Data/1DOF_150_Ramp_check/1DOF_150Hz` (copia + `h_min_ramp` / `h_max_ramp`, cono en h_min cuando Ap baja, alfombra
  con `R_ext_max`, marca `# ramps: both directions`). Comprobado sin simular: la extensión radial en cada altura de
  la herramienta da Ap(w) exacto en los dos sentidos y la rampa creciente es idéntica a la plantilla original.
  La app (`sim_problems`) da ERROR a una rampa decreciente sobre un caso cuya db_def no tiene la marca.
- **Simulación real `ramp_check`** (experimento nuevo, `Data/1DOF_150_Ramp_check/ramp_check`): caso 0 = 15 → 5 mm
  (decreciente), caso 1 = 5 → 15 mm (control, igual al cono existente: comprueba también la nueva db_def).

### F5 — la regla de la alarma temprana en `DOE_Test_1DOF_150_n12000` (recalculado a un archivo temporal; el del usuario no se tocó)

| variante | antes (cualquier alarma = TP): bal.acc / MCC | regla nueva, early_tol 0.5 s | early_tol 5 s |
|---|---|---|---|
| green_fixed_revo_dec7_1step | 0.94 / 0.90 (TP 11, FN 0) | 0.48 / -0.05 (TP 1, FN 10: 10 alarmas tempranas) | 0.94 / 0.90 |
| ssq_revo_aux4_n_aux4_dec7_1step | 0.94 / 0.90 | 0.48 / -0.05 (10 alarmas tempranas) | 0.94 / 0.90 |
| maxent_revo_dec7_1step | 0.88 / 0.80 | 0.38 / -0.40 (11 alarmas tempranas) | 0.69 / 0.38 |
| rms_cv_revo_aux4_n_aux4_dec7_1step | 0.56 / 0.28 | 0.06 / -0.90 (11 alarmas tempranas) | 0.34 / -0.35 |

**Lectura (importante, para decidir el valor por defecto de early_tol_s):** RMS-CV alarma a 0.075 s en todos los casos
(el transitorio de entrada: alarma falsa de verdad). Pero Green, SST y MaxEnt detectan el chatter **mientras crece**,
mucho antes de que la amplitud llegue al 40 % del avance: p. ej. κ = 1.08 → Green detecta a 7.2 s y la amplitud cruza
el límite a 11.9 s; en todos los casos inestables t_det ≈ 0.6 · t_onset_amp (Green / SST) y ≈ 0.4 · t_onset_amp
(MaxEnt). Con 0.5 s esas detecciones pasan a "alarma temprana" y cuentan como FN. Con 5 s Green y SST vuelven a
0.94, MaxEnt queda penalizado (detecta hasta 7 s antes cerca de κ = 1) y RMS-CV sigue penalizado por el transitorio.
El valor por defecto se dejó en 0.5 s (decisión del plan); se cambia por experimento en Validate > Edit config.
Las validaciones ya hechas quedan en naranja (early_tol_s entra en la huella de Validate) y se rehacen en segundos.

### E2E real (2026-10-05) — experimento `ramp_check` (`Data/1DOF_150_Ramp_check/ramp_check`)

- **Simulate** (66 min por caso, 2 en paralelo) → **Extract** → **Label template/build** → **Indicators** (4 variantes,
  6 min) → **Validate** contra el entrenamiento constante: todo en verde desde la app (`experiment.py run`).
- **Física (cierra F0):** rampa decreciente 15 → 5 mm: F/Ap = 50.0 N/mm constante mientras Ap baja (la `db_def` de dos
  sentidos funciona); el control 5 → 15 mm es **idéntico bit a bit** al cono simulado antes (misma plantilla).
- **Verdad por ventanas (7 vueltas, paso 1):** 5 → 15: estable hasta 9.99 s, gray, **inestable desde 10.46 s**
  (Ap ≈ 12 mm, κ ≈ 1.40). 15 → 5: estable 0.05–1.04 s (el chatter aún no creció aunque κ = 1.74), gray, **inestable
  1.33–11.04 s**, gray, estable desde 12.15 s.
- **Indicadores (primera detección / retraso al cruce):** 5 → 15: Green 9.27 s (−1.20), SST 9.22 s (−1.25),
  MaxEnt 8.51 s (−1.95), RMS-CV 0.075 s (transitorio). 15 → 5: Green 0.71 s (−0.63), SST 0.69 s (−0.65), MaxEnt
  0.43 s (−0.90), RMS-CV 0.075 s. Con `early_tol_s` = 0.5 s todas son alarma temprana (ramp_early_alarm_rate 1.00);
  con 2 s, Green / SST / MaxEnt detectan 2/2 anticipadas (retraso mediano −0.91 / −0.95 / −1.43 s) y RMS-CV 1/2.
- Errores encontrados en el E2E y arreglados: el visor elegía `t_onset` como variable de color (las rampas nuevas no
  tienen `kappa` suelto); el panel de Validate imprimía un ranking vacío sin casos constantes.
- Aviso previo, no causado por esto: `s/gen_tool.py` de la plantilla falla (`gent_straight_insert() ... id_node_dyn`)
  en todas las simulaciones (también en `DOE_Test_1DOF150_n5189`); Nessy2m usa la herramienta ya generada en `tool/`.
