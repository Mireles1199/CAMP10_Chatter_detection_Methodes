# PLAN — Validación con ruido blanco (Ap constante)

**Estado (2026-10-05):** plan escrito, nada implementado. Rama de los scripts: `wt-validacion`. Rama de la interfaz: `wt-interfaz`. Coordina: sesión `DOE_utils/PLAN_app_v2.md` (manager).
**Cómo retomar:** leer §0 y la tabla de fases (§9); la última fase marcada ✅ es donde se quedó.

Estilo de trabajo: **ponytail** (reusar lo que ya existe, el cambio más corto que funcione, un selftest por lógica nueva) y figuras con **article-plot-style** (`DOE_plots/plot_style.py`: `FIGSIZE_SIMPLE`/`FIGSIZE_WIDE` × `FIGSCALE`, `figsize_grid`, `lang_text`, paleta Okabe-Ito, `fig._keep_size`).

---

## 0. Resumen en 6 líneas

1. Hoy existen las etapas `noise` (ruido a **un** caso de control) y `noise_indicators` (indicadores sobre ese ruido), pero **no hay validación después**.
2. Se amplía `noise` para poner ruido a **varios casos de validación**, con **SNR absoluto**, **varios niveles** y **varias realizaciones**.
3. `noise_indicators` sigue igual, pero **sin copiar las señales** (tamaño).
4. Etapa **nueva** `noise_validate`: puntúa cada copia ruidosa contra la verdad del caso **limpio**, tomada de la validación limpia ya hecha.
5. Métricas por nivel de SNR y realización, resumidas en media / mín / máx entre realizaciones, más la referencia limpia.
6. Tres figuras nuevas (métricas contra SNR, matriz caso × SNR, anticipación contra SNR) en el mismo estilo y la misma API que las de validación.

---

## 1. Contexto y objetivo

La validación con Ap constante ya está cerrada (`validate_indicators.py`, `validation_figures.py`, `GUIA_metricas_validacion.md`). La pregunta nueva es de **robustez**: *¿un indicador calibrado con datos limpios sigue funcionando cuando la señal tiene ruido de sensor?*

- **Entrenamiento limpio, validación con ruido.** Los umbrales (μ + 3σ, etc.) se fijan con el dataset de entrenamiento limpio (como hoy en `noise_indicators`). Solo los DOE de validación reciben ruido.
- **Verdad limpia.** La etiqueta de cada copia ruidosa es la del caso limpio del que viene: la verdad no cambia con el ruido, solo se mide el indicador.
- **Se reutilizan los experimentos existentes**: se añaden etapas a su YAML, no se crean experimentos nuevos.

## 2. Decisiones tomadas (con el usuario)

| # | Decisión | Valor |
|---|---|---|
| 1 | Casos con ruido | Los de validación del experimento. Primera pasada: una lista de ~12 casos representativos (4 estables, 2 frontera, 6 inestables); después `all`. |
| 2 | Definición del SNR | **Absoluto**: `sigma = sqrt(P_ref / 10^(SNR/10))`, un mismo sigma para todos los casos. `P_ref` = varianza de la señal del **caso inestable más débil** (menor kappa etiquetado inestable), por señal. Opción: `snr_ref_case` explícito. |
| 3 | Niveles | Lista explícita, primera pasada **80, 60, 40, 30, 20, 10 dB**. El rango antiguo (`SNR_RANGE`, 5–200 dB) **desaparece** (se acepta y se ignora con aviso). |
| 4 | Realizaciones | **3** en la primera pasada (semillas independientes). Métricas por realización y resumen media / mín / máx entre ellas (no se mezclan como casos). Dentro de una realización, el mismo ruido unitario reescalado en todos los niveles (curvas más limpias). |
| 5 | Etiqueta | La del caso limpio. Entrenamiento limpio. |
| 6 | Señales con ruido | `Axial_disp` y `Axial_vel`, ruido independiente en cada una (**simplificación declarada**; opción futura: derivar la velocidad ruidosa del desplazamiento). |
| 7 | Grises | Mismo modo que la validación limpia (`gray_mode` del archivo limpio). |
| 8 | Tamaño | No copiar señales en los resultados de indicadores y de validación con ruido (§8). |

Referencia en n12000 (medida): caso de referencia kappa 1.082, RMS desplazamiento 1.28e-5 m, velocidad 1.2e-2 m/s → σ_disp = 1.3e-8 m a 60 dB, 1.3e-6 m a 20 dB. Los casos estables vibran 1.3e-8 – 2.4e-8 m (RMS): desde ~60 dB el ruido ya iguala su vibración.

## 3. Flujo

```
doe_results.h5 (limpio) ──label_build──> reference_dataset_amp.h5 ──┐
        │                                                            │
        ├──indicators──> doe_indicator_results.h5 ──validate──> doe_validation_results.h5 (limpio; verdad + t_onset + gray_mode)
        │                                                                                     │
        └──noise──> doe_noise_results.h5 ──noise_indicators──> doe_noise_indicator_results.h5 ┴──noise_validate──> doe_noise_validation_results.h5
             (copias ruidosas)          (umbrales del entrenamiento limpio, sin señales)        (métricas por SNR × realización)
```

`noise_validate` depende de `noise_indicators` **y** de `validate` (limpio): de este último toma la verdad (intervalos), `t_onset_amp`, el modo de grises y las métricas limpias de referencia. No necesita el archivo de etiquetas.

## 4. Formatos de archivo (CONTRATO entre ramas)

Todo es **aditivo**: el modo antiguo de un solo caso de control sigue funcionando igual.

### 4.1 `doe_noise_results.h5` (modo multi-caso)
- **Grupos de primer nivel:** uno por copia ruidosa, nombre `snr_{SNR:06.2f}__{case}__r{K:02d}` (ej. `snr_040.00__case_011__r02`). Empieza por `snr_` a propósito: el visor y `doe_indicators.py` lo reconocen como archivo de ruido (no hay grupos `case_*` en la raíz).
- **Dentro:** `Axial_disp/{time,values}`, `Axial_vel/{time,values}` (ruidosas).
- **Attrs del grupo:** los del caso original (`kappa`, `$Ap_start$`, `$spin_rate$`, ...) + `snr_db`, `case_source`, `realization`, `seed`, `sigma_Axial_disp`, `sigma_Axial_vel`.
- **Attrs raíz:** `noise_layout = "multi"` (ausente = modo antiguo), `snr_mode = "absolute"`, `snr_ref_case`, `snr_ref_power_Axial_disp`, `snr_ref_power_Axial_vel`, `snr_levels` (array), `realizations`, `seed`, `cases` (array).
- Sin grupo `control` (lo limpio ya está en `doe_results.h5`).

### 4.2 `doe_noise_indicator_results.h5`
Igual que hoy (`<grupo>/<run>/{t, I_t, t_d}` + attrs `pp_*`, `meta_*`), con los attrs del grupo copiados. **Sin señales** cuando se pide `--no-signals` (§5.2).

### 4.3 `doe_noise_validation_results.h5` (nuevo)
- **Attrs raíz:** `schema = "doe_noise_validation_results/1"`, `gray_mode`, `clean_results` (nombre del archivo limpio), los attrs de §4.1 relevantes (`snr_mode`, `snr_ref_case`, `snr_levels`, `realizations`).
- **`/summary/<run>`:** una fila por copia ruidosa: `group`, `case_source`, `snr_db`, `realization`, `truth`, `is_gray`, `outcome`, `first_detection_t`, `t_ratio`, `delay_det_s`, `score_max`, `kappa` (mismas columnas que la validación limpia + `case_source`, `snr_db`, `realization`).
- **`/metrics/<run>`:** tabla (datasets alineados) con una fila por (`snr_db`, `realization`): `TP FN TN FP TPR TNR balanced_accuracy MCC AUC mean_alarm_fraction_stable median_t_ratio n_gray`.
- **`/by_snr/<run>`:** una fila por nivel: `snr_db` y, para cada métrica de arriba, `<m>_mean`, `<m>_min`, `<m>_max` entre realizaciones. Attr `snr_breakdown_db` = mayor SNR al que la balanced accuracy media cae más de 0.05 bajo la limpia (NaN si no cae).
- **`/clean/<run>`:** attrs con las métricas limpias (copiadas de `/metrics/<run>` del archivo limpio): son el punto de referencia "sin ruido".
- **CSV:** `<out>_by_snr.csv` (una fila por indicador y nivel).
- **Sin** grupos por caso con señales ni ROC por nivel (ponytail; añadir si se pide).

## 5. Cambios en los scripts — rama `wt-validacion` (yo)

### 5.1 `DOE_simulacion/doe_noise.py`
- Modo **multi-caso** cuando la sección `noise` trae `cases` (lista o `all`); sin `cases` = modo antiguo de un caso de control, intacto.
- Claves nuevas de la sección `noise` (y CLI equivalente): `cases`, `snr_list` (default `[80, 60, 40, 30, 20, 10]`), `realizations` (default 3), `snr_ref_case` (`auto` por defecto), `seed`, `signals`. `snr_range` se acepta e ignora con un aviso.
- `snr_ref_case: auto` → el caso etiquetado `unstable` con menor kappa, leído de las etiquetas del experimento (`ex.label_info`, como hace `doe_indicators.py`); sin etiquetas → error claro pidiendo `snr_ref_case`.
- Ruido: `rng = np.random.default_rng([seed, K, idx_caso, idx_señal])` → `z` unitario por (caso, realización, señal), reescalado por el sigma de cada nivel. Se **reusa** `add_gaussian_noise` cambiando solo de dónde sale sigma (absoluto, de `P_ref`).
- **Check:** selftest con un `doe_results` sintético: nombres de grupo, attrs, sigma igual para todos los casos de un nivel, reproducibilidad con la misma semilla, realizaciones distintas entre sí, modo antiguo sin cambios.

### 5.2 `DOE_analisis/doe_indicators.py`
- Flag `--no-signals`: `write_results` no copia `Axial_*` (sí los attrs del grupo). Por defecto copia, como hoy.
- Nada más: ya trata como "ruido" un archivo sin grupos `case_*` y usa `snr_db` como etiqueta.
- **Check:** caso en el selftest con `--no-signals`.

### 5.3 `DOE_analisis/validate_noise.py` (nuevo, CLI)
```
python validate_noise.py --noise_ind X/doe_noise_indicator_results.h5 --clean X/doe_validation_results.h5 [--out X/doe_noise_validation_results.h5]
```
- **Reusa** de `validate_indicators.py`: `pred_windows`, `truth_windows`, `case_truth`, `score`, `run_metrics`, `gray_suffix`, `GRAY_MODES`, `wilson`. Para no duplicar el bloque "verdad efectiva según `--gray`", se extrae a una función compartida `effective_truth(intervals, gray)` en `validate_indicators.py` (refactor sin cambio de comportamiento; su selftest lo cubre).
- Por copia ruidosa: intervalos del caso limpio (`case_NNN/truth_t0, truth_t1, truth_label` del archivo limpio), `t_onset_amp` del limpio, modo de grises del limpio → `score()` → fila de `/summary`.
- Métricas por (nivel, realización) con `run_metrics`; resumen por nivel; `snr_breakdown_db`; `/clean` copiado.
- **Check:** selftest sintético: un archivo limpio mínimo + un archivo de indicadores ruidosos con 2 niveles × 2 realizaciones; verdad tomada del limpio, conteos por nivel, media/mín/máx, `snr_breakdown_db`, grises según el modo.

### 5.4 `DOE_plots/validation_figures.py`
- Registro aparte `NOISE_FIGURES = {nombre: fn(h5_path, out_dir=None) -> Figure}` con la **misma API** y la misma tubería (`_figure`, `_fig`, `_grid`, `T`, `RUN_COLOR`, `OUTCOMES`, `_keep_size`, nota de modo de grises).
- `figs_dir(h5)` → `figs_noise_validation[_gray-<modo>]/` cuando `schema` empieza por `doe_noise_validation`.
- `make_all` elige `FIGURES` o `NOISE_FIGURES` según el `schema` del archivo.
- Figuras: §6.
- **Check:** el selftest genera un archivo de ruido sintético (vía 5.3) y dibuja todas las `NOISE_FIGURES`.

### 5.5 Documentación
- `GUIA_metricas_validacion.md`: sección nueva "Validación con ruido" (qué es el SNR absoluto, por qué entrenamiento limpio, realizaciones, cómo leer las 3 figuras), en el mismo tono para no especialistas.
- `INFORME_validacion.md`: estado y decisiones.

## 6. Figuras (article-plot-style)

**¿Se heredan las 15 actuales?** No directamente: leen `/summary` y `/metrics` de **un** archivo de validación (una fila por caso, un número por métrica), y aquí hay una dimensión más (SNR × realización). Se hereda **toda la tubería y el estilo**, no las figuras. Para mirar un nivel concreto con las figuras de siempre no se añade nada (ponytail): basta comparar con la validación limpia.

| Nombre | Preset | Datos | Pregunta |
|---|---|---|---|
| `noise_metrics` | grid 2×2 (`figsize_grid(2, 2)` × `FIGSCALE`) | `/by_snr/<run>` + `/clean/<run>` | ¿Cómo caen balanced accuracy, TPR, TNR y la fracción de alarma en estables al bajar el SNR? Línea = media, banda = mín–máx entre realizaciones, marcador a la derecha = limpio, línea vertical discontinua = `snr_breakdown_db`. Eje X de SNR **invertido** (de limpio a ruidoso). |
| `noise_case_matrix` | grid por indicador | `/summary/<run>` | ¿Qué casos dejan de acertar y a qué SNR? Filas = casos por kappa (S/U/g), columnas = niveles, color = fracción de realizaciones con acierto (0–1, escala secuencial apta para daltónicos). |
| `noise_anticipation` | SIMPLE | `/by_snr/<run>` `median_t_ratio_*` | ¿Se pierde anticipación con el ruido? `t_det/t_onset` mediana contra SNR, banda mín–máx. |

Todas con `lang_text` (EN/FR/both), `constrained_layout`, leyendas fuera de los ejes si tapan datos, y la nota de modo de grises cuando no es `ignore`.

## 7. Parte de `wt-interfaz` (la gestiona wt-interfaz; yo entrego contrato, scripts y selftests)

### 7.1 Creación y configuración (YAML + `experiment.py`)
- **Sección `noise`**: claves nuevas `cases`, `snr_list`, `realizations`, `snr_ref_case`, `seed`, `signals` (defaults de §5.1). Quitar `snr_range` del formulario (el script lo acepta e ignora). Mantener el modo antiguo si no hay `cases`.
- **Stage `noise`**: comando igual (`doe_noise.py --doe_results ... --out ... --experiment ...`). Con `snr_ref_case: auto` el script necesita las etiquetas del experimento: añadir la dependencia `(exp, "label_build")` (y su salida como entrada) cuando `noise.cases` esté definido. `out` por defecto sigue en la carpeta de datos (`shared=True`): avisar en el formulario que configuraciones distintas sobre los mismos datos necesitan `out` distinto.
- **Stage `noise_indicators`**: añadir `--no-signals` cuando `noise.cases` esté definido.
- **Stage nueva `noise_validate`**: dependencias `(exp, "noise_indicators")` y `(exp, "validate")`; entradas `[ni_out, val_out]`; salida `doe_noise_validation_results{gray_suffix(gray)}.h5` en `out_dir`; comando `validate_noise.py --noise_ind <ni_out> --clean <val_out> --out <nv_out>`; sección `noise_validate` (por ahora solo `out`). Añadir a `STAGES`, `OPTIONAL`, títulos, descripción y al grupo "Noise robustness".
- **Huellas**: comprobar que añadir estas etapas a un experimento existente **no** marca como desactualizadas las etapas limpias.
- **Formularios** (`launcher.py`): `NoiseForm` con las claves nuevas (lista de casos con selector, niveles, realizaciones, caso de referencia auto/manual); `noise_validate` sin campos salvo `out`.
- **Tarjeta de la etapa `noise_validate`**: por indicador, balanced accuracy limpia → al nivel más ruidoso, y `snr_breakdown_db`.

### 7.2 Visor y exportación
- `doe_unified_selector.py`: si el archivo tiene `schema` `doe_noise_validation_results/*` → nuevo tipo; entradas del combo de resumen desde `vf.NOISE_FIGURES` (igual que hoy con `vf.FIGURES`); "Save PNG" → `vf.figs_dir(h5)`.
- Archivos `doe_noise_results.h5` / `doe_noise_indicator_results.h5` en modo multi (`noise_layout = "multi"`): mostrar columnas `case_source`, `snr_db`, `realization` en la tabla, y **ocultar** las figuras antiguas de un solo caso (`doe_noise_plotter`, que leen el SNR del nombre del grupo y suponen un control).
- Botón "Viewer" de la etapa `noise_validate` → abre `doe_noise_validation_results*.h5`.
- Exportación: las figuras van a `figs_noise_validation[_gray-<modo>]/` junto al `.h5`; el CSV `<out>_by_snr.csv` ya lo escribe el script.

## 8. Tamaño y coste (medido en n12000)

- Una señal ≈ 4.5 MB (float64, 562 570 muestras). Cada copia ruidosa lleva 2 señales ≈ 9 MB; el ruido gaussiano casi no se comprime.
- Primera pasada con 12 casos × 6 niveles × 3 realizaciones = **216 copias ≈ 2 GB** de `doe_noise_results.h5`.
- `doe_noise_indicator_results.h5` sin señales: solo `t`, `I_t`, `t_d` por indicador (del orden de MB por copia, a medir en la fase E2E).
- `doe_noise_validation_results.h5`: solo tablas (pocos MB).
- Tiempo: 216 copias × 4 indicadores; se mide en la fase E2E antes de pasar a `all`.
- **Alternativa anotada** (no se hace ahora): no guardar las señales ruidosas y regenerarlas desde la semilla dentro de `noise_indicators`.

## 9. Fases

| Fase | Quién | Qué | Commit / señal |
|---|---|---|---|
| N1 | wt-validacion | `doe_noise.py` multi-caso + selftest | `feat(noise): multi-case absolute-SNR noise with realizations` |
| N2 | wt-validacion | `doe_indicators.py --no-signals` + selftest | `feat(noise): --no-signals for noisy indicator results` |
| N3 | wt-validacion | `effective_truth` compartida + `validate_noise.py` + selftest | `feat(validation): validate_noise.py (metrics per SNR x realization)` |
| N4 | wt-validacion | `NOISE_FIGURES` + `figs_dir` + selftest | `feat(validation): noise validation figures` |
| N5 | wt-validacion | E2E por CLI en n12000 con 12 casos: tamaños y tiempos reales, revisión a ojo de las figuras | ajustes + `docs(validation)` GUIA/INFORME |
| N6 | wt-validacion → manager + wt-interfaz | Mensaje de entrega: hash, contrato (§4), comandos (§5), resultados E2E | — |
| I1 | wt-interfaz | YAML + `experiment.py` (stages `noise` ampliado, `noise_indicators --no-signals`, `noise_validate`) | su rama |
| I2 | wt-interfaz | Formularios y tarjetas | su rama |
| I3 | wt-interfaz | Visor: tipo nuevo, `NOISE_FIGURES`, columnas multi, ocultar figuras antiguas, `figs_dir` | su rama |
| J1 | ambos | E2E desde la app sobre n12000 (12 casos): crear etapas, configurar, correr, ver y exportar | lista de problemas por mensaje |
| J2 | quien corresponda | Correcciones; luego `cases: all` y, si sirve, más realizaciones | — |

## 10. Protocolo de coordinación entre sesiones

- **Una sola fuente de verdad:** este archivo (§4 = contrato). Cualquier cambio de formato o de CLI se escribe **aquí primero** y se avisa por mensaje.
- **Propiedad de archivos:** wt-validacion = `doe_noise.py`, `doe_indicators.py`, `validate_indicators.py`, `validate_noise.py`, `validation_figures.py`, `GUIA`, `INFORME`, este plan. wt-interfaz = `experiment.py`, `launcher.py`, `doe_unified_selector.py`, `sld_model.py`, `check_app_dialogs.py`, YAML, `doe_noise_plotter.py`. Nadie edita archivos del otro: se pide por mensaje.
- **Mensajes** (siempre con copia al manager): al terminar N4 y N5 (hash + qué probar), al terminar I1–I3 (hash + qué cambió), y cuando algo del contrato no encaje (antes de improvisar).
- **Reglas comunes:** cambios aditivos (si una clave o flag desaparece, se acepta e ignora); finales de línea LF verificados por bytes; un selftest por lógica nueva; commits locales pequeños; sin push ni ramas nuevas sin el usuario.
- **Integración:** wt-interfaz hace `git merge wt-validacion` al recibir el mensaje de N6.

## 11. Verificación de punta a punta

1. Selftests: `doe_noise.py --selftest`, `doe_indicators.py --selftest`, `validate_indicators.py --selftest`, `validate_noise.py --selftest`, `validation_figures.py --selftest` (con `entorno_CAMP10\Scripts\python.exe`).
2. CLI en n12000 (12 casos, 6 niveles, 3 realizaciones): `doe_noise.py --experiment ...` → `doe_indicators.py --doe_results <ruido> --no-signals` → `validate_noise.py --noise_ind ... --clean <validación limpia>` → `validation_figures.py --results <validación con ruido>`; tamaños y tiempos anotados en §8.
3. Comprobaciones de sentido: a 80 dB las métricas ≈ limpias; el sigma de un nivel es el mismo para todos los casos; misma semilla → mismo archivo; las tres realizaciones difieren.
4. Desde la app (J1): añadir las etapas al YAML de n12000, correr, abrir el visor, guardar figuras.

## 12. Fuera de alcance (anotado)

- Entrenar con ruido (variante: experimento de entrenamiento ruidoso como `reference`).
- Ruido coherente entre desplazamiento y velocidad.
- Rampas con ruido (las rampas siguen diferidas).
- ROC y prueba entre indicadores por nivel de SNR.
