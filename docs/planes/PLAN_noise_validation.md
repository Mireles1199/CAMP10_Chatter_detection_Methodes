# PLAN — Validación con ruido blanco (Ap constante)

**Estado (2026-10-05):** N1–N4 ✅ hechos en `wt-validacion` (a57c847 doe_noise, 4cabb64 + 5036ca0 doe_indicators, b80745c refactor, cbb3296 validate_noise, 86df0a2 figuras; selftests OK). N5 ✅ (n12000 por CLI, 12 casos × 6 niveles × 3 realizaciones; medidas en §8). J2 ✅ (2026-10-06): `cases: all` (22) × 6 × 5 realizaciones de ruido, indicadores y validación de la realización 0 (`validacion_figs/noise_all22_r5/`, §8); `doe_indicators.py --only/--resume/--realizations`, `validate_noise.py --realizations`. wt-interfaz: I1–I3 ✅ (merge de wt-validacion 8788d2e en wt-interfaz → 6e7f0bb; e6b374b etapas, 1aa3bd0 formularios, ee0b09a visor). Rama de la interfaz: `wt-interfaz`. Coordina: sesión `docs/planes/PLAN_app_v2.md` (manager).
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
| 2 | Definición del SNR | **Absoluto**: `sigma = sqrt(P_ref / 10^(SNR/10))`, un mismo sigma para todos los casos. `P_ref` = varianza de la señal del **caso inestable más débil** (menor η etiquetado inestable), por señal. Opción: `snr_ref_case` explícito. |
| 3 | Niveles | Lista explícita, primera pasada **80, 60, 40, 30, 20, 10 dB**. El rango antiguo (`SNR_RANGE`, 5–200 dB) **desaparece** (se acepta y se ignora con aviso). |
| 4 | Realizaciones | **3** en la primera pasada (semillas independientes). Métricas por realización y resumen media / mín / máx entre ellas (no se mezclan como casos). Dentro de una realización, el mismo ruido unitario reescalado en todos los niveles (curvas más limpias). |
| 5 | Etiqueta | La del caso limpio. Entrenamiento limpio. |
| 6 | Señales con ruido | `Axial_disp` y `Axial_vel`, ruido independiente en cada una (**simplificación declarada**; opción futura: derivar la velocidad ruidosa del desplazamiento). |
| 7 | Grises | Mismo modo que la validación limpia (`gray_mode` del archivo limpio). |
| 8 | Tamaño | No copiar señales en los resultados de indicadores y de validación con ruido (§8). |

Referencia en n12000 (medida): caso de referencia η 1.082, RMS desplazamiento 1.28e-5 m, velocidad 1.2e-2 m/s → σ_disp = 1.3e-8 m a 60 dB, 1.3e-6 m a 20 dB. Los casos estables vibran 1.3e-8 – 2.4e-8 m (RMS): desde ~60 dB el ruido ya iguala su vibración.

## 3. Flujo

```
doe_results.h5 (limpio) ──label_build──> reference_dataset_amp.h5 ──┐
        │                                                            │
        ├──indicators──> doe_indicator_results.h5 ──validate──> doe_validation_results.h5 (limpio; verdad + t_onset + gray_mode)
        │                                                                                     │
        └──noise──> doe_noise_multi_results.h5 ──noise_indicators──> doe_noise_indicator_results.h5 ┴──noise_validate──> doe_noise_validation_results.h5
             (copias ruidosas)          (umbrales del entrenamiento limpio, sin señales)        (métricas por SNR × realización)
```

`noise_validate` depende de `noise_indicators` **y** de `validate` (limpio): de este último toma la verdad (intervalos), `t_onset_amp`, el modo de grises y las métricas limpias de referencia. No necesita el archivo de etiquetas.

## 4. Formatos de archivo (CONTRATO entre ramas)

Todo es **aditivo**: el modo antiguo de un solo caso de control sigue funcionando igual.

### 4.1 `doe_noise_multi_results.h5` (modo multi-caso)
- **Nombre y lugar propios:** en modo multi el archivo por defecto es `doe_noise_multi_results.h5` y va a la carpeta de salida **del experimento** (`out_dir`), no a la de datos: depende de las etiquetas y de la lista de casos de ese experimento, así que dos experimentos sobre los mismos datos no se pisan. El modo de un solo caso sigue igual (`doe_noise_results.h5`, compartido).
- **`time` sin duplicar:** el `time` de cada señal se guarda una vez por caso; las demás copias de ese caso (y `Axial_vel/time` si es igual a `Axial_disp/time`) son **enlaces duros de HDF5** al primero. Transparente para quien lee, coste cero.
- **Grupos de primer nivel:** uno por copia ruidosa, nombre `snr_{SNR:06.2f}__{case}__r{K:02d}` (ej. `snr_040.00__case_011__r02`). Empieza por `snr_` a propósito: el visor y `doe_indicators.py` lo reconocen como archivo de ruido (no hay grupos `case_*` en la raíz).
- **Dentro:** `Axial_disp/{time,values}`, `Axial_vel/{time,values}` (ruidosas).
- **Attrs del grupo:** los del caso original (`eta`, `$Ap_start$`, `$spin_rate$`, ...) + `snr_db`, `case_source`, `realization`, `seed`, `sigma_Axial_disp`, `sigma_Axial_vel`. **Contrato:** `realization` está en cada grupo de este archivo y del de indicadores (doe_indicators copia los attrs de grupo); el visor lo usa para reconocer el modo multi.
- **Attrs raíz:** `noise_layout = "multi"` (ausente = modo antiguo), `snr_mode = "absolute"`, `snr_ref_case`, `snr_ref_power_Axial_disp`, `snr_ref_power_Axial_vel`, `snr_levels` (array), `realizations`, `seed`, `cases` (array).
- Sin grupo `control` (lo limpio ya está en `doe_results.h5`).

### 4.2 `doe_noise_indicator_results.h5`
Igual que hoy (`<grupo>/<run>/{t, I_t, t_d}` + attrs `pp_*`, `meta_*`), con los attrs del grupo copiados. **Sin señales** cuando se pide `--no-signals` (§5.2).

### 4.3 `doe_noise_validation_results.h5` (nuevo)
- **Attrs raíz:** `schema = "doe_noise_validation_results/1"`, `gray_mode`, `clean_results` (nombre del archivo limpio), los attrs de §4.1 relevantes (`snr_mode`, `snr_ref_case`, `snr_levels`, `realizations`).
- **`/summary/<run>`:** una fila por copia ruidosa: `copy` (nombre del grupo ruidoso), `case` (el caso limpio, = attr `case_source`), `snr_db`, `realization`, `truth`, `is_gray`, `outcome`, `eta`, `first_detection_t`, `t_ratio`, `delay_det_s`, `score_max`, `alarm_fraction` (nombres como los escribe `validate_noise.py`).
- **`/metrics/<run>`:** grupo de **datasets 1-D alineados** (no attrs: en la validación limpia `/metrics/<run>` son attrs; aquí, mismo nombre y otro formato) con una fila por (`snr_db`, `realization`): `TP FN TN FP TPR TNR balanced_accuracy MCC AUC mean_alarm_fraction_stable median_t_ratio n_gray`.
- **`/by_snr/<run>`:** grupo de datasets 1-D alineados, una fila por nivel: `snr_db` y, para cada métrica de arriba, `<m>_mean`, `<m>_min`, `<m>_max` entre realizaciones. Attr `snr_breakdown_db` = mayor SNR al que la balanced accuracy media cae más de 0.05 bajo la limpia **de los mismos casos** (`/clean/<run>` balanced_accuracy; NaN si no cae).
- **`/clean/<run>`:** attrs con las métricas limpias **sobre los mismos casos que tienen copias ruidosas** (las corridas limpias de esos casos, leídas del archivo limpio y puntuadas con las mismas reglas y el mismo modo de grises): es la referencia "sin ruido" comparable. Además, `<m>_all` = la métrica limpia sobre **todos** los casos del archivo limpio (copiada de su `/metrics/<run>`). Si un indicador no está en la validación limpia, `/clean/<run>` existe con NaN y nada falla. *(Cambio de semántica, mismo nombre, tras J1 de wt-interfaz: antes era la métrica de todos los casos y producía quiebres falsos, p. ej. rms_cv 0.56 con 22 casos frente a 0.50 con los 3 casos con ruido.)*
- **CSV:** `<out>_by_snr.csv` (una fila por indicador y nivel).
- **Sin** grupos por caso con señales ni ROC por nivel (ponytail; añadir si se pide).

### 4.4 Orígenes de los datos (attrs raíz; sin duplicar señales) — contrato 2026-10-06
Cada archivo de resultados dice de dónde vienen sus datos. Para cada origen `K` hay **dos attrs raíz**: `K_rel` (ruta **relativa a la carpeta de este .h5**; ausente si no se puede relativizar, p. ej. otra unidad) y `K_abs` (absoluta, de respaldo). Resolución: primero `K_rel`, luego `K_abs`; si ninguna existe, el origen falta.

| Clave `K` | Qué es | `doe_noise_multi_results.h5` | `doe_noise_indicator_results.h5` | `doe_noise_validation_results.h5` |
|---|---|---|---|---|
| `source_signals` | `doe_results.h5` limpio (señales de los casos) | ✔ | ✔ | ✔ |
| `clean_indicators` | `doe_indicator_results.h5` limpio (I(t) de los casos) | ✔ solo si se genera con `--experiment` | ✔ (heredado) | ✔ |
| `noise_results` | `doe_noise_multi_results.h5` (señales ruidosas) | — | ✔ | ✔ |
| `noise_indicators` | `doe_noise_indicator_results.h5` (I(t) ruidosos) | — | — | ✔ |
| `clean_validation` | `doe_validation_results*.h5` limpio (verdad, métricas) | — | — | ✔ |

- Los archivos **limpios** también los llevan, aditivos: `doe_indicator_results.h5` → `source_signals`; `doe_validation_results*.h5` → `clean_indicators` (y `source_signals` si el de indicadores lo trae). Los archivos viejos no los tienen: el lector devuelve "origen ausente", no falla.
- Los attrs de nombre-solo existentes (`clean_results`, `noise_indicator_results`, `indicator_results_file`...) se **conservan**.
- **Vínculo copia → original:** el attr de grupo `case_source` de cada copia ruidosa (caso limpio, p. ej. `case_011`); `realization` y `snr_db` como hoy. Nombre de grupo de la copia **sin cambios** (`snr_040.00__case_011__r00`), igual en el archivo de ruido, el de indicadores y los subárboles. En modo multi **no hay grupo `control`**: los casos limpios no son grupos de ningún archivo de ruido; se leen del origen.
- **Helper** (solo h5py + numpy + os, sin matplotlib ni paquetes de indicadores): `DOE_utils/noise_origins.py`
  - `origins(h5_path) -> {clave: ruta | None}` (las cinco claves; `None` = ausente o no encontrada).
  - `OriginMissing(KeyError)` con `.key` y `.tried` (rutas probadas).
  - `read_signal(origin_path, case, name) -> (t, y)` (origin = `doe_results.h5` o `doe_noise_multi_results.h5`; `case` = `case_NNN` o nombre de copia).
  - `read_indicator(origin_path, case, variant) -> {"t", "I_t", "t_d", "attrs"}` (origin = archivo de indicadores, limpio o ruidoso).
  - `signal_of(h5_path, group, name) -> (t, y)`: desde cualquiera de los tres archivos de ruido o un subárbol de nivel; `group` = copia (→ `noise_results`), caso limpio (→ `source_signals`) o caso de un subárbol (usa su attr `copy`). Lanza `OriginMissing` si falta el origen.
  - Escritura (la usan los scripts): `set_origins(h5_path, **{clave: ruta})`, `inherit_origins(dst_h5, src_h5)` (copia los orígenes de otro archivo y recalcula las rutas relativas).

### 4.5 `doe_noise_validation_results.h5`: subárbol por nivel (schema `doe_noise_validation_results/2`)
Además de `/summary`, `/metrics`, `/by_snr`, `/clean` (sin cambios), **un grupo de primer nivel por nivel de SNR** con **exactamente la estructura de `doe_validation_results.h5`** (schema `doe_validation_results/4`): `/<sub>/{summary, metrics, ranking, roc, pairwise, training, case_NNN/...}` y los mismos attrs raíz de la validación limpia (`channel`, `gray_mode`, `labeling_*`...) más `snr_db` y `realization`.
- **Nombre del subárbol:** `snr_{SNR:06.2f}` (ej. `snr_040.00`) si se puntuó **una** realización; con varias, `snr_{SNR:06.2f}__r{K:02d}`. El orden y los nombres están en el attr raíz **`snr_subtrees`** (array de str, de limpio a ruidoso; dentro de un nivel por realización) — no hay que adivinarlos. Raíz, además: `snr_levels`, `realizations_scored`, `source_signals_*`... (§4.4).
- **Dentro de `case_NNN`:** attrs `$eta$ $truth$ $gray$ $t_onset_amp$ $group$ $t_onset$ $outcome_<run>$` (como la limpia), `copy` (nombre de la copia ruidosa) y `case_source`; `truth_t0/t1/label`; subgrupos de indicador con `t, I_t, t_d, pred, truth_w` + métricas. **Sin `Axial_*`**: las señales se leen del origen (`noise_origins.signal_of`; la ruidosa vía `copy` → `noise_results`).
- `/training` va una sola vez en la raíz y los subárboles lo enlazan (enlace duro **interno**, no externo).
- Un nivel con los mismos datos que la limpia da las mismas métricas (selftest). Con 1 realización cada nivel es una validación limpia de 22 casos: Wilson, McNemar y ROC válidos.

## 5. Cambios en los scripts — rama `wt-validacion` (yo)

### 5.1 `DOE_simulacion/doe_noise.py`
- Modo **multi-caso** cuando la sección `noise` trae `cases` (lista o `all`); sin `cases` = modo antiguo de un caso de control, intacto (mismo comando, misma huella). Sin `--out`, el nombre por defecto en modo multi es `doe_noise_multi_results.h5`.
- Claves nuevas de la sección `noise` (y CLI equivalente): `cases`, `snr_list` (default `[80, 60, 40, 30, 20, 10]`), `realizations` (default 3), `snr_ref_case` (`auto` por defecto), `seed`, `signals`. `snr_range` se acepta e ignora con un aviso.
- `snr_ref_case: auto` → el caso etiquetado `unstable` con menor η, leído de las etiquetas **propias** del experimento (`exp.label["out"]` con `ex.label_info`, como `doe_indicators.py`), **no** de `exp.reference` (en una validación es el entrenamiento, con otros casos); sin etiquetas → error claro pidiendo `snr_ref_case`.
- `time` una vez por caso y enlaces duros de HDF5 en las demás copias (§4.1).
- Ruido: `rng = np.random.default_rng([seed, K, idx_caso, idx_señal])` → `z` unitario por (caso, realización, señal), reescalado por el sigma de cada nivel. Se **reusa** `add_gaussian_noise` cambiando solo de dónde sale sigma (absoluto, de `P_ref`).
- **Check:** selftest con un `doe_results` sintético: nombres de grupo, attrs, sigma igual para todos los casos de un nivel, reproducibilidad con la misma semilla, realizaciones distintas entre sí, modo antiguo sin cambios.

### 5.2 `DOE_analisis/doe_indicators.py`
- Flag `--no-signals`: `write_results` no copia `Axial_*` (sí los attrs del grupo). Por defecto copia, como hoy.
- En el layout de ruido, **ignorar `indicators.cases`** (usar `all`): hoy `ENABLED_CASES` también filtraría los grupos de ruido y daría 0 grupos si un experimento limita `indicators.cases`; la selección de casos ya la hace `noise.cases`.
- Nada más: ya trata como "ruido" un archivo sin grupos `case_*` y usa `snr_db` como etiqueta.
- **Check:** caso en el selftest con `--no-signals`.

### 5.3 `DOE_analisis/validate_noise.py` (nuevo, CLI)
```
python validate_noise.py --noise_ind X/doe_noise_indicator_results.h5 --clean X/doe_validation_results.h5 [--out X/doe_noise_validation_results.h5]
```
- **Reusa** de `validate_indicators.py`: `pred_windows`, `truth_windows`, `case_truth`, `score`, `run_metrics`, `gray_suffix`, `GRAY_MODES`, `wilson`. Para no duplicar el bloque "verdad efectiva según `--gray`", se extrae a una función compartida `effective_truth(intervals, gray)` en `validate_indicators.py` (refactor sin cambio de comportamiento; su selftest lo cubre).
- Por copia ruidosa: intervalos del caso limpio (`case_NNN/truth_t0, truth_t1, truth_label` del archivo limpio), `t_onset_amp` del limpio, modo de grises del limpio → `score()` → fila de `/summary`.
- Métricas por (nivel, realización) con `run_metrics`; resumen por nivel; `snr_breakdown_db`; `/clean` copiado (NaN si el indicador falta en el limpio).
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
| `noise_case_matrix` | grid por indicador | `/summary/<run>` | ¿Qué casos dejan de acertar y a qué SNR? Filas = casos por η (S/U/g), columnas = niveles, color = fracción de realizaciones con acierto (0–1, escala secuencial apta para daltónicos). |
| `noise_anticipation` | SIMPLE | `/by_snr/<run>` `median_t_ratio_*` | ¿Se pierde anticipación con el ruido? `t_det/t_onset` mediana contra SNR, banda mín–máx. |

Todas con `lang_text` (EN/FR/both), `constrained_layout`, leyendas fuera de los ejes si tapan datos, y la nota de modo de grises cuando no es `ignore`.

### 6.1 API de figuras con nivel (2026-10-06)
- `FIGURES[<nombre>](h5_path, out_dir=None, snr=None, realization=None)` (las 15): con un archivo de validación **con ruido** y `snr=40` dibuja ese nivel (subárbol §4.5; `realization` por defecto = la primera puntuada); sin `snr` en un archivo con ruido, error claro. En un archivo limpio, `snr` se rechaza. `NOISE_FIGURES[...](h5_path, out_dir=None, snr=None)` ignora `snr` (resumen entre niveles).
- `figs_dir(h5, snr=None)`: `.../figs_noise_validation[_gray-<modo>]` sin `snr`; con él `.../figs_noise_validation[_gray-<modo>]/snr_040`. `make_all(h5, out_dir=None, snr=None)`: en un archivo con ruido, `NOISE_FIGURES` siempre + las 15 de cada nivel de `snr` (número, lista o `"all"`).
- CLI: `validation_figures.py --results <h5> [--snr 40 [20 ...] | --snr all]`.

## 7. Parte de `wt-interfaz` (la gestiona wt-interfaz; yo entrego contrato, scripts y selftests)

### 7.1 Creación y configuración (YAML + `experiment.py`)
- **Sección `noise`**: claves nuevas `cases`, `snr_list`, `realizations`, `snr_ref_case`, `seed`, `signals` (defaults de §5.1). Quitar `snr_range` del formulario (el script lo acepta e ignora). Mantener el modo antiguo si no hay `cases`.
- **Stage `noise`**: comando igual (`doe_noise.py --doe_results ... --out ... --experiment ...`). Con `snr_ref_case: auto` el script necesita las etiquetas del experimento: añadir la dependencia `(exp, "label_build")` (y su salida como entrada) cuando `noise.cases` esté definido. `out` por defecto sigue en la carpeta de datos (`shared=True`); **en modo multi el `out` por defecto es `doe_noise_multi_results.h5` en `out_dir` del experimento** (propuesta de wt-interfaz, aceptada); el de un solo caso sigue en la carpeta de datos (`doe_noise_results.h5`, sin cambios).
- **Stage `noise_indicators`**: añadir `--no-signals` cuando `noise.cases` esté definido (su huella cambia en ese caso, re-ejecución legítima). Sin `cases`, comando idéntico al actual.
- **Stage nueva `noise_validate`**: dependencias `(exp, "noise_indicators")` y `(exp, "validate")`; entradas `[ni_out, val_out]`; salida `doe_noise_validation_results{gray_suffix(gray)}.h5` en `out_dir`; comando `validate_noise.py --noise_ind <ni_out> --clean <val_out> --out <nv_out>`, con `val_out` = **`validation_path(exp)`** (la misma función que usa la etapa `validate`, con el sufijo del modo de grises) y `nv_out` con el mismo sufijo; el script lee `gray_mode` del archivo limpio; sección `noise_validate` (por ahora solo `out`). Añadir a `STAGES`, `OPTIONAL`, títulos, descripción y al grupo "Noise robustness".
- **Huellas**: comprobar que añadir estas etapas a un experimento existente **no** marca como desactualizadas las etapas limpias.
- **Formularios** (`launcher.py`): `NoiseForm` con las claves nuevas (lista de casos con selector, niveles, realizaciones, caso de referencia auto/manual); `noise_validate` sin campos salvo `out`.
- **Tarjeta de la etapa `noise_validate`**: por indicador, balanced accuracy limpia → al nivel más ruidoso, y `snr_breakdown_db`.

### 7.2 Visor y exportación
- `doe_unified_selector.py`: si el archivo tiene `schema` `doe_noise_validation_results/*` → nuevo tipo; entradas del combo de resumen desde `vf.NOISE_FIGURES` (igual que hoy con `vf.FIGURES`); "Save PNG" → `vf.figs_dir(h5)`.
- Archivos `doe_noise_multi_results.h5` / `doe_noise_indicator_results.h5` en modo multi (`noise_layout = "multi"`): mostrar columnas `case_source`, `snr_db`, `realization` en la tabla, y **ocultar** las figuras antiguas de un solo caso (`doe_noise_plotter`, que leen el SNR del nombre del grupo y suponen un control).
- Detección del modo multi por el attr de grupo `realization`; `load_doe_noise` perezoso (hoy lee todas las señales al abrir: un archivo multi cargaría GB). Guard en el CLI de `doe_noise_plotter.py` para nombres compuestos. Objetivo nuevo "Noise validation" → [noise_validate]; "Noise robustness" → [noise_indicators] se mantiene. Sin `noise.cases`, `noise_validate` aparece como no ejecutable con su motivo.
- Botón "Viewer" de la etapa `noise_validate` → abre `doe_noise_validation_results*.h5`.
- Exportación: las figuras van a `figs_noise_validation[_gray-<modo>]/` junto al `.h5`; el CSV `<out>_by_snr.csv` ya lo escribe el script.

## 8. Tamaño y coste (medido en n12000)

- Una señal ≈ 4.5 MB (float64, 562 570 muestras). Cada copia ruidosa lleva 2 señales ≈ 9 MB; el ruido gaussiano casi no se comprime.
- Primera pasada con 12 casos × 6 niveles × 3 realizaciones = **216 copias ≈ 2 GB** de `doe_noise_multi_results.h5`. **Se pide confirmación al usuario antes de generarlo (N5).**
- `doe_noise_indicator_results.h5` sin señales: solo `t`, `I_t`, `t_d` por indicador (del orden de MB por copia, a medir en la fase E2E).
- `doe_noise_validation_results.h5`: solo tablas (pocos MB).
- Con `time` en enlaces duros solo cuentan los `values` (≈ 9 MB por copia → ≈ 2 GB); sin ellos serían ≈ 3.9 GB.
- Tiempo: 216 copias × 4 indicadores; se mide en la fase E2E antes de pasar a `all`, **con más `workers`** (n12000 tiene `indicators.workers: 1`; `workers` no entra en la huella).
- No abrir el archivo multi en el visor hasta que I3 haga la carga perezosa.
- **Medido en N5 (n12000, 12 casos × 6 niveles × 3 realizaciones = 216 copias):** `doe_noise_multi_results.h5` = 2.13 GB, generado en 6 s. `doe_noise_indicator_results.h5` (sin señales) = 65 MB. `doe_noise_validation_results.h5` = 0.36 MB (+ `_by_snr.csv` 8 KB); `validate_noise.py` tarda 2 s. Indicadores (864 tareas): un grupo con los 4 indicadores en un proceso = 87 s → 1 worker ≈ 5.2 h; 3 workers ≈ 5.8 tareas/min (≈ 2.5 h); 6 workers ≈ 11 tareas/min (≈ 1.3 h) pero sin memoria virtual (ver arriba). Windows también puede suspender el equipo y cortar la corrida: se reanuda con `--cases` de los grupos incompletos (en N5 se usó un lanzador que reintenta y pide no suspender, `SetThreadExecutionState`).
- **Memoria (medido en N5):** cada worker de `noise_indicators` llega a ~3 GB de pico (un grupo, 4 indicadores, 1 proceso: 87 s, 2.9 GB). Con 6 workers Windows se quedó sin memoria virtual (log System 18:27:02, "Mémoire virtuelle minimale insuffisante") y el pool se cayó tras 399/864 tareas. **Usar `workers` <= 3.** Como los resultados se escriben incrementalmente, se retoma pasando `--cases` con los grupos incompletos.
- **Medido en la ampliación (22 casos × 6 niveles × 5 realizaciones = 660 copias, seed 42, 2026-10-06):** `doe_noise_multi_results.h5` = 6.45 GB en 44 s; indicadores solo de r00 (132 grupos × 4) = 69 MB (incluye ~25 grupos de r03/r04 a 10 dB que sobraron de una corrida cortada; se ignoran con `--realizations 0`); `doe_noise_validation_results.h5` = 0.27 MB. Tiempo de indicadores ~4.7 h con 3 workers; ritmo 3–4.5 tareas/min, Green domina (~6 min por tarea a 10 dB, mucho menos con poco ruido). **Fuga de memoria encontrada y corregida:** `run_all` guardaba todos los resultados en `future_map` (~190 MB por tarea, 30 GB tras 93 tareas); con 2640 tareas habría agotado la memoria virtual (probable causa real del fallo de N5 con 6 workers). Con la corrección el proceso principal se queda en ~2.7 GB. Quiebres con 22 casos: green/ssq 20 dB, maxent 40 dB, rms_cv 80 dB (iguales que con 12 casos).
- **Alternativa anotada** (no se hace ahora): no guardar las señales ruidosas y regenerarlas desde la semilla dentro de `noise_indicators`.

## 9. Fases

| Fase | Quién | Qué | Commit / señal |
|---|---|---|---|
| N1 ✅ | wt-validacion | `doe_noise.py` multi-caso + selftest | `feat(noise): multi-case absolute-SNR noise with realizations` |
| N2 ✅ | wt-validacion | `doe_indicators.py --no-signals` + selftest | `feat(noise): --no-signals for noisy indicator results` |
| N3 ✅ | wt-validacion | `effective_truth` compartida + `validate_noise.py` + selftest | `feat(validation): validate_noise.py (metrics per SNR x realization)` |
| N4 ✅ | wt-validacion | `NOISE_FIGURES` + `figs_dir` + selftest | `feat(validation): noise validation figures` |
| N5 ✅ | wt-validacion | E2E por CLI en n12000 con 12 casos (~2 GB en disco: **confirmar con el usuario antes**): tamaños y tiempos reales, revisión a ojo de las figuras | ajustes + `docs(validation)` GUIA/INFORME |
| N6 | wt-validacion → manager + wt-interfaz | Mensaje de entrega: hash, contrato (§4), comandos (§5), resultados E2E | — |
| I1 ✅ | wt-interfaz | YAML + `experiment.py` (stages `noise` ampliado, `noise_indicators --no-signals`, `noise_validate`) | su rama |
| I2 ✅ | wt-interfaz | Formularios y tarjetas | su rama |
| I3 ✅ | wt-interfaz | Visor: tipo nuevo, `NOISE_FIGURES`, columnas multi, ocultar figuras antiguas, `figs_dir` | su rama |
| J1 | ambos | E2E desde la app sobre n12000 (12 casos): crear etapas, configurar, correr, ver y exportar | lista de problemas por mensaje |
| J4 ✅ | wt-validacion | A) `noise_all22_r1` (solo r00, 132 copias, bit a bit = r00 de r5) + archivo de indicadores con solo r00; B) orígenes §4.4 + `noise_origins.py`; C) subárboles por nivel §4.5 + figuras por nivel §6.1 | 767a43d, 3e253bb (A+B); d815c7d, 766d866, fb8d605 (C) |
| J2 ✅ | wt-validacion | `cases: all` (22) con 5 realizaciones de ruido; indicadores y validación de r00; `--only`, `--resume`, `--realizations` (doe_indicators y validate_noise), `prepare_run`, fuga de memoria; líneas de quiebre de `noise_metrics` | 298ca86, 0394156, 7146aa0, 66d0b2a, 409123b |
| J3 | wt-interfaz | Probar los flags reales desde la app; ampliar a r01–r02 con `--resume` si hace falta | — |

## 10. Protocolo de coordinación entre sesiones

- **Una sola fuente de verdad:** este archivo (§4 = contrato). Cualquier cambio de formato o de CLI se escribe **aquí primero** y se avisa por mensaje.
- **Propiedad de archivos:** wt-validacion = `doe_noise.py`, `doe_indicators.py`, `validate_indicators.py`, `validate_noise.py`, `validation_figures.py`, `GUIA`, `INFORME`, este plan. wt-interfaz = `DOE_utils/experiment.py`, `DOE_utils/launcher.py`, `DOE_utils/DOE_plots/doe_unified_selector.py`, `DOE_utils/DOE_plots/sld_model.py`, `DOE_utils/check_app_dialogs.py`, YAML, `DOE_utils/DOE_plots/doe_noise_plotter.py`. (Rutas de §5 relativas a `DOE_utils/`.) Nadie edita archivos del otro: se pide por mensaje.
- **Mensajes** (siempre con copia al manager): al terminar N4 y N5 (hash + qué probar), al terminar I1–I3 (hash + qué cambió), y cuando algo del contrato no encaje (antes de improvisar).
- **Reglas comunes:** cambios aditivos (si una clave o flag desaparece, se acepta e ignora); finales de línea LF verificados por bytes; un selftest por lógica nueva; commits locales pequeños; sin push ni ramas nuevas sin el usuario.
- **Integración:** wt-interfaz hace `git merge wt-validacion` al recibir el mensaje de N6. Si el usuario lo autoriza, wt-interfaz puede empezar I1 antes, contra el contrato de §4.
- **Revisión de wt-interfaz (2026-10-05):** etiquetas propias para la referencia del SNR, `out` multi en `out_dir`, `time` en enlaces duros, `indicators.cases` ignorado en layout de ruido, `realization` como marca de modo multi, formatos 1-D explícitos en §4.3, `/clean` con NaN si falta el indicador, `workers` en N5, visor perezoso en I3.
- **Revisión del manager (2026-10-05):** huellas por etapa (añadir `noise_validate` no toca las limpias), nombre `noise_validate` confirmado, `doe_noise_plotter.py` del lado de wt-interfaz, out multi propio, `--clean` vía `validation_path(exp)`.

## 11. Verificación de punta a punta

1. Selftests: `doe_noise.py --selftest`, `doe_indicators.py --selftest`, `validate_indicators.py --selftest`, `validate_noise.py --selftest`, `validation_figures.py --selftest` (con `entorno_CAMP10\Scripts\python.exe`).
2. CLI en n12000 (12 casos, 6 niveles, 3 realizaciones): `doe_noise.py --experiment ...` → `doe_indicators.py --doe_results <ruido> --no-signals` → `validate_noise.py --noise_ind ... --clean <validación limpia>` → `validation_figures.py --results <validación con ruido>`; tamaños y tiempos anotados en §8.
3. Comprobaciones de sentido: a 80 dB las métricas ≈ limpias; el sigma de un nivel es el mismo para todos los casos; misma semilla → mismo archivo; las tres realizaciones difieren.
4. Desde la app (J1): añadir las etapas al YAML de n12000, correr, abrir el visor, guardar figuras.

## 11b. Notas de implementación (aditivas al contrato)

- `doe_indicators.py` copia al archivo de indicadores los attrs **raíz** del archivo multi (`noise_layout`, `snr_mode`, `snr_ref_case`, `snr_levels`, `realizations`, `seed`, `cases`, `snr_ref_power_*`), y `validate_noise.py` los pasa a su salida.
- `doe_noise.py` en modo multi sin `--out` escribe `doe_noise_multi_results.h5` **junto a `doe_results.h5`**; la etapa de la app debe pasar `--out` en `out_dir` (§7.1).
- `validate_noise.py` sin `--out` escribe `doe_noise_validation_results{gray_suffix}.h5` junto a `--noise_ind` y siempre `<out>_by_snr.csv`.
- `doe_noise.py` mantiene sus finales de línea CRLF (así estaba en el repo).
- J4 (2026-10-06): `validate_indicators.write_validation()` (núcleo reutilizable de `validate()`); `validate_noise.write_levels()` escribe los subárboles de nivel (§4.5); validación con ruido en schema `/2`; `noise_all22_r1` = r00 de 22 casos, ruido verificado bit a bit contra r00 de `noise_all22_r5` (132 copias, 528 datasets), indicadores 38 MB (solo r00), validación 43 MB.
- `validate_noise.py --realizations K [K …]` puntúa solo esas realizaciones (índice del attr `realization` de cada copia); attr raíz **aditivo** `realizations_scored` = las puntuadas, mientras `realizations` sigue siendo las del archivo de ruido.
- `doe_indicators.py`: `--only X [X …]` (prefijo o nombre de variante; error si algún X no coincide), `--resume` (salta tareas grupo × variante con attr `id` ya escrito), `--realizations K [K …]` (sufijo `__rKK`, se interseca con `--cases`).

## 12. Fuera de alcance (anotado)

- Entrenar con ruido (variante: experimento de entrenamiento ruidoso como `reference`).
- Ruido coherente entre desplazamiento y velocidad.
- Rampas con ruido (las rampas siguen diferidas).
- ROC y prueba entre indicadores por nivel de SNR.
