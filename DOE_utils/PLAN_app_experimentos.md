# Plan — App de experimentos DOE (launcher.py → app central)

Fecha: 2026-10-01 · Estado: propuesto, sin empezar

## 1. Objetivo

Una sola ventana para **configurar y seguir** cada experimento DOE de principio a fin, sin editar `.py`:

- configurar cada etapa desde un solo lugar (un YAML por experimento, con formularios para el camino principal);
- ver en un **diagrama** en qué etapa está cada experimento, qué falta y qué etapa sigue para conseguir algo concreto;
- crear experimentos **desde cero** (DOE nuevo a simular, o importar una carpeta ya simulada) o **derivando** de otro (otra `n`, otros `ap`, otras variantes de indicadores);
- evitar errores de configuración con validaciones antes de ejecutar.

Las etapas se siguen ejecutando **en consola**, como hoy. La app solo arma la configuración, lanza la consola con el comando correcto y lee el estado.

Decisiones ya tomadas (2026-10-01):

| Tema | Decisión |
|---|---|
| Etapas incluidas | Todas. Formularios solo en el camino principal; deflexión, ruido y SNR del modelo se configuran editando su sección del YAML |
| Estado "corriendo / falló" | Sí, con un envoltorio que guarda inicio, fin, código de salida y log de cada etapa |
| Launcher | `launcher.py` se convierte en la app (un solo punto de entrada) |
| Plan | Este archivo, en `DOE_utils/` |
| Interfaz | tkinter, textos en inglés (como las demás herramientas) |
| Commits | Solo cuando se pidan |

## 2. Piezas (archivos)

| Archivo | Qué es |
|---|---|
| `DOE_utils/experiment.py` **(nuevo)** | Núcleo sin interfaz: leer/guardar experimentos, registro de etapas, cálculo de estado, metas, validaciones, comandos. CLI: `status`, `run` (el envoltorio), `resolve` (configuración final de una variante para un caso), `selftest` |
| `DOE_utils/launcher.py` **(reescrito)** | La app: lista de experimentos, diagrama, panel de etapa, formularios, creación, herramientas sueltas |
| `DOE_utils/experiments/<nombre>.yaml` **(nuevos)** | Un archivo por experimento |
| `DOE_utils/experiments/indicator_variants.yaml` **(nuevo)** | Biblioteca de variantes de indicadores (sale de los `INDICATOR_CONFIG_*` de hoy) |
| `DOE_utils/experiments/.runs/` | Registros y logs de cada ejecución (ignorado por git) |

Scripts existentes que cambian (siempre manteniendo su uso actual por consola):

| Script | Cambio |
|---|---|
| `DOE_analisis/doe_indicators.py` | `--experiment PATH`: variantes, referencia y casos desde el experimento; valores físicos leídos del `.h5` por caso. El bloque `CONFIG` queda como respaldo |
| `DOE_simulacion/doe_noise.py`, `static_deflection.py`, `DOE_analisis/doe_model_snr.py` | `--experiment PATH`: su sección del YAML sobrescribe sus constantes de `CONFIG` (`doe_model_snr` además necesita la carpeta del DOE) |
| `DOE_simulacion/doe_val_planner.py` | Argumentos para que la app lo abra con el dataset y la ruta de salida ya puestos |
| `doe_runner.py`, `reference_dataset.py`, `validate_indicators.py` | Sin cambios: ya aceptan todo por línea de comandos |

## 3. El archivo de experimento

Entrenamiento (ejemplo real: el tuyo, dos corridas fusionadas):

```yaml
name: train_tube_12098
kind: training                 # training | validation
description: Tube 1DOF 150 Hz, n = 12098 rpm, kappa 0.5-2.0 + 0.91-1.09
runs:                          # una o varias corridas
  - dir: .../4_DOE_Data_Training_Tube/DOE_Training_Tube_dxl_20e-5_RUN_10_0.5-2.0    # importada (ya simulada)
  - dir: .../4_DOE_Data_Training_Tube/DOE_Training_Tube_dxl_20e-5_RUN_10_0.91-1.09
  # - config: tube_ap_sweep    # o una config de configs/ que la app puede simular
merge: {into: DOE_Training_Tube_dxl_20e-5_RUN_10_0.5-2.0, in_place: true}   # importada: se fusionó en su sitio (las nuevas van a carpeta aparte)
label:                         # mismos nombres que reference_dataset.py template
  strategy: amplitude
  amp_signal: Axial_disp
  base_attr: $f_tooth$
  base_scale: 1.0e-3
  lim_inf_pct: 10
  lim_sup_pct: 40
  warmup: 0.0
  out: reference_dataset_amp.h5
indicators:
  variants: [maxent_revo, rms_cv_revo, ssq_revo, green_lyap_revo]   # nombres de indicator_variants.yaml
  f_modal: 150.0
  cases: all
  workers: 6
# secciones opcionales (solo YAML): static_deflection, noise, model_snr
```

Validación (enlazada a su entrenamiento):

```yaml
name: val_tube_12098
kind: validation
training: train_tube_12098     # referencia + parámetros de etiquetado salen de aquí
runs:
  - config: test_validaicon    # el YAML que ya generaste con doe_val_planner
indicators: {variants: inherit}   # por defecto, las mismas del entrenamiento
validate: {channel: Axial_disp}
```

Reglas:

- `extends: <otro experimento>` hereda y sobrescribe por sección (igual que `configs/*.yaml`).
- Valores físicos (`spin_rate`, `fs`, `f_tooth`, `ap_ref`, duración de la señal) **nunca se escriben**: se leen del `.h5`, caso por caso. Así un DOE con varias `n` calcula `T_rev = 60/n` para cada caso.
- En un experimento de validación, la sección `label` **no se edita**: se copia del entrenamiento. Si alguien la cambia a mano, la app lo marca como error.

### 3.1 Dónde vive cada valor

| Dónde | Qué guarda | ¿Se escribe a mano? |
|---|---|---|
| `configs/<doe>.yaml` (existe, formato sin cambios) | Qué simular: `Ap`, `spin_rate`, `dxl_size`, `nb_dt_rev`, `ap_ref` | Sí, con `doe_planner`, `doe_val_planner` o la app al derivar |
| `experiments/<exp>.yaml` | Corridas del experimento, etiquetado, variantes elegidas, casos, workers | Sí, con formularios |
| `experiments/indicator_variants.yaml` | Biblioteca de variantes de indicadores | Solo al crear o editar una variante |
| `.h5` del DOE | `spin_rate`, `fs`, duración, `f_tooth`, `Ap`, `κ` de cada caso | **Nunca**: se lee |

El experimento no repite nada de la simulación: la carpeta (`base_dir/doe_name`) y su `doe_results.h5` se deducen de la config.

**Salidas por experimento** (hueco 3). Lo que depende de la configuración del experimento va en una subcarpeta propia, para que dos experimentos sobre el mismo DOE no se pisen:

| Salida | Dónde |
|---|---|
| carpetas de casos, `doe_results.h5`, `Out_Deflex`, `doe_noise_results.h5`, `doe_model_snr_results.h5` | carpeta del DOE (compartida: solo dependen de la simulación) |
| `reference_labels.yaml`, `reference_dataset_*.h5`, `doe_indicator_results.h5`, `doe_noise_indicator_results.h5`, `doe_validation_results.h5` (+ `_metrics.csv`) | `<carpeta del DOE>/<experimento>/` |

Un experimento importado puede apuntar a salidas que ya existen en otro sitio (ej. `label.out: <ruta>/reference_dataset_amp.h5` del entrenamiento actual); la app escribe esas rutas explícitas al importar.

**Trazabilidad** (hueco 5). Al terminar bien una etapa, el envoltorio añade al `.h5` de salida los atributos `experiment`, `experiment_stage`, `experiment_hash` (hash de la sección usada) y `experiment_date`. Así cualquier resultado se puede rastrear hasta su configuración. No hace falta cambiar los scripts.

**Derivar otra `n`** (ej. 10000 rpm): la app escribe `configs/<config>_n10000.yaml` (copia con `spin_rate` y `doe_name` nuevos; los `Ap` se recalculan con el `ap_ref` de esa `n` si se pide) y `experiments/<exp>_10000.yaml` con `extends:` y su propio `runs`. Lo que depende de `n` en los indicadores se recalcula solo (§3.2).

### 3.2 Indicadores: tres niveles

Hoy todo está en el bloque `CONFIG` de `doe_indicators.py` (`_RPM`, `_T_REV`, `_T_MODAL`, `_CUT_START`, `_CUT_END`, `_T_GT`, `_REFERENCE_H5`, `INDICATOR_CONFIG_*`, `RUNS`). Pasa a:

**Nivel 1 — biblioteca `indicator_variants.yaml`** (se escribe una vez, copiando exactamente los valores actuales):

```yaml
defaults:                        # comunes por indicador (hoy _COMMON_MAXENT, _COMMON_SSQ, _COMMON_RMS, _COMMON_GREEN_*)
  MaxEnt_SPRT: {alpha: 0.00135, beta: 0.00135, reset_on_H0: true}
  SST_SVD:     {n_fft_power: 3, mode: causal_inclusive, sigma: 6.0, alpha: 0.05, z: 3.0, fallback_mad: false}
  ...
cut_start: 0.05                  # [s] inicio del análisis (hoy _CUT_START)
variants:                        # clave = nombre del grupo en el .h5 de resultados (el mismo que hoy)
  maxent_revo_dec4_1step:
    indicator: MaxEnt_SPRT
    signal: Axial_vel
    mode: by_revolution
    window: {N_rev_window: 4, step_rev: 1}
    params: {segmentation: opr}
  green_fixed_revo_dec4_1step:
    indicator: Green_Integral
    func: Lyapunov
    signal: Axial_disp
    mode: by_revolution
    window: {N_rev_window: 4, step_rev: 1}
  ...                            # todas las INDICATOR_CONFIG_* de hoy, también las desactivadas
```

**Nivel 2 — el experimento elige** (formulario: casillas con las variantes de la biblioteca + casos, workers, `f_modal`):

```yaml
indicators:
  variants: [maxent_revo_dec4_1step, rms_cv_revo_aux4_n_aux4_dec7_1step,
             ssq_revo_aux4_n_aux4_dec7_1step, green_fixed_revo_dec4_1step]
  f_modal: 150.0                 # solo la usan las variantes by_modal
  cases: all
  workers: 6
```

En validación, `variants: inherit` toma las del entrenamiento.

**Nivel 3 — calculado por caso** (nunca se escribe), ej. `case_009`:

| Valor | De dónde | Resultado |
|---|---|---|
| `spin_rate` | `$spin_rate$` del caso | 12098.28 rpm |
| `T_rev = 60/n` | calculado | 4.959 ms (ventana de 4 rev = 19.84 ms = 800 muestras a 40327.6 Hz) |
| `T_modal = 1/f_modal` | experimento | 6.67 ms |
| `cut_end` | fin de la señal del caso | 14.0 s (hoy fijo en 16 s, por encima de la señal) |
| referencia | `label.out` del entrenamiento | `reference_dataset_amp.h5` |
| `t_theorical` / `_T_GT` | no se usa | con referencia externa no interviene en la detección (desde que se quitó `t_d_no_FAR`) |

El resultado es el mismo diccionario que hoy recibe `run_maxent_sprt` / `run_rms_cv` / `run_sst_svd` / `run_green_std`, con `T_rev` del caso.

### 3.3 Nombres de variantes únicos

Hallazgo: hoy `_run_name()` arma el nombre del grupo solo con modo, ventana y paso. Dos configuraciones que difieren en otro parámetro (`alpha`, `sigma`, `z`…) reciben el mismo nombre y la segunda **sobrescribe** los resultados de la primera en el `.h5`.

Regla nueva: el nombre del grupo es la **clave de la variante** en la biblioteca. La app no deja guardar dos variantes con el mismo nombre. "Duplicate variant" propone un nombre según la plantilla actual (`maxent_revo_dec6_1step`) y, si ya existe, pide otro.

**Variantes usadas no se editan, se duplican** (mejora 4). Una variante que ya tiene resultados en algún experimento queda bloqueada: el botón "Edit" pasa a "Duplicate to change". Así nunca cambia algo que otro experimento ya corrió, y no hace falta propagar estados desactualizados entre experimentos. Una variante sin resultados se edita libremente.

### 3.4 Vista "configuración resuelta"

En la etapa indicators, "Show resolved config for case…" muestra el diccionario final que se pasará al indicador para un caso concreto (con `T_rev`, `cut_end` y referencia ya calculados), para comprobar qué se va a correr antes de lanzarlo. Lo mismo por CLI: `experiment.py resolve <exp> <variante> <caso>`.

## 4. Etapas y estado

Registro de etapas (en `experiment.py`; añadir una etapa futura = una entrada nueva):

| Etapa | Necesita | Produce | Comando | Formulario |
|---|---|---|---|---|
| simulate | config de la corrida | carpetas de casos | `doe_runner --config X --command n2m_sch` | se abre `doe_planner` con esa config |
| extract | simulate | `<doe>/doe_results.h5` | `doe_runner --config X --command extract` | — |
| merge (si hay >1 corrida) | extract de todas | `<into>/doe_results.h5` (carpeta **nueva**) | `doe_runner --command merge --doe_name … --merge_from … --merge_out <into>` (siempre con `--merge_out`, nunca en el sitio) | sí |
| label_template | extract / merge | `<doe>/<exp>/reference_labels.yaml` | `reference_dataset.py template …` | sí |
| (revisar etiquetas, opcional) | label_template | — | botón "Abrir YAML de etiquetas" | — |
| label_build | label_template | `<doe>/<exp>/reference_dataset_*.h5` | `reference_dataset.py build …` | — |
| indicators | extract / merge + label_build **del entrenamiento** | `<doe>/<exp>/doe_indicator_results.h5` | `doe_indicators --experiment E` | sí (variantes, casos, workers) |
| validate (solo validación) | indicators + label_build propio | `<doe>/<exp>/doe_validation_results.h5` + `_metrics.csv` | `validate_indicators …` | sí (canal) |
| static_deflection (opc.) | extract | grupo `Out_Deflex` dentro de `doe_results.h5` | `static_deflection --experiment E` | YAML |
| noise (opc.) | extract | `doe_noise_results.h5` | `doe_noise --experiment E` | YAML |
| noise_indicators (opc.) | noise + label_build del entrenamiento | `<doe>/<exp>/doe_noise_indicator_results.h5` | `doe_indicators --experiment E` sobre el de ruido | YAML |
| model_snr (opc.) | simulate | `doe_model_snr_results.h5` | `doe_model_snr --experiment E` | YAML |
| view | cualquier `.h5` | — | abre `doe_unified_selector --h5 …` | — |

Colores del diagrama:

| Estado | Regla |
|---|---|
| ✓ hecha (verde) | existe la salida y el último registro terminó con código 0 |
| ⚠ desactualizada (naranja) | la salida es más antigua que alguna entrada, o la sección del YAML cambió desde la última ejecución (se guarda un hash de la sección en el registro) |
| ▶ corriendo (azul) | hay registro de inicio sin fin y el proceso sigue vivo |
| ✗ falló (rojo) | el último registro terminó con código ≠ 0 (el panel muestra el final del log) |
| · pendiente (gris) | no hay salida |
| ⊘ bloqueada | falta una entrada (la etapa previa no está hecha) |

Si una etapa queda desactualizada, todas las que dependen de ella (dentro del mismo experimento) también.

**Motivo del bloqueo, explícito** (mejora 2). Una etapa bloqueada dice qué falta y dónde, por ejemplo "Blocked: needs `label_build` of experiment `train_tube_12098`", con un botón "Go to" que abre ese experimento y su etapa. Lo mismo en `experiment.py status`.

Límites conocidos (se muestran en la ayuda de la app):
- una etapa lanzada **fuera** de la app (consola a mano) no aparece como "corriendo"; se ve como hecha cuando aparece su salida;
- la revisión del YAML de etiquetas es manual: la app no sabe si se hizo, por eso es un paso opcional.

Metas (selector "Goal"): resaltan la cadena necesaria y dicen cuál es el **siguiente paso**.

**Meta por defecto según el tipo** (mejora 1): un experimento `training` abre con "Dataset de entrenamiento" y uno `validation` con "Validación de indicadores". El siguiente paso se ve sin elegir nada; el selector solo se usa para otra meta.

| Meta | Cadena |
|---|---|
| Datos simulados | simulate → extract (→ merge) |
| Dataset de entrenamiento | … → label_template → label_build |
| Indicadores calculados | … → indicators |
| Validación de indicadores | entrenamiento listo + simulate → extract → label → indicators → validate |
| Robustez al ruido | extract → noise → noise_indicators |
| SNR del modelo | simulate → model_snr |
| Deflexión estática | extract → static_deflection |

## 5. Envoltorio de ejecución (`experiment.py run <exp> <etapa>`)

- La app abre una consola nueva con `python experiment.py run <exp> <etapa>`.
- El envoltorio arma el comando real, lo ejecuta, copia su salida a la consola **y** a `experiments/.runs/<exp>/<etapa>.log`, y escribe `<etapa>.json` con: comando, inicio, fin, código de salida, PID, hash de la sección.
- Para saber si el proceso sigue vivo en Windows se usa la API del sistema (`OpenProcess`/`GetExitCodeProcess` vía `ctypes`), **no** `os.kill`, que en Windows termina el proceso.
- `doe_runner` con `n2m_sch` usa su propio Python de Nessy2m: el envoltorio solo lanza `doe_runner` como hoy, sin cambiarle el Python.
- **Python correcto** (hueco 6): el envoltorio reutiliza `pick_python()` de `doe_planner` (el Python actual si tiene los paquetes, si no `entorno_CAMP10`, si no avisa y no lanza). No se escribe otra comprobación.
- **Sin escrituras simultáneas** (hueco 2): antes de lanzar, se miran los registros de las etapas que están corriendo (en todos los experimentos). Si alguna escribe un archivo que esta etapa lee o escribe, no se lanza y se dice cuál la bloquea. Ej.: `static_deflection` escribe dentro de `doe_results.h5`, así que mientras corre no se lanza nada que lea ese archivo.
- **Trazabilidad** (hueco 5): al terminar con código 0, añade los atributos `experiment*` al `.h5` de salida (§3.1).

## 6. La ventana

```
┌ Experiments ───────────┐┌ Goal: [Validación de indicadores ▼]   Next step: label_build ┐
│ ● train_tube_12098  ███│|                                                             │
│   training · 12098 rpm ││  [simulate]→[extract]→[merge]→[label_tpl]→[label_build]     │
│ ● val_tube_12098    █░░││                          │                │                 │
│   validation · 17 cases││                          └→[indicators]←──┘→[validate]      │
│ [New] [Derive] [Dup]   ││   optional: [deflection] [noise]→[noise_ind] [model_snr]    │
├ Tools ─────────────────┤├ Stage: label_build ─────────────────────────────────────────┤
│ planner · val planner  ││ status ✗ failed (exit 1, 19:02) · log tail …                 │
│ selector · unified …   ││ inputs ✓  outputs ✗   [Run in console] [Copy cmd] [Log]      │
└────────────────────────┘│ [Edit config] [Open output in viewer]                        │
                          └──────────────────────────────────────────────────────────────┘
```

- Lista: nombre, tipo, `n`, número de casos, descripción corta y una barra con las etapas hechas.
- Diagrama: clic en una caja → panel de la etapa. Una caja bloqueada muestra el motivo y el botón "Go to".
- Goal arranca con la meta por defecto del tipo de experimento.
- La pestaña "Tools" conserva lo que hace hoy `launcher.py` (abrir cada herramienta con argumentos).
- El estado se refresca solo cada pocos segundos y con un botón.

## 7. Crear experimentos

| Opción | Qué hace |
|---|---|
| New → simular DOE nuevo | **Asistente** (mejora 3) con lo que cambia siempre: config base (heredada de `base.yaml` o de otra config), `n`, lista o rango de `κ` (o de `Ap`) y nombre. Escribe `configs/<nombre>.yaml` y crea el experimento de una vez. "Advanced…" abre `doe_planner` para todo lo demás |
| New → importar carpeta | Eliges la carpeta del DOE; se rellenan `runs`, casos, `n` y `κ` desde su `doe_results.h5`, y las rutas de las salidas que ya existen. **Registro de partida** (hueco 4): se guarda un registro "imported" por cada salida existente, con su fecha y el hash de la sección; desde ahí solo cuentan los cambios posteriores, así nada sale en naranja por una historia que la app no conoce. Si la carpeta tiene una fusión hecha **en su sitio** (más casos que los simulados en ella, como tu 0.5–2.0 con 34), su extract queda bloqueado (hueco 1) con el aviso "re-extracting would drop the merged cases" |
| New → validación | Eliges el entrenamiento; se abre `doe_val_planner` con su dataset; al guardar, el experimento queda enlazado y con `label` heredado |
| Derive | Copia otro experimento con `extends:` y cambias solo lo que difiere: otra `n` (copia la config de simulación con el nuevo `spin_rate` y otro `doe_name`), otros `ap`/`κ`, otras variantes |
| Duplicate | Copia completa con otro nombre |
| Delete | Borra **solo el YAML** del experimento, nunca datos; pide confirmación |

**Nombres automáticos** (mejora 5): al crear o derivar, la app propone `<tipo>_<máquina>_n<rpm>_k<κmin>-<κmax>` (ej. `val_1DOF150_n12098_k0.5-2.0`) y una línea de descripción con lo mismo en palabras. Ambos se pueden cambiar; el nombre no puede repetirse.

## 8. Validaciones antes de ejecutar

Errores (bloquean el botón Run):
- rutas o configs que no existen;
- entradas de la etapa sin hacer;
- `label` de una validación distinto al de su entrenamiento;
- discretización distinta entre validación y entrenamiento (`dxl_size`, `nb_dt_rev`, `f_tooth`, canal);
- variante de indicador inexistente en la biblioteca;
- dos variantes con el mismo nombre en la biblioteca;
- otra etapa corriendo escribe un archivo que esta etapa lee o escribe (hueco 2);
- extract sobre una carpeta con fusión hecha en su sitio (hueco 1).

Confirmación explícita (no bloquea, pero hay que aceptar):
- cualquier etapa que vaya a **sobrescribir** una salida existente: la app dice qué archivo reemplaza y su fecha (hueco 1);
- `doe_name` que ya existe en la simulación.

Una variante que ya tiene resultados no se puede editar, solo duplicar (§3.3).

Avisos (no bloquean):
- `κ` de validación repetidos en el entrenamiento;
- `n` de validación distinta a la del entrenamiento (prueba de generalización);
- muchos casos `gray` en el etiquetado.

## 9. Fases (de principio a fin)

Cada fase deja algo usable y una comprobación ejecutable.

| # | Fase | Entregable | Comprobación |
|---|---|---|---|
| F1 | Núcleo `experiment.py` | Esquema, `extends`, registro de todas las etapas, comandos, estado por archivos, metas (con meta por defecto según tipo), siguiente paso, motivo de bloqueo, validaciones; CLI `status`; salidas por experimento (§3.1); registro de partida al importar; bloqueo de extract en fusiones hechas en su sitio | `selftest` con carpetas temporales; `status` sobre tu entrenamiento real importado coincide con lo que existe (extract/merge/label hechos, indicators pendiente) y **nada sale en naranja**; su extract aparece bloqueado por la fusión en el sitio; una validación sin entrenamiento listo dice qué le falta y de qué experimento |
| F2 | Envoltorio `run` | Registros `.json` + logs; estados corriendo / falló / desactualizada por hash; `pick_python()`; bloqueo de escrituras simultáneas; confirmación de sobrescritura; atributos `experiment*` en la salida | `selftest` con una etapa ficticia que termina bien, otra que falla y otra que sigue viva; una segunda etapa que escribe el mismo archivo no se lanza; la salida de la que termina bien tiene los atributos `experiment*` |
| F3 | `doe_indicators --experiment` | Biblioteca `indicator_variants.yaml` con todas las configs actuales; física por caso desde el `.h5`; nombre de grupo = clave de la variante; `experiment.py resolve` | Mismo caso corrido por `CONFIG` y por `--experiment` da los mismos `t_d` e `I_t` (con `cut_end` igualado, porque hoy es 16 s fijo); los nombres de grupo coinciden con los de hoy |
| F4 | `--experiment` en las etapas opcionales + `doe_val_planner` con argumentos | Ruido, deflexión y SNR configurables por YAML | `--dry-run`/`--list` de cada una con el experimento; selftests existentes siguen pasando |
| F5 | App: lista + diagrama + panel | `launcher.py` nuevo con colores, meta por defecto, siguiente paso, motivo de bloqueo con "Go to", Run/Copy/Log/Viewer, pestaña Tools | Abres la app con los dos experimentos reales y el diagrama coincide con `experiment.py status`; "Go to" lleva al experimento y etapa que bloquean |
| F6 | Formularios del camino principal | merge, label, indicators (casillas de variantes, Duplicate / Edit variant con variantes usadas bloqueadas, "Show resolved config"), validate; validaciones al guardar | Errores de §8 bloquean Run en un experimento hecho a propósito con fallos; duplicar una variante con nombre repetido se rechaza; una variante con resultados no se deja editar |
| F7 | Crear experimentos | Asistente de DOE nuevo (`n`, `κ`/`Ap`, nombre), importar, validación, Derive, Duplicate, Delete; nombres y descripción propuestos | Crear uno de cada tipo; el asistente y el derivado con otra `n` generan una config de simulación correcta (`doe_runner --dry-run`); el nombre propuesto sigue la plantilla y no se repite |
| F8 | Comparar experimentos | Tabla con las métricas de `/ranking` de dos validaciones lado a lado | Dos `doe_validation_results.h5` de prueba |
| F9 | Cierre | `FLUJO_DOE_utils.md` actualizado, guía corta de uso, experimentos reales migrados (entrenamiento + `test_validaicon`) | Recorrido completo con tus datos: importar entrenamiento → crear validación → ver siguiente paso → lanzar extract/label/indicators/validate (las simulaciones largas las lanzas tú) |

Orden: F1 → F2 → F3 son la base; F5 puede empezar con F1–F2 hechas. F4, F6, F7, F8 son independientes entre sí después de F5.

## 10. Riesgos y cómo se cubren

| Riesgo | Cobertura |
|---|---|
| Refactor de `doe_indicators` cambia resultados | Comparación exacta `CONFIG` vs `--experiment` (F3) antes de usarlo |
| Estado engañoso si se copian/borran archivos a mano | Botón refrescar; la regla de estado solo mira archivos y registros, documentada en la app |
| Borrar datos por error | La app nunca borra `.h5` ni carpetas; Delete solo quita el YAML |
| Perder casos fusionados al re-extraer | Fusiones nuevas siempre con `--merge_out`; extract bloqueado en fusiones hechas en su sitio; confirmación antes de sobrescribir cualquier salida |
| Corromper un `.h5` con dos etapas a la vez | El envoltorio no lanza una etapa si otra que corre escribe uno de sus archivos |
| Dos experimentos pisándose sobre el mismo DOE | Salidas dependientes de la configuración en `<carpeta DOE>/<experimento>/` |
| No saber con qué configuración se hizo un resultado | Atributos `experiment*` en cada `.h5` de salida |
| Experimentos con rutas absolutas de otra máquina | Validación de rutas al abrir; aviso en rojo |
| Scripts usados también desde otras sesiones | Todos los cambios son flags nuevos opcionales; el uso actual por consola no cambia |

## 11. Fuera de alcance (por ahora)

- Ejecutar etapas dentro de la ventana (siguen en consola).
- Fundir `doe_planner` / `doe_unified_selector` dentro de la app (se abren desde ahí).
- Formularios para deflexión, ruido y SNR del modelo (YAML).
- Cola de ejecuciones automáticas (lanzar toda la cadena sola): se puede añadir después sobre el envoltorio.

## 12. Hecho = 

- Tu entrenamiento y tu validación existen como experimentos y su diagrama refleja la realidad.
- Puedes crear una validación nueva o derivar otra `n` sin editar ningún `.py`.
- Cada etapa se lanza desde la app y su estado (hecha, corriendo, falló, desactualizada) se ve en el diagrama.
- Todos los selftests pasan.
