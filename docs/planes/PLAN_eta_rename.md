# PLAN — Renombrar κ (kappa) → η (eta)

**Estado (2026-10-06):** lado wt-validacion HECHO: Fase 1 (lectores, eta_compat.py) 8165b7c/a8115e2/8b31502; Fase 2 (escritores) 91f3a87/ca197f1; Fase 3 (textos, figuras score_vs_eta / delay_vs_eta, docs de validación) 557f885/08c3a85. Quedan en mi parte solo alias, lectores de ambos nombres y selftests con archivos viejos. Pendiente: lado wt-interfaz (su Fase 2/3) y `indicators/*` (decide el usuario). Contrato original:  Decisión del usuario: lo que el código y las figuras llaman κ (kappa) es en realidad η (eta). Coordina: manager (`DOE_utils/PLAN_app_v2.md`). Reparto: **wt-validacion** = scripts, figuras y documentos de validación; **wt-interfaz** = app, visor, SLD, formularios, tutorial.

## 1. Qué es y qué NO se toca

- η = Ap / AP_REF (adimensional; hasta hoy "kappa"). Texto de pantalla `η`; LaTeX `$\eta$`; atributo / columna / clave `eta`.
- **No se reescriben los .h5 de datos existentes** (cientos de MB; los de simulación y los reference_dataset llevan `kappa`): todos los LECTORES aceptan ambos nombres.
- **No se tocan** binarios (figuras PNG/PDF antiguas, PPTX, PDF de referencias) ni `Data_Exported/*.h5`. Los PNG de `Optimizacion/` y `Convergency_Simulation/Latex/` quedan como están (salida histórica).
- Documentos de `docs/historico/` (PLAN_ramps, PLAN_figuras, PLAN_app_experimentos) y el INFORME de rampas: **se dejan con "kappa"** (son historia); se añade una línea de equivalencia en `docs/README.md`.
- Paquetes `indicators/*`: solo aparece `indicators/green_integral/examples/augmented_trajectory_exploration.py` (19 apariciones). **No se toca** sin acuerdo del manager. (`Optimizacion/` usa "K" para otra cosa; no es kappa.)

## 2. Contrato de nombres

| Qué | Antes | Después (escritores nuevos) | Lectores |
|---|---|---|---|
| attr de caso (simulación, `doe_results.h5`) | `kappa`, `kappa_start`, `kappa_end` | `eta`, `eta_start`, `eta_end` | ambos, prioridad `eta` |
| attrs de pieza (reference_dataset) | `kappa`, `kappa_start`, `kappa_end`, `kappa_t0`, `kappa_t1` | `eta`, `eta_start`, `eta_end`, `eta_t0`, `eta_t1` | ambos |
| validación, attrs de caso | `$kappa$`, `$kappa_start$`, `$kappa_end$` | `$eta$`, `$eta_start$`, `$eta_end$` | ambos |
| validación, `/summary/<run>` y CSV | columnas `kappa`, `kappa_start`, `kappa_end` | `eta`, `eta_start`, `eta_end` | ambos |
| validación con ruido `/summary` | columna `kappa` | `eta` | ambos |
| estrategia de etiquetado | valor `kappa` (`--strategy kappa`, `labeling_strategy`) | `eta` (alias: `kappa` sigue valiendo) | ambos |
| umbral | `--kappa-threshold`, `DEFAULT_KAPPA_THRESHOLD` | `--eta-threshold`, `DEFAULT_ETA_THRESHOLD` (alias vivos) | ambos |
| etiquetas de figura | `κ`, `$\kappa$`, "kappa" | `η`, `$\eta$`, "eta" | — |
| nombres de figura (registro `FIGURES`) | `score_vs_kappa`, `delay_vs_kappa` | `score_vs_eta`, `delay_vs_eta` (+ `FIGURE_ALIASES` viejo→nuevo para quien los pida; el registro solo lleva los nuevos) | — |
| claves YAML de experimentos | cualquiera con `kappa` (`kappa_range`, listas…) | `eta…` | ambas; la vieja y la nueva son **equivalentes** (huella, ver 4) |

Los nombres internos de funciones/variables con "kappa" se renombran solo si es barato y sin romper la API entre módulos; si un nombre público lo usa otro módulo, se deja un alias (`kappa_txt = eta_txt`). Mensajes de log y docstrings: "eta".

## 3. Helper común — `DOE_utils/eta_compat.py` (solo stdlib; sin h5py obligatorio)

Lo importan los lectores de **ambos** lados. Un solo sitio para la regla "acepta los dos, manda `eta`".

- `get(d, name, default=nan)`: `name` empieza por `eta` o `$eta` (`"eta"`, `"eta_start"`, `"$eta$"`…). Devuelve `d[name]` si existe, si no `d[name con kappa]`, si no `default`. `d` = attrs de h5py, dict o cualquier objeto con `in` y `[]`.
- `has(d, name)`.
- `col(group, name)`: lo mismo para datasets/columnas de un grupo HDF5 o dict de arrays (`/summary`).
- `old_name(name)` / `canon(key)`: `eta…`↔`kappa…` (solo el token; `meta`, `beta`, `delta` no se tocan).
- `canon_keys(mapping)`: copia recursiva con las claves `kappa…` renombradas a `eta…` (para la huella: la vieja y la nueva dan el mismo hash).
- `LABEL = "η"`, `TEX = r"$\eta$"`.
- Selftest: attrs viejos (`kappa`), nuevos (`eta`), ambos (gana `eta`), ninguno (default), `meta`/`beta` intactos, `canon_keys` recursivo.

## 4. Huella y estados (wt-interfaz)

El renombrado **no debe cambiar** la huella ni el estado de los experimentos existentes (los 9 deben quedar idénticos). Si una clave renombrada entra en la huella, se hashea con `eta_compat.canon_keys(...)` (vieja ≡ nueva). Texto de pantalla: no entra en la huella. Los YAML existentes **no se reescriben** automáticamente; los nuevos usan `eta`.

## 5. Orden (para no romper a nadie)

1. **Fase 1 — lectores** (ambos lados, en paralelo): `eta_compat.py` (yo, primero: aviso con hash), y cada lector pasa a `eta_compat.get/col`. Selftests con archivos "viejos" (kappa) y "nuevos" (eta). **Ningún escritor cambia todavía.**
2. **Fase 2 — escritores** (cuando wt-interfaz confirme que sus lectores están listos y mergeados): `doe_runner` (attrs de caso), `reference_dataset` (piezas), `validate_indicators`/`validate_noise` (atributos y columnas), CSV.
3. **Fase 3 — textos**: figuras (ejes, leyendas, títulos, `figure_texts.yaml`, `fig_lang.KEEP`), docs de validación (`GUIA_*`, `INFORME_validacion`, `PLAN_noise_validation`, `FLUJO`), `docs/README.md`; interfaz/tutorial/diálogos: wt-interfaz.
4. Cada fase = commits pequeños + aviso con hash a wt-interfaz y al manager. Sin push, sin ramas nuevas.

## 6. Reparto de ficheros

**wt-validacion:** `DOE_analisis/{validate_indicators, validate_noise, doe_indicators, check_indicators_experiment}.py`, `DOE_simulacion/{doe_runner, doe_planner, doe_val_planner, reference_dataset, doe_noise}.py` + sus YAML/`configs` (comentarios), `DOE_plots/{validation_figures, figure_texts.yaml, fig_lang}.py`, `noise_origins.py` (0 apariciones), `eta_compat.py`, `docs/{guias,informes,planes,flujos}` de validación.
**wt-interfaz:** `DOE_utils/{experiment, launcher, label_grid, check_app_dialogs}.py`, `DOE_plots/{doe_unified_selector, sld_model, indicator_plots}.py`, `experiments/*.yaml`, `TUTORIAL.md` y `tutorial_img` (imágenes: se regeneran si se quiere), `docs/contratos`, `docs/planes/PLAN_app_v2.md`.
Cualquier fichero que aparezca y no esté aquí: se pregunta antes.

## 7. Magnitud (medida en archivos de texto versionados)

~690 apariciones en 36 ficheros: `experiment.py` 136, `launcher.py` 60, `doe_unified_selector.py` 53, `reference_dataset.py` 52, `validation_figures.py` 39, `check_app_dialogs.py` 37, `doe_val_planner.py` 23, `doe_runner.py` 17, `validate_indicators.py` 16, `doe_indicators.py` 16, `doe_noise.py` 12, `validate_noise.py` 6, `TUTORIAL.md` 22, `GUIA_metricas_validacion.md` 25.

## 8. Verificación

Selftest de cada módulo con datos viejos y nuevos; los 9 experimentos con huella/estado idénticos; una validación n12000 regenerada con escritores nuevos == la actual salvo `kappa`→`eta`; `git grep -i "kappa\|κ"` en los ficheros de cada reparto solo deja alias/lectores y `docs/historico`.
