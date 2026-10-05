# PLAN — Exportación configurable de toda figura de todo visor + eje κ / Ap en el SLD

Fecha: 2026-10-05 · Rama: `wt-interfaz` · Estado: **aprobado por el usuario**; fases 3, 5 y 6 esperan la autorización del manager para tocar `DOE_plots/` y `DOE_simulacion/`.

## Inventario (verificado en el código, 2026-10-05)
| Etapa | Archivo | Visor | Paneles | Cómo se guarda hoy |
|---|---|---|---|---|
| Simulate / Extract | `doe_results.h5` | `DoeSelectorUnifiedApp` | Signals, Forces, Out Deflex; SLD; convergencia | barra de matplotlib; `Save PNG` (300 dpi fijo); `Figures…` |
| Label template / build | `reference_dataset_*.h5` | `ReferenceViewerApp` (tramos / combinado) | señal, distribución, rejilla | `💾 Export figure` (300 dpi fijo) |
| Label build | ídem | `label_grid.py` | rejilla / histogramas | barra de matplotlib |
| Indicators | `doe_indicator_results.h5` | `DoeSelectorUnifiedApp` | Signals, I_t(t); SLD; t_d, overlay I(t) | barra; `Save PNG`; `Figures…` |
| Validate | `doe_validation_results.h5` | `DoeSelectorUnifiedApp` | + 12 de validación, SLD por outcome | barra; `Save PNG`; `Figures…` |
| Validate (Compare) | 2 × validación | ventana *Plot* del launcher | A/B | se guarda sola (png 300 dpi) |
| Noise / Noise ind. / Model SNR | `doe_noise*.h5`, `doe_model_snr_results.h5` | `DoeSelectorUnifiedApp` | señales, I(t), overlays, SNR | barra; `Save PNG` |
| (Tools) | — | `doe_planner.py`, `doe_val_planner.py` | SLD con casos / κ | barra de matplotlib |


## Context
Hoy las figuras se ven en los visores de cada etapa, pero solo algunas se pueden guardar y casi ninguna con la
configuración de la ventana nueva `Figures…` (idioma, escala, dpi, formato). El usuario quiere que **toda figura de
todo visor de etapa** (más `doe_planner` y `doe_val_planner`) se pueda guardar con esa configuración, desde un **botón
general por visor**, también en los visores con varios paneles. Además quiere elegir en el SLD del panel derecho si el
eje vertical es **κ o Ap**, con Ap rotulado FR « Largeur de coupe $a_p$ [mm] » / EN « Width of cut $a_p$ [mm] ».

Decisiones del usuario (2026-10-05): κ/Ap a elegir (κ = Ap / límite del SLD a esa velocidad, límite = línea κ = 1);
inglés *Width of cut*; **idioma EN/FR/both en todas las figuras**; alcance = visores de las etapas + `doe_planner` +
`doe_val_planner` (no `doe_selector` ni el selector del SLD de *New experiment*).

## Enfoque

### 1. Una sola ventana de exportación — generalizar `DOE_utils/DOE_plots/figures_window.py` (mío, ya existe)
- Elementos `(nombre, kind, render)`:
  - **generada**: `render(language, scale) -> Figure` (validación, SLD, convergencia, t_d…, como hoy);
  - **panel vivo** (Signals, Forces, I(t), Out Deflex, tramos, rejilla de etiquetas, Compare, planner): copia
    independiente de la figura del panel con `pickle.loads(pickle.dumps(fig))` (probado: conserva selección y zoom, no toca
    el panel, se redibuja a otro tamaño). Si no se deja copiar: se exporta tal cual, sin vista previa.
- Controles para todos: **tamaño** (*como en pantalla* | `FIGSIZE_SIMPLE` | `FIGSIZE_WIDE` | rejilla n×m) × **escala**,
  proporciones de artículo, **idioma** EN/FR/both, **dpi** 200/300/600, **formato** png/pdf/svg, **carpeta + nombre**
  (por defecto los de hoy: `figs_validation/`, `figs_indicators/`, carpeta de la referencia…), Save / Save all / Open folder.
  Errores (sin datos, fallos al dibujar) en la barra de estado (ya existe).
- Reusar: `plot_style.figsize_from_scale`, `figsize_grid`, `FIGSIZE_SIMPLE/WIDE`, `lang_text`.

### 2. Idioma en todas las figuras sin reescribir los plotters — traducción al exportar
- Nuevo `DOE_plots/fig_lang.py`: `translate_figure(fig, language)` recorre los textos de la **copia** (títulos, ejes,
  leyendas, `suptitle`, anotaciones) y los sustituye según una tabla; `both` usa el formato de `lang_text`
  (`[EN] …\n[FR] …`, leyendas con ` / `). Los textos en pantalla no cambian.
- Tabla `DOE_plots/figure_texts.yaml`: texto tal como está en el código (hoy mezcla ES/EN: « Señales: disp », « Convergencia
  RMS », « todos », « modo ») → `{en, fr}`; coincidencia exacta y luego por fragmentos para textos con valores
  (« κ=0.5 | run »). La relleno escaneando los ~170 textos fijos (`doe_unified_selector` 59, `doe_indicator_plotter` 41,
  `doe_noise_plotter` 28, `doe_plotter` 23, `doe_model_snr_plotter` 15, planners, `label_grid`).
- Textos sin traducción: se avisan en la barra de estado y *Save all* los lista para completar la tabla.
- `validation_figures` y `sld_model` ya dibujan en el idioma elegido (su `LANGUAGE`): no se traducen.

### 3. Botón **Export…** en cada visor (abre la ventana del §1 con todos sus paneles)
| Visor | Elementos |
|---|---|
| `DoeSelectorUnifiedApp` (`doe_unified_selector.py`; todos los tipos de `.h5`) | Signals, Forces, I(t), Out Deflex (los que haya), panel derecho actual, todos los SLD, todas las de `Figures…` |
| `ReferenceViewerApp` (mismo archivo; tramos / combinado) | señal, distribución, rejilla |
| `label_grid.py` | página actual y todas las páginas |
| Compare (`launcher.py`, `plot_compare`) | la figura A/B (ya no se guarda sola) |
| `DOE_simulacion/doe_planner.py`, `doe_val_planner.py` | su figura (SLD con casos / κ) |
`Save PNG`, `💾 Export figure` y `Figures…` abren la misma ventana con su panel preseleccionado (una sola manera de guardar).
Reusar: `_make_summary_figure`, `_is_validation_h5`, `_sim_models`, `_fig_style`/`_apply_fig_style` (`doe_unified_selector.py`).

### 4. SLD con eje κ / Ap — `DOE_plots/sld_model.py` + panel derecho
- `plot_sld(..., y_axis="Ap"|"kappa")`. κ: lóbulos divididos por el límite del preset a cada velocidad (`ap_lim`,
  `lobes` ya existen), línea κ = 1, cada caso en su κ (rampas: segmento κ_start → κ_end, `case_points`).
- Etiquetas: Ap → EN « Width of cut $a_p$ [mm] » / FR « Largeur de coupe $a_p$ [mm] »; κ → « $\kappa = a_p / a_{p,\lim}(\Omega)$ [–] ».
- Casilla **κ axis** en el panel derecho (redibuja el SLD actual) y opción en la ventana de exportación para las entradas SLD.

## Fases (un commit cada una; LF verificado por bytes, CRLF en `doe_unified_selector.py`)
0. Actualizar `DOE_utils/PLAN_figuras.md` con estas decisiones y **pedir al manager** la autorización para tocar
   `DOE_plots/` (`doe_unified_selector`, `sld_model`) y `DOE_simulacion/` (planners). No tocar esos archivos hasta el OK.
1. Ventana genérica (tamaño por preset, carpeta/nombre, elementos vivos por pickle).
2. `fig_lang.py` + `figure_texts.yaml` (tabla rellenada) e idioma en la ventana.
3. Export… en `DoeSelectorUnifiedApp` y `ReferenceViewerApp`; botones existentes → misma ventana.
4. `label_grid.py` y Compare por la ventana.
5. SLD κ / Ap + etiquetas *Width of cut / Largeur de coupe*.
6. Planners (`doe_planner`, `doe_val_planner`).
7. `CONTRATO_interfaz.md`, `TUTORIAL.md`, mensaje al manager.

## Verificación
- Selftests: `experiment.py selftest`, `launcher.py --selftest`, `DOE_plots/doe_unified_selector.py --selftest`,
  `validation_figures.py --selftest`, `label_grid.py --selftest`, `check_app_dialogs.py` (añadir: botón Export… presente,
  ventana con N elementos, guardado de un elemento vivo y uno generado).
- Selftest nuevo de `fig_lang.py`: figura sintética con textos de la tabla → EN / FR / both correctos; texto ausente → avisado.
- Por visor, sobre archivos reales en **solo lectura** (guardando en el scratchpad): abrir, *Save all* en png y pdf, EN y FR,
  SIMPLE y WIDE × 1.5; comprobar tamaños en pulgadas del archivo, captura de cada ventana, y el SLD en κ y en Ap con
  `n12000` (schema 4) y `ramp_check`.
- Python: `D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\Env\entorno_CAMP10\Scripts\python.exe`. Commits locales en
  `wt-interfaz`, sin push.
