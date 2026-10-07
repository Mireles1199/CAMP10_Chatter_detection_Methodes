# PLAN — Ronda 3: ruido (3 realizaciones), visor con panel central, flujo "extender" y estilo unificado

**Estado (2026-10-06, noche):** plan escrito para que cada sesión (wt-interfaz, wt-validacion) lo lea en frío tras un `/clear`; el manager manda el "GO" por tarea. Nada de esta ronda está commiteado salvo lo marcado.
**Cómo retomar:** leer §0 (reglas), §1 (de dónde partimos) y la sección de tu sesión en §2; responder al manager con UNA línea ("listo" + estado de tu árbol) y esperar el GO.

---

## 0. Reglas comunes (valen para todas las tareas)

- Sin `git push`, sin ramas nuevas. Los merges hacia `REPO-Utils` los hace SOLO wt-interfaz (carpeta principal `CAMP10_Chatter_detection_Methodes`), con: árbol limpio salvo cambios ajenos del usuario, sin `index.lock`, hash+estado de los 9 experimentos idénticos, selftests de todo lo tocado y `check_app_dialogs.py`. Cada sesión hace `git merge REPO-Utils` antes de editar documentación.
- Propiedad de ficheros: wt-validacion = `DOE_analisis/*`, `DOE_simulacion/*`, `validation_figures.py`, `figure_texts.yaml`, `fig_lang`, `noise_origins.py`, `eta_compat.py`, `plot_style.py` canónico, guías/informes de validación. wt-interfaz = `experiment.py`, `launcher.py`, `doe_unified_selector.py`, `sld_model.py`, `figures_window.py`, `indicator_plots.py`, `label_grid.py`, `check_app_dialogs.py`, `doe_indicator_plotter`, `doe_noise_plotter`, YAML de experimentos, TUTORIAL. Nadie edita lo del otro: se pide por mensaje (copia al manager).
- Datos: no borrar nada sin autorización explícita del usuario (vía manager). `workers ≤ 3` en indicadores (con 6 se agotó la memoria virtual). Lecturas perezosas; sin cómputo pesado en paralelo con la corrida de T1.
- Estilo de trabajo: **ponytail** (reusar lo que existe, el cambio más corto que funcione, un selftest por lógica nueva). Figuras: skill **article-plot-style** (invocarlo con la herramienta Skill al tocar figuras).
- Python: `D:/Thesis/03-Code_Storage/02-Altintlas_Nessy2m_Storage/Env/entorno_CAMP10/Scripts/python.exe`. Finales de línea: verificar por bytes (algunos ficheros son CRLF: `sed -b -i`).
- Mensajes entre sesiones: siempre con copia al manager; un cambio de contrato se escribe primero en el plan y luego se avisa.

## 1. De dónde partimos (2026-10-06)

- `REPO-Utils` = `41b3c9c`; `wt-interfaz` y `wt-validacion` = `e7900e4` (todo fusionado, sin push). κ → η terminado (`docs/planes/PLAN_eta_rename.md`; los `.h5` existentes siguen con `kappa`, los lectores aceptan ambos).
- n12000 (`DOE_Test_1DOF_150_n12000`): 22 casos × 6 niveles (80, 60, 40, 30, 20, 10 dB) × **1 realización** (r00, seed 42, referencia `case_011`). Etapas `noise`, `noise_indicators`, `noise_validate` adoptadas ("imported"). Datos en `D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\Data\1DOF_150_Ap_Cont_test_ind\Ap_Cons_test_ind\DOE_Test_1DOF_150_n12000\`. Respaldos en `CAMP10_wt-validacion\validacion_figs\` (`noise_all22_r1`, `noise_all22_r5`, `noise`).
- Orígenes sin duplicar: attrs `source_signals`, `clean_indicators`, `noise_results`, `noise_indicators`, `clean_validation` (+ helper `DOE_utils/noise_origins.py`); validación con ruido en un solo `.h5` con un subárbol por nivel (`docs/planes/PLAN_noise_validation.md` §4.4–§4.5).
- Quiebres (exactitud balanceada): Green y SSQ 20 dB, MaxEnt 40 dB, RMS-CV 80 dB (calentamiento abierto, TNR 0.00 siempre). Con 1 realización las bandas mín–máx de `noise_metrics` no significan nada.
- Ritmo medido: ≈ 3.6 tareas de indicador por minuto con 3 workers → ≈ 2.4 h por realización (132 grupos × 4 variantes).

## 2. Tareas

| Id | Quién | Qué | Depende de | Estado |
|---|---|---|---|---|
| T1 | wt-validacion | Ampliar n12000 a 3 realizaciones (noche) | — | **lanzada 23:26** en `validacion_figs\noise_all22_r3\` |
| T2 | wt-validacion | Plan de estilo `PLAN_plot_style.md` | — | pendiente |
| T3 | wt-interfaz | Panel central Signals/I(t) en el visor de validación con ruido | — | en curso, sin commit |
| T4 | wt-interfaz | `truth` en los visores de ruido | T3 | pendiente |
| T5 | wt-interfaz (+ wt-validacion) | Flujo "extender sin rehacer" en la app | notas de T1 | pendiente |
| T6 | ambas | Implementar el estilo unificado | T2 + OK del usuario | pendiente |
| T7 | wt-interfaz | Adoptar T1 en n12000 | T1 terminada | pendiente |
| T8 | usuario | Decisiones y limpieza (§4) | — | pendiente |

### T1 — Ampliar a 3 realizaciones (wt-validacion)

- Carpeta de trabajo `validacion_figs\noise_all22_r3\`; no tocar la carpeta de datos de n12000 ni `noise_all22_r1`.
- (1) Regenerar el ruido con `realizations=3` (22 × 6 × 3 = 396 copias, seed 42) y verificar bit a bit que las 132 copias r00 son idénticas a las de `noise_all22_r1`. (2) Partir del archivo de indicadores de r1 (copiarlo) y calcular solo r01 y r02 con `--resume` / `--realizations`, workers 3, lanzador con reintentos y `SetThreadExecutionState` (el equipo se suspendió una vez). (3) `validate_noise` con las 3 realizaciones (subárboles `snr_XXX__rKK`) y figuras: las 3 de resumen y las 15 de un par de niveles para revisión.
- Anotar: tareas saltadas por `--resume`, tareas calculadas, tiempos, tamaños y toda fricción (alimenta T5).
- Mientras corre NO editar `doe_noise.py`, `doe_indicators.py`, `validate_noise.py`, `noise_origins.py` ni `eta_compat.py`.
- Estado al escribir este plan: ruido generado (3.9 GB), indicadores en curso (264 grupos faltan = 1 056 tareas; ≈ 4.5–5 h). Si se cortó: relanzar con el lanzador (reanuda).
- Hecho cuando: validación con 3 realizaciones + figuras + mensaje con hash/tiempos/tamaños y quiebres con bandas reales.

### T2 — Plan de estilo (wt-validacion)

Instrucción del usuario: *unificar el estilo; el plot style debe ser el del skill article-plot-style; borrar todo rastro de estilo antiguo o heredado; si el skill está mal, decirlo.* Escribir `docs/planes/PLAN_plot_style.md` (+ línea en `docs/README.md`) con:
- Inventario de TODAS las fuentes de estilo: 6 copias de `plot_style.py` que difieren entre sí (`DOE_utils/DOE_plots`, `indicators/{maxent_sprt,rms_cv,ssq_chatter,green_integral,emd_hht}/…/viz/`) más `Optimizacion/study_phase3/`; tamaños de letra a mano (`validation_figures.py` 7/8/9/14, `sld_model.py` 9, …); `rcParams` locales; estilos propios de las figuras de los paquetes de indicadores; `figure_texts.yaml` / `fig_lang`.
- Diferencia de cada una respecto al skill (`C:\Users\quiqu\.claude\skills\article-plot-style\SKILL.md` y `.github/skills/article-plot-style/SKILL.md`). Observación: el skill pide una fuente única "sin copias que puedan divergir".
- Propuesta: una fuente canónica (`DOE_utils/DOE_plots/plot_style.py`); para los paquetes de indicadores (instalables, no deben importar de `DOE_utils`) copias IDÉNTICAS a la canónica con un script de sincronización y un selftest que falle si divergen; tamaños de letra relativos (`small`, `x-small` o múltiplos de `font.size`) en lugar de números a mano.
- **Texto y escala.** Hoy las fuentes son puntos absolutos (12/16/14/10) y `figsize_from_scale` solo escala el lienzo, así que al exportar con escala mayor el texto queda relativamente más pequeño. El usuario espera que el texto acompañe a la talla. Propuesta: función o contexto en `plot_style` (p. ej. `rc_scaled(scale)`) y un control "texto" en la ventana de exportar, por defecto "sigue a la escala" (como un zoom) y opción "fijo" (puntos del skill). Si el skill no dice cómo escalar el texto, NO editarlo: describir la mejora propuesta y el usuario decide.
- Reparto de ficheros (§0) y fases con selftests; verificar con renders reales (`indicator_plots.py` lanza las figuras de cada paquete) que no se rompe ninguna figura ni cambia ningún resultado numérico.
- Enviar el plan a wt-interfaz y al manager; NO implementar sin el OK del usuario (T6).

### T3 — Panel central en el visor de validación con ruido (wt-interfaz)

- Problema: en el visor de `noise_validate` hay botones ("Signals of copy…", "I(t) of copy…") que abren ventanas. El usuario quiere lo de todos los demás steps: señales (`Axial_disp`/`Axial_vel`, limpias del origen y ruidosas) y curvas I(t) (con umbrales y detecciones) en el PANEL CENTRAL, pestañas Signals / I(t), seleccionando filas de la tabla (copias y sus casos "clean" seleccionables, como en `noise` y `noise_indicators`). Sin botones que abran otras figuras (quitarlos). Mantener márgenes de etiquetado 10 %/40 %, exportación (Figures…/Export…), lectura perezosa, decimación y tope de curvas. Revisar que ningún otro visor de ruido use botones donde los demás steps usan el panel central.
- Estado: hay un cambio SIN COMMIT en `wt-interfaz` (`DOE_plots/doe_unified_selector.py`; panel central para `noise_validate`, botones eliminados; selftest del visor OK). Revisarlo, completar y commitear.

### T4 — Verdad (label true) en los visores de ruido (wt-interfaz)

Columna `truth` (stable / unstable / gray) en las tablas de `noise` y `noise_indicators`, resuelta del caso limpio de origen (`clean_validation` / orígenes; en `noise_validate` ya está en `/summary`), y opción para sombrear en las gráficas de señal e I(t) los intervalos de la verdad (como el SLD y la validación limpia). Nota visible si falta el origen.

### T5 — Flujo "extender sin rehacer" (wt-interfaz, con apoyo de wt-validacion)

- Hoy un cambio de configuración (más casos, más variantes, más realizaciones, más niveles) deja la etapa "stale" y todo lo posterior sin decir cuánto falta realmente. Los scripts ya tienen `--only`, `--resume`, `--realizations`, `--cases`, `--dry_run`.
- Pedido: en la tarjeta de una etapa stale por ampliación, mostrar "missing: N de M" (solo lectura, con `--dry_run`/`--resume`) y un botón "Run missing" que corra con `--resume` (+ `--only` / `--realizations` / `--cases` según lo ampliado) sin recalcular lo hecho; las etapas dependientes quedan stale como ahora.
- Primero PROBAR en una copia con un caso pequeño (p. ej. 1 → 2 realizaciones sobre un archivo pequeño, o añadir una variante) y documentar qué ampliaciones son seguras (más realizaciones, niveles y casos añadidos al final: las semillas son anidadas y por índice de caso) y cuáles no.
- wt-validacion aporta las notas de fricción de T1 y, si falta algo en los scripts (p. ej. un conteo de faltantes), lo implementa.

### T6 — Implementar el estilo unificado (ambas, tras el OK del usuario a T2)

Por fases con hash y selftests; el reparto de ficheros está en T2/§0. Entregable adicional de wt-interfaz: control "texto" en `figures_window.py` (defecto: sigue a la escala).

### T7 — Adoptar T1 en n12000 (wt-interfaz, por la mañana)

YAML de n12000: `noise.realizations: 3`; `noise_indicators` / `noise_validate` con `resume` y, si se quiere, `realizations_run` (o sin la clave = todas); copiar a la carpeta de datos los 3 `.h5`, el CSV y `figs_noise_validation\` de `noise_all22_r3` (reescribir en las copias los attrs de origen con `noise_origins.set_origins`); renombrar los de r1 a `*.old_r1` y pedir al manager autorización para borrarlos; re-adoptar las etapas ("imported"); verificar estado, tarjetas y visor.

## 3. Orden y riesgos

- Esta noche: T1 corre sola (≈ 3 workers, ≈ 9 GB de RAM). En paralelo solo trabajo ligero (T3, T4, T2 de redacción): sin cómputo pesado ni ediciones de los ficheros de T1.
- Mañana: T7 → T5 (con las notas de T1) → T6 tras el OK a T2. T4 puede ir en cualquier momento después de T3.
- Riesgos: suspensión del equipo (corta la corrida; el lanzador reanuda), memoria (≤ 3 workers), figuras de MaxEnt/SST piden varios GB al dibujar, merges mientras el usuario edita ficheros en la carpeta principal.

## 4. Decisiones del usuario pendientes

1. ¿El `kappa` de `indicators/green_integral/examples/augmented_trajectory_exploration.py` es la misma magnitud (η)? Mientras no se decida, no se toca.
2. Borrar los respaldos de `validacion_figs\` (≈ 6.5 GB: `noise_all22_r5` y `noise`) y, tras T7, los `*.old_r1`.
3. `git push` de `REPO-Utils` (hoy ≈ 33 commits por delante de `origin/REPO-Utils`) y si se fusiona en `main`.
4. Si el skill article-plot-style debe ampliarse (escala del texto) tras el diagnóstico de T2.
5. Prueba manual de lo ya hecho (lista dada el 2026-10-06): ruido en n12000, "Indicator plots…", márgenes 10 %/40 %, exportación de señales, η en pantalla.
