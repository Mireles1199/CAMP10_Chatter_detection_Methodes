# Plan v2 — App de experimentos DOE: generalizar y mejorar la experiencia

Fecha: 2026-10-02 · Estado: **pendiente de 4 decisiones (§4)**, nada implementado todavía.
Plan anterior (v1, implementado): `DOE_utils/PLAN_app_experimentos.md` (bitácora §13, filas F0–F9, R2, R3).
Código: `DOE_utils/launcher.py` (ventana), `DOE_utils/experiment.py` (núcleo), `DOE_utils/experiments/` (YAML de
experimentos y `indicator_variants.yaml`), `DOE_utils/TUTORIAL.md` (tutorial, también pestaña de la app).
Rama: `Aplication-Indicateur-Validacion-Training` (último commit de v1: `cfa314e`, sin push).

## 1. Qué pidió el usuario (revisión del 2026-10-02, resumida)

Idea general: la app gusta, pero está **demasiado orientada a entrenamiento y validación**. Debe servir para
cualquier flujo de trabajo (entrenamiento y validación serían solo un caso particular), ser más fácil de editar y
de crear casos nuevos **sin tener que tocar YAML**, y dejar claro qué config, qué entradas y qué salidas usa
cada cosa.

Puntos concretos (todos):

1. No solo training / validation: que sirva para los distintos flujos.
2. Mejorar la modificación de las distintas configuraciones; no estar obligado a editar YAML.
3. El botón **Viewer** no funciona.
4. No funciona si se elige **un solo `Ap`**.
5. **New DOE** se basa en una config ya hecha: quiere crear **desde cero**, con más personalización. (Eso también
   provocó que configuraciones de indicadores quedaran mal.)
6. Si se elige `ap_ref` del SLD y un modo, la config no queda en `mode: model` / `model_at_spin` con su modelo.
7. El campo **"training (validation only)"** no deja elegir nada.
8. Nombre propuesto: no forzar `DOE_`; puede ser un solo caso, no un DOE.
9. **Edit config** de la simulación no hace nada.
10. No siempre se sabe qué config usa un experimento.
11. Las configs heredan de `base.yaml` (bien), pero al crear/editar se debería **reescribir explícito** todo lo que
    se hereda, para ver qué config se usa realmente.
12. Elegir fácilmente **dónde se guardan** los `.h5`, las carpetas de referencia y dónde están los datos.
13. Las variantes de indicadores están bien, pero es pesado declararlas, encontrarlas y modificarlas.
14. No se entiende para qué sirve el botón **Show**.
15. No está claro qué **entrada** y qué **salida** tiene cada etapa.
16. New DOE dio `Ap_start`/`Ap_end` = **−inf** y la etapa Simulate no marcó ningún error: tuvo que editar el YAML.
17. Robustez cuando no se declaran cosas (p. ej. `ap_ref`): quiere un **dry-run** que valide antes.
18. ¿Para qué sirve el log?
19. Al borrar un experimento, ¿se borran sus `.runs`? (Revisar: debería.)
20. El panel (comando / barra de progreso) **solo se actualiza con Refresh**.
21. No se ve dónde activar `--timed` ni `--auto-extract`.
22. Una etapa con archivo importado aparece como **pending**: indicarlo o resolverlo.
23. ¿Aún se necesita `t_gt` (`t_theorical` / `_T_GT`) en la declaración de indicadores?
24. ¿Qué pasa si se corre Indicators sin la etapa de etiquetado?
25. ¿Por qué sale "Advertencia: varianza muy pequeña, aplicando floor para evitar sigma=0." con MaxEnt?
26. Cada indicador muestra un resumen al final: mejor mostrarlo **de forma sistemática al terminar cada caso**.
27. Más UX en general; que no sea exclusivo de validación y entrenamiento.
28. El `.h5` final de indicadores **no está conectado a κ ni a la etiqueta** (versión antigua de `doe_indicators.py`).
29. En el visor del dataset de referencia sale κ; quiere ver también **`Ap`**.
30. Hay **restos de `t_gt`** al visualizar el `.h5` de indicadores.
31. El estado **stale** funciona demasiado bien: cambios mínimos lo activan aunque apunten a la misma carpeta.
32. Revisar la manera en que se guardan los distintos archivos (organización de salidas).
33. Visor del `.h5` de indicadores: poder **activar/desactivar la ley normal de entrenamiento** y, si se lee un
    dato de entrenamiento, que se indique.
34. ¿Que la consola **se cierre sola** al terminar una etapa? (opción)
35. Una forma **más amigable de ver el log**.
36. No se entiende el botón **Derive** ni su utilidad.
37. No está claro **cuándo se edita la configuración y cuándo el experimento**.
38. ¿Por qué hay un YAML de config y otro de experimento?
39. Especificar **qué canal** se usa para etiquetar el entrenamiento y **qué canal** se analiza.
40. El visor del DOE: a veces no se sabe si se está abriendo o no.
41. En archivos de indicadores y derivados, qué es "Axial" (nombres de canal poco claros).
42. En los DOE de validación y entrenamiento no queda claro **cuál es la etiqueta verdadera, qué estrategia se usó**
    ni qué dice el indicador cuando se usa el visor.
43. Poder **crear el YAML de experimento para `.h5` importados** (calculados fuera de la app, sin el formato que
    exige la app) para **estandarizarlos**.
44. Añadir a los parámetros de los casos **con qué modelo se simuló** (atributo), sobre todo para `.h5` antiguos
    que no lo tienen (va con el punto 43).

## 2. Causas ya identificadas (para no rediagnosticar)

| Punto | Causa | Arreglo |
|---|---|---|
| 3, 9 | `launcher.launch(kind, script, args)` recibe los argumentos como **texto** y los parte con `shlex.split(posix=False)`, que **deja las comillas** dentro del argumento: el visor y `doe_planner` reciben `"D:\…"` con comillas y no encuentran el archivo | Que `launch` acepte una **lista** de argumentos; usarla desde la app (`open_output`, `edit_stage`, `NewValidationDialog`) |
| 20 | `App.redraw` solo redibuja si cambia el *snapshot* (estados); el progreso y el log no forman parte de él | Añadir al snapshot el progreso (`stage_progress`) y la fecha del log de la etapa seleccionada |
| 16 | `NewDoeDialog`/`new_doe_config` no comprueba que los `Ap` sean finitos y > 0; `sld_model.ap_lim` devuelve `inf` en un hueco entre lóbulos | Validar `Ap` (finito, > 0, sin duplicados) antes de escribir; avisar del hueco del SLD |
| 6 | `new_doe_config` siempre escribe `ap_ref: {mode: manual}` | Permitir `model` / `model_at_spin` con su `model:` (preset de `sld_model.MODELS`) |
| 4 | Sin reproducir aún: un solo `Ap` (barrido de longitud 1) | Reproducir con `doe_runner --dry-run` y arreglar |
| 7 | Combobox "training (validation only)" en `NewDoeDialog`/`ImportDialog` | Revisar estado/valores; desaparece si se generaliza (§4 decisión 1) |
| 31 | La huella (`Stage.hash`) incluye rutas como texto y la config completa leída por `doe_runner` (con rutas normalizadas distinto según el caso) | Normalizar rutas (`os.path.normcase(abspath)`) y hashear solo lo que cambia el resultado |
| 23 | `indicator_variants.yaml` conserva `t_theorical`, `t_stable_total`, `cut_end_time` del CONFIG viejo; con referencia externa no se usan | Quitarlos de las variantes (verificar con `check_indicators_experiment.py` que los resultados no cambian) |
| 28, 30 | `doe_indicators.write_results` copia los attrs del caso (que traen `kappa`), pero no la etiqueta; el visor/plotter aún usa `T_GT` fijo | Escribir en cada caso `kappa`, `Ap`, etiqueta y estrategia; quitar `T_GT` de `doe_indicator_plotter` / `doe_unified_selector` |
| 19 | `experiment.delete()` sí borra `experiments/.runs/<nombre>`; verificar el caso que vio el usuario | Comprobar y, si falla, corregir |
| 25 | `MaxEnt_SPRT`: un tramo estable con entropía casi constante da σ ≈ 0; se aplica un mínimo y se imprime una línea por segmento | No es un fallo: contarlo y mostrarlo una vez en el resumen del caso |
| 24 | Indicators necesita el dataset de etiquetas como referencia (umbrales); sin él no puede entrenar | Explicarlo en el panel; con generalización, la referencia puede ser cualquier dataset elegido |

Funciones/archivos relevantes: `experiment.py` → `stages()`, `status()`, `_stage_state()`, `next_step()`,
`run_blockers()`, `new_doe_config()`, `derive()`, `import_dir()`, `stage_summary()`; `launcher.py` → `launch()`,
`App.redraw()`, `App._show_stage()`, `NewDoeDialog`, `IndicatorsForm`, `render_markdown()`.

## 3. Propuesta (a confirmar con §4)

### 3.1 Fallos (sin decisiones; primero)
Puntos 3, 4, 6, 7, 9, 16, 19, 20, 23, 31, más el control de `Ap` y un **dry-run** (17): antes de guardar o lanzar,
`doe_runner --dry-run` sobre la config y mostrar los casos que saldrían; bloquear Run si falla.

### 3.2 Generalizar (puntos 1, 27, 37, 38, 10, 11, 12, 36, 14)
- Experimento = un DOE (o un solo caso) + las **etapas que el usuario active**; entrenamiento y validación pasan a
  ser **plantillas de flujo**. Un experimento puede apuntar opcionalmente a otro como **referencia** (dataset para
  los indicadores, verdad para validar).
- Config de simulación **explícita**: todo lo heredado de `base.yaml` escrito; ver en la app qué config se usa.
- Elegir en un formulario **dónde se guardan** las salidas y dónde están los datos / la referencia (12, 32).
- Quitar o renombrar **Show** (14) y **Derive** (36) → p. ej. "Copy with changes" con explicación.
- Distinguir claramente "editar la configuración de simulación" vs "editar el experimento" (37, 38).

### 3.3 Crear y editar sin YAML (2, 5, 8, 13, 21, 34)
- **New** desde cero, formulario completo: carpeta, caso, `n`, lista/rango de `Ap` **o** de κ (también 1 caso),
  `ap_ref` manual o por modelo (`model`, `model_at_spin` + preset), discretización (`dxl_size`, `nb_dt_rev`,
  `f_tooth`), `nb_proc`, señales a extraer; nombre libre (sin `DOE_` forzado). Plantilla opcional solo para
  precargar valores.
- Opciones de ejecución en Simulate: `--timed`, `--auto-extract`, y "cerrar la consola al terminar".
- Indicadores más ligeros (13): tabla editable (indicador, señal, modo, ventana, paso) + presets; parámetros raros
  en un desplegable.

### 3.4 Información y visor (15, 18, 22, 26, 29, 33, 35, 39–42)
- Cada etapa muestra **entradas y salidas** con su ruta y estado, y **qué canal** usa (etiquetado vs análisis).
- Etapas importadas: estado propio "imported" en vez de "pending".
- Visor de log integrado (coloreado, búsqueda, se actualiza solo) en vez de abrir el bloc de notas.
- Resumen de indicadores **por caso**, sistemático, al terminar cada uno.
- Visor (`doe_unified_selector`): mostrar `Ap` además de κ; quitar `t_gt`; activar/desactivar la ley normal de
  entrenamiento; marcar cuándo el `.h5` es de entrenamiento; mostrar etiqueta verdadera y estrategia; indicar que
  se está abriendo.

### 3.5 Estandarizar datos externos (43, 44)
- Asistente "Standardize / import any .h5": detecta el formato, crea el YAML del experimento y completa atributos
  que falten en los casos (p. ej. **modelo de simulación**, `kappa`, `Ap`, etiquetas), sin tocar los datos
  originales salvo atributos añadidos con confirmación.

## 4. Decisiones pendientes (preguntar al empezar la próxima sesión)

1. **Generalizar**: (a, recomendado) sin tipos: cada experimento elige sus etapas y puede apuntar a otro como
   referencia; entrenamiento/validación = plantillas de flujo. (b) mantener tipos y añadir "general".
2. **Archivos**: (a, recomendado) un archivo por experimento con la simulación completa y explícita dentro; la app
   genera por debajo la config de `doe_runner`. (b) dos archivos (configs/ + experimento), ambos explícitos y
   editados en el mismo formulario.
3. **Indicadores**: (a, recomendado) tabla en el experimento + presets; la biblioteca deja de verse. (b) mantener la
   biblioteca con buscador y edición en formulario.
4. **Cómo proceder**: (a, recomendado) arreglar primero los fallos de §3.1, luego el resto en bloques con un commit
   por bloque, decisiones menores anotadas en una bitácora al final de este archivo. (b) solo los fallos y revisar.

Nota: el usuario pidió aclarar estas preguntas antes de contestarlas; empezar la sesión preguntando qué quiere
aclarar.

## 5. Bitácora v2

| Bloque | Decisión | Por qué |
|---|---|---|
| — | (vacía) | |
