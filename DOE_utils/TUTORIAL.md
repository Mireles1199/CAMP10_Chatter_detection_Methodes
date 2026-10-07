# Tutorial: app de experimentos DOE

Se abre con `python launcher.py` (Python de `entorno_CAMP10`). Esta guía se lee dentro de la app (pestaña **Tutorial**) y también como archivo: `DOE_utils/TUTORIAL.md`. Usa el índice de la izquierda para saltar a una sección y **A+ / A−** para cambiar el tamaño del texto.

> En una frase: cada experimento es un solo archivo con todo lo que usa; la app te dice en qué etapa está, qué falta y cuál es el siguiente paso; tú pulsas un botón y la etapa corre en una consola aparte.

## 1. La idea

Un **experimento** es un DOE (o un solo caso) y las **etapas** que tú activas. Todo vive en un archivo, `experiments/<nombre>.yaml`: la simulación escrita completa (nada heredado), las etapas, el etiquetado, la tabla de variantes de indicadores y dónde se guardan las salidas.

No hay "tipos". Entrenamiento y validación son solo dos **flujos** posibles:

| Flujo | Etapas | Para qué |
|---|---|---|
| Simulation only | Simulate, Extract | Simular y tener las señales (un estudio de convergencia, un solo caso…) |
| Labelled dataset (training) | + Label template, Label build | Un dataset etiquetado: la verdad, y lo que aprenden los indicadores |
| Indicators | + Indicators | Correr los indicadores sobre esos casos |
| Validation against a reference | + Validate | Medir los indicadores contra otro experimento |

Un experimento puede apuntar a otro como **referencia**: el dataset etiquetado de la referencia es con lo que aprenden los indicadores, y sus etiquetas se calculan con los mismos parámetros que la referencia. Sin referencia, el experimento aprende de su propio dataset.

| Etapa | Qué hace | Entrada | Salida |
|---|---|---|---|
| Simulate | Corre Nessy2m para cada caso | la simulación del experimento | una carpeta por caso |
| Extract | Junta los casos en un archivo | carpetas de casos | `doe_results.h5` (señales + `η`, `Ap`, `n`…) |
| Label template | Propone estable / gray / inestable por caso | `doe_results.h5` | `reference_labels.yaml` (revísalo) |
| Label build | Corta las señales según esas etiquetas | datos + YAML de etiquetas | `reference_dataset_*.h5` (la verdad) |
| Indicators | Corre cada variante sobre cada caso | datos + dataset etiquetado de la referencia (o el propio) | `doe_indicator_results.h5` |
| Validate | Compara detecciones con la verdad | resultados de indicadores + dataset etiquetado propio | `doe_validation_results.h5` |
| Static deflection, Noise, Noise indicators, Model SNR | Etapas opcionales | | cada una, su archivo |

> Las etapas **no corren dentro de la app**: se abren en una consola aparte. Cerrar la app no las detiene y, al volver a abrirla, el estado se recupera solo.

## 2. Leer la pantalla

![Pantalla principal: un entrenamiento con la etapa Indicators seleccionada](01_overview.png)

- **Izquierda**: los experimentos y su flujo. Verde = meta alcanzada, azul = algo corriendo, rojo = una etapa falló.
- **Arriba a la derecha (tarjeta)**: flujo, `n`, casos, `η`, etapas hechas y, en gris, **de dónde sale todo**: cada corrida (simulación escrita en el experimento, config o carpeta importada) y su carpeta de datos, la carpeta de salidas del experimento y el archivo. En azul, su referencia y de quién es referencia.
- **Goal**: lo que quieres conseguir. Empieza en la etapa más lejana que activaste. **Next step** dice la siguiente etapa que falta; **Select that stage** la selecciona y **▶ Run next step** la lanza. **▶▶ Run to goal** corre, en una sola consola, todo lo que la meta necesita, una etapa detrás de otra, **también las de la referencia**: desde un test cuyo entrenamiento aún no se ha corrido, simula, extrae y etiqueta primero el entrenamiento y después hace lo del test hasta Validate. Antes te enseña la lista. La casilla **stop after Label template (review)**, junto al botón, decide si se para tras cada Label template para que revises las etiquetas (vuelve a pulsarlo y sigue) o corre de seguido; se recuerda entre sesiones. Se para si una etapa falla. Vale igual para etapas naranjas (*stale*): si cambiaste algo del entrenamiento, rehace lo que haga falta del entrenamiento y del test.
- **Notificaciones**: cuando termina (bien o mal) una etapa que tardó más de un minuto, o una cadena *Run to goal*, Windows muestra una notificación. **Dry-run (check all)** muestra, sin ejecutar nada, los casos de cada corrida (`Ap`, `η`, `n`), las carpetas, los comandos y todos los problemas.
- **Diagrama**: una caja por etapa, con su color y lo que contiene su resultado. El ratón encima dice por qué está en ese estado. La caja punteada *reference dataset* es el dataset de la referencia (clic para ir a él).
- **Panel de abajo**: qué hace la etapa, su **Resultado**, qué revisar, el progreso, la última corrida, por qué no puede correr y qué hacer, sus **entradas y salidas** (qué es cada archivo y si existe) y **qué canal** usa.

### Los colores

| Color | Significado |
|---|---|
| Verde | Hecha |
| Verde claro (*not needed*) | No se corrió, pero una etapa posterior ya tiene su resultado (p. ej. un dataset importado): no hay nada que hacer |
| Naranja | Desactualizada: su configuración o una entrada cambió después de correrla |
| Azul | Corriendo, con el % cuando se puede calcular |
| Rojo | Falló: el panel muestra las últimas líneas del log |
| Gris | Pendiente y lista para correr |
| Borde punteado | Bloqueada: falta una etapa anterior |

Borde oscuro = etapas necesarias para la meta, borde azul = siguiente paso, morado = la etapa que estás viendo.

### Botones de la etapa

| Botón | Qué hace |
|---|---|
| ▶ Run in console | La lanza en una consola nueva. Con **close the console when the stage ends OK** (activada por defecto) la consola se cierra sola si termina bien; si falla, se queda abierta hasta que pulses Enter. El log siempre se queda |
| Copy command | Copia el comando, para lanzarlo a mano |
| Log | Abre el visor del log (ver §5) |
| Viewer | Abre su `.h5` de salida en el visor; abajo aparece "Opening the viewer…" mientras carga |
| Edit config | Los ajustes **de esa etapa** (en Simulate y Extract: la simulación) |
| Labels YAML | Abre el YAML de etiquetas para revisarlo |
| Label grid | (en Label build) Abre el dataset etiquetado como una **rejilla de casos**, ordenados por `η` y paginados: cada panel es la señal de un caso coloreada por su etiqueta (azul estable, gris gray, naranja inestable: la paleta de las figuras de artículo), con las líneas `±lim_inf` / `±lim_sup` del criterio de amplitud. Se pueden **quitar las líneas** (casilla *criterion lines*) cuando la señal es muy pequeña frente a los límites y no se ve; la barra de matplotlib hace zoom por panel. La casilla *histogram + fitted normal* cambia la señal por el histograma de sus muestras con la normal ajustada (discontinua) y marca `μ` y `μ±σ`; `μ` y `σ` de cada caso (por etiqueta) están escritos en su panel en los dos modos. Elige canal, etiqueta y filas × columnas. **💾 Export…** guarda cada página con estilo de artículo (ver *Guardar figuras*, en el paso 9). Solo lee el `.h5`; sirve para juzgar de un vistazo si el etiquetado tiene sentido |
| Folder | Abre la carpeta de sus salidas |
| Go to blocker | Va a la etapa (de otro experimento) que la bloquea |
| Mark up to date | En una etapa naranja por un cambio de configuración que no altera su resultado: la marca al día |

## 3. Recorrido completo: validar los indicadores

### Paso 1. Mira el entrenamiento

Selecciona `train_…`. En **Label build** ves cuántos casos son estables, gray e inestables y dónde cae la frontera de `η`: es lo que aprenden los indicadores. Label template aparece *not needed*: su dataset se hizo con otro YAML (ver §5).

### Paso 2. Crea la validación

Pulsa **New experiment…**.

![New experiment: un solo Ap, simulación escrita completa y vista previa de los casos](03_new_experiment.png)

1. **what for** = *Validation against a reference* y **reference experiment** = tu entrenamiento.
2. Rellena la simulación: carpeta (`base_dir`), caso (el modelo Nessy2m), `n`, y las profundidades **en `Ap` [mm] o en `η`**. Con `η` eliges `ap_ref`: *manual* (una profundidad), *model* (el mínimo del SLD de un modelo) o *model_at_spin* (el límite del SLD a la `n` de cada caso), con el modelo del SLD.
3. **Check / preview cases** muestra la tabla de casos (`Ap`, `η`, `n`) y los problemas. Nada se guarda mientras haya un error en rojo (p. ej. una `n` en un hueco entre lóbulos, donde el límite es infinito). También avisa de los `η` que ya están (o casi) en la referencia, que no prueban nada nuevo, y estima el tiempo de simulación con las simulaciones anteriores de esa carpeta (`wall_time_s.txt`, con la misma discretización).
4. **Propose names** propone el nombre del experimento y de la carpeta de datos (sin prefijo obligatorio; puede ser un solo caso).
5. **Pick the depths on the SLD…** abre los lóbulos del modelo con la línea de tu `n`, su límite y los casos de la referencia coloreados por etiqueta. Con **η** marcado el eje vertical está en `η` (= `Ap` / límite a tu `n`, con una línea en `η` = 1); con **Ap [mm]**, en milímetros. Clic = añade una profundidad en esa `n`, en las unidades del eje (clic derecho = quita la más cercana), o rellena un rango. La lista dice el `η` y la zona de cada uno. **Use these Ap** los escribe en el formulario (y, si lo dejas marcado, pone `ap_ref = model_at_spin` de ese modelo). Dos botones llevan la `n` a un punto notable del SLD: **n at the minimum limit** (el fondo del lóbulo donde estás, de todo el SLD o de un modo, según el desplegable) y **n at the intersection**, la `n` corregida donde se cruzan los lóbulos de dos modos (la misma intersección que marca el visor; con varias, la más cercana a la `n` que tienes). Este segundo botón solo se activa con un modelo de dos o más modos, p. ej. `2DOF_150_250` (9989 rpm, 15.557 mm); con un modo, los lóbulos no se cruzan y queda desactivado.
6. **load values from** también acepta experimentos importados: su simulación se reconstruye desde la carpeta (`doe_config.yaml`, los `var_val.py` o los atributos del `.h5`). Una carpeta importada también se abre en el planificador desde **Edit config** de su Simulate.
7. ¿Quieres que el planificador de validación elija los `η` alrededor de los casos del entrenamiento? **Pick η with the validation planner…**, guarda su YAML en `configs/` y elígelo en **load values from**, que solo rellena los campos.

### Paso 3. Simula

Selecciona la validación: **Next step** dice *Simulate*. Pulsa **▶ Run next step**. La caja pasa a azul con el % de casos hechos, y el panel se actualiza solo.

> Tarda horas. Puedes cerrar la app y volver. Las opciones `--timed` (casos uno por uno, `nb_proc` en paralelo, tiempo por caso guardado) y `--auto-extract` (Extract justo después) están en **Edit config** de Simulate.

### Paso 4. Extract

El mismo botón. Junta los casos en `doe_results.h5`.

### Paso 5. Label template y revisión

Genera la propuesta de etiquetas con los parámetros **de la referencia** (aquí no se editan: la verdad se etiqueta igual que el dataset con el que aprendieron los indicadores). Pulsa **Labels YAML** y revísalo.

> Esas etiquetas son la verdad contra la que se mide. Un caso mal etiquetado altera todas las métricas.

### Paso 6. Label build

Corta las señales según el YAML. Los casos gray no cuentan en las métricas. Pulsa **Label grid** para ver todos los casos etiquetados de un vistazo y juzgar si el corte de amplitud tiene sentido.

### Paso 7. Indicators

Corre cada variante sobre cada caso, aprendiendo de la referencia. Al terminar **cada caso** la consola (y el log) muestra un resumen: `η`, `Ap`, la etiqueta verdadera y, por variante, si detectó, en qué momento y si acertó (`OK` / `MAL`). En una rampa que cruza muestra además dónde la verdad pasa a inestable, el retraso con signo y si fue acierto, falsa alarma (antes del inicio) o fallo (la regla de Validate).

### Paso 8. Validate

Por variante: TP, FN, TN, FP, exactitud balanceada, MCC, AUC, tiempos de detección. El panel muestra el ranking. La exactitud balanceada y el MCC llevan su **intervalo del 95 %** entre corchetes (con pocos casos es ancho: dice cuánto fiarse del número) y, si hay aciertos, la **razón mediana `t_det / t_onset`** (< 1: el indicador alarmó antes de que la vibración alcanzara el límite de amplitud; no depende de η). Compare muestra las mismas columnas.

**Casos grises.** Un caso cuya etiqueta entera es *gray* (Ap constante) no se puntúa por defecto. En **Edit config** de Validate, el campo **gray cases** elige cómo tratarlos: *ignore* (no se puntúan; lo de siempre), *stable (pessimistic)* (una alarma cuenta como falsa alarma, ninguna como acierto) o *unstable (optimistic)* (una alarma cuenta como acierto, ninguna como fallo). El modo recalcula todo (conteos, métricas, ranking, ROC y figuras) y **cada modo tiene su propio archivo** (`doe_validation_results.h5`, `…_gray-stable.h5`, `…_gray-unstable.h5`) y su carpeta de figuras (`figs_validation`, `figs_validation_gray-stable`…): para tener los tres, corre Validate una vez por modo. Cambiar el modo deja Validate desactualizada. La tarjeta dice el modo activo y, aunque no se puntúen, cuántos grises hay, cuántos dan alarma y las cotas «si todos fueran estables / inestables»: son cotas, no un veredicto. En el visor, la leyenda de esos casos dice «(gray)». Compare compara los archivos del modo de cada experimento.

Un caso inestable se puntúa con su **primera detección** frente al inicio de su verdad (`t_onset`: la primera muestra sobre el límite de amplitud en un caso constante): en un **caso constante** cualquier alarma → TP y ninguna → FN; el retraso (`delay_onset_s`, `delay_det_s`) queda como dato, no decide el acierto. En una **rampa que cruza**, una alarma antes del inicio de su verdad es falsa alarma, después es TP y ninguna es FN. No hay tolerancia que configurar (antes existía `early_tol_s`; se quitó y un YAML que aún la lleve se ignora).

### Paso 9. Ver y comparar

![Validación completa](02_validation.png)

**Viewer** abre el resultado. Hay **una sola ventana de visor** para todos los archivos: si ya tienes un visor abierto, el archivo entra en esa ventana como una pestaña nueva (y si ese archivo ya estaba abierto, su pestaña se recarga con lo último, útil tras volver a correr la etapa); solo si no hay ninguno abierto se abre una ventana nueva. Esto vale para el botón *Viewer* de cualquier experimento o etapa; el botón *Unified viewer* de la pestaña Tools sigue abriendo uno propio. En la pestaña **Compare** eliges dos validaciones y ves sus métricas lado a lado; el botón **Plot** abre la comparación A/B en la ventana de exportación (se guarda al pulsar *Save*, en `figs_validation/`).

En el visor, el panel de la derecha (*Summary plots*) conserva las curvas de referencia: los SLD y, en un archivo de validación, el SLD con el resultado (TP/TN/FN/FP) de cada indicador, solo del modelo con que se simularon los casos. En el SLD, los lóbulos y los casos van siempre en Ap (eje izquierdo: *Width of cut* / *Largeur de coupe* `a_p` [mm]). La casilla **η axis** añade un **eje derecho en η** = Ap / límite del SLD a la velocidad de los casos (la mediana de sus rpm; el límite es η = 1). Es exacto cuando los casos comparten velocidad; si esa velocidad cae en un hueco entre lóbulos, o no hay casos, η se calcula contra `a_p,min` del modelo y el rótulo lo dice.

En la gráfica de `I(t)`, cada línea tiene un significado y la leyenda lo dice:
- **Trazo discontinuo con un punto = `t_d`**, la primera detección de ese indicador (del color de su curva).
- **Línea de puntos = `t_onset`**, el instante en que la verdad pasa a inestable: en una rampa, su primera ventana inestable; en un caso constante de un archivo de validación, cuando la vibración alcanza el límite de amplitud del etiquetado.
- **Trazo y punto alternados horizontal = límite de detección del indicador**, el umbral que `I(t)` cruza en `t_d`: SST, `lim_sup` (y `lim_inf` si es positivo); RMS-CV, el umbral del CV; MaxEnt-SPRT, las dos cotas ln((1−β)/α) y ln(β/(1−α)). Green no guarda un umbral, así que no se dibuja. Los límites se leen del archivo de indicadores que está junto al de validación.
- **Δ en la leyenda de cada curva** (solo en validación) = `t_d − t_onset`; negativo = el indicador alarmó antes de que la vibración llegara a la amplitud del etiquetado.
Con un solo indicador, las curvas se colorean por caso y aparece la **barra de color** de η, con una marca por cada caso dibujado, como en las demás pestañas. Las demás figuras (las de validación: ranking, ROC, matriz de casos, tiempos de detección…; y las de `t_d` e `I(t)`) están en **Figures…**.

### Guardar figuras (cualquier visor)

Todos los visores tienen el botón **💾 Export…** (el visor, cada vista de la referencia, *Label grid*, *Compare › Plot*, `doe_planner` y `doe_val_planner`); en el visor, **Save…** y **Figures…** abren la misma ventana con su figura elegida. La ventana lista **todas** las figuras de ese visor: los paneles tal como están en pantalla (*Panel — Signals*, *I(t)*, *Forces*…, con su selección y su zoom; se exporta una copia, el panel no cambia) y todas las figuras de resumen. Controles:

| Control | Qué hace |
|---|---|
| size | *own* (la figura tal cual: su tamaño de artículo, o el de pantalla para un panel), *SIMPLE* (1 columna), *WIDE* (página), *grid* (columnas × filas de *SIMPLE*; se rellena solo con la disposición de ejes del panel) |
| scale | multiplica el tamaño (1 = el del artículo, 1.5 por defecto) |
| text | *follows scale* (por defecto): letra, líneas y marcadores crecen con la escala como un zoom, así la figura se ve igual en grande o en pequeño (a 1.5 queda como antes). *fixed*: los puntos del estilo del artículo a cualquier escala (al ampliar, el texto parece más pequeño). En un panel o una figura no nativa solo actúa con *size* SIMPLE / WIDE / grid (con *own* el lienzo no cambia) |
| language | EN / FR / both. Las figuras de validación y el SLD se dibujan en ese idioma; las demás se traducen al exportar con la tabla `DOE_plots/figure_texts.yaml` (si falta una frase, la barra de estado lo dice: añádela a la tabla) |
| article proportions | en pantalla, la vista previa con la proporción del archivo |
| dpi, format | 200 / 300 / 600; png / pdf / svg |
| folder, name | carpeta (por defecto `figs_validation/`, `figs_indicators/` o `figs_reference/` junto al `.h5`) y nombre del archivo |

**Save** guarda la figura elegida; **Save all**, todas (dice cuáles no se pudieron hacer y qué textos quedaron sin traducir). Una figura sin datos (por ejemplo, las de casos constantes en un archivo que solo tiene rampas) muestra el error abajo en vez de dibujarse.

## 4. Otras tareas

### Cambiar la simulación (otra n, otros Ap)

**Edit config** de Simulate abre el formulario de la simulación. Si el experimento leía una config de `configs/`, al guardar la simulación queda escrita completa dentro del experimento (la config no se toca). **Open in the planner** la muestra sobre el SLD.

### Probar otra n sin perder lo que hay

**Copy…** con un sufijo de carpeta (p. ej. `_n10000`): la copia es un archivo completo, con sus propias salidas y su propia carpeta de datos. Después, **Edit config** de su Simulate para cambiar `n`, `Ap` o `η`.

### Probar otras variantes de indicadores

**Edit config** de Indicators.

![Tabla de variantes](04_indicators_form.png)

- Cada fila es una variante (un grupo en el archivo de resultados): indicador, canal, modo, ventana, paso y aux.
- **Add** añade una fila desde un *preset* (`experiments/indicator_variants.yaml`); el experimento guarda su propia copia.
- **Edit row…** (o doble clic): canal, modo, ventana, paso; el resto de parámetros en *Advanced*. **Propose name** pone el nombre según los parámetros.
- Con una referencia, **Same variants as the reference** usa las de la referencia.
- **Show resolved config for a case…** muestra la configuración final de un caso (con `T_rev` de su `n`).

### Cambiar qué es el experimento

**Experiment settings…**: descripción, etapas activas (o una plantilla de flujo), referencia y **carpeta de salidas** (vacía = `<carpeta del DOE>/<experimento>`).

![Experiment settings](05_settings.png)

### Traer datos que ya existen

| Botón | Para qué |
|---|---|
| Import folder… | Una carpeta ya simulada con `doe_results.h5` (u otro `.h5` con el mismo formato). Antes de aceptar muestra qué hay dentro (casos simulados, `n`, `η`, datasets etiquetados, resultados) y qué etapas activará; eliges qué dataset etiquetado es la verdad (p. ej. `amp` o `eta`) y el nombre es el de la carpeta. Lo que ya existe aparece en verde. Si la carpeta está simulada pero **sin extraer** (no hay `.h5`), elige *(not extracted yet)* y el `ap_ref` para `η`: la app reconstruye la simulación desde el `var_val.py` de cada caso, deja Simulate hecha y bloqueada (relanzarla borraría la carpeta) y Extract lista para correr |
| Standardize an .h5… | Un `.h5` hecho fuera de la app: dice qué tiene y qué falta, añade los atributos que faltan a cada caso (modelo simulado `sim_case` / `sim_model`, `n`, `η` desde un `ap_ref`) sin tocar las señales, y crea su experimento |
| Delete | Borra el archivo del experimento y sus registros (`.runs/<experimento>`), nunca datos; se niega si otro experimento lo usa |

### Etapas opcionales

Deflexión estática, ruido y SNR del modelo se activan en **Experiment settings**. **Noise** y **Noise validation** tienen formulario en **Edit config**; las otras se configuran en su sección del YAML (**Edit config** abre el archivo): las claves son las constantes del script en minúscula (`k_cut`, `control_idx`…).

### Validación con ruido

¿Un indicador calibrado con datos limpios sigue funcionando cuando la señal tiene ruido de sensor? Los umbrales se quedan como salieron del entrenamiento limpio; solo se ensucian las señales de **validación**, y la verdad de cada copia ruidosa es la de su caso limpio.

1. **Noise** (Edit config → *Noise for the validation*): elige los casos (**Pick…** los lista con su η y su etiqueta), los **niveles de SNR** en dB y las **realizaciones**. El SNR es **absoluto**: el mismo ruido para todos los casos, calculado a partir del caso inestable más débil (**reference case**: `auto`, o uno a mano). Las copias son casos × niveles × realizaciones; el formulario dice cuántas son y cuánto pesa el archivo (`doe_noise_multi_results.h5`, en la carpeta de salida del experimento). Sin marcar la casilla queda el modo antiguo: un solo caso de control, sin validación.
2. **Noise indicators**: corre los indicadores sobre cada copia (sin guardar las señales otra vez). En su **Viewer**, selecciona la fila *clean* de un caso y sus copias para ver juntas las I(t) limpia y ruidosas (umbrales y detecciones incluidos): con un indicador marcado la limpia va en negro y las copias por SNR; con varios, cada indicador tiene su color y la limpia va en un tono más oscuro y más gruesa.
3. **Noise validation** (necesita **Validate** hecha): puntúa cada copia contra la verdad del caso limpio, con el mismo modo de grises que la validación limpia. En su **Viewer** hay tres figuras: métricas contra SNR (línea = media, banda = mín–máx entre realizaciones, marcador = limpio, línea discontinua = el SNR donde la exactitud equilibrada cae más de 0.05), matriz caso × SNR y anticipación. **Figures…** las guarda en `figs_noise_validation/`.

La etapa se activa en **Experiment settings** (**Noise validation** trae **Noise indicators**, **Noise** y **Validate**). Si **Noise** no tiene `cases`, **Noise validation** no se puede correr y dice por qué.

### Ampliar sin rehacer (Run missing)

Si después de correr **Indicators** o **Noise indicators** solo **añades** trabajo (más casos, más realizaciones o niveles de ruido, más variantes), la etapa queda *stale* pero lo ya calculado sigue valiendo. Su tarjeta dice `missing: N of M tasks (group x variant)` y **Run missing** calcula solo esas `N` (`--resume`), sin tocar el resto. Las etapas que leen esos resultados quedan *stale* como siempre: córrelas después (**Noise validation** y **Validate** tardan poco).

Qué es seguro (probado en una copia pequeña, 2026-10-07):

| Cambio | ¿Run missing? |
|---|---|
| Más realizaciones, más niveles de SNR o más casos en **Noise** (vuelve a correr **Noise**: las copias que ya existían salen idénticas bit a bit, la semilla va por realización y por número de caso) | sí |
| Más variantes en la tabla de Indicators | sí |
| `realizations_run` que crece (p. ej. `[0]` → todas) | sí |
| Otros parámetros de una variante que ya existe (mismo nombre) | **no**: corre la etapa entera |
| Otra semilla, otro caso de referencia del SNR, menos casos o niveles, otras etiquetas o referencia | **no**: corre la etapa entera |

La app lo decide comparando la configuración con la que corrió la etapa (se guarda en su registro desde esta versión) con la actual: si algo **cambió** (no solo creció), **Run missing** se niega y dice qué. Si la etapa corrió antes de esta versión (o se importó), la app no lo sabe y **Run missing** pregunta. Con `snr_ref_case: auto`, cambiar las etiquetas puede cambiar el caso de referencia sin que la configuración lo diga: en ese caso corre todo.

### Rampas de Ap

Un caso en **rampa** cambia la profundidad durante el corte (con `n` fija): `Ap` va de `Ap_start` a `Ap_end`, en línea recta con el tiempo de la señal. Sirven para **validar**; el entrenamiento sigue siendo de casos constantes. Un experimento puede mezclar casos constantes y rampas.

- **Crear.** En el formulario de la simulación, **depths end (ramps)** junto a **depths**: vacío = todo constante; un valor por profundidad (misma unidad, `Ap` o `η`) = el `Ap` al final del corte de cada caso. Un fin distinto del inicio es una rampa; igual, un caso constante. La vista previa muestra `Ap end`, `η end` y el tipo de cada caso.
- **En el SLD.** **Pick the depths on the SLD…** → **add: ramps**. Dos clics = una rampa (inicio y luego fin); clic derecho quita la más cercana. **Fill** y **Propose** reparten los **inicios** en `[from, to]` y les suman el **ramp span** (en la unidad elegida; negativo = `Ap` decreciente). Aceptar, quitar y el zoom funcionan como con los puntos. **Use these Ap** rellena `depths` y `depths end`.

![SLD en modo rampas](07_ramps_sld.png)

- **Rampas decrecientes.** La pieza de la plantilla `1DOF_150Hz` solo sabe hacer crecer `Ap`; con ella una rampa decreciente saldría como un caso constante, y la app la rechaza. Usa un caso cuya `db_def` admita los dos sentidos (marca `# ramps: both directions`), p. ej. `Data/1DOF_150_Ramp_check/1DOF_150Hz`.
- **η.** Una rampa tiene `eta_start` y `eta_end`; su `eta` suelto (si existe) se ignora en toda la app. (Los archivos hechos antes del cambio de nombre llevan `kappa`, `kappa_start` y `kappa_end`: la app lee ambos.)
- **La verdad.** La misma regla de amplitud (`max|y|` frente a `lim_inf` / `lim_sup`), pero **ventana a ventana**, con ventanas como las de un indicador (`window_mode`, `window_N`, `window_step` en **Edit config** del etiquetado; por defecto los de las variantes, hoy 7 vueltas con paso 1). Ventanas seguidas iguales forman un intervalo; si alternan cerca del umbral se dejan tal cual (revísalo en el YAML). `t_onset` = inicio de la primera ventana inestable. No se usa ningún tiempo teórico de cruce `η = 1`. Los casos constantes se etiquetan igual que siempre.
- **Validación.** Las rampas que cruzan (estable → inestable) se puntúan con la misma regla que los constantes, contra su `t_onset`, pero **fuera** de las métricas globales, del ranking y del ROC: tienen sus métricas `ramp_*` (tasa de detección, alarmas antes del inicio, fallos, retraso con signo, alarmas en el tramo estable, persistencia). Una rampa que no cruza se puntúa como un caso constante. El panel de Validate y la pestaña **Compare** las muestran.
- **Visor.** Las tablas tienen `eta_start`, `eta_end` y `t_onset`; las rampas se ordenan por su `η` de inicio. En señal e `I(t)`, una línea punteada marca dónde la verdad pasa a inestable y, con un solo caso, se sombrean sus intervalos estable / gray / inestable.

## 5. Cosas que conviene saber

- **Desactualizada (naranja).** Cambiar la configuración de una etapa ya corrida la pone en naranja, y también a las siguientes. La huella ignora cómo se escribe una ruta (mayúsculas, `/` o `\`), el número de procesos (`nb_proc`, `workers`) y el `n2m.bat`. Si cambiaste algo que no altera el resultado, **Mark up to date**.
- **El log** es todo lo que la etapa imprimió en su consola. Se guarda en `.runs/<experimento>/<etapa>.log` y sigue ahí aunque cierres la consola. El visor colorea los errores (rojo), avisos (naranja), el progreso (azul) y los resúmenes por caso, busca texto y sigue el archivo mientras la etapa corre.

![Visor del log](06_log.png)

- **Reemplazar salidas.** Si una etapa va a sobrescribir un archivo, la app pregunta. En Label template el YAML anterior se guarda como `.bak-<fecha>`.
- **Etapas lanzadas fuera de la app** no aparecen como "corriendo"; se ven hechas cuando aparece su salida.
- **Dos etapas que escriben el mismo archivo** no se lanzan a la vez.
- **Trazabilidad.** Cada `.h5` de salida guarda de qué experimento y etapa viene, qué es (`experiment_role`) y con qué dataset aprendieron los indicadores. El visor lo muestra en una banda amarilla arriba.
- **El `.h5` de indicadores lleva la verdad**: cada caso tiene `eta`, `Ap_mm`, `true_label` y `label_strategy`. Si haces el etiquetado después de los indicadores, se añade al terminar Label build, sin recalcular nada.

## 6. Desde la consola

```
python experiment.py status [EXP]         estado y siguiente paso de cada experimento
python experiment.py check EXP            errores y avisos de la configuración
python experiment.py dryrun EXP           qué se correría: casos, carpetas, comandos (no corre nada)
python experiment.py accept EXP [ETAPA]   marca al día etapas naranjas por un cambio de configuración sin efecto
python experiment.py run EXP ETAPA        corre una etapa (lo mismo que "Run in console")
python experiment.py chain EXP [--goal G] corre las etapas que faltan hasta la meta (lo mismo que "Run to goal")
python experiment.py import NOMBRE CARPETA [--reference EXP] [--h5 ARCHIVO]
python experiment.py selftest
```

## 7. Dónde está cada cosa

| Qué | Dónde |
|---|---|
| Experimentos (todo lo que usan) | `DOE_utils/experiments/<nombre>.yaml` |
| Presets de variantes de indicadores | `DOE_utils/experiments/indicator_variants.yaml` |
| Configs antiguas de simulación (las lee un experimento que aún no se reescribió completo) | `DOE_utils/DOE_simulacion/configs/` |
| Registros, logs y el archivo que lee `doe_runner` | `DOE_utils/experiments/.runs/<experimento>/` |
| Datos de la simulación: carpetas de casos, `doe_results.h5`, deflexión, ruido, SNR del modelo | `base_dir/doe_name` de la simulación |
| Etiquetas, dataset etiquetado, indicadores, validación | la carpeta de salidas del experimento (por defecto `<carpeta del DOE>/<experimento>/`) |
| Capturas de este tutorial | `DOE_utils/tutorial_img/` (`python launcher.py --screenshot ARCHIVO.png EXP ETAPA`) |
| Diseño y decisiones | `docs/historico/PLAN_app_experimentos.md`, `docs/planes/PLAN_app_v2.md` |

## 8. Preguntas frecuentes

### ¿Cuándo edito la configuración y cuándo el experimento? ¿Por qué antes había dos YAML?

Antes la simulación estaba en `configs/` (heredando de `base.yaml`) y el resto en el experimento. Ahora hay **un solo archivo**: el experimento lleva su simulación escrita completa. **Experiment settings…** cambia qué es el experimento (descripción, etapas, referencia, carpeta de salidas); **Edit config** de una etapa cambia los ajustes de esa etapa. Un experimento antiguo que todavía lee `configs/` se reescribe completo al guardar su simulación, o con **Write it out in full** en Experiment settings.

### ¿Cómo sabe la app si una etapa terminó, está pendiente o falló?

Por tres cosas: **los archivos de salida** (si existen, está hecha); **un registro por etapa** en `.runs/<experimento>/<etapa>.json` (inicio, fin, código de salida, proceso y una huella de la configuración); y **las fechas** (si una entrada es más reciente que la etapa, queda desactualizada). Los registros no están en git.

### ¿Qué pasa si corro Indicators sin la etapa de etiquetado?

No se puede: cada variante aprende su umbral de los tramos estables (y, MaxEnt, también inestables) de un dataset etiquetado. Es el de la referencia o, sin referencia, el propio. Por eso Indicators espera a Label build, y el panel lo explica.

### ¿Hace falta todavía `t_gt` (`t_theorical`, `_T_GT`)?

No. Con el dataset etiquetado como referencia, la detección no lo usa. Se quitó de los presets (y en MaxEnt también `t_stable_total` y `cut_end_time`), y el visor ya no dibuja una línea `t_GT` fija: cada caso tiene su propia etiqueta.

### ¿Por qué MaxEnt dice "varianza muy pequeña, aplicando floor"?

En un tramo estable con la entropía casi constante, la desviación típica sale casi cero y MaxEnt usa un mínimo para no dividir por cero. No es un error. Ahora se cuenta y aparece una vez en el resumen de cada caso.

### ¿Qué es "Axial" en los nombres de canal?

Son los sensores de la herramienta en Nessy2m, en la dirección de su eje (z del sistema de la herramienta): `Axial_disp` desplazamiento [m], `Axial_vel` velocidad [m/s], `Axial_acc` aceleración [m/s²]; `res_R_p` es la fuerza de corte resultante [N]. El panel dice qué canal usa cada etapa: el del **etiquetado** (Label template / build), el **analizado** por cada variante (Indicators) y el de la **verdad** (Validate).

### ¿Al borrar un experimento se borran sus `.runs`?

Sí: Delete borra el archivo y `.runs/<experimento>`. Si renombras o borras un YAML a mano, su carpeta de `.runs` queda huérfana (`experiment.orphan_runs()` las lista); se puede borrar sin perder datos.

### ¿Qué pasa si uso un `.h5` que no generé con la app?

Usa **Import folder…** (si ya tiene el formato de `doe_results.h5`) o **Standardize an .h5…** (si le faltan atributos). Si un resultado está en otro sitio, el YAML acepta rutas explícitas (`label: {out: …}`, `indicators: {out: …}`, `validate: {out: …}`).

### ¿El planificador (doe_planner) sigue sirviendo?

Sí, para ver los casos sobre el SLD: **Open in the planner** en el formulario de la simulación. Si lanzas la simulación desde su botón, corre fuera del envoltorio y la app no guarda registro ni log; lánzala desde la app.

### ¿Tengo que usar la app para todo?

No. Todo lo que guarda son archivos de texto (YAML) y los scripts de siempre; puedes editarlos y correrlos a mano. La app relee los archivos cuando cambian. Lo que pierdes al saltártela es el registro, el log y los avisos (desactualizada, sobrescritura, dos escrituras a la vez, etiquetado distinto de la referencia).
