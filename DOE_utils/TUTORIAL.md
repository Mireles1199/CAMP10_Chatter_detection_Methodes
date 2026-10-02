# Tutorial: app de experimentos DOE

Se abre con `python launcher.py` (Python de `entorno_CAMP10`). Esta guía se lee dentro de la app (pestaña **Tutorial**) y también como archivo: `DOE_utils/TUTORIAL.md`. Usa el índice de la izquierda para saltar a una sección y **A+ / A−** para cambiar el tamaño del texto.

> En una frase: la app te dice en qué etapa está cada experimento, qué falta y cuál es el siguiente paso; tú pulsas un botón y la etapa corre en una consola aparte.

## 1. La idea

Un **experimento** es un DOE llevado desde la simulación hasta los resultados. Hay dos tipos:

- **training**: el DOE cuyos casos etiquetados enseñan a los indicadores qué es estable y qué es inestable.
- **validation**: un DOE con casos nuevos (otros `ap`) para comprobar que los indicadores clasifican bien. Siempre apunta a un training.

Cada experimento recorre **etapas**, que la app dibuja como cajas conectadas:

| Etapa | Qué hace | Qué produce |
|---|---|---|
| Simulate | Corre Nessy2m para cada caso | Una carpeta por caso |
| Extract | Junta los casos en un solo archivo | `doe_results.h5` |
| Label template | Propone la etiqueta de cada caso: estable, gray o inestable | `reference_labels.yaml` |
| Label build | Corta las señales según esas etiquetas | `reference_dataset_*.h5` |
| Indicators | Corre las variantes de indicadores sobre cada caso | `doe_indicator_results.h5` |
| Validate | Compara lo que detectaron con las etiquetas (solo en validación) | `doe_validation_results.h5` |
| Static deflection, Noise, Noise indicators, Model SNR | Etapas opcionales | Cada una, su archivo |

> Las etapas **no corren dentro de la app**: se abren en una consola aparte. Cerrar la app no las detiene, y al volver a abrirla el estado se recupera solo.

## 2. Leer la pantalla

![Pantalla principal: experimento de entrenamiento con la etapa Indicators seleccionada](01_overview.png)

- **Izquierda**: la lista de experimentos. Verde = meta alcanzada, azul = algo corriendo, rojo = una etapa falló.
- **Arriba a la derecha**: nombre, tipo, `n`, casos, rango de `κ` y con qué entrenamiento o validaciones está enlazado.
- **Goal**: lo que quieres conseguir. Se pone sola según el tipo. Junto a ella, **Next step**: la siguiente etapa que falta, con los botones **Show** (la selecciona) y **▶ Run next step** (la lanza).
- **Diagrama**: cada caja tiene un color y una tercera línea con lo que contiene su resultado. Pasando el ratón sobre una caja se ve por qué está en ese estado.
- **Panel de abajo** (al hacer clic en una caja): qué hace la etapa, su **Resultado**, un **Check** (qué conviene mirar), el progreso, la última corrida, por qué no puede correr y qué hacer, y los archivos.

### Los colores

| Color | Significado |
|---|---|
| Verde | Hecha |
| Naranja | Desactualizada: su configuración o una entrada cambió después de correrla |
| Azul | Corriendo, con el % cuando se puede calcular |
| Rojo | Falló: el panel muestra las últimas líneas del log |
| Gris | Pendiente y lista para correr |
| Borde punteado | Bloqueada: falta una etapa anterior |

El borde oscuro marca las etapas necesarias para la meta, el borde azul grueso el siguiente paso y el morado la etapa que estás viendo.

## 3. Recorrido completo: validar los indicadores

Es el flujo de tu caso: ya tienes el entrenamiento y quieres validarlo con casos nuevos.

### Paso 1. Mira el entrenamiento

Selecciona `train_…` en la lista. Debe estar verde en Simulate, Extract, Label build e Indicators. En **Label build** verás cuántos casos son estables, gray e inestables y dónde cae la frontera de `κ`: es lo que aprenden los indicadores.

> Label template aparece pendiente en este entrenamiento. Es normal: su dataset se hizo con otro YAML de etiquetas (ver la sección 5). No lo necesitas para validar.

### Paso 2. Crea la validación

Pulsa **New validation…**.

![Diálogo New validation](03_new_validation.png)

1. Elige el entrenamiento y deja el nombre propuesto (o cámbialo).
2. Pulsa **Create and open planner**. Se abre el planificador de casos con el dataset del entrenamiento y el `doe_name` ya puestos.
3. En el planificador, ajusta las zonas de `κ` y el número de casos, pulsa **Generate cases** y después **Save YAML…**, con el nombre propuesto y en la carpeta `configs/` que ya aparece.
4. Vuelve a la app. Hasta que guardes el YAML, la validación dice "No stages yet" y el motivo.

El número de casos nuevos, sus `κ` y su separación respecto a los del entrenamiento los decides en el planificador. `n` y la discretización salen del entrenamiento y no se editan.

### Paso 3. Simula

![Validación nueva: el siguiente paso es Simulate](02_validation.png)

Selecciona la validación. **Next step** dice *Simulate*. Pulsa **▶ Run next step**. Se abre una consola y la caja pasa a azul, con el % de casos hechos.

> Tarda horas. Puedes cerrar la app y volver: el estado se recupera solo.

### Paso 4. Extract

El mismo botón. Junta los casos en `doe_results.h5`, y la caja muestra cuántos son.

### Paso 5. Label template y revisión

El mismo botón. Genera la propuesta de etiquetas con los parámetros **heredados del entrenamiento** (en una validación no se editan). Pulsa **Labels YAML** y revísalo.

> Esas etiquetas son la verdad contra la que se mide. Un caso mal etiquetado altera todas las métricas.

### Paso 6. Label build

Corta las señales según el YAML. En el resultado mira cuántos casos quedaron estables, gray e inestables. Los gray no cuentan en las métricas.

### Paso 7. Indicators

Corre las variantes elegidas sobre cada caso, usando el dataset del entrenamiento como referencia. Es la etapa larga después de simular: para 34 casos y 4 variantes tardó unos 57 minutos. La caja muestra el progreso.

### Paso 8. Validate

Calcula por variante los casos TP, FN, TN y FP, la exactitud balanceada, el MCC, el AUC, los tiempos de detección y la calidad de la alarma. El panel muestra el ranking de las variantes.

### Paso 9. Ver y comparar

En Validate, **Viewer** abre el resultado en el visor, con los casos sobre el SLD coloreados por acierto. En la pestaña **Compare** eliges dos validaciones (por ejemplo, otra `n`) y ves sus métricas lado a lado.

## 4. Otras tareas

### Probar otra velocidad (n)

Selecciona un experimento y pulsa **Derive…**. Escribe la nueva `n`. Se crea una copia de su config de simulación con esa `n` y un experimento nuevo. En *Ap at the new n* puedes mantener los mismos `Ap`, o usar los mismos `κ` con el límite del SLD de un modelo a esa `n`. `T_rev` de los indicadores se recalcula solo.

### Probar otra configuración de indicadores

Selecciona la etapa Indicators y pulsa **Edit config**.

![Formulario de indicadores](04_indicators_form.png)

- Marca o desmarca variantes de la biblioteca.
- Selecciona una con el botón redondo y pulsa **Duplicate variant…** para crear una nueva (por ejemplo, ventana de 6 revoluciones). Cambia lo que quieras en el YAML y guarda.
- **Show resolved config for a case…** muestra la configuración final que recibirá un caso concreto, con `T_rev` ya calculado.
- En una validación puedes dejar marcado **Same variants as the training**.

> Las variantes que ya tienen resultados no se editan, se duplican. Así nunca cambia algo que otro experimento ya corrió.

### Otras formas de crear un experimento

| Botón | Para qué |
|---|---|
| New DOE… | Un DOE desde cero. Eliges una config plantilla (de ahí salen carpeta, caso y discretización), `n`, la lista de `κ` (`0.5, 0.9` o `0.5:2.0:0.1`) y el nombre |
| Import folder… | Un DOE que ya simulaste. Lo hecho aparece en verde. Sus etapas de simulación no se pueden relanzar desde la app, porque no tiene config |
| Duplicate… | Una copia de un experimento |
| Delete | Borra solo el YAML del experimento y sus registros, nunca datos, y se niega si otro experimento depende de él |

### Etapas opcionales

Deflexión estática, ruido y SNR del modelo se configuran en su sección del YAML del experimento: **Edit config** abre el archivo. Las claves son los nombres de las constantes del script en minúscula (`snr_range`, `k_cut`, `control_idx`…).

## 5. Cosas que conviene saber

- **Desactualizada (naranja).** Cambiar la configuración de una etapa ya corrida, por ejemplo un parámetro de etiquetado, pone esa etapa y las siguientes en naranja. Es una advertencia: los resultados que hay no corresponden a la configuración actual. Vuelve a correrlas cuando quieras.
- **Reemplazar salidas.** Si una etapa va a sobrescribir un archivo existente, la app te pregunta y dice cuál. En Label template, el YAML anterior se guarda como `.bak-<fecha>`.
- **Etapas lanzadas fuera de la app.** Una etapa que lances a mano en consola no aparece como "corriendo". La verás hecha cuando exista su salida.
- **Label template pendiente en el entrenamiento importado.** El dataset se hizo con otro YAML de etiquetas. Si lo regeneras, el dataset y los indicadores pasan a naranja.
- **Dos etapas que escriben el mismo archivo** no se lanzan a la vez. La app dice cuál bloquea.
- **Trazabilidad.** Cada archivo de resultados guarda de qué experimento viene (atributos `experiment`, `experiment_stage`, `experiment_hash`, `experiment_date`).
- **Salidas por experimento.** Las etiquetas, los indicadores y la validación van en `<carpeta del DOE>/<experimento>/`: dos experimentos sobre el mismo DOE no se pisan.
- **Si una etapa falla**, el panel muestra las últimas líneas del log. El botón **Log** abre el archivo completo.

## 6. Desde la consola

```
python experiment.py status [EXP]         estado y siguiente paso de cada experimento
python experiment.py check EXP            errores y avisos de la configuración
python experiment.py run EXP ETAPA        corre una etapa (lo mismo que "Run in console")
python experiment.py import NOMBRE CARPETA
python experiment.py selftest
```

Etapas: `simulate`, `extract`, `merge`, `label_template`, `label_build`, `indicators`, `validate`, `static_deflection`, `noise`, `noise_indicators`, `model_snr`.

## 7. Dónde está cada cosa

| Qué | Dónde |
|---|---|
| Experimentos | `DOE_utils/experiments/<nombre>.yaml` |
| Variantes de indicadores | `DOE_utils/experiments/indicator_variants.yaml` |
| Configs de simulación | `DOE_utils/DOE_simulacion/configs/` |
| Registros y logs de cada corrida | `DOE_utils/experiments/.runs/<experimento>/` |
| Capturas de este tutorial | `DOE_utils/tutorial_img/` (se regeneran con `python launcher.py --screenshot ARCHIVO.png EXP ETAPA`) |
| Diseño y decisiones | `DOE_utils/PLAN_app_experimentos.md` |
| Cada script suelto | pestaña **Tools** |

## 8. Preguntas frecuentes

### ¿Cómo sabe la app si una etapa terminó, está pendiente o falló? ¿Dónde se guarda?

Combina tres cosas:

- **Los archivos de salida.** Si existen, la etapa está hecha; si no, pendiente.
- **Un registro por etapa**, que escribe el envoltorio cuando la lanzas desde la app: `DOE_utils/experiments/.runs/<experimento>/<etapa>.json` (inicio, fin, código de salida, número de proceso y una huella de la configuración usada) y su `.log`. Con él sabe si está **corriendo** (el proceso sigue vivo), si **falló** (código de salida distinto de 0, o el proceso murió sin terminar) o si está **desactualizada** (la huella de la configuración cambió).
- **Las fechas**: si una entrada es más reciente que la etapa, queda desactualizada.

Estos registros no están en git. Si los borras, las etapas con salida existente siguen apareciendo como hechas ("no run record"); solo pierdes el historial y el log.

### ¿Qué pasa si uso un .h5 que no generé con la app?

Funciona, con una condición: el archivo tiene que estar donde la app lo espera, o decírselo.

- Si existe en la ruta esperada, la etapa aparece como **hecha** ("outputs exist, no run record: made outside the app"). Pasa a **desactualizada** si una de sus entradas es más reciente.
- Si está en otro sitio, ponle la ruta en el YAML del experimento (`label: {out: …}`, `indicators: {out: …}`, `validate: {out: …}`, `labels_yaml`), o usa **Import folder…**, que lo detecta y deja las etapas hechas en verde.
- La app **no revisa el contenido**, salvo un caso: avisa en rojo si el YAML de etiquetas y el dataset de etiquetas no coinciden.

### ¿Se cubren las etapas aunque no llegue a los indicadores?

Sí. Cada meta ("Goal") termina en una etapa distinta: *Simulated data* (hasta Extract), *Training dataset* (hasta Label build), *Indicators computed*, *Indicator validation*, además de ruido, SNR del modelo y deflexión estática. Un experimento de entrenamiento abre con *Training dataset*: no necesita indicadores para estar completo. Cambia la meta con el selector.

### ¿Dónde se guardan los .h5?

| Archivo | Dónde |
|---|---|
| `doe_results.h5`, carpetas de casos, deflexión estática (dentro del mismo `.h5`), ruido, SNR del modelo | La carpeta del DOE: `base_dir/doe_name` de su config, por ejemplo `Convergency_Simulation/4_DOE_Data_Training_Tube/<DOE>/` |
| Etiquetas, dataset de etiquetas, indicadores, validación | `<carpeta del DOE>/<experimento>/` |
| DOE fusionado (varias corridas) | Una carpeta nueva junto a las corridas |
| El experimento mismo | `DOE_utils/experiments/<nombre>.yaml` |

En el panel de cada etapa, la sección **Files** muestra la ruta exacta, y el botón **Folder** la abre.

### ¿Cómo se configura el runner (la simulación)?

Con el YAML de `DOE_simulacion/configs/` que usa el experimento (`runs: - config: <nombre>`), el mismo que usaba `doe_runner`. Cada config hereda de `base.yaml` (ruta de Nessy2m, procesos en paralelo, señales a extraer) y define `base_dir`, `case`, `doe_name`, el barrido de `Ap` y `spin_rate` y el `ap_ref`. La etapa Simulate ejecuta `doe_runner.py --config <nombre> --command n2m_sch`, y Extract el mismo con `--command extract`.

Puedes crear la config con **New DOE…** (desde una config plantilla), con **New validation…** (planificador de validación) o editándola a mano, y abrirla en el planificador con **Edit config**.

### ¿El planificador (doe_planner) ya no sirve?

Sí sirve. Es la mejor forma de **ver el DOE sobre el SLD** y de ajustar una config. **Edit config** de las etapas Simulate y Extract lo abre con la config del experimento. Una diferencia: si lanzas la simulación desde su botón "Lanzar", se ejecuta fuera del envoltorio, así que la app no guarda registro ni log; solo ve los archivos que van apareciendo. Para tener estado, progreso y log, lanza desde la app.

### ¿El planificador de validación (doe_val_planner) ya no es útil?

Sí es útil, y **New validation…** lo abre. Elige los casos nuevos por zonas de `κ`, evitando los `κ` del entrenamiento, y los muestra junto a los de entrenamiento. **New DOE…** también acepta una lista de `κ`, pero no hace eso. Para una validación, úsalo.

### ¿Tengo que editar algo desde la app? ¿Todo?

No. La app es una comodidad, no un requisito.

- Todo lo que guarda la app son archivos de texto (YAML) y los scripts de siempre. Puedes editarlos a mano: la app los relee cuando cambian, incluidos los `configs/*.yaml` y la biblioteca de variantes.
- Los formularios existen para el camino principal (etiquetado, indicadores, validación, fusión). Para lo demás (deflexión, ruido, SNR) se edita el YAML.
- Puedes correr cualquier script por tu cuenta (`doe_runner.py`, `doe_indicators.py`…). La app ve los archivos que aparezcan y los marca como hechos; solo no tendrá el registro ni el log de esa corrida.
- Lo que la app **sí** vigila por ti: avisos de etapa desactualizada, de sobrescritura, de dos etapas escribiendo el mismo archivo y de etiquetado distinto entre validación y entrenamiento. Eso se pierde si te saltas la app.
