# Tutorial: app de experimentos DOE

Esta guía se lee dentro de la app (pestaña **Tutorial**) y también como archivo: `DOE_utils/TUTORIAL.md`.
Se abre la app con `python launcher.py` (Python de `entorno_CAMP10`).

## 1. La idea en un minuto

Un **experimento** es un DOE llevado desde la simulación hasta los resultados. Hay dos tipos:

- **training**: el DOE cuyos casos etiquetados enseñan a los indicadores qué es estable y qué es inestable.
- **validation**: un DOE con casos nuevos (otros `ap`) para comprobar que los indicadores clasifican bien. Siempre apunta a un training.

Cada experimento recorre **etapas**, que la app dibuja como cajas conectadas:

| Etapa | Qué hace | Qué produce |
|---|---|---|
| Simulate | Corre Nessy2m para cada caso | Una carpeta por caso |
| Extract | Junta los casos en un solo archivo | `doe_results.h5` |
| Label template | Propone la etiqueta de cada caso (estable, gray, inestable) | `reference_labels.yaml` |
| Label build | Corta las señales según esas etiquetas | `reference_dataset_*.h5` |
| Indicators | Corre las variantes de indicadores sobre cada caso | `doe_indicator_results.h5` |
| Validate | Compara lo que detectaron con las etiquetas (solo validación) | `doe_validation_results.h5` |
| Static deflection, Noise, Noise indicators, Model SNR | Opcionales | Cada una su archivo |

Las etapas **no corren dentro de la app**: se abren en una consola aparte. Cerrar la app no las detiene.

## 2. Leer la pantalla

- **Izquierda**: lista de experimentos. Verde = meta alcanzada, azul = algo corriendo, rojo = una etapa falló.
- **Arriba a la derecha**: nombre, tipo, `n`, casos, rango de `κ`, y con qué entrenamiento o validaciones está enlazado.
- **Goal**: lo que quieres conseguir. Se pone sola según el tipo. Junto a ella, **Next step**: la siguiente etapa que falta.
- **Diagrama**: cada caja tiene un color y una tercera línea con lo que contiene su resultado.

  | Color | Significado |
  |---|---|
  | Verde | Hecha |
  | Naranja | Desactualizada: su configuración o una entrada cambió después de correrla |
  | Azul | Corriendo (con el % si se puede calcular) |
  | Rojo | Falló |
  | Gris | Pendiente, lista para correr |
  | Borde punteado | Bloqueada: falta una etapa anterior |

  Borde oscuro = necesaria para la meta. Borde azul grueso = siguiente paso. Borde morado = la que estás viendo.
- **Pasar el ratón por una caja** dice por qué está en ese estado.
- **Panel de abajo** (al hacer clic en una caja): qué hace la etapa, el **Resultado** (qué contiene su salida), un **Check** (qué conviene mirar), el progreso, la última corrida con su duración, por qué no puede correr y qué hacer, y los archivos.

## 3. Recorrido completo: validar los indicadores

Este es el flujo de tu caso: ya tienes el entrenamiento y quieres validarlo con casos nuevos.

**Paso 1. Mira el entrenamiento.** Selecciona `train_…` en la lista. Debe estar verde en Simulate, Extract, Label build e Indicators. En Label build verás cuántos casos son estables, gray e inestables y dónde cae la frontera de `κ`. Eso es lo que aprenden los indicadores. (Label template aparece pendiente: es normal, ver sección 5.)

**Paso 2. Crea la validación.** Botón **New validation…**:
1. Elige el entrenamiento y deja el nombre propuesto (o cámbialo).
2. **Create and open planner**. Se abre el planificador de casos con el dataset del entrenamiento y el `doe_name` ya puestos.
3. En el planificador: ajusta las zonas de `κ` y el número de casos, **Generate cases**, y **Save YAML…** con el nombre propuesto, en la carpeta `configs/` que ya aparece.
4. Vuelve a la app: la validación aparece con sus etapas. Hasta que guardes el YAML dice "No stages yet" y el motivo.

El número de casos nuevos, sus `κ` y su separación respecto a los del entrenamiento los decides en el planificador. La discretización y `n` salen del entrenamiento y no se editan.

**Paso 3. Simula.** Selecciona la validación; Next step dice **Simulate**. Pulsa **▶ Run next step** (o haz clic en la caja y **▶ Run in console**). Se abre una consola y la caja pasa a azul con el % de casos. Tarda horas: puedes cerrar la app y volver; el estado se recupera solo.

**Paso 4. Extract.** Mismo botón. Junta los casos en `doe_results.h5` (la caja muestra cuántos).

**Paso 5. Label template y revisión.** Mismo botón. Genera la propuesta de etiquetas con los parámetros **heredados del entrenamiento** (no se editan en una validación). Pulsa **Labels YAML** y revísalo: esas etiquetas son la verdad contra la que se mide. Un caso mal etiquetado altera todas las métricas.

**Paso 6. Label build.** Corta las señales según el YAML. En el resultado mira cuántos casos quedaron estables, gray e inestables. Los gray no cuentan en las métricas.

**Paso 7. Indicators.** Corre las variantes elegidas sobre cada caso usando el dataset del entrenamiento como referencia. Es la etapa larga después de simular (para 34 casos y 4 variantes tardó unos 57 minutos). La caja muestra el progreso.

**Paso 8. Validate.** Calcula por variante: casos TP, FN, TN, FP, exactitud balanceada, MCC, AUC, tiempos de detección y calidad de la alarma. El panel muestra el ranking.

**Paso 9. Ver y comparar.** En Validate, **Viewer** abre el resultado en el visor (casos sobre el SLD coloreados por acierto). En la pestaña **Compare** eliges dos validaciones (por ejemplo, otra `n`) y ves sus métricas lado a lado.

## 4. Otras tareas

**Probar otra velocidad (`n`).** Selecciona un experimento, **Derive…**, escribe la nueva `n`. Se crea una copia de su config de simulación con esa `n` y un experimento nuevo. En "Ap at the new n" puedes mantener los mismos `Ap` o usar el mismo `κ` con el límite del SLD de un modelo a esa `n`. `T_rev` de los indicadores se recalcula solo.

**Probar otra configuración de indicadores.** Selecciona la etapa Indicators, **Edit config**:
- Marca o desmarca variantes de la biblioteca.
- Selecciona una con el botón redondo y **Duplicate variant…** para crear una nueva (por ejemplo, ventana de 6 revoluciones). Cámbiale lo que quieras en el YAML y guarda. Las variantes que ya tienen resultados no se editan, se duplican.
- **Show resolved config for a case…** muestra la configuración final que recibirá un caso concreto (con `T_rev` calculado).
- En una validación puedes dejar **Same variants as the training**.

**Crear un DOE desde cero.** **New DOE…**: eliges una config plantilla (de ahí salen carpeta, caso y discretización), `n`, la lista de `κ` (`0.5, 0.9` o `0.5:2.0:0.1`) y el nombre. Genera la config y el experimento.

**Importar un DOE ya simulado.** **Import folder…**: eliges la carpeta con `doe_results.h5`. Lo hecho aparece en verde. Las etapas de una carpeta importada no se pueden relanzar desde la app (no tiene config).

**Etapas opcionales** (deflexión estática, ruido, SNR del modelo): se configuran en su sección del YAML del experimento. **Edit config** abre el archivo. Las claves son los nombres de las constantes del script en minúscula (`snr_range`, `k_cut`, `control_idx`…).

**Duplicate… / Delete.** Duplicar copia el experimento. Borrar elimina **solo su YAML** y sus registros, nunca datos, y se niega si otro experimento depende de él.

## 5. Cosas que conviene saber

- **Desactualizada (naranja).** Cambiar la configuración de una etapa ya corrida (por ejemplo, un parámetro de etiquetado) pone esa etapa y las siguientes en naranja. Es una advertencia: los resultados que hay no corresponden a la configuración actual. Vuelve a correrlas cuando quieras.
- **Reemplazar salidas.** Si una etapa va a sobrescribir un archivo existente, la app te pregunta y dice cuál. En Label template el YAML anterior se guarda como `.bak-<fecha>`.
- **Una etapa lanzada fuera de la app** (a mano en consola) no aparece como "corriendo". La verás hecha cuando exista su salida.
- **Label template pendiente en el entrenamiento importado.** Su dataset se hizo con otro YAML de etiquetas. Si lo regeneras, el dataset y los indicadores pasan a naranja. No lo necesitas para validar.
- **Dos etapas que escriben el mismo archivo** no se lanzan a la vez. La app dice cuál bloquea.
- **Cada archivo de resultados guarda de qué experimento viene** (atributos `experiment`, `experiment_stage`, `experiment_hash`, `experiment_date`).
- **Salidas por experimento.** Las etiquetas, los indicadores y la validación van en `<carpeta del DOE>/<experimento>/`; dos experimentos sobre el mismo DOE no se pisan.
- Si una etapa falla, el panel muestra las últimas líneas del log; **Log** abre el archivo completo.

## 6. Desde la consola

```
python experiment.py status [EXP]         estado y siguiente paso de cada experimento
python experiment.py check EXP            errores y avisos de la configuración
python experiment.py run EXP ETAPA        corre una etapa (lo mismo que "Run in console")
python experiment.py import NOMBRE CARPETA
python experiment.py selftest
```

Etapas: `simulate extract merge label_template label_build indicators validate static_deflection noise noise_indicators model_snr`.

## 7. Dónde está cada cosa

| Qué | Dónde |
|---|---|
| Experimentos | `DOE_utils/experiments/<nombre>.yaml` |
| Variantes de indicadores | `DOE_utils/experiments/indicator_variants.yaml` |
| Configs de simulación | `DOE_utils/DOE_simulacion/configs/` |
| Registros y logs de cada corrida | `DOE_utils/experiments/.runs/<experimento>/` |
| Diseño y decisiones | `DOE_utils/PLAN_app_experimentos.md` |
| Cada script suelto | pestaña **Tools** |
