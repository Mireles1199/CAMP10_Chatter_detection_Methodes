# Guía de la validación con ruido

**Para quién es:** para entender y explicar a los directores cómo se prueba si un indicador de chatter sigue funcionando cuando la señal trae **ruido**. No hace falta saber estadística: cada concepto se explica desde cero, con un ejemplo y con los números reales de la tesis.

Esta guía es la hermana de `GUIA_metricas_validacion.md`: allí se explican las métricas (TPR, TNR, balanced accuracy...) y aquí solo se explica **qué cambia al añadir ruido**. Los ejemplos vienen del experimento **n12000** (1DOF, 12098 rpm), con Ap constante. Las rampas de Ap quedan fuera.

**Cómo está organizada**
1. La pregunta y una analogía.
2. El vocabulario.
3. El flujo paso a paso, con un ejemplo.
4. Qué se puede configurar y cómo se corre.
5. Cómo se calculan las métricas con ruido.
6. Las gráficas, una por una.
7. Qué salió en la primera prueba.
8. Cómo contárselo a los directores.
9. Glosario y dónde está cada cosa.

---

## 1. La pregunta y una analogía

Todo lo que se validó hasta ahora usa señales **limpias** de simulación. Un sensor real añade **ruido**. La pregunta es:

> *Un indicador cuyo umbral se calibró con datos limpios, ¿sigue funcionando cuando la señal tiene ruido?*

Es una pregunta de **robustez**.

**Analogía:** un detector de humo calibrado en una casa silenciosa. ¿Sigue sin dar falsas alarmas cuando alguien empieza a cocinar con la radio encendida? Y, al revés, ¿sigue detectando un incendio de verdad?

La validación con ruido **no es un experimento nuevo**: es la misma validación de siempre, repetida con ruido sumado a las señales.

---

## 2. Vocabulario

| Palabra | Significado sencillo |
|---|---|
| **Ruido blanco gaussiano** | Ruido aleatorio sin ningún patrón, como el "siseo" de un sensor. Es el tipo de ruido que se usa. |
| **SNR (dB)** | "Relación señal/ruido": cuánto más fuerte es la señal que el ruido. **Más dB = menos ruido.** Cada 20 dB el ruido se divide por 10 en amplitud (por 100 en potencia). |
| **SNR absoluto** | El ruido de un nivel tiene **el mismo tamaño en todos los casos**, como un sensor real. Se mide respecto a una señal de referencia fija. |
| **Caso de referencia** | El **caso inestable más débil** del experimento (en n12000, kappa 1.082). Sirve para fijar el tamaño del ruido: "SNR 20 dB" = ruido con el 1 % de la potencia de ese chatter. |
| **Copia ruidosa** | Un caso con ruido de un nivel y una realización concretos. Es una señal nueva. |
| **Realización** | Una "tirada" de números aleatorios. Con otra tirada el ruido es distinto aunque tenga el mismo tamaño. |
| **Entrenamiento limpio** | Los umbrales de los indicadores se calcularon con datos sin ruido y **no se recalculan**. |
| **SNR de quiebre** | El mayor SNR al que el indicador empieza a empeorar de forma clara. Es "a partir de cuánto ruido falla". |

---

## 3. El flujo paso a paso

```
 casos limpios ──(1) añadir ruido──> copias ruidosas ──(2) indicadores──> ¿alarmó? por copia
                                                                              │
 etiquetas limpias + validación limpia ───────(3) comparar con la verdad──────┘
                                                                              │
                         (4) métricas por nivel de ruido ──> (5) resumen + figuras
```

### Paso 1: añadir ruido (`doe_noise.py`)
- Se eligen los casos de validación que recibirán ruido: todos o una lista. En la primera prueba, **12 de los 22** de n12000: 5 estables, 1 gris y 6 inestables.
- A cada caso se le suma ruido a varios **niveles de SNR**. En la primera prueba: **80, 60, 40, 30, 20 y 10 dB**.
- El tamaño del ruido se calcula **una sola vez** a partir de la señal del caso de referencia y vale para todos los casos. Ejemplos en n12000 (desplazamiento):

  | SNR | Tamaño del ruido |
  |---|---|
  | 60 dB | 1.3e-8 m |
  | 40 dB | 1.3e-7 m |
  | 20 dB | 1.3e-6 m |
  | 10 dB | 4.0e-6 m |

- **Por qué importa que sea absoluto:** los casos estables vibran muy poco (la mayoría entre 1e-8 y 5e-8 m; el de kappa 1.05, unos 8e-7 m). A 60 dB el ruido ya es del tamaño de toda su vibración. Un caso inestable, que vibra ~1e-5 m o más, sigue viéndose muy por encima del ruido.
- Cada combinación (caso, nivel, realización) es una **copia ruidosa**. Con 12 casos, 6 niveles y 3 realizaciones son **216 copias** (2.1 GB).

### Paso 2: correr los indicadores (`doe_indicators.py`)
- Los 4 indicadores corren sobre cada copia con los **umbrales del entrenamiento limpio**: no se recalibran. Son 216 × 4 = **864 corridas**.
- Cada corrida responde: ¿alarmó en algún momento? Sí o no.

### Paso 3: la verdad (`validate_noise.py`)
- La respuesta correcta de una copia es la **etiqueta del caso limpio** del que viene. El ruido no cambia lo que el caso es: un caso inestable sigue siendo inestable. Lo que se prueba es al indicador.
- Los casos grises se tratan con el mismo modo que la validación limpia (`ignore`, `stable` o `unstable`).

### Paso 4: las métricas
- Para cada **nivel** y cada **realización** se hace la misma cuenta que en la validación limpia (guía de métricas, sección 2): aciertos con alarma en inestables, silencios correctos en estables, falsas alarmas y fallos, y de ahí TPR, TNR y balanced accuracy.
- Eso es **una validación completa de los 12 casos**. Como hay 6 niveles y 3 realizaciones, son 18 validaciones por indicador.

### Paso 5: resumen y figuras
- Las realizaciones se resumen con **media, mínimo y máximo** (ver sección 5).
- Se dibuja una curva por indicador: métrica contra SNR, con el punto "sin ruido" como referencia.

### Un ejemplo completo (green, n12000)

| SNR | Lo que pasa | Balanced accuracy |
|---|---|---|
| sin ruido | de los 5 estables, 4 en silencio y 1 falsa alarma; los 6 inestables se detectan | 0.90 |
| 80, 60, 40, 30 dB | igual que sin ruido | 0.90 |
| **20 dB** | los 5 estables alarman | **0.50** |
| 10 dB | igual | 0.50 |

El quiebre de green es **20 dB**: a partir de ahí el ruido hace sonar la alarma en los casos estables.

---

## 4. Qué se puede configurar y cómo se corre

### Lo que se puede cambiar
| Opción | Qué controla | Valor de la primera prueba |
|---|---|---|
| `cases` | Casos con ruido: `all` o una lista | 12 casos elegidos |
| `snr_list` | Niveles de SNR en dB | 80, 60, 40, 30, 20, 10 |
| `realizations` | Realizaciones por caso y nivel | 3 |
| `snr_ref_case` | Caso de referencia del SNR: `auto` (el inestable de menor kappa) o uno concreto | `auto` (case_011) |
| `seed` | Semilla: la misma semilla da exactamente el mismo ruido | 42 |
| `signals` | Señales con ruido | `Axial_disp` y `Axial_vel` |
| `workers` (indicadores) | Procesos en paralelo | 3 (con 6 se agota la memoria) |
| Modo de grises | Cómo se tratan los casos grises | el de la validación limpia |

### Lo que está fijo (por decisión)
Ruido blanco gaussiano; SNR absoluto; ruido independiente en desplazamiento y velocidad; entrenamiento limpio; la caída de 0.05 que define el quiebre.

### Cómo se corre
En la aplicación, añadiendo al experimento las etapas **Noise**, **Noise indicators** y **Noise validate** (formulario "Noise" para los parámetros). Por línea de comandos:

1. `doe_noise.py --doe_results <doe_results.h5> --labels <etiquetas.h5> --cases ... --realizations 3 --out <archivo de ruido>`
2. `doe_indicators.py --experiment <exp> --doe_results <archivo de ruido> --out <resultados> --no-signals --workers 3`
3. `validate_noise.py --noise_ind <resultados> --clean <validación limpia> [--realizations 0]`
4. `validation_figures.py --results <validación con ruido>`

Opciones de `doe_indicators.py` útiles en estas corridas largas: `--realizations K [K …]` (solo esas realizaciones), `--only green rms_cv` (solo esos indicadores; el resto del archivo no se toca) y `--resume` (salta lo que ya está calculado: sirve para retomar tras un corte o para ampliar de 1 a 3 realizaciones sin repetir la 0). `validate_noise.py --realizations K` puntúa solo esas realizaciones y deja en el archivo cuáles fueron.

### Tamaño y tiempo (medidos en n12000)
- Archivo de ruido: **2.1 GB**, se genera en segundos.
- Resultados de indicadores (sin señales): **65 MB**. Validación con ruido: **0.4 MB**.
- Indicadores: **unas 2.5 horas con 3 procesos** (con 1 proceso, unas 5 horas). Es la parte lenta.
- Ampliación a 22 casos × 5 realizaciones de ruido (6.45 GB, generado en 44 s): los indicadores de la realización 0 tardaron ~4.7 h en total con 3 procesos (Green, a 10 dB, ~6 min por tarea, domina); con las 5 realizaciones habrían sido ~8 h.
- Si el equipo se suspende, la corrida se corta pero lo calculado se conserva y se puede reanudar con `--resume` (o pasando solo los grupos que faltan).

---

## 5. Cómo se calculan las métricas con ruido

### 5.1 Métricas por nivel y por realización
Son **las mismas** de la guía de métricas (TPR, TNR, balanced accuracy, MCC, AUC, fracción de alarma en estables, anticipación `t_det / t_onset`). Cambia solo que se calculan para cada nivel de ruido.

### 5.2 Por qué hay realizaciones, y cómo se juntan
- **Qué son:** repeticiones del paso 1 con otra tirada de ruido (misma intensidad, distinto detalle).
- **Para qué sirven:** saber si el resultado depende de la suerte del ruido, sobre todo cerca del límite donde el indicador empieza a fallar. Con una sola tirada no se puede saber.
- **Cómo se juntan:** para cada nivel se calcula la métrica en cada realización y se reporta la **media**, el **mínimo** y el **máximo**. Es la banda de las gráficas.
- **Lo que NO se hace:** mezclar las realizaciones como si fueran casos nuevos. Son el mismo caso con otro ruido, y mezclarlas haría que los resultados parecieran más seguros de lo que son.
- **No agregan gráficas:** las figuras de ruido son 3, con 1 realización o con 10. Solo agregan tiempo de cómputo (×3 con 3 realizaciones).
- **Cuántas usar:** en la primera prueba las 3 coincidieron casi siempre. Para explorar basta **1**; conviene subir a 3 o más cuando se afine alrededor del quiebre.

### 5.3 La referencia sin ruido
El punto "sin ruido" (el rombo hueco de las gráficas) es la **validación limpia de esos mismos casos**, puntuada con las mismas reglas. Es importante que sean **los mismos casos**: si se comparara contra los 22 casos de la validación limpia, el indicador podría parecer que "empeora" solo porque cambió la población de casos.

### 5.4 El SNR de quiebre
> **SNR de quiebre** = el mayor SNR al que la balanced accuracy **media** cae **más de 0.05** por debajo de la referencia sin ruido.

Si nunca cae, no tiene valor. Si el indicador ya falla sin ruido y con ruido queda igual de mal, el quiebre puede aparecer en el primer nivel (como rms_cv); en ese caso se lee junto con la referencia.

---

## 6. Las gráficas, una por una

Se generan con `validation_figures.py`. En el visor aparecen con el mismo nombre; en disco, en `figs_noise_validation/`. Eje horizontal siempre = SNR **de limpio (izquierda) a ruidoso (derecha)**, con el valor "clean" como primer punto.

### `noise_metrics` (4 paneles)
- **Qué muestra:** balanced accuracy, TPR, TNR y fracción de alarma en casos estables, contra SNR. Línea = media entre realizaciones; banda = mínimo–máximo; rombo hueco = sin ruido; línea vertical discontinua (panel de balanced accuracy) = SNR de quiebre.
- **Cómo leerla:** una curva plana hasta niveles muy ruidosos = indicador robusto. Si cae el **TNR** (y sube la fracción de alarma en estables), el ruido causa **falsas alarmas**. Si cae el **TPR**, el ruido **tapa** el chatter.
- **En n12000:** el TPR se queda en 1.0 en todos los niveles; lo que cae es el TNR.

### `noise_case_matrix` (un panel por indicador)
- **Qué muestra:** filas = casos ordenados por kappa (S estable, U inestable, g gris); columnas = niveles de SNR; color = fracción de realizaciones en que el caso salió **bien** (amarillo = siempre bien, morado = siempre mal; gris = no puntuado).
- **Cómo leerla:** dice **qué casos** fallan y a qué nivel. En n12000, los inestables nunca fallan y los estables pasan juntos de acertar a fallar al cruzar el quiebre. El estable de kappa 1.05 falla incluso sin ruido (es el caso ambiguo de siempre).

### `noise_anticipation`
- **Qué muestra:** la mediana de `t_det / t_onset` contra SNR (1 = alarma justo cuando la amplitud llega al límite; menos de 1 = se anticipa).
- **Cómo leerla:** con poco ruido la anticipación se mantiene (~0.57 en green y ssq). Al llegar al quiebre se desploma hacia 0: el indicador alarma desde el principio, es decir, **falsas alarmas disfrazadas de detección temprana**. Léela siempre junto con `noise_metrics`.

---

## 7. Qué salió en la primera prueba (n12000)

12 casos, 6 niveles, 3 realizaciones. Balanced accuracy sin ruido (en esos mismos 12 casos) y SNR de quiebre:

| Indicador | Sin ruido | A qué SNR empieza a fallar |
|---|---|---|
| green | 0.90 | **20 dB** |
| ssq (SST-SVD) | 0.90 | **20 dB** |
| maxent | 0.80 | **40 dB** |
| rms_cv | 0.60 | ya falla a 80 dB (0.50) |

**Lectura:**
1. **El ruido no tapa el chatter:** el TPR se mantiene en 1.0 en todo el rango.
2. **El ruido sí provoca falsas alarmas:** como los casos estables vibran muy poco, a partir del quiebre se disparan. Con el entrenamiento limpio, los umbrales no aguantan ruido de ese tamaño.
3. **Green y ssq aguantan bastante más que maxent** (hasta ~30 dB frente a ~60 dB).
4. **rms_cv** ya fallaba sin ruido (alarma desde el arranque), así que el ruido casi no empeora algo que ya estaba mal.
5. **Las realizaciones coinciden:** las bandas son casi invisibles. El resultado de cada caso es de todo o nada y el ruido a ese nivel no lo cambia.

### 7.1 Ampliación a los 22 casos (1 realización)
Los 22 casos de n12000 (los 12 anteriores + 10 más), 6 niveles y solo la realización 0 (con 3 realizaciones los indicadores habrían tardado el triple; la 0 de la ampliación es la misma tirada de ruido que en una corrida de 5):

| Indicador | Sin ruido (22 casos) | SNR de quiebre | Falsas alarmas (TNR) |
|---|---|---|---|
| green | 0.94 | **20 dB** | 0.88 hasta 30 dB, 0.12 a 20 dB, 0.00 a 10 dB |
| ssq (SST-SVD) | 0.94 | **20 dB** | 0.88 hasta 30 dB, 0.00 desde 20 dB |
| maxent | 0.88 | **40 dB** | 0.75 hasta 60 dB, 0.00 desde 40 dB |
| rms_cv | 0.56 | 80 dB (ya 0.50 con ruido débil) | 0.00 siempre |

**Las conclusiones no cambian al pasar de 12 a 22 casos:** mismos quiebres, mismo TPR = 1.0 en todos los niveles. Sube la referencia limpia de green y ssq (0.90 → 0.94) porque ahora hay más casos estables bien clasificados.

**Observación sobre rms_cv (no es del ruido):** alarma en la primera ventana de casi todos los casos, también sin ruido (el arranque del indicador no se ignora, `warmup_ignore_alerts: false`). Por eso su TNR es 0.00 a cualquier nivel y su "quiebre" aparece en el primer nivel. Es una configuración del indicador, no se cambió aquí; con esa configuración la comparación con ruido no le dice nada nuevo.

---

## 8. Cómo contárselo a los directores

### Mensajes principales
1. **La validación con ruido es la validación de siempre con ruido sumado a las señales**, con los mismos casos, los mismos umbrales (calibrados sin ruido) y las mismas métricas.
2. **El ruido es absoluto** (mismo tamaño en todos los casos, como un sensor real) y se mide en dB respecto al chatter más débil que queremos detectar.
3. **Los indicadores siguen detectando el chatter con ruido**: el TPR no baja en ningún nivel probado.
4. **Lo que se rompe son las falsas alarmas**: green y ssq aguantan hasta ~30 dB; maxent, hasta ~60 dB; rms_cv ya fallaba sin ruido.
5. **Los umbrales se calibraron sin ruido**, y eso explica el quiebre: son sensibles porque los casos estables de entrenamiento vibran muy poco.

### Limitaciones que conviene decir antes de que pregunten
- **Pocas realizaciones:** 3 en la primera pasada (12 casos) y 1 en la ampliación (22 casos). Las diferencias pequeñas entre indicadores no son concluyentes, y con 1 realización no hay banda mín–máx.
- **Ruido idealizado:** blanco, gaussiano e independiente en desplazamiento y velocidad. No es el ruido de un sensor concreto.
- **Entrenamiento sin ruido:** no se probó qué pasa si se calibran los umbrales **con** ruido. Es otra pregunta, anotada para el futuro.
- **Un solo experimento** (1DOF, 12098 rpm).
- **Rampas de Ap:** no incluidas.

### Preguntas probables
| Pregunta | Respuesta corta |
|---|---|
| ¿Por qué el ruido es "absoluto"? | Porque un sensor real tiene el mismo ruido venga lo que venga. Si fuera relativo a cada señal, los casos más fuertes recibirían más ruido y no se compararía bien. |
| ¿Por qué no se entrena con ruido? | Es otra pregunta: aquí se mide qué pasa cuando el indicador calibrado en limpio se enfrenta a ruido. Entrenar con ruido es el siguiente experimento. |
| ¿Para qué las 3 realizaciones? | Para comprobar que el resultado no depende de la suerte de una tirada de ruido. Aquí coincidieron. |
| ¿Por qué el TPR no baja y el TNR sí? | Un chatter real es mucho más grande que el ruido; los casos estables vibran tan poco que el ruido los hace parecer chatter. |
| ¿Qué significa el SNR de quiebre? | El mayor ruido al que el indicador todavía se comporta como sin ruido (dentro de 0.05 de balanced accuracy). |

---

## 9. Glosario y dónde está cada cosa

| Término | Una frase |
|---|---|
| dB | Escala logarítmica: cada 20 dB el ruido cambia 10 veces en amplitud. |
| Copia ruidosa | Un caso con ruido de un nivel y una realización. |
| Realización | Una tirada del ruido aleatorio. |
| Referencia del SNR | El caso inestable más débil: fija el tamaño del ruido. |
| SNR de quiebre | Mayor SNR al que la balanced accuracy cae más de 0.05. |
| Robustez | Mantener el rendimiento cuando hay perturbaciones (aquí, ruido). |

| Qué | Dónde |
|---|---|
| Ruido (genera las copias) | `DOE_utils/DOE_simulacion/doe_noise.py` |
| Indicadores sobre el ruido | `DOE_utils/DOE_analisis/doe_indicators.py` (`--no-signals`) |
| Métricas con ruido | `DOE_utils/DOE_analisis/validate_noise.py` |
| Figuras | `DOE_utils/DOE_plots/validation_figures.py` (`noise_*`) |
| Plan y contrato de formatos | `docs/planes/PLAN_noise_validation.md` |
| Guía de las métricas | `docs/guias/GUIA_metricas_validacion.md` |
| Resultados de la prueba real | `validacion_figs/noise/` (en el worktree, fuera de git) |
