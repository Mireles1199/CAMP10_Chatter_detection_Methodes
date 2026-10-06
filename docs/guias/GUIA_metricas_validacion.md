# Guía de métricas y gráficas de la validación de indicadores

**Para quién es:** para entender y explicar a los directores cómo se evalúa si un indicador de chatter funciona bien. No hace falta saber estadística ni clasificación: cada concepto se explica desde cero, con un ejemplo y con los números reales de la tesis.

Los ejemplos vienen de dos experimentos con Ap constante: **n12000** (12098 rpm, 22 casos) y **n5189** (5189 rpm, 17 casos), con la regla actual (sin tolerancia de tiempo). Las rampas de Ap están fuera: se tratarán aparte.

**Cómo está organizada**
1. La idea general y una analogía para no perderse.
2. El vocabulario.
3. Las métricas (números), una por una.
4. Los casos grises y los dos escenarios (pesimista y optimista).
5. Las gráficas, una por una.
6. Cómo contárselo a los directores.
7. Glosario y dónde está cada cosa.
8. Validación con ruido (robustez).

---

## 1. La idea general

### 1.1 Qué se está comprobando

Tenemos cuatro **indicadores** (MaxEnt-SPRT, RMS-CV, SST-SVD y Green). Cada uno mira la señal de vibración de una simulación de fresado y, de vez en cuando, **levanta una alarma** que dice "esto es chatter". La pregunta de la validación es: **¿hacen sonar la alarma cuando debe sonar, y no cuando no debe?**

Para saberlo hace falta una **respuesta correcta** con la que comparar. Esa respuesta es la **etiqueta**.

### 1.2 La analogía del detector de humo

Piensa en un detector de humo en una casa:

| En la casa | En la tesis |
|---|---|
| Hay un incendio | El caso es **inestable** (hay chatter) |
| No hay incendio | El caso es **estable** (sin chatter) |
| Suena la alarma | El indicador **alarma** |
| Un detector bueno | Suena cuando hay fuego y se calla cuando no lo hay |

Un detector puede fallar de dos maneras distintas: **no sonar cuando hay fuego** (peligroso) o **sonar sin fuego** (molesto y, a la larga, hace que nadie le haga caso). La validación mide ambas cosas por separado.

### 1.3 Cómo sabemos si hay "fuego": la etiqueta

Cada simulación (un **caso**) tiene una profundidad de corte Ap fija. Se etiqueta mirando la amplitud de la vibración (|Axial_disp|) comparada con una **base** (`f_tooth·1e-3`):

| Si la amplitud máxima del caso... | Etiqueta |
|---|---|
| supera el **40 %** de la base | **inestable** |
| queda por debajo del **10 %** | **estable** |
| queda **entre 10 % y 40 %** | **gris** (no está claro) |

Es una etiqueta **operacional**: dice "el chatter ya se ve en la amplitud". No usa la teoría. De hecho, cada caso trae un número `eta = Ap / Ap_límite` (qué tan por encima del límite teórico de estabilidad está), pero ese número es **solo informativo**: no se usa para etiquetar.

### 1.4 Qué hace la validación, paso a paso

1. Los indicadores ya corrieron en una etapa anterior y guardaron sus resultados.
2. La validación **no vuelve a correrlos**: lee esos resultados.
3. Para cada caso mira **una sola cosa**: ¿el indicador alarmó alguna vez? (sí / no)
4. Compara con la etiqueta y obtiene uno de cuatro resultados (sección 2).
5. Con esos resultados cuenta aciertos y fallos y calcula las métricas (sección 3).
6. Dibuja las gráficas (sección 5).

### 1.5 Un caso real, de principio a fin

Caso de n12000 con **η = 1.19** (Ap ≈ 10.25 mm):
- Su amplitud llega al 40 % de la base a los **5.29 s** (ese es su `t_onset`) → etiqueta: **inestable**.
- El indicador Green alarma por primera vez a los **3.11 s**.
- Como era inestable y alarmó: resultado **TP** (acierto).
- La alarma llegó **2.18 s antes** de que el chatter se viera en la amplitud (retraso = 3.11 − 5.29 = −2.18 s). Eso es una ventaja, no un error (ver sección 3.8).

---

## 2. Vocabulario mínimo

| Palabra | Significado sencillo |
|---|---|
| **Caso** | Una simulación con una profundidad de corte Ap fija. |
| **Etiqueta** | La respuesta correcta del caso: estable, inestable o gris. |
| **Alarma / detección** | El indicador marca que hay chatter en alguna ventana de tiempo. |
| **Primera detección (`t_det`)** | El primer instante en que el indicador alarma. |
| **`I_t`** | El número que calcula el indicador en cada ventana. La alarma suena cuando `I_t` supera un **umbral**. |
| **Umbral** | La "perilla de sensibilidad": el valor de `I_t` a partir del cual se alarma. Se calibra con casos de entrenamiento. |
| **`t_onset`** | El instante en que la amplitud del caso supera el límite del 40 %. Solo se usa para medir adelantos. |
| **Ventana** | Un trozo corto de señal (unas vueltas de la herramienta) sobre el que el indicador calcula un valor. |

### Los cuatro resultados posibles de un caso

| | El indicador **alarma** | El indicador **no alarma** |
|---|---|---|
| El caso es **inestable** | **TP** (verdadero positivo): acertó, detectó el chatter | **FN** (falso negativo): se le escapó |
| El caso es **estable** | **FP** (falso positivo): falsa alarma | **TN** (verdadero negativo): acertó, se quedó callado |

Los casos **grises** no entran en esta tabla (sección 4). Se cuentan **casos**, no ventanas.

---

## 3. Las métricas, una por una

Ejemplo de referencia: **Green en n12000**: TP = 11, FN = 0, TN = 7, FP = 1. Son 19 casos puntuables (los otros 3 son grises).

### 3.1 TPR (sensibilidad): "¿cuántos incendios detecta?"

- **Pregunta que responde:** de todos los casos inestables, ¿en cuántos alarma?
- **Cálculo:** TP / (TP + FN) = 11 / 11 = **1.00**.
- **Cómo leerlo:** 1.0 = no se le escapa ninguno. 0 = nunca detecta nada.
- **Trampa:** un indicador que alarma **siempre** también tiene TPR = 1. Por eso nunca se mira solo: va con el TNR.

### 3.2 TNR (especificidad): "¿cuántas veces se calla cuando no hay fuego?"

- **Pregunta que responde:** de todos los casos estables, ¿en cuántos NO alarma?
- **Cálculo:** TN / (TN + FP) = 7 / 8 = **0.875**.
- **Cómo leerlo:** 1.0 = nunca da falsa alarma. Valores bajos = alarma sin motivo (el detector que suena con las tostadas).
- **Trampa:** un indicador que nunca alarma tiene TNR = 1 y no sirve de nada. Va junto al TPR.

> TPR y TNR son las dos caras de la moneda: uno mide "detecta lo que debe" y el otro "no molesta cuando no debe". Un buen indicador necesita los dos altos.

### 3.3 Accuracy (exactitud): "¿qué porcentaje acierta?"

- **Cálculo:** (TP + TN) / total = (11 + 7) / 19 = **0.947**.
- **Cómo leerlo:** el porcentaje de casos bien clasificados.
- **Trampa:** si hay más casos de una clase que de otra, engaña. Con 11 inestables y 8 estables, un indicador que alarma siempre ya acertaría 11/19 = 58 %.

### 3.4 Balanced accuracy (exactitud balanceada): "el promedio justo"

- **Cálculo:** (TPR + TNR) / 2 = (1.00 + 0.875) / 2 = **0.9375**.
- **Para qué sirve:** da el mismo peso a las dos clases, aunque haya más casos de una que de otra. **Es la métrica que ordena el ranking.**
- **Cómo leerlo:** 1.0 = perfecto. **0.5 = no mejor que tirar una moneda**. Por debajo de 0.5 = peor que el azar.

### 3.5 F1: "equilibrio entre detectar y no dar falsas alarmas"

- **En palabras:** mira dos cosas a la vez: de las alarmas que sonaron, ¿cuántas eran reales? y de los incendios que hubo, ¿cuántos se detectaron? F1 las combina en un número. **No mira los TN** (los casos estables donde no pasó nada).
- **Cálculo:** 2·TP / (2·TP + FP + FN) = 22 / (22 + 1 + 0) = **0.957**.
- **Cómo leerlo:** 1.0 = perfecto, 0 = muy malo.
- **Cuándo es útil:** cuando lo que más importa es la clase "inestable".

### 3.6 MCC (coeficiente de Matthews): "la correlación entre la alarma y la realidad"

- **En palabras:** es como un coeficiente de correlación entre "lo que dijo el indicador" y "lo que era verdad". Usa los cuatro conteos (TP, FN, TN, FP) a la vez, así que es difícil de engañar.
- **Cálculo:** (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)) = (77 − 0) / √(12·11·8·7) = **0.90**.
- **Cómo leerlo:** **+1 = perfecto, 0 = azar, −1 = siempre al revés.**
- **Cuándo sale "NaN" (sin valor):** si alguna de las cuatro cantidades del denominador es 0, la fórmula no se puede calcular. Ejemplo: rms_cv en n5189 no tiene ningún TN, y su MCC queda sin definir.

### 3.7 Intervalo de confianza (de Wilson): "¿qué tan seguro es este número?"

Esto es **lo más importante para no sobreinterpretar**.

- **El problema:** con 11 casos inestables, "detectó 11 de 11" no prueba que detecte siempre. Con otros 11 casos quizá habría fallado uno.
- **Analogía:** es como una encuesta electoral con "margen de error". Si preguntas a 8 personas y 7 dicen sí, no puedes decir que el 87.5 % de todo el país dice sí. El intervalo dice en qué rango razonable está el valor real.
- **Qué es:** un rango (`_lo` a `_hi`) que **probablemente contiene el valor real**. Se calcula para TPR, TNR y accuracy con el método de Wilson al 95 %.
- **Qué significa "95 %":** si se repitiera el experimento muchas veces con casos nuevos, en el 95 % de las repeticiones el intervalo contendría el valor real.
- **De qué depende su ancho:** de cuántos casos hay. **Pocos casos = intervalo ancho.**

Ejemplos de n12000:

| Medida | Valor | Intervalo |
|---|---|---|
| TPR de Green | 11/11 = 1.00 | [0.74 – 1.00] |
| TNR de Green | 7/8 = 0.875 | [0.53 – 0.98] |
| TNR de MaxEnt | 6/8 = 0.75 | [0.41 – 0.93] |
| TNR de rms_cv | 1/8 = 0.125 | [0.02 – 0.47] |

**Cómo usarlo para comparar dos indicadores:**
- Si los intervalos **no se tocan** (Green [0.53–0.98] frente a rms_cv [0.02–0.47]): la diferencia probablemente es real.
- Si **se solapan mucho** (Green frente a MaxEnt): con estos datos **no se puede afirmar quién es mejor**. Solapar no prueba que sean iguales; solo dice que los datos no alcanzan.

### 3.7b Intervalo de balanced accuracy y de MCC: "¿y el número del ranking?"

El ranking se ordena por **balanced accuracy** y **MCC**, así que ellas también necesitan su rango de incertidumbre. No tienen una fórmula cerrada sencilla como la de Wilson, así que se calcula **por simulación**:

1. Con lo observado (por ejemplo, 11 de 11 inestables detectados y 7 de 8 estables en silencio), se generan 4000 "versiones posibles" del TPR y del TNR reales. Cada versión es plausible con esos conteos.
2. En cada versión se calcula la balanced accuracy y el MCC.
3. El intervalo es el rango que contiene el 95 % central de esos 4000 valores.

La semilla es fija: los mismos conteos dan siempre el mismo intervalo. El intervalo siempre contiene el valor medido.

Ejemplo (n12000):

| Indicador | Balanced accuracy | MCC |
|---|---|---|
| green | 0.94 [0.74 – 0.99] | 0.90 [0.53 – 0.98] |
| ssq | 0.94 [0.74 – 0.99] | 0.90 [0.53 – 0.98] |
| maxent | 0.88 [0.67 – 0.96] | 0.80 [0.40 – 0.94] |
| rms_cv | 0.56 [0.45 – 0.71] | 0.28 [−0.16 – 0.52] |

**Cómo leerlo:** los intervalos de green, ssq y maxent se **solapan mucho**: con estos casos no se puede afirmar quién es mejor entre ellos. El de rms_cv queda claramente por debajo (y su MCC incluso llega a valores negativos).

### 3.7c ¿La diferencia entre dos indicadores es real? Prueba de McNemar

Los intervalos se pueden comparar a ojo; la **prueba de McNemar** lo hace con un número. Compara a los indicadores **de dos en dos, caso por caso**:

- Se mira solo en qué casos **discrepan**: casos que el indicador A acierta y el B falla (A solo), y casos que el B acierta y el A falla (B solo).
- Si los dos fueran igual de buenos, esas discrepancias se repartirían al azar mitad y mitad. La prueba calcula qué tan raro sería el reparto observado.
- El resultado es un **valor p**. **p < 0.05** significa que la diferencia es **más que casualidad** con estos casos. **p grande** significa que no hay evidencia de diferencia (no que sean iguales).

Ejemplo (n12000):

| Pareja | A solo \| B solo | p | Lectura |
|---|---|---|---|
| green frente a ssq | 0 \| 0 | 1.00 | Idénticos en todos los casos. |
| green frente a maxent | 1 \| 0 | 1.00 | Una discrepancia: sin evidencia de diferencia. |
| green frente a rms_cv | 6 \| 0 | **0.031** | Green acierta 6 casos que rms_cv falla y ninguno al revés: diferencia real. |
| maxent frente a rms_cv | 5 \| 0 | 0.062 | Casi, pero no llega a 0.05. |

En n5189 las parejas con rms_cv dan p ≈ 0.001. **Conclusión que respalda la prueba:** rms_cv es peor que los otros tres; entre los otros tres no se puede declarar diferencia.

Un detalle: p = 1.00 en "0 | 0" no quiere decir "demostrado igual", solo que no discreparon en ningún caso.

### 3.8 ROC y AUC: "¿qué pasa si muevo la perilla de sensibilidad?"

Hasta aquí hemos evaluado cada indicador **con su umbral actual**. La curva ROC y el AUC evalúan algo más general: **si el indicador, con el umbral ideal, podría separar bien los casos estables de los inestables.**

- **El puntaje de un caso:** el valor máximo de `I_t` en toda la señal. Un caso "alarma" justo cuando ese máximo supera el umbral.
- **Curva ROC:** se baja y sube el umbral poco a poco. Con cada umbral hay una tasa de detecciones (TPR, eje vertical) y una tasa de falsas alarmas (FPR = 1 − TNR, eje horizontal). Unidas dan una curva.
  - Una curva pegada a la **esquina superior izquierda** = muy bueno (muchas detecciones con pocas falsas alarmas).
  - La **diagonal** = azar.
- **AUC (área bajo la curva):** un solo número. En palabras: *"si tomo un caso inestable y uno estable al azar, ¿con qué probabilidad el indicador da un valor más alto al inestable?"*
  - **1.0** = existe un umbral que separa perfectamente a los dos grupos.
  - **0.5** = el indicador no separa nada.
  - **Menos de 0.5** = está invertido (rms_cv en n5189 da 0.37).
- **Orientación:** algunos indicadores marcan chatter con valores **altos** de `I_t` y otros con **bajos**. El sistema lo decide mirando las propias alarmas del indicador (¿las ventanas marcadas tienen valores más altos o más bajos?), **nunca mirando las etiquetas**.
- **Punto operativo:** es el círculo en el gráfico que indica dónde está el indicador **con su umbral real**: (1 − TNR, TPR). La curva muestra lo que *podría* lograr con otro umbral; el círculo, lo que logra *hoy*. Si el círculo está lejos de la esquina pero la curva llega, el umbral está mal calibrado, no el indicador.
- **Intervalo del AUC** (método Hanley-McNeil): igual idea que el de Wilson. **Ojo:** con 11 inestables y 8 estables perfectamente separados, un AUC = 1.00 tiene intervalo [1.00, 1.00] que **no informa**: no significa "infalible", significa que con tan pocos casos el método no puede decir más.

### 3.8b El puntaje del caso, el umbral y los puntos de la curva (en detalle)

**El puntaje de cada caso.** Cada indicador da un valor `I_t` en cada ventana. El puntaje del caso es el **máximo** de `I_t` en toda la señal (el mínimo, con signo cambiado, para los indicadores que marcan chatter con valores bajos). Razón: "el indicador alarma en algún momento" es lo mismo que "su `I_t` máximo supera el umbral". El sistema decide la orientación (valores altos o bajos) mirando las alarmas del propio indicador, nunca las etiquetas.

**Tres cosas distintas se llaman "umbral":**

| # | Qué es | ¿Quién lo fija? | ¿Se mueve? |
|---|---|---|---|
| 1 | **Umbral de la etiqueta** (10 % y 40 % de la base) | El criterio operacional | No: es parte de la verdad, no del indicador. |
| 2 | **Umbral real del indicador** (con el que alarma) | El propio indicador, a partir del entrenamiento: por ejemplo μ + 3σ en SST-SVD, Green y RMS-CV; `alpha` y `beta` en MaxEnt | Sí, con sus parámetros (el 3 de "3σ", etc.). Es el que da los TP, FN, TN y FP de las tablas. |
| 3 | **Umbral hipotético de la ROC** | Es un ejercicio de cálculo | Se simula, sin tocar nada. |

**La "perilla" es una simulación.** Nadie gira nada. Como de cada caso ya se guardó su puntaje, se puede preguntar: *"si el umbral hubiera estado en X, ¿qué casos habrían alarmado?"*. Los estables que lo superan serían falsas alarmas; los inestables que no lo superan serían fallos. Subir el umbral hace al indicador más exigente (menos falsas alarmas, pero pueden escaparse inestables); bajarlo, más sensible (más detecciones, pero más falsas alarmas). Es como cambiar la regla de 3σ por 2σ o 4σ, pero solo en papel.

**El círculo de la figura `roc`** es el rendimiento con el umbral **real** (el 2): (falsas alarmas, detecciones) = (1 − TNR, TPR). Cae sobre la curva. Si el círculo está lejos de la esquina pero la curva llega a ella, el problema es el umbral y no el indicador.

**Cuántos puntos tiene la curva.** El umbral no se mueve en pasos fijos: se coloca justo en cada puntaje que existe, de mayor a menor, porque entre dos puntajes seguidos el resultado no cambia. Por eso el número de puntos es **el número de puntajes distintos + 1** (el +1 es el punto de partida (0, 0), con el umbral tan alto que nadie alarma). En n12000: 19 casos puntuables, 19 umbrales y **20 puntos**. Cada salto vale 1/11 = 0.09 en detecciones (si pasa un caso inestable) o 1/8 = 0.125 en falsas alarmas (si pasa uno estable). Esa es la resolución con tan pocos casos.

**Una salvedad con MaxEnt.** En Green, SST-SVD y RMS-CV mover el umbral sobre el puntaje equivale a cambiar su regla de σ. MaxEnt usa un contraste secuencial (SPRT, con `alpha` y `beta`) que no es un simple "`I_t` mayor que X", así que para MaxEnt la ROC es una aproximación razonable, no una réplica exacta de lo que pasaría al cambiar sus parámetros.

### 3.9 Tiempos: "¿con cuánta anticipación detecta?"

Estas cifras **no deciden si un caso es acierto o fallo**. Solo cuentan **cuánto se anticipa o retrasa** la alarma.

| Métrica | Qué es | Cómo leerla |
|---|---|---|
| `first_detection_t` | El primer instante con alarma. | Menor = detecta antes. |
| `delay_onset_s` | `t_det − t_onset`, con signo, solo en los casos detectados (en el resumen: la mediana). | **Negativo = alarmó antes de que la amplitud llegara al 40 %.** Eso es anticipación, no un error. |
| `delay_det_s` | El mismo retraso para cualquier primera detección. | Es el que usan las figuras. |
| `delay_onset_p25_s`, `delay_onset_p75_s` | Los cuartiles (25 % y 75 %) del retraso `delay_onset_s` entre los casos detectados. | La mediana sola esconde la dispersión: el rango entre estos dos números contiene a la mitad central de los casos. Green n12000: mediana −1.91 s, rango [−4.06, −0.93] s. |
| `t_ratio` (por caso) y `median_t_ratio` (por indicador) | `t_det / t_onset`: en qué fracción del tiempo hasta el límite de amplitud llega la alarma. | **1 = alarma justo cuando la amplitud llega al límite; 0.5 = a la mitad de ese tiempo; menos de 1 = anticipa.** No depende de η, así que permite comparar. n12000: green 0.58, ssq 0.57, maxent 0.37. rms_cv da 0.02, pero porque alarma en el arranque. |
| `delay_start_s` | Primera ventana marcada en la parte inestable − inicio de esa parte. | **Hoy no es un retraso real y no se debe usar**: el registro entero está etiquetado como inestable desde 0.05 s, así que es casi el tiempo de detección. Se conserva en el archivo solo por compatibilidad. |

Ejemplo (Green, n12000): el adelanto es de **−4.7 s con η 1.08** y de **−0.45 s con η 2.0**. Cuanto más inestable el corte, más rápido crece el chatter y menos margen hay.

**Por qué anticipar no es un error:** el caso está etiquetado inestable, es decir, el chatter existe. Que el indicador lo detecte cuando solo tiene 2–5 % de la amplitud límite es **sensibilidad**, no una falsa alarma.

### 3.10 Calidad de la alarma

| Métrica | Qué es | Cómo leerla |
|---|---|---|
| `alarm_fraction` (media en `mean_alarm_fraction_stable`) | De todas las ventanas de los casos **estables**, qué fracción está marcada. | **0 = nunca alarma donde no debe.** Green: 0.03. rms_cv: 0.29. |
| `persistence` (media en `mean_persistence`) | Una vez que el indicador alarma en un caso inestable, qué fracción de las ventanas siguientes **mantiene** la alarma. | **1 = mantiene la alarma encendida.** Valores bajos = alarma intermitente, poco confiable. |

### 3.11 Qué métricas NO usar

Los conteos **por ventana** (TP/FP/TN/FN y `tpr`/`tnr` por caso) están guardados, pero **no son métrica**: como todo el registro está etiquetado como inestable, el transitorio inicial cuenta como inestable y los números no se pueden interpretar.

### 3.12 Ranking

Se ordena por **balanced accuracy**, luego **MCC**, luego **AUC**. Los NaN quedan al final.

### 3.13 Resumen de resultados (sin tolerancia de tiempo)

| Indicador | n12000: bal.acc / MCC / AUC | n5189: bal.acc / MCC / AUC |
|---|---|---|
| green | 0.94 / 0.90 / 1.00 | 0.96 / 0.87 / 1.00 |
| ssq (SST-SVD) | 0.94 / 0.90 / 1.00 | 0.96 / 0.87 / 1.00 |
| maxent | 0.88 / 0.80 / 1.00 | 0.92 / 0.77 / 1.00 |
| rms_cv | 0.56 / 0.28 / 0.80 | 0.50 / NaN / 0.37 |

---

## 4. Los casos grises y los dos escenarios (pesimista y optimista)

### 4.1 Qué es un caso gris y por qué no se puntúa

Un caso es **gris** cuando su amplitud máxima queda entre el 10 % y el 40 % de la base: la etiqueta no puede decir si es estable o inestable. En n12000 hay **3 casos grises** (η 1.057 a 1.066). En n5189 no hay ninguno.

Por defecto (modo `ignore`, ver 4.8) los grises **no se puntúan**: su resultado es `n/a` (no aplica), no suman en TP/FN/TN/FP, ni en el ROC, ni en el ranking. **Razón:** no hay una respuesta correcta contra la cual comparar. Forzarles una (decir "estable" o "inestable") sería inventarla.

### 4.2 El riesgo de ignorarlos

Los grises suelen ser los casos **más difíciles** (están justo en la frontera). Si se omiten, las métricas hablan solo de casos claros, y **los indicadores pueden parecer mejores de lo que serían con todos los casos**. Por eso nunca se dice "los indicadores son perfectos": se dice "con casos de etiqueta clara...".

### 4.3 Cómo se reporta lo omitido

Para que la omisión no esconda nada, cada indicador reporta (en `/metrics` y en el CSV):

| Métrica | Qué es |
|---|---|
| `n_gray` | Cuántos casos grises hay. |
| `n_gray_alarm` y `gray_alarm_rate` | Cuántos de esos alarman, y qué fracción. |
| `gray_as_stable_*` (TPR, TNR, balanced_accuracy, MCC) | Las métricas **si todos los grises se contaran como estables**. |
| `gray_as_unstable_*` (TPR, TNR, balanced_accuracy, MCC) | Las métricas **si todos los grises se contaran como inestables**. |

### 4.4 Los dos escenarios, explicados

Como no sabemos qué eran realmente, se calculan **los dos extremos**:

| | Si el gris **alarma** | Si el gris **no alarma** |
|---|---|---|
| **Escenario pesimista** (gris = estable) | cuenta como **falsa alarma (FP)** | cuenta como acierto (TN) |
| **Escenario optimista** (gris = inestable) | cuenta como **detección (TP)** | cuenta como fallo (FN) |

- Son **cotas**: la verdad está en algún lugar **entre** los dos, no son un veredicto.
- El nombre "optimista" es válido mientras el indicador alarme en los grises (como en n12000, donde alarman todos). Si algún gris no alarmara, en el escenario optimista sería un fallo.

### 4.5 Resultados en n12000

Los 3 grises: los **cuatro indicadores alarman en los tres**.

| Indicador | Sin grises (base) TNR / bal.acc / MCC | **Pesimista** (grises = estables) TNR / bal.acc / MCC | **Optimista** (grises = inestables) TPR / bal.acc / MCC |
|---|---|---|---|
| green | 0.88 / 0.94 / 0.90 | **0.64 / 0.82 / 0.68** | **1.00 / 0.94 / 0.90** |
| ssq | 0.88 / 0.94 / 0.90 | **0.64 / 0.82 / 0.68** | **1.00 / 0.94 / 0.90** |
| maxent | 0.75 / 0.88 / 0.80 | **0.55 / 0.77 / 0.61** | **1.00 / 0.88 / 0.81** |
| rms_cv | 0.12 / 0.56 / 0.28 | **0.09 / 0.55 / 0.22** | **1.00 / 0.56 / 0.29** |

### 4.6 Cómo leerlo y cómo decirlo

- En el escenario optimista **casi nada cambia** (las alarmas en los grises se vuelven aciertos).
- En el pesimista **sí baja**: el TNR de Green pasa de 0.875 a 0.64.
- **El orden de los indicadores no cambia** en ningún escenario: Green y ssq arriba, luego MaxEnt, rms_cv al final.

> Frase para los directores: *"Sobre los casos con etiqueta clara, Green detecta todos los inestables y se calla en la gran mayoría de los estables. En los 3 casos ambiguos (η ≈ 1.06) todos los indicadores alarman. Si esos 3 se contaran como estables, el TNR de Green bajaría de 0.88 a 0.64; si se contaran como inestables, no cambiaría. Las conclusiones sobre el orden de los indicadores no dependen de cómo se cuenten."*

### 4.7 Dónde aparece esto

| Dónde | ¿Aparece? |
|---|---|
| **Cálculos** (`/metrics` y `_metrics.csv`) | **Sí**, todas las métricas de la tabla de arriba. |
| **Resumen de consola** | Sí, una línea por indicador cuando hay grises. |
| **Figuras** | Los casos grises **se dibujan** (círculo hueco gris) en `detection_time`, `score_vs_eta` y `score_dist`, y salen en gris en `case_matrix`. **Las cotas pesimista y optimista todavía no se grafican**: hoy solo están como números. |
| **Interfaz** (tarjeta de Validate, pestaña Compare) | Pendiente: ya se pasó la petición a wt-interfaz. |
| **Calcular todo en un modo** (ver 4.8) | **Sí**, desde la línea de comandos (`--gray ignore\|stable\|unstable`); el selector en la interfaz está pendiente (wt-interfaz). |

### 4.8 Elegir cómo se tratan los grises (`--gray`): tres modos

Además de las cotas, se puede **recalcular todo** (conteos, métricas, ranking, ROC y figuras) con un tratamiento concreto de los grises:

| Modo | Qué hace con los casos grises | Cuándo usarlo |
|---|---|---|
| `ignore` (por defecto) | No se puntúan (resultado `n/a`). | La visión principal: solo casos con etiqueta clara. |
| `stable` (pesimista) | Se cuentan como **estables**: si alarman, falsa alarma. | Ver el peor caso: ¿qué pasa si esos casos eran en realidad estables? |
| `unstable` (optimista) | Se cuentan como **inestables**: si alarman, detección. | Ver el mejor caso. |

Cada modo se guarda en **su propio archivo**, para no pisar los otros:

| Modo | Archivo de resultados | Carpeta de figuras |
|---|---|---|
| `ignore` | `doe_validation_results.h5` | `figs_validation/` |
| `stable` | `doe_validation_results_gray-stable.h5` | `figs_validation_gray-stable/` |
| `unstable` | `doe_validation_results_gray-unstable.h5` | `figs_validation_gray-unstable/` |

Las figuras de los modos `stable` y `unstable` llevan una nota gris abajo ("gray cases counted as stable / unstable") para que nadie las confunda con la visión principal. Los casos grises se dibujan siempre con círculo hueco gris.

**Ejemplo (n12000, Green):**

| Modo | TP / FN / TN / FP | TNR | Balanced accuracy | MCC |
|---|---|---|---|---|
| `ignore` | 11 / 0 / 7 / 1 | 0.88 | 0.94 | 0.90 |
| `stable` | 11 / 0 / 7 / 4 | 0.64 | 0.82 | 0.68 |
| `unstable` | 14 / 0 / 7 / 1 | 0.88 | 0.94 | 0.90 |

En los tres modos el **orden** de los indicadores es el mismo (Green y ssq, MaxEnt, rms_cv). Las columnas `gray_as_stable_*` y `gray_as_unstable_*` de 4.3 se siguen reportando en cualquier modo, siempre sobre los casos que no son grises.

---

## 5. Las gráficas, una por una

Se generan con `DOE_utils/DOE_plots/validation_figures.py`. En el visor aparecen como "Validation — <nombre>"; en disco, en `<carpeta del .h5>/figs_validation/<nombre>.png`.

Los colores son una paleta segura para daltónicos. Verde = acierto TP, azul = TN, amarillo = FP, rosa = FN, gris = no puntuado.

### 5.1 `ranking`: barras de balanced accuracy, MCC y AUC
- **Qué muestra:** tres barras por indicador, ordenados del mejor al peor. La barrita de error sobre el AUC es su intervalo.
- **Cómo leerla:** más alto es mejor. Buscar el orden y los saltos grandes.
- **Las barras de error** son los intervalos de 95 % de las tres métricas (balanced accuracy y MCC por simulación, AUC por Hanley-McNeil; ver 3.7b y 3.8).
- **Qué vemos:** Green y ssq empatados arriba; MaxEnt un poco abajo; rms_cv claramente peor. Pero las barras de Green, ssq y MaxEnt se solapan mucho: **no se puede afirmar quién es mejor entre los tres**.
- **Qué no concluir:** una barrita de error plana en 1.0 (AUC) no es certeza (sección 3.8). Para ver si una diferencia es más que casualidad, ver `pairwise_test`.

### 5.2 `roc`: curvas ROC con el punto operativo
- **Ejes:** horizontal = falsas alarmas (FPR); vertical = detecciones (TPR).
- **Cómo leerla:** cuanto más pegada la curva a la esquina superior izquierda, mejor. El **marcador hueco** (uno distinto por indicador: círculo, cuadrado, rombo, triángulo) es el indicador con su umbral real. Dos indicadores con el mismo resultado se ven uno dentro del otro.
- **Qué vemos:** Green, ssq y MaxEnt tocan la esquina (AUC 1.00): existe un umbral perfecto. Pero el círculo de MaxEnt queda a la derecha de la esquina: su umbral actual da falsas alarmas que otro umbral evitaría. rms_cv tiene curva escalonada y un círculo lejano (FPR 0.875).
- **Qué no concluir:** las curvas de Green y ssq se superponen y una tapa a la otra; no es un error.

### 5.3 `tpr_tnr`: TPR y TNR con intervalo de Wilson
- **Cómo leerla:** dos puntos por indicador (naranja = TPR, azul = TNR) con su barra de error. **Barra larga = pocos casos = poca certeza.**
- **Qué vemos:** todos detectan los inestables (TPR = 1). La diferencia entre indicadores está en el TNR (falsas alarmas). Los intervalos de Green, ssq y MaxEnt se solapan.
- **Conclusión honesta:** rms_cv es claramente peor, pero entre los otros tres no se puede declarar un ganador.
- **Es la figura que mejor muestra la incertidumbre.**

### 5.4 `confusion`: matriz de confusión 2×2 por indicador
- **Cómo leerla:** filas = la verdad (inestable arriba, estable abajo); columnas = lo que hizo el indicador (alarma / sin alarma). Cada celda lleva su conteo.
- **Qué vemos:** Green: 11 aciertos con alarma, 0 omitidos, 1 falsa alarma, 7 aciertos en silencio. rms_cv: 7 falsas alarmas y solo 1 acierto en silencio.
- **Para qué sirve:** lectura inmediata de dónde pierde cada indicador.

### 5.5 `case_matrix`: resultado por caso e indicador
- **Cómo leerla:** cada columna es un caso, ordenado por η (S = estable, U = inestable, g = gris); cada fila, un indicador; el color es el resultado.
- **Qué vemos:** los amarillos (falsas alarmas) de Green, ssq y MaxEnt aparecen en η 1.03–1.05, justo antes de los grises. rms_cv falla en casi todos los estables, incluso con η 0.5.
- **Para qué sirve:** ver **dónde** (en qué zona de η) falla cada uno.

### 5.6 `detection_time`: primera detección frente al inicio por amplitud
- **Ejes:** horizontal = η; vertical (escala logarítmica) = tiempo en segundos.
- **Cómo leerla:** la **línea negra** es `t_onset` (cuándo la amplitud llega al 40 %). Los **círculos llenos** son detecciones en casos inestables; las **"x"**, en casos estables; el **círculo hueco**, en casos grises.
- **Qué vemos:** todos los círculos llenos están **por debajo de la línea negra**: los indicadores alarman antes de que el chatter sea visible. rms_cv está pegado al fondo (0.07 s) en todos los casos, estables incluidos: eso es la primera ventana, no una detección real.

### 5.7 `delay_vs_eta`: retraso con signo frente a η
- **Cómo leerla:** el eje vertical es `t_det − t_onset`. **La línea en 0 = el instante en que la amplitud llega al límite.** Negativo = alarmó antes.
- **Qué vemos:** todos los retrasos son negativos y se acercan a 0 al aumentar η (el chatter crece más rápido y deja menos margen). MaxEnt anticipa más que Green y ssq; rms_cv "anticipa" aún más, pero porque alarma desde el arranque.

### 5.8 `detection_amp`: cuánta amplitud hay cuando alarma
- **Cómo leerla:** el eje vertical es |Axial_disp| como % de la base en el momento de la primera alarma, en los casos inestables. Líneas de referencia: 40 % (límite de la etiqueta) y 10 % (límite inferior).
- **Qué vemos:** los indicadores alarman con **2–5 %** de la base, muy por debajo de los dos límites. Es evidencia directa de que detectan chatter incipiente.
- **Ojo:** la curva de rms_cv refleja el arranque, no crecimiento.

### 5.9 `score_vs_eta`: el puntaje de cada caso frente a η
- **Cómo leerla:** un panel por indicador. Azul = casos estables, naranja = inestables, círculo hueco gris = grises. La línea punteada es η = 1.
- **Qué vemos:** esto es lo que el ROC "umbraliza". En Green, ssq y MaxEnt hay un salto de órdenes de magnitud entre estables e inestables (por eso AUC = 1). En rms_cv las nubes se mezclan.

### 5.10 `score_dist`: distribución del puntaje, por tipo de caso
- **Cómo leerla:** tres columnas por indicador (estable, inestable, gris), escala logarítmica. **Cuanto más separadas las nubes, mejor.**
- **Qué vemos:** Green, ssq y MaxEnt separan completamente estables de inestables, y **los grises quedan en medio** (en Green y ssq, entre ambas nubes): coherente con que son casos ambiguos. Los puntos "estables" más altos son los casos de η 1.03–1.05. En rms_cv hay solape.
- **Ojo:** si el indicador usa valores negativos (MaxEnt), la escala es mixta.

### 5.10b `anticipation`: cuánto se anticipa, independiente de η
- **Ejes:** horizontal = η; vertical = `t_det / t_onset` (ver 3.9), solo casos inestables detectados. La línea punteada en 1 = "alarma cuando la amplitud llega al límite".
- **Cómo leerla:** más abajo = más anticipación. Una línea **plana** significa que el indicador alarma siempre en la misma fracción del camino, sea cual sea η.
- **Qué vemos:** Green y ssq alarman a ~0.5–0.6 del tiempo; MaxEnt a ~0.4; las tres casi planas. rms_cv queda pegado a 0 porque alarma en el arranque, no porque detecte antes.
- **Para qué sirve:** es la forma más limpia de decir "cuánto antes" sin depender de los segundos concretos de cada caso.

### 5.10c `pairwise_test`: ¿las diferencias entre indicadores son más que casualidad?
- **Cómo leerla:** matriz con los indicadores en filas y columnas. Cada celda lleva el valor p de la prueba de McNemar (3.7c) y dos números: `fila acierta y columna falla | al revés`. **Celda naranja = p < 0.05** (diferencia real).
- **Qué vemos:** solo las celdas con rms_cv son naranjas; entre green, ssq y maxent no hay ninguna.
- **Qué no concluir:** una celda gris no demuestra que dos indicadores sean iguales; solo que estos casos no alcanzan para distinguirlos.

### 5.10d `gray_bounds`: las cotas de los casos grises
- **Cómo leerla:** dos paneles (balanced accuracy y MCC). Por indicador, un cuadrado azul = si los grises fueran estables (pesimista), un triángulo naranja = si fueran inestables (optimista), y un punto negro = el valor de este archivo. La línea gris entre los extremos es el rango posible.
- **Qué vemos:** el rango de green y ssq es el más amplio en balanced accuracy (≈ 0.82 a 0.94); el de rms_cv, muy estrecho. El orden de los indicadores es el mismo en los dos extremos.
- **Si no hay casos grises** (n5189) la figura no se genera.

### 5.11 `alarm_quality`: alarmas en estables y persistencia
- **Cómo leerla:** izquierda = fracción de ventanas con alarma en casos estables (**menor es mejor**); derecha = persistencia de la alarma una vez que arranca (**mayor es mejor**).
- **Qué vemos:** Green y ssq casi no alarman donde no deben (0.03); rms_cv, 0.29. Los tres primeros mantienen la alarma (persistencia 1.0).

### 5.12 `training_coverage`: entrenamiento frente a validación
- **Cómo leerla:** ejes = η y rpm. Puntos de color = casos del entrenamiento (azul estable, naranja inestable); "x" = casos de validación.
- **Para qué sirve:** si la validación cae dentro de la región entrenada, es una validación "en dominio". Si cae fuera, hay extrapolación y hay que decirlo.
- **Requisito:** la validación debe haberse corrido con `--reference`; si no, la figura da un aviso en vez de dibujarse.

### 5.13 `compare`: dos experimentos, A frente a B
- **Cómo leerla:** un tramo por indicador: círculo hueco = A, círculo lleno = B, en balanced accuracy, MCC y AUC. **Tramo largo = cambió mucho.**
- **Qué vemos:** al pasar de 12000 a 5189 rpm, Green y ssq casi no cambian; rms_cv cae.

---

## 6. Cómo contárselo a los directores

### Mensajes principales
1. **La validación compara cada indicador con una etiqueta operacional de amplitud, caso por caso:** acierta si detecta un caso inestable y no alarma en uno estable.
2. **Green, ssq y MaxEnt detectan todos los casos inestables** (TPR = 1) y **separan perfectamente** las dos clases (AUC = 1). **rms_cv no:** alarma casi siempre desde el arranque.
3. **Los indicadores detectan antes de que el chatter sea visible:** con 2–5 % de la base, entre ~0.4 y ~7 s antes del límite del 40 %, según η y el indicador. Es una ventaja, no un error.
4. **Los pocos falsos positivos** de Green, ssq y MaxEnt ocurren con η 1.03–1.05: casos teóricamente inestables que la etiqueta operacional aún llama estables.
5. **Las diferencias finas entre Green y ssq no son concluyentes** con 11–19 casos: los intervalos se solapan.
6. **Los casos grises no se puntúan, pero se reportan:** todos los indicadores alarman en ellos, y el ranking no cambia con ninguno de los dos escenarios extremos.

### Limitaciones que conviene decir antes de que pregunten
- **Pocos casos** (11–19): intervalos anchos, AUC = 1.00 sin intervalo informativo.
- **La etiqueta es operacional, no teórica:** depende del horizonte de la simulación y del 40 % elegido.
- **Etiqueta de todo el registro:** no se puede medir "retraso desde el inicio del chatter", solo anticipación frente al límite de amplitud.
- **Los casos grises se omiten** del conteo principal (con cotas pesimista/optimista reportadas).
- **rms_cv:** el problema es de configuración del indicador (arranque), no necesariamente del método.
- **Rampas de Ap:** no incluidas; se evaluarán aparte.

### Preguntas probables y respuesta corta
| Pregunta | Respuesta |
|---|---|
| ¿Por qué un indicador que alarma "antes" no se cuenta como error? | Porque ese caso está etiquetado inestable: el chatter existe. Detectarlo cuando solo tiene 2–5 % de la amplitud límite es sensibilidad. El tiempo se reporta aparte. |
| ¿Por qué balanced accuracy y no accuracy? | Hay más casos inestables que estables: balanced accuracy da el mismo peso a las dos clases. |
| ¿AUC = 1 significa que no falla nunca? | No: significa que existe un umbral que separa perfectamente en estos datos. Con pocos casos el intervalo no informa. |
| ¿Para qué MCC y F1? | MCC usa los cuatro conteos y no se infla con clases desbalanceadas; F1 se centra en la clase inestable. |
| ¿Qué tiene de raro rms_cv? | Alarma en la primera ventana (t ≈ 0.07 s) en casi todos los casos; eso hunde su TNR. |
| ¿No están inflados los resultados al quitar los grises? | Es una preocupación válida. Por eso se reportan los grises y los escenarios pesimista y optimista: el orden de los indicadores no cambia, aunque el TNR de Green baje a 0.64 en el pesimista. |
| ¿Qué es un intervalo de confianza? | El rango razonable del valor real, dado que se midió con pocos casos. Más casos, rango más estrecho. |

---

## 7. Glosario y dónde está cada cosa

### Glosario rápido
| Término | Una frase |
|---|---|
| Alarma | El indicador marca "chatter" en una ventana. |
| AUC | Probabilidad de que el indicador dé más puntaje a un caso inestable que a uno estable. 1 = separa perfecto, 0.5 = azar. |
| Balanced accuracy | Promedio de TPR y TNR; da igual peso a las dos clases. |
| Caso | Una simulación con Ap constante. |
| Caso gris | Amplitud entre 10 % y 40 % de la base; sin respuesta correcta clara; no se puntúa. |
| Etiqueta | La respuesta correcta del caso (estable, inestable, gris), por amplitud. |
| F1 | Combina cuántas alarmas eran reales y cuántos inestables se detectaron. |
| FN / FP / TN / TP | Fallo (no detectó), falsa alarma, silencio correcto, detección correcta. |
| Intervalo de confianza | Rango razonable del valor real de una métrica medida con pocos casos. |
| η | Qué tan por encima del límite teórico de estabilidad está el corte. Solo informativo. |
| MCC | Correlación entre lo que dijo el indicador y la verdad, de −1 a +1. |
| Operacional | Basado en lo que se observa (amplitud visible), no en la teoría. |
| Punto operativo | Dónde está el indicador en la curva ROC con su umbral real. |
| ROC | Curva que muestra detecciones frente a falsas alarmas al mover el umbral. |
| `t_onset` | Instante en que la amplitud llega al 40 % de la base. |
| TPR / TNR | Fracción de inestables detectados / fracción de estables sin alarma. |
| Umbral | Valor de `I_t` a partir del cual se alarma (la "perilla de sensibilidad"). |

### Dónde está cada cosa
| Qué | Dónde |
|---|---|
| Código de métricas | `DOE_utils/DOE_analisis/validate_indicators.py` |
| Código de figuras | `DOE_utils/DOE_plots/validation_figures.py` |
| Informe técnico (decisiones, estado) | `docs/informes/INFORME_validacion.md` |
| Esta guía | `docs/guias/GUIA_metricas_validacion.md` |
| Resultados | `doe_validation_results.h5` y `doe_validation_results_metrics.csv` (junto a los datos de cada experimento) |
| Figuras guardadas | `<carpeta del .h5>/figs_validation/` |
| Validación con ruido | `DOE_utils/DOE_simulacion/doe_noise.py` (ruido), `DOE_utils/DOE_analisis/validate_noise.py` (métricas), plan en `docs/planes/PLAN_noise_validation.md` |
| Figuras con ruido | `<carpeta del .h5>/figs_noise_validation/` |

---

## 8. Validación con ruido (robustez)

> Hay una guía propia, más detallada y paso a paso: **`GUIA_validacion_ruido.md`**. Esta sección es el resumen.

### 8.1 La pregunta

Todo lo anterior usa señales **limpias** de simulación. Un sensor real añade **ruido**. La pregunta es: *¿un indicador cuyo umbral se calibró con datos limpios sigue funcionando cuando la señal tiene ruido?* Es una pregunta de **robustez**.

Analogía: un detector de humo calibrado en una casa sin cocina, ¿sigue sin dar falsas alarmas cuando alguien empieza a cocinar?

### 8.2 Qué se hace

1. Se toman los casos de validación (un subconjunto representativo en la primera pasada) y a cada uno se le suma **ruido blanco gaussiano** a varios niveles.
2. Los indicadores se corren sobre esas señales ruidosas **con el mismo umbral del entrenamiento limpio**.
3. Cada copia ruidosa se compara con la **etiqueta del caso limpio** del que viene: el ruido no cambia la verdad (el caso sigue siendo estable o inestable); solo pone a prueba al indicador.
4. Se calculan las mismas métricas de la sección 3, **por nivel de ruido**.

### 8.3 El SNR (relación señal/ruido) y por qué es "absoluto"

- **SNR en dB:** cuánto más fuerte es la señal que el ruido. **Más dB = menos ruido.** Cada 10 dB el ruido se divide por ~3 en amplitud (por 10 en potencia): 20 dB = ruido de potencia 1 % de la señal; 40 dB = 0.01 %.
- **Absoluto:** el ruido de un nivel tiene **el mismo tamaño en todos los casos**, como pasa con un sensor real. Se mide respecto a una señal de referencia fija: la del **caso inestable más débil** (el chatter más pequeño que queremos detectar). "SNR 20 dB" significa "ruido con el 1 % de la potencia de ese chatter".
- **Consecuencia importante:** los casos estables vibran muy poco (en n12000, entre ~15 y ~1000 veces menos que la referencia en amplitud RMS). Por eso, a un nivel donde el chatter todavía se ve perfectamente, el ruido ya puede ser **más grande que toda la vibración de un caso estable**. Ahí es donde se esperan las primeras falsas alarmas.
- Niveles de la primera pasada: **80, 60, 40, 30, 20 y 10 dB**.

### 8.4 Realizaciones: por qué se repite con varias semillas

El ruido es aleatorio: con otra "tirada" (semilla) el resultado de un caso puede cambiar, sobre todo cerca del límite. Por eso cada nivel se repite con **3 realizaciones** independientes.

- Cada realización es una validación completa de todos los casos a ese nivel.
- Se reporta la **media** entre realizaciones y su **rango (mínimo–máximo)**.
- Las realizaciones **no se cuentan como casos nuevos** (son el mismo caso con otro ruido): mezclarlas haría que los resultados parecieran más seguros de lo que son.

### 8.5 Qué métricas nuevas aparecen

| Métrica | Qué es |
|---|---|
| Métricas por nivel (`/by_snr`) | Las mismas de la sección 3 (balanced accuracy, TPR, TNR, MCC, AUC, fracción de alarma en estables, anticipación), con media, mínimo y máximo entre realizaciones. |
| **SNR de quiebre** (`snr_breakdown_db`) | El **mayor SNR** al que la balanced accuracy media cae **más de 0.05** por debajo de la limpia. Es "a partir de cuánto ruido el indicador empieza a fallar". Sin valor si nunca cae. |
| Referencia limpia (`/clean`) | Las métricas sin ruido, para comparar. |

### 8.6 Las gráficas de ruido

**`noise_metrics`** (4 paneles: balanced accuracy, TPR, TNR, fracción de alarma en estables)
- Eje horizontal: SNR, **de limpio (izquierda) a ruidoso (derecha)**. El primer punto, "clean", es el valor sin ruido (rombo hueco).
- Línea = media entre realizaciones; banda = mínimo–máximo.
- Línea vertical discontinua (panel de balanced accuracy) = SNR de quiebre de ese indicador.
- **Cómo leerla:** una curva que se mantiene plana hasta niveles muy ruidosos = indicador robusto. Si cae el **TNR** (y sube la fracción de alarma), el ruido provoca **falsas alarmas**; si cae el **TPR**, el ruido **tapa** el chatter.

**`noise_case_matrix`** (un panel por indicador)
- Filas = casos ordenados por η (S estable, U inestable, g gris); columnas = niveles de SNR.
- Color = fracción de realizaciones en que el caso salió **bien** (amarillo = siempre bien, morado = siempre mal); gris = no puntuado.
- **Cómo leerla:** muestra **qué casos** empiezan a fallar primero. Lo esperable: los estables (por las falsas alarmas) antes que los inestables.

**`noise_anticipation`**
- Anticipación `t_det / t_onset` (sección 3.9) contra el SNR.
- **Cómo leerla:** si el ruido hace alarmar antes (valor más bajo) o más tarde (más alto). Ojo: alarmar "antes" con ruido puede ser una falsa alarma disfrazada; hay que leerla junto con `noise_metrics`.

### 8.7 Resultados de la primera pasada (n12000)

12 casos (5 estables, 1 gris, 6 inestables), niveles 80, 60, 40, 30, 20 y 10 dB, 3 realizaciones. Balanced accuracy sin ruido (en esos mismos 12 casos) y SNR de quiebre:

| Indicador | Sin ruido | A qué SNR empieza a fallar |
|---|---|---|
| green | 0.90 | **20 dB** |
| ssq | 0.90 | **20 dB** |
| maxent | 0.80 | **40 dB** |
| rms_cv | 0.60 | ya falla a 80 dB (0.50) |

**Cómo se lee:**
- **El TPR no baja:** con ruido los indicadores siguen detectando todos los casos inestables. El ruido no tapa el chatter en este rango.
- **Lo que cae es el TNR:** los casos estables vibran muy poco (~1e-8 m), así que el ruido los hace parecer chatter y se disparan **falsas alarmas**. Con el entrenamiento limpio, los umbrales no aguantan ruido de ese tamaño.
- **Green y ssq aguantan mucho más que maxent** (hasta ~30 dB frente a ~60 dB). rms_cv ya fallaba sin ruido (por su arranque), así que el ruido casi no empeora algo que ya estaba mal.
- **En `noise_case_matrix`** se ve el mismo patrón por caso: los estables pasan todos a la vez de acertar a fallar al cruzar el quiebre; los inestables nunca fallan. El estable de η 1.05 falla incluso sin ruido (es el caso ambiguo de siempre).
- **En `noise_anticipation`:** al llegar al quiebre la anticipación se desploma (`t_det/t_onset` → ~0): el indicador alarma desde el principio, es decir, falsas alarmas, no detección temprana.

Las bandas mínimo–máximo entre realizaciones son casi invisibles: las 3 realizaciones coinciden en casi todo (el resultado de cada caso es de todo o nada y el ruido a ese nivel no lo cambia). Eso es buena señal, pero con 12 casos y 3 realizaciones sigue siendo una primera pasada.

### 8.8 Qué se puede y qué no se puede concluir

- **Sí:** a qué nivel de ruido cada indicador, **con su calibración limpia**, empieza a fallar, y si falla por falsas alarmas o por perder el chatter.
- **No:** cómo se comportaría si se **entrenara con ruido** (es otra pregunta, anotada para el futuro), ni cómo es el ruido de un sensor concreto (aquí es ruido blanco ideal, independiente en desplazamiento y velocidad: una simplificación declarada).
- Con 12 casos y 3 realizaciones, las diferencias pequeñas entre indicadores no son concluyentes (igual que en la sección 3.7).
