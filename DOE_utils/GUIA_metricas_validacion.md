# Guía de métricas y gráficas de la validación de indicadores

Para entender y explicar a los directores qué mide la etapa de validación, cómo leer cada número y cada figura, y qué conclusiones se pueden (y no se pueden) sacar.
Los ejemplos numéricos vienen de los experimentos de Ap constante `n12000` (12098 rpm, 22 casos) y `n5189` (5189 rpm, 17 casos), con la regla actual (sin tolerancia de tiempo). Las rampas de Ap quedan fuera: se tratarán aparte.

---

## 0. La idea en cinco líneas

1. Se simulan varios **casos** de fresado, cada uno con una profundidad de corte Ap constante distinta (y por tanto un `kappa = ap / ap_lim` distinto; `kappa` es solo informativo, no entra en la etiqueta).
2. Cada caso se **etiqueta por amplitud** (criterio operacional): si |Axial_disp| supera el 40 % de la base (`f_tooth·1e-3`) el caso es **inestable**; si no llega al límite inferior (10 %) es **estable**; entre medio es **gris** y se ignora.
3. Cada **indicador** (MaxEnt-SPRT, RMS-CV, SST-SVD, Green) recorre la señal por ventanas, calcula un valor `I_t` y levanta **alarma** cuando `I_t` cruza su umbral (calculado con el entrenamiento). Si nunca alarma, "no detecta".
4. La **validación** no vuelve a correr los indicadores: lee lo que ya guardaron y **compara con la etiqueta**, caso por caso.
5. De esa comparación salen las métricas (tablas) y las figuras.

> Mensaje clave: la etiqueta es la verdad "operacional" (lo que se ve en la amplitud). Un indicador puede detectar el crecimiento antes de que se vea; eso no es un error, y por eso el tiempo se reporta aparte y no decide el acierto.

---

## 1. Vocabulario mínimo

| Término | Significado |
|---|---|
| Caso | Una simulación con un Ap fijo. Tiene una etiqueta: estable, inestable o gris. |
| Alarma / detección | El indicador marca chatter en alguna ventana. `t_det` = primera ventana con alarma. |
| **TP** (verdadero positivo) | Caso inestable en el que el indicador alarma. |
| **FN** (falso negativo) | Caso inestable en el que nunca alarma (se le escapó). |
| **TN** (verdadero negativo) | Caso estable sin ninguna alarma. |
| **FP** (falso positivo) | Caso estable en el que alarma alguna vez (falsa alarma). |
| `I_t` | Valor del indicador en cada ventana (p. ej. el primer valor singular en SST-SVD). |
| `t_onset` | Instante en que la amplitud supera el límite de la etiqueta (40 %). Sirve para medir retrasos. |

Se cuentan **casos**, no ventanas. Los casos grises no cuentan.

---

## 2. Las métricas, una por una

Ejemplo de referencia: **green, n12000**: TP=11, FN=0, TN=7, FP=1 (19 casos puntuables, 3 grises ignorados).

### 2.1 Tasas básicas

| Métrica | Fórmula | Para qué sirve | Cómo leerla |
|---|---|---|---|
| **TPR** (sensibilidad) | TP / (TP + FN) | ¿Cuántos de los casos inestables detecta? | 1.0 = no se le escapa ninguno. Green: 11/11 = **1.00**. |
| **TNR** (especificidad) | TN / (TN + FP) | ¿Cuántos de los casos estables deja en paz? | 1.0 = nunca da falsa alarma. Green: 7/8 = **0.875**. |
| **Accuracy** | (TP + TN) / total | Porcentaje de casos bien clasificados. | Engañosa si hay más casos de una clase que de otra. Green: 18/19 = **0.947**. |
| **Balanced accuracy** | (TPR + TNR) / 2 | Igual que accuracy, pero cada clase pesa lo mismo. | **Es la que ordena el ranking.** Green: **0.9375**. 0.5 = no mejor que el azar. |
| **F1** | 2·TP / (2·TP + FP + FN) | Compromiso entre detectar y no dar falsas alarmas, **sin mirar los TN**. | 1.0 = perfecto. Útil si lo que importa es la clase "inestable". Green: **0.957**. |
| **MCC** (Matthews) | (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | Un único número que usa los cuatro conteos. Robusto a clases desbalanceadas. | −1 = siempre se equivoca, 0 = azar, +1 = perfecto. Green: **0.90**. Si el denominador es 0 (p. ej. ningún TN) no está definido (NaN). |

### 2.2 Incertidumbre: intervalo de Wilson (`_lo`, `_hi`)

TPR, TNR y accuracy llevan un **intervalo de confianza de Wilson al 95 %**. Con pocos casos, una tasa sola es muy burda: "11 de 11" suena a 100 %, pero el intervalo dice que el valor real podría ser de 0.74 a 1.00.

- Cómo leerlo: si dos indicadores tienen intervalos que se **solapan**, con estos datos **no se puede afirmar** que uno sea mejor que otro.
- Ejemplo: TNR de green 0.875 [0.53–0.98] y de maxent 0.75 [0.41–0.93]: se solapan mucho; la diferencia no es concluyente.

### 2.3 Curva ROC y AUC

El indicador alarma cuando `I_t` supera un umbral. La **ROC** responde: *¿qué pasaría con cada umbral posible?*

- **Score del caso**: el máximo de `I_t` en toda la señal (el mínimo si el indicador marca chatter con valores bajos). Superar un umbral "en algún momento" equivale a que el máximo supere el umbral.
- **Orientación** (`roc_direction`): si chatter corresponde a `I_t` alto (+1) o bajo (−1). Se decide con las propias alarmas del indicador (¿las ventanas marcadas tienen `I_t` más alto o más bajo?), **nunca con las etiquetas**.
- **AUC**: la probabilidad de que un caso inestable elegido al azar tenga mayor score que un caso estable elegido al azar.
  - 1.0 = existe un umbral que separa **perfectamente** estables de inestables.
  - 0.5 = el score no separa nada.
  - < 0.5 = la orientación está invertida (p. ej. rms_cv en n5189: 0.37).
- **Intervalo del AUC** (`AUC_lo`, `AUC_hi`, Hanley-McNeil). Con 11 inestables y 8 estables, un AUC = 1.00 tiene intervalo [1.00, 1.00] que **degenera**: no es que sea "infalible", es que la fórmula no puede decir más con estos datos.
- **Punto operativo**: (1 − TNR, TPR). Es dónde cae el indicador **con su umbral real**. La curva es lo que *podría* lograr con otro umbral.

### 2.4 Tiempos

| Métrica | Qué es | Cómo leerla |
|---|---|---|
| `first_detection_t` | Primer instante con alarma. | Menor = detecta antes. |
| `delay_onset_s` | `t_det − t_onset`, con signo, solo en los TP. En el resumen: mediana. | **Negativo = alarmó antes de que la amplitud llegara al límite del 40 %.** Eso no es un fallo: es anticipación. |
| `delay_det_s` | El mismo retraso para cualquier primera detección. | Se usa en las figuras. |
| `delay_start_s` | Primera ventana marcada en la parte inestable − inicio de esa parte. | **Hoy no es un retraso real**: el registro entero está etiquetado como inestable desde 0.05 s, así que es casi el tiempo de detección. |

Ejemplo (green, n12000): el adelanto es −4.7 s con kappa 1.08 y −0.45 s con kappa 2.0. Cuanto más inestable, más rápido crece el chatter y menos margen hay.

### 2.5 Calidad de la alarma

| Métrica | Qué es | Cómo leerla |
|---|---|---|
| `alarm_fraction` (media: `mean_alarm_fraction_stable`) | Ventanas marcadas / ventanas, en casos **estables**. | 0 = nunca alarma donde no debe. Green: 0.03; rms_cv: 0.29. |
| `persistence` (media: `mean_persistence`) | Ventanas marcadas / ventanas desde la primera detección, en los TP. | 1 = una vez que alarma, **mantiene** la alarma. Valores bajos = alarma intermitente. |

### 2.6 Qué NO usar

Los conteos **por ventana** (TP/FP/TN/FN y `tpr`/`tnr` por caso) están en el archivo pero **no son métrica**: como el registro completo está etiquetado inestable, el transitorio inicial cuenta como inestable y los números no se pueden interpretar.

### 2.6b Casos grises: qué son y cómo se reportan

Un caso es **gris** cuando su amplitud máxima queda entre el límite inferior (10 %) y el superior (40 %) de la base: la etiqueta no puede decir si es estable o inestable. Esos casos **no se puntúan**: su resultado es `n/a`, no suman en TP/FN/TN/FP, ni en el ROC, ni en el ranking. Forzarles una verdad sería inventarla.

Para que la omisión no esconda nada, cada indicador reporta (en `/metrics` y el CSV):

| Métrica | Qué es | Cómo leerla |
|---|---|---|
| `n_gray` | Cuántos casos grises hay. | 0 = no hay nada omitido (n5189). 3 en n12000. |
| `n_gray_alarm`, `gray_alarm_rate` | Cuántos de ellos alarman, y la fracción. | 1.0 = alarma en todos. |
| `gray_as_stable_TNR` (y `_TPR`, `_balanced_accuracy`, `_MCC`) | Las métricas **si todos los grises se contaran como estables** (alarma = falsa alarma). | Es la cota **pesimista**. n12000: TNR de green/ssq 0.64 (en vez de 0.875), maxent 0.55, rms_cv 0.09. |
| `gray_as_unstable_TPR` (y `_TNR`, `_balanced_accuracy`, `_MCC`) | Las métricas **si todos los grises se contaran como inestables** (alarma = acierto). | Es la cota **optimista** mientras el indicador alarme. n12000: TPR 1.00. |

Cómo decirlo: *"sobre los casos con etiqueta clara, esto; los 3 casos ambiguos (kappa ≈ 1.06) los alarman todos los indicadores; si se contaran como estables el TNR de green bajaría a 0.64, si se contaran como inestables el TPR seguiría en 1.00"*. Son **cotas**, no un veredicto. En las figuras los grises salen con un círculo hueco gris.

### 2.7 Ranking

Se ordena por **balanced accuracy**, luego **MCC**, luego **AUC** (los NaN quedan al final).

### 2.8 Resumen de resultados (sin tolerancia de tiempo)

| Indicador | n12000: bal.acc / MCC / AUC | n5189: bal.acc / MCC / AUC |
|---|---|---|
| green | 0.94 / 0.90 / 1.00 | 0.96 / 0.87 / 1.00 |
| ssq (SST-SVD) | 0.94 / 0.90 / 1.00 | 0.96 / 0.87 / 1.00 |
| maxent | 0.88 / 0.80 / 1.00 | 0.92 / 0.77 / 1.00 |
| rms_cv | 0.56 / 0.28 / 0.80 | 0.50 / NaN / 0.37 |

---

## 3. Las gráficas, una por una

Se generan con `DOE_utils/DOE_plots/validation_figures.py` (en el visor, entradas "Validation — <nombre>"; en disco, `figs_validation/<nombre>.png`).

### 3.1 `ranking` — barras de balanced accuracy, MCC y AUC

- **Cómo leerla:** una terna de barras por indicador, ordenados del mejor al peor. La barra de error del AUC es el intervalo de Hanley-McNeil.
- **Qué concluir:** green y ssq empatan arriba; maxent un poco abajo; rms_cv claramente peor.
- **Cuidado:** barras de error de AUC "planas" en 1.0 no significan certeza (ver 2.3). Mirar `tpr_tnr` para ver si las diferencias son significativas.

### 3.2 `roc` — curvas ROC con punto operativo

- **Cómo leerla:** eje X = falsas alarmas (FPR), eje Y = detecciones (TPR). Cuanto más pegada a la esquina superior izquierda, mejor. El círculo hueco es el punto operativo real.
- **Qué concluir:** green, ssq y maxent tocan la esquina (AUC 1.00): *existe* un umbral perfecto. Pero el círculo de maxent está a la derecha de la esquina: su umbral actual da falsas alarmas que otro umbral evitaría. rms_cv tiene una curva escalonada y un punto operativo lejos (FPR = 0.875).
- **Cuidado:** las curvas de green y ssq se superponen y una tapa a la otra; no es un error.

### 3.3 `tpr_tnr` — TPR y TNR con intervalo de Wilson

- **Cómo leerla:** dos puntos por indicador (naranja = TPR, azul = TNR) con su barra de error. La barra larga significa pocos casos.
- **Qué concluir:** todos detectan los inestables (TPR = 1); la diferencia entre indicadores está en el TNR (falsas alarmas). Los intervalos de green, ssq y maxent se solapan: con estos datos **no se puede declarar un ganador entre ellos**; solo que rms_cv es peor.
- **Es la figura más honesta para los directores** porque muestra la incertidumbre.

### 3.4 `confusion` — matriz de confusión 2×2 por indicador

- **Cómo leerla:** filas = verdad (inestable arriba, estable abajo), columnas = lo que hizo el indicador (alarma / sin alarma). Verde = TP, rosa = FN, amarillo = FP, azul = TN. Cada celda tiene su conteo.
- **Qué concluir:** una lectura inmediata de dónde se pierde cada indicador. rms_cv: 7 FP, casi ninguna celda azul.

### 3.5 `case_matrix` — resultado por caso e indicador

- **Cómo leerla:** columnas = casos ordenados por kappa (S estable, U inestable, g gris); filas = indicadores; el color es el resultado (verde TP, azul TN, amarillo FP, rosa FN, gris = no puntuado).
- **Qué concluir:** los FP de green, ssq y maxent aparecen en **kappa 1.03–1.05**, justo antes de los casos grises y los inestables: son casos teóricamente inestables que aún no se ven en la amplitud. rms_cv falla en casi todos los estables, incluso con kappa 0.5.

### 3.6 `detection_time` — primera detección vs el inicio por amplitud

- **Cómo leerla:** eje X = kappa, eje Y (log) = tiempo. La línea negra es `t_onset` (cuándo la amplitud llega al 40 %). Los círculos son detecciones en casos inestables; las "x", detecciones en casos etiquetados estables.
- **Qué concluir:** todos los círculos están **por debajo de la línea negra**: los indicadores alarman antes de que el chatter sea visible. rms_cv está pegado al fondo (0.07 s) para todos los casos, estables incluidos: eso es la primera ventana, no detección.

### 3.7 `delay_vs_kappa` — retraso con signo vs kappa

- **Cómo leerla:** Y negativo = alarma antes del inicio por amplitud. La línea en 0 es el instante en que la amplitud llega al límite.
- **Qué concluir:** todos los retrasos son negativos y se acercan a 0 al aumentar kappa (el chatter crece más rápido y deja menos margen). maxent anticipa más que green y ssq; rms_cv "anticipa" más aún, pero porque alarma en el arranque.

### 3.8 `detection_amp` — amplitud en el momento de la primera alarma

- **Cómo leerla:** Y (log) = |Axial_disp| como % de la base en la primera alarma, en los casos inestables. Líneas: `lim_sup` (40 %, límite de la etiqueta) y `lim_inf` (10 %).
- **Qué concluir:** los indicadores alarman con **2–5 %** de la base, muy por debajo del 10 % y del 40 %. Es la evidencia directa de que detectan crecimiento incipiente. Ojo con rms_cv: esa curva refleja el arranque, no crecimiento.

### 3.9 `score_vs_kappa` — el score del caso (máx. de `I_t`) vs kappa

- **Cómo leerla:** un panel por indicador; azul = casos estables, naranja = inestables; la línea punteada es kappa = 1.
- **Qué concluir:** lo que umbraliza el ROC. En green, ssq y maxent hay un salto de órdenes de magnitud entre estables e inestables (por eso AUC = 1). En rms_cv las nubes se mezclan.

### 3.10 `score_dist` — distribución del score, estables vs inestables

- **Cómo leerla:** dos columnas por indicador (estable / inestable), eje Y en escala log. Cuanto más separadas, mejor.
- **Qué concluir:** green, ssq y maxent separan completamente las dos clases; los puntos "estables" más altos son los casos de kappa 1.03–1.05. En rms_cv hay solape.
- **Cuidado:** si el indicador usa valores negativos (maxent, razón de verosimilitud) la escala es symlog.

### 3.11 `alarm_quality` — alarmas en estables y persistencia

- **Cómo leerla:** izquierda = fracción de ventanas con alarma en casos estables (menor es mejor); derecha = persistencia de la alarma tras la primera detección (mayor es mejor).
- **Qué concluir:** green y ssq casi no alarman donde no deben (0.03); rms_cv, 0.29. Los tres primeros mantienen la alarma (persistencia 1.0).

### 3.12 `training_coverage` — entrenamiento vs validación

- **Cómo leerla:** ejes kappa y rpm; puntos = casos del dataset de entrenamiento (azul estable, naranja inestable), "x" = casos de validación.
- **Qué concluir:** si la validación cae **dentro** de la región entrenada, es una validación "dentro de dominio". Si no, hay extrapolación y hay que decirlo. Necesita que la validación se haya corrido con `--reference`.

### 3.13 `compare` — comparación de dos experimentos (A vs B)

- **Cómo leerla:** un tramo por indicador: círculo hueco = A, círculo lleno = B, en balanced accuracy, MCC y AUC. Cuanto más largo el tramo, mayor el cambio.
- **Qué concluir:** cómo cambia cada indicador al cambiar de condiciones (p. ej. 12000 vs 5189 rpm). Aquí green y ssq son estables entre ambas, mientras que rms_cv cae.

---

## 4. Cómo contárselo a los directores

### Mensajes principales
1. **La validación compara cada indicador con una etiqueta operacional de amplitud, caso por caso**: acierta si detecta un caso inestable y no alarma en uno estable.
2. **Green, ssq y maxent detectan todos los casos inestables** (TPR = 1) y separan perfectamente las dos clases (AUC = 1). **rms_cv no**: alarma casi siempre desde el arranque.
3. **Los indicadores detectan antes de que el chatter sea visible**: con 2–5 % de la base, entre ~0.4 y ~7 s antes del límite del 40 % según kappa y el indicador. Es una ventaja, no un error.
4. **Los pocos falsos positivos** de green, ssq y maxent ocurren con kappa 1.03–1.05, casos teóricamente inestables que la etiqueta operacional todavía llama estables. Se leen como "detecta antes de que sea visible".
5. **Diferencias finas entre green y ssq no son concluyentes** con 11–19 casos: los intervalos se solapan.

### Limitaciones que conviene decir antes de que pregunten
- **Pocos casos** (11–19): intervalos anchos, AUC = 1.00 sin intervalo informativo.
- **La etiqueta es operacional, no teórica**: depende del horizonte de la simulación y del 40 % elegido.
- **Etiqueta de todo el registro**: no hay "retraso desde el inicio del chatter" medible; solo la anticipación respecto al límite de amplitud.
- **rms_cv**: el problema es de configuración del indicador (warmup), no necesariamente del método.
- **Rampas de Ap**: no incluidas; se evaluarán aparte.

### Preguntas probables y respuesta corta
| Pregunta | Respuesta |
|---|---|
| ¿Por qué un indicador que alarma "antes" no se cuenta como error? | Porque en el caso etiquetado inestable el chatter existe; que se detecte antes de ser visible es sensibilidad. El tiempo se reporta aparte. |
| ¿Por qué balanced accuracy y no accuracy? | Hay más casos inestables que estables (o al revés): balanced accuracy pesa las dos clases por igual. |
| ¿AUC = 1 significa que no falla nunca? | No: significa que existe un umbral que separa perfectamente en estos datos. Con pocos casos el intervalo no informa. |
| ¿Por qué MCC y F1? | MCC usa los cuatro conteos y no se infla con clases desbalanceadas; F1 se centra en la clase inestable. |
| ¿Qué hace rms_cv diferente? | Alarma en la primera ventana (t ≈ 0.07 s) en casi todos los casos; eso hunde su TNR. |

---

## 5. Dónde está cada cosa

| Qué | Dónde |
|---|---|
| Código de métricas | `DOE_utils/DOE_analisis/validate_indicators.py` |
| Código de figuras | `DOE_utils/DOE_plots/validation_figures.py` |
| Informe técnico (decisiones, estado) | `DOE_utils/INFORME_validacion.md` |
| Resultados | `doe_validation_results.h5` y `doe_validation_results_metrics.csv` (junto a los datos del experimento) |
| Figuras guardadas | `<carpeta del .h5>/figs_validation/` |
