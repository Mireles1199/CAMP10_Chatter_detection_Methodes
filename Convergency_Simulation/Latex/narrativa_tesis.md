# Historia narrativa completa: Chatter Detection (tesis) — de la idea propia a la validación del modelo

> Esta es la narrativa de nivel superior de la tesis (o al menos de la parte relevante para el
> CSI). Envuelve y se apoya en `narrativa_madre_detection_chatter.md` (la crisis del dexel y las
> 8 etapas), que aquí se entiende como la sub-sección "Validación del modelo".

## 1. Origen: de la idea a su limitación

Motivación de fondo: detectar la inestabilidad del maquinado **antes** de que esté totalmente
establecida, siguiendo la evolución -- típicamente exponencial -- de la amplitud de vibración de
la herramienta.

Primer enfoque intentado: a partir de simulaciones, buscar la cresta de cada ciclo de vibración
para estimar si esa amplitud crece o decrece con el tiempo.

**Primer problema de discretización (temporal, en la detección de picos):** al venir de
simulaciones, existe una discretización temporal del tiempo; el punto muestreado no siempre
corresponde exactamente a la cresta real del ciclo. La estimación de la tasa de
crecimiento/decrecimiento quedaba entonces contaminada por cómo estaba hecha esa discretización
temporal -- no era una medida limpia del fenómeno físico.

## 2. El pivote: un indicador propio

En vez de seguir con un estudio basado en máximos/mínimos, se propone un indicador basado en la
**energía disipada ciclo a ciclo** (el indicador Green Integral / Green-Area) -- permite decidir,
ciclo a ciclo, si la amplitud de vibración crece o no, sin depender de encontrar exactamente la
cresta de cada ciclo. **Este indicador es la aportación propia de la tesis.**

## 3. De la invención a la comparación justa

Con el indicador propio ya propuesto, el objetivo pasa a ser posicionarlo frente a otros
indicadores ya existentes en la literatura, cada uno de un área/fundamento muy distinto entre sí:

- MaxEnt-SPRT
- SST / SSQ-STFT
- RMS-CV

Para que la comparación fuera justa, se propuso además una **metodología de homogeneización**:
una forma común de aplicar y evaluar los cuatro indicadores (los tres de la literatura + el
propio), respetando el dominio/fundamento propio de cada uno. Esta metodología de comparación
**es también aportación propia de la tesis**, no solo la aplicación de un estándar ya existente.

## 4. Puesta en marcha de la comparación

Se decide estudiar un caso de \(a_p\) variable, cruzando el lóbulo de estabilidad, para ver cuál
de los cuatro indicadores avisa antes del inicio de la inestabilidad.

Para ejecutar esa comparación hace falta fijar la configuración de las simulaciones que
alimentarán a los indicadores -- lo que exige, antes que nada, validar que el modelo de
simulación sea numéricamente confiable.

## 5. Validación del modelo (sub-sección -- ver narrativa madre CAMP10)

Aquí es donde entra completa la narrativa ya construida en
`narrativa_madre_detection_chatter.md`: la crisis de confianza en el modelo, y el
replanteamiento en 8 etapas (Etapa0 a Etapa7) que aíslan un parámetro de discretización o de
modelo a la vez antes de volver a confiar en él para comparar indicadores.

## 6. El giro: la segunda discretización

Con los parámetros de simulación ya fijados y "confiables" (validados en las primeras etapas),
al pasar al caso real de \(a_p\) variable aparece el **segundo problema de discretización**
--esta vez espacial, del tamaño de dexel-- descrito en detalle en la narrativa madre: el tiempo
teórico de cruce del lóbulo deja de servir como referencia de comparación, porque el propio
tamaño de dexel desplaza el instante de aparición del chatter simulado.

## Hilo narrativo global

La discretización traiciona la historia **dos veces**, en dos momentos y por dos razones
distintas:

1. Al intentar medir picos de vibración directamente sobre la señal simulada (discretización
   **temporal** de muestreo) -- esto es lo que motivó abandonar el enfoque de máximos/mínimos e
   inventar el indicador de energía ciclo a ciclo.
2. Al intentar validar el modelo de simulación que alimentaría la comparación de indicadores
   (discretización **espacial**, el tamaño de dexel) -- esto es lo que obligó a retroceder y
   replantear el trabajo en 8 etapas.

No es una coincidencia forzada: es el mismo enemigo (la discretización numérica) apareciendo en
dos escalas distintas de la misma investigación, y ambas veces la respuesta fue la misma actitud
metodológica -- desconfiar de la medida directa y buscar una vía más robusta (un indicador nuevo
en el primer caso, una validación sistemática etapa por etapa en el segundo).

## Estructura completa (para el CSI)

1. Origen -- limitación del enfoque de picos (discretización temporal).
2. Indicador propio -- energía disipada ciclo a ciclo (Green Integral).
3. Comparación justa -- metodología de homogeneización de los 4 indicadores.
4. Puesta en marcha -- caso \(a_p\) variable, necesidad de simulaciones confiables.
5. Validación del modelo -- Etapa0 a Etapa3 (ver narrativa madre CAMP10).
6. El giro -- segunda discretización (espacial), Etapa4.
7. Robustez al ruido -- Etapa5 y Etapa6 (en curso).
8. Practicidad y comparación final -- Etapa7 (pendiente).

## Figuras sugeridas para la sección de origen (1-3)

Conceptuales, pendientes de confirmar si existen datos/figuras reales para ellas:

1. **Esquema del problema de detección de picos bajo discretización temporal** -- una señal
   continua con su cresta real marcada, y los puntos muestreados discretos que no coinciden
   exactamente con ella -- respalda el punto 1 (Origen).
2. **Ilustración conceptual del indicador de energía ciclo a ciclo** -- por ejemplo la
   trayectoria en espacio de fase de un ciclo de vibración con el área (energía disipada)
   sombreada -- respalda el punto 2 (El pivote).
3. **Esquema de la metodología de homogeneización** -- cómo los 4 indicadores, de dominios
   distintos, se llevan a una salida/criterio común comparable -- respalda el punto 3.

¿Existen ya figuras reales para estos tres puntos (quizás en las sesiones `Green-Area`, `MaxEnt`,
etc.), o hay que crearlas?
