# Historia narrativa: Étape 0 (parte de Détection du début du chatter — CAMP10)

> Narrativa madre: `../narrativa_madre_detection_chatter.md`

## Objetivo específico de esta etapa

Escoger el tamaño de dexel más adecuado, decidido en base al comportamiento del error
relativo de la fuerza de corte simulada frente a la fuerza nominal analítica — no es "usar
la fuerza" como fin en sí, sino usarla como criterio de decisión para elegir el dexel que
alimentará las etapas siguientes.

## Método / Solución

Se trabaja con el **modelo tubo** (\(\ap\) constante, 1 grado de libertad) y **dinámica
desactivada** (estudio puramente cinemático, sin retardo regenerativo ni vibración
estructural), de modo que cualquier oscilación de la fuerza simulada sea atribuible
únicamente a la discretización geométrica, no a un fenómeno físico.

DOE de doce tamaños de dexel \(\dxlsize \in [4\times10^{-5},\ 2{,}56\times10^{-3}]\) m. Para
cada caso, la fuerza simulada se compara contra la fuerza nominal analítica (modelo
mecanicista linealizado, \(\kf\) constante) mediante cuatro estimadores de error relativo:
bias, pico, rango (étendue) y desviación estándar.

*Nota técnica:* los valores numéricos concretos del caso (\(\ap = 15\) mm, \(\Omega = 12099\)
rpm, \(N_{\mathrm{rev}} = 200\) pasos/revolución, \(\fz = 0{,}05\) mm/diente, 1 diente) son
detalle de modelo y le corresponden a la sección de descripción del modelo del `.tex`
(donde ya están las fórmulas simbólicas) -- no están capturados aquí porque este brief se
queda a nivel narrativo/conceptual. Al revisar `Etapa0.tex` estos valores numéricos todavía
no aparecen explícitos ahí (solo las fórmulas simbólicas) -- posible pendiente a completar
en el `.tex`, fuera del alcance de esta skill.

## Resultado clave

Se retiene \(\dxlsize = 2\times10^{-4}\) m -- el tamaño más grueso (menor costo
computacional) para el cual los cuatro estimadores de error se mantienen simultáneamente
bajo el umbral de tolerancia de 10 %, siendo el estimador de rango (étendue) el más
restrictivo. Este tamaño de dexel es el que se usará como base en las etapas siguientes del
pipeline.

## Hilo narrativo específico de la etapa

A diferencia del momento de inflexión que motiva la narrativa madre, esta etapa no tuvo
nada tedioso ni sorprendente -- es un paso de validación rutinario y necesario: antes de
poder confiar en cualquier resultado aguas abajo, había que primero asegurarse de que la
fuerza simulada no estuviera contaminada por ruido puramente numérico.

## Figuras (ya realizadas, no requieren sugerencia)

Ya existen y están insertadas en `Etapa0.tex`:

1. `fig3_error_summary` -- barras de error relativo (los 4 estimadores) vs. \(\dxlsize\),
   con línea de umbral al 10 % y el dexel retenido resaltado -- respalda: **Resultado clave**.
2. `fig_time_series` -- series temporales de la fuerza simulada para los doce casos del DOE,
   coloreadas por \(\dxlsize\) -- respalda: **Método** (evidencia visual de por qué el dexel
   grueso genera más ruido de discretización).
3. `fig_computational_cost` -- tiempo de cálculo vs. \(\dxlsize\) -- respalda: **Resultado
   clave** (el compromiso finura/costo que justifica no ir más fino de lo necesario).
