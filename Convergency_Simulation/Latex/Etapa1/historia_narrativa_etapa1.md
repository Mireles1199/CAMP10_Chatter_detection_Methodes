# Historia narrativa: Étape 1 (parte de Détection du début du chatter — CAMP10)

> Narrativa madre: `../narrativa_madre_detection_chatter.md`

## Objetivo específico de esta etapa

Verificar que el modelo tubo (\(\ap\) constante, ya con la dinámica activada y usando el
dexel retenido en Etapa0) predice correctamente el \(a_p\) límite de estabilidad,
comparándolo contra el \(a_p\) crítico teórico a un rpm (velocidad de giro) dado.

## Método / Solución

Barrido de \(\epsilon = a_p / a_{p,\mathrm{crit,teo}}\) iniciando en 0,5. Para cada caso
(dinámica activa), se calcula el RMS móvil de la velocidad estructural y se clasifica el
caso como estable/inestable según la tendencia (pendiente) de ese RMS -- método simple y
rápido.

El barrido no es un único paso fijo: se refina iterativamente buscando el primer par
consecutivo estable→inestable, continuando hasta que la diferencia entre ambos \(\epsilon\)
es de 0,01 (precisión sin nombre definido aún -- pendiente que Enrique decida si la llama
"1 %" o "2 %"). El punto medio de ese par final se toma como \(\lambda_{\mathrm{crit,sim}}\),
y \(a_{p,\mathrm{crit,sim}} = \lambda_{\mathrm{crit,sim}} \times a_{p,\mathrm{crit,teo}}\). El
error se mide como \(|\lambda_{\mathrm{crit,sim}} - 1{,}0| \times 100\).

## Resultado clave

El modelo tiende a ser **conservador/pesimista**: en general \(\lambda_{\mathrm{crit,sim}} <
1\), es decir, predice inestabilidad a un \(a_p\) menor que el teórico (sobreestima la
propensión a volverse inestable) -- no se considera un defecto grave, lo relevante es la
precisión del error de detección.

Con el dexel retenido en Etapa0, el modelo predice bien el \(\epsilon\) teórico, y afinar
aún más el dexel no cambia significativamente el resultado (ya convergido). En cambio, con
un dexel más grande (más grueso), el modelo **subestima** la inestabilidad (predice más
estabilidad de la que hay en teoría) -- comportamiento opuesto al caso fino.

Conclusión: el modelo tubo, con el dexel adecuado, predice bien el límite teórico, pero la
dirección del sesgo (conservador vs. optimista) depende del tamaño de dexel usado -- punto
que se profundiza específicamente en Etapa2.

## Hilo narrativo específico de la etapa

Etapa de validación tranquila y esperada -- el resultado coincidió con lo anticipado, sin
sorpresas ni contratiempos.

## Figuras (ya existen en el código, pendientes de insertar en un `.tex`)

`Etapa_1.py` ya genera estas figuras (guardadas junto a las demás en `Latex/plots/`):

1. `fig1_stability_map` -- mapa de estabilidad (\(a_p\)/\(\epsilon\) vs. caso, coloreado
   estable/inestable) con el \(a_p\) crítico teórico y simulado marcados -- respalda:
   **Resultado clave**.
2. `fig2_rms_vel_vs_epsilon` -- RMS de la velocidad estructural vs. \(\epsilon\), mostrando
   la transición estable→inestable -- respalda: **Método** (evidencia del criterio de
   clasificación).
3. `fig5_crit_comparison` -- comparación entre \(a_{p,\mathrm{crit,sim}}\) y
   \(a_{p,\mathrm{crit,teo}}\) -- respalda directamente el **Resultado clave** (el sesgo
   conservador/optimista según el dexel).
