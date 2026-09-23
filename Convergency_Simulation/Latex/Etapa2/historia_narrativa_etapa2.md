# Historia narrativa: Étape 2 (parte de Détection du début du chatter — CAMP10)

> Narrativa madre: `../narrativa_madre_detection_chatter.md`

## Objetivo específico de esta etapa

Medir cómo influye el tamaño de dexel en la predicción del \(a_{p,\mathrm{crit}}\) --
repitiendo el ejercicio de Etapa1 (bisección hasta \(\Delta\epsilon = 0{,}01\)) para varios
tamaños de dexel, con el fin de ver si el modelo converge a medida que se refina la
discretización -- esta es, literalmente, la etapa que responde a la duda de convergencia
que motivó el replanteamiento en la narrativa madre.

## Método / Solución

9 tamaños de dexel (potencias de 2 alrededor de la base retenida en Etapa0, de 1/16× a
16×: 1,25; 2,5; 5; 10; 20; 40; 80; 160; 320, todo ×10⁻⁵ m), cada uno corrido de forma
independiente con la misma metodología de Etapa1 (bisección hasta \(\Delta\eta = 0{,}01\)).
`Etapa_2.py` agrega los resultados de las 9 carpetas `_RUN_10` (\(\eta_{\mathrm{crit,sim}} =
a_{p,\mathrm{crit,sim}}/a_{p,\mathrm{crit,theo}}\), banda \([\lambda_-,\lambda_+]\)) y los
grafica vs. tamaño de dexel (eje log), con la línea teórica de referencia \(\eta=1\).

## Resultado clave

El modelo converge rápidamente con el tamaño de dexel: \(\eta_{\mathrm{crit,sim}}\) se
mantiene dentro de ~1–2% de la teoría desde 1,25×10⁻⁵ hasta 160×10⁻⁵ m, con la mejor
predicción en 20×10⁻⁵ m -- el dexel base ya elegido en Etapa0. Refinar más allá de ese punto
no se justifica frente al costo computacional. Solo el dexel más grueso del barrido
(320×10⁻⁵ m) rompe la convergencia, subestimando la inestabilidad en ~50% (\(\eta≈1{,}5\)).

Se repite el mismo patrón cualitativo visto en Etapa1 (dexel fino/medio → conservador,
predice inestabilidad antes de lo real; dexel grueso → subestima la inestabilidad), pero acá
la convergencia se alcanza rápido del lado fino, así que en la práctica el dexel no tiene
tanta influencia sobre la detección salvo en el extremo más grueso del barrido.

## Hilo narrativo específico de la etapa

Esta etapa fue, en el fondo, una prueba de confianza en la herramienta misma. Después de la
duda que motivó todo el replanteamiento en la narrativa madre -- ¿el modelo seguía derivando
sin converger? --, Etapa2 responde que no, al menos no por este lado: en el modelo más
simple (ap constante), el simulador Nessy2 predice bien la inestabilidad y es fiable para
reproducir la realidad. El tamaño de dexel converge rápido y su efecto resulta negligible
frente a la detección de la inestabilidad. Es la primera pieza que da tranquilidad -- el
problema real (la deriva sin converger) no está acá; habrá que seguir buscando en las etapas
siguientes (dexel temporal, modelo cónico) para encontrar dónde sí importa.

## Figuras

`Etapa_2.py` (función `plot_epsilon_convergence`) genera la figura de convergencia y la
guarda en `Latex/plots/etapa2_convergence_epsilon_vs_dxl.png` (ya no depende de `plt.show()`).

1. **Convergencia \(\eta_{\mathrm{crit,sim}}\) vs. tamaño de dexel** (eje Y partido en dos
   paneles para separar el outlier de 320×10⁻⁵ m del resto; banda \([\lambda_-,\lambda_+]\);
   línea teórica \(\eta=1\)) -- respalda directamente el **Resultado clave** y el punto
   crítico de convergencia de la **Justificación** heredada de la madre.
