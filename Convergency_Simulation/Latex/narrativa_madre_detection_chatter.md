# Historia narrativa: Détection du début du chatter (CAMP10) — narrativa madre

## Problemática

Detectar en señales temporales de torneado/fresado el **inicio** del chatter (no solo si existe
o no), típicamente con un comportamiento exponencial, sin que la detección dependa de la
discretización temporal de la señal. Los métodos clásicos (pico a pico, máximos) ven su
detección sesgada por dónde caen los picos según esa discretización — el problema que este
trabajo busca evitar.

## Objetivo

Proponer un indicador basado en ciclos capaz de detectar el inicio del chatter, y posicionarlo
frente a otros indicadores de detección de inicio de chatter existentes en la literatura,
mediante una comparación sobre una base común unificada (esa base de comparación se define en
otra sección de la tesis, no en esta, pero es un prerequisito de todo lo que sigue).

## Justificación

Al afinar el modelo (dexel más pequeño) se observó que la aparición del chatter se retrasaba
cada vez más, sin evidencia de que ese comportamiento convergiera. Esto puso en duda la
confiabilidad del modelo cónico (ap variable) como referencia para comparar indicadores:
cualquier conclusión sobre qué indicador detecta mejor o antes podía ser en realidad un
artefacto de la discretización del modelo, no un mérito real del indicador. Por esta razón se
retrocede y se replantea el trabajo en etapas que aíslan un parámetro a la vez, empezando por el
modelo más simple (ap constante), con el fin de validar primero la confiabilidad del modelo
antes de usarlo como base de comparación entre indicadores.

## Método / Solución

Recorrido general del proyecto:

1. Caso simple de la literatura (torneado, 1 grado de libertad, modelo "tubo" = ap constante):
   verificar qué indicadores distinguen estable/chatter según el ap.
2. Pregunta siguiente: ¿qué indicador avisa *antes* del inicio de la inestabilidad? Se pasa a un
   modelo de ap variable ("cono"), cruzando el lóbulo de estabilidad desde la zona estable hasta
   la zona de chatter.
3. Hallazgo: todos los indicadores detectan después del tiempo teórico de cruce del lóbulo —
   explicado porque el modelo cónico retrasa la aparición del chatter respecto al modelo de ap
   constante usado para calcular los lóbulos de estabilidad clásicos.
4. Se evalúa la robustez a la discretización de pieza (tamaño de dexel): usando los mismos
   parámetros ya "convergidos" en el modelo tubo, variar el tamaño de dexel en el modelo cónico
   también desplaza el inicio de chatter (dexel más fino → retrasa; dexel más grueso →
   adelanta) — un hallazgo inesperado.
5. Hipótesis: el retraso proviene de las condiciones/perturbaciones iniciales — un dexel más
   pequeño genera una perturbación inicial más pequeña, lo que retrasa el inicio de chatter
   (validar esta hipótesis por una segunda vía es tarea de la Etapa4).
6. Como el modelo depende fuertemente de la discretización, se retrocede a aislar e identificar
   los parámetros que influyen, replanteando el trabajo completo en 8 etapas (Etapa0 a Etapa7).

## Resultado clave (trabajo en curso)

- Trabajar con ap variable dificulta el análisis frente a los lóbulos de estabilidad clásicos,
  ya que estos se calculan asumiendo ap constante.
- La influencia del tamaño de dexel en el modelo de ap variable no es despreciable (evidencia
  recabada en la Etapa4).
- Pendiente: definir un umbral mínimo de aplicabilidad de los indicadores según el orden de
  magnitud de la señal/ruido (Etapas 5 y 6).
- Pendiente: la comparación definitiva entre indicadores, una vez validada la base del modelo.
- El proyecto sigue en curso — las conclusiones detalladas por etapa se documentarán punto por
  punto más adelante, con este mismo skill, en cada carpeta de etapa.

## Hilo narrativo

Toda la comparación entre indicadores se apoyaba en un mismo criterio: qué indicador detectaba
el chatter antes, usando como referencia el tiempo teórico en que se cruza el lóbulo de
estabilidad. Pero al descubrir que el tamaño de dexel del modelo influye en cuándo aparece el
chatter, esa referencia teórica dejó de ser confiable: ya no había manera de saber si una serie
temporal del modelo reflejaba un chatter genuinamente tardío o uno simplemente retrasado por la
discretización. Esto hacía que la comparación, tal como estaba planteada, perdiera toda su
fuerza — no se podía afirmar con certeza qué indicador reaccionaba antes si el propio punto de
referencia era movedizo. Ese fue el momento de inflexión que obligó a retroceder: antes de poder
comparar indicadores de forma confiable, había que primero validar y entender el modelo mismo.
De ahí nacen las 8 etapas.

## Etapas del proyecto

1. **Etapa0** — Validar qué tamaño de dexel mantiene el error en las fuerzas de corte por debajo
   del 10 % (modelo estático, ap constante, dinámica desactivada).
2. **Etapa1** — Con ese dexel y la dinámica ya activada (modelo tubo), verificar que el ap
   límite de estabilidad predicho por el modelo coincide con el teórico, a un rpm dado.
3. **Etapa2** — Repetir la comparación anterior variando el tamaño de dexel, para medir su
   influencia en la predicción del ap límite.
4. **Etapa3** — Repetir variando la discretización temporal — conclusión: influye menos que el
   dexel espacial en la predicción del ap límite.
5. **Etapa4** — Pasar al modelo de ap variable (cono) y validar, por una segunda vía, la
   hipótesis de que el retraso del chatter proviene de las perturbaciones/condiciones iniciales
   asociadas al tamaño de dexel.
6. **Etapa5** — Robustez de los indicadores frente a ruido blanco añadido directamente a la
   señal; determinar a partir de qué orden de magnitud son aplicables los indicadores. Nace de
   observar que, en las señales temporales simuladas, el orden de magnitud es en muchos casos
   tan pequeño que analizarlo directamente no tiene sentido físico — de ahí la necesidad de
   establecer un umbral mínimo antes de poder explotar los indicadores con confianza.
7. **Etapa6** — Robustez frente a "ruido de modelo" (degradar el tamaño de dexel); problema
   abierto: aún no está claro cómo cuantificar este tipo de ruido, a diferencia del ruido
   blanco de la Etapa5. Misma motivación de fondo que la Etapa5 (umbral de magnitud explotable),
   pero explorada degradando el modelo en vez de inyectando ruido directo a la señal.
8. **Etapa7** — Configuración y zonas de parámetros de practicidad para el uso de los
   indicadores (aún poco desarrollado).

## Figuras sugeridas

Estas son sugerencias conceptuales (no se partió de código específico para la narrativa madre;
cada figura real se afinará al construir la narrativa de su etapa correspondiente con código):

1. **Diagrama/roadmap de las 8 etapas**, mostrando qué variable aísla cada una y el orden en que
   se conectan (dexel espacial → dexel temporal → modelo cónico → ruido de señal → ruido de
   modelo → practicidad) -- respalda: **Método** e **Hilo narrativo**, útil como figura de
   apertura tanto del capítulo como de una presentación.
2. **Esquema modelo tubo vs. modelo cono** (ap constante vs. ap variable) con el lóbulo de
   estabilidad superpuesto, mostrando visualmente por qué el cruce teórico del lóbulo ya no es
   una referencia directa en ap variable -- respalda: **Justificación** y **Resultado clave**.
3. **Estudio de convergencia**: tiempo (o ap) de inicio de chatter vs. tamaño de dexel en el
   modelo cónico, mostrando si la curva converge o sigue derivando al afinar el dexel -- respalda
   directamente el punto crítico de la **Justificación** (la duda sobre si el modelo converge).
4. **Comparación esquemática de detección por indicador vs. tiempo teórico de cruce del lóbulo**,
   ilustrando cómo el desplazamiento por dexel invalida usar ese tiempo teórico como referencia
   fija de comparación -- respalda el **Hilo narrativo**.
