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
independiente con la misma metodología de Etapa1. `Etapa_2.py` agrega los resultados
(\(a_{p,\mathrm{crit,sim}}\), % error, banda de transición \([a_{p,-}, a_{p,+}]\)) de las
carpetas correspondientes y los grafica juntos vs. tamaño de dexel (eje log), con la línea
teórica de referencia y el error porcentual en un eje secundario.

**Pendiente técnico:** `Etapa_2.py` actualmente apunta a 5 carpetas antiguas y parciales
(`_5e-5`, `_10e-5`, `_15e-5`, `_20e-5`, `_25e-5`, de mediados de junio) en vez de las 9
carpetas correctas y más recientes con sufijo `_RUN_10` (confirmadas por Enrique como "las
buenas"). Hay que actualizar la lista `convergence_folders` en el script antes de generar
la figura real de convergencia.

## Resultado clave

**Pendiente -- Enrique aún no ha analizado los resultados de las 9 corridas.** Preguntas
abiertas a responder cuando se analicen:
- ¿\(a_{p,\mathrm{crit,sim}}\) converge al afinar el dexel, o sigue derivando (la duda
  original que motivó toda la narrativa madre)?
- ¿Se repite el patrón visto en Etapa1 (dexel fino → conservador/pesimista, dexel grueso →
  subestima la inestabilidad), o aparece algo distinto al ver el barrido completo de 9
  puntos?
- Conclusión de la etapa en una frase (pendiente).

## Hilo narrativo específico de la etapa

**Pendiente**, a completar junto con el Resultado clave una vez analizados los datos.

## Figuras

`Etapa_2.py` ya genera la figura de convergencia (`plot_dxl_convergence`), pero **no la
guarda automáticamente a disco** (solo `plt.show()`) -- habría que agregarle el guardado
antes de insertarla en un `.tex`.

1. **Convergencia \(a_{p,\mathrm{crit,sim}}\) vs. tamaño de dexel** (con banda de
   transición y línea teórica) -- respalda directamente el **Resultado clave** (pendiente)
   y el punto crítico de convergencia de la **Justificación** heredada de la madre.
