# Historia narrativa: Étape 3 (parte de Détection du début du chatter — CAMP10)

> Narrativa madre: `../narrativa_madre_detection_chatter.md`

## Objetivo específico de esta etapa

Medir la influencia de la discretización temporal (\(N_{dt}\), pasos por revolución) en la
predicción del \(a_{p,\mathrm{crit}}\), para compararla con la influencia de la
discretización espacial (dexel) ya estudiada en Etapa2, y determinar cuál de las dos pesa
más en la predicción del límite de estabilidad.

## Método / Solución

Barrido de 5 valores de \(N_{dt}\) (potencias de 2 dividiendo la base): 200 (base), 100,
50, 25 y 12,5 pasos/revolución. Cada caso se corre con la misma metodología de bisección de
Etapa1 (buscar la transición estable→inestable hasta \(\Delta\epsilon = 0{,}01\)) para
obtener su propio \(a_{p,\mathrm{crit,sim}}\). `Etapa_3.py` agrega los resultados de las 5
carpetas y grafica \(a_{p,\mathrm{crit,sim}}\) vs. \(N_{dt}\) (eje log), con banda de
transición, línea teórica de referencia y el error porcentual en un eje secundario --
mismo formato de figura que Etapa2, pero con \(N_{dt}\) en el eje X en vez del dexel.

**Pendiente técnico:** igual que en Etapa2, `Etapa_3.py` apunta a carpetas sin el sufijo
`_RUN_10` (`DOE_Detection_Limite_Lobes_dt_50`, etc.), pero las reales sí lo llevan
(`_dt_200_RUN_10`, `_dt_100_RUN_10`, `_dt_50_RUN_10`, `_dt_25_RUN_10`,
`_dt_12.5_RUN_10`, creadas el 18/09/2026). Hay que actualizar `convergence_folders` en el
script antes de generar la figura real.

## Resultado clave

**Pendiente -- las simulaciones están corriendo actualmente (18/09/2026), Enrique aún no
tiene los resultados.** Hipótesis a verificar, no todavía confirmada: que la discretización
temporal influye menos en la predicción del \(a_{p,\mathrm{crit}}\) que la discretización
espacial (dexel) vista en Etapa2 -- queda pendiente confirmar esto con los datos reales en
vez de darlo por sentado.

## Hilo narrativo específico de la etapa

**Pendiente**, a completar junto con el Resultado clave una vez terminen las simulaciones.

## Figuras

`Etapa_3.py` genera la figura de convergencia (`plot_dt_convergence`), con el mismo
formato que la de Etapa2 pero en función de \(N_{dt}\) -- tampoco se guarda a disco
automáticamente (solo `plt.show()`).

1. **Convergencia \(a_{p,\mathrm{crit,sim}}\) vs. \(N_{dt}\)** (con banda de transición y
   línea teórica) -- respalda el **Resultado clave** (pendiente), y sirve para comparar
   directamente contra la figura equivalente de Etapa2 (dexel) para responder cuál
   discretización pesa más.
