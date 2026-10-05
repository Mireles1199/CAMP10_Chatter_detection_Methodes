# Informe nocturno: rampas de Ap (`PLAN_ramps.md`) — 2026-10-05

El plan está completo, de F0 a F8, y la prueba real de punta a punta funcionó desde la app. Hay 13 commits en
`Aplication-Indicateur-Validacion-Training` (`3d0c314`..`361e547`), sin push. Todos los selftests y
`check_app_dialogs.py` (con su nueva sección de rampas) pasan.

## Prueba real con el experimento `ramp_check`

Simulé dos casos: 15→5 mm (decreciente) y 5→15 mm (de control). Tardaron 66 min cada uno. Después corrí Extract,
Label template, Label build, Indicators y Validate contra tu entrenamiento constante, y todo quedó en verde.

- **Física:** en la rampa decreciente la fuerza sigue a Ap(t). F/Ap se mantiene en 50.0 N/mm mientras Ap baja. El
  control 5→15 sale **idéntico bit a bit** a tu cono, así que Ap(t) lineal en el tiempo queda confirmado.
- **Verdad por ventanas** (7 vueltas, paso 1):
  - 5→15: pasa a inestable a los **10.46 s**.
  - 15→5: es inestable de **1.33 a 11.04 s** y vuelve a estable desde 12.15 s.

## Lo que tienes que decidir tú

**1. El valor por defecto de `early_tol_s` (0.5 s) es demasiado corto para estos datos.** Lo dejé en 0.5 s porque era
la decisión cerrada del plan, pero el efecto es grande:
- Green, SST y MaxEnt detectan el chatter **mientras crece**, antes de que la amplitud llegue al 40 % del avance. En
  los casos inestables detectan siempre antes de que la amplitud cruce el límite: Green y SST hacia el 60 % de ese
  tiempo, MaxEnt hacia el 40 %. Con 0.5 s todas esas detecciones cuentan como "alarma temprana" y por tanto como FN.
- Lo recalculé en `DOE_Test_1DOF_150_n12000`, escribiendo a un archivo temporal sin tocar el tuyo:

| Variante | bal.acc antes | Con 0.5 s | Con 5 s |
|---|---|---|---|
| Green | 0.94 | 0.48 | 0.94 |
| SST | 0.94 | 0.48 | 0.94 |
| MaxEnt | 0.88 | 0.38 | 0.69 |
| RMS-CV | 0.56 | 0.06 | 0.34 |

- RMS-CV sí alarma en el transitorio de entrada (0.075 s); esa alarma es falsa de verdad.
- En `ramp_check`, con 0.5 s todas las variantes quedan en "alarma temprana". Con 2 s, Green, SST y MaxEnt aciertan
  las dos rampas como anticipadas.
- Se cambia por experimento en Validate > Edit config. Las tablas están en la bitácora de `PLAN_ramps.md`.

**2. Etapas tuyas en naranja, a propósito; no las re-ejecuté:**
- La Label template y la Label build del cono, porque la verdad de una rampa ahora se calcula por ventanas.
  Precalculado en temporal, el cono pasa a inestable a los 10.46 s.
- El Validate de `DOE_Test_1DOF_150_n12000`, porque cambió la regla de puntuación.

**3. Rampas decrecientes:** tu plantilla de caso `1DOF_150Hz` solo sabe hacer crecer Ap; con ella una rampa
decreciente se simula como un caso constante. No toqué tus carpetas de caso. Creé
`Data/1DOF_150_Ramp_check/1DOF_150Hz` con una pieza que admite los dos sentidos (marca `# ramps: both directions` en
su `db_def`). La app rechaza una rampa decreciente si el caso no lleva esa marca. Si quieres rampas decrecientes en
otras carpetas, hay que copiar ese cambio de la `db_def`.

## Fallos encontrados y arreglados
- **Por qué falló el Validate de tu cono (22:51):** `validate_indicators` tomaba los parámetros del primer grupo de
  etiquetas, que estaba vacío (`gray`). Arreglado.
- **Visor:** usaba `t_onset` como variable de color y orden en los archivos de rampas reales. Ahora usa κ (el κ de
  inicio en las rampas).
- **Panel de Validate:** mostraba un ranking vacío cuando solo había rampas.
- **Aviso de casos gray:** contaba dos veces una rampa que pasa por gray.

## Decisiones que tomé por ti (y qué cuestan si me equivoqué)
- **Ventana de etiquetado por defecto:** sale de las variantes del experimento, y solo se aplica si hay rampas. Por
  eso tus experimentos constantes no cambiaron. Para RMS-CV y SST cuento su ventana de decisión completa (la "dec7"
  del nombre). Si no era eso, la ventana sería de 4 vueltas en vez de 7.
- **Una rampa que pasa por gray e inestable, sin tramo estable,** cuenta como inestable. Si no era eso, cambia la
  etiqueta de esos casos.
- **Rampa decreciente detectada solo en su tramo estable final:** cuenta como TP, que es lo que dice la regla del
  plan. Añadí la columna de diagnóstico `hit_in_unstable` para verlo. Si deberían ser FN, hay que cambiar la regla.
- **`doe_indicator_plotter.py` (figuras de consola):** no lo modifiqué; las rampas se ven en el visor unificado.
- **Rampa de control 5→15:** la elegí igual que el cono para que sirviera también de prueba de la nueva plantilla.

## Cosas menores pendientes
- En los archivos de validación el visor sigue mostrando `Axial_acc` como si fuera un "run". Ya pasaba antes.
- `s/gen_tool.py` de la plantilla falla en todas las simulaciones, también en tu `DOE_Test_1DOF150_n5189`. Nessy2m
  usa la herramienta ya generada en `tool/`, así que no afecta a los resultados.

## Revisión final y seguimiento
- La revisión final del branch la hice yo mismo: no lancé un revisor aparte porque no lo pediste. Es más débil que
  una revisión independiente; si la quieres, corre `/code-review` antes de hacer merge.
- La bitácora y el punto donde retomar están en `DOE_utils/PLAN_ramps.md`; también están en el tutorial (sección
  "Rampas de Ap") y en el HELP de la app.
- `experiments/ramp_check.yaml` queda sin commitear, como tus otros experimentos.
