# Informe de validación (wt-validacion, 2026-10-05)

Datos leídos (solo lectura): `Data/1DOF_150_Ap_Cont_test_ind/{Ap_Cons_test_ind/DOE_Test_1DOF_150_n12000, DOE_Test_1DOF150_n5189/DOE_Test_1DOF150_n5189}` y `Data/1DOF_150_Ramp_check/...`
(los .h5 NO están en la carpeta CAMP10 original sino en `Data/`). Re-corrí `validate_indicators.py --out validacion_figs/<x>/` con el código actual
(el .h5 guardado de n12000 es schema/2, regla vieja). Figuras: `validacion_figs/<x>/fig/*.png` (excluida vía .git/info/exclude), generadas con
`DOE_utils/DOE_plots/validation_figures.py --results X/doe_validation_results.h5 --out-dir D`.

## 0. DECISIÓN DEL USUARIO (2026-10-05): `early_tol` eliminada por completo
Implementado en `validate_indicators.py` (selftest OK) y `validation_figures.py`:
- Caso constante inestable: TP si el indicador alarma en cualquier momento, FN si nunca. Estable: FP/TN igual que antes. Sin tolerancia, ni informativa. `delay_onset_s`/`delay_det_s` siguen como dato del retraso (no deciden nada).
- Desaparecen outcome `TP_early`, attr raíz `early_tol_s`, métricas `n_anticipated`, `n_early_alarm`, `early_alarm_rate`, `ramp_anticipated_rate*`.
- Rampas que cruzan (etiqueta en el tiempo): alarma antes del inicio de la primera ventana inestable = FA (sin tolerancia), después = TP, ninguna = FN. **Criterio a confirmar por el usuario**: con 2 rampas (ramp_check) ambas salen FA.
- Resultados sin tolerancia: n12000 green/ssq bal.acc 0.94 (MCC 0.90), maxent 0.88, rms_cv 0.56; n5189 green/ssq 0.96, maxent 0.92, rms_cv 0.50.
- `validate()` ahora crea el directorio de `--out`.
- COMPAT (contrato): `--early-tol` se sigue aceptando pero se ignora (oculto en --help), y `EARLY_TOL_S` (=0.0, sin efecto), `OUTCOME_TEXT`, `detection_outcome(t, onset, early_tol=None)` siguen exportados porque `doe_indicators.py` los importa. Quedan referencias a limpiar por wt-interfaz / dueños:
  `experiment.py` (l.729-733 pasa `--early-tol`; l.1137, 1448-1463 texto/regla; l.1848-1850 METRIC_COLUMNS y EARLY_TOL_S; l.1899, 2228 sección validate), `launcher.py` (l.231, 1492-1508 campo early_tol_s, 536, 2695),
  `check_app_dialogs.py` l.588, `doe_indicators.py` (l.43, 338-358, 424, 928-964, 1009, 1083-1106: usa la tolerancia en el texto de progreso de rampas; `detection_outcome` ya no la aplica) y `PLAN_ramps.md`.

---

## 1. Qué hace la validación (resumen)
- Verdad por CASO (etiqueta de amplitud: |Axial_disp| > 40 % de f_tooth·1e-3). `t_onset_amp` = primer instante sobre ese límite.
- Primer detectado: `t_det` = primera ventana con alarma. Inestable: TP si `t_det >= t_onset`; TP_early si cae en `[t_onset-0.5 s, t_onset)`; FA si antes; FN si nunca.
  Estable: FP si alarma alguna vez, TN si no. Gris: ignorado. En la 2x2, TP_early cuenta TP y **FA cuenta FN**.
- AUC: score del caso = max(I_t) en toda la señal (min si "bajo = chatter"; orientación sacada de las propias alarmas). Hanley-McNeil 95 %.
- Rampas: aparte (`ramp_*`), t_onset = inicio de la primera ventana inestable.

## 2. ¿Convencen las definiciones? Puntos discutibles (con evidencia)
1. **Tolerancia fija 0.5 s vs. alarma temprana real (el más grave).** Con la regla actual: n12000 TP=1/FN=10 (bal.acc 0.48, MCC -0.05), n5189 TP=0/FN=5, **pero AUC=1.00** en green/ssq/maxent.
   El .h5 viejo de n12000 daba TP=11 (bal.acc 0.94). Los indicadores alarman con |Axial_disp| ≈ 2–5 % de la base (`detection_amp.png`), 1–7 s antes del límite de 40 %.
   El adelanto escala con 1/crecimiento (kappa 1.08 → -4.7 s; kappa 2.0 → -0.45 s; `delay_vs_kappa.png`), así que ninguna tolerancia fija sirve. Una "alarma temprana" así es detección de crecimiento real, no falsa alarma.
   El ranking por balanced accuracy y el AUC se contradicen: ranking.png.
2. **Etiqueta 'stable' con kappa>1.** Casos kappa 1.01–1.05 (teóricamente inestables, amplitud no alcanza el límite en el horizonte 15/35 s) son "stable"; green/ssq/maxent los alarman a t=10–26 s → FP.
   Son errores de la etiqueta (horizonte finito) tanto como del indicador (`detection_time.png`, cruces sobre 'x'). Los 'unlabelled' (kappa 1.057–1.066) sí se ignoran.
3. **rms_cv alarma en la 1ª ventana (t≈0.07–0.11 s) en casi todos los casos** (estables incluidos): transitorio/warmup, no discriminación. Hunde TNR y su AUC (0.80 / 0.37); `max(I_t)` también lo ve. Es configuración del indicador (`warmup_ignore_alerts: false`), no de la métrica.
4. **t_start = 0.05 s en todo caso inestable** (etiqueta "inestable" sobre todo el registro) → `delay_start_s` = tiempo de detección, no un retraso; conteos por ventana (TP/FP por ventana) no son interpretables. Solo `delay_onset_s`/`delay_det_s` sirven.
5. **AUC con 17–22 casos perfectamente separados por kappa**: AUC=1.00 con IC [1,1] (Hanley-McNeil degenera en 1). Poco informativo para distinguir green/ssq/maxent; separa por kappa, no por calidad de detección.
6. Detalle menor: `validate()` no crea la carpeta de `--out` (falla con FileNotFoundError); cambio compatible, no hecho.

Alternativas para decidir (NO decididas, usuario): (a) hit = "alarma en algún momento de un caso inestable", retraso solo informativo (reproduce el esquema viejo, = lo que mide el ROC);
(b) early_tol relativo (en revoluciones/e-foldings de crecimiento, o fracción de t_onset; el cociente t_det/t_onset ronda 0.55–0.7 en todos los kappa);
(c) t_onset por umbral bajo (p. ej. 2–5 % de la base) — más cercano a la sensibilidad real de los indicadores;
(d) kappa>1 & "stable" → gris o 'slow unstable'; (e) persistencia de la alarma para separar alarma real de transitorio.

## 3. Figuras que valen la pena en la interfaz, y qué lee cada una
| Figura | Archivo | Datos del .h5 | Valor |
|---|---|---|---|
| Ranking (bal.acc, MCC, AUC ± IC) | ranking.png | `/ranking/{run,rank,balanced_accuracy,MCC,AUC}`, `/metrics/<run>` attrs `AUC_lo/AUC_hi` | alto: portada; mostrar junto a AUC por la contradicción |
| ROC por indicador + punto operativo | roc.png | `/roc/<run>/{high,low}/{fpr,tpr,thr}` (según `roc_direction`), `/metrics` `TPR,TNR,AUC` | alto: el punto operativo muy por debajo de la curva ES el hallazgo 1 |
| Matriz run × caso (outcome por kappa) | case_matrix.png | `/summary/<run>/{outcome,kappa,truth,group,ap_mm,ap_end_mm}` | alto: ve dónde falla cada indicador |
| Detección vs t_onset (log) | detection_time.png | `/summary/<run>/{first_detection_t,t_onset_amp,kappa,truth}` | alto: muestra adelanto y FP con kappa>1 |
| Retraso firmado vs kappa + banda early_tol | delay_vs_kappa.png | `/summary/<run>/delay_det_s`, attrs raíz `early_tol_s` | alto: justifica (o refuta) la tolerancia |
| Amplitud (% base) en la 1ª alarma | detection_amp.png | `case_NNN/Axial_disp/{time,values}`, `case_NNN` attr `$f_tooth$`, attrs raíz `labeling_*`, `/summary first_detection_t` | medio: costoso (lee señales), pero es la mejor evidencia de sensibilidad |
| max(I_t) por caso vs kappa | score_vs_kappa.png | `/summary/<run>/{score_max,score_min,truth,kappa}`, `/metrics roc_direction` | medio: explica qué umbraliza el ROC |
| Tabla de métricas | (tabla) | `*_metrics.csv` o `/metrics/<run>` attrs (TPR/TNR/F1/MCC con Wilson, `n_early_alarm`, `early_alarm_rate`, `mean_alarm_fraction_stable`, `mean_persistence`) | alto |
| Rampas: tasas ramp_* | (tabla/barras) | `/metrics/<run>` attrs `ramp_*` | pendiente: ramp_check tiene solo 2 casos, sin datos globales ni kappa; sin evidencia aún |
| Entrenamiento vs validación (cobertura kappa/spin) | (no hecha) | `/training/{kappa,ap_mm,spin_rpm,label}` | bajo/opcional |

Notas de contrato: no se tocó nada del contrato ni launcher/experiment/yaml. Cualquier cambio de regla (§2) debería ser aditivo (nuevas columnas/attrs, p. ej. un `early_tol` relativo o un outcome alternativo), manteniendo `outcome`/`TP...` actuales.
