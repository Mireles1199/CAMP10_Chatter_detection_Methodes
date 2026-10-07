# Documentación del repo — índice

Un solo lugar para planes, guías, informes, contratos y flujos. Si agregas un documento, ponlo en su carpeta y añade una línea aquí con su estado.

Estados: **activo** (se sigue editando), **vigente** (terminado, sirve como referencia), **histórico** (ya implementado; solo contexto).

| Carpeta | Documento | Qué es | Estado |
|---|---|---|---|
| `planes/` | `PLAN_app_v2.md` | Plan v2 de la app de experimentos (bitácora y punto donde retomar, §0) | activo |
| `planes/` | `PLAN_noise_validation.md` | Validación con ruido blanco: decisiones, contrato de formatos (§4), fases N/I/J | vigente (hecho; §4 es el contrato) |
| `planes/` | `PLAN_ronda3.md` | Ronda 3: ruido con 3 realizaciones, panel central en el visor de validación con ruido, verdad en los visores, flujo "extender", estilo unificado | activo |
| `planes/` | `PLAN_eta_rename.md` | Renombrar κ (kappa) → η (eta): contrato de nombres, helper `eta_compat.py`, fases, reparto | activo |
| `planes/` | `PLAN_plot_style.md` | Estilo de figuras unificado (skill `article-plot-style`): inventario, escala del texto, fases; pendiente del OK del usuario | propuesta |
| `guias/` | `GUIA_metricas_validacion.md` | Métricas y gráficas de la validación, para no especialistas | vigente |
| `guias/` | `GUIA_validacion_ruido.md` | Validación con ruido paso a paso, para no especialistas | vigente |
| `guias/` | `indicadores_modos_explicacion.md` | Derivación de los modos de parametrización física de los indicadores | vigente |
| `informes/` | `INFORME_validacion.md` | Estado y decisiones de la validación (incluye ruido) | vigente |
| `informes/` | `INFORME_ramps_2026-10-05.md` | Informe nocturno de las rampas de Ap | histórico |
| `contratos/` | `CONTRATO_interfaz.md` | Cambios de wt-interfaz que afectan al merge | vigente |
| `flujos/` | `FLUJO_DOE_utils.md` / `.html` | Orden de scripts, entradas y salidas de DOE_utils | vigente |
| `historico/` | `PLAN_app_experimentos.md` | Plan v1 de la app (implementado) | histórico |
| `historico/` | `PLAN_ramps.md` | Rampas de Ap (implementado; estudio de rampas diferido) | histórico |
| `historico/` | `PLAN_figuras.md` | Exportación configurable de figuras (implementado) | histórico |

## Fuera de `docs/` a propósito

- `DOE_utils/TUTORIAL.md` (+ `DOE_utils/tutorial_img/`): la app lo lee en tiempo de ejecución (pestaña Tutorial); no moverlo.
- `indicators/<indicador>/README.md` y `indicators/<indicador>/docs/`: documentación de cada paquete; viven con su código.
- `indicators/COMMON_TEMPLATE.md`: plantilla común de los paquetes de indicadores.
- `Convergency_Simulation/Latex/`: narrativas de la tesis.

## Notas

- Las rutas de este índice son relativas a `docs/`. Muchos comentarios del código citan solo el nombre del archivo (por ejemplo `PLAN_ramps.md`): búscalo aquí.
- κ (kappa) pasó a llamarse η (eta) el 2026-10-06 (`planes/PLAN_eta_rename.md`): los documentos de `historico/` conservan "kappa"; equivale a η.
- Hasta el 2026-10-06 los documentos de planes, guías e informes estaban sueltos en `DOE_utils/` y en la raíz.
