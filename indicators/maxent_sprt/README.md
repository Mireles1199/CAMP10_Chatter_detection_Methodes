
# MaxEnt_SPRT

Paquete Python para detección de *early chatter* con indicador **MaxEnt** por segmentos OPR
y test secuencial **SPRT** (Wald). Arquitectura modular, comentarios en español.

## Instalación (editable)

```bash
pip install -e .
```

## Requisitos
- NumPy (rama estable 1.24–2.x; fijado `<3` en este paquete)
- Matplotlib (opcional para ejemplos)

## Ejemplo mínimo

```python
from MaxEnt_SPRT.lib.detector import MaxEntSPRTDetector, MaxEntSPRTConfig
```

## Licencia
MIT
