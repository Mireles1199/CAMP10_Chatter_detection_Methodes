# ssq-chatter (reorganizado por módulos)

- No existe `chatter_hht_pyemd.py`. Todas sus funciones se distribuyeron en:
  - `lib/pipeline.py` — `detect_chatter_from_force(...)`
  - `lib/emd_preproc.py` — preprocesado (`_maybe_*`)
  - `lib/hht.py` — Hilbert/HHS
  - `lib/selection.py` — selección de IMF
  - `lib/metrics.py` — conteos/energías
  - `lib/misc.py` — utilidades restantes
  - `viz/plotting.py` — funciones `plot_*`
  - `lib/datatypes.py` — `ChatterResult`
- `utils/signal_chatter.py` contiene generación de señales/FFT.

## Instalación
```bash
pip install -e .
```

## Ejemplo rápido
```python
from ssq_chatter.lib.pipeline import detect_chatter_from_force
from ssq_chatter.utils.signal_chatter import make_chatter_like_signal

sig, meta = make_chatter_like_signal(fs=2000.0, T=6.0, signal_chatter=True)
res = detect_chatter_from_force(sig, fs=meta["fs"], hhs_enable=True)
```
