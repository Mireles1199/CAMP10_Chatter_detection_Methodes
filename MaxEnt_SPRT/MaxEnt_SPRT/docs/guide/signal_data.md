[← Signal Data](signal_data.md){ .md-button } [Indicator Configuration →](indicator_config.md){ .md-button }
# `SignalData` — Signal Container

`SignalData` is a Python **dataclass** that packages a vibration signal and its metadata before passing it to `run_maxent_sprt()` and `plots_maxent_sprt()`.

---

## Import

```python
from MaxEnt_SPRT import SignalData
```

---

## Fields

### Required fields

These four fields must always be provided:

| Field | Type | Description |
|---|---|---|
| `t_analysis` | `np.ndarray` | Time array of the signal (seconds). Shape: `(N,)`. |
| `signal_analysis` | `np.ndarray` | Signal values (velocity, displacement, acceleration, etc.). Shape: `(N,)`. |
| `fs` | `float` | Sampling frequency in Hz. Must satisfy `fs / (rpm/60) = integer`. |
| `path` | `str` | File path of the original data, or any descriptive string (used for logging and plots). |

### Optional fields

| Field | Type | Default | Description |
|---|---|---|---|
| `meta` | `dict` | `{}` | Freeform metadata dictionary. Stored in the result and displayed in some plots. Typical keys: `"rpm"`, `"AP"` (axial depth of cut), `"note"`. |

---

## Minimal Example

```python
import numpy as np
from MaxEnt_SPRT import SignalData

fs = 20_000.0
t  = np.linspace(0, 10, int(10 * fs))
y  = np.sin(2 * np.pi * 200 * t)

sig = SignalData(
    t_analysis      = t,
    signal_analysis = y,
    fs              = fs,
    path            = "my_experiment",
)
```

---

## Full Example

```python
sig = SignalData(
    t_analysis      = t,
    signal_analysis = velocity_signal,
    fs              = 20_000.0,
    path            = r"D:\data\experiment_01\out.hdf5",
    meta            = {
        "rpm"   : 12_000,
        "AP"    : "5 mm",       # Axial depth of cut
        "tool"  : "4-flute endmill, D=12 mm",
        "date"  : "2026-03-24",
    },
)
```

---

## Common Mistakes

!!! danger "`t_analysis` and `signal_analysis` must have the same length"
    ```python
    # Wrong — different shapes
    sig = SignalData(t_analysis=t[:-1], signal_analysis=y, fs=fs, path="x")

    # Correct
    sig = SignalData(t_analysis=t, signal_analysis=y, fs=fs, path="x")
    ```

!!! warning "`fs` must match the actual time array spacing"
    The pipeline derives the rotation frequency from `fs` and `rpm`.  
    If `fs` does not match `1 / (t[1] - t[0])`, the OPR downsampling step will fail.

    ```python
    fs = 1.0 / (t[1] - t[0])   # Always compute fs from the time array
    ```

---

## Where `SignalData` Is Used

| Function | How it uses `SignalData` |
|---|---|
| `run_maxent_sprt(sig, config)` | Reads `t_analysis`, `signal_analysis`, `fs` for all processing steps |
| `plots_maxent_sprt(signal=sig, ...)` | Reads `signal_analysis`, `t_analysis`, `meta["rpm"]` for panel titles |

---

[← Signal Data](signal_data.md){ .md-button } [Indicator Configuration →](indicator_config.md){ .md-button }
