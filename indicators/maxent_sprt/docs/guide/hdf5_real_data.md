[← Run & Plot](run_and_plot.md){ .md-button } [ API Reference Overview →](../api/index.md){.md-button }
# Loading Real Data with `HDF5Reader`

This guide shows how to replace the synthetic signal from the [Quick Start](quickstart.md) with a real measurement stored in an HDF5 file.

---

## What Is an HDF5 File?

HDF5 (Hierarchical Data Format 5) organises data like a filesystem:  
groups (folders) contain datasets (arrays). A typical measurement file looks like:

```
out.hdf5
├── tool_dyn/
│   └── data          ← [N×2] array: column 0 = time, column 1 = displacement
├── tool_dyn_o/
│   └── data          ← [N×2] array: column 0 = time, column 1 = velocity
└── res_R_p/
    └── data          ← [N×2] array: force in Newtons
```

---

## Import

```python
from MaxEnt_SPRT import HDF5Reader
```

---

## Opening a File

```python
reader = HDF5Reader(r"D:\data\experiment_01\out.hdf5")
```

The entire file is loaded into memory as a Python dictionary on construction.

---

## Reading Datasets

### Slash-separated path

```python
tool_dyn = reader.get_element("tool_dyn/data")   # returns np.ndarray shape (N, 2)
```

### Separate arguments (equivalent)

```python
tool_dyn = reader.get_element("tool_dyn", "data")
```

### Slicing

```python
first_1000 = reader.get_element("tool_dyn/data", "0:1000")   # rows 0..999
```

### Inspect the full structure

```python
full_dict = reader.get_data()
print(full_dict.keys())
```

---

## Complete Example — From HDF5 to Detection

```python
import os
import numpy as np
from MaxEnt_SPRT import HDF5Reader, SignalData, run_maxent_sprt, plots_maxent_sprt

# ── 1. Load HDF5 ──────────────────────────────────────────────────────
data_path = r"D:\data\experiment_01\out.hdf5"
reader    = HDF5Reader(data_path)

# Read tool velocity (column 0 = time, column 1 = velocity)
raw_vel  = reader.get_element("tool_dyn_o/data")
t        = raw_vel[:, 0]
velocity = raw_vel[:, 1]

# ── 2. Compute fs from time array ─────────────────────────────────────
fs = 1.0 / (t[1] - t[0])
print(f"Sampling frequency: {fs:.1f} Hz")

# ── 3. (Optional) Cut to region of interest ───────────────────────────
t_start, t_end = 0.05, 15.0
mask      = (t >= t_start) & (t <= t_end)
t_cut     = t[mask]
vel_cut   = velocity[mask]

# ── 4. Package signal ─────────────────────────────────────────────────
sig = SignalData(
    t_analysis      = t_cut,
    signal_analysis = vel_cut,
    fs              = fs,
    path            = data_path,
    meta            = {"rpm": 12_000, "AP": "5 mm"},
)

# ── 5. Configure ──────────────────────────────────────────────────────
config = {
    "func": "Default",
    "params": {
        "rpm"            : 12_000.0,
        "ratio_sampling" : 50.0,
        "N_seg"          : 2,
        "t_stable_total" : 5.365,    # ← adjust to when chatter starts in your signal
        "alpha"          : 0.05,
        "beta"           : 0.05,
        "reset_on_H0"    : True,
        "cut_start_time" : 1.006,    # skip initial transient
        "cut_end_time"   : 10.303,
    },
}

# ── 6. Run + plot ─────────────────────────────────────────────────────
result = run_maxent_sprt(sig, config)
plots_maxent_sprt(signal=sig, result=result, show_signal=True, vlines=[5.365])
```

---

## Key Differences from the Synthetic Example

| Step | Synthetic | Real HDF5 |
|---|---|---|
| Signal source | Generated with `numpy` | `HDF5Reader.get_element(...)` |
| `fs` | Fixed constant | Computed as `1/(t[1]-t[0])` |
| `t_stable_total` | `5.0` (designed) | Must be determined from data or annotations |
| `cut_start_time` | `0.0` | Skip initial transient (machine ramp-up) |
| `path` field | Any string | Full path to the `.hdf5` file |

---

## Tips

!!! tip "Finding `t_stable_total`"
    If you do not know exactly when chatter starts in your recording:
    
    1. Plot the raw signal over time.
    2. Look for where the amplitude distinctly increases or the frequency content changes.
    3. Use that time as `t_stable_total`.  
    A rough estimate is acceptable — the MaxEnt models are robust to some label noise.

!!! warning "OPR integer requirement"
    Verify that `fs / (rpm/60)` is an integer **before** running the pipeline:
    
    ```python
    fr = rpm / 60
    assert abs((fs / fr) - round(fs / fr)) < 1e-6, \
        f"fs={fs}, fr={fr}: ratio {fs/fr} is not an integer"
    ```

!!! tip "Multiple channels"
    If your HDF5 has velocity, displacement, and force, you can store all in `meta` and pick the best channel:
    
    ```python
    sig = SignalData(
        t_analysis      = t_cut,
        signal_analysis = vel_cut,       # velocity is typically most sensitive
        fs              = fs,
        path            = data_path,
        meta            = {
            "displacement": disp_cut,
            "force"       : force_cut,
        },
    )
    ```

---

[← Run & Plot](run_and_plot.md){ .md-button } [ API Reference Overview →](../api/index.md){.md-button }
