[← Installation](installation.md){ .md-button } [Signal Data →](signal_data.md){ .md-button }
# Quick Start — Step-by-Step Example

This guide walks through a **complete detection run using a synthetic signal**.  
No external files are needed — everything runs with `numpy` and the installed package.

The synthetic signal simulates:

- **0 –“ 5 s** — Stable, chatter-free vibration (pure sine + small noise).
- **5 –“ 10 s** — Chatter-like vibration (frequency-modulated / chirp + larger amplitude).

---

## Step 0 — Imports

```python
import numpy as np
import matplotlib.pyplot as plt
from MaxEnt_SPRT import SignalData, run_maxent_sprt, plots_maxent_sprt
```

All three names are part of the public API exported from `MaxEnt_SPRT.__init__`.

---

## Step 1 — Generate the Synthetic Signal

```python
# ===== Sampling parameters =========
fs   = 20_000.0   # Sampling frequency [Hz] — must satisfy fs/fr = integer
rpm  = 12_000.0   # Spindle speed [RPM]
fr   = rpm / 60   # Rotation frequency [Hz] = 200 Hz

# fs / fr = 20000 / 200 = 100  - (integer → OPR sampling is valid)

t = np.arange(0, 10, 1 / fs)   # 10-second signal, 200 000 samples

rng = np.random.default_rng(42)

# ==== Stable segment (0–“5 s) ===========
mask_stable  = t < 5.0
t_stable     = t[mask_stable]
y_stable     = 0.5 * np.sin(2 * np.pi * 200 * t_stable)   \
             + rng.normal(0, 0.02, t_stable.size)

# ===== Chatter segment (5–“10 s) =====
mask_chatter = t >= 5.0
t_chatter    = t[mask_chatter]
# Frequency increases from 200 Hz to 600 Hz (chirp)
inst_freq    = 200 + 80 * (t_chatter - 5.0)
y_chatter    = 1.5 * np.sin(2 * np.pi * inst_freq * (t_chatter - 5.0)) \
             + rng.normal(0, 0.10, t_chatter.size)

signal = np.concatenate([y_stable, y_chatter])
```

!!! tip "Physical interpretation"
    - `y_stable`: small-amplitude vibration at the tooth-passing frequency — no chatter.
    - `y_chatter`: growing amplitude at a frequency not locked to the spindle — chatter regime.
    - The amplitude ratio (0.5 vs 1.5) and noise level (0.02 vs 0.10) both increase entropy.

---

## Step 2 — Create `SignalData`

`SignalData` is a lightweight dataclass that packages the signal for the indicator.

```python
sig = SignalData(
    t_analysis      = t,        # Full time array
    signal_analysis = signal,   # Full signal array
    fs              = fs,       # Sampling frequency [Hz]
    path            = "synthetic_example",   # Identifier (any string)
    meta            = {
        "rpm"  : rpm,
        "note" : "synthetic chirp chatter onset at t=5 s",
    },
)
```

**Required fields:**

| Field | Type | Description |
|---|---|---|
| `t_analysis` | `np.ndarray` | Time axis for the signal |
| `signal_analysis` | `np.ndarray` | Signal values (velocity, displacement, etc.) |
| `fs` | `float` | Sampling frequency in Hz |
| `path` | `str` | File path or descriptive identifier |

See [SignalData reference](signal_data.md) for all fields.

---

## Step 3 — Define `INDICATOR_CONFIG`

This dictionary controls every aspect of the detection. Do not worry — each key is explained fully in [INDICATOR_CONFIG reference](indicator_config.md).

```python
INDICATOR_CONFIG = {
    "id"  : "MaxEnt_SPRT",       # Optional label for logging
    "func": "Default",           # "Default" activates the built-in pipeline
    "params": {
        # == Mechanical / signal parameters ==========================
        "rpm"           : rpm,      # Spindle speed [RPM]
        "ratio_sampling": 50.0,     # OPR sub-sampling ratio (50 x fr = 10 kHz effective)

        # == Segmentation =============================================
        "N_seg"         : 2,        # 2 revolutions per segment = 2/200 = 10 ms per segment

        # == Training split ===========================================
        "t_stable_total": 5.0,      # Everything BEFORE t=5 s is labelled (no chatter)
                                    # Everything AFTER  t=5 s is labelled (chatter)

        # == SPRT error rates =========================================
        "alpha"         : 0.05,     # Max false-alarm rate (5 %)
        "beta"          : 0.05,     # Max missed-detection rate (5 %)
        "reset_on_H0"   : True,     # Reset S_n when Hâ‚€ is accepted

        # == Signal window =============================================
        "cut_start_time": 0.0,      # Start analysis from t=0
        "cut_end_time"  : 10.0,     # End analysis at t=10
    },
}
```

!!! warning "Training split must match the signal"
    `t_stable_total` **must** be set to the time at which chatter starts in your signal.  
    If you set it too late, chatter samples contaminate the $P_0$ model — the detector will not work correctly.

---

## Step 4 — Run the Indicator

```python
result = run_maxent_sprt(sig, INDICATOR_CONFIG)
```

`run_maxent_sprt` executes three phases internally:

1. **Split** — divides the signal at `t_stable_total` to create training data for $P_0$ and $P_1$.
2. **Train** — fits two Gaussian MaxEnt models from the OPR-sampled training segments.
3. **Detect** — runs SPRT on the full signal segment by segment; accumulates $S_n$; flags chatter where $S_n \geq b$.

The function returns an `IndicatorResult` object:

```python
print(result.name)     # "MaxEnt_SPRT"
print(result.t)        # array: midpoint time of each segment
print(result.I_t)      # array: S_n history (the SPRT statistic)
print(result.t_d)      # array: times where chatter was detected (S_n >= b)
print(result.meta)     # dict: trained model parameters, config, intermediate signals
```

---

## Step 5 — Visualise the Result

```python
plots_maxent_sprt(
    signal     = sig,
    result     = result,
    show_signal= True,
    vlines     = [5.0],    # Mark the true onset at t=5 s
)
```

This produces a **6-panel figure**:

| Panel | Content |
|---|---|
| 1 | Raw signal — stable (blue) vs chatter (orange) portions |
| 2 | OPR samples from the stable training window |
| 3 | OPR samples from the chatter training window |
| 4 | Gaussian PDF fitted to a single segment |
| 5 | $P_0(H)$ and $P_1(H)$ probability densities side by side |
| 6 | Entropy sequence $H_n$ over time |
| 7 | SPRT statistic $S_n$ with thresholds $a$ and $b$ |

See [Run & Plot reference](run_and_plot.md) for all parameters.

---

## Complete Runnable Script

Copy-paste this into a `.py` file and run it directly:

```python
import numpy as np
from MaxEnt_SPRT import SignalData, run_maxent_sprt, plots_maxent_sprt

# == 1. Generate synthetic signal ======================================
fs, rpm = 20_000.0, 12_000.0
fr = rpm / 60
t = np.arange(0, 10, 1 / fs)
rng = np.random.default_rng(42)

mask_s = t < 5.0
y_s  = 0.5 * np.sin(2 * np.pi * 200 * t[mask_s]) + rng.normal(0, 0.02, mask_s.sum())
t_ch = t[~mask_s]
y_ch = 1.5 * np.sin(2 * np.pi * (200 + 80 * (t_ch - 5)) * (t_ch - 5)) \
     + rng.normal(0, 0.10, (~mask_s).sum())
signal = np.concatenate([y_s, y_ch])

# == 2. Package signal =================================================
sig = SignalData(
    t_analysis=t, signal_analysis=signal,
    fs=fs, path="synthetic",
    meta={"rpm": rpm},
)

# == 3. Configure indicator ============================================
config = {
    "func": "Default",
    "params": {
        "rpm": rpm, "ratio_sampling": 50.0, "N_seg": 2,
        "t_stable_total": 5.0, "alpha": 0.05, "beta": 0.05,
        "reset_on_H0": True,
        "cut_start_time": 0.0, "cut_end_time": 10.0,
    },
}

# == 4. Run ============================================================
result = run_maxent_sprt(sig, config)

# == 5. Plot ===========================================================
plots_maxent_sprt(signal=sig, result=result, show_signal=True, vlines=[5.0])
```

Expected output: the SPRT statistic $S_n$ should cross the upper threshold $b \approx +2.94$ near $t = 5\,\text{s}$, confirming chatter detection.

---

[← Installation](installation.md){ .md-button } [Signal Data →](signal_data.md){ .md-button }

