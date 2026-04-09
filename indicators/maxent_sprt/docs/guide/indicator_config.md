[← Signal Data](signal_data.md){ .md-button } [Run & Plot →](run_and_plot.md){ .md-button }
# `INDICATOR_CONFIG` — Detection Parameters

`INDICATOR_CONFIG` is a plain Python dictionary that controls every aspect of the MaxEnt-SPRT pipeline.  
Pass it as the second argument to `run_maxent_sprt(signal, INDICATOR_CONFIG)`.

---

## Structure

```python
INDICATOR_CONFIG = {
    "id"  : "MaxEnt_SPRT",      # (optional) label for logging
    "func": "Default",           # pipeline selector
    "params": {                  # keyword arguments forwarded to the pipeline
        ...
    },
}
```

---

## Top-Level Keys

| Key | Type | Required | Description |
|---|---|---|---|
| `"func"` | `str` or `callable` | Yes | `"Default"` uses the built-in `_maxent_sprt_pipeline`. Pass a callable to use a custom pipeline. |
| `"params"` | `dict` | Yes | All pipeline parameters (see below). |
| `"id"` | `str` | No | Human-readable label stored in logs and results. |

---

## `"params"` Dictionary — Complete Reference

### Mechanical / Signal Parameters

| Key | Type | Units | Description |
|---|---|---|---|
| `rpm` | `float` | RPM | Spindle rotational speed. Used to compute $f_r = \text{rpm}/60$ and the OPR downsampling step. |
| `ratio_sampling` | `float` | — | Sub-sampling multiplier applied after OPR. Effective rate = `ratio_sampling Ã— fr`. Typical value: `50.0`. Set to `1.0` for pure OPR (1 sample/revolution). |

### Segmentation

| Key | Type | Units | Description |
|---|---|---|---|
| `N_seg` | `int` | revolutions | Number of OPR samples per segment. One entropy value $H_n$ is computed per segment. Each segment spans $N_\text{seg}/f_r$ seconds. Small values → finer resolution but noisier. Typical: 2–“5. |

### Training Split

| Key | Type | Units | Description |
|---|---|---|---|
| `t_stable_total` | `float` | seconds | Time at which chatter begins. All signal **before** this time is used to train $P_0$ (no chatter). All signal **after** is used to train $P_1$ (chatter). **Must match the actual chatter onset in your signal.** |

### SPRT Error Rates

| Key | Type | Range | Description |
|---|---|---|---|
| `alpha` | `float` | (0, 1) | Desired Type I error rate (false alarm probability). Smaller = fewer false alarms but slower detection. |
| `beta` | `float` | (0, 1) | Desired Type II error rate (missed detection probability). |
| `reset_on_H0` | `bool` | — | If `True`, the SPRT statistic $S_n$ is reset to 0 each time the test accepts $H_0$. Enables detection of **multiple** chatter events in one signal. Set `False` for a one-shot test. |

### Signal Window (Optional)

| Key | Type | Units | Default | Description |
|---|---|---|---|---|
| `cut_start_time` | `float` | seconds | `t_analysis[0]` | Start of the analysis window. Ignores signal before this time. Useful to skip transient startup. |
| `cut_end_time` | `float` | seconds | `t_analysis[-1]` | End of the analysis window. |

---

## How the Thresholds Are Computed

The SPRT thresholds are derived automatically from `alpha` and `beta`:

$$a = \ln\frac{\beta}{1-\alpha} \qquad b = \ln\frac{1-\beta}{\alpha}$$

With `alpha = beta = 0.05`:

$$a \approx -2.944 \qquad b \approx +2.944$$

You do not need to set them manually.

---

## Effect of Changing Parameters

### `N_seg`

| Value | Segment duration at 12 000 rpm | Effect |
|---|---|---|
| 1 | 5 ms | Fast response, noisy entropy |
| 2 | 10 ms | Recommended starting point |
| 5 | 25 ms | Smoother, slower to react |
| 10 | 50 ms | Reliable for gradual onsets |

### `ratio_sampling`

| Value | Effective sampling rate (at 12 000 rpm, $f_r=200$ Hz) |
|---|---|
| 1.0 | 200 Hz — pure OPR |
| 10.0 | 2 000 Hz |
| 50.0 | 10 000 Hz (recommended) |
| 100.0 | 20 000 Hz = original $f_s$ |

### `alpha` / `beta`

| Value | Effect on thresholds | False alarms | Missed detections |
|---|---|---|---|
| 0.01 | $b \approx 4.6$ — harder to cross | Very few | More |
| 0.05 | $b \approx 2.9$ — balanced | Few | Few |
| 0.10 | $b \approx 2.2$ — easier to cross | More | Very few |

---

## Minimal Config Example

```python
config = {
    "func": "Default",
    "params": {
        "rpm": 12_000.0,
        "ratio_sampling": 50.0,
        "N_seg": 2,
        "t_stable_total": 5.0,   # chatter starts at t = 5 s
        "alpha": 0.05,
        "beta": 0.05,
        "reset_on_H0": True,
        "cut_start_time": 0.0,
        "cut_end_time": 10.0,
    },
}
```

---

[← Signal Data](signal_data.md){ .md-button } [Run & Plot →](run_and_plot.md){ .md-button }

