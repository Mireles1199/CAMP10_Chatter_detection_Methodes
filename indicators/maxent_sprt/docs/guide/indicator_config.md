[← Signal Data](signal_data.md){ .md-button } [Run & Plot →](run_and_plot.md){ .md-button }
# `INDICATOR_CONFIG` — Detection Parameters

`INDICATOR_CONFIG` is a plain Python dictionary that controls every aspect of the MaxEnt-SPRT pipeline.  
Pass it as the second argument to `run_maxent_sprt(signal, INDICATOR_CONFIG)`.

---

This indicator follows the shared CAMP10 indicator contract — see
[`indicators/COMMON_TEMPLATE.md`](../../../COMMON_TEMPLATE.md) for the
convention shared with `rms_cv`, `ssq_chatter` and `green_integral`
(`SignalData`/`IndicatorResult`/`INDICATOR_CONFIG`/`param_mode`).

## Structure

```python
INDICATOR_CONFIG = {
    "id"        : "MaxEnt_SPRT",   # (optional) label for logging
    "func"      : "Default",        # pipeline selector
    "param_mode": "native",         # "native" (default) | "by_revolution" | "by_modal"
    "params"          : { ... },    # used when param_mode == "native"
    "params_physical" : { ... },    # used when param_mode != "native"
}
```

---

## Top-Level Keys

| Key | Type | Required | Description |
|---|---|---|---|
| `"func"` | `str` or `callable` | Yes | `"Default"` uses the built-in `_maxent_sprt_pipeline`. Pass a callable to use a custom pipeline. |
| `"param_mode"` | `str` | No (default `"native"`) | `"native"`, `"by_revolution"`, or `"by_modal"` — see below. |
| `"params"` | `dict` | Only in `native` mode | All native pipeline parameters (see below). |
| `"params_physical"` | `dict` | Only in `by_revolution`/`by_modal` mode | Physical "decision window" parameters, resolved to native params by `_resolve_physical_params_maxent`. |
| `"id"` | `str` | No | Human-readable label stored in logs and results. |

---

## `param_mode`: physical decision-window parametrization

Instead of specifying `N_seg`/`rpm` directly (`native` mode), you can describe the
analysis window in physical units and let the resolver derive the native
parameters:

| `param_mode` | required in `params_physical` | optional |
|---|---|---|
| `by_revolution` | `T_rev` [s], `N_rev_window`, `step_rev` | `segmentation` |
| `by_modal` | `T_modal` [s], `N_modal_window`, `step_modal` | `T_rev` (informational only, not used for windowing), `segmentation` |

- `T_rev`/`T_modal` set the spindle-revolution / modal-period duration.
- `N_rev_window`/`N_modal_window` set the segment length in cycles.
- `step_rev`/`step_modal` (mandatory) set the hop between consecutive segments — `step == N_window` means no overlap.
- `segmentation` (default `"opr"`):
  - `"opr"` — one sample per cycle (OPR decimation). `N_*_window`/`step_*` must be exact integers; a fractional value raises `ValueError` instead of being silently truncated.
  - `"raw"` — uses the raw signal (no OPR decimation) inside each block. Accepts fractional `N_*_window`/`step_*`; the resolver converts them to a raw sample count via `ceil`. Produces `N_samples_per_seg` and overrides `step_seg` to a raw-sample hop.

Any pass-through parameter (`alpha`, `beta`, `reset_on_H0`, `t_stable_total`,
`training_intervals`, `cut_start_time`, `cut_end_time`, `ratio_sampling`,
`use_sprt`, `H_threshold`, `t_theorical`, `segmentation`) can be placed directly
in `params_physical` and is forwarded unchanged to the pipeline.

## `"params"` / native parameters — Complete Reference

### Mechanical / Signal Parameters

| Key | Type | Units | Description |
|---|---|---|---|
| `rpm` | `float` | cycles/min | Cycle rate of the analysis window: revolutions/min in `by_revolution` (or native), modal cycles/min (`60/T_modal`) in `by_modal`. Used to compute $f_r = \text{rpm}/60$ and the OPR downsampling step. |
| `ratio_sampling` | `float` | — | Sub-sampling multiplier applied after OPR. Effective rate = `ratio_sampling × fr`. Typical value: `50.0`. Set to `1.0` for pure OPR (1 sample/revolution). |

### Segmentation

| Key | Type | Units | Description |
|---|---|---|---|
| `N_seg` | `int` | cycles | Number of OPR samples per segment (native equivalent of `N_rev_window`/`N_modal_window`). One entropy value $H_n$ is computed per segment. Each segment spans $N_\text{seg}/f_r$ seconds. Small values → finer resolution but noisier. Typical: 2–5. |
| `step_seg` | `int` | cycles (or raw samples in `raw` mode) | Hop between consecutive segments (native equivalent of `step_rev`/`step_modal`). |
| `segmentation` | `str` | — | `"opr"` (default) or `"raw"` — see above. |
| `N_samples_per_seg` | `int` | raw samples | Block length in raw samples, only used when `segmentation="raw"`. Resolved automatically in physical modes. |

### Training Split

| Key | Type | Units | Description |
|---|---|---|---|
| `training_intervals` | `list[(t0, t1, label)]` | seconds | Preferred way to define training regions. Each tuple labels a `[t0, t1]` interval as `"stable"` (trains $P_0$) or `"chatter"` (trains $P_1$); multiple intervals per label are concatenated. Supersedes `t_stable_total` when given. |
| `t_stable_total` | `float` | seconds | **Legacy fallback**, only used when `training_intervals is None`. Time at which chatter begins: all signal before trains $P_0$, all signal after trains $P_1$. |

### SPRT Error Rates

| Key | Type | Range | Description |
|---|---|---|---|
| `alpha` | `float` | (0, 1) | Desired Type I error rate (false alarm probability). Smaller = fewer false alarms but slower detection. |
| `beta` | `float` | (0, 1) | Desired Type II error rate (missed detection probability). |
| `reset_on_H0` | `bool` | — | If `True`, the SPRT statistic $S_n$ is reset to 0 each time the test accepts $H_0$. Enables detection of **multiple** chatter events in one signal. Set `False` for a one-shot test. |
| `use_sprt` | `bool` | — | If `True` (default), accumulate the SPRT statistic $S_n$ across segments. If `False`, each segment is thresholded independently against `H_threshold` instead of running the sequential test. |
| `H_threshold` | `float` | — | Per-segment entropy threshold used when `use_sprt=False`. Ignored when `use_sprt=True`. |

### Signal Window (Optional)

| Key | Type | Units | Default | Description |
|---|---|---|---|---|
| `cut_start_time` | `float` | seconds | `t_analysis[0]` | Start of the analysis window. Ignores signal before this time. Useful to skip transient startup. |
| `cut_end_time` | `float` | seconds | `t_analysis[-1]` | End of the analysis window. |
| `t_theorical` | `float` | seconds | `None` | Theoretical/ground-truth chatter onset, used only for debug/plots (`t_d_no_FAR`) — not used in detection itself. |

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

## Minimal Config Examples

Native mode:

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

`by_revolution` mode:

```python
config = {
    "func": "Default",
    "param_mode": "by_revolution",
    "params_physical": {
        "T_rev": 60.0 / 12_000.0,
        "N_rev_window": 5,
        "step_rev": 1,           # mandatory
        "t_stable_total": 5.0,
        "alpha": 0.05,
        "beta": 0.05,
        "reset_on_H0": True,
    },
}
```

---

[← Signal Data](signal_data.md){ .md-button } [Run & Plot →](run_and_plot.md){ .md-button }

