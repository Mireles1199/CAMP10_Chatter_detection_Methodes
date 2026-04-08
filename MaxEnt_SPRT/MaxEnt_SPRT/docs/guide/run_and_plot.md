[← Indicator Configuration](indicator_config.md){ .md-button } [Real HDF5 Data →](hdf5_real_data.md){ .md-button }
# `run_maxent_sprt` and `plots_maxent_sprt`

This page documents the two main entry-point functions you call at the end of every detection run.

---

## `run_maxent_sprt`

```python
from MaxEnt_SPRT import run_maxent_sprt

result = run_maxent_sprt(signal: SignalData, INDICATOR_CONFIG: dict) -> IndicatorResult
```

### Description

Runs the complete MaxEnt-SPRT detection pipeline on a signal:

1. **Signal preparation** — splits at `t_stable_total` into stable and chatter training windows; applies `cut_start_time` / `cut_end_time`.
2. **OPR downsampling** — reduces `fs` to `ratio_sampling Ã— (rpm/60)`.
3. **Offline training** — fits Gaussian MaxEnt models $P_0$ and $P_1$ from OPR segments.
4. **Online detection** — segments the full signal, computes entropy $H_n$ per segment, accumulates $S_n$ with SPRT.
5. **Result packaging** — collects all intermediate values into `IndicatorResult`.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `signal` | `SignalData` | Packaged signal (see [SignalData](signal_data.md)) |
| `INDICATOR_CONFIG` | `dict` | Detection configuration (see [INDICATOR_CONFIG](indicator_config.md)) |

### Returns — `IndicatorResult`

```python
result.name    # str  — "MaxEnt_SPRT"
result.t       # np.ndarray — midpoint time of each segment
result.I_t     # np.ndarray — S_n SPRT statistic for each segment
result.t_d     # np.ndarray — timestamps where S_n â‰¥ b (chatter detected)
result.meta    # dict       — all intermediate results (see below)
```

#### `result.meta` keys

| Key | Content |
|---|---|
| `"fs"` | Sampling frequency |
| `"Rotational_Frequency_Hz"` | $f_r = \text{rpm}/60$ |
| `"N_seg"` | Segments per entropy value |
| `"alpha"`, `"beta"` | Configured error rates |
| `"rpm"` | Spindle speed |
| `"ratio_sampling"` | Downsampling ratio |
| `"Total_segments"` | Number of segments processed |
| `"Size_signal_free"` | Samples in the stable training window |
| `"SPRT_final_state"` | `"chatter"`, `"free"`, or `"indeterminado"` |
| `"SPRT_decision_index"` | Segment index of the first decision |
| `"SPRT_a"`, `"SPRT_b"` | Computed lower / upper thresholds |
| `"H_seq_online"` | Full entropy sequence $H_n$ |
| `"t_mid_segments"` | Time array for segments |
| `"opr_free"`, `"opr_chat"` | OPR signals used for training |
| `"t_opr_free"`, `"t_opr_chat"` | Corresponding time arrays |

### Example

```python
result = run_maxent_sprt(sig, INDICATOR_CONFIG)

# Check detection
if result.t_d.size > 0:
    print(f"Chatter first detected at t = {result.t_d[0]:.3f} s")
else:
    print("No chatter detected")

# Access intermediate entropy sequence
H_seq = result.meta["H_seq_online"]
t_seg = result.meta["t_mid_segments"]
```

---

## `plots_maxent_sprt`

```python
from MaxEnt_SPRT import plots_maxent_sprt

fig = plots_maxent_sprt(
    signal    : SignalData,
    result    : IndicatorResult,
    show_signal: bool = True,
    show      : bool = True,
    zoom_x    : tuple[float, float] | None = None,
    zoom_y    : tuple[float, float] | None = None,
    vlines    : list[float] | None = None,
    hlines    : list[float] | None = None,
) -> matplotlib.figure.Figure
```

### Description

Produces a multi-panel diagnostic figure covering the entire detection workflow from raw signal to SPRT decision.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `signal` | `SignalData` | — | Signal object (from `run_maxent_sprt`) |
| `result` | `IndicatorResult` | — | Result object (output of `run_maxent_sprt`) |
| `show_signal` | `bool` | `True` | If `True`, includes the raw signal panel |
| `show` | `bool` | `True` | If `True`, calls `plt.show()` automatically |
| `zoom_x` | `(float, float)` or `None` | `None` | Time window for zooming in on all panels |
| `zoom_y` | `(float, float)` or `None` | `None` | Y-axis limits for the signal panel |
| `vlines` | `list[float]` or `None` | `None` | List of times to draw vertical reference lines (e.g. known chatter onset) |
| `hlines` | `list[float]` or `None` | `None` | List of values to draw horizontal reference lines |

### Returns

A `matplotlib.figure.Figure` object. You can save it:

```python
fig = plots_maxent_sprt(signal=sig, result=result, show=False)
fig.savefig("detection_result.png", dpi=200, bbox_inches="tight")
```

### Panels Produced

| Panel # | Title | What it shows |
|---|---|---|
| 1 | Tool Velocity | Full signal; stable portion in blue, chatter in orange |
| 2 | OPR Stable | Stable signal with OPR sample points overlaid |
| 3 | OPR Chatter | Chatter signal with OPR sample points overlaid |
| 4 | Segment PDF | Gaussian PDF fitted to one representative segment |
| 5 | MaxEnt Models | $P_0(H)$ and $P_1(H)$ densities with their means |
| 6 | Entropy $H_n$ | Entropy sequence over time with mean $\mu_0$ and $\mu_1$ markers |
| 7 | SPRT $S_n$ | Cumulative LLR with thresholds $a$ (lower) and $b$ (upper) highlighted |

### Example

```python
# Basic plot
plots_maxent_sprt(signal=sig, result=result, show_signal=True)

# With known onset marked and saved
fig = plots_maxent_sprt(
    signal=sig, result=result,
    show_signal=True,
    vlines=[5.365],     # known chatter onset time
    zoom_x=(0, 12),
    show=False,
)
fig.savefig("result.png", dpi=150, bbox_inches="tight")
```

---

[← Indicator Configuration](indicator_config.md){ .md-button } [Real HDF5 Data →](hdf5_real_data.md){ .md-button }

