# Visualization

Plotting and diagnostic functions for inspecting MaxEnt-SPRT detection results.
All functions take data produced by the pipeline and return Matplotlib figures.

[← Data & I/O](data_io.md){ .md-button } [API Overview](index.md){ .md-button }

---

## `plots_maxent_sprt` — Main diagnostic figure

Generates the standard multi-panel diagnostic plot: raw signal (optional), entropy
sequence, and cumulative SPRT statistic with decision thresholds.

```python
from MaxEnt_SPRT.viz.maxent_sprt_plots import plots_maxent_sprt

fig = plots_maxent_sprt(
    signal=signal,       # SignalData — set None to hide signal panel
    result=result,       # IndicatorResult
    show_signal=True,
    zoom_x=(t0, t1),     # optional x-axis zoom (seconds)
    vlines=[t_chatter],  # optional vertical marker lines
)
fig.savefig("output.pdf")
```

| Parameter | Type | Purpose |
|---|---|---|
| `signal` | `SignalData \| None` | Raw signal panel (hidden if `None`) |
| `result` | `IndicatorResult` | Detection output to visualize |
| `show_signal` | `bool` | Toggle signal panel visibility |
| `zoom_x` | `tuple[float,float] \| None` | X-axis zoom window |
| `zoom_y` | `tuple[float,float] \| None` | Y-axis zoom window |
| `vlines` | `Sequence[float] \| None` | Vertical reference lines |
| `hlines` | `Sequence[float] \| None` | Horizontal reference lines |

[→ Full reference (Sphinx)](../technical/autoapi/MaxEnt_SPRT/viz/maxent_sprt_plots/index.html){ .md-button .md-button--primary }

---

## Style utilities

```python
from MaxEnt_SPRT.viz.maxent_sprt_plots import configurar_estilo_global, fig_size

configurar_estilo_global()        # apply publication-ready rcParams globally
w, h = fig_size(scale=1.0)      # compute figure dimensions for paper columns
```
