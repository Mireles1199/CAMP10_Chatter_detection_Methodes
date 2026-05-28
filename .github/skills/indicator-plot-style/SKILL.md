---
name: indicator-plot-style
description: >
  Canonical matplotlib plotting style for chatter-detection indicator figures in
  CAMP10 (maxent_sprt, rms_cv, ssq_chatter, emd_hht, green_integral).
  Use when: creating any plot function for an indicator; adding vertical event-time
  lines or time annotations (t_gt, t_d); adding horizontal threshold labels (b, a,
  mu±3sigma); deciding figure colors for stable vs chatter regions; setting up a
  new viz/*.py module; asked about "indicator figure style", "cómo graficar",
  "add vline", "annotate threshold", "nueva figura indicador", "estilo gráfica".
  Always apply ALL conventions from this skill (style, fig_size, colors, helpers)
  when writing or modifying any indicator visualization file.
---

# Indicator Plot Style — CAMP10

All indicator visualization files (`viz/*.py`) in CAMP10 must follow these
conventions exactly. The canonical reference is
`indicators/maxent_sprt/src/MaxEnt_SPRT/viz/maxent_sprt_plots.py`.

---

## 1. Global Style — `configurar_estilo_global()`

**Rule**: call at module level (top of each `viz/*.py` file, not inside functions).
Copy this function verbatim into every new indicator viz module.

```python
import colorsys
import matplotlib.pyplot as plt

def configurar_estilo_global() -> None:
    local_style = {
        # Typography
        'font.family': 'serif',
        'font.size': 9,
        # Titles and labels
        'axes.titlesize': 25,
        'axes.labelsize': 25,
        'xtick.labelsize': 23,
        'ytick.labelsize': 23,
        'legend.fontsize': 23,
        # Lines
        'lines.linewidth': 1.25,
        'lines.markersize': 6,
        # Axes borders
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        # Ticks
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2.5,
        'ytick.minor.size': 2.5,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        # Math text
        'mathtext.fontset': 'stix',
        'axes.formatter.use_mathtext': True,
        # Legend
        'legend.frameon': False,
        'legend.loc': 'best',
        'legend.handlelength': 2.0,
        'legend.borderaxespad': 0.5,
        # Export
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'savefig.transparent': True,
        # Background
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    }
    plt.rcParams.update(local_style)

configurar_estilo_global()   # ← call immediately at module level
```

> **Known deviations to fix**: `rms_cv` and `ssq_chatter` currently use
> `font.size=16`, `axes.labelsize=18` — update to match the values above.

---

## 2. Figure Size — `fig_size()`

**Rule**: always use `fig_size()` instead of hardcoded `figsize` values.

```python
def fig_size(scale=1.0, ncols=1, base_width=3.4):
    """Return (width, height) in inches for IEEE/Elsevier journals."""
    width = base_width * ncols * scale
    height = width * 0.70   # canonical aspect ratio
    return (width, height)
```

Usage examples:
```python
fig, ax  = plt.subplots(figsize=fig_size(scale=3.0))           # single panel
fig, axes = plt.subplots(1, 2, figsize=fig_size(scale=3.0, ncols=2))  # side-by-side
fig, axes = plt.subplots(2, 1, figsize=fig_size(scale=3.0, ncols=1))  # stacked
```

> **Known deviation**: `rms_cv` and `ssq_chatter` use `height = width * 0.70` — must
> be changed to `0.40`.

---

## 3. Color Palette

Copy these definitions verbatim. Use only these five canonical names.

```python
import colorsys

r, g, b = colorsys.hls_to_rgb(346/360, 0.45, 0.99)
color_red    = (r, g, b)   # alarm / SPRT threshold b / false-alarm scatter

r, g, b = colorsys.hls_to_rgb(36/360, 0.45, 0.99)
color_orange = (r, g, b)   # chatter signal / detection time td / P1 PDF

r, g, b = colorsys.hls_to_rgb(279/360, 0.36, 0.99)
color_purple = (r, g, b)   # SPRT statistic Sk / auxiliary curves

r, g, b = colorsys.hls_to_rgb(98/360, 0.36, 0.99)
color_verde  = (r, g, b)   # stable threshold a / P0 PDF / Lambda=0 region

r, g, b = colorsys.hls_to_rgb(206.957/360, 0.40941, 0.55603)
color_azul   = (r, g, b)   # stable signal / P0 histogram
```

> **Forbidden names** (legacy `rms_cv`): `color_rms`, `color_CV` — rename to
> `color_azul` and `color_orange` respectively.

---

## 4. Signal / Region Color Convention

| Signal / event | Color |
|---|---|
| Stable signal or stable training data | `color_azul` |
| Chatter signal or chatter training data | `color_orange` |
| Detection time $t_d$ vertical line | `color_orange` |
| Ground-truth time $t_{gt}$ vertical line | `"black"` |
| SPRT upper threshold $b$ | `color_red` |
| SPRT lower threshold $a$ | `color_verde` |
| SPRT statistic $S_k$ | `color_purple` |
| OPR sample scatter | `color_red` |
| Gaussian mean $\mu$ vertical line | `color_red` |

---

## 5. Helper — `_draw_vlines()`

Add this helper as a nested function inside the main plot function.
It accepts plain floats or labeled tuples.

```python
def _draw_vlines(ax, vlines, default_color="black", default_ls="--"):
    """Draw vertical lines with optional rotated text labels.

    Each entry in vlines may be:
      - float                     → dashed line, no label
      - (float, label)            → dashed line + vertical label (default_color)
      - (float, label, color)     → dashed line + vertical label (custom color)
    """
    if vlines is None:
        return
    for entry in vlines:
        if isinstance(entry, (list, tuple)):
            vx    = float(entry[0])
            label = str(entry[1]) if len(entry) > 1 else None
            color = entry[2]      if len(entry) > 2 else default_color
        else:
            vx, label, color = float(entry), None, default_color
        ax.axvline(x=vx, color=color, linestyle=default_ls, lw=1.2)
        if label:
            ax.text(
                vx, 0.97, f"  {label}",
                rotation=90, va="top", ha="right", fontsize=16,
                color=color, transform=ax.get_xaxis_transform(),
            )
```

Build `vlines` lists as labeled tuples so labels are automatic:
```python
auto_vlines = []
if t_gt is not None:
    auto_vlines.append((t_gt,   f"$t_{{gt}}={t_gt:.3f}$ s",  "black"))
if t_det is not None:
    auto_vlines.append((t_det,  f"$t_d={t_det:.3f}$ s",       color_orange))
```

---

## 6. Vertical Event-Line Annotation Pattern

For inline use (without `_draw_vlines`), always use this exact pattern.
**Never** use `ax.annotate` with `xytext` / `offset points` for time labels.

```python
# Draw line
ax.axvline(x=t_val, color=color, ls="--", lw=1.2)
# Draw rotated label (positioned at 97 % of axes height)
ax.text(
    t_val, 0.97, f"  $t_{{gt}}={t_val:.3f}$ s",
    rotation=90, va="top", ha="right",
    fontsize=16, color=color,
    transform=ax.get_xaxis_transform(),
)
```

Key parameters (do NOT change):
- `y = 0.97` — near top of axes
- `rotation=90` — vertical
- `va="top"`, `ha="right"` — text hangs left of the line
- `transform=ax.get_xaxis_transform()` — x in data coords, y in axes [0,1]
- Leading spaces `"  $label$"` — small gap between line and text

---

## 7. Horizontal Threshold Annotation Pattern

For hline labels, always use `get_yaxis_transform()`.
**Never** use `ax.annotate` for threshold labels.

```python
# Draw line
ax.axhline(y=val, color=color, ls="--", lw=1.4)
# Draw label (positioned at 99 % of axes width, at the y data value)
ax.text(
    0.99, val, rf"$b = {val:.4g}$",
    transform=ax.get_yaxis_transform(),
    color=color, ha='right', va='bottom', fontsize=16,
)
```

Use `va='bottom'` when the label should appear above the line,
`va='top'` when it should appear below (e.g. for the lower threshold `a`).

Full example (SPRT thresholds `b` and `a`):
```python
ax.axhline(y=b_val, color=color_red,   ls="--", lw=1.4)
ax.axhline(y=a_val, color=color_verde, ls="--", lw=1.0)
ax.axhline(y=0,     color="gray",      ls=":",  lw=0.8)
ax.text(0.99, b_val, rf"$b={b_val:.4g}$",
        transform=ax.get_yaxis_transform(),
        color=color_red, ha='right', va='bottom', fontsize=16)
ax.text(0.99, a_val, rf"$a={a_val:.4g}$",
        transform=ax.get_yaxis_transform(),
        color=color_verde, ha='right', va='top', fontsize=16)
```

---

## 8. Background Shading by Training Interval — `_shade_intervals_local()`

Add this helper whenever a plot has training-interval metadata.
`training_intervals` is a list of `(t_start, t_end, label)` tuples where
`label` is `"stable"` or `"chatter"`.

```python
def _shade_intervals_local(ax, training_intervals, alpha_bg=0.06):
    """Tint axis background by training-interval label."""
    if training_intervals is None:
        return
    for t_lo, t_hi, lbl in training_intervals:
        c = color_azul if str(lbl).lower() == "stable" else color_orange
        ax.axvspan(t_lo, t_hi, alpha=alpha_bg, color=c, zorder=0)
```

Call it after all data plots and before `ax.legend()`:
```python
_shade_intervals_local(ax, training_intervals)
```

---

## 9. `fill_between` for Region Highlighting

Use translucent fills to highlight which distribution or signal region is
active at each x/y position.

```python
# Highlight where chatter is more likely (orange) vs stable (blue)
ax.fill_between(x, y0, y1, where=(y1 > y0), alpha=0.12, color=color_orange,
                label=r"chatter region")
ax.fill_between(x, y0, y1, where=(y1 < y0), alpha=0.12, color=color_azul,
                label=r"stable region")

# Highlight ±3σ band (PDF figures)
ax.fill_between(t_arr, thr_lo, thr_hi, alpha=0.07, color=color_verde,
                label=r"$\pm 3\sigma_0$ band")
```

Alpha guide: `0.06–0.08` for backgrounds, `0.10–0.15` for overlapping regions,
`0.20–0.40` for probability fills (alpha/beta error areas).

---

## 10. OPR Sample Scatter

Always use these exact parameters for scatter plots of OPR samples:

```python
ax.scatter(t_opr, v_opr, color=color_red, s=7, zorder=5, label="OPR Sampled")
```

The continuous signal behind the scatter uses `color_azul` (stable) or
`color_orange` (chatter) with `alpha=0.9`.

---

## 11. Histogram + Gaussian PDF Overlay

Standard pattern for any entropy / statistic distribution figure:

```python
# Histogram
ax.hist(H_data, density=True, alpha=0.5, color=color_hist, bins=50,
        label="Histogram")

# Gaussian PDF curve
x_pdf = np.linspace(mu - 4.5*sigma, mu + 4.5*sigma, 1000)
y_pdf = np.exp(-0.5*((x_pdf - mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi))
ax.plot(x_pdf, y_pdf, color=color_pdf,
        label=rf"PDF  $\mu$={mu:.4f}, $\sigma$={sigma:.4f}")

# Reference lines: mean + ±1σ / ±2σ / ±3σ
ax.axvline(mu, color=color_red, ls='-', label=rf"$\mu$ = {mu:.4f}")
sigma_styles = [('--', 0.85), (':', 0.65), ('-.', 0.45)]
for k, (ls, alpha) in enumerate(sigma_styles, start=1):
    ax.axvline(mu + k*sigma, color='gray', ls=ls, alpha=alpha)
    ax.axvline(mu - k*sigma, color='gray', ls=ls, alpha=alpha)
```

Stable data: `color_hist=color_azul`, `color_pdf=color_verde`.
Chatter data: `color_hist=color_orange`, `color_pdf=color_red`.

---

## 12. Scientific Notation on Y-Axis

For cumulative statistics (`S_k`) and large-magnitude signals:

```python
ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
```

Add immediately after all `ax.plot()` calls, before `ax.set_title()`.

---

## 13. Complete Module Template

Minimum boilerplate for a new `viz/plots_<indicator>.py` file:

```python
from __future__ import annotations
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence

# ── Color palette ─────────────────────────────────────────────────────────────
r, g, b = colorsys.hls_to_rgb(346/360, 0.45, 0.99);  color_red    = (r, g, b)
r, g, b = colorsys.hls_to_rgb(36/360,  0.45, 0.99);  color_orange = (r, g, b)
r, g, b = colorsys.hls_to_rgb(279/360, 0.36, 0.99);  color_purple = (r, g, b)
r, g, b = colorsys.hls_to_rgb(98/360,  0.36, 0.99);  color_verde  = (r, g, b)
r, g, b = colorsys.hls_to_rgb(206.957/360, 0.40941, 0.55603); color_azul = (r, g, b)

# ── Figure helpers ─────────────────────────────────────────────────────────────
def fig_size(scale=1.0, ncols=1, base_width=3.4):
    width = base_width * ncols * scale
    return (width, width * 0.40)

def configurar_estilo_global() -> None:
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 9,
        'axes.titlesize': 25,   'axes.labelsize': 25,
        'xtick.labelsize': 23,  'ytick.labelsize': 23, 'legend.fontsize': 23,
        'lines.linewidth': 1.25, 'lines.markersize': 6,
        'axes.linewidth': 0.8,   'grid.linewidth': 0.5,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.direction': 'in',  'ytick.direction': 'in',
        'xtick.major.size': 4,  'ytick.major.size': 4,
        'xtick.minor.size': 2.5, 'ytick.minor.size': 2.5,
        'xtick.minor.width': 0.6, 'ytick.minor.width': 0.6,
        'mathtext.fontset': 'stix', 'axes.formatter.use_mathtext': True,
        'legend.frameon': False,   'legend.loc': 'best',
        'figure.dpi': 100, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'savefig.transparent': True,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    })

configurar_estilo_global()

# ── Main plot function ─────────────────────────────────────────────────────────
def plots_<indicator>(signal, result, t_gt=None, scale=5.0, **kwargs):

    def _draw_vlines(ax, vlines, default_color="black", default_ls="--"):
        if vlines is None:
            return
        for entry in vlines:
            if isinstance(entry, (list, tuple)):
                vx    = float(entry[0])
                label = str(entry[1]) if len(entry) > 1 else None
                color = entry[2]      if len(entry) > 2 else default_color
            else:
                vx, label, color = float(entry), None, default_color
            ax.axvline(x=vx, color=color, linestyle=default_ls, lw=1.2)
            if label:
                ax.text(vx, 0.97, f"  {label}", rotation=90, va="top", ha="right",
                        fontsize=16, color=color, transform=ax.get_xaxis_transform())

    def _shade_intervals_local(ax, training_intervals, alpha_bg=0.06):
        if training_intervals is None:
            return
        for t_lo, t_hi, lbl in training_intervals:
            c = color_azul if str(lbl).lower() == "stable" else color_orange
            ax.axvspan(t_lo, t_hi, alpha=alpha_bg, color=c, zorder=0)

    # ... sub-plot functions here ...
```
