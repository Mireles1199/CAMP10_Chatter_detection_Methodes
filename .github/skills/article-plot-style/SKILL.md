---
name: time-plot-style
description: >
  Canonical matplotlib plotting style for frozen-time stability figures (ChatterPlotter
  in time.py — Artiuclo_Manuf21_2026). Use when: creating any frozen-time stability
  plot; adding stability-crossing markers (t_E, a_E); plotting depth-of-cut profiles
  with R markers; plotting visible chatter sweeps (a_vis, t_vis vs R); plotting
  cumulative growth G(t) or relative amplitude e^G(t); adding a synchronized upper
  X-axis (ap [mm] ↔ t [s]); asked about "estilo time.py", "plot estabilidad", "frozen
  time figure style", "scientific axis formatter", "secondary x-axis ap". Always apply
  ALL conventions (rcParams, colors, formatters, markers) when writing or modifying any
  frozen-time stability visualization file.
---

# Frozen-Time Stability Plot Style — time.py (Artiuclo_Manuf21_2026)

Canonical reference: `Artiuclo_Manuf21_2026/Comparation/time.py` → `ChatterPlotter`.

---

## 1. Global rcParams — `_configurar_estilo_global()`

Call once at the start (inside `__init__` of the plotter class, or at module level).

```python
import matplotlib.pyplot as plt

local_style = {
    # Typography
    'font.family': 'serif',
    'font.size': 12,
    # Titles and labels
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 10,
    # Lines
    'lines.linewidth': 1.2,
    'lines.markersize': 10,
    # Axes borders
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    # Ticks — inward, with minor ticks
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
    # Math text — STIX font
    'mathtext.fontset': 'stix',
    'axes.formatter.use_mathtext': True,
    # Legend — no frame
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
    # Background — white
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
}
plt.rcParams.update(local_style)
```

---

## 2. Figure sizes (shared layout)

| # panels | figsize |
|----------|---------|
| 1–3 | `(5*n + 1, 4)` with `plt.subplots(1, n)` |
| 4–5 | `(6*ncols, 5*nrows)` with `ncols=3 if n==5 else 2` |
| Single standalone | `(3.5*3.5, 2.5*3.5)` ≈ `(12.25, 8.75)` |

Use `constrained_layout=True` on the figure. Do NOT call `tight_layout()`.

---

## 3. Colors

| Element | Color |
|---------|-------|
| Main stability curve Re{λ_max} | `'steelblue'` |
| Stable fill (σ < 0) | `color='green', alpha=0.15` |
| Unstable fill (σ ≥ 0) | `color='red', alpha=0.15` |
| Stability-crossing marker (t_E, a_E, axvline) | `'crimson'` |
| Depth-of-cut profile line | `'navy'` |
| Cumulative growth G(t) | `'darkorange'` |
| a_vis curve (twin axis) | `'crimson'` |
| t_vis curve (twin axis) | `'steelblue'` |
| R-sweep color gradient | `plt.cm.plasma(np.linspace(0.1, 0.85, n_R))` |

---

## 4. Scatter markers for key events

```python
# Stability crossing at t_E / a_E
ax.scatter([x_mark], [0.0],
           color='crimson', zorder=5, s=75,
           edgecolor='k', linewidths=0.6)

# Visible chatter crossings (per R)
ax.scatter([x_mark], [target],
           color=col, zorder=5, s=75,
           edgecolor='k', linewidths=0.6)
```

---

## 5. Line styles for events

```python
ax.axhline(0.0, color='k', linewidth=0.8, linestyle='--')   # zero line
ax.axvline(t_E, color='crimson', linewidth=1.2, linestyle=':')   # t_E marker
ax.axhline(a_E, color='crimson', linewidth=1.0, linestyle='-.')  # a_E marker
ax.axvline(t_vis, color=col, linewidth=1.0, linestyle='--')      # R-sweep lines
```

---

## 6. Scientific Y-axis formatter (apply to every ax)

```python
import matplotlib.ticker as mticker

fmt = mticker.ScalarFormatter(useMathText=True)
fmt.set_scientific(True)
fmt.set_powerlimits((-2, 2))
ax.yaxis.set_major_formatter(fmt)
ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

# Move the offset text (×10^n label)
off = ax.yaxis.get_offset_text()
off.set_size(14)
off.set_x(-0.12)
off.set_y(-0.05)
```

---

## 7. Synchronized upper X-axis — ap [mm] ↔ t [s]

Use `_add_top_xaxis_interactive` when `x_axis='both'`. Do NOT use `secondary_xaxis`
for non-linear profiles — use `twiny()` with a manual formatter instead.

```python
def add_top_xaxis_ap(ax, t, a_mm, mode="auto", tick_step=1):
    """
    Add synchronized upper X-axis: ap [mm] ← t [s].
    mode: 'linear' | 'step' | 'auto' (auto-detects constant segments)
    """
    import numpy as np

    secax = ax.twiny()
    secax.set_xlabel(r"Depth $a_p$ [mm]")
    secax.set_navigate(False)

    t     = np.asarray(t, dtype=float)
    a_mm  = np.asarray(a_mm, dtype=float)

    def detect_mode(a):
        da = np.diff(a)
        if np.all(np.isclose(da, 0.0, rtol=1e-9, atol=1e-12)):
            return "step"
        if np.any(np.isclose(da, 0.0, rtol=1e-9, atol=1e-12)):
            return "step"
        return "linear"

    if mode == "auto":
        mode = detect_mode(a_mm)

    def map_t_to_ap(x):
        x = np.asarray(x, dtype=float)
        if mode == "step":
            idx = np.searchsorted(t, x, side="right") - 1
            idx = np.clip(idx, 0, len(a_mm) - 1)
            return a_mm[idx]
        return np.interp(x, t, a_mm)

    # Copy lower ticks and relabel with a_p values
    ax.figure.canvas.draw()
    lower_ticks = ax.get_xticks()
    xmin, xmax  = ax.get_xlim()
    visible     = lower_ticks[(lower_ticks >= xmin) & (lower_ticks <= xmax)]
    if tick_step > 1:
        visible = visible[::tick_step]

    secax.set_xlim(ax.get_xlim())
    secax.set_xticks(visible)
    ap_labels = map_t_to_ap(visible)
    secax.set_xticklabels([f"{v:.1f}" for v in ap_labels])
    return secax
```

---

## 8. Legend conventions

- No frame: `legend.frameon = False` (via rcParams, already set)
- Stability plot: `loc='upper left'`
- Twin-axis plots: combine handles from both axes:
  ```python
  lines1, labs1 = ax.get_legend_handles_labels()
  lines2, labs2 = ax2.get_legend_handles_labels()
  ax.legend(lines1 + lines2, labs1 + labs2, loc='lower right')
  ```
- Growth / amplitude plots: `loc='upper left'`

---

## 9. Grid

Grid is OFF by default in time.py style:
```python
# ax.grid(False)   ← default, no explicit call needed
```

---

## 10. Export

```python
fig.savefig(str(out_path), dpi=200, bbox_inches='tight')
```
- Interactive HTML: `fig.write_html(str(html_path), include_plotlyjs='cdn')` (Plotly only)
- Use `dpi=300` for final publication figures, `dpi=200` for intermediate exports.

---

## 11. Plot catalogue (ChatterPlotter)

| Method | X-axis | Y-axis | Key elements |
|--------|--------|--------|-------------|
| `plot_stability` | t [s] or ap [mm] | Re{λ_max} [s⁻¹] | fill_between green/red, axvline t_E, scatter at crossing |
| `plot_depth_profile` | t [s] | ap [mm] | navy line, axvline t_E, axhline a_E, plasma R markers |
| `plot_visible_sweep` | R factor | a_vis [mm] (left) / t_vis [s] (right) | twin-axis, 'o-' crimson + 's--' steelblue |
| `plot_growth` | t [s] | G(t) [—] | darkorange line, axhline G=0, plasma R scatter |
| `plot_amplitude` | t [s] | e^G(t) [—] | darkorange, axhline 1.0, zoom inset per R |

All methods accept `ax=None` (create their own axes) or an existing `Axes`.

---

## 12. Quick start template

```python
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── 1. Apply style ───────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12,
    'axes.titlesize': 16, 'axes.labelsize': 16,
    'xtick.labelsize': 14, 'ytick.labelsize': 14,
    'legend.fontsize': 10, 'lines.linewidth': 1.2,
    'lines.markersize': 10, 'axes.linewidth': 0.8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'mathtext.fontset': 'stix', 'axes.formatter.use_mathtext': True,
    'legend.frameon': False, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ── 2. Figure ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

# ── 3. Plot data ─────────────────────────────────────────
ax.fill_between(t, re, 0, where=(re < 0),  alpha=0.15, color='green', label='Stable')
ax.fill_between(t, re, 0, where=(re >= 0), alpha=0.15, color='red',   label='Unstable')
ax.plot(t, re, color='steelblue', linewidth=1.5, label=r'$\mathrm{Re}\{\lambda_{\max}\}$')
ax.axhline(0.0, color='k', linewidth=0.8, linestyle='--')
ax.axvline(t_E, color='crimson', linewidth=1.2, linestyle=':')
ax.scatter([t_E], [0.0], color='crimson', s=75, edgecolor='k', linewidths=0.6, zorder=5)

# ── 4. Scientific formatter ───────────────────────────────
fmt = mticker.ScalarFormatter(useMathText=True)
fmt.set_scientific(True); fmt.set_powerlimits((-2, 2))
ax.yaxis.set_major_formatter(fmt)
off = ax.yaxis.get_offset_text()
off.set_size(14); off.set_x(-0.12); off.set_y(-0.05)

# ── 5. Labels & legend ────────────────────────────────────
ax.set_xlabel('Time [s]')
ax.set_ylabel(r'$\mathrm{Re}\{\lambda_{\max}\}$ [s$^{-1}$]')
ax.set_title('Frozen-time stability')
ax.legend(loc='upper left')

plt.show()
```
