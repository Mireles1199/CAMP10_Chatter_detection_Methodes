"""Plotting helpers for the green_integral indicator."""

from __future__ import annotations

import colorsys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm as _scipy_norm

# ---------------------------------------------------------------------------
# Canonical colour palette
# ---------------------------------------------------------------------------
color_red    = colorsys.hls_to_rgb(346/360, 0.45, 0.99)
color_orange = colorsys.hls_to_rgb(36/360,  0.45, 0.99)
color_purple = colorsys.hls_to_rgb(279/360, 0.36, 0.99)
color_verde  = colorsys.hls_to_rgb(98/360,  0.36, 0.99)
color_azul   = colorsys.hls_to_rgb(206.957/360, 0.40941, 0.55603)


def fig_size(scale: float = 5.0) -> tuple[float, float]:
    """Return ``(width, height)`` in inches with height = width * 0.70."""
    w = scale
    return (w, w * 0.70)


def _configurar_estilo() -> None:
    mpl.rcParams.update({
        'font.family':                 'serif',
        'font.size':                   18,
        'axes.titlesize':              18,
        'axes.labelsize':              18,
        'xtick.labelsize':             16,
        'ytick.labelsize':             16,
        'lines.linewidth':             1.5,
        'mathtext.fontset':            'stix',
        'axes.formatter.use_mathtext': True,
        'legend.frameon':              False,
        'legend.loc':                  'best',
        'savefig.dpi':                 300,
        'savefig.bbox':                'tight',
        'savefig.transparent':         True,
        'figure.facecolor':            'white',
        'axes.facecolor':              'white',
    })


_configurar_estilo()


def _draw_vlines(ax, vlines, default_color="black", default_ls="--"):
    if vlines is None:
        return
    for entry in vlines:
        if isinstance(entry, (int, float)):
            ax.axvline(entry, color=default_color, ls=default_ls, lw=1.2)
        elif len(entry) == 2:
            x, label = entry
            ax.axvline(x, color=default_color, ls=default_ls, lw=1.2)
            ax.text(x, 0.97, f"  {label}",
                    rotation=90, va="top", ha="right", fontsize=16,
                    color=default_color, transform=ax.get_xaxis_transform())
        else:
            x, label, col = entry[0], entry[1], entry[2]
            ax.axvline(x, color=col, ls=default_ls, lw=1.2)
            ax.text(x, 0.97, f"  {label}",
                    rotation=90, va="top", ha="right", fontsize=16,
                    color=col, transform=ax.get_xaxis_transform())



def plot_windows_local(result: Dict[str, Any], name: str = "") -> plt.Figure:
    """Scatter/line plot of per-cycle areas, grouped by window."""
    data_windows = result["data_window"]
    agrupamiento = result["agrupamiento"]
    type_method  = result.get("global_data", {}).get("type_method", "GreenIntegral")

    fig, axes = plt.subplots(figsize=fig_size(scale=5.0))
    axes.set_title(f"{type_method} \u2014 Mean Area per Cycle \u2014 {name}")
    axes.set_xlabel("Time (s)")
    axes.set_ylabel("Area")
    axes.set_yscale("log")

    # Per-cycle aggregated mean areas (from grouping)
    t_vals, y_vals_mean = [], []
    for ciclo, datos in agrupamiento.items():
        t_vals.append(datos["promedio_tiempo_window"])
        y_vals_mean.append(datos["promedio_area_window"])

    if t_vals:
        ordenados = sorted(zip(t_vals, y_vals_mean), key=lambda p: p[0])
        t_s, y_s = zip(*ordenados)
        axes.plot(
            t_s, y_s,
            color=color_azul,
            marker="o", linewidth=1, markersize=1, alpha=0.99, label="Mean Area",
        )

    # Secondary x-axis: window index
    N = len(data_windows)
    t_n_values    = np.array([dw["indicadores"]["t_n"] for dw in data_windows])
    window_indices = np.arange(N)

    ax2 = axes.twiny()
    ax2.set_xlabel("Index of data_window")
    ax2.set_xlim(axes.get_xlim())

    def _update(event=None):
        xmin, xmax = axes.get_xlim()
        # Keep the index axis synced with the time axis at every zoom level —
        # previously this only ran when >1 point was visible, so zooming in
        # past a single point left ax2 showing stale ticks from before the zoom.
        ax2.set_xlim(xmin, xmax)
        visible = (t_n_values >= xmin) & (t_n_values <= xmax)
        vt = t_n_values[visible]
        vi = window_indices[visible]
        if len(vt) > 1:
            step = max(1, len(vt) // 20)
            ax2.set_xticks(vt[::step])
            ax2.set_xticklabels(vi[::step], rotation=0)

    fig.canvas.mpl_connect("draw_event", _update)
    axes.callbacks.connect("xlim_changed", _update)
    _update()

    # --- mu ± z*sigma threshold lines (optional) --------------------------
    thr = result.get("global_data", {}).get("area_mu_3sigma", {})
    if thr:
        z_lbl = f"{thr['z']:.0f}"
        axes.axhline(thr["upper"], color=color_red,   ls="--", lw=1.4)
        axes.text(0.99, thr["upper"],
                  rf"$\mu+{z_lbl}\sigma={thr['upper']:.3g}$",
                  transform=axes.get_yaxis_transform(),
                  color=color_red, ha='right', va='bottom', fontsize=16)
        axes.axhline(thr["lower"], color=color_red,   ls=":",  lw=1.2)
        axes.text(0.99, thr["lower"],
                  rf"$\mu-{z_lbl}\sigma={thr['lower']:.3g}$",
                  transform=axes.get_yaxis_transform(),
                  color=color_red, ha='right', va='top', fontsize=16)
        axes.axhline(thr["mu"], color=color_verde, ls="-", lw=1.0)
        axes.text(0.99, thr["mu"], rf"$\mu={thr['mu']:.3g}$",
                  transform=axes.get_yaxis_transform(),
                  color=color_verde, ha='right', va='bottom', fontsize=16)

    t_d = result.get("t_d")
    if t_d is not None:
        _draw_vlines(axes,
                     [(t_d, rf"$t_d={t_d:.3f}$ s", color_orange)])

    axes.legend()
    fig.tight_layout()
    return fig


def plot_windows_duration(result: Dict[str, Any], name: str = "") -> List[plt.Figure]:
    """Two-panel plot of window durations (by index and by time)."""
    data_windows = result["data_window"]
    type_method  = result.get("global_data", {}).get("type_method", "GreenIntegral")

    t_n_values = np.array([dw["indicadores"]["t_n"] for dw in data_windows])
    durations  = np.array([dw["window_duration"] for dw in data_windows])

    # Figure 1: duration vs window index
    fig1, ax1 = plt.subplots(figsize=fig_size(scale=5.0))
    ax1.set_title(f"Window Duration \u2014 {type_method} \u2014 {name}")
    ax1.set_xlabel("Window Number")
    ax1.set_ylabel("Window Duration [s]")
    ax1.plot(np.arange(len(data_windows)), durations,
             color=color_azul, lw=1.5, marker="o", markersize=5)
    fig1.tight_layout()

    # Figure 2: duration vs time
    fig2, ax2 = plt.subplots(figsize=fig_size(scale=5.0))
    ax2.set_title(f"Window Duration (Time) \u2014 {type_method} \u2014 {name}")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Window Duration [s]")
    ax2.plot(t_n_values, durations,
             color=color_azul, lw=1.5, marker="o", markersize=3)
    fig2.tight_layout()

    return [fig1, fig2]


def plot_indicator_local(result: Dict[str, Any], name: str = "") -> plt.Figure:
    """Plot the per-window ``delta_n`` indicator over time."""
    data_windows = result["data_window"]
    type_method  = result.get("global_data", {}).get("type_method", "GreenIntegral")

    t_n_values    = np.array([dw["indicadores"]["t_n"]    for dw in data_windows])
    delta_n_values = np.array([dw["indicadores"]["delta_n"] for dw in data_windows])
    window_indices = np.arange(len(data_windows))

    fig, axes = plt.subplots(figsize=fig_size(scale=5.0))
    axes.set_title(f"{type_method} \u2014 Delta_n per Window \u2014 {name}")
    axes.set_xlabel("Time (s)")
    axes.set_ylabel("Delta_n")

    for dw in data_windows:
        axes.plot(
            dw["indicadores"]["t_n"], dw["indicadores"]["delta_n"],
            color=color_azul, marker="o", markersize=3,
            linestyle="-", linewidth=1.5,
        )

    axes.plot(t_n_values, delta_n_values,
              color="black", marker="", linewidth=1, alpha=0.75)
    axes.set_yscale("linear")

    # Secondary x-axis
    ax2 = axes.twiny()
    ax2.set_xlabel("Index of data_window")
    ax2.set_xlim(axes.get_xlim())

    def _update(event=None):
        xmin, xmax = axes.get_xlim()
        # Keep the index axis synced with the time axis at every zoom level —
        # previously this only ran when >1 point was visible, so zooming in
        # past a single point left ax2 showing stale ticks from before the zoom.
        ax2.set_xlim(xmin, xmax)
        visible = (t_n_values >= xmin) & (t_n_values <= xmax)
        vt = t_n_values[visible]
        vi = window_indices[visible]
        if len(vt) > 1:
            step = max(1, len(vt) // 20)
            ax2.set_xticks(vt[::step])
            ax2.set_xticklabels(vi[::step], rotation=0)

    fig.canvas.mpl_connect("draw_event", _update)
    axes.callbacks.connect("xlim_changed", _update)
    _update()

    fig.tight_layout()
    return fig


def plot_training_distribution(
    global_data: Dict[str, Any],
    name: str = "",
    log_transform: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
) -> List[plt.Figure]:
    """Diagnose the population that actually trained the mu +- z*sigma threshold.

    Reads ``global_data["training_areas"]``/``["training_t_wins"]`` — the
    exact windows used, populated by the runner regardless of whether the
    source was ``training_intervals`` (``training_source="internal"``) or an
    external ``reference_signal`` (``training_source="external_reference"``).
    Unlike re-deriving a "stable" slice from ``training_intervals`` against
    whatever signal is being plotted, this can never point at the wrong one.

    ``figsize`` must match the actual (width, height) inches used by the
    sibling figures in the caller's own figure set (default: this module's
    own ``fig_size(scale=5.0)``). Callers with a *different* figure size —
    e.g. green_integral_plots.py's plots_lyapunov, whose C1-C3/Ĝ panels use
    ITS OWN ``fig_size(scale=3.0)`` (a same-named but differently-scaled
    helper — passing a bare number here would silently use the wrong
    formula) — must compute their own size and pass the tuple explicitly,
    otherwise these two figures render at a visibly different size than the
    rest of the set.

    Returns two figures (empty list if there isn't enough trained data):

    1. Histogram of the training population + the actual fitted Gaussian
       (mu/sigma taken from ``global_data["area_mu_3sigma"]``, never
       recomputed, so the curve always matches the real threshold).
    2. The same values plotted as a curve (vs. their own time axis when
       available, else sample index) — to visually verify the "roughly
       Gaussian" assumption (trend, outliers) that fed the normal law.
    """
    thr = global_data.get("area_mu_3sigma") or {}
    train_areas = np.asarray(global_data.get("training_areas", []), dtype=float)
    train_t = np.asarray(global_data.get("training_t_wins", []), dtype=float)

    valid = np.isfinite(train_areas) & (train_areas > 0)
    if not thr or valid.sum() < 5 or not (float(thr.get("sigma", 0.0)) > 0):
        return []

    vals = np.log10(train_areas[valid]) if log_transform else train_areas[valid]
    t_vals = train_t[valid] if train_t.shape == train_areas.shape else np.array([])

    mu_h, std_h = float(thr["mu"]), float(thr["sigma"])
    # Verify the plotted population is genuinely what trained the threshold
    # above, not a stale/mismatched array: mu_h/std_h were derived by the
    # runner FROM this exact `vals`, so recomputing them here must reproduce
    # the same numbers. Catches a future regression silently decoupling the
    # two (which is exactly the bug this function was written to fix).
    assert np.isclose(float(np.mean(vals)), mu_h, rtol=1e-6, atol=1e-9), (
        "plot_training_distribution: training_areas does not match the "
        "population that trained area_mu_3sigma — mu mismatch"
    )
    assert np.isclose(float(np.std(vals, ddof=1)), std_h, rtol=1e-6, atol=1e-9), (
        "plot_training_distribution: training_areas does not match the "
        "population that trained area_mu_3sigma — sigma mismatch"
    )
    z = float(thr.get("z", 3.0))
    hi = float(thr.get("upper", mu_h + z * std_h))
    lo = float(thr.get("lower", mu_h - z * std_h))
    z_lbl = f"{z:.0f}"
    xlabel = r"$\log_{10}(A_k)$" if log_transform else r"$A_k$"
    _figsize = figsize if figsize is not None else fig_size(scale=5.0)

    figs: List[plt.Figure] = []

    # ── Histogram + Gaussian PDF (mu/sigma from the real threshold) ────────
    fig_h, ax_h = plt.subplots(figsize=_figsize)
    ax_h.set_title(f"Training Area Distribution — {name}")
    ax_h.set_xlabel(xlabel)
    ax_h.set_ylabel("Density")
    ax_h.hist(vals, bins=40, density=True, alpha=0.55, color=color_azul,
              label=f"Training pop. (n={len(vals)})")
    xs = np.linspace(mu_h - 4 * std_h, mu_h + 4 * std_h, 300)
    ax_h.plot(xs, _scipy_norm.pdf(xs, mu_h, std_h), color=color_azul, lw=1.8,
              label=rf"$\mathcal{{N}}(\mu={mu_h:.3g},\,\sigma={std_h:.3g})$")
    ax_h.axvline(mu_h, color=color_verde, ls="-", lw=1.4)
    ax_h.axvline(hi, color=color_red, ls="--", lw=1.4, label=rf"$\mu+{z_lbl}\sigma$")
    ax_h.axvline(lo, color=color_red, ls=":", lw=1.2, label=rf"$\mu-{z_lbl}\sigma$")
    ax_h.legend()
    fig_h.tight_layout()
    figs.append(fig_h)

    # ── Curve: same values, in the order used to fit the normal law ────────
    fig_c, ax_c = plt.subplots(figsize=_figsize)
    ax_c.set_title(f"Training Curve — normal-law input — {name}")
    if t_vals.size == vals.size:
        x_axis, ax_c_xlabel = t_vals, "Time [s]"
    else:
        x_axis, ax_c_xlabel = np.arange(len(vals)), "Sample index"
    ax_c.set_xlabel(ax_c_xlabel)
    ax_c.set_ylabel(xlabel)
    ax_c.plot(x_axis, vals, color=color_azul, lw=1.0, marker="o", markersize=2,
              label=f"Training pop. (n={len(vals)})")
    ax_c.axhline(mu_h, color=color_verde, ls="-", lw=1.4, label=rf"$\mu={mu_h:.3g}$")
    ax_c.axhline(hi, color=color_red, ls="--", lw=1.4, label=rf"$\mu+{z_lbl}\sigma={hi:.3g}$")
    ax_c.axhline(lo, color=color_red, ls=":", lw=1.2, label=rf"$\mu-{z_lbl}\sigma={lo:.3g}$")
    ax_c.legend()
    fig_c.tight_layout()
    figs.append(fig_c)

    return figs
