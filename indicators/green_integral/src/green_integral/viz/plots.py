"""Plotting helpers for the green_integral indicator."""

from __future__ import annotations

import colorsys
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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
    axes.set_title(f"Signal Analysis Method: {type_method} Local Data \u2014 {name}")
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
        visible = (t_n_values >= xmin) & (t_n_values <= xmax)
        vt = t_n_values[visible]
        vi = window_indices[visible]
        if len(vt) > 1:
            step = max(1, len(vt) // 20)
            ax2.set_xticks(vt[::step])
            ax2.set_xticklabels(vi[::step], rotation=0)
            ax2.set_xlim(axes.get_xlim())

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
    axes.set_title(
        f"Signal Analysis Method: {type_method} Local Indicator \u2014 {name}"
    )
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
        visible = (t_n_values >= xmin) & (t_n_values <= xmax)
        vt = t_n_values[visible]
        vi = window_indices[visible]
        if len(vt) > 1:
            step = max(1, len(vt) // 20)
            ax2.set_xticks(vt[::step])
            ax2.set_xticklabels(vi[::step], rotation=0)
            ax2.set_xlim(axes.get_xlim())

    fig.canvas.mpl_connect("draw_event", _update)
    axes.callbacks.connect("xlim_changed", _update)
    _update()

    fig.tight_layout()
    return fig
