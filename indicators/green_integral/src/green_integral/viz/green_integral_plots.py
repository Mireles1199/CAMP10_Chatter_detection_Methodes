"""Main entry point for green_integral visualization."""

from __future__ import annotations

import colorsys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as _scipy_norm

from ..utils.types import SignalData, GreenIntegralResult, FixedWindowResult
from .plots import plot_windows_local, plot_windows_duration, plot_indicator_local

# ── Color palette ─────────────────────────────────────────────────────────────────────────────
r, g, b = colorsys.hls_to_rgb(346/360, 0.45, 0.99);  color_red    = (r, g, b)
r, g, b = colorsys.hls_to_rgb(36/360,  0.45, 0.99);  color_orange = (r, g, b)
r, g, b = colorsys.hls_to_rgb(279/360, 0.36, 0.99);  color_purple = (r, g, b)
r, g, b = colorsys.hls_to_rgb(98/360,  0.36, 0.99);  color_verde  = (r, g, b)
r, g, b = colorsys.hls_to_rgb(206.957/360, 0.40941, 0.55603)
color_azul = (r, g, b)


def fig_size(scale=1.0, ncols=1, base_width=3.4):
    """Return (width, height) in inches for IEEE/Elsevier journals."""
    width = base_width * ncols * scale
    return (width, width * 0.70)


def configurar_estilo_global() -> None:
    plt.rcParams.update({
        'font.family':                 'serif',
        'font.size':                   9,
        'axes.titlesize':              25,
        'axes.labelsize':              25,
        'xtick.labelsize':             23,
        'ytick.labelsize':             23,
        'legend.fontsize':             23,
        'lines.linewidth':             1.25,
        'axes.linewidth':              0.8,
        'grid.linewidth':              0.5,
        'xtick.direction':             'in',
        'ytick.direction':             'in',
        'xtick.major.size':            4,
        'ytick.major.size':            4,
        'xtick.minor.size':            2.5,
        'ytick.minor.size':            2.5,
        'xtick.major.width':           0.8,
        'ytick.major.width':           0.8,
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


configurar_estilo_global()


def _add_hline_label(ax, y_data, text, **kwargs):
    """Add a text label next to an axhline that auto-hides when y is out of view."""
    txt = ax.text(0.99, y_data, text, transform=ax.get_yaxis_transform(), **kwargs)

    def _update(ax_ref):
        ylo, yhi = ax_ref.get_ylim()
        txt.set_visible(ylo <= y_data <= yhi)

    ax.callbacks.connect('ylim_changed', _update)
    _update(ax)
    return txt


def _add_vline_label(ax, x_data, text, y_frac=0.97, **kwargs):
    """Add a rotated text label next to an axvline that auto-hides when x is out of view."""
    txt = ax.text(x_data, y_frac, f"  {text}",
                  rotation=90, va='top', ha='right',
                  transform=ax.get_xaxis_transform(), **kwargs)

    def _update(ax_ref):
        xlo, xhi = ax_ref.get_xlim()
        txt.set_visible(xlo <= x_data <= xhi)

    ax.callbacks.connect('xlim_changed', _update)
    _update(ax)
    return txt


def _draw_vlines(ax, vlines, default_color="black", default_ls="--"):
    """Draw vertical lines with optional rotated text labels.

    Each entry in vlines may be:
      - float                  → plain dashed line (default_color)
      - (float, label)         → dashed line + vertical label
      - (float, label, color)  → dashed line + vertical label (custom color)
    """
    if vlines is None:
        return
    for entry in vlines:
        if isinstance(entry, (int, float)):
            ax.axvline(entry, color=default_color, ls=default_ls, lw=1.2)
        elif len(entry) == 2:
            x, label = entry
            ax.axvline(x, color=default_color, ls=default_ls, lw=1.2)
            _add_vline_label(ax, x, label, fontsize=16, color=default_color)
        else:
            x, label, col = entry[0], entry[1], entry[2]
            ax.axvline(x, color=col, ls=default_ls, lw=1.2)
            _add_vline_label(ax, x, label, fontsize=16, color=col)


def plots_green_integral(
    signal: SignalData,
    result: GreenIntegralResult,
    show: bool = True,
    **kwargs: Any,
) -> None:
    """Produce the standard set of plots for a green_integral result.

    Parameters
    ----------
    signal : :class:`~green_integral.utils.types.SignalData`
        Original input signal (used for the signal name).
    result : :class:`~green_integral.utils.types.GreenIntegralResult`
        Output from :func:`~green_integral.run_green_integral`.
    show   : bool, default ``True``
        Call ``plt.show()`` after creating all figures.
    **kwargs
        Forwarded to individual plot functions (currently unused).
    """
    name = signal.name if signal.name else ""

    result_dict: Dict[str, Any]
    if isinstance(result, dict):
        result_dict = result
    else:
        result_dict = {
            "data_window": result.data_window,
            "agrupamiento": result.agrupamiento,
            "Mediana_delta_n": result.Mediana_delta_n,
            "global_data": result.global_data,
            "Name": result.Name,
            "t_d": result.t_d,
        }

    plot_windows_local(result_dict, name=name)
    plot_windows_duration(result_dict, name=name)
    plot_indicator_local(result_dict, name=name)

    if show:
        plt.show(block=True)


# ---------------------------------------------------------------------------
# Fixed-window indicator plots
# ---------------------------------------------------------------------------

def plots_fixed_window(
    signal: SignalData,
    result: FixedWindowResult,
    t_gt: Optional[float] = None,
    training_intervals=None,
    show: bool = True,
) -> None:
    """Produce the standard set of plots for a :class:`FixedWindowResult`.

    Figures produced
    ----------------
    C1. **Signal split** — displacement and velocity, stable (azul) vs chatter (orange).
    C2. **Areas** — shoelace area per window on a log scale.
    C3. **Lyapunov** — raw σ̂ and (if available) σ̂_EWMA.
    C4. **Histogram** — stable areas (log₁₀), Gaussian PDF, μ + μ±3σ lines.
    Ĝ  **Accumulator** — only when ``result.G_hat`` is non-empty.
    Ĝs **Sliding** — only when ``result.G_hat_sliding`` is non-empty.

    Parameters
    ----------
    signal             : original input signal.
    result             : output of :func:`run_fixed_window`.
    t_gt               : ground-truth chatter onset [s] (optional).
    training_intervals : list of ``(t0, t1, label)`` tuples. Entries whose
                         label starts with ``"stable"`` define the stable
                         training region. Falls back to ``global_data`` or
                         ``t < t_gt`` when not provided.
    show               : call ``plt.show(block=True)`` after all figures.
    """
    name   = signal.name or ""
    t_wins = np.asarray(result.t_wins)
    areas  = np.asarray(result.areas)
    trayectory_C = np.asarray(result.trayectory_C)
    trayectory_K = np.asarray(result.trayectory_K)
    sigma  = np.asarray(result.sigma)
    s_ewma = np.asarray(result.sigma_ewma)
    t_d    = result.t_d
    gd     = result.global_data or {}
    thr    = gd.get("area_mu_3sigma") or {}
    area_threshold_enabled = bool(gd.get("use_area_threshold", False))

    # training_intervals: direct param overrides global_data
    if training_intervals is None:
        training_intervals = gd.get("training_intervals")
    # If training_intervals was not explicitly provided (None), treat it as
    # "no explicit training" and avoid plotting histograms / Gaussian PDFs
    # that rely on a user-defined stable training region. This prevents
    # showing μ±3σ curves when no training intervals were supplied.
    explicit_training_provided = training_intervals is not None

    _all_stable_ranges = [
        (_t0, _t1)
        for _t0, _t1, _lbl in (training_intervals or [])
        if str(_lbl).startswith("stable")
    ]

    # shared event vlines
    auto_vlines = []
    if t_gt is not None:
        auto_vlines.append((t_gt, rf"$t_{{gt}}={t_gt:.3f}$ s", "black"))
    if t_d is not None:
        _td_val = float(t_d[0]) if len(t_d) > 0 else float("nan")
        _td_valid = (t_gt is None) or (_td_val > t_gt)
        _td_lbl = rf"$t_d^+={_td_val:.3f}$ s" if _td_valid else rf"$t_d={_td_val:.3f}$ s"
        auto_vlines.append((_td_val, _td_lbl, color_orange))

    # ── C1: Signal split (displacement + velocity) ────────────────────────
    t_arr = np.asarray(gd.get("t", []))
    q_arr = np.asarray(gd.get("q_signal", []))
    v_arr = np.asarray(gd.get("q_o_signal", []))
    if t_arr.size > 0 and q_arr.size == t_arr.size and v_arr.size == t_arr.size:
        fig_c1, (ax_x, ax_v) = plt.subplots(
            2, 1, figsize=fig_size(scale=3.0), sharex=True,
            constrained_layout=True,
        )
        fig_c1.suptitle(f"C1 — Signal — {name}")
        _split_t = t_gt if t_gt is not None else t_d
        if _all_stable_ranges:
            # plot each stable interval separately (avoids connecting lines)
            for _bi, (_t0, _t1) in enumerate(_all_stable_ranges):
                _m = (t_arr >= _t0) & (t_arr <= _t1)
                if _m.any():
                    ax_x.plot(t_arr[_m], q_arr[_m], color=color_azul,
                              lw=0.8, label="Stable" if _bi == 0 else "_nolegend_")
            if _split_t is not None:
                _mc = t_arr >= _split_t
                if _mc.any():
                    ax_x.plot(t_arr[_mc], q_arr[_mc], color=color_orange,
                              lw=0.8, label="Chatter")
        else:
            if _split_t is not None:
                mask_s = t_arr < _split_t
                mask_c = t_arr >= _split_t
            else:
                mask_s = np.ones(len(t_arr), dtype=bool)
                mask_c = np.zeros(len(t_arr), dtype=bool)
            if mask_s.any():
                ax_x.plot(t_arr[mask_s], q_arr[mask_s], color=color_azul,
                          lw=0.8, label="Stable")
            if mask_c.any():
                ax_x.plot(t_arr[mask_c], q_arr[mask_c], color=color_orange,
                          lw=0.8, label="Chatter")
        ax_x.set_ylabel("Displacement [m]")
        ax_x.legend()
        _draw_vlines(ax_x, auto_vlines)
        if _all_stable_ranges:
            for _bi, (_t0, _t1) in enumerate(_all_stable_ranges):
                _m = (t_arr >= _t0) & (t_arr <= _t1)
                if _m.any():
                    ax_v.plot(t_arr[_m], v_arr[_m], color=color_azul,
                              lw=0.8, label="Stable" if _bi == 0 else "_nolegend_")
            if _split_t is not None:
                _mc = t_arr >= _split_t
                if _mc.any():
                    ax_v.plot(t_arr[_mc], v_arr[_mc], color=color_orange,
                              lw=0.8, label="Chatter")
        else:
            _v_split_t = t_gt if t_gt is not None else t_d
            if _v_split_t is not None:
                _ms = t_arr < _v_split_t
                _mc = t_arr >= _v_split_t
            else:
                _ms = np.ones(len(t_arr), dtype=bool)
                _mc = np.zeros(len(t_arr), dtype=bool)
            if _ms.any():
                ax_v.plot(t_arr[_ms], v_arr[_ms], color=color_azul,
                          lw=0.8, label="Stable")
            if _mc.any():
                ax_v.plot(t_arr[_mc], v_arr[_mc], color=color_orange,
                          lw=0.8, label="Chatter")
        ax_v.set_ylabel("Velocity [m/s]")
        ax_v.set_xlabel("Time [s]")
        ax_v.legend()
        _draw_vlines(ax_v, auto_vlines)
        # constrained_layout handles spacing for 2-subplot figure

    # ── C2: Areas per window ──────────────────────────────────────────────
    fig_c2, ax_c2 = plt.subplots(figsize=fig_size(scale=3.0))
    ax_c2.set_title(f"C2 — Areas per Window — {name}")
    ax_c2.set_xlabel("Time [s]")
    ax_c2.set_ylabel("Shoelace area [m·m/s]")
    valid = np.isfinite(areas)
    if valid.any():
        # ax_c2.semilogy(t_wins[valid], areas[valid], color=color_azul,
        #                lw=1.0, marker="o", markersize=2, label="$A_k$")
        ax_c2.plot(t_wins[valid], areas[valid], color=color_azul,
                   lw=1.0, marker="o", markersize=2, label="$A_k$")
        ax_c2.set_yscale("log")
    if thr and explicit_training_provided and area_threshold_enabled:
        z_lbl   = f"{thr['z']:.0f}"
        y_upper = 10 ** thr["upper"]
        y_lower = 10 ** thr["lower"]
        y_mu    = 10 ** thr["mu"]
        # y_upper = thr["upper"]
        # y_lower = thr["lower"]
        # y_mu    = thr["mu"]
        ax_c2.axhline(y_upper, color=color_red, ls="--", lw=1.4)
        _add_hline_label(ax_c2, y_upper, rf"$\mu+{z_lbl}\sigma={thr['upper']:.3g}$",
                         color=color_red, ha='right', va='bottom', fontsize=16)
        ax_c2.axhline(y_lower, color=color_red, ls=":", lw=1.2)
        _add_hline_label(ax_c2, y_lower, rf"$\mu-{z_lbl}\sigma={thr['lower']:.3g}$",
                         color=color_red, ha='right', va='top', fontsize=16)
        ax_c2.axhline(y_mu, color=color_verde, ls="-", lw=1.0)
        _add_hline_label(ax_c2, y_mu, rf"$\mu={thr['mu']:.3g}$",
                         color=color_verde, ha='right', va='bottom', fontsize=16)
    _draw_vlines(ax_c2, auto_vlines)
    ax_c2.legend()

    # ── C2-b: Trajectory per window ──────────────────────────────────────────────
    fig_c2b, ax_c2b = plt.subplots(figsize=fig_size(scale=3.0))
    ax_c2b.set_title(f"C2-b — Trajectory per Window — {name}")
    ax_c2b.set_xlabel("Time [s]")
    ax_c2b.set_ylabel("Shoelace area [m·m/s]")
    valid = np.isfinite(trayectory_C)
    trayectory_K = abs(trayectory_K)

    area_c_k = trayectory_C + trayectory_K
    if valid.any():
        # ax_c2.semilogy(t_wins[valid], areas[valid], color=color_azul,
        #                lw=1.0, marker="o", markersize=2, label="$A_k$")
        ax_c2b.plot(t_wins[valid], trayectory_C[valid], color=color_azul,
                   lw=1.0, marker="o", markersize=2, label="$C_k$")
        ax_c2b.plot(t_wins[valid], trayectory_K[valid], color=color_orange,
                   lw=1.0, marker="o", markersize=2, label="$K_k$")
        ax_c2b.plot(t_wins[valid], areas[valid], color=color_verde,
                   lw=1.0, marker="o", markersize=2, label="$A_k$")
        ax_c2b.plot(t_wins[valid], area_c_k[valid], color=color_purple,
                   lw=1.0, marker="o", markersize=2, label="$C_k+K_k$") 
        ax_c2b.set_yscale("linear")
    if thr and explicit_training_provided and area_threshold_enabled:
        z_lbl   = f"{thr['z']:.0f}"
        y_upper = 10 ** thr["upper"]
        y_lower = 10 ** thr["lower"]
        y_mu    = 10 ** thr["mu"]
        # y_upper = thr["upper"]
        # y_lower = thr["lower"]
        # y_mu    = thr["mu"]
        ax_c2b.axhline(y_upper, color=color_red, ls="--", lw=1.4)
        _add_hline_label(ax_c2b, y_upper, rf"$\mu+{z_lbl}\sigma={thr['upper']:.3g}$",
                         color=color_red, ha='right', va='bottom', fontsize=16)
        ax_c2b.axhline(y_lower, color=color_red, ls=":", lw=1.2)
        _add_hline_label(ax_c2b, y_lower, rf"$\mu-{z_lbl}\sigma={thr['lower']:.3g}$",
                         color=color_red, ha='right', va='top', fontsize=16)
        ax_c2b.axhline(y_mu, color=color_verde, ls="-", lw=1.0)
        _add_hline_label(ax_c2b, y_mu, rf"$\mu={thr['mu']:.3g}$",
                         color=color_verde, ha='right', va='bottom', fontsize=16)
    _draw_vlines(ax_c2b, auto_vlines)
    ax_c2b.legend()


    # ── C3: Lyapunov exponent σ̂ ──────────────────────────────────────────
    fig_c3, ax_c3 = plt.subplots(figsize=fig_size(scale=3.0), layout='tight')
    ax_c3.set_title(rf"C3 — Lyapunov $\hat{{\sigma}}$(t) — {name}")
    ax_c3.set_xlabel("Time [s]")
    ax_c3.set_ylabel(r"$\hat{\sigma}$ [1/s]")
    valid_s = np.isfinite(sigma)
    if valid_s.any():
        ax_c3.plot(t_wins[valid_s], sigma[valid_s], color=color_azul,
                   lw=0.8, alpha=0.7, marker=".", markersize=3,
                   label=r"$\hat{\sigma}$ raw")
    ewma_differs = (valid_s.any() and not np.allclose(
        sigma[valid_s], s_ewma[valid_s], equal_nan=True))
    if ewma_differs:
        valid_e = np.isfinite(s_ewma)
        if valid_e.any():
            ax_c3.plot(t_wins[valid_e], s_ewma[valid_e], color=color_orange,
                       lw=1.5, label=r"$\hat{\sigma}$ EWMA")
    ax_c3.axhline(0, color="black", lw=0.8, ls="--",
                  label=r"$\hat{\sigma}=0$")
    _draw_vlines(ax_c3, auto_vlines)
    ax_c3.legend()

    # ── C4: Histogram of stable areas ─────────────────────────────────────
    N_wins = len(t_wins)
    if _all_stable_ranges:
        stable_mask = np.zeros(N_wins, dtype=bool)
        for _t0, _t1 in _all_stable_ranges:
            stable_mask |= (t_wins >= _t0) & (t_wins <= _t1)
    else:
        _stable_split = t_gt if t_gt is not None else t_d
        if _stable_split is not None:
            stable_mask = t_wins < _stable_split
        else:
            stable_mask = np.zeros(N_wins, dtype=bool)
            stable_mask[:max(3, int(0.30 * N_wins))] = True
    stable_areas = areas[stable_mask]
    valid_sa = np.isfinite(stable_areas) & (stable_areas > 0)
    # Only show stable-area histogram / Gaussian fit if the user provided
    # explicit training intervals (otherwise we assume no training-based
    # thresholding / annotation is desired).
    if valid_sa.sum() >= 5 and explicit_training_provided and area_threshold_enabled:
        log10_a = np.log10(stable_areas[valid_sa])
        fig_c4, ax_c4 = plt.subplots(figsize=fig_size(scale=3.0), layout='tight')
        ax_c4.set_title(f"C4 — Stable Area Distribution — {name}")
        ax_c4.set_xlabel(r"$\log_{10}(A_k)$")
        ax_c4.set_ylabel("Density")
        ax_c4.hist(log10_a, bins=40, density=True, alpha=0.55,
                   color=color_azul, label=f"Stable (n={int(valid_sa.sum())})")
        mu_h  = float(np.mean(log10_a))
        std_h = float(np.std(log10_a))
        if std_h > 0:
            xs = np.linspace(mu_h - 4 * std_h, mu_h + 4 * std_h, 300)
            ax_c4.plot(xs, _scipy_norm.pdf(xs, mu_h, std_h),
                       color=color_azul, lw=1.8, ls="-")
            ax_c4.axvline(mu_h, color=color_verde, ls="-", lw=1.4)
            _add_vline_label(ax_c4, mu_h, rf"$\mu={mu_h:.3g}$", fontsize=14, color=color_verde)
            if thr and thr.get("upper", 0) > 0 and thr.get("lower", 0) > 0:
                lo3   = np.log10(float(thr["lower"]))
                hi3   = np.log10(float(thr["upper"]))
                z_lbl = f"{thr['z']:.0f}"
            else:
                lo3, hi3, z_lbl = mu_h - 3 * std_h, mu_h + 3 * std_h, "3"
            ax_c4.axvline(hi3, color=color_red, ls="--", lw=1.4)
            _add_vline_label(ax_c4, hi3, rf"$\mu+{z_lbl}\sigma={hi3:.3g}$", fontsize=14, color=color_red)
            ax_c4.axvline(lo3, color=color_red, ls=":", lw=1.2)
            _add_vline_label(ax_c4, lo3, rf"$\mu-{z_lbl}\sigma={lo3:.3g}$", fontsize=14, color=color_red)
        ax_c4.legend()
        ax_c4.grid(False)

        # times matching the log10_a array (needed for per-label D1b / D4b)
        _t_stable = t_wins[stable_mask][valid_sa]

        # ── D1: Time series — stable log₁₀(A)  [MaxEnt F1 analog] ────────
        fig_d1, ax_d1 = plt.subplots(figsize=fig_size(scale=3.0), layout='tight',
                                     num=f"D1 — Stable Areas — {name}")
        ax_d1.set_title(f"D1 — Stable Areas — {name}")
        ax_d1.set_xlabel("Time [s]")
        ax_d1.set_ylabel(r"$\log_{10}(A_k)$")
        if _all_stable_ranges:
            for _bi, (_t0, _t1) in enumerate(_all_stable_ranges):
                _m = (t_wins >= _t0) & (t_wins <= _t1) & np.isfinite(areas) & (areas > 0)
                if _m.any():
                    ax_d1.plot(
                        t_wins[_m], np.log10(areas[_m]),
                        color=color_azul, lw=1.0, marker="o", markersize=2,
                        label="Stable" if _bi == 0 else "_nolegend_",
                    )
        else:
            ax_d1.plot(_t_stable, log10_a, color=color_azul,
                       lw=1.0, marker="o", markersize=2, label="Stable")
        _draw_vlines(ax_d1, auto_vlines)
        ax_d1.legend()

        # ── D1b / D4b: per-label (only when ≥2 distinct stable labels) ────
        _stable_label_groups: dict = {}
        for _t0, _t1, _lbl in (training_intervals or []):
            if str(_lbl).startswith("stable"):
                _stable_label_groups.setdefault(str(_lbl), []).append((_t0, _t1))

        # D1b — one figure per stable label  [MaxEnt F1b analog]
        if len(_stable_label_groups) >= 2:
            for _gi, (_lbl_name, _ranges) in enumerate(_stable_label_groups.items()):
                fig_d1b, ax_d1b = plt.subplots(
                    figsize=fig_size(scale=3.0), layout='tight',
                    num=f"D1b.{_gi} — {_lbl_name}",
                )
                ax_d1b.set_title(rf"D1b — Stable Areas | {_lbl_name} — {name}")
                ax_d1b.set_xlabel("Time [s]")
                ax_d1b.set_ylabel(r"$\log_{10}(A_k)$")
                # background: all stable intervals, faded (per interval, no connecting lines)
                for _bi, (_t0b, _t1b) in enumerate(_all_stable_ranges):
                    _mb = (t_wins >= _t0b) & (t_wins <= _t1b) & np.isfinite(areas) & (areas > 0)
                    if _mb.any():
                        ax_d1b.plot(
                            t_wins[_mb], np.log10(areas[_mb]),
                            color=color_azul, alpha=0.5, lw=0.8,
                            marker="o", markersize=1,
                            label="All stable" if _bi == 0 else "_nolegend_",
                        )
                # highlighted: specific label intervals, per interval
                _ranges_str = ", ".join(rf"$[{r0:.2f},\,{r1:.2f}]$" for r0, r1 in _ranges)
                for _ri, (_t0, _t1) in enumerate(_ranges):
                    _m = (t_wins >= _t0) & (t_wins <= _t1) & np.isfinite(areas) & (areas > 0)
                    if _m.any():
                        ax_d1b.plot(
                            t_wins[_m], np.log10(areas[_m]),
                            color=color_azul, alpha=1.0, lw=1.6,
                            marker="o", markersize=2,
                            label=rf"{_lbl_name}  {_ranges_str} s" if _ri == 0 else "_nolegend_",
                        )
                if std_h > 0:
                    _hi3_d1b = mu_h + 3 * std_h
                    ax_d1b.axhline(_hi3_d1b, color=color_red, ls="--", lw=1.4)
                    _add_hline_label(ax_d1b, _hi3_d1b, rf"$\mu+3\sigma={_hi3_d1b:.3g}$",
                                     color=color_red, ha='right', va='bottom', fontsize=14)
                ax_d1b.legend()

        # D4b — one figure per stable label histogram  [MaxEnt F4b analog]
        if len(_stable_label_groups) >= 2 and std_h > 0:
            _n_bins_d4b   = 40
            _counts_all_d, _bin_edges_d = np.histogram(log10_a, bins=_n_bins_d4b)
            _widths_d      = np.diff(_bin_edges_d)
            _heights_all_d = _counts_all_d / (len(log10_a) * _widths_d)
            for _gi, (_lbl_name, _ranges) in enumerate(_stable_label_groups.items()):
                fig_d4b, ax_d4b = plt.subplots(
                    figsize=fig_size(scale=3.0), layout='tight',
                    num=f"D4b.{_gi} — {_lbl_name}",
                )
                ax_d4b.set_title(rf"D4b — Area PDF | {_lbl_name} — {name}")
                ax_d4b.set_xlabel(r"$\log_{10}(A_k)$")
                ax_d4b.set_ylabel("Density")
                # full stable histogram (light)
                ax_d4b.bar(
                    _bin_edges_d[:-1], _heights_all_d, width=_widths_d,
                    color=color_azul, alpha=0.35, align="edge",
                    label=f"All stable",
                )
                # highlighted segment (same normalisation denominator)
                _mask_seg = np.zeros(len(log10_a), dtype=bool)
                for _t0, _t1 in _ranges:
                    _mask_seg |= (_t_stable >= _t0) & (_t_stable <= _t1)
                if _mask_seg.sum() >= 2:
                    _counts_seg, _ = np.histogram(log10_a[_mask_seg], bins=_bin_edges_d)
                    _heights_seg   = _counts_seg / (len(log10_a) * _widths_d)
                    _ranges_str    = ", ".join(rf"$[{r0:.2f},\,{r1:.2f}]$" for r0, r1 in _ranges)
                    ax_d4b.bar(
                        _bin_edges_d[:-1], _heights_seg, width=_widths_d,
                        color=color_azul, alpha=0.72, align="edge",
                        label=rf"{_lbl_name}  {_ranges_str} s",
                    )
                # Gaussian PDF + μ and σ reference lines
                _xs_d = np.linspace(mu_h - 4.5 * std_h, mu_h + 4.5 * std_h, 300)
                ax_d4b.plot(_xs_d, _scipy_norm.pdf(_xs_d, mu_h, std_h),
                            color=color_verde, lw=1.8,
                            label=rf"PDF  $\mu$={mu_h:.3g}, $\sigma$={std_h:.3g}")
                ax_d4b.axvline(mu_h, color=color_verde, ls="-", lw=1.4)
                _add_vline_label(ax_d4b, mu_h, rf"$\mu={mu_h:.3g}$", fontsize=14, color=color_verde)
                ax_d4b.axvline(mu_h + 3 * std_h, color=color_red, ls="--", lw=1.4)
                _add_vline_label(ax_d4b, mu_h + 3 * std_h, rf"$\mu+3\sigma={mu_h + 3 * std_h:.3g}$",
                                 fontsize=14, color=color_red)
                ax_d4b.axvline(mu_h - 3 * std_h, color=color_red, ls=":", lw=1.2)
                _add_vline_label(ax_d4b, mu_h - 3 * std_h, rf"$\mu-3\sigma={mu_h - 3 * std_h:.3g}$",
                                 fontsize=14, color=color_red)
                ax_d4b.legend()
                ax_d4b.grid(False)

    # ── Ĝ accumulator (optional) ──────────────────────────────────────────
    G = np.asarray(result.G_hat)
    if G.size > 0:
        fig_g, ax_g = plt.subplots(figsize=fig_size(scale=3.0), layout='tight')
        ax_g.set_title(rf"$\hat{{G}}$ Accumulator — {name}")
        ax_g.set_xlabel("Time [s]")
        ax_g.set_ylabel(r"$\hat{G}$ [m·m/s · s]")
        ax_g.plot(t_wins[:len(G)], G, color=color_orange, lw=1.5,
                  label=r"$\hat{G}(t)$")
        ax_g.axhline(0, color="black", lw=0.8, ls="--",
                     label=r"$\hat{G}=0$")
        ax_g.fill_between(t_wins[:len(G)], G, 0,
                          where=(G > 0),  alpha=0.15, color=color_red,
                          label="chatter")
        ax_g.fill_between(t_wins[:len(G)], G, 0,
                          where=(G <= 0), alpha=0.10, color=color_verde,
                          label="stable")
        _draw_vlines(ax_g, auto_vlines)
        ax_g.legend()

    # ── Ĝ sliding window (optional) ───────────────────────────────────────
    Gs = np.asarray(result.G_hat_sliding)
    if Gs.size > 0:
        fig_gs, ax_gs = plt.subplots(figsize=fig_size(scale=3.0), layout='tight')
        ax_gs.set_title(rf"$\hat{{G}}$ Sliding Window — {name}")
        ax_gs.set_xlabel("Time [s]")
        ax_gs.set_ylabel(r"$\hat{G}_{slide}$ [m·m/s · s]")
        ax_gs.plot(t_wins[:len(Gs)], Gs, color=color_purple, lw=1.5,
                   label=r"$\hat{G}_{slide}(t)$")
        ax_gs.axhline(0, color="black", lw=0.8, ls="--",
                      label=r"$\hat{G}_{slide}=0$")
        ax_gs.fill_between(t_wins[:len(Gs)], Gs, 0,
                           where=(Gs > 0),  alpha=0.15, color=color_red,
                           label="chatter")
        ax_gs.fill_between(t_wins[:len(Gs)], Gs, 0,
                           where=(Gs <= 0), alpha=0.10, color=color_verde,
                           label="stable")
        _draw_vlines(ax_gs, auto_vlines)
        ax_gs.legend()

    if show:
        plt.show(block=True)


# ---------------------------------------------------------------------------
# Signal diagnostics plots
# ---------------------------------------------------------------------------

def plots_signal_diagnostics(
    signal: SignalData,
    result: FixedWindowResult,
    stable_range: Tuple[float, float] = (0.5, 4.0),
    zoom_range: Tuple[float, float] = (1.0, 1.2),
    eq_smooth_s: float = 0.050,
    show: bool = True,
) -> None:
    """Diagnostic plots for understanding signal structure and area variability.

    Figures produced
    ----------------
    **Fig A — Frequency content & area autocorrelation**
        A1. FFT of *x* in the stable zone (``stable_range``).
        A2. Autocorrelation of ln(areas) in the stable zone — reveals
            periodic modulation (beat, tooth-pass, etc.).

    **Fig B — Quasi-static equilibrium & dynamic decomposition**
        B1. Full *x* signal with the quasi-static equilibrium
            ``x_eq ≈ EWMA_slow(x)`` overlaid.
        B2. Dynamic component ``x_dyn = x − x_eq``.
        B3. Phase portrait (orbit) of the centered signal
            ``(x_dyn, v_dyn)`` coloured by time.

    **Fig C — Phase portrait evolution (stable → chatter)**
        Three orbit snapshots: one in the stable zone, one at the
        transition, one in full chatter — using the *centered* orbit.

    Parameters
    ----------
    signal        : original input signal.
    result        : output of :func:`run_fixed_window`.
    stable_range  : ``(t_start, t_end)`` [s] defining the "stable" zone
                    used for the FFT and autocorrelation.
    zoom_range    : ``(t_start, t_end)`` [s] for the signal zoom (B1/B2).
                    Should be a short window (≤ 0.5 s).
    eq_smooth_s   : half-width [s] of the moving-average used to estimate
                    ``x_eq``.  Default 50 ms — slow enough to follow AP
                    drift but fast enough to not absorb dynamic vibration.
    show          : call ``plt.show()`` when done.
    """
    name = signal.name or ""
    t    = np.asarray(signal.t)
    x    = np.asarray(signal.displacement)
    v    = np.asarray(signal.velocity)
    dt   = float(t[1] - t[0])
    fs   = 1.0 / dt

    t_wins = np.asarray(result.t_wins)
    areas  = np.asarray(result.areas)

    # ── helpers ────────────────────────────────────────────────────────────
    def _mask(t_arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
        return (t_arr >= lo) & (t_arr <= hi)

    def _moving_avg(arr: np.ndarray, half_n: int) -> np.ndarray:
        """Causal moving average of half-width *half_n* samples."""
        n = 2 * half_n + 1
        kernel = np.ones(n) / n
        padded = np.pad(arr, (n - 1, 0), mode="edge")
        return np.convolve(padded, kernel, mode="valid")[: len(arr)]

    # quasi-static equilibrium estimate (slow moving average of x)
    half_n_eq = max(1, int(round(eq_smooth_s / dt)))
    x_eq  = _moving_avg(x, half_n_eq)
    x_dyn = x - x_eq
    v_eq  = _moving_avg(v, half_n_eq)
    v_dyn = v - v_eq

    # ── Fig A — FFT + autocorrelation ─────────────────────────────────────
    figA, (aA1, aA2) = plt.subplots(2, 1, figsize=(12, 7))
    figA.suptitle(f"Signal diagnostics — frequency content & area statistics — {name}")

    # A1: FFT of x in stable zone
    ms = _mask(t, *stable_range)
    x_stab = x[ms]
    N_fft  = len(x_stab)
    if N_fft > 0:
        fft_mag = np.abs(np.fft.rfft(x_stab))
        freqs   = np.fft.rfftfreq(N_fft, 1.0 / fs)
        mask_f  = freqs < min(fs / 2, 800.0)
        aA1.semilogy(freqs[mask_f], fft_mag[mask_f],
                     color="steelblue", lw=0.8, label="FFT |X(f)|")
        for fmark, col, lbl in [
            (150.0, "red",    "f_modal 150 Hz"),
            (200.0, "green",  "f_tool  200 Hz"),
            (50.0,  "purple", "f_beat   50 Hz"),
        ]:
            if fmark < freqs[-1]:
                aA1.axvline(fmark, color=col, lw=1.5, ls="--", label=lbl)
    aA1.set_xlabel("Frequency [Hz]")
    aA1.set_ylabel("|FFT(x)|")
    aA1.set_title(f"FFT of x in stable zone t=[{stable_range[0]:.1f}, {stable_range[1]:.1f}] s")
    aA1.legend(fontsize=9)
    aA1.grid(True, alpha=0.3)

    # A2: autocorrelation of ln(areas) in stable zone
    mw = _mask(t_wins, *stable_range)
    A_stab = areas[mw]
    with np.errstate(divide="ignore", invalid="ignore"):
        logA = np.where(A_stab > 1e-50, np.log(A_stab), np.nan)
    ok = np.isfinite(logA)
    if ok.sum() > 10:
        la = logA[ok] - np.nanmean(logA[ok])
        acorr = np.correlate(la, la, mode="full")
        acorr = acorr[len(acorr) // 2:]
        acorr /= acorr[0]
        lags_ms = np.arange(len(acorr)) * (t_wins[1] - t_wins[0]) * 1000.0
        n_show  = min(len(lags_ms), 200)
        aA2.plot(lags_ms[:n_show], acorr[:n_show],
                 color="darkorange", lw=1.2, label="autocorr ln(A)")
        aA2.axvline(20.0, color="purple", lw=1.2, ls="--",
                    label="T_beat=20 ms (expected)")
        aA2.axhline(0, color="black", lw=0.5)
    aA2.set_xlabel("Lag [ms]")
    aA2.set_ylabel("Autocorrelation")
    aA2.set_title("Autocorrelation of ln(areas) in stable zone")
    aA2.legend(fontsize=9)
    aA2.grid(True, alpha=0.3)
    figA.tight_layout()

    # ── Fig B — Equilibrium decomposition ─────────────────────────────────
    figB, (aB1, aB2, aB3) = plt.subplots(3, 1, figsize=(13, 10))
    figB.suptitle(f"Quasi-static equilibrium & dynamic decomposition — {name}")

    mz = _mask(t, *zoom_range)
    tz = t[mz]; xz = x[mz]; xeqz = x_eq[mz]; xdynz = x_dyn[mz]
    vz = v[mz]; vdynz = v_dyn[mz]

    # B1: x with x_eq overlay
    aB1.plot(tz * 1000, xz,    color="steelblue", lw=0.9, label="x  (absoluto)")
    aB1.plot(tz * 1000, xeqz,  color="red",       lw=2.0, ls="--",
             label=f"x_eq ≈ moving avg ({eq_smooth_s*1000:.0f} ms)")
    aB1.set_ylabel("x [m]")
    aB1.set_title(f"Señal x y equilibrio cuasi-estático — zoom t=[{zoom_range[0]:.2f}, {zoom_range[1]:.2f}] s")
    aB1.legend(fontsize=9)
    aB1.grid(True, alpha=0.3)

    # B2: dynamic residual
    aB2.plot(tz * 1000, xdynz, color="forestgreen", lw=0.9, label="x_dyn = x − x_eq")
    aB2.axhline(0, color="black", lw=0.5)
    aB2.set_ylabel("x_dyn [m]")
    aB2.set_title("Componente dinámica (vibración alrededor del equilibrio)")
    aB2.legend(fontsize=9)
    aB2.grid(True, alpha=0.3)

    # B3: orbit of centered signal
    sc = aB3.scatter(xdynz, vdynz,
                     c=tz, cmap="viridis", s=4, alpha=0.8,
                     label="órbita centrada (x_dyn, v_dyn)")
    plt.colorbar(sc, ax=aB3, label="time [s]")
    aB3.axhline(0, color="black", lw=0.3)
    aB3.axvline(0, color="black", lw=0.3)
    aB3.set_xlabel("x_dyn [m]")
    aB3.set_ylabel("v_dyn [m/s]")
    aB3.set_title("Diagrama de fase (centrado) — zona zoom")
    aB3.legend(fontsize=9)
    aB3.grid(True, alpha=0.2)
    figB.tight_layout()

    # ── Fig C — Phase portrait snapshots: stable / transition / chatter ────
    t_total = float(t[-1] - t[0])
    t0      = float(t[0])

    # pick three snapshot times: 20 % (stable), 55 % (transition), 85 % (chatter)
    snap_fracs  = [0.20, 0.55, 0.85]
    snap_labels = ["Estable (20%)", "Transición (55%)", "Chatter (85%)"]
    snap_colors = ["steelblue", "darkorange", "crimson"]
    snap_dur    = min(0.05, t_total * 0.03)   # 50 ms per snapshot

    figC, axes_c = plt.subplots(1, 3, figsize=(13, 5))
    figC.suptitle(f"Diagrama de fase (centrado) — estable / transición / chatter — {name}")

    for ax_c, frac, lbl, col in zip(axes_c, snap_fracs, snap_labels, snap_colors):
        t_snap_lo = t0 + frac * t_total
        t_snap_hi = t_snap_lo + snap_dur
        ms2 = _mask(t, t_snap_lo, t_snap_hi)
        if ms2.sum() < 5:
            ax_c.set_title(f"{lbl}\n(no data)")
            continue
        ax_c.plot(x_dyn[ms2], v_dyn[ms2], color=col, lw=0.8, alpha=0.9)
        ax_c.scatter(x_dyn[ms2][[0]], v_dyn[ms2][[0]], color="black", s=30, zorder=5)
        ax_c.axhline(0, color="black", lw=0.3)
        ax_c.axvline(0, color="black", lw=0.3)
        ax_c.set_xlabel("x_dyn [m]")
        ax_c.set_ylabel("v_dyn [m/s]")
        ax_c.set_title(f"{lbl}\nt=[{t_snap_lo:.1f}, {t_snap_hi:.2f}] s")
        ax_c.grid(True, alpha=0.2)
    figC.tight_layout()

    if show:
        plt.show()

