"""Main entry point for green_integral visualization."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
_MPL_PLT = plt  # unshadowed alias -- see each dispatcher's per-call `plt` shadow
from scipy.stats import norm as _scipy_norm

from ..utils.types import SignalData, GreenIntegralResult, LyapunovResult
from .plots import (
    plot_windows_local,
    plot_windows_duration,
    plot_indicator_local,
    plot_training_distribution,
)
from .colors import COLORS
from .plot_style import (
    FIGSIZE_SIMPLE, FIGSIZE_WIDE, SCALE,
    apply_sci_yaxis, figsize_from_scale, figsize_grid,
)
from .plot_style import apply as _apply_style

_apply_style()   # article-plot-style, at the default scale (was an import-time side effect of plot_style.py)

color_red    = COLORS["threshold"]
color_orange = COLORS["chatter"]
color_purple = COLORS["secondary"]
color_verde  = COLORS["mean"]
color_azul   = COLORS["stable"]

# Most panels in plots_lyapunov (C2-C3, D1b/D4b, Ĝ, Ĝs) and the training
# distribution pair share this one size — required by
# test_plot_reliability.py's check #6 (Training figures must match the
# sibling C-panels' size). C1 and the C6/C7 per-case grids are sized off
# FIGSIZE_SIMPLE instead (single/grid panels, not this WIDE composite).
_LYAP_FIGSIZE = figsize_from_scale(FIGSIZE_WIDE, SCALE)


def _diag_figsize(
    nrows: int = 1, ncols: int = 1, scale: float = SCALE,
    base_wide: Tuple[float, float] = FIGSIZE_WIDE,
) -> Tuple[float, float]:
    """Compose a figsize for plots_signal_diagnostics' multi-panel figures.

    ncols == 1 (Figs A/B: a vertical stack of full-width time-series panels)
    keeps each row at the full WIDE preset width via ``figsize_grid`` —
    these panels genuinely need FIGSIZE_WIDE's aspect (long time/frequency
    axes), so this is the documented exception to the SIMPLE/WIDE-preset
    rule rather than an ad-hoc tuple.
    ncols > 1 (Fig C: side-by-side phase-portrait snapshots) tiles the
    near-square SIMPLE preset per panel instead, matching a phase portrait's
    natural aspect ratio.
    """
    if ncols == 1:
        return figsize_grid(1, nrows, base=figsize_from_scale(base_wide, scale))
    return figsize_grid(ncols, nrows, base=figsize_from_scale(FIGSIZE_SIMPLE, scale))


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


def _plot_pieces_grid(
    pieces: Sequence[Tuple[np.ndarray, np.ndarray]],
    color: str, title: str,
    ylabel: str = r"$v(t)$",
    hlines: Optional[Sequence[Tuple[float, str, str]]] = None,
    yscale: Optional[str] = None,
    figsize_simple: Tuple[float, float] = FIGSIZE_SIMPLE,
    scale: float = 1.0,
    fig_label: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Per-case grid: one square subplot per training piece (falls back to a
    single panel when there's only one piece) -- same per-case-grid idea as
    MaxEnt's F0a/F1 and SST's C6/C7."""
    pieces = [
        (np.asarray(t, dtype=float), np.asarray(v, dtype=float))
        for t, v in pieces if t is not None and len(t) > 0
    ]
    if not pieces:
        pieces = [(np.array([]), np.array([]))]

    def _fill(ax, t_i, v_i):
        if t_i.size > 0:
            ax.plot(t_i, v_i, color=color, alpha=0.9, lw=0.8, marker="o", markersize=2)
        if yscale:
            ax.set_yscale(yscale)
        for hy, hcol, hls in (hlines or []):
            ax.axhline(hy, color=hcol, ls=hls, lw=1.0)

    if len(pieces) <= 1:
        fig, ax = plt.subplots(figsize=figsize_from_scale(figsize_simple, scale), num=fig_label)
        _fill(ax, *pieces[0])
        ax.set_title(title)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        fig.tight_layout()  # one-shot, not layout='tight' — see C1 for why
        return fig, np.array([[ax]])

    n = len(pieces)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    cell = figsize_simple[1]
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell * ncols * scale, cell * nrows * scale),
                              num=fig_label, squeeze=False)
    for i, (t_i, v_i) in enumerate(pieces):
        ax = axes[i // ncols, i % ncols]
        _fill(ax, t_i, v_i)
        ax.set_title(f"Piece {i + 1}", fontsize=10)
        ax.set_box_aspect(1)
    for j in range(n, nrows * ncols):
        fig.delaxes(axes[j // ncols, j % ncols])
    fig.suptitle(title, y=1.02)
    fig.supxlabel("Time [s]")
    fig.supylabel(ylabel)
    fig.tight_layout()  # one-shot, not layout='tight' — see C1 for why
    return fig, axes


def plots_green_integral(
    signal: SignalData,
    result: GreenIntegralResult,
    show: bool = True,
    scale: float = SCALE,
    figsize_wide: Tuple[float, float] = FIGSIZE_WIDE,
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
    scale  : float, default ``plot_style.SCALE``
        article-plot-style figure-size multiplier for this call, independent
        of the shared ``plot_style.SCALE`` default (callers, e.g. the example
        script, can expose their own editable variable and pass it here).
    figsize_wide : tuple, default ``plot_style.FIGSIZE_WIDE``
        Base (width, height) preset this call scales from, independent of
        the shared ``plot_style.FIGSIZE_WIDE`` default.
    **kwargs
        Forwarded to individual plot functions (currently unused).
    """
    name = signal.name if signal.name else ""
    local_figsize = figsize_from_scale(figsize_wide, scale)

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

    plot_windows_local(result_dict, name=name, figsize=local_figsize)
    plot_windows_duration(result_dict, name=name, figsize=local_figsize)
    plot_indicator_local(result_dict, name=name, figsize=local_figsize)
    # Histogram + verification curve for the mu +- z*sigma training
    # population (internal training_intervals or external reference_signal —
    # see COMMON_TEMPLATE.md Fase 3).
    plot_training_distribution(result_dict.get("global_data", {}), name=name, log_transform=False,
                                figsize=local_figsize)

    if show:
        plt.show(block=True)


# ---------------------------------------------------------------------------
# Lyapunov indicator plots
# ---------------------------------------------------------------------------

def plots_lyapunov(
    signal: SignalData,
    result: LyapunovResult,
    training_intervals=None,
    reference_signal: Optional[Sequence[SignalData]] = None,
    show: bool = True,
    scale: float = SCALE,
    grid_scale: Optional[float] = None,
    figsize_simple: Tuple[float, float] = FIGSIZE_SIMPLE,
    figsize_wide: Tuple[float, float] = FIGSIZE_WIDE,
) -> None:
    """Produce the standard set of plots for a :class:`LyapunovResult`.

    Figures produced
    ----------------
    C1. **Signal** — velocity, single panel, single color, with the
        first-detection vline.
    C2. **Areas** — shoelace area per window on a log scale.
    C3. **Lyapunov** — raw σ̂ and (if available) σ̂_EWMA.
    Training population — histogram + Gaussian fit, and a verification curve
        of the exact windows that trained mu +- z*sigma (see
        ``plots.plot_training_distribution``); sourced from
        ``global_data["training_areas"]``, so it always matches whichever
        ``training_source`` ("internal" training_intervals or
        "external_reference" reference_signal) actually trained the threshold.
    C6/C7. Per-case grid — one square subplot per training piece/case (raw
        signal and decision variable/area, respectively). Stable only.
    D1b/D4b. Per-label breakdown — only when training_source == "internal"
        and >=2 distinct "stable*" labels were passed via training_intervals.
    Ĝ  **Accumulator** — only when ``result.G_hat`` is non-empty.
    Ĝs **Sliding** — only when ``result.G_hat_sliding`` is non-empty.

    Parameters
    ----------
    signal             : original input signal.
    result             : output of :func:`run_lyapunov`.
    training_intervals : list of ``(t0, t1, label)`` tuples. Entries whose
                         label starts with ``"stable"`` define the stable
                         training region. Falls back to ``global_data``
                         when not provided.
    show               : call ``plt.show(block=True)`` after all figures.
    scale              : article-plot-style figure-size multiplier for this
                         call (shadows the module-level ``_LYAP_FIGSIZE``
                         default with ``figsize_from_scale(FIGSIZE_WIDE, scale)``
                         for the duration of this call).
    """
    # Shadows the module-level _LYAP_FIGSIZE for this call only -- every use
    # of that name below resolves to this local (Python scoping), so a
    # caller-supplied `scale`/`figsize_wide` reaches every panel without
    # touching each site.
    _LYAP_FIGSIZE = figsize_from_scale(figsize_wide, scale)
    if grid_scale is None:
        grid_scale = scale
    if isinstance(reference_signal, SignalData):
        reference_signal = [reference_signal]
    name   = signal.name or "signal"
    # Every title in this function already embeds `name` -- but num= (window
    # identity) is still a fixed literal ("C6 — ...", etc.), so calling this
    # dispatcher again for a DIFFERENT case in the same process would
    # silently reuse/overwrite this case's windows instead of opening new
    # ones. Shadow plt.subplots to qualify every num= with `name` too.
    _real_plt = _MPL_PLT
    class _PltShadow:
        @staticmethod
        def subplots(*_args, num=None, **_kwargs):
            if num is not None:
                num = f"{num} — {name}"
            return _real_plt.subplots(*_args, num=num, **_kwargs)

        def __getattr__(self, _name):
            return getattr(_real_plt, _name)

    plt = _PltShadow()
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
    training_source = gd.get("training_source", "internal")

    # training_intervals: direct param overrides global_data
    if training_intervals is None:
        training_intervals = gd.get("training_intervals")

    _all_stable_ranges = [
        (_t0, _t1)
        for _t0, _t1, _lbl in (training_intervals or [])
        if str(_lbl).startswith("stable")
    ]

    # shared event vlines -- only the first-detection marker, no t_gt.
    auto_vlines = []
    if t_d is not None and len(t_d) > 0:
        _td_val = float(t_d[0])
        auto_vlines.append((_td_val, rf"$t_d={_td_val:.3f}$ s", color_orange))

    # ── C1: Signal (single panel, article-plot-style FIGSIZE_SIMPLE) ───────
    t_arr = np.asarray(gd.get("t", []))
    v_arr = np.asarray(gd.get("q_o_signal", []))
    if t_arr.size > 0 and v_arr.size == t_arr.size:
        fig_c1, ax_c1 = plt.subplots(figsize=figsize_from_scale(figsize_simple, scale))
        ax_c1.plot(t_arr, v_arr, color=color_azul, lw=0.8)
        ax_c1.set_xlabel("Time [s]")
        ax_c1.set_ylabel("Velocity [m/s]")
        ax_c1.set_title("C1 — Signal")
        _draw_vlines(ax_c1, auto_vlines)
        # One-shot layout, not a persistent engine (layout='tight' /
        # constrained_layout=True recompute on every draw/savefig, so
        # ax.get_position() would visibly wobble depending on which
        # vline/hline text labels happen to be in view at that moment —
        # see plot_training_distribution's sibling fix in plots.py history).
        fig_c1.tight_layout()

    # ── C2: Areas per window ──────────────────────────────────────────────
    fig_c2, ax_c2 = plt.subplots(figsize=_LYAP_FIGSIZE)
    ax_c2.set_title("C2 — Areas per Window")
    ax_c2.set_xlabel("Time [s]")
    ax_c2.set_ylabel("Shoelace area [m·m/s]")
    valid = np.isfinite(areas)
    if valid.any():
        # ax_c2.semilogy(t_wins[valid], areas[valid], color=color_azul,
        #                lw=1.0, marker="o", markersize=2, label="$A_k$")
        ax_c2.plot(t_wins[valid], areas[valid], color=color_azul,
                   lw=1.0, marker="o", markersize=2, label="$A_k$")
        ax_c2.set_yscale("log")
    if thr and area_threshold_enabled:
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
    fig_c2b, ax_c2b = plt.subplots(figsize=_LYAP_FIGSIZE)
    ax_c2b.set_title("C2-b — Trajectory per Window")
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
        apply_sci_yaxis(ax_c2b)
    if thr and area_threshold_enabled:
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
    fig_c3, ax_c3 = plt.subplots(figsize=_LYAP_FIGSIZE)
    ax_c3.set_title(r"C3 — Lyapunov $\hat{\sigma}$(t)")
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
    fig_c3.tight_layout()  # one-shot, not layout='tight' — see C1 for why

    # ── Training population diagnostics (histogram + verification curve) ───
    # Sourced from global_data["training_areas"]/["training_t_wins"] — the
    # exact windows that trained mu/sigma above, correct regardless of
    # training_source ("internal" training_intervals or "external_reference"
    # reference_signal — see COMMON_TEMPLATE.md Fase 3). Replaces the old
    # C4/D1 figures, which always re-derived a "stable" slice from
    # training_intervals against THIS signal even when the threshold had
    # actually been trained on a separate reference_signal.
    # figsize matches this figure set's own C1-C3/Ĝ/Ĝs sizing (_LYAP_FIGSIZE),
    # not plots.py's own default (its callers use their own default instead).
    plot_training_distribution(gd, name=name, log_transform=True, figsize=_LYAP_FIGSIZE)

    # ── C6/C7: per-case grid (one square subplot per training piece), same
    # idea as MaxEnt's F0a/F1 and SST's C6/C7. Prefer external reference_signal
    # pieces (own case per piece); else per training_intervals stable range;
    # else a single fallback panel. Only stable -- Green-Area, like SST, has
    # no separate "chatter training" population to grid.
    t_arr_gd = np.asarray(gd.get("t", []))
    q_arr_gd = np.asarray(gd.get("q_signal", []))
    training_areas  = np.asarray(gd.get("training_areas", []), dtype=float)
    training_t_wins = np.asarray(gd.get("training_t_wins", []), dtype=float)
    _piece_counts = gd.get("reference_piece_window_counts") or []

    if training_areas.size > 0:
        _grid_sig_pieces: list = []
        _grid_area_pieces: list = []
        if training_source == "external_reference" and reference_signal and len(reference_signal) > 1:
            _MAX_GRID_PTS = 50_000
            _cap = max(1, _MAX_GRID_PTS // len(reference_signal))
            for _p in reference_signal:
                _pt = np.asarray(_p.t_analysis, dtype=float)
                _px = np.asarray(_p.signal_analysis, dtype=float)
                if _pt.size > _cap:
                    _pstep = _pt.size // _cap
                    _pt, _px = _pt[::_pstep], _px[::_pstep]
                _grid_sig_pieces.append((_pt, _px))
            if len(_piece_counts) > 1 and sum(_piece_counts) == training_areas.size:
                _bounds = np.cumsum(_piece_counts)
                _starts = np.concatenate(([0], _bounds[:-1]))
                for _s0, _s1 in zip(_starts, _bounds):
                    _grid_area_pieces.append((training_t_wins[_s0:_s1], training_areas[_s0:_s1]))
        elif _all_stable_ranges and len(_all_stable_ranges) > 1:
            for _r0, _r1 in _all_stable_ranges:
                _m_sig = (t_arr_gd >= _r0) & (t_arr_gd <= _r1)
                _grid_sig_pieces.append((t_arr_gd[_m_sig], q_arr_gd[_m_sig]))
                _m_a = (training_t_wins >= _r0) & (training_t_wins <= _r1)
                _grid_area_pieces.append((training_t_wins[_m_a], training_areas[_m_a]))

        if not _grid_sig_pieces:
            _t0f, _t1f = float(np.min(training_t_wins)), float(np.max(training_t_wins))
            _m_fb = (t_arr_gd >= _t0f) & (t_arr_gd <= _t1f)
            _grid_sig_pieces = [(t_arr_gd[_m_fb], q_arr_gd[_m_fb])]
        if not _grid_area_pieces:
            _grid_area_pieces = [(training_t_wins, training_areas)]

        _area_hlines = []
        if thr and area_threshold_enabled:
            _area_hlines = [
                (10 ** thr["upper"], color_red,  "--"),
                (10 ** thr["lower"], color_red,  ":"),
                (10 ** thr["mu"],    color_verde, "-"),
            ]
        _plot_pieces_grid(
            _grid_sig_pieces, color=color_azul,
            title="Stable Training Signal (per case)",
            ylabel="Displacement [m]",
            figsize_simple=figsize_simple, scale=grid_scale,
            fig_label="C6 — Stable Training Signal (per case)",
        )
        _plot_pieces_grid(
            _grid_area_pieces, color=color_purple,
            title="Stable Decision Variable — Area (per case)",
            ylabel="Shoelace area [m·m/s]",
            hlines=_area_hlines, yscale="log",
            figsize_simple=figsize_simple, scale=grid_scale,
            fig_label="C7 — Stable Decision Variable (per case)",
        )

    # ── D1b / D4b: per-label breakdown — only meaningful for internal
    # training (needs this signal's own training_intervals labels; skipped
    # for training_source == "external_reference", where they don't apply).
    _stable_label_groups: dict = {}
    for _t0, _t1, _lbl in (training_intervals or []):
        if str(_lbl).startswith("stable"):
            _stable_label_groups.setdefault(str(_lbl), []).append((_t0, _t1))

    if (training_source != "external_reference" and area_threshold_enabled
            and len(_stable_label_groups) >= 2
            and thr and float(thr.get("sigma", 0.0)) > 0):
        mu_h, std_h = float(thr["mu"]), float(thr["sigma"])
        stable_mask = np.zeros(len(t_wins), dtype=bool)
        for _t0, _t1 in _all_stable_ranges:
            stable_mask |= (t_wins >= _t0) & (t_wins <= _t1)
        stable_areas = areas[stable_mask]
        valid_sa = np.isfinite(stable_areas) & (stable_areas > 0)
        log10_a = np.log10(stable_areas[valid_sa])
        _t_stable = t_wins[stable_mask][valid_sa]

        # D1b — one figure per stable label  [MaxEnt F1b analog]
        for _gi, (_lbl_name, _ranges) in enumerate(_stable_label_groups.items()):
                fig_d1b, ax_d1b = plt.subplots(
                    figsize=_LYAP_FIGSIZE,
                    num=f"D1b.{_gi} — {_lbl_name}",
                )
                ax_d1b.set_title(rf"D1b — Stable Areas | {_lbl_name}")
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
                fig_d1b.tight_layout()  # one-shot, not layout='tight' — see C1 for why

        # D4b — one figure per stable label histogram  [MaxEnt F4b analog]
        if len(_stable_label_groups) >= 2 and std_h > 0:
            _n_bins_d4b   = 40
            _counts_all_d, _bin_edges_d = np.histogram(log10_a, bins=_n_bins_d4b)
            _widths_d      = np.diff(_bin_edges_d)
            _heights_all_d = _counts_all_d / (len(log10_a) * _widths_d)
            for _gi, (_lbl_name, _ranges) in enumerate(_stable_label_groups.items()):
                fig_d4b, ax_d4b = plt.subplots(
                    figsize=_LYAP_FIGSIZE,
                    num=f"D4b.{_gi} — {_lbl_name}",
                )
                ax_d4b.set_title(rf"D4b — Area PDF | {_lbl_name}")
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
                fig_d4b.tight_layout()  # one-shot, not layout='tight' — see C1 for why

    # ── Ĝ accumulator (optional) ──────────────────────────────────────────
    G = np.asarray(result.G_hat)
    if G.size > 0:
        fig_g, ax_g = plt.subplots(figsize=_LYAP_FIGSIZE)
        ax_g.set_title(r"$\hat{G}$ Accumulator")
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
        apply_sci_yaxis(ax_g)
        _draw_vlines(ax_g, auto_vlines)
        ax_g.legend()
        fig_g.tight_layout()  # one-shot, not layout='tight' — see C1 for why

    # ── Ĝ sliding window (optional) ───────────────────────────────────────
    Gs = np.asarray(result.G_hat_sliding)
    if Gs.size > 0:
        fig_gs, ax_gs = plt.subplots(figsize=_LYAP_FIGSIZE)
        ax_gs.set_title(r"$\hat{G}$ Sliding Window")
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
        apply_sci_yaxis(ax_gs)
        _draw_vlines(ax_gs, auto_vlines)
        ax_gs.legend()
        fig_gs.tight_layout()  # one-shot, not layout='tight' — see C1 for why

    if show:
        plt.show(block=True)


# ---------------------------------------------------------------------------
# Signal diagnostics plots
# ---------------------------------------------------------------------------

def plots_signal_diagnostics(
    signal: SignalData,
    result: LyapunovResult,
    stable_range: Tuple[float, float] = (0.5, 4.0),
    zoom_range: Tuple[float, float] = (1.0, 1.2),
    eq_smooth_s: float = 0.050,
    freq_markers: Optional[Dict[str, float]] = None,
    t_beat_ms: Optional[float] = None,
    show: bool = True,
    scale: float = SCALE,
    figsize_wide: Tuple[float, float] = FIGSIZE_WIDE,
) -> None:
    """Diagnostic plots for understanding signal structure and area variability.

    Figures produced
    ----------------
    **Fig A — Frequency content & area autocorrelation**
        A1. FFT of *x* in the stable zone (``stable_range``), with vertical
            markers at ``freq_markers`` (if given — see below).
        A2. Autocorrelation of ln(areas) in the stable zone — reveals
            periodic modulation (beat, tooth-pass, etc.); marks ``t_beat_ms``
            (if given).

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
    result        : output of :func:`run_lyapunov`.
    stable_range  : ``(t_start, t_end)`` [s] defining the "stable" zone
                    used for the FFT and autocorrelation.
    zoom_range    : ``(t_start, t_end)`` [s] for the signal zoom (B1/B2).
                    Should be a short window (≤ 0.5 s).
    eq_smooth_s   : half-width [s] of the moving-average used to estimate
                    ``x_eq``.  Default 50 ms — slow enough to follow AP
                    drift but fast enough to not absorb dynamic vibration.
    freq_markers  : optional ``{label: frequency_hz}`` vertical markers drawn
                    on the FFT panel (A1) — e.g. the actual ``f_modal``/tooth-
                    pass/beat frequencies of *this* run's config. ``None``
                    (default) draws no markers, rather than guessing them.
    t_beat_ms     : optional expected beat period [ms] marked on the
                    autocorrelation panel (A2). ``None`` (default) draws no
                    marker.
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
    figA, (aA1, aA2) = plt.subplots(2, 1, figsize=_diag_figsize(nrows=2, scale=scale, base_wide=figsize_wide),
                                     constrained_layout=True)
    figA.suptitle("Signal diagnostics — frequency content & area statistics")

    # A1: FFT of x in stable zone
    ms = _mask(t, *stable_range)
    x_stab = x[ms]
    N_fft  = len(x_stab)
    if N_fft > 0:
        fft_mag = np.abs(np.fft.rfft(x_stab))
        freqs   = np.fft.rfftfreq(N_fft, 1.0 / fs)
        mask_f  = freqs < min(fs / 2, 800.0)
        aA1.semilogy(freqs[mask_f], fft_mag[mask_f],
                     color=color_azul, lw=0.8, label="FFT |X(f)|")
        _marker_colors = [color_red, color_verde, color_purple, color_orange, color_azul]
        for _mi, (lbl, fmark) in enumerate((freq_markers or {}).items()):
            if fmark < freqs[-1]:
                col = _marker_colors[_mi % len(_marker_colors)]
                aA1.axvline(fmark, color=col, lw=1.5, ls="--",
                            label=f"{lbl} {fmark:.1f} Hz")
    aA1.set_xlabel("Frequency [Hz]")
    aA1.set_ylabel("|FFT(x)|")
    aA1.set_title("Frequency content — stable zone")
    aA1.text(0.98, 0.95, f"t = [{stable_range[0]:.1f}, {stable_range[1]:.1f}] s",
             transform=aA1.transAxes, ha="right", va="top", fontsize=9)
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
                 color=color_orange, lw=1.2, label="autocorr ln(A)")
        if t_beat_ms is not None:
            aA2.axvline(t_beat_ms, color=color_purple, lw=1.2, ls="--",
                        label=f"T_beat={t_beat_ms:.1f} ms (expected)")
        aA2.axhline(0, color="black", lw=0.5)
    aA2.set_xlabel("Lag [ms]")
    aA2.set_ylabel("Autocorrelation")
    aA2.set_title("Autocorrelation of ln(areas) in stable zone")
    aA2.legend(fontsize=9)
    aA2.grid(True, alpha=0.3)

    # ── Fig B — Equilibrium decomposition ─────────────────────────────────
    figB, (aB1, aB2, aB3) = plt.subplots(3, 1, figsize=_diag_figsize(nrows=3, scale=scale, base_wide=figsize_wide),
                                          constrained_layout=True)
    figB.suptitle("Quasi-static equilibrium & dynamic decomposition")

    mz = _mask(t, *zoom_range)
    tz = t[mz]; xz = x[mz]; xeqz = x_eq[mz]; xdynz = x_dyn[mz]
    vz = v[mz]; vdynz = v_dyn[mz]

    # B1: x with x_eq overlay
    aB1.plot(tz * 1000, xz,    color=color_azul, lw=0.9, label="x  (absolute)")
    aB1.plot(tz * 1000, xeqz,  color=color_red,  lw=2.0, ls="--",
             label=f"x_eq ≈ moving avg ({eq_smooth_s*1000:.0f} ms)")
    aB1.set_ylabel("x [m]")
    aB1.set_title("Signal and quasi-static equilibrium")
    aB1.text(0.98, 0.95, f"t = [{zoom_range[0]:.2f}, {zoom_range[1]:.2f}] s",
             transform=aB1.transAxes, ha="right", va="top", fontsize=9)
    aB1.legend(fontsize=9)
    aB1.grid(True, alpha=0.3)

    # B2: dynamic residual
    aB2.plot(tz * 1000, xdynz, color=color_verde, lw=0.9, label="x_dyn = x − x_eq")
    aB2.axhline(0, color="black", lw=0.5)
    aB2.set_ylabel("x_dyn [m]")
    aB2.set_title("Dynamic component (vibration about equilibrium)")
    aB2.legend(fontsize=9)
    aB2.grid(True, alpha=0.3)

    # B3: orbit of centered signal
    sc = aB3.scatter(xdynz, vdynz,
                     c=tz, cmap="viridis", s=4, alpha=0.8,
                     label="centered orbit (x_dyn, v_dyn)")
    plt.colorbar(sc, ax=aB3, label="time [s]")
    aB3.axhline(0, color="black", lw=0.3)
    aB3.axvline(0, color="black", lw=0.3)
    aB3.set_xlabel("x_dyn [m]")
    aB3.set_ylabel("v_dyn [m/s]")
    aB3.set_title("Centered phase portrait")
    aB3.legend(fontsize=9)
    aB3.grid(True, alpha=0.2)

    # ── Fig C — Phase portrait snapshots: stable / transition / chatter ────
    t_total = float(t[-1] - t[0])
    t0      = float(t[0])

    # pick three snapshot times: 20 % (stable), 55 % (transition), 85 % (chatter)
    snap_fracs  = [0.20, 0.55, 0.85]
    snap_labels = ["Stable (20%)", "Transition (55%)", "Chatter (85%)"]
    snap_colors = [color_azul, color_orange, color_red]
    snap_dur    = min(0.05, t_total * 0.03)   # 50 ms per snapshot

    figC, axes_c = plt.subplots(1, 3, figsize=_diag_figsize(ncols=3, scale=scale, base_wide=figsize_wide),
                                 constrained_layout=True)
    figC.suptitle("Phase portrait evolution — stable / transition / chatter")

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

    if show:
        plt.show()

