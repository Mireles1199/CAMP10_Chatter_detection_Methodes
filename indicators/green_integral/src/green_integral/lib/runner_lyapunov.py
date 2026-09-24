"""Fixed-window Lyapunov chatter indicator.

Differences from the standard green_integral indicator:

* **No zero-crossing detection** — windows have a fixed duration of
  ``num_T × T_modal`` seconds, exactly as specified.
* **No clustering** — one shoelace area per window, no cross-window grouping.
* **Lyapunov exponent** σ̂ estimated from consecutive log-area ratios or a
  local linear fit (frozen-time mode).
* **Optional EWMA smoothing** of σ̂ (set ``lambda_ewma`` to a float ∈ (0,1];
  pass ``None`` to disable).
* **Optional accumulation** Ĝ = ∫ σ̂_EWMA dt, analogous to the RALE
  indicator (set ``accumulate=True`` to enable; ``None`` / ``False`` disables).

Decision rule
-------------
    σ̂ > 0   →  chatter (orbit growing)
    Ĝ > 0   →  chatter confirmed (accumulated evidence, if enabled)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.linear_model import TheilSenRegressor
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

from ..utils.types import SignalData, FixedWindowConfig, FixedWindowResult
from ..utils.signal_filter import savgol_filter_window
from ..logging_setup import LOGGING_LEVELS, configure_logging
from .diagnostics import estimate_center, center_trajectory, compute_local_phase, drift_ratio

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter




# ---------------------------------------------------------------------------
# Stable-region mask helper (shared with runner.py logic)
# ---------------------------------------------------------------------------

def _select_stable_mask(
    t_wins: np.ndarray,
    training_intervals: Optional[List[Tuple[float, float, str]]],
    stable_time: Optional[Tuple[float, float]],
    frac_stable: float,
) -> np.ndarray:
    """Boolean mask of windows belonging to the stable training region."""
    if training_intervals is not None:
        mask = np.zeros(len(t_wins), dtype=bool)
        for t0, t1, label in training_intervals:
            if str(label).startswith("stable"):
                mask |= (t_wins >= t0) & (t_wins <= t1)
    elif stable_time is not None:
        mask = (t_wins >= stable_time[0]) & (t_wins <= stable_time[1])
    else:
        n_stable = max(3, int(len(t_wins) * frac_stable))
        mask = np.zeros(len(t_wins), dtype=bool)
        mask[:n_stable] = True
    return mask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _shoelace(x: np.ndarray, v: np.ndarray) -> float:
    """Shoelace (Green's theorem) area of the phase-space orbit.

    Returns ``np.nan`` for degenerate windows (< 3 points).
    """
    if len(x) < 3:
        return np.nan
    return 0.5 * abs(float(
        np.dot(x, np.roll(v, -1)) - np.dot(v, np.roll(x, -1))
    ))

def _shoelace_oriented(x: np.ndarray, v: np.ndarray) -> float:
    if len(x) < 3:
        return np.nan

    return 0.5 * float(
        np.dot(x, np.roll(v, -1)) - np.dot(v, np.roll(x, -1))
    )

def _shoelace_open_contribution(x, y):
    """
    Contribution orientée d'une trajectoire ouverte.
    Ne ferme PAS la courbe.
    """
    return 0.5 * float(
        np.sum(x[:-1] * y[1:] - y[:-1] * x[1:])
    )

def _closure_contribution(x_start, y_start, x_end, y_end):
    """
    Contribution orientée du segment de fermeture :
    point final -> point initial.
    """
    return 0.5 * float(
        x_end * y_start - y_end * x_start
    )

def _ajouter_valeurs_barres(ax, bars, fmt="{:.6e}"):
    """
    Ajoute les valeurs numériques au-dessus ou au-dessous des barres.
    """
    for bar in bars:
        hauteur = bar.get_height()
        x_pos = bar.get_x() + bar.get_width() / 2

        if hauteur >= 0:
            va = "bottom"
            y_pos = hauteur
        else:
            va = "top"
            y_pos = hauteur

        ax.text(
            x_pos,
            y_pos,
            fmt.format(hauteur),
            ha="center",
            va=va,
            fontsize=14,
            rotation=0
        )

def winding_number_point(px, py, x, y):
    """
    Nombre d'enroulement de la courbe fermée autour du point (px, py).
    """
    dx = x - px
    dy = y - py

    dx_next = np.roll(dx, -1)
    dy_next = np.roll(dy, -1)

    cross = dx * dy_next - dy * dx_next
    dot = dx * dx_next + dy * dy_next

    angles = np.arctan2(cross, dot)

    return np.sum(angles) / (2 * np.pi)


def _estimate_sigma(
    areas: np.ndarray,
    t_wins: np.ndarray,
    T_window: float,
    eps: float,
    method: str,
    local_n: int,
) -> np.ndarray:
    """Estimate instantaneous Lyapunov exponent σ̂ from area sequence.

    Parameters
    ----------
    areas   : per-window shoelace areas (positive floats).
    t_wins  : window start times.
    T_window : window duration = num_T * T_modal [s].
    eps     : minimum valid area threshold.
    method  : ``"ratio"`` or ``"frozen_time"``.
    local_n : neighbourhood half-width for frozen-time mode.

    Returns
    -------
    sigma : array same length as *areas*, NaN where insufficient data.

    Notes
    -----
    ``A_k ∝ ‖δx_k‖² ∝ exp(2σ k T_window)``
    → slope of ln(A) vs k*T_window = 2σ
    → σ̂ = Δln(A) / (2 T_window)
    """
    A = np.where(areas > eps, areas, np.nan)
    sigma = np.full(len(A), np.nan)

    if method.strip().lower() == "ratio":
        log_A = np.log(A)
        # σ̂_k = (ln A_k - ln A_{k-1}) / (2 * T_window)
        sigma[1:] = (log_A[1:] - log_A[:-1]) / (2.0 * T_window)

    else:  # frozen_time
        n_local = max(3, int(local_n))
        if n_local % 2 == 0:
            n_local += 1
        half = n_local // 2

        for k in range(len(A)):
            i0 = max(0, k - half)
            i1 = min(len(A), k + half + 1)
            A_loc = A[i0:i1]
            t_loc = t_wins[i0:i1]
            valid = np.isfinite(A_loc) & np.isfinite(t_loc)
            if np.count_nonzero(valid) < 2:
                continue

            y_fit = np.log(A_loc[valid])
            x_fit = t_loc[valid]

            if _SKLEARN_OK and len(x_fit) >= 3:
                model = TheilSenRegressor(random_state=0)
                model.fit(x_fit.reshape(-1, 1), y_fit)
                slope = float(model.coef_[0])
            else:
                slope = float(np.polyfit(x_fit, y_fit, 1)[0])

            sigma[k] = slope / 2.0  # A ∝ exp(2σt) → slope = 2σ

    return sigma


def _apply_ewma(sigma: np.ndarray, lam: float) -> np.ndarray:
    """Causal EWMA smoother.  NaN inputs use hold-last-value."""
    out = np.full_like(sigma, np.nan)
    s_prev = np.nan
    for i, s in enumerate(sigma):
        if np.isnan(s):
            out[i] = s_prev
        elif np.isnan(s_prev):
            out[i] = s
        else:
            out[i] = (1.0 - lam) * s_prev + lam * s
        s_prev = out[i]
    return out


def _integrate_G(sigma_ewma: np.ndarray, t_wins: np.ndarray) -> np.ndarray:
    """Ĝ(t) = ∫ σ̂_EWMA dt  (trapezoidal rule)."""
    G = np.zeros(len(sigma_ewma), dtype=float)
    for i in range(1, len(sigma_ewma)):
        s0 = 0.0 if np.isnan(sigma_ewma[i - 1]) else sigma_ewma[i - 1]
        s1 = 0.0 if np.isnan(sigma_ewma[i])     else sigma_ewma[i]
        dt = max(0.0, float(t_wins[i] - t_wins[i - 1]))
        G[i] = G[i - 1] + 0.5 * (s0 + s1) * dt
    return G


def _integrate_G_sliding(
    sigma_ewma: np.ndarray,
    t_wins: np.ndarray,
    T_memory: float,
) -> np.ndarray:
    """Sliding-window Ĝ:  ∫_{t - T_memory}^{t} σ̂_EWMA dτ  (trapezoidal rule).

    Parameters
    ----------
    sigma_ewma : smoothed Lyapunov exponent array.
    t_wins     : window start times.
    T_memory   : width of the sliding integration window [s].

    Returns
    -------
    G_slide : same length as sigma_ewma.  Tracks current state — drops back
              below 0 when the system stabilises after a chatter episode.
    """
    n = len(sigma_ewma)
    G_slide = np.zeros(n, dtype=float)
    for k in range(1, n):
        t_lo = t_wins[k] - T_memory
        # find the first index inside the memory window
        i0 = np.searchsorted(t_wins, t_lo, side="left")
        i0 = max(0, i0)
        # trapezoidal integral from i0 to k
        acc = 0.0
        for j in range(i0 + 1, k + 1):
            s0 = 0.0 if np.isnan(sigma_ewma[j - 1]) else sigma_ewma[j - 1]
            s1 = 0.0 if np.isnan(sigma_ewma[j])     else sigma_ewma[j]
            dt = max(0.0, float(t_wins[j] - t_wins[j - 1]))
            acc += 0.5 * (s0 + s1) * dt
        G_slide[k] = acc
    return G_slide


# ---------------------------------------------------------------------------
# Zero-crossing cycle extractor
# ---------------------------------------------------------------------------

def extract_complete_cycles(
    t_win: np.ndarray,
    q_win: np.ndarray,
    v_win: np.ndarray,
    frac_min: float = 0.4,
    direction: str = "up",
    v_ref: Optional[np.ndarray] = None,
    v_cyc_src: Optional[np.ndarray] = None,
    force_zero_endpoints: bool = False,
) -> list:
    """
    Extrae ciclos completos usando cruces interpolados de v=0.

    No requiere conocer la frecuencia del ciclo.
    El debounce es auto-calibrado: la separación mínima entre cruces válidos
    se estima como frac_min * mediana(gaps entre todos los cruces detectados).

    Parámetros
    ----------
    frac_min   : fracción de la mediana de gaps → umbral de debounce.
    direction  : "up" (v: -→+), "down" (v: +→-), "any".
    v_ref      : señal usada SOLO para detectar cruces (e.g. v_win detrended).
                 Si es None se usa v_win para detección.
    v_cyc_src  : señal usada para construir los arrays v_cyc retornados.
                 None → usa v_win (Opciones 1 y 2).
                 v_ref → usa señal detrended (Opción 3).
    force_zero_endpoints : si True, los endpoints del ciclo se fijan a 0.0
                 independientemente de v_cyc_src (Opción 1). El interior
                 usa v_for_cyc (= v_win cuando v_cyc_src es None).

    Retorna
    -------
    Lista de (t_cyc, q_cyc, v_cyc). Cada ciclo empieza y termina en un cruce
    del mismo tipo con los puntos interpolados incluidos.
    """
    v_detect  = v_ref     if v_ref     is not None else v_win
    v_for_cyc = v_cyc_src if v_cyc_src is not None else v_win

    # ── 1. Detectar todos los cruces de signo ────────────────────────────
    signs = np.sign(v_detect).astype(float)
    signs[signs == 0] = 1.0   # evita doble-detección en ceros exactos

    raw_idx = np.where(np.diff(signs) != 0)[0]   # índice ANTES del cruce

    if len(raw_idx) == 0:
        return []

    # ── 2. Interpolar t, q y dirección en cada cruce ─────────────────────
    raw_t   = []
    raw_dir = []
    raw_k   = []
    raw_fr  = []

    for k in raw_idx:
        v0, v1 = float(v_detect[k]), float(v_detect[k + 1])
        frac = -v0 / (v1 - v0)
        tc   = float(t_win[k]) + frac * (float(t_win[k + 1]) - float(t_win[k]))
        raw_t.append(tc)
        raw_dir.append(+1 if v1 > v0 else -1)
        raw_k.append(k)
        raw_fr.append(frac)

    raw_t = np.array(raw_t)

    # ── 3. Debounce auto-calibrado (sin T_nominal) ────────────────────────
    gaps = np.diff(raw_t)
    if len(gaps) == 0:
        return []

    dt_ref = float(np.median(gaps))
    dt_min = frac_min * dt_ref

    valid = []
    t_last = -np.inf
    for tc, d, k, fr in zip(raw_t, raw_dir, raw_k, raw_fr):
        if tc - t_last >= dt_min:
            valid.append((tc, d, k, fr))
            t_last = tc

    # ── 4. Filtrar por dirección ──────────────────────────────────────────
    if direction == "up":
        selected = [c for c in valid if c[1] == +1]
    elif direction == "down":
        selected = [c for c in valid if c[1] == -1]
    else:
        selected = valid

    if len(selected) < 2:
        return []

    # ── 5. Construir ciclos con puntos interpolados ───────────────────────
    cycles = []
    for i in range(len(selected) - 1):
        tc0, _, k0, fr0 = selected[i]
        tc1, _, k1, fr1 = selected[i + 1]

        q_c0 = float(q_win[k0]) + fr0 * (float(q_win[k0 + 1]) - float(q_win[k0]))
        q_c1 = float(q_win[k1]) + fr1 * (float(q_win[k1 + 1]) - float(q_win[k1]))

        # v endpoints: modo según v_cycle_mode
        # Opción 1 (force_zero_endpoints) → 0.0 hardcodeado, K = 0
        # Opción 2 (v_for_cyc = v_win)    → valor real interpolado, K ≈ offset×Δq
        # Opción 3 (v_for_cyc = detrend)  → ≈ 0.0 interpolado, K = 0
        if force_zero_endpoints:
            v_c0 = 0.0
            v_c1 = 0.0
        else:
            v_c0 = float(v_for_cyc[k0]) + fr0 * (float(v_for_cyc[k0 + 1]) - float(v_for_cyc[k0]))
            v_c1 = float(v_for_cyc[k1]) + fr1 * (float(v_for_cyc[k1 + 1]) - float(v_for_cyc[k1]))

        sl = slice(k0 + 1, k1 + 1)
        t_cyc = np.concatenate([[tc0],  t_win[sl],       [tc1]])
        q_cyc = np.concatenate([[q_c0], q_win[sl],       [q_c1]])
        v_cyc = np.concatenate([[v_c0], v_for_cyc[sl],   [v_c1]])

        cycles.append((t_cyc, q_cyc, v_cyc))

    return cycles


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def _fixed_window_pipeline(
    signal: SignalData,
    config: FixedWindowConfig,
) -> FixedWindowResult:
    """Fixed-window Lyapunov indicator pipeline."""

    t   = np.asarray(signal.t,            dtype=float)
    q   = np.asarray(signal.displacement, dtype=float)
    q_o = np.asarray(signal.velocity,     dtype=float)

    T_win  = config.T_window
    T_cycle = 1.0 / config.f_modal
    dt_sig = float(t[1] - t[0])
    N_win  = max(2, int(round(T_win / dt_sig)))  # samples per window
    N_win_cycle  = max(2, int(round(T_cycle / dt_sig)))  # samples per window

    # step between window starts
    if config.dt is None:
        step = N_win  # non-overlapping
    else:
        step = max(1, int(round(config.dt / dt_sig)))


    # ---- 1. Build windows and compute areas --------------------------------
    areas_list: list  = []
    areas_list_subwin: list = []
    areas_list_subwin_oriented: list = []
    areas_list_oriented: list = []

    C_list: list = []
    K_list: list = []

    t_wins_list: list = []
    error_list: list = []
    _cycles_count_history: list = []   # (t_win_end, n_cycles) for all windows

    i = 0
    flag_plot_signal = True
    _dbg_lo, _dbg_hi = config.debug_window_range
    while i + N_win <= len(t):

        # t_win = t[i:i + N_win]
        # q_win = q[i:i + N_win]
        # v_win = q_o[i:i + N_win]

        # if config.data_filtrated and len(q_win) >= 7:
        #     q_win = savgol_filter_window(q_win)
        #     v_win = savgol_filter_window(v_win)

        # areas_list.append(_shoelace(q_win, v_win))
        # t_wins_list.append(float(t_win[-1]))
        # i += step

        t_win = t[i:i + N_win]
        q_win = q[i:i + N_win]
        v_win = q_o[i:i + N_win]

        # ── Debug gate ──────────────────────────────────────────────────
        _t0_win = float(t_win[0])
        _do_debug = (
            config.debug_level >= 2
            and _t0_win >= _dbg_lo
            and (_dbg_hi is None or _t0_win < _dbg_hi)
        )

        if _do_debug:
            if flag_plot_signal:
                fig_signal_x, ax_signal_x = plt.subplots(figsize=(12, 6))
                ax_signal_x.plot(t, q, label='Displacement (q)')
                ax_signal_x.set_title("Full Signal: Displacement vs Time")
                ax_signal_x.legend()

                fig_signal_v, ax_signal_v = plt.subplots(figsize=(12, 6))
                ax_signal_v.plot(t, q_o, label='Velocity (q_o)', color='orange')
                ax_signal_v.set_title("Full Signal: Velocity vs Time")
                ax_signal_v.legend()

                # ── Historial de ciclos hasta el inicio del rango de debug ──
                _hist_pre = [(t_end, n) for t_end, n in _cycles_count_history
                             if t_end < _dbg_lo]
                if _hist_pre:
                    _ht = [h[0] for h in _hist_pre]
                    _hn = [h[1] for h in _hist_pre]
                    fig_ncyc, ax_ncyc = plt.subplots(figsize=(12, 4))
                    ax_ncyc.step(_ht, _hn, where='post', color='steelblue')
                    ax_ncyc.set_xlabel("t fin ventana [s]")
                    ax_ncyc.set_ylabel("N ciclos detectados")
                    ax_ncyc.set_title(
                        f"Ciclos por ventana — hasta inicio debug (t < {_dbg_lo:.3f} s)"
                    )
                    ax_ncyc.axvline(_dbg_lo, color='red', linestyle='--',
                                    label=f'debug_range start = {_dbg_lo:.3f} s')
                    ax_ncyc.legend()
                    plt.tight_layout()

                flag_plot_signal = False

            fig_disp, ax_disp = plt.subplots(figsize=(10, 6))
            ax_disp.plot(t_win, q_win, label='Displacement (q)',)
            fig_disp.suptitle(f"Displacement (q) vs Time (window end at t={t_win[-1]:.2f}s)")
            ax_disp.legend()

            fig_vel, ax_vel = plt.subplots(figsize=(10, 6))
            ax_vel.plot(t_win, v_win, label='Velocity (q_o)')
            fig_vel.suptitle(f"Velocity (q_o) vs Time (window end at t={t_win[-1]:.2f}s)")
            ax_vel.legend()

            fig_phase, ax_phase = plt.subplots(figsize=(10, 6))
            ax_phase.plot(q_win, v_win, label='Phase Space (q vs q_o)')
            fig_phase.suptitle(f"Phase Space (q vs q_o) - window end at t={t_win[-1]:.2f}s")
            ax_phase.legend()

            _q_trend = np.polyval(np.polyfit(t_win, q_win, 1), t_win)
            _q_detrended = q_win - _q_trend
            fig_disp_dc, ax_disp_dc = plt.subplots(figsize=(10, 6))
            ax_disp_dc.plot(t_win, _q_detrended, label='q sin DC',        color='steelblue')
            fig_disp_dc.suptitle(
                f"Displacement sin DC — ventana end t={t_win[-1]:.2f}s"
            )
            ax_disp_dc.set_xlabel("t [s]")
            ax_disp_dc.set_ylabel("q")
            ax_disp_dc.legend()
            ax_disp_dc.axhline(0, color='k', linewidth=0.6, linestyle=':')

        if config.data_filtrated and len(q_win) >= 7:


            q_win = savgol_filter_window(q_win)
            v_win = savgol_filter_window(v_win)

            if _do_debug:
                ax_disp.plot(t_win, q_win, label='Displacement (q) - filtered')
                ax_disp.legend()
                ax_vel.plot(t_win, v_win, label='Velocity (q_o) - filtered')
                ax_vel.legend()

                _q_trend = np.polyval(np.polyfit(t_win, q_win, 1), t_win)
                _q_detrended = q_win - _q_trend

                ax_disp_dc.plot(t_win, _q_detrended, label='q filtrado sin DC', color='orange', alpha=0.8)
                ax_disp_dc.legend()

        if _do_debug:
            # ── Portrait centrado + fase local ───────────────────────────
            _cx, _cv = estimate_center(q_win, v_win, config.center_win)
            _xr, _vr = center_trajectory(q_win, v_win, _cx, _cv)
            _rho     = drift_ratio(_cx, _cv, _xr, _vr)

            # Figura: portrait original (izq) vs centrado (der)
            fig_cent, axes_cent = plt.subplots(1, 2, figsize=(14, 6))
            fig_cent.suptitle(
                f"Portrait original vs centrado — t_end={t_win[-1]:.2f}s  "
                f"(\u03c1={_rho:.2f})",
                fontsize=13,
            )
            # Panel izquierdo — trayectoria original + trayectoria del centro
            axes_cent[0].plot(q_win, v_win, color='steelblue', linewidth=1, label='trayectoria')
            axes_cent[0].plot(_cx, _cv, color='orange', linewidth=1.5,
                              linestyle='--', label='centro lento')
            axes_cent[0].scatter(q_win[0],  v_win[0],  color='green', s=60, zorder=5, label='inicio')
            axes_cent[0].scatter(q_win[-1], v_win[-1], color='red',   s=60, zorder=5, label='fin')
            # axes_cent[0].set_aspect( adjustable='datalim')
            axes_cent[0].set_xlabel('q'); axes_cent[0].set_ylabel('v')
            axes_cent[0].set_title('Original'); axes_cent[0].legend(fontsize=9)

            # Panel derecho — trayectoria centrada
            axes_cent[1].plot(_xr, _vr, color='steelblue', linewidth=1, label='centrada')
            axes_cent[1].scatter(_xr[0],  _vr[0],  color='green', s=60, zorder=5, label='inicio')
            axes_cent[1].scatter(_xr[-1], _vr[-1], color='red',   s=60, zorder=5, label='fin')
            axes_cent[1].axhline(0, color='k', linewidth=0.5, linestyle=':')
            axes_cent[1].axvline(0, color='k', linewidth=0.5, linestyle=':')
            # axes_cent[1].set_aspect(adjustable='datalim')
            axes_cent[1].set_xlabel('xr'); axes_cent[1].set_ylabel('vr')
            axes_cent[1].set_title(f'Centrado  (\u03c1={_rho:.6f})')
            axes_cent[1].legend(fontsize=9)
            plt.tight_layout()

            # Figura: fase local phi y dphi
            _phi, _dphi = compute_local_phase(_xr, _vr, t_win)
            _n_cyc_phi = abs(float(_phi[-1] - _phi[0])) / (2 * np.pi)
            _sign_dom  = np.sign(np.nanmedian(_dphi))
            _pct_inv   = 100.0 * np.sum(np.sign(_dphi[np.isfinite(_dphi)]) != _sign_dom) \
                         / np.sum(np.isfinite(_dphi)) if np.sum(np.isfinite(_dphi)) > 0 else 0.0

            fig_phi, axes_phi = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
            fig_phi.suptitle(
                f"Fase local — t_end={t_win[-1]:.2f}s  "
                f"N_ciclos\u2248{_n_cyc_phi:.2f}  dphi_inv={_pct_inv:.1f}%",
                fontsize=13,
            )
            axes_phi[0].plot(t_win, np.degrees(_phi), color='steelblue')
            axes_phi[0].set_ylabel('phi [\u00b0]')
            axes_phi[0].set_title('Fase desenvolta')
            axes_phi[0].axhline(0, color='k', linewidth=0.5, linestyle=':')

            _dphi_deg = np.degrees(_dphi)
            _colors_dphi = np.where(_dphi_deg >= 0, 'steelblue', 'red')
            for j in range(len(t_win) - 1):
                axes_phi[1].plot(t_win[j:j+2], _dphi_deg[j:j+2],
                                 color=_colors_dphi[j], linewidth=1)
            axes_phi[1].axhline(0, color='k', linewidth=0.8, linestyle='--')
            axes_phi[1].set_ylabel('d\u03c6/dt [\u00b0/s]')
            axes_phi[1].set_xlabel('t [s]')
            axes_phi[1].set_title(
                f'Velocidad angular local  (rojo = inversi\u00f3n, {_pct_inv:.1f}% del tiempo)'
            )
            plt.tight_layout()






        
        # ── Ciclos ──────────────────────────────────────────────────────
        if config.use_zero_crossing_cycles:
            # Detrend lineal de v solo para detección de cruces (opcional)
            if config.zc_detrend:
                _q_trend = np.polyval(np.polyfit(t_win, q_win, 1), t_win)
                _q_detrended = q_win - _q_trend
                
                _trend = np.polyval(np.polyfit(t_win, v_win, 1), t_win)
                _v_for_zc = v_win - _trend

                q_win  = _q_detrended  
                v_win  = _v_for_zc     
                if _do_debug:
                    ax_vel.plot(t_win, _v_for_zc, label='Velocity for ZC detection (detrended)', color='green')
                    ax_vel.legend()

                    fig_phase_detrend, ax_phase_detrend = plt.subplots(figsize=(10, 6))
                    ax_phase_detrend.plot(_q_detrended, _v_for_zc, label='Phase Space for ZC detection (q vs v_detrended)', color='purple')
                # Opción 3: v_cyc usa señal detrended → endpoints en v=0 exacto
                # Opción 2: v_cyc usa v_win original  → endpoints en v real ≠ 0
                # Opción 1: v_cyc usa v_win, endpoints hardcodeados a 0.0
                _mode = config.v_cycle_mode
                _v_cyc_src         = _v_for_zc if _mode == "detrended" else None
                _force_zero_endpts = (_mode == "zero")
            else:
                _v_for_zc          = None
                _v_cyc_src         = None
                _force_zero_endpts = True   # sin detrend → comportamiento original
            # Ciclos completos por cruces v=0 (debounce auto-calibrado)
            cycles = extract_complete_cycles(t_win, q_win, v_win,
                                             frac_min=0.4, direction="up",
                                             v_ref=_v_for_zc,
                                             v_cyc_src=_v_cyc_src,
                                             force_zero_endpoints=_force_zero_endpts)
            _cycles_count_history.append((float(t_win[-1]), len(cycles)))
            # print(f"Numero de ciclos detectados en ventana: {len(cycles)} en t={t_win[-1]:.3f}s")
        else:
            # Fallback: dividir ventana en sub-ciclos fijos de longitud T_modal
            cycles = []
            j = 0
            while j + N_win_cycle <= len(t_win):
                cycles.append((
                    t_win[j:j + N_win_cycle],
                    q_win[j:j + N_win_cycle],
                    v_win[j:j + N_win_cycle],
                ))
                j += N_win_cycle

        # ── Beta = unión de ciclos completos ──────────────────────────────
        if config.use_beta_from_cycles and config.use_zero_crossing_cycles and cycles:
            q_beta = np.concatenate([c[1] for c in cycles])
            v_beta = np.concatenate([c[2] for c in cycles])
        else:
            q_beta = q_win.copy()
            v_beta = v_win.copy()

        C_beta = _shoelace_open_contribution(q_beta, v_beta)
        
        K_beta = _closure_contribution(q_beta[0], v_beta[0], q_beta[-1], v_beta[-1])
        A_beta = _shoelace_oriented(q_beta, v_beta)
        # A_beta = abs(C_beta) + abs(K_beta)

        if _do_debug:
            fig_cycles, ax_cycles = plt.subplots(figsize=(10, 6))
            fig_cycles.suptitle(f"Phase Space Cycles (window end at t={t_win[-1]:.2f}s)")
            ax_cycles.plot(q_beta, v_beta, label='Full Window', color='gray', alpha=0.5)
            ax_cycles.plot([q_beta[-1], q_beta[0]], [v_beta[-1], v_beta[0]],
                           color='gray', alpha=0.75, linestyle='--')

        # ── Sobrantes (solo con ZC activado) ─────────────────────────────
        if config.use_zero_crossing_cycles and config.use_beta_from_cycles and cycles:
            tc_first = cycles[0][0][0]
            k_first  = int(np.searchsorted(t_win, tc_first, side='right'))
            q_ini = np.concatenate([q_win[:k_first],
                                    [float(np.interp(tc_first, t_win, q_win))]])
            v_ini = np.concatenate([v_win[:k_first], [0.0]])

            tc_last = cycles[-1][0][-1]
            k_last  = int(np.searchsorted(t_win, tc_last, side='left'))
            q_fin = np.concatenate([[float(np.interp(tc_last, t_win, q_win))],
                                     q_win[k_last:]])
            v_fin = np.concatenate([[0.0], v_win[k_last:]])
        else:
            q_ini = v_ini = q_fin = v_fin = np.array([], dtype=float)

        def _safe_C(q, v): return _shoelace_open_contribution(q, v) if len(q) >= 2 else 0.0
        def _safe_A(q, v): return _shoelace_oriented(q, v)           if len(q) >= 3 else 0.0
        def _safe_K(q, v): return _closure_contribution(q[0], v[0], q[-1], v[-1]) if len(q) >= 2 else 0.0

        C_ini = _safe_C(q_ini, v_ini);  A_ini = _safe_A(q_ini, v_ini);  K_ini = _safe_K(q_ini, v_ini)
        C_fin = _safe_C(q_fin, v_fin);  A_fin = _safe_A(q_fin, v_fin);  K_fin = _safe_K(q_fin, v_fin)

        # ── Alpha = ciclos individuales ───────────────────────────────────
        C_alpha = []
        A_alpha = []
        K_alpha = []

        for t_subwin, q_subwin, v_subwin in cycles:
            if _do_debug:
                ax_cycles.plot(q_subwin, v_subwin,
                               label=f'Cycle t={t_subwin[0]*1000:.2f} ms')
                color = ax_cycles.get_lines()[-1].get_color()
                ax_cycles.scatter(q_subwin[0],  v_subwin[0],  color=color, marker='o', s=50)
                ax_cycles.scatter(q_subwin[-1], v_subwin[-1], color=color, marker='X', s=75)
                ax_cycles.plot([q_subwin[-1], q_subwin[0]],
                               [v_subwin[-1], v_subwin[0]],
                               color=color, alpha=0.6, linestyle='--')
                ax_cycles.legend(fontsize=12)

            C_alpha.append(_shoelace_open_contribution(q_subwin, v_subwin))
            A_alpha.append(_shoelace_oriented(q_subwin, v_subwin))
            K_alpha.append(_closure_contribution(
                q_subwin[0], v_subwin[0], q_subwin[-1], v_subwin[-1]))

        C_alpha_sum = np.sum(C_alpha) if len(C_alpha) > 0 else 0.0
        diff_C = C_beta - C_alpha_sum

        A_alpha_sum = np.sum(A_alpha) if len(A_alpha) > 0 else 0.0
        diff_A = A_beta - A_alpha_sum

        K_alpha_sum = np.sum(K_alpha) if len(K_alpha) > 0 else 0.0
        diff_K = K_beta - K_alpha_sum

        # areas_list_subwin.append(float(np.sum(areas_cycle)))

        if _do_debug:
            # ── Winding number map ────────────────────────────────────────
            marge = 0.0
            xmin, xmax = q_beta.min() - marge, q_beta.max() + marge
            ymin, ymax = v_beta.min() - marge, v_beta.max() + marge

            q_beta_closed = np.concatenate([q_beta, [q_beta[0]]])
            v_beta_closed = np.concatenate([v_beta, [v_beta[0]]])

            nx = 500
            ny = 500

            xg = np.linspace(xmin, xmax, nx)
            yg = np.linspace(ymin, ymax, ny)

            W = np.zeros((ny, nx))

            for iy, yy in enumerate(yg):
                for ix, xx in enumerate(xg):
                    W[iy, ix] = winding_number_point(xx, yy, q_beta_closed, v_beta_closed)

            # W_round = np.round(W)
            W_round = np.copy(W)

            fig_multi, axes_multi = plt.subplots(figsize=(16, 12))

            levels = np.arange(W_round.min() - 0.5, W_round.max() + 1.5, 1)

            contour = axes_multi.contourf(
                xg,
                yg,
                W_round,
                levels=levels,
                alpha=0.99,
                cmap="viridis"
            )

            cbar = fig_multi.colorbar(contour, ax=axes_multi)
            cbar.set_label("multiplicite / winding number", fontsize=12)
            cbar.set_ticks(np.arange(W_round.min(), W_round.max() + 1, 1))
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

            axes_multi.plot(q_beta, v_beta, color="black", linewidth=1.4, label="trajectoire")
            axes_multi.plot(q_win, v_win, color="gray", linewidth=1.4, label="fenetre")

            axes_multi.plot(
                [q_beta_closed[-2], q_beta_closed[-1]],
                [v_beta_closed[-2], v_beta_closed[-1]],
                "--",
                color="gray",
                linewidth=2,
                label="fermeture globale"
            )

            axes_multi.scatter(q_beta_closed[-1], v_beta_closed[-1], marker="o", s=80, label="debut")
            axes_multi.scatter(q_beta_closed[-2], v_beta_closed[-2], marker="o", s=80, label="fin")

            axes_multi.set_xlabel("x")
            axes_multi.set_ylabel("v")
            axes_multi.set_title("5. Multiplicite", fontsize=14)
            axes_multi.legend()
            plt.tight_layout()

            fig_bar, axes_bar = plt.subplots(2, 3, figsize=(24, 14))

            for ax, labels, values, title in [
                (axes_bar[0, 0],
                 ["C b\nouv.", "SUM C a\nouv.", "Cb - SCa"],
                 [C_beta, C_alpha_sum, diff_C],
                 "Ouvertes : b vs Sa"),
                (axes_bar[0, 1],
                 ["A b\nferm.", "SUM A a\nferm.", "Ab - SAa"],
                 [A_beta, A_alpha_sum, diff_A],
                 "Fermees : b vs Sa"),
                (axes_bar[0, 2],
                 ["K b", "SUM K a", "Kb - SKa"],
                 [K_beta, K_alpha_sum, diff_K],
                 "Fermetures : b vs Sa"),
            ]:
                bars = ax.bar(labels, values)
                ax.set_xticklabels(labels, rotation=0, fontsize=10)
                ax.tick_params(axis='y', labelsize=12)
                ax.set_title(title, fontsize=12)
                ax.set_ylabel("valeur orientee", fontsize=10)
                ax.axhline(0, linewidth=0.8)
                _ajouter_valeurs_barres(ax, bars)

            for ax, labels, values, title in [
                (axes_bar[1, 0],
                 ["C ini", "C fin", "C ini+fin"],
                 [C_ini, C_fin, C_ini + C_fin],
                 "Ouvertes sobrantes"),
                (axes_bar[1, 1],
                 ["A ini", "A fin", "A ini+fin"],
                 [A_ini, A_fin, A_ini + A_fin],
                 "Fermees sobrantes"),
                (axes_bar[1, 2],
                 ["K ini", "K fin", "K ini+fin"],
                 [K_ini, K_fin, K_ini + K_fin],
                 "Fermetures sobrantes"),
            ]:
                bars = ax.bar(labels, values)
                ax.set_xticklabels(labels, rotation=0, fontsize=10)
                ax.tick_params(axis='y', labelsize=12)
                ax.set_title(title, fontsize=12)
                ax.set_ylabel("valeur orientee", fontsize=10)
                ax.axhline(0, linewidth=0.8)
                _ajouter_valeurs_barres(ax, bars)

            fig_bar.suptitle(
                "Resume : b (ciclos completos), a (cada ciclo) y sobrantes",
                fontsize=14
            )
            plt.tight_layout()
            plt.show()

        # ── Normalización por número de ciclos ───────────────────────────
        _n_cyc = len(cycles) if cycles else 0
        _norm  = config.cycle_area_norm
        if _norm == "none" or _n_cyc == 0:
            _area_out = abs(A_beta)
            _C_out    = C_beta
            _K_out    = K_beta
        elif _norm == "mean":
            _area_out = abs(A_beta) / _n_cyc
            _C_out    = abs(C_beta) / _n_cyc
            _K_out    = K_beta / _n_cyc
        elif _norm == "median":
            _area_out = float(np.median([abs(_shoelace_oriented(c[1], c[2])) for c in cycles]))
            _C_out    = float(np.median([abs(_shoelace_open_contribution(c[1], c[2])) for c in cycles]))
            _K_out    = float(np.median([abs(_closure_contribution(c[1][0], c[2][0], c[1][-1], c[2][-1])) for c in cycles]))
        else:
            raise ValueError(f"cycle_area_norm must be 'none', 'mean' or 'median', got {_norm!r}")

        areas_list.append(_area_out)
        C_list.append(_C_out)
        K_list.append(_K_out)
        t_wins_list.append(float(t_win[-1]))

        i += step

    areas  = np.array(areas_list,  dtype=float)
    t_wins = np.array(t_wins_list, dtype=float)
    trayectory_C = np.array(C_list, dtype=float)
    trayectory_K = np.array(K_list, dtype=float)

    # Replace sub-noise-floor areas with NaN so downstream consumers
    # (HMM, sigma estimator) treat them as missing rather than near-zero.
    below_floor = ~np.isfinite(areas) | (areas <= config.area_noise_eps)
    areas[below_floor] = np.nan

    n_valid = int(np.sum(np.isfinite(areas)))
    logger.info_plus(
        "Fixed-Window: %d windows computed, %d valid (area > eps=%.2e)",
        len(areas), n_valid, config.area_noise_eps,
    )

    # ---- 2. Lyapunov exponent σ̂ -------------------------------------------
    sigma = _estimate_sigma(
        areas, t_wins, T_win,
        eps=config.area_noise_eps,
        method=config.sigma_method,
        local_n=config.sigma_local_n,
    )

    # ---- 3. Optional EWMA smoothing ----------------------------------------
    if config.lambda_ewma is not None:
        sigma_ewma = _apply_ewma(sigma, float(config.lambda_ewma))
        logger.info_plus("Fixed-Window: EWMA applied (λ=%.3f).", config.lambda_ewma)
    else:
        sigma_ewma = sigma.copy()

    # ---- 4a. Optional Ĝ accumulation (from t=0) ----------------------------
    if config.accumulate:
        G_hat = _integrate_G(sigma_ewma, t_wins)
        logger.info_plus(
            "Fixed-Window: Ĝ_final = %.4f  (%s)",
            float(G_hat[-1]) if len(G_hat) else float("nan"),
            "CHATTER" if len(G_hat) and G_hat[-1] > 0 else "stable",
        )
    else:
        G_hat = np.array([], dtype=float)

    # ---- 4b. Optional sliding-window Ĝ -------------------------------------
    if config.G_memory is not None:
        G_hat_sliding = _integrate_G_sliding(sigma_ewma, t_wins, float(config.G_memory))
        logger.info_plus(
            "Fixed-Window: Ĝ_sliding_final = %.4f  (T_memory=%.3f s, %s)",
            float(G_hat_sliding[-1]) if len(G_hat_sliding) else float("nan"),
            float(config.G_memory),
            "CHATTER" if len(G_hat_sliding) and G_hat_sliding[-1] > 0 else "stable",
        )
    else:
        G_hat_sliding = np.array([], dtype=float)

    # ---- 5. Pack result ----------------------------------------------------
    area_mu_3sigma: Dict[str, Any] = {}
    t_d_detected: Optional[float] = None
    mu_log    = None
    sigma_log = None
    upper_log = None
    lower_log = None

    # Only compute μ±zσ area threshold when the user explicitly enabled
    # `use_area_threshold` AND provided `training_intervals`. Skip automatic
    # fallback selection of stable windows when training_intervals is None.
    if config.use_area_threshold and config.training_intervals is not None:
        stab = _select_stable_mask(
            t_wins, config.training_intervals,
            config.stable_time, config.frac_stable,
        )
        valid_mask = np.isfinite(areas) & (areas > config.area_noise_eps)
        stab_valid = stab & valid_mask
        if stab_valid.sum() >= 3:
            # Work in log10 space — areas are approximately log-normal
            log10_stab = np.log10(areas[stab_valid])
            mu_log    = float(np.mean(log10_stab))
            sigma_log = float(np.std(log10_stab, ddof=1))
            upper_log = mu_log + config.z_sigma * sigma_log
            lower_log = mu_log - config.z_sigma * sigma_log

            # log10_stab = areas[stab_valid]
            # mu_log    = float(np.mean(log10_stab))
            # sigma_log = float(np.std(log10_stab, ddof=1))
            # upper_log = mu_log + config.z_sigma * sigma_log
            # lower_log = mu_log - config.z_sigma * sigma_log


            area_mu_3sigma = {
                "mu": mu_log, "sigma": sigma_log,
                "upper": upper_log, "lower": lower_log, "z": config.z_sigma,
            }
            # detection in linear space: area > 10^upper_log
            det_idx = np.where( valid_mask & (areas > 10 ** upper_log))[0]
            # det_idx = np.where(~stab & valid_mask & (areas > upper_log))[0]
            if det_idx.size > 0:
                t_d_detected = np.float64(t_wins[det_idx])
            logger.info_plus(
                "Fixed-Window area threshold (log10): mu=%.4g, sigma=%.4g, upper=%.4g",
                mu_log, sigma_log, upper_log,
            )
        else:
            logger.warning(
                "Fixed-Window area threshold: not enough stable windows (%d < 3), skipped.",
                stab_valid.sum(),
            )

    global_data: Dict[str, Any] = {
        "q_signal":           q.tolist(),
        "q_o_signal":         q_o.tolist(),
        "t":                  t.tolist(),
        "type_signal":        "FixedWindow",
        "type_method":        "FixedWindow",
        "area_mu_3sigma":     area_mu_3sigma,
        "training_intervals": list(config.training_intervals) if config.training_intervals else None,
        "use_area_threshold": bool(config.use_area_threshold),
    }
    t_d_detected_no_FAR_idx =  np.where(t_d_detected >= config.t_theorical)[0] if t_d_detected is not None else np.array([], dtype=int)
    td_detected_no_FAR = t_d_detected[t_d_detected_no_FAR_idx] if t_d_detected is not None else None



    return FixedWindowResult(
        t_wins=t_wins,
        areas=areas,
        trayectory_C=trayectory_C,
        trayectory_K=trayectory_K,
        sigma=sigma,
        sigma_ewma=sigma_ewma,
        G_hat=G_hat,
        G_hat_sliding=G_hat_sliding,
        global_data=global_data,
        Name=signal.name,
        t_d=t_d_detected,
        t_d_no_FAR=td_detected_no_FAR,
        mu_log=mu_log,
        sigma_log=sigma_log,
        upper_log=upper_log,
        lower_log=lower_log,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_DEFAULT_FW_PARAMS: Dict[str, Any] = {
    "num_T":              6,
    "dt":                 None,
    "data_filtrated":     True,
    "lambda_ewma":        None,
    "accumulate":         None,
    "G_memory":           None,
    "sigma_method":       "ratio",
    "sigma_local_n":      5,
    "area_noise_eps":     1e-30,
    "use_area_threshold": False,
    "training_intervals": None,
    "frac_stable":        0.30,
    "stable_time":        None,
    "z_sigma":            3.0,
    "debug_level":        0,
    "debug_window_range": (0.0, None),
    "t_theorical":       None,
    "use_beta_from_cycles":    True,
    "use_zero_crossing_cycles": True,
    "zc_detrend":              True,
    "v_cycle_mode":            "zero",  # "zero" | "original" | "detrended"
    "cycle_area_norm":         "none",  # "none" | "mean" | "median"
    "center_win":              0,        # half-width [samples] for slow-centre estimate
}

FIXED_WINDOW_CONFIG: Dict[str, Any] = {
    "func":   "FixedWindow",
    "params": _DEFAULT_FW_PARAMS,
}


def run_fixed_window(
    signal: SignalData,
    config: Dict[str, Any],
) -> FixedWindowResult:
    """Run the Fixed-Window Lyapunov chatter indicator.

    Parameters
    ----------
    signal : :class:`~green_integral.utils.types.SignalData` input.
    config : dict with keys ``"func"`` (ignored) and ``"params"``
        (merged on top of defaults).  Alternatively, pass a
        :class:`~green_integral.utils.types.FixedWindowConfig` directly.

    Returns
    -------
    :class:`~green_integral.utils.types.FixedWindowResult`
    """
    if isinstance(config, FixedWindowConfig):
        cfg = config
    else:
        params = config.get("params", {})
        merged = {**_DEFAULT_FW_PARAMS, **params}
        f_modal = merged.pop("f_modal", None)
        if f_modal is None:
            raise ValueError("run_fixed_window: 'f_modal' is required in params.")
        cfg = FixedWindowConfig(f_modal=f_modal, **{
            k: merged[k] for k in merged if k in FixedWindowConfig.__dataclass_fields__
        })

    return _fixed_window_pipeline(signal, cfg)
