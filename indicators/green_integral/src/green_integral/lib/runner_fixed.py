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
    dt_sig = float(t[1] - t[0])
    N_win  = max(3, int(round(T_win / dt_sig)))  # samples per window

    # step between window starts
    if config.dt is None:
        step = N_win  # non-overlapping
    else:
        step = max(1, int(round(config.dt / dt_sig)))

    logger.info("=" * 60)
    logger.info("Fixed-Window Indicator  |  signal: %s", signal.name)
    logger.info("  f_modal   = %.2f Hz  |  T_modal = %.4e s", config.f_modal, config.T_modal)
    logger.info("  num_T     = %d  |  T_window = %.4e s", config.num_T, T_win)
    logger.info("  N_win     = %d samples  |  step = %d samples", N_win, step)
    logger.info(
        "  filtered  = %s  |  lambda_ewma = %s  |  accumulate = %s",
        config.data_filtrated, config.lambda_ewma, config.accumulate,
    )
    logger.info("  sigma_method = %s", config.sigma_method)
    logger.info("=" * 60)

    # ---- 1. Build windows and compute areas --------------------------------
    areas_list: list  = []
    t_wins_list: list = []

    i = 0
    while i + N_win <= len(t):
        t_win = t[i:i + N_win]
        q_win = q[i:i + N_win]
        v_win = q_o[i:i + N_win]

        if config.data_filtrated and len(q_win) >= 7:
            q_win = savgol_filter_window(q_win)
            v_win = savgol_filter_window(v_win)

        areas_list.append(_shoelace(q_win, v_win))
        t_wins_list.append(float(t_win[0]))
        i += step

    areas  = np.array(areas_list,  dtype=float)
    t_wins = np.array(t_wins_list, dtype=float)

    # Replace sub-noise-floor areas with NaN so downstream consumers
    # (HMM, sigma estimator) treat them as missing rather than near-zero.
    below_floor = ~np.isfinite(areas) | (areas <= config.area_noise_eps)
    areas[below_floor] = np.nan

    n_valid = int(np.sum(np.isfinite(areas)))
    logger.info(
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
        logger.info("Fixed-Window: EWMA applied (λ=%.3f).", config.lambda_ewma)
    else:
        sigma_ewma = sigma.copy()

    # ---- 4a. Optional Ĝ accumulation (from t=0) ----------------------------
    if config.accumulate:
        G_hat = _integrate_G(sigma_ewma, t_wins)
        logger.info(
            "Fixed-Window: Ĝ_final = %.4f  (%s)",
            float(G_hat[-1]) if len(G_hat) else float("nan"),
            "CHATTER" if len(G_hat) and G_hat[-1] > 0 else "stable",
        )
    else:
        G_hat = np.array([], dtype=float)

    # ---- 4b. Optional sliding-window Ĝ -------------------------------------
    if config.G_memory is not None:
        G_hat_sliding = _integrate_G_sliding(sigma_ewma, t_wins, float(config.G_memory))
        logger.info(
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

    if config.use_area_threshold:
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
            area_mu_3sigma = {
                "mu": mu_log, "sigma": sigma_log,
                "upper": upper_log, "lower": lower_log, "z": config.z_sigma,
            }
            # detection in linear space: area > 10^upper_log
            det_idx = np.where(~stab & valid_mask & (areas > 10 ** upper_log))[0]
            if det_idx.size > 0:
                t_d_detected = float(t_wins[det_idx[0]])
            logger.info(
                "Fixed-Window area threshold (log10): mu=%.4g, sigma=%.4g, upper=%.4g | t_d=%s",
                mu_log, sigma_log, upper_log, t_d_detected,
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
    }

    return FixedWindowResult(
        t_wins=t_wins,
        areas=areas,
        sigma=sigma,
        sigma_ewma=sigma_ewma,
        G_hat=G_hat,
        G_hat_sliding=G_hat_sliding,
        global_data=global_data,
        Name=signal.name,
        t_d=t_d_detected,
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
