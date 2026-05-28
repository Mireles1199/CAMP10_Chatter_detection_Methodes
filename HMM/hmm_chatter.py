"""hmm_chatter.py — 2-state Hidden Markov Model chatter detector.

Post-processor over the Green Integral Fixed Window indicator.
Receives `areas` and `t_wins` from run_fixed_window(), applies a
Bayesian forward filter (Markov-chain prior + Gaussian emission),
and returns a chatter probability curve  p_k = P(C | y_{1:k})
plus a detection time t_d.

Theory
------
    Observation  : y_k = log10(max(area_k, eps))
    Hidden state : z_k ∈ {S=0, C=1}
    Emission     : y_k | z_k ~ N(μ_z, σ_z²)
    Transition   : P = [[1-p_SC, p_SC],
                        [p_CS,  1-p_CS]]
    Forward step : α̂_k = Pᵀ · α_{k-1}          (predict)
                   α_k  = b(y_k) ⊙ α̂_k / Z      (update)
    Output       : p_k  = α_k[1]  = P(C | y_{1:k})

Usage
-----
    from hmm_chatter import HMMConfig, run_hmm_detector

    config = HMMConfig(
        training_intervals=[
            (0.05,  5.366, "stable"),
            (5.366, 16.0,  "chatter"),
        ],
        p_SC=0.005, p_CS=0.002,
        rho=0.95, m_consecutive=3,
    )
    result = run_hmm_detector(areas, t_wins, config)
    print(f"t_d = {result.t_d:.4f} s")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class HMMConfig:
    """Configuration for the 2-state HMM chatter detector.

    Parameters
    ----------
    training_intervals : list of (t_start, t_end, label) tuples, optional
        Labeled time intervals.  label ∈ {"stable", "chatter"}.
        - "stable" intervals  → estimate μ_S, σ_S.
        - "chatter" intervals → estimate μ_C, σ_C  (required).
    frac_stable : float
        Fallback fraction of windows used as stable reference when
        training_intervals and stable_time are both None.
    stable_time : (t0, t1) tuple, optional
        Explicit stable interval [s].  Used when training_intervals is None.
    p_SC : float
        Per-window transition probability S → C.
    p_CS : float
        Per-window transition probability C → S.
    rho : float
        Detection threshold on p_chatter ∈ (0, 1).
    m_consecutive : int
        Number of consecutive windows with p_k > rho required to declare t_d.
    eps : float
        Floor value for log10(area): y = log10(max(area, eps)).
    """

    training_intervals: Optional[List[Tuple[float, float, str]]] = None
    frac_stable: float = 0.30
    stable_time: Optional[Tuple[float, float]] = None
    p_SC: float = 0.005
    p_CS: float = 0.002
    rho: float = 0.95
    m_consecutive: int = 3
    eps: float = 1e-8
    # ── Detection mode ────────────────────────────────────────────────────
    mode: str = "2state"
    # "2state" : classic Bayesian forward filter (needs chatter intervals)
    # "1class" : stable-only; p_k = norm.cdf(y_k, mu_S, sigma_S)
    # "auto"   : tries 2state; falls back to 1class if separation < auto_sep_min
    z_sigma_1class: float = 3.0    # threshold for 1class: mu_S + z*sigma_S
    auto_sep_min:   float = 2.0    # min (mu_C-mu_S)/sigma_S to keep 2state in auto
    # ── Observation clipping ─────────────────────────────────────────────
    y_clip_n_sigma: float = 4.0
    # Clips log10(area) from below at mu_S - y_clip_n_sigma*sigma_S.
    # Prevents very-low-area windows (eps floor) from inflating P(chatter)
    # because sigma_C >> sigma_S makes the chatter tail wider than stable's.
    # Set to None to disable clipping.
    # ── Manual transition matrix ─────────────────────────────────────────
    transition_matrix: Optional[np.ndarray] = None
    # Full 2x2 matrix P where P[i,j] = P(z_k=j | z_{k-1}=i).
    # Rows are normalised automatically.  If None, built from p_SC / p_CS.


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class HMMResult:
    """Output of run_hmm_detector().

    Attributes
    ----------
    t_wins : ndarray, shape (N,)
        Window start times [s] (same as input t_wins).
    p_chatter : ndarray, shape (N,)
        Causal chatter probability P(C | y_{1:k}) for each window.
    y_obs : ndarray, shape (N,)
        Log10-area observations: log10(max(areas, eps)).
    mu_S, sigma_S : float
        Estimated Gaussian emission parameters for the stable state.
    mu_C, sigma_C : float
        Estimated Gaussian emission parameters for the chatter state.
    t_d : float or None
        First window time where p_chatter exceeded `rho` for
        `m_consecutive` consecutive windows. None if never met.
    info : dict
        Internal metadata: transition matrix P, initial distribution
        alpha_0, separation score (μ_C − μ_S)/σ_S, window counts.
    """

    t_wins: np.ndarray
    p_chatter: np.ndarray
    p_chatter_predict: np.ndarray   # α̂_k[1] = P(C | y_{1:k-1})  before update
    y_obs: np.ndarray
    mu_S: float
    sigma_S: float
    mu_C: float
    sigma_C: float
    t_d: Optional[float]
    mode_used: str = "2state"   # mode actually selected (useful when mode="auto")
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _select_mask(
    t_wins: np.ndarray,
    training_intervals: Optional[List[Tuple[float, float, str]]],
    label: str,
    stable_time: Optional[Tuple[float, float]] = None,
    frac_stable: float = 0.30,
) -> np.ndarray:
    """Return a boolean mask selecting windows with the given label.

    Priority for "stable":
        1. training_intervals windows labelled "stable"
        2. stable_time explicit range
        3. first frac_stable fraction of windows  (fallback)

    For "chatter":
        Only training_intervals are used.  If none found → ValueError.
    """
    N = len(t_wins)

    if label == "stable":
        if training_intervals is not None:
            mask = np.zeros(N, dtype=bool)
            for t0, t1, lbl in training_intervals:
                if lbl == "stable":
                    mask |= (t_wins >= t0) & (t_wins <= t1)
            if mask.any():
                return mask

        if stable_time is not None:
            return (t_wins >= stable_time[0]) & (t_wins <= stable_time[1])

        # fallback: first frac_stable windows
        n_st = max(3, int(N * frac_stable))
        mask = np.zeros(N, dtype=bool)
        mask[:n_st] = True
        return mask

    elif label == "chatter":
        if training_intervals is None:
            raise ValueError(
                "No 'chatter' intervals available: training_intervals is None. "
                "Provide at least one (t0, t1, 'chatter') entry."
            )
        mask = np.zeros(N, dtype=bool)
        for t0, t1, lbl in training_intervals:
            if lbl == "chatter":
                mask |= (t_wins >= t0) & (t_wins <= t1)
        if not mask.any():
            raise ValueError(
                "No 'chatter' windows found in training_intervals. "
                "Check that interval times overlap with t_wins."
            )
        return mask

    else:
        raise ValueError(f"Unknown label '{label}'. Use 'stable' or 'chatter'.")


def _estimate_emission(
    y: np.ndarray,
    mask: np.ndarray,
    min_sigma: float = 0.05,
) -> Tuple[float, float]:
    """Estimate (mu, sigma) of a Gaussian emission from masked observations.

    NaN entries in *y* are ignored (they correspond to sub-eps-floor windows
    that were mapped to NaN instead of an artificial log10(eps) value).
    """
    y_sel = y[mask]
    y_sel = y_sel[np.isfinite(y_sel)]   # skip NaN / Inf (noise-floor windows)
    if len(y_sel) < 2:
        raise ValueError(
            f"Too few finite samples ({len(y_sel)}) to estimate emission parameters."
        )
    mu = float(np.mean(y_sel))
    sigma = float(np.std(y_sel, ddof=1))
    sigma = max(sigma, min_sigma)   # prevent degenerate / zero emission
    return mu, sigma


def _forward_filter(
    y: np.ndarray,
    mu_S: float,
    sigma_S: float,
    mu_C: float,
    sigma_C: float,
    P: np.ndarray,
) -> np.ndarray:
    """Bayesian forward filter for a 2-state HMM with Gaussian emissions.

    States: 0 = Stable,  1 = Chatter.

    Parameters
    ----------
    P : ndarray, shape (2, 2)
        Transition matrix: P[i, j] = P(z_k = j | z_{k-1} = i). Rows sum to 1.

    Returns
    -------
    (p_chatter, p_predict) : tuple of ndarrays, shape (N,)
    """
    N = len(y)

    # Initial distribution: start almost certainly stable
    alpha = np.array([0.999, 0.001], dtype=float)

    p_chatter = np.empty(N, dtype=float)
    p_predict = np.empty(N, dtype=float)   # α̂_k[1] before observation update

    for k in range(N):
        # ── Predict: propagate previous posterior through transition ─────
        alpha_hat = P.T @ alpha           # shape (2,)
        p_predict[k] = alpha_hat[1]       # P(C | y_{1:k-1})

        # ── Update: multiply by emission likelihood ──────────────────────
        b_S = float(norm.pdf(y[k], mu_S, sigma_S))
        b_C = float(norm.pdf(y[k], mu_C, sigma_C))
        b = np.array([b_S, b_C])

        alpha_new = b * alpha_hat

        # ── Normalise (guard against numerical underflow) ────────────────
        s = alpha_new.sum()
        if s > 0.0:
            alpha = alpha_new / s
        else:
            # Both likelihoods ≈ 0: keep the prior (predict-only step)
            alpha = alpha_hat / alpha_hat.sum()

        p_chatter[k] = alpha[1]

    return p_chatter, p_predict


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_hmm_detector(
    areas: np.ndarray,
    t_wins: np.ndarray,
    config: HMMConfig,
) -> HMMResult:
    """Run the 2-state HMM chatter detector on phase-space areas.

    Parameters
    ----------
    areas : ndarray, shape (N,)
        Shoelace phase-space area per window  (output of run_fixed_window).
    t_wins : ndarray, shape (N,)
        Window start times [s]  (output of run_fixed_window).
    config : HMMConfig
        Detector configuration.

    Returns
    -------
    HMMResult
    """
    areas = np.asarray(areas, dtype=float)
    t_wins = np.asarray(t_wins, dtype=float)

    if areas.shape != t_wins.shape:
        raise ValueError(
            f"areas (len={len(areas)}) and t_wins (len={len(t_wins)}) "
            "must have the same length."
        )

    # ── 1. Log10-area observations ──────────────────────────────────────────
    # Areas below the noise floor (config.eps) are mapped to NaN so they are
    # excluded from emission training and treated as missing in the filter.
    y = np.where(areas > config.eps, np.log10(areas), np.nan)

    # ── 2. Stable emission parameters (always needed) ───────────────────────
    stable_mask = _select_mask(
        t_wins,
        config.training_intervals,
        "stable",
        stable_time=config.stable_time,
        frac_stable=config.frac_stable,
    )
    mu_S, sigma_S = _estimate_emission(y, stable_mask)

    # ── 3. Determine effective mode ──────────────────────────────────────────
    if config.mode == "1class":
        mode_used = "1class"
    elif config.mode == "2state":
        mode_used = "2state"
    else:  # "auto"
        try:
            _cm_test = _select_mask(t_wins, config.training_intervals, "chatter")
            _mu_C_test, _ = _estimate_emission(y, _cm_test)
            _sep_test = (_mu_C_test - mu_S) / sigma_S if sigma_S > 0 else 0.0
            mode_used = "2state" if _sep_test >= config.auto_sep_min else "1class"
        except (ValueError, Exception):
            mode_used = "1class"  # no chatter intervals or poor separation

    # ── 4. Chatter emission + filter (mode-dependent) ────────────────────────
    if mode_used == "1class":
        # Stable-only: p_k = CDF(y_k | N(mu_S, sigma_S))
        # Detection fires when p_k > rho  <=>  y_k > norm.ppf(rho)*sigma_S + mu_S
        # mu_C stored as the z_sigma threshold level (for plotting)
        mu_C              = mu_S + config.z_sigma_1class * sigma_S
        sigma_C           = sigma_S
        p_chatter         = norm.cdf(y, mu_S, sigma_S)
        p_chatter_predict = p_chatter.copy()   # no transition model in 1class
        chatter_mask      = y > mu_C
    else:
        # Classic 2-state HMM

        # ── Build transition matrix ──────────────────────────────────────
        if config.transition_matrix is not None:
            _P = np.asarray(config.transition_matrix, dtype=float)
            if _P.shape != (2, 2):
                raise ValueError("transition_matrix must be shape (2, 2).")
            _P = _P / _P.sum(axis=1, keepdims=True)   # normalise rows
        else:
            _P = np.array([[1 - config.p_SC, config.p_SC],
                           [config.p_CS,    1 - config.p_CS]], dtype=float)

        chatter_mask = _select_mask(
            t_wins,
            config.training_intervals,
            "chatter",
        )
        mu_C, sigma_C = _estimate_emission(y, chatter_mask)

        # ── Clip y from below: prevents eps-floor areas from inflating P(C)
        # Root cause: sigma_C >> sigma_S makes the wide chatter tail assign
        # higher likelihood to very-low observations than the tight stable tail.
        if config.y_clip_n_sigma is not None:
            _y_filt = np.clip(y, mu_S - config.y_clip_n_sigma * sigma_S, None)
        else:
            _y_filt = y

        p_chatter, p_chatter_predict = _forward_filter(
            _y_filt, mu_S, sigma_S, mu_C, sigma_C, _P
        )

    # ── 5. Detection: first run of m_consecutive windows above rho ──────────
    t_d: Optional[float] = None
    m = config.m_consecutive
    N = len(p_chatter)
    for k in range(N - m + 1):
        if np.all(p_chatter[k : k + m] > config.rho):
            t_d = float(t_wins[k])
            break

    # ── 6. Pack result ───────────────────────────────────────────────────────
    separation = (mu_C - mu_S) / sigma_S if sigma_S > 0 else float("nan")
    # Transition matrix actually used (build from p_SC/p_CS if not 2state or no override)
    if mode_used == "2state":
        _P_info = _P
    else:
        _P_info = np.array([[1 - config.p_SC, config.p_SC],
                            [config.p_CS,    1 - config.p_CS]], dtype=float)
    info = {
        "P": _P_info,
        "alpha_0": np.array([0.999, 0.001]),
        "separation_z": separation,
        "n_stable": int(stable_mask.sum()),
        "n_chatter": int(chatter_mask.sum()),
        "stable_mask": stable_mask,
        "chatter_mask": chatter_mask,
        "mode_used": mode_used,
    }

    return HMMResult(
        t_wins=t_wins,
        p_chatter=p_chatter,
        p_chatter_predict=p_chatter_predict,
        y_obs=y,
        mu_S=mu_S,
        sigma_S=sigma_S,
        mu_C=mu_C,
        sigma_C=sigma_C,
        t_d=t_d,
        mode_used=mode_used,
        info=info,
    )
