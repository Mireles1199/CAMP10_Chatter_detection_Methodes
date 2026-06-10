"""Data containers for the green_integral indicator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

@dataclass
class SignalData:
    """Input signal container.

    Attributes
    ----------
    t : array of time stamps [s].
    displacement : displacement signal q(t) [m].
    velocity : velocity signal dq/dt(t) [m/s].
    fs : sampling frequency [Hz]. Inferred from *t* when not supplied.
    name : short identifier used in plot titles and exported filenames.
    """
    t: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    name: str = "signal"
    fs: Optional[float] = None

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        self.displacement = np.asarray(self.displacement, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)
        if self.fs is None:
            dt = float(self.t[1] - self.t[0])
            self.fs = 1.0 / dt


@dataclass
class GreenIntegralConfig:
    """Configuration for the Green Integral chatter indicator.

    Parameters
    ----------
    f_modal : Modal frequency of the dominant chatter mode [Hz]. **Required.**
    num_T : Number of modal periods per analysis window.
    dt : Window step (time increment between consecutive window starts) [s].
    data_filtrated : Apply Savitzky–Golay pre-filtering inside each window.
    hilbert : Use Hilbert-transform zero-crossings instead of simple sign-change.
    while_loop_extend : Extend the window if fewer than *num_T* cycles are found.
    cycles_cluster_points : Max index-distance for grouping crossing candidates.
        ``None`` disables clustering.
    thein_sen : Use Theil–Sen regression instead of consecutive log-ratios for
        computing delta_n.
    debug_level : 0 = off, 1 = minimal (key events), 2 = full (per-window).
    debug_window_range : (min, max) window indices for detailed debug output.
    save_figures_windows : Save per-window figures to *work_space*.
    work_space : Base directory for saved figures and exported files.
        Required when *save_figures_windows* is True.
    """
    # --- required ---
    f_modal: float

    # --- windowing ---
    num_T: int = 6
    dt: float = 1e-2

    # --- signal processing ---
    data_filtrated: bool = True
    hilbert: bool = False
    while_loop_extend: bool = False
    cycles_cluster_points: Optional[int] = None
    thein_sen: bool = False

    # --- mu ± 3σ area threshold ---
    use_area_threshold: bool = False
    training_intervals: Optional[List[Tuple[float, float, str]]] = None
    frac_stable: float = 0.30
    stable_time: Optional[Tuple[float, float]] = None
    z_sigma: float = 3.0

    # --- debug / output ---
    debug_level: int = 0
    debug_window_range: Tuple[int, Optional[int]] = (0, None)
    save_figures_windows: bool = False
    work_space: Optional[str] = None

    def __post_init__(self) -> None:
        self.T_modal: float = 1.0 / self.f_modal


# ---------------------------------------------------------------------------
# Per-window result (internal dict schema — exposed here for type hints)
# ---------------------------------------------------------------------------

WindowResult = Dict[str, Any]
"""Dict with keys:
    window_number, start_time, end_time, window_duration,
    num_processed_data, exp_fit_times, exp_fit_values,
    window_times, window_q_signal, window_q_o_signal,
    cycle_key_agrupation, time_key_agrupation,
    indicadores (dict with t_n, delta_n),
    centers_x, centers_v,
    median_area, geometric_mean_area, center_area_value, t_centre.
"""


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class GreenIntegralResult:
    """Detection output returned by :func:`run_green_integral`.

    Attributes
    ----------
    data_window : List of per-window result dicts.
    agrupamiento : Cross-window cycle accumulation dict.
    Mediana_delta_n : Median of per-window delta_n values (scalar summary).
    global_data : Dict with global signal arrays and metadata.
    Name : Signal/run name (from ``SignalData.name``).
    """
    data_window: List[WindowResult]
    agrupamiento: Dict[int, Any]
    Mediana_delta_n: float
    global_data: Dict[str, Any]
    Name: str
    t_d: Optional[float] = None


# ---------------------------------------------------------------------------
# Fixed-window indicator (no zero-crossing, no clustering)
# ---------------------------------------------------------------------------

@dataclass
class FixedWindowConfig:
    """Configuration for the Fixed-Window Lyapunov chatter indicator.

    Parameters
    ----------
    f_modal : Modal frequency [Hz]. **Required.**
    num_T : Number of modal periods per window.
    dt : Time step between consecutive window starts [s].
        If ``None``, windows are non-overlapping (step = window duration).
    data_filtrated : Apply Savitzky–Golay pre-filtering inside each window.
    lambda_ewma : EWMA smoothing parameter λ ∈ (0, 1].
        ``None`` disables smoothing (sigma_ewma = sigma).
    accumulate : When ``True``, compute the accumulated indicator
        Ĝ = ∫ σ̂_EWMA dt.  ``None`` is treated as ``False``.
    sigma_method : ``"ratio"`` (consecutive log-ratios) or
        ``"frozen_time"`` (local linear fit of ln A vs t).
    sigma_local_n : Half-neighbourhood size for frozen-time mode.
    area_noise_eps : Minimum valid area threshold (windows below this
        are treated as NaN).
    debug_level : 0 = off, 1 = INFO checkpoints.
    """
    # --- required ---
    f_modal: float

    # --- windowing ---
    num_T: int = 6
    dt: Optional[float] = None

    # --- signal processing ---
    data_filtrated: bool = True

    # --- EWMA / accumulation ---
    lambda_ewma: Optional[float] = None
    accumulate: Optional[bool] = None
    G_memory: Optional[float] = None  # sliding-window memory [s]; None = disabled

    # --- Lyapunov estimator ---
    sigma_method: str = "ratio"
    sigma_local_n: int = 5

    # --- noise floor ---
    area_noise_eps: float = 1e-30

    # --- mu ± 3σ area threshold ---
    use_area_threshold: bool = False
    training_intervals: Optional[List[Tuple[float, float, str]]] = None
    frac_stable: float = 0.30
    stable_time: Optional[Tuple[float, float]] = None
    z_sigma: float = 3.0

    # --- debug ---
    debug_level: int = 0

    t_theorical: Optional[float] = None  # for debug/plots, not used in detection

    def __post_init__(self) -> None:
        self.T_modal: float = 1.0 / self.f_modal
        self.T_window: float = self.num_T * self.T_modal


@dataclass
class FixedWindowResult:
    """Output of :func:`run_fixed_window`.

    Attributes
    ----------
    t_wins : Window start times [s].
    areas : Shoelace area per window [m·m/s].
    sigma : Raw instantaneous Lyapunov exponent σ̂ [1/s].
    sigma_ewma : EWMA-smoothed σ̂ (equals *sigma* if lambda_ewma is None).
    G_hat : Accumulated indicator Ĝ = ∫σ̂_EWMA dt (from t=0).
        Empty array if *accumulate* is False/None.
    G_hat_sliding : Sliding-window Ĝ = ∫_{t-T_mem}^{t} σ̂_EWMA dt.
        Tracks current state; detects recovery.
        Empty array if *G_memory* is None.
    global_data : Dict with raw signal arrays and metadata.
    Name : Signal/run identifier.
    """
    t_wins: np.ndarray
    areas: np.ndarray
    sigma: np.ndarray
    sigma_ewma: np.ndarray
    G_hat: np.ndarray
    G_hat_sliding: np.ndarray
    global_data: Dict[str, Any]
    Name: str
    t_d: Optional[float] = None
    t_d_no_FAR: Optional[float] = None
    mu_log: Optional[float] = None
    sigma_log: Optional[float] = None
    upper_log: Optional[float] = None
    lower_log: Optional[float] = None


# ---------------------------------------------------------------------------
# Standard interface (compatible with other CAMP10 indicators)
# ---------------------------------------------------------------------------

@dataclass
class StdSignalData:
    """Standard signal container compatible with other CAMP10 indicators.

    This mirrors the ``SignalData`` of MaxEnt-SPRT so that ``run_green_std``
    can be driven by the same input objects used for maxent, rms_cv, and ssq.

    ``signal_analysis`` is interpreted as the **displacement** signal.
    Velocity is taken from ``meta["velocity"]`` when supplied; otherwise it is
    estimated via central-difference differentiation.
    """
    t_analysis: np.ndarray
    signal_analysis: np.ndarray
    path: str
    fs: float
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t_analysis = np.asarray(self.t_analysis, dtype=float)
        self.signal_analysis = np.asarray(self.signal_analysis, dtype=float)


@dataclass
class IndicatorResult:
    """Standard result compatible with other CAMP10 indicators.

    Mirrors the ``IndicatorResult`` of MaxEnt-SPRT so downstream analysis
    (``doe_noise_indicators.py``, plotters) can handle all indicators uniformly.
    """
    name: str
    """Human-readable identifier of the indicator/variant that produced this result."""
    t: np.ndarray
    """Time axis for the indicator trajectory."""
    I_t: np.ndarray
    """Indicator values along ``t``."""
    t_d: Optional[float] = None
    """Detection time [s]; ``None`` when not detected."""
    t_d_no_FAR: Optional[float] = None
    """Detection time without false alarms [s]; ``None`` when not detected."""
    meta: Dict[str, Any] = field(default_factory=dict)
    """Auxiliary artifacts (raw result, resolved config, resolver trace, etc.)."""
