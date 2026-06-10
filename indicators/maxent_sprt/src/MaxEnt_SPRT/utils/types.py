from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


# =========================
# 1) Datos de entrada comunes
# =========================
@dataclass
class SignalData:
    """
    Container for the signal array, its analysis timeline, and related metadata.

    This structure is the main input type for the MaxEnt-SPRT workflow.
    It bundles the analysis-ready signal and timeline together with source path,
    sampling frequency, and optional context metadata so pipelines can remain
    function-oriented without passing many independent arguments.
    """

    # t_cut: np.ndarray
    # v_cut: np.ndarray
    # x_cut: np.ndarray
    # force_cut: np.ndarray
    # t_original: np.ndarray
    # x_original: np.ndarray
    # v_original: np.ndarray
    t_analysis: np.ndarray
    """Time axis associated with ``signal_analysis``."""
    signal_analysis: np.ndarray
    """Analysis-ready 1D signal used by OPR sampling, segmentation, entropy extraction, and sequential detection."""
    # force_original: np.ndarray
    path: str
    """Source identifier of the signal (typically a file path) used for traceability and experiment bookkeeping."""
    fs: float
    """Sampling frequency in Hz of ``signal_analysis``."""
    meta: Dict[str, Any] = field(default_factory=dict)
    """Flexible metadata dictionary for optional context (machine setup, test labels, channel info, units, notes, etc.)."""



# =========================
# 2) Standard indicator result
# =========================
@dataclass
class IndicatorResult:
    """
    Result of an indicator calculation.

    Stores the computed time axis, indicator history, detection time information,
    and any additional metadata required for analysis or plotting.

    The ``meta`` dictionary is intentionally flexible and can include trained
    models, intermediate signals, thresholds, or references to detector objects
    needed for reproducibility and debugging.
    """

    name: str
    """Human-readable identifier of the indicator or method that produced this result."""
    t: np.ndarray
    """Time axis for the computed indicator trajectory."""
    I_t: np.ndarray
    """Indicator values evaluated along ``t``."""
    t_d: Optional[float] = None
    """Detection timestamp in seconds when available; ``None`` when no detection time is defined."""
    t_d_no_FAR: Optional[np.ndarray] = None
    """Detection timestamps in seconds when available and above the theoretical threshold; ``None`` when no detection time is defined."""
    meta: Dict[str, Any] = field(default_factory=dict)
    """Auxiliary artifacts required for analysis, visualization, reproducibility, or post-hoc debugging."""


# =========================
# 3) Scenario metadata (optional)
# =========================
@dataclass
class ScenarioMetadata:
    """
    Metadata attached to a simulated or experimental scenario.

    This lightweight record keeps experimental conditions (such as spindle
    speed and SNR) close to generated or measured signals, which simplifies
    reporting and batch comparisons across scenarios.
    """

    scenario_id: str
    """Unique scenario label used in logs, exports, and benchmark tables."""
    ap_ramp: Optional[tuple[float, float]] = None
    """Optional lower/upper range of axial depth-of-cut ramp (or analogous scalar progression)."""
    rpm: Optional[float] = None
    """Spindle speed in revolutions per minute."""
    snr_db: Optional[float] = None
    """Signal-to-noise ratio in dB when known."""
    extra: Dict[str, Any] = field(default_factory=dict)
    """Arbitrary extension fields for experiment-specific metadata without changing the dataclass schema."""
