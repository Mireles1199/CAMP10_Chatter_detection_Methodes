"""Shared data container classes for the RMS-CV chatter detection pipeline.

Defines three :mod:`dataclasses <dataclasses>` used throughout the package:

* :class:`SignalData` — raw signal arrays and acquisition metadata.
* :class:`IndicatorResult` — detection output produced by the pipeline.
* :class:`ScenarioMetadata` — optional descriptor for experimental scenarios.

All classes can be passed between pipeline stages by reference and converted
to plain dictionaries via :func:`dataclasses.asdict` when needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


# =========================
# 1) Datos de entrada comunes
# =========================
@dataclass
class SignalData:
    """Input signal container shared across all indicator pipeline stages.

    Stores the raw time series arrays alongside acquisition and cut-window
    metadata.  Both the full original recording and the trimmed analysis
    segment are kept so that plots can display context outside the analysis
    range.

    Attributes:
        t_cut (np.ndarray): Time vector of the analysis window [s],
            shape ``(N,)``.
        v_cut (np.ndarray): Tool velocity in the analysis window [m/s],
            shape ``(N,)``.
        x_cut (np.ndarray): Tool displacement in the analysis window [m],
            shape ``(N,)``.
        force_cut (np.ndarray): Cutting force in the analysis window [N],
            shape ``(N,)``.
        t_original (np.ndarray): Full-length original time vector [s].
        x_original (np.ndarray): Full-length original displacement [m].
        v_original (np.ndarray): Full-length original velocity [m/s].
        t_analysis (np.ndarray): Time vector actually fed to the indicator.
            Typically equals ``t_cut`` but may be a sub-range.
        signal_analysis (np.ndarray): Signal values fed to the indicator.
            Typically equals ``v_cut``.
        force_original (np.ndarray): Full-length original force signal [N].
        path (str): Absolute path to the source ``.hdf5`` file, or the
            string ``"synthetic"`` for programmatically generated signals.
        fs (float): Acquisition sampling frequency [Hz].  Must be > 0.
        meta (Dict[str, Any]): Free-form key-value metadata
            (e.g. ``{"RPM": 12000, "AP": "5mm"}``).  Defaults to ``{}``.
    """

    t_cut: np.ndarray
    v_cut: np.ndarray
    x_cut: np.ndarray
    force_cut: np.ndarray
    t_original: np.ndarray
    x_original: np.ndarray
    v_original: np.ndarray
    t_analysis: np.ndarray
    signal_analysis: np.ndarray
    force_original: np.ndarray
    path: str
    fs: float
    meta: Dict[str, Any] = field(default_factory=dict)



# =========================
# 2) Standard indicator result
# =========================
@dataclass
class IndicatorResult:
    """Detection output returned by any indicator pipeline function.

    Carries the full CV (or other indicator) time series together with the
    first detection instant and extended metadata echoed from the pipeline.

    Attributes:
        name (str): Human-readable indicator identifier, e.g. ``"RMS_CV"``.
        t (np.ndarray): Time stamps associated with each value in ``I_t``
            [s], shape ``(F,)``.
        I_t (np.ndarray): Indicator values at each frame.  For the RMS-CV
            pipeline this is the Coefficient of Variation sequence,
            shape ``(F,)``.
        t_d (Optional[np.ndarray]): Array of times [s] where the indicator
            exceeded the detection threshold.  ``None`` if no detection
            occurred during the recording.
        meta (Dict[str, Any]): Extended pipeline outputs echoed for
            diagnostics and plotting: RMS values, CV values, window
            parameters, and alert arrays.  Defaults to ``{}``.
    """

    name: str
    t: np.ndarray
    I_t: np.ndarray
    t_d: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


# =========================
# 3) Scenario metadata (optional)
# =========================
@dataclass
class ScenarioMetadata:
    """Optional descriptor for experimental machining scenarios.

    Intended for bookkeeping when running batch experiments so that each
    :class:`IndicatorResult` can be traced back to its physical conditions.

    Attributes:
        scenario_id (str): Unique string identifier for the scenario
            (e.g. ``"exp_01_5mm_12000rpm"``).
        ap_ramp (Optional[tuple[float, float]]): Axial depth-of-cut ramp as
            ``(ap_start_mm, ap_end_mm)``.  ``None`` for constant ap.
        rpm (Optional[float]): Spindle speed [rev/min].  ``None`` if unknown.
        snr_db (Optional[float]): Signal-to-noise ratio of the recording
            [dB].  ``None`` if unmeasured.
        extra (Dict[str, Any]): Any additional key-value pairs specific to
            the experiment.  Defaults to ``{}``.
    """

    scenario_id: str
    ap_ramp: Optional[tuple[float, float]] = None
    rpm: Optional[float] = None
    snr_db: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)
