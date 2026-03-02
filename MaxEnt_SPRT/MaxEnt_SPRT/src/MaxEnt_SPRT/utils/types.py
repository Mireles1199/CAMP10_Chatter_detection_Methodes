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
    A data class for storing signal data and its analysis results.
    Attributes:
        t_cut (np.ndarray): Time array of the cut signal segment.
        v_cut (np.ndarray): Velocity array of the cut signal segment.
        x_cut (np.ndarray): Position array of the cut signal segment.
        force_cut (np.ndarray): Force array of the cut signal segment.
        t_original (np.ndarray): Time array of the original signal.
        x_original (np.ndarray): Position array of the original signal.
        v_original (np.ndarray): Velocity array of the original signal.
        t_analysis (np.ndarray): Time array used for analysis.
        signal_analysis (np.ndarray): Processed signal array for analysis.
        force_original (np.ndarray): Force array of the original signal.
        path (str): File path to the signal data source.
        fs (float): Sampling frequency of the signal in Hz.
        meta (Dict[str, Any]): Metadata dictionary containing additional information about the signal.
            Defaults to an empty dictionary.
    """

    # t_cut: np.ndarray
    # v_cut: np.ndarray
    # x_cut: np.ndarray
    # force_cut: np.ndarray
    # t_original: np.ndarray
    # x_original: np.ndarray
    # v_original: np.ndarray
    t_analysis: np.ndarray
    signal_analysis: np.ndarray
    # force_original: np.ndarray
    path: str
    fs: float
    meta: Dict[str, Any] = field(default_factory=dict)



# =========================
# 2) Standard indicator result
# =========================
@dataclass
class IndicatorResult:
    """
    Data class representing the result of an indicator calculation.
    Attributes:
        name (str): The name of the indicator.
        t (np.ndarray): Time values or time points associated with the indicator.
        I_t (np.ndarray): Indicator values at corresponding time points.
        t_d (Optional[float]): Detection time or threshold time, if applicable. Defaults to None.
        meta (Dict[str, Any]): Dictionary containing metadata or additional information about the indicator result. Defaults to an empty dictionary.
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
    """
    Metadata container for experimental scenarios.
    Attributes:
        scenario_id (str): Unique identifier for the scenario.
        ap_ramp (Optional[tuple[float, float]]): Acceleration/power ramp parameters
            specified as a tuple of (start, end) values. Defaults to None.
        rpm (Optional[float]): Revolutions per minute value. Defaults to None.
        snr_db (Optional[float]): Signal-to-noise ratio in decibels. Defaults to None.
        extra (Dict[str, Any]): Additional metadata as key-value pairs.
            Defaults to an empty dictionary.
    """

    scenario_id: str
    ap_ramp: Optional[tuple[float, float]] = None
    rpm: Optional[float] = None
    snr_db: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)
