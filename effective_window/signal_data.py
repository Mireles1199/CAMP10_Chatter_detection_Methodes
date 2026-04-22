"""
signal_data.py
==============
Standalone signal container for the effective_window framework.

Independent of any indicator library (MaxEnt_SPRT, rms_cv, ssq_chatter).
Each indicator adapter is responsible for converting this object into
the library-specific SignalData expected by that indicator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class SignalData:
    """
    Complete signal container for the effective-window framework.

    This is the canonical input object for :class:`~effective_window.runner.WindowRunner`.
    It carries all physical channels recorded during a milling experiment so that
    any indicator adapter can pick the channels it needs.

    Parameters
    ----------
    t_analysis : np.ndarray
        Time axis fed to the indicators (typically the cut portion).
    signal_analysis : np.ndarray
        1-D signal fed to the indicators (typically tool velocity v(t)).
    fs : float
        Acquisition sampling frequency [Hz].
    path : str
        Source identifier — absolute path to the HDF5 file or ``"synthetic"``.
    t_cut : np.ndarray, optional
        Time axis of the analysis window (same as ``t_analysis`` in most cases).
    v_cut : np.ndarray, optional
        Tool velocity in the analysis window [m/s].
    x_cut : np.ndarray, optional
        Tool displacement in the analysis window [m].
    force_cut : np.ndarray, optional
        Cutting force in the analysis window [N].
    t_original : np.ndarray, optional
        Full original time vector before any cut [s].
    x_original : np.ndarray, optional
        Full original tool displacement [m].
    v_original : np.ndarray, optional
        Full original tool velocity [m/s].
    force_original : np.ndarray, optional
        Full original cutting force [N].
    meta : dict
        Free-form metadata dict: RPM, axial depth of cut (AP), test labels, etc.
        Example: ``{"RPM": 12_000, "AP": "5mm"}``.
    """

    # ── mandatory ────────────────────────────────────────────────────────────
    t_analysis: np.ndarray
    signal_analysis: np.ndarray
    fs: float
    path: str

    # ── optional channels ────────────────────────────────────────────────────
    # t_cut: Optional[np.ndarray] = None
    # v_cut: Optional[np.ndarray] = None
    # x_cut: Optional[np.ndarray] = None
    # force_cut: Optional[np.ndarray] = None
    # t_original: Optional[np.ndarray] = None
    # x_original: Optional[np.ndarray] = None
    # v_original: Optional[np.ndarray] = None
    # force_original: Optional[np.ndarray] = None

    meta: Dict[str, Any] = field(default_factory=dict)

    # ── convenience ──────────────────────────────────────────────────────────
    def duration(self) -> float:
        """Total duration of the analysis window [s]."""
        return float(self.t_analysis[-1] - self.t_analysis[0])

    def n_samples(self) -> int:
        """Number of samples in the analysis window."""
        return len(self.t_analysis)

    def __repr__(self) -> str:
        return (
            f"SignalData(fs={self.fs:.1f} Hz, "
            f"duration={self.duration():.3f} s, "
            f"n_samples={self.n_samples()}, "
            f"path='{self.path}')"
        )
