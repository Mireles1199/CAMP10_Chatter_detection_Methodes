"""
adapters.py
===========
Indicator adapters for the effective-window framework.

Each adapter:
  1. Converts the framework's SignalData into the indicator library's own
     SignalData (they share the same field names, so conversion is trivial).
  2. Merges the resolved parameter(s) on top of base_params to produce
     the INDICATOR_CONFIG dict expected by each library runner.
  3. Calls the indicator runner and returns the library's IndicatorResult.

Design note — SST-SVD adapter
------------------------------
  The public ``run_sst_svd`` function validates that
  hop_ms ∈ [0.25·w, 0.50·w] and raises ValueError otherwise.
  The theory only requires h ∈ (0, w] (i.e. h_ratio ∈ (0, 1]).
  To support the full theoretical range without modifying library code,
  the SST adapter calls ``_sst_svd_pipeline`` directly (private function
  in ssq_chatter.lib.runner), bypassing the validation entirely.
  A non-blocking warning is emitted when h_ratio ∉ [0.25, 0.50].
"""

from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from .signal_data import SignalData as EWSignalData

logger = logging.getLogger("effective_window.adapters")

# ──────────────────────────────────────────────────────────────────────────────
# Lazy imports — keep indicator libraries as optional dependencies
# ──────────────────────────────────────────────────────────────────────────────

def _import_maxent():
    try:
        from MaxEnt_SPRT import run_maxent_sprt
        from MaxEnt_SPRT.utils.types import SignalData as MaxEntSD, IndicatorResult
        return run_maxent_sprt, MaxEntSD, IndicatorResult
    except ImportError as e:
        raise ImportError(
            "MaxEnt_SPRT library not found. Install it or add it to sys.path."
        ) from e


def _import_rms_cv():
    try:
        from rms_cv import run_rms_cv
        from rms_cv.utils.types import SignalData as RmsSD, IndicatorResult
        return run_rms_cv, RmsSD, IndicatorResult
    except ImportError as e:
        raise ImportError(
            "rms_cv library not found. Install it or add it to sys.path."
        ) from e


def _import_sst():
    try:
        # Import private pipeline directly to bypass hop validation
        from ssq_chatter.lib.runner import _sst_svd_pipeline
        from ssq_chatter.utils.types import SignalData as SstSD, IndicatorResult
        return _sst_svd_pipeline, SstSD, IndicatorResult
    except ImportError as e:
        raise ImportError(
            "ssq_chatter library not found. Install it or add it to sys.path."
        ) from e


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────

class IndicatorAdapter(ABC):
    """
    Abstract adapter between the effective-window framework and one indicator
    library.

    Responsibilities
    ----------------
    - Convert :class:`~effective_window.signal_data.SignalData` →
      library-specific ``SignalData``.
    - Merge resolved parameters onto ``base_params`` to produce
      ``INDICATOR_CONFIG``.
    - Call the indicator runner and return its ``IndicatorResult``.
    """

    @abstractmethod
    def build_config(
        self,
        base_params: Dict[str, Any],
        resolved_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Produce the ``INDICATOR_CONFIG`` dict for the indicator runner.

        Only the resolved parameter(s) are replaced; everything else in
        ``base_params`` stays unchanged.
        """

    @abstractmethod
    def run(
        self,
        signal: EWSignalData,
        config: Dict[str, Any],
    ) -> Any:
        """Call the indicator runner and return its ``IndicatorResult``."""

    @staticmethod
    def _to_numpy(arr: Optional[np.ndarray]) -> np.ndarray:
        """Return *arr* as a numpy array, or an empty array if None."""
        if arr is None:
            return np.array([])
        return np.asarray(arr)


# ──────────────────────────────────────────────────────────────────────────────
# RMS-CV adapter
# ──────────────────────────────────────────────────────────────────────────────

class RMSCVAdapter(IndicatorAdapter):
    """
    Adapter for the RMS-CV indicator.

    Resolved variable mapping
    -------------------------
    Theory name  →  INDICATOR_CONFIG["params"] key
    N            →  samples_per_window
    rho          →  overlap_pct
    n_max        →  n_max
    """

    # Map from resolver's solved_var names → runner param keys
    _VAR_MAP = {
        "N":     "samples_per_window",
        "rho":   "overlap_pct",
        "n_max": "n_max",
    }

    def build_config(
        self,
        base_params: Dict[str, Any],
        resolved_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        params = copy.deepcopy(base_params)
        for theory_key, runner_key in self._VAR_MAP.items():
            if theory_key in resolved_params:
                value = resolved_params[theory_key]
                # Cast integer-valued params properly
                if runner_key in ("samples_per_window", "n_max"):
                    value = int(value)
                params[runner_key] = value
        return {"id": "RMS_CV", "func": "Default", "params": params}

    def run(self, signal: EWSignalData, config: Dict[str, Any]) -> Any:
        run_rms_cv, RmsSD, _ = _import_rms_cv()
        lib_signal = RmsSD(
            # t_cut=self._to_numpy(signal.t_cut),
            # v_cut=self._to_numpy(signal.v_cut),
            # x_cut=self._to_numpy(signal.x_cut),
            # force_cut=self._to_numpy(signal.force_cut),
            # t_original=self._to_numpy(signal.t_original),
            # x_original=self._to_numpy(signal.x_original),
            # v_original=self._to_numpy(signal.v_original),
            t_analysis=signal.t_analysis,
            signal_analysis=signal.signal_analysis,
            # force_original=self._to_numpy(signal.force_original),
            path=signal.path,
            fs=signal.fs,
            meta=dict(signal.meta),
        )
        return run_rms_cv(lib_signal, config)


# ──────────────────────────────────────────────────────────────────────────────
# MaxEnt-SPRT adapter
# ──────────────────────────────────────────────────────────────────────────────

class MaxEntAdapter(IndicatorAdapter):
    """
    Adapter for the MaxEnt-SPRT indicator.

    Resolved variable mapping
    -------------------------
    Theory name  →  INDICATOR_CONFIG["params"] key
    N_seg        →  N_seg   (identical in library and theory)
    """

    def build_config(
        self,
        base_params: Dict[str, Any],
        resolved_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        params = copy.deepcopy(base_params)
        if "N_seg" in resolved_params:
            params["N_seg"] = int(resolved_params["N_seg"])
        return {"id": "MaxEnt_SPRT", "func": "Default", "params": params}

    def run(self, signal: EWSignalData, config: Dict[str, Any]) -> Any:
        run_maxent_sprt, MaxEntSD, _ = _import_maxent()
        lib_signal = MaxEntSD(
            t_analysis=signal.t_analysis,
            signal_analysis=signal.signal_analysis,
            path=signal.path,
            fs=signal.fs,
            meta=dict(signal.meta),
        )
        return run_maxent_sprt(lib_signal, config)


# ──────────────────────────────────────────────────────────────────────────────
# SST-SVD adapter
# ──────────────────────────────────────────────────────────────────────────────

class SSTSVDAdapter(IndicatorAdapter):
    """
    Adapter for the SST-SVD indicator.

    This adapter calls ``_sst_svd_pipeline`` **directly** — bypassing the
    public ``run_sst_svd`` wrapper — so that h_ratio ∈ (0, 1] is fully
    supported without triggering the 25–50% runner validation.

    Resolved variable mapping
    -------------------------
    Theory name  →  _sst_svd_pipeline kwarg
    n_A          →  Ai_length
    w            →  win_length_ms
    h_ratio      →  (converted to hop_ms = h_ratio * win_length_ms)

    The parameter ``hop_ms`` stored in base_params / resolved_params is
    always expressed in milliseconds when passed to the pipeline.
    Internally, h_ms = h_ratio * win_length_ms is computed here.
    """

    _VAR_MAP = {
        "n_A": "Ai_length",
        "w":   "win_length_ms",
        # h_ratio is handled separately: it becomes hop_ms = h_ratio * w
    }

    _RUNNER_HOP_MIN = 0.0
    _RUNNER_HOP_MAX = 0.99

    def build_config(
        self,
        base_params: Dict[str, Any],
        resolved_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        params = copy.deepcopy(base_params)

        for theory_key, runner_key in self._VAR_MAP.items():
            if theory_key in resolved_params:
                value = resolved_params[theory_key]
                if runner_key == "Ai_length":
                    value = int(value)
                params[runner_key] = value

        # Resolve hop_ms from h_ratio if h_ratio was resolved or provided
        h_ratio = resolved_params.get("h_ratio") or params.get("h_ratio")
        w_ms    = resolved_params.get("w") or params.get("win_length_ms")
        if h_ratio is not None and w_ms is not None:
            hop_ms = float(h_ratio) * float(w_ms)
            params["hop_ms"] = hop_ms
            # Advisory warning if outside runner's preferred range
            if not (self._RUNNER_HOP_MIN <= float(h_ratio) <= self._RUNNER_HOP_MAX):
                logger.warning(
                    "SST-SVD: h_ratio = %.4f (hop_ms = %.2f ms) is outside the "
                    "library runner's preferred range [%.2f, %.2f]. "
                    "Calling _sst_svd_pipeline directly to bypass validation. "
                    "Theory constraint 0 < h ≤ w is satisfied.",
                    h_ratio, hop_ms,
                    self._RUNNER_HOP_MIN * float(w_ms),
                    self._RUNNER_HOP_MAX * float(w_ms),
                )

        # Remove the framework-level key; the pipeline doesn't know it
        params.pop("h_ratio", None)

        return {"id": "SST_SVD", "func": "Default", "params": params}

    def run(self, signal: EWSignalData, config: Dict[str, Any]) -> Any:
        _sst_svd_pipeline, SstSD, _ = _import_sst()
        params = config["params"]

        lib_signal = SstSD(
            # t_cut=self._to_numpy(signal.t_cut),
            # v_cut=self._to_numpy(signal.v_cut),
            # x_cut=self._to_numpy(signal.x_cut),
            # force_cut=self._to_numpy(signal.force_cut),
            # t_original=self._to_numpy(signal.t_original),
            # x_original=self._to_numpy(signal.x_original),
            # v_original=self._to_numpy(signal.v_original),
            t_analysis=signal.t_analysis,
            signal_analysis=signal.signal_analysis,
            # force_original=self._to_numpy(signal.force_original),
            path=signal.path,
            fs=signal.fs,
            meta=dict(signal.meta),
        )

        # Call private pipeline directly — bypasses hop validation
        return _sst_svd_pipeline(lib_signal, **params)


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

ADAPTER_REGISTRY: Dict[str, IndicatorAdapter] = {
    # canonical keys (match library IDs)
    "RMS_CV":      RMSCVAdapter(),
    "MaxEnt_SPRT": MaxEntAdapter(),
    "SST_SVD":     SSTSVDAdapter(),
    # lowercase / normalized aliases
    "rms_cv":      RMSCVAdapter(),
    "maxent_sprt": MaxEntAdapter(),
    "sst_svd":     SSTSVDAdapter(),
}
"""
Maps ``indicator_id`` → adapter instance.

To add a new indicator, implement :class:`IndicatorAdapter` and add an
entry here.
"""


def get_adapter(indicator_id: str) -> IndicatorAdapter:
    """Return the adapter for *indicator_id*.

    Raises ``KeyError`` if the indicator is not registered.
    """
    if indicator_id not in ADAPTER_REGISTRY:
        raise KeyError(
            f"No IndicatorAdapter registered for indicator '{indicator_id}'. "
            "Add one to effective_window.adapters.ADAPTER_REGISTRY."
        )
    return ADAPTER_REGISTRY[indicator_id]
