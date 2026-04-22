"""
resolvers.py
============
Algebraic window-parameter resolvers.

Each resolver implements one or more directed parametrization paths as defined
in the theory.  Every path solves algebraically for one indicator parameter
given T_des and the values of the remaining (fixed) parameters.

Theory traceability
-------------------
  RMS-CV
    T_w = (N/fs) * [1 + (n_max - 1)(1 - rho)]
    solve_N     →  N*     = fs * T_des / [1 + (n_max-1)(1-rho)]
    solve_rho   →  rho*   = 1 - [(fs*T_des/N - 1) / (n_max - 1)]
    solve_n_max →  n_max* = 1 + [(fs*T_des/N - 1) / (1 - rho)]

  MaxEnt-SPRT
    T_w = N_seg * T_rev
    solve_N_seg →  N_seg* = T_des / T_rev

  SST-SVD   (all quantities in milliseconds)
    T_w_ms = w + (n_A - 1) * h_ms,  where h_ms = h_ratio * w
    solve_n_A     →  n_A*     = 1 + (T_des_ms - w) / h_ms
    solve_w       →  w*       = T_des_ms - (n_A - 1) * h_ms
    solve_h_ratio →  h_ratio* = (T_des_ms - w) / [(n_A - 1) * w]
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict

from .config import ParameterResolutionConfig, RoundingPolicy


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ResolverResult:
    """
    Output of a single resolver path invocation.

    Attributes
    ----------
    solved_var : str
        Name of the parameter that was solved.
    raw_value : float
        Continuous algebraic solution (before rounding).
    rounded_value : float
        Value after applying the configured :class:`~effective_window.config.RoundingPolicy`.
        Equal to ``raw_value`` when policy is NONE.
    T_w_actual : float
        Effective window [s] computed with the rounded parameter.
    T_des : float
        Target window [s] that was imposed.
    delta_T_w : float
        T_w_actual − T_des  [s].  Positive = window is slightly larger than target.
    resolved_params : dict
        Merged dict: fixed_vars + {solved_var: rounded_value}.
        Ready to be used directly by an adapter.
    """

    solved_var: str
    raw_value: float
    rounded_value: float
    T_w_actual: float
    T_des: float
    delta_T_w: float
    resolved_params: Dict[str, Any]


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────

class WindowResolver(ABC):
    """
    Abstract base for all indicator window resolvers.

    Subclasses must implement :meth:`resolve` for each supported
    ``resolution_path`` and expose them via the ``_PATHS`` dispatch table.
    """

    #: Mapping  path_name → handler method.  Populated by subclasses.
    _PATHS: Dict[str, Callable[..., ResolverResult]] = {}

    @abstractmethod
    def resolve(
        self,
        T_des: float,
        fs: float,
        config: ParameterResolutionConfig,
    ) -> ResolverResult:
        """
        Resolve the indicator parameter for the given target window.

        Parameters
        ----------
        T_des : float
            Target effective decision window [s].
        fs : float
            Signal sampling frequency [Hz].
        config : ParameterResolutionConfig
            Which variable to solve, fixed values, rounding policy.

        Returns
        -------
        ResolverResult
        """

    def _apply_rounding(self, value: float, policy: RoundingPolicy) -> float:
        return policy.apply(value)

    @classmethod
    def supported_paths(cls) -> list[str]:
        return list(cls._PATHS.keys())


# ──────────────────────────────────────────────────────────────────────────────
# RMS-CV resolver
# ──────────────────────────────────────────────────────────────────────────────

class RMSCVResolver(WindowResolver):
    """
    Algebraic resolver for the RMS-CV indicator.

    Effective window (theory):
        T_w = (N / fs) * [1 + (n_max - 1)(1 - rho)]

    Supported paths:  ``solve_N`` | ``solve_rho`` | ``solve_n_max``
    """

    def resolve(
        self,
        T_des: float,
        fs: float,
        config: ParameterResolutionConfig,
    ) -> ResolverResult:
        path = config.resolution_path
        fv = config.fixed_vars
        rounding = config.rounding

        if path == "solve_N":
            return self._solve_N(T_des, fs, fv, rounding)
        if path == "solve_rho":
            return self._solve_rho(T_des, fs, fv, rounding)
        if path == "solve_n_max":
            return self._solve_n_max(T_des, fs, fv, rounding)

        raise ValueError(
            f"RMSCVResolver: unknown path '{path}'. "
            f"Supported: solve_N, solve_rho, solve_n_max."
        )

    # ── T_w formula helper ────────────────────────────────────────────────────

    @staticmethod
    def _T_w(N: float, fs: float, n_max: float, rho: float) -> float:
        """T_w = (N/fs) * [1 + (n_max-1)(1-rho)]"""
        return (N / fs) * (1.0 + (n_max - 1.0) * (1.0 - rho))

    # ── path: solve N ────────────────────────────────────────────────────────

    def _solve_N(
        self,
        T_des: float,
        fs: float,
        fv: Dict[str, Any],
        rounding: RoundingPolicy,
    ) -> ResolverResult:
        n_max = float(fv["n_max"])
        rho   = float(fv["rho"])
        # N* = fs * T_des / [1 + (n_max-1)(1-rho)]
        denom = 1.0 + (n_max - 1.0) * (1.0 - rho)
        raw = fs * T_des / denom
        rounded = self._apply_rounding(raw, rounding)
        T_w_actual = self._T_w(rounded, fs, n_max, rho)
        return ResolverResult(
            solved_var="N",
            raw_value=raw,
            rounded_value=rounded,
            T_w_actual=T_w_actual,
            T_des=T_des,
            delta_T_w=T_w_actual - T_des,
            resolved_params={**fv, "N": rounded},
        )

    # ── path: solve rho ──────────────────────────────────────────────────────

    def _solve_rho(
        self,
        T_des: float,
        fs: float,
        fv: Dict[str, Any],
        rounding: RoundingPolicy,
    ) -> ResolverResult:
        N     = float(fv["N"])
        n_max = float(fv["n_max"])
        # rho* = 1 - [(fs*T_des/N - 1) / (n_max - 1)]   requires n_max != 1
        if n_max == 1.0:
            raise ValueError(
                "RMSCVResolver(solve_rho): n_max == 1 → "
                "rho is not identifiable (T_w = N/fs regardless of rho). "
                "Use solve_N instead, or choose n_max > 1."
            )
        raw = 1.0 - (fs * T_des / N - 1.0) / (n_max - 1.0)
        rounded = self._apply_rounding(raw, rounding)
        T_w_actual = self._T_w(N, fs, n_max, rounded)
        return ResolverResult(
            solved_var="rho",
            raw_value=raw,
            rounded_value=rounded,
            T_w_actual=T_w_actual,
            T_des=T_des,
            delta_T_w=T_w_actual - T_des,
            resolved_params={**fv, "rho": rounded},
        )

    # ── path: solve n_max ────────────────────────────────────────────────────

    def _solve_n_max(
        self,
        T_des: float,
        fs: float,
        fv: Dict[str, Any],
        rounding: RoundingPolicy,
    ) -> ResolverResult:
        N   = float(fv["N"])
        rho = float(fv["rho"])
        # n_max* = 1 + [(fs*T_des/N - 1) / (1 - rho)]   requires rho != 1
        if rho == 1.0:
            raise ValueError(
                "RMSCVResolver(solve_n_max): rho == 1 → "
                "n_max is not identifiable (T_w = N/fs regardless of n_max). "
                "Use solve_N instead, or choose rho < 1."
            )
        raw = 1.0 + (fs * T_des / N - 1.0) / (1.0 - rho)
        rounded = self._apply_rounding(raw, rounding)
        T_w_actual = self._T_w(N, fs, rounded, rho)
        return ResolverResult(
            solved_var="n_max",
            raw_value=raw,
            rounded_value=rounded,
            T_w_actual=T_w_actual,
            T_des=T_des,
            delta_T_w=T_w_actual - T_des,
            resolved_params={**fv, "n_max": rounded},
        )


# ──────────────────────────────────────────────────────────────────────────────
# MaxEnt-SPRT resolver
# ──────────────────────────────────────────────────────────────────────────────

class MaxEntResolver(WindowResolver):
    """
    Algebraic resolver for the MaxEnt-SPRT indicator.

    Effective window (theory):
        T_w = N_seg * T_rev,   T_rev = 60 / rpm

    Supported paths:  ``solve_N_seg``

    Note: In the code, the parameter is named ``N_seg``; in the theory it is
    labelled N_rev (number of revolution-synchronous segments).
    """

    def resolve(
        self,
        T_des: float,
        fs: float,
        config: ParameterResolutionConfig,
    ) -> ResolverResult:
        path = config.resolution_path
        if path == "solve_N_seg":
            return self._solve_N_seg(T_des, config.fixed_vars, config.rounding)
        raise ValueError(
            f"MaxEntResolver: unknown path '{path}'. Supported: solve_N_seg."
        )

    def _solve_N_seg(
        self,
        T_des: float,
        fv: Dict[str, Any],
        rounding: RoundingPolicy,
    ) -> ResolverResult:
        rpm = float(fv["rpm"])
        T_rev = 60.0 / rpm
        # N_seg* = T_des / T_rev
        raw = T_des / T_rev
        rounded = self._apply_rounding(raw, rounding)
        T_w_actual = rounded * T_rev
        return ResolverResult(
            solved_var="N_seg",
            raw_value=raw,
            rounded_value=rounded,
            T_w_actual=T_w_actual,
            T_des=T_des,
            delta_T_w=T_w_actual - T_des,
            resolved_params={**fv, "N_seg": rounded},
        )


# ──────────────────────────────────────────────────────────────────────────────
# SST-SVD resolver
# ──────────────────────────────────────────────────────────────────────────────

class SSTSVDResolver(WindowResolver):
    """
    Algebraic resolver for the SST-SVD indicator.

    Effective window (theory, all in milliseconds):
        T_w_ms = w + (n_A - 1) * h_ms

    The hop is expressed as an adimensional ratio:
        h_ratio ∈ (0, 1]  →  h_ms = h_ratio * w

    Supported paths:  ``solve_n_A`` | ``solve_w`` | ``solve_h_ratio``

    The theory's admissibility constraint is 0 < h ≤ w, which translates
    to h_ratio ∈ (0, 1].  The SST-SVD library runner additionally prefers
    h_ratio ∈ [0.25, 0.50]; the adapter bypasses that check when h_ratio
    falls outside that range, while still logging a warning.
    """

    def resolve(
        self,
        T_des: float,
        fs: float,
        config: ParameterResolutionConfig,
    ) -> ResolverResult:
        path = config.resolution_path
        fv   = config.fixed_vars
        rounding = config.rounding

        if path == "solve_n_A":
            return self._solve_n_A(T_des, fv, rounding)
        if path == "solve_w":
            return self._solve_w(T_des, fv, rounding)
        if path == "solve_h_ratio":
            return self._solve_h_ratio(T_des, fv, rounding)

        raise ValueError(
            f"SSTSVDResolver: unknown path '{path}'. "
            f"Supported: solve_n_A, solve_w, solve_h_ratio."
        )

    # ── T_w formula helper ────────────────────────────────────────────────────

    @staticmethod
    def _T_w_ms(w: float, n_A: float, h_ratio: float) -> float:
        """T_w_ms = w + (n_A - 1) * h_ratio * w  [ms]"""
        return w + (n_A - 1.0) * h_ratio * w

    # ── path: solve n_A ──────────────────────────────────────────────────────

    def _solve_n_A(
        self,
        T_des: float,
        fv: Dict[str, Any],
        rounding: RoundingPolicy,
    ) -> ResolverResult:
        w       = float(fv["w"])           # [ms]
        h_ratio = float(fv["h_ratio"])     # adimensional ∈ (0, 1]
        h_ms    = h_ratio * w
        T_des_ms = 1000.0 * T_des
        # n_A* = 1 + (T_des_ms - w) / h_ms
        if h_ms <= 0:
            raise ValueError("SSTSVDResolver(solve_n_A): h_ms must be > 0.")
        raw = 1.0 + (T_des_ms - w) / h_ms
        rounded = self._apply_rounding(raw, rounding)
        T_w_ms_actual = self._T_w_ms(w, rounded, h_ratio)
        return ResolverResult(
            solved_var="n_A",
            raw_value=raw,
            rounded_value=rounded,
            T_w_actual=T_w_ms_actual / 1000.0,
            T_des=T_des,
            delta_T_w=T_w_ms_actual / 1000.0 - T_des,
            resolved_params={**fv, "n_A": rounded},
        )

    # ── path: solve w ────────────────────────────────────────────────────────

    def _solve_w(
        self,
        T_des: float,
        fv: Dict[str, Any],
        rounding: RoundingPolicy,
    ) -> ResolverResult:
        n_A     = float(fv["n_A"])
        h_ratio = float(fv["h_ratio"])
        T_des_ms = 1000.0 * T_des
        # w* = T_des_ms - (n_A - 1) * h_ratio * w*
        # → w* = T_des_ms / [1 + (n_A - 1)*h_ratio]
        denom = 1.0 + (n_A - 1.0) * h_ratio
        if denom <= 0:
            raise ValueError(
                "SSTSVDResolver(solve_w): denominator is non-positive. "
                "Check n_A and h_ratio values."
            )
        raw = T_des_ms / denom
        rounded = self._apply_rounding(raw, rounding)
        h_ms = h_ratio * rounded
        T_w_ms_actual = rounded + (n_A - 1.0) * h_ms
        return ResolverResult(
            solved_var="w",
            raw_value=raw,
            rounded_value=rounded,
            T_w_actual=T_w_ms_actual / 1000.0,
            T_des=T_des,
            delta_T_w=T_w_ms_actual / 1000.0 - T_des,
            resolved_params={**fv, "w": rounded},
        )

    # ── path: solve h_ratio ──────────────────────────────────────────────────

    def _solve_h_ratio(
        self,
        T_des: float,
        fv: Dict[str, Any],
        rounding: RoundingPolicy,
    ) -> ResolverResult:
        n_A = float(fv["n_A"])
        w   = float(fv["w"])    # [ms]
        if n_A == 1.0:
            raise ValueError(
                "SSTSVDResolver(solve_h_ratio): n_A == 1 → "
                "h_ratio is not identifiable (T_w = w regardless of h_ratio). "
                "Use solve_w instead, or choose n_A > 1."
            )
        T_des_ms = 1000.0 * T_des
        # h_ratio* = (T_des_ms - w) / [(n_A - 1) * w]
        raw = (T_des_ms - w) / ((n_A - 1.0) * w)
        rounded = self._apply_rounding(raw, rounding)
        h_ms = rounded * w
        T_w_ms_actual = w + (n_A - 1.0) * h_ms
        return ResolverResult(
            solved_var="h_ratio",
            raw_value=raw,
            rounded_value=rounded,
            T_w_actual=T_w_ms_actual / 1000.0,
            T_des=T_des,
            delta_T_w=T_w_ms_actual / 1000.0 - T_des,
            resolved_params={**fv, "h_ratio": rounded},
        )


# ──────────────────────────────────────────────────────────────────────────────
# Not-implemented resolver (extension point for future indicators, e.g. EMD-HHT)
# ──────────────────────────────────────────────────────────────────────────────

class NotImplementedResolver(WindowResolver):
    """
    Placeholder resolver for indicators whose window formula is not yet defined.

    Raises ``NotImplementedError`` with a descriptive message so the absence
    of a real resolver is never silently ignored.
    """

    def __init__(self, indicator_id: str) -> None:
        self._id = indicator_id

    def resolve(
        self,
        T_des: float,
        fs: float,
        config: ParameterResolutionConfig,
    ) -> ResolverResult:
        raise NotImplementedError(
            f"No window resolver is available for indicator '{self._id}'. "
            "Add a WindowResolver subclass and register it in "
            "effective_window.resolvers.RESOLVER_REGISTRY to enable it."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

RESOLVER_REGISTRY: Dict[str, WindowResolver] = {
    # canonical keys (match library IDs)
    "RMS_CV":      RMSCVResolver(),
    "MaxEnt_SPRT": MaxEntResolver(),
    "SST_SVD":     SSTSVDResolver(),
    # lowercase / normalized aliases (accepted in IndicatorWindowConfig)
    "rms_cv":      RMSCVResolver(),
    "maxent_sprt": MaxEntResolver(),
    "sst_svd":     SSTSVDResolver(),
}
"""
Maps ``indicator_id`` → resolver instance.

Both the canonical ID (``"RMS_CV"``) and the lowercase alias
(``"rms_cv"``) are accepted.
"""


def get_resolver(indicator_id: str) -> WindowResolver:
    """Return the resolver for *indicator_id*, or a ``NotImplementedResolver``."""
    return RESOLVER_REGISTRY.get(indicator_id, NotImplementedResolver(indicator_id))
