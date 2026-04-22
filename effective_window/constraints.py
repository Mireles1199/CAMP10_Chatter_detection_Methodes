"""
constraints.py
==============
Admissibility checkers for the effective-window framework.

Three levels of checking per indicator:

  Level 1 — Basic (type & range)
      Verify that each parameter satisfies the domain constraints stated in
      the theory (e.g. N ∈ ℕ₊, 0 ≤ ρ < 1).

  Level 2 — Algebraic feasibility
      Verify that the algebraically resolved value lies within its valid
      domain (e.g. the solved ρ* must be in [0, 1); w* must be > 0).

  Level 3 — Degenerate cases
      Detect configurations where the chosen parameter is not identifiable
      from the equality constraint (e.g. ρ when n_max = 1, h_ratio when
      n_A = 1).  These are not errors in the data but structural collapses
      of the parametrization.

Theory reference
----------------
  RMS-CV  : F_RMS  = {(N, ρ, n_max) : (N/fs)[1+(n_max-1)(1-ρ)] = T_des}
             N ∈ ℕ₊,  n_max ∈ ℕ₊,  0 ≤ ρ < 1
  MaxEnt  : N_seg ∈ ℕ₊
  SST-SVD : n_A ∈ ℕ₊,  w > 0,  h_ratio ∈ (0, 1]  (theory: 0 < h ≤ w)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .config import ParameterResolutionConfig


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ConstraintReport:
    """
    Outcome of one constraint-checking call.

    Attributes
    ----------
    passed : bool
        True if all checks passed.
    level_failed : str or None
        ``"basic"`` | ``"feasibility"`` | ``"degenerate"`` | None.
    message : str
        Human-readable description of the first failure found.
    details : dict
        All parameter values that were checked (for logging / debug plots).
    """

    passed: bool
    level_failed: Optional[str] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, details: Dict[str, Any] | None = None) -> "ConstraintReport":
        return cls(passed=True, details=details or {})

    @classmethod
    def fail(
        cls,
        level: str,
        message: str,
        details: Dict[str, Any] | None = None,
    ) -> "ConstraintReport":
        return cls(
            passed=False,
            level_failed=level,
            message=message,
            details=details or {},
        )


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────

class ConstraintChecker(ABC):
    """
    Abstract base for all per-indicator admissibility checkers.

    Parameters are passed as a flat ``dict`` so that the checker does not
    depend on whether the parameter came from ``base_params`` or was freshly
    resolved.
    """

    @abstractmethod
    def check(
        self,
        params: Dict[str, Any],
        config: ParameterResolutionConfig,
    ) -> ConstraintReport:
        """
        Run all three levels of admissibility checks.

        Parameters
        ----------
        params : dict
            Merged parameter dict produced by the resolver
            (fixed_vars + solved variable at rounded value).
        config : ParameterResolutionConfig
            Resolution configuration — used to detect degenerate cases.

        Returns
        -------
        ConstraintReport
        """


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _is_positive_int(value: Any) -> bool:
    """True if value is a finite float/int that equals a positive integer."""
    try:
        v = float(value)
        return math.isfinite(v) and v >= 1.0 and math.isclose(v, round(v), rel_tol=1e-9)
    except (TypeError, ValueError):
        return False


# ──────────────────────────────────────────────────────────────────────────────
# RMS-CV
# ──────────────────────────────────────────────────────────────────────────────

class RMSCVConstraintChecker(ConstraintChecker):
    """
    Admissibility checker for the RMS-CV indicator.

    Parameters checked
    ------------------
    Level 1 (basic):
        N ∈ ℕ₊  (positive integer)
        n_max ∈ ℕ₊
        0 ≤ ρ < 1

    Level 2 (algebraic feasibility):
        - If solved_var = "rho" : ρ* must be in [0, 1)
        - If solved_var = "n_max": n_max* must be ≥ 1 and finite
        - If solved_var = "N"   : N* must be ≥ 1 and finite

    Level 3 (degenerate):
        - "rho" with n_max == 1  → not identifiable
        - "n_max" with rho == 1  → not identifiable
    """

    def check(
        self,
        params: Dict[str, Any],
        config: ParameterResolutionConfig,
    ) -> ConstraintReport:
        details = dict(params)
        solved_var = config.solved_var
        fv = config.fixed_vars

        # ── Level 3: degenerate cases ─────────────────────────────────────────
        if solved_var == "rho":
            n_max_fv = float(fv.get("n_max", params.get("n_max", 2)))
            if n_max_fv == 1.0:
                return ConstraintReport.fail(
                    "degenerate",
                    "RMS-CV: n_max == 1 → rho is not identifiable. "
                    "T_w = N/fs regardless of rho. "
                    "Use solve_N or set n_max > 1.",
                    details,
                )

        if solved_var == "n_max":
            rho_fv = float(fv.get("rho", params.get("rho", 0.0)))
            if rho_fv == 1.0:
                return ConstraintReport.fail(
                    "degenerate",
                    "RMS-CV: rho == 1 → n_max is not identifiable. "
                    "T_w = N/fs regardless of n_max. "
                    "Use solve_N or set rho < 1.",
                    details,
                )

        # ── Level 1: basic domain checks ─────────────────────────────────────
        N     = params.get("N")
        n_max = params.get("n_max")
        rho   = params.get("rho")

        if N is not None and not _is_positive_int(N):
            return ConstraintReport.fail(
                "basic",
                f"RMS-CV: N must be a positive integer; got N = {N!r}.",
                details,
            )
        if n_max is not None and not _is_positive_int(n_max):
            return ConstraintReport.fail(
                "basic",
                f"RMS-CV: n_max must be a positive integer; got n_max = {n_max!r}.",
                details,
            )
        if rho is not None:
            rho_f = float(rho)
            if not (0.0 <= rho_f < 1.0):
                return ConstraintReport.fail(
                    "basic",
                    f"RMS-CV: rho must be in [0, 1); got rho = {rho_f}.",
                    details,
                )

        # ── Level 2: algebraic feasibility ────────────────────────────────────
        if solved_var == "rho" and rho is not None:
            rho_f = float(rho)
            if not (0.0 <= rho_f < 1.0):
                return ConstraintReport.fail(
                    "feasibility",
                    f"RMS-CV(solve_rho): resolved ρ* = {rho_f:.6g} is outside [0, 1). "
                    "Adjust T_des, N, or n_max.",
                    details,
                )

        if solved_var == "n_max" and n_max is not None:
            if not _is_positive_int(n_max):
                return ConstraintReport.fail(
                    "feasibility",
                    f"RMS-CV(solve_n_max): resolved n_max* = {n_max!r} is not a "
                    "positive integer. Adjust T_des, N, or rho.",
                    details,
                )

        return ConstraintReport.ok(details)


# ──────────────────────────────────────────────────────────────────────────────
# MaxEnt-SPRT
# ──────────────────────────────────────────────────────────────────────────────

class MaxEntConstraintChecker(ConstraintChecker):
    """
    Admissibility checker for the MaxEnt-SPRT indicator.

    Parameters checked
    ------------------
    Level 1: N_seg ∈ ℕ₊
    Level 2: resolved N_seg* ≥ 1 and finite
    Level 3: no degenerate case (single-parameter indicator)
    """

    def check(
        self,
        params: Dict[str, Any],
        config: ParameterResolutionConfig,
    ) -> ConstraintReport:
        details = dict(params)
        N_seg = params.get("N_seg")

        # Level 1
        if N_seg is not None and not _is_positive_int(N_seg):
            return ConstraintReport.fail(
                "basic",
                f"MaxEnt-SPRT: N_seg must be a positive integer; got N_seg = {N_seg!r}.",
                details,
            )

        # Level 2 (same check — N_seg is the only resolved var)
        if config.solved_var == "N_seg" and N_seg is not None:
            if not _is_positive_int(N_seg):
                return ConstraintReport.fail(
                    "feasibility",
                    f"MaxEnt-SPRT(solve_N_seg): resolved N_seg* = {N_seg!r} is not a "
                    "positive integer. Adjust T_des or rpm.",
                    details,
                )

        return ConstraintReport.ok(details)


# ──────────────────────────────────────────────────────────────────────────────
# SST-SVD
# ──────────────────────────────────────────────────────────────────────────────

class SSTSVDConstraintChecker(ConstraintChecker):
    """
    Admissibility checker for the SST-SVD indicator.

    Theory constraint:  0 < h ≤ w  (expressed as h_ratio ∈ (0, 1]).

    Parameters checked
    ------------------
    Level 1 (basic):
        n_A ∈ ℕ₊
        w > 0  [ms]
        h_ratio ∈ (0, 1]
        h_ms = h_ratio * w > 0

    Level 2 (algebraic feasibility):
        - If solved_var = "n_A"     : n_A* ∈ ℕ₊
        - If solved_var = "w"       : w* > 0
        - If solved_var = "h_ratio" : h_ratio* ∈ (0, 1]

    Level 3 (degenerate):
        - "h_ratio" with n_A == 1 → not identifiable

    Note: h_ratio ∈ (0.25, 0.50] is the library runner's preference; it is
    NOT a theoretical constraint.  A warning is logged when h_ratio falls
    outside that range, but the check still passes.
    """

    _RUNNER_HOP_MIN = 0.25
    _RUNNER_HOP_MAX = 0.50

    def check(
        self,
        params: Dict[str, Any],
        config: ParameterResolutionConfig,
    ) -> ConstraintReport:
        details = dict(params)
        solved_var = config.solved_var
        fv = config.fixed_vars

        # ── Level 3: degenerate ───────────────────────────────────────────────
        if solved_var == "h_ratio":
            n_A_fv = float(fv.get("n_A", params.get("n_A", 2)))
            if n_A_fv == 1.0:
                return ConstraintReport.fail(
                    "degenerate",
                    "SST-SVD: n_A == 1 → h_ratio is not identifiable. "
                    "T_w = w regardless of h_ratio. "
                    "Use solve_w or set n_A > 1.",
                    details,
                )

        # ── Level 1: basic domain checks ─────────────────────────────────────
        n_A     = params.get("n_A")
        w       = params.get("w")
        h_ratio = params.get("h_ratio")

        if n_A is not None and not _is_positive_int(n_A):
            return ConstraintReport.fail(
                "basic",
                f"SST-SVD: n_A must be a positive integer; got n_A = {n_A!r}.",
                details,
            )

        if w is not None:
            w_f = float(w)
            if not (w_f > 0):
                return ConstraintReport.fail(
                    "basic",
                    f"SST-SVD: w must be > 0 ms; got w = {w_f}.",
                    details,
                )

        if h_ratio is not None:
            h_f = float(h_ratio)
            if not (0.0 < h_f <= 1.0):
                return ConstraintReport.fail(
                    "basic",
                    f"SST-SVD: h_ratio must be in (0, 1]; got h_ratio = {h_f:.6g}.",
                    details,
                )
            # Non-blocking advisory: flag if outside the runner's preferred range
            details["_runner_hop_advisory"] = not (
                self._RUNNER_HOP_MIN <= h_f <= self._RUNNER_HOP_MAX
            )

        # ── Level 2: algebraic feasibility ────────────────────────────────────
        if solved_var == "n_A" and n_A is not None:
            if not _is_positive_int(n_A):
                return ConstraintReport.fail(
                    "feasibility",
                    f"SST-SVD(solve_n_A): resolved n_A* = {n_A!r} is not a "
                    "positive integer. Adjust T_des, w, or h_ratio.",
                    details,
                )

        if solved_var == "w" and w is not None:
            if float(w) <= 0:
                return ConstraintReport.fail(
                    "feasibility",
                    f"SST-SVD(solve_w): resolved w* = {w} ms is not positive. "
                    "Adjust T_des, n_A, or h_ratio.",
                    details,
                )

        if solved_var == "h_ratio" and h_ratio is not None:
            h_f = float(h_ratio)
            if not (0.0 < h_f <= 1.0):
                return ConstraintReport.fail(
                    "feasibility",
                    f"SST-SVD(solve_h_ratio): resolved h_ratio* = {h_f:.6g} is "
                    "outside (0, 1]. "
                    "Adjust T_des, w, or n_A.",
                    details,
                )

        return ConstraintReport.ok(details)


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

CONSTRAINT_REGISTRY: Dict[str, ConstraintChecker] = {
    # canonical keys (match library IDs)
    "RMS_CV":      RMSCVConstraintChecker(),
    "MaxEnt_SPRT": MaxEntConstraintChecker(),
    "SST_SVD":     SSTSVDConstraintChecker(),
    # lowercase / normalized aliases
    "rms_cv":      RMSCVConstraintChecker(),
    "maxent_sprt": MaxEntConstraintChecker(),
    "sst_svd":     SSTSVDConstraintChecker(),
}
"""
Maps ``indicator_id`` → constraint checker instance.

Both the canonical ID (``"RMS_CV"``) and the lowercase alias
(``"rms_cv"``) are accepted.
"""


def get_checker(indicator_id: str) -> ConstraintChecker:
    """Return the constraint checker for *indicator_id*.

    Raises ``KeyError`` if the indicator is not registered, prompting the
    developer to add an entry in ``CONSTRAINT_REGISTRY``.
    """
    if indicator_id not in CONSTRAINT_REGISTRY:
        raise KeyError(
            f"No ConstraintChecker registered for indicator '{indicator_id}'. "
            "Add one to effective_window.constraints.CONSTRAINT_REGISTRY."
        )
    return CONSTRAINT_REGISTRY[indicator_id]
