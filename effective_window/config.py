"""
config.py
=========
Configuration dataclasses and enumerations for the effective-window framework.

Theory traceability
-------------------
- ``WindowBasis``              → two physical definitions of T_des (modal / revolution)
- ``WindowSpec``               → T_des = n_m * T_modal  or  T_des = n_r * T_rev
- ``RoundingPolicy``           → floor / ceil / round / none  (NONE = raw float)
- ``ParameterResolutionConfig``→ directed parametrization: solved_var + fixed_vars + path
- ``IndicatorWindowConfig``    → per-indicator assembly of resolution + base params
- ``RunnerConfig``             → top-level configuration passed to WindowRunner
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────

class WindowBasis(Enum):
    """Physical basis used to define the target observation window T_des."""

    MODAL = "modal"
    """T_des = n_m / f_modal  (modal-cycle basis)."""

    REVOLUTION = "revolution"
    """T_des = n_r * 60 / rpm  (spindle-revolution basis)."""


class RoundingPolicy(Enum):
    """
    Discretisation policy applied to a continuously solved parameter.

    ============  =============================================================
    FLOOR         math.floor — round down to nearest integer
    CEIL          math.ceil  — round up to nearest integer
    ROUND         round()    — round to nearest integer (Python built-in)
    NONE          pass the raw float through without any rounding
                  (useful for continuous-valued parameters such as ρ or h_ratio,
                  or for Phase 2 optimisation where integer relaxation is needed)
    ============  =============================================================
    """

    FLOOR = "floor"
    CEIL  = "ceil"
    ROUND = "round"
    NONE  = "none"

    def apply(self, value: float) -> float:
        """Apply this policy to *value* and return the result."""
        if self is RoundingPolicy.FLOOR:
            return float(math.floor(value))
        if self is RoundingPolicy.CEIL:
            return float(math.ceil(value))
        if self is RoundingPolicy.ROUND:
            return float(round(value))
        # NONE — return as-is
        return float(value)


# ──────────────────────────────────────────────────────────────────────────────
# Window specification  (T_des)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WindowSpec:
    """
    Specifies the common physical effective decision window T_des.

    Either ``f_modal`` (MODAL basis) or ``rpm`` (REVOLUTION basis) must be
    provided, depending on ``basis``.

    Parameters
    ----------
    basis : WindowBasis
        Which physical cycle type is used to define T_des.
    n_cycles : float
        Number of periods / revolutions.
        Theory notation: n_m (modal) or n_r (revolution).
    f_modal : float, optional
        Structural modal frequency [Hz].  Required when basis=MODAL.
    rpm : float, optional
        Spindle speed [rev/min].  Required when basis=REVOLUTION.
    """

    basis: WindowBasis
    n_cycles: float
    f_modal: Optional[float] = None   # [Hz]  — required for MODAL
    rpm:     Optional[float] = None   # [rpm] — required for REVOLUTION

    def compute_T_des(self) -> float:
        """
        Compute and return T_des [s].

        MODAL:      T_des = n_m / f_modal        (n_m * T_modal)
        REVOLUTION: T_des = n_r * 60 / rpm       (n_r * T_rev)

        Raises
        ------
        ValueError
            If the required physical parameter is missing or non-positive.
        """
        if self.basis is WindowBasis.MODAL:
            if self.f_modal is None or self.f_modal <= 0:
                raise ValueError(
                    "WindowSpec: f_modal must be a positive float when basis=MODAL."
                )
            return self.n_cycles / self.f_modal

        # REVOLUTION
        if self.rpm is None or self.rpm <= 0:
            raise ValueError(
                "WindowSpec: rpm must be a positive float when basis=REVOLUTION."
            )
        T_rev = 60.0 / self.rpm
        return self.n_cycles * T_rev

    def T_rev(self) -> float:
        """Spindle revolution period [s].  Requires ``rpm`` to be set."""
        if self.rpm is None or self.rpm <= 0:
            raise ValueError("WindowSpec: rpm must be set to access T_rev.")
        return 60.0 / self.rpm

    def T_modal(self) -> float:
        """Structural modal period [s].  Requires ``f_modal`` to be set."""
        if self.f_modal is None or self.f_modal <= 0:
            raise ValueError("WindowSpec: f_modal must be set to access T_modal.")
        return 1.0 / self.f_modal

    def __repr__(self) -> str:
        T = self.compute_T_des()
        return (
            f"WindowSpec(basis={self.basis.value}, n_cycles={self.n_cycles}, "
            f"T_des={T*1000:.2f} ms)"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Parameter resolution configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ParameterResolutionConfig:
    """
    Specifies the directed parametrization for one indicator.

    Theory: the equality T_w(θ) = T_des removes one degree of freedom.
    One parameter is solved algebraically; the others are held fixed.

    Parameters
    ----------
    solved_var : str
        Name of the parameter to be algebraically solved.
        Must match a key in the corresponding resolver's supported paths.

        ================  =====================================================
        Indicator         Valid values
        ================  =====================================================
        RMS_CV            ``"N"`` | ``"rho"`` | ``"n_max"``
        MaxEnt_SPRT       ``"N_seg"``
        SST_SVD           ``"n_A"`` | ``"w"`` | ``"h_ratio"``
        ================  =====================================================

    fixed_vars : dict
        Values of ALL parameters that are NOT being solved.
        Must contain every parameter required by the chosen resolution path
        except ``solved_var``.
    resolution_path : str, optional
        Explicit label of the algebraic path to use.
        Defaults to ``f"solve_{solved_var}"``.
    rounding : RoundingPolicy
        Discretisation policy applied to the raw solved value.
        Use NONE for continuous parameters (ρ, h_ratio) or when the raw
        float is needed by a Phase 2 optimisation layer.
    """

    solved_var: str
    fixed_vars: Dict[str, Any]
    rounding: RoundingPolicy = RoundingPolicy.ROUND
    resolution_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.resolution_path is None:
            self.resolution_path = f"solve_{self.solved_var}"


# ──────────────────────────────────────────────────────────────────────────────
# Per-indicator window configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class IndicatorWindowConfig:
    """
    Full configuration for one indicator inside the WindowRunner.

    Parameters
    ----------
    indicator_id : str
        Registry key.  One of ``"RMS_CV"``, ``"MaxEnt_SPRT"``, ``"SST_SVD"``.
    base_params : dict
        Baseline parameter dict (same structure as ``INDICATOR_CONFIG["params"]``
        used by the existing indicator runners).
        The resolved parameter(s) will be merged on top of this dict;
        everything else remains unchanged.
    resolution : ParameterResolutionConfig
        Which variable to solve, which to fix, and how to round.
    strict_constraints : bool
        If True, a constraint failure causes the indicator to be skipped
        (with a warning logged).  If False, the run proceeds regardless.
    """

    indicator_id: str
    base_params: Dict[str, Any]
    resolution: ParameterResolutionConfig
    strict_constraints: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# Top-level runner configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RunnerConfig:
    """
    Top-level configuration object passed to :class:`~effective_window.runner.WindowRunner`.

    Parameters
    ----------
    window_spec : WindowSpec
        Target window T_des — shared across all indicators.
    indicators : list of IndicatorWindowConfig
        One entry per indicator to run.
    show_plots : bool
        Generate production plots after the run.
    debug_level : int
        Verbosity level for :class:`~effective_window.debug.DebugManager`.

        =====  =====================
        0      OFF  — no output
        1      INFO — key events
        2      VERBOSE — per-indicator detail
        3      DEBUG — all plots
        =====  =====================
    """

    window_spec: WindowSpec
    indicators: List[IndicatorWindowConfig]
    show_plots: bool = False
    debug_level: int = 0
