"""
sweep/__init__.py
=================
Public API for the discrete-parameter sweep study package.

Typical usage
-------------
    from sweep import StudyBasis, SweepMode, SweepResult
    from sweep import enumerate_feasible, build_indicator_config, run_combo
    from sweep.debug import DebugManager
"""

from .basis          import StudyBasis
from .enumerator     import SweepMode, enumerate_feasible
from .config_builder import build_indicator_config
from .metrics        import MetricResult, evaluate
from .run_one        import RunResult, run_combo
from .sweep_result   import SweepResult
from .debug          import DebugManager

__all__ = [
    "StudyBasis",
    "SweepMode",
    "enumerate_feasible",
    "build_indicator_config",
    "MetricResult",
    "evaluate",
    "RunResult",
    "run_combo",
    "SweepResult",
    "DebugManager",
]
