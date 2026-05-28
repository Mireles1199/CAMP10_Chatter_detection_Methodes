"""green_integral — Green Integral chatter detection indicator.

Provides a clean, package-level API for detecting machining chatter
using the Green Integral Method (area growth of displacement-velocity
phase-plane cycles).

Quick start
-----------
    from green_integral import SignalData, run_green_integral, plots_green_integral
    from green_integral.logging_setup import configure_logging, LOGGING_LEVELS

    configure_logging(level=LOGGING_LEVELS["info"])

    sig = SignalData(t=t, displacement=x, velocity=v, name="test")
    config = {
        "func": "Default",
        "params": {"f_modal": 150.0, "num_T": 6, "dt": 0.005},
    }
    result = run_green_integral(sig, config)
    plots_green_integral(signal=sig, result=result)
"""

from __future__ import annotations

import logging as _logging

# -----------------------------------------------------------------------
# Register INFO_PLUS custom level (level 15, between DEBUG and INFO)
# -----------------------------------------------------------------------
INFO_PLUS_LEVEL: int = 15
_logging.addLevelName(INFO_PLUS_LEVEL, "INFO_PLUS")


def _add_verbose_method(level: int) -> None:
    """Monkey-patch ``Logger.verbose()`` that logs at *level*."""
    def verbose(self, msg, *args, **kwargs):  # type: ignore[override]
        if self.isEnabledFor(level):
            self._log(level, msg, args, **kwargs)

    if not hasattr(_logging.Logger, "verbose"):
        _logging.Logger.verbose = verbose  # type: ignore[attr-defined]


_add_verbose_method(INFO_PLUS_LEVEL)

# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------
from .utils.types import (  # noqa: E402
    SignalData,
    GreenIntegralConfig,
    GreenIntegralResult,
    FixedWindowConfig,
    FixedWindowResult,
)
from .utils.hdf5_utils import HDF5Reader  # noqa: E402
from .lib.runner import run_green_integral, INDICATOR_CONFIG  # noqa: E402
from .lib.runner_fixed import run_fixed_window, FIXED_WINDOW_CONFIG  # noqa: E402
from .viz.green_integral_plots import plots_green_integral, plots_fixed_window, plots_signal_diagnostics  # noqa: E402

__version__: str = "0.1.0"

__all__ = [
    # original indicator
    "SignalData",
    "GreenIntegralConfig",
    "GreenIntegralResult",
    "HDF5Reader",
    "run_green_integral",
    "plots_green_integral",
    "INDICATOR_CONFIG",
    "INFO_PLUS_LEVEL",
    # fixed-window indicator
    "FixedWindowConfig",
    "FixedWindowResult",
    "run_fixed_window",
    "FIXED_WINDOW_CONFIG",
    "plots_fixed_window",
    "plots_signal_diagnostics",
]
