"""rms_cv -- RMS/CV online chatter detection indicator.

This package detects machining chatter by computing Root Mean Square (RMS)
values over a sliding window of the vibration signal, then monitoring the
Coefficient of Variation (CV) of that RMS sequence online.  A sudden rise
in CV indicates the onset of chatter.

Typical usage::

    from rms_cv import SignalData, run_rms_cv, plots_rms_cv

    sig = SignalData(
        t_analysis=t, signal_analysis=v, fs=fs,
        # ... other required fields ...
    )
    config = {
        "func": "Default",
        "params": {
            "n_max": 20,
            "samples_per_window": 512,
            "cv_threshold": 1.05,
        },
    }
    result = run_rms_cv(sig, config)
    plots_rms_cv(signal=sig, result=result, show_signal=True)

Public API
----------
:class:`SignalData`
    Input signal container.
:class:`IndicatorResult`
    Detection output (CV sequence, detection instants).
:func:`run_rms_cv`
    Top-level pipeline dispatcher.
:func:`rms_sequence`
    Windowed RMS computation.
:class:`CVOnlineConfig`
    Configuration for the online CV monitor.
:class:`CVOnlineState`
    Mutable running-statistics state.
:class:`CVOnlineMonitor`
    Streaming CV monitor.
:class:`HDF5Reader`
    Eager-loading HDF5 file reader.
:func:`plots_rms_cv`
    Coordinated publication figures.
"""

# ── Custom log level (INFO_PLUS = 15, between DEBUG=10 and INFO=20) ───────────
INFO_PLUS_LEVEL = 15


def _register_info_plus_level() -> None:
    import logging as _lg
    if not hasattr(_lg, "INFO_PLUS"):
        _lg.INFO_PLUS = INFO_PLUS_LEVEL  # type: ignore[attr-defined]
    if _lg.getLevelName(INFO_PLUS_LEVEL) != "INFO_PLUS":
        _lg.addLevelName(INFO_PLUS_LEVEL, "INFO_PLUS")
    if not hasattr(_lg.Logger, "verbose"):
        def _verbose(self, msg, *args, **kwargs):
            if self.isEnabledFor(INFO_PLUS_LEVEL):
                self._log(INFO_PLUS_LEVEL, msg, args, **kwargs)
        _lg.Logger.verbose = _verbose  # type: ignore[attr-defined]
    if not hasattr(_lg.Logger, "info_plus"):
        _lg.Logger.info_plus = _lg.Logger.verbose  # type: ignore[attr-defined]
    if not hasattr(_lg, "info_plus"):
        def _module_info_plus(msg, *args, **kwargs):
            _lg.log(INFO_PLUS_LEVEL, msg, *args, **kwargs)
        _lg.info_plus = _module_info_plus  # type: ignore[attr-defined]


_register_info_plus_level()
del _register_info_plus_level

# ── Public API exports ────────────────────────────────────────────────────────
from .utils.signals import five_senos, signal_1
from .utils.rms import rms_sequence
from .lib.cv_monitor import CVOnlineConfig, CVOnlineState, CVOnlineMonitor
from .viz.plots import plot_signal, plot_rms, plot_cv
from .lib.runner import run_rms_cv
from .utils.types import SignalData, IndicatorResult
from .utils.hdf5_utils import HDF5Reader
from .viz.rms_cv_plots import plots_rms_cv
from .logging_setup import LOGGING_LEVELS

__all__ = [
    # Signal generators
    "five_senos",
    "signal_1",
    # Data containers
    "SignalData",
    "IndicatorResult",
    # I/O
    "HDF5Reader",
    # Pipeline
    "run_rms_cv",
    "rms_sequence",
    # CV monitor
    "CVOnlineConfig",
    "CVOnlineState",
    "CVOnlineMonitor",
    # Visualisation (legacy)
    "plot_signal",
    "plot_rms",
    "plot_cv",
    # Visualisation (main)
    "plots_rms_cv",
    # Logging
    "INFO_PLUS_LEVEL",
    "LOGGING_LEVELS",
]

__version__ = "0.1.0"
