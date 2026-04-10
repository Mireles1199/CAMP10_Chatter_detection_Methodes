"""rms_cv — RMS/CV online chatter detection indicator.

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

# ── Public API exports ────────────────────────────────────────────────────────
from .utils.signals import five_senos, signal_1
from .utils.rms import rms_sequence
from .lib.cv_monitor import CVOnlineConfig, CVOnlineState, CVOnlineMonitor
from .viz.plots import plot_signal, plot_rms, plot_cv
from .lib.runner import run_rms_cv
from .utils.types import SignalData, IndicatorResult
from .utils.hdf5_utils import HDF5Reader
from .viz.rms_cv_plots import plots_rms_cv

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
]

__version__ = "0.1.0"
