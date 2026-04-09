# Comentario: exporta la API pública de la librería
from .utils.signals import five_senos, signal_1
from .utils.rms import rms_sequence
from .lib.cv_monitor import CVOnlineConfig, CVOnlineState, CVOnlineMonitor
from .viz.plots import plot_signal, plot_rms, plot_cv
from .lib.runner import run_rms_cv
from .utils.types import SignalData, IndicatorResult
from .utils.hdf5_utils import HDF5Reader
from .viz.rms_cv_plots import plots_rms_cv

__all__ = [
    "five_senos",
    "signal_1",
    SignalData,
    IndicatorResult,
    HDF5Reader,
    run_rms_cv,
    "rms_sequence",
    "CVOnlineConfig",
    "CVOnlineState",
    "CVOnlineMonitor",
    "plot_signal",
    "plot_rms",
    "plot_cv",
    "plots_rms_cv",
]

__version__ = "0.1.0"
