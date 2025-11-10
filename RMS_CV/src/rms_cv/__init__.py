# Comentario: exporta la API pública de la librería
from .utils.signals import five_senos, signal_1
from .utils.rms import rms_sequence
from .lib.cv_monitor import CVOnlineConfig, CVOnlineState, CVOnlineMonitor
from .viz.plots import plot_signal, plot_rms, plot_cv

__all__ = [
    "five_senos",
    "signal_1",
    "rms_sequence",
    "CVOnlineConfig",
    "CVOnlineState",
    "CVOnlineMonitor",
    "plot_signal",
    "plot_rms",
    "plot_cv",
]

__version__ = "0.1.0"
