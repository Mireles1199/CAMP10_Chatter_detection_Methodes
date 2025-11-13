from __future__ import annotations
# Comentario: API pública del paquete

from .utils.signal_chatter import make_chatter_like_signal, amplitude_spectrum
from .utils.signal_senos import five_senos
from .lib.core import detect_chatter_from_force
from .lib.datatypes import ChatterResult
from .viz.plotting import (
    plot_imfs,
    plot_imf_seleccionado,
    plot_imfs_separados,
    plot_tendencia,
    plot_HHS,
)

__all__ = [
    "make_chatter_like_signal",
    "amplitude_spectrum",
    "five_senos",
    "detect_chatter_from_force",
    "ChatterResult",
    "plot_imfs",
    "plot_imf_seleccionado",
    "plot_imfs_separados",
    "plot_tendencia",
    "plot_HHS",
]