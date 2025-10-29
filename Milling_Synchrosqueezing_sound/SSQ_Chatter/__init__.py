# Comentario: exportar la API pública de la librería
from .lib.core import sqq_chatter
from .lib.detection import detectar_chatter_3sigma
from .utils.signal_generators import five_senos, signal_1
from .utils.tf_windows import extract_local_windows, compute_svd

__all__ = [
    "sqq_chatter",
    "detectar_chatter_3sigma",
    "five_senos",
    "signal_1",
    "extract_local_windows",
    "compute_svd",
]
