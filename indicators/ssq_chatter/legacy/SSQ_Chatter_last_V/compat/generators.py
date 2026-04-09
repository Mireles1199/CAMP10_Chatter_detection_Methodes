# Comentario: compatibilidad para generadores de señal
from __future__ import annotations
from typing import Tuple, Optional
import numpy as np

# Comentario: reexportar implementaciones originales para mantener fidelidad
from ..legacy.utils.signal_generators import five_senos as five_senos  # noqa: F401
from ..legacy.utils.signal_generators import signal_1 as signal_1      # noqa: F401
