from __future__ import annotations
import numpy as np
from typing import Tuple, Union, Optional



def _time_axis(n: int, fs: Optional[float]) -> Tuple[np.ndarray , str]:
    """Devuelve eje x y etiqueta apropiada según disponibilidad de `fs`.

    Args:
        n (int): Número de muestras.
        fs (Optional[float]): Frecuencia de muestreo (Hz) o None.

    Returns:
        Tuple[np.ndarray, str]: (x, xlabel) donde x es tiempo (s) si `fs` válido, o índices.
    """
    if fs is None or fs <= 0:
        return np.arange(n), "muestras"
    else:
        return np.arange(n) / fs, "s"



def _smoothstep(z: Union[np.ndarray , float]) -> Union[np.ndarray , float]:
    """Función paso-suave: 0 fuera, y 3z^2 - 2z^3 para z en [0,1].

    Args:
        z (np.ndarray | float): Valor(es) de entrada.

    Returns:
        np.ndarray | float: Salida con la misma forma que `z`.
    """
    z = np.clip(z, 0.0, 1.0)
    return z*z*(3 - 2*z)  # type: ignore[operator]
