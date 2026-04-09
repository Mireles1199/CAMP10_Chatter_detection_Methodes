from __future__ import annotations
import numpy as np
from typing import Optional, Tuple


def _maybe_detrend(x: np.ndarray ) -> np.ndarray :
    """Elimina tendencia lineal (detrend) de una señal.

    Args:
        x (np.ndarray): Señal 1D original (tipo real).

    Returns:
        np.ndarray: Señal 1D sin tendencia lineal (misma longitud que `x`).

    Notas:
        - Se usa una regresión lineal simple (mínimos cuadrados) sobre un eje temporal
          normalizado (media 0, var ~1) para mayor estabilidad numérica.
        - No se modifica la entrada in-place; se retorna una nueva vista/array.
    """
    n: int = len(x)
    t: np.ndarray  = np.arange(n, dtype=float)
    t = (t - t.mean()) / (t.std() + 1e-12)
    A: np.ndarray  = np.vstack([t, np.ones_like(t)]).T
    m, b = np.linalg.lstsq(A, x, rcond=None)[0]
    return x - (m * t + b)



def _maybe_clip(x: np.ndarray , clip: Optional[Tuple[float, float]]) -> np.ndarray :
    """Aplica recorte (clip) duro a la señal si se provee un rango.

    Args:
        x (np.ndarray): Señal 1D de entrada.
        clip (Optional[Tuple[float, float]]): Par (vmin, vmax). Si es None, no se aplica.

    Returns:
        np.ndarray: Señal recortada (o la original si `clip` es None).

    Precaución:
        - El clipping duro puede distorsionar espectro y fase; úsese solo para mitigar outliers.
    """
    if clip is None:
        return x
    vmin, vmax = clip
    return np.clip(x, vmin, vmax)



def _maybe_bandlimit(x: np.ndarray , fs: float, band: Optional[Tuple[float, float]]) -> np.ndarray :
    """Filtra band-pass (Butterworth) si se especifica banda válida.

    Args:
        x (np.ndarray): Señal 1D real.
        fs (float): Frecuencia de muestreo en Hz.
        band (Optional[Tuple[float, float]]): Par (lo, hi) en Hz. Si None, no se filtra.

    Returns:
        np.ndarray: Señal filtrada (o la original si `band` es None).

    Raises:
        ImportError: Si `scipy.signal` no está disponible.

    Detalles:
        - Orden fijo N=4; se usa filtfilt para fase cero.
        - Asegura límites (0, fs/2) para evitar errores en normalización.
    """
    if band is None:
        return x
    lo, hi = band
    if hi >= fs * 0.5:
        hi = fs * 0.5 * 0.99
    if lo <= 0:
        lo = 1e-6
    try:
        from scipy.signal import butter, filtfilt
    except Exception as e:  # pragma: no cover - import guard
        raise ImportError("Se requiere scipy.signal para bandlimit_pre") from e
    b, a = butter(N=4, Wn=[lo/(fs*0.5), hi/(fs*0.5)], btype='band')
    return filtfilt(b, a, x).astype(x.dtype, copy=False)

