"""Phase-space diagnostics for the fixed-window Green integral indicator.

Functions
---------
estimate_center       : suavizado lento de (x, v) → centro (cx, cv).
center_trajectory     : quita el centro lento → (xr, vr).
compute_local_phase   : fase local phi = unwrap(arctan2(vr, xr)), dphi.
drift_ratio           : rho = desplazamiento_centro / radio_local_medio.

Uso típico (dentro del debug loop de runner_fixed.py):
    cx, cv = estimate_center(q_win, v_win, center_win)
    xr, vr = center_trajectory(q_win, v_win, cx, cv)
    phi, dphi = compute_local_phase(xr, vr)
    rho = drift_ratio(cx, cv, xr, vr)
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Media móvil causal (modo 'same') usando np.convolve.

    Los primeros w-1 puntos se rellenan repitiendo el primer valor válido
    para evitar artefactos de borde.
    """
    if w <= 1:
        return x.copy()
    kernel = np.ones(w) / w
    out = np.convolve(x, kernel, mode="full")[:len(x)]
    # Corregir los bordes iniciales (divisor real < w)
    for i in range(min(w - 1, len(x))):
        out[i] = np.mean(x[: i + 1])
    return out


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def estimate_center(
    x: np.ndarray,
    v: np.ndarray,
    center_win: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estima el centro lento de la trayectoria (cx, cv).

    Parameters
    ----------
    x, v        : señales de desplazamiento y velocidad.
    center_win  : semi-ancho de la ventana de suavizado en muestras.
                  0 → usa la media global (escalar broadcast).
                  > 0 → media móvil de anchura ``2*center_win + 1``.
                  Debe ser < len(x)/2 para que tenga sentido.

    Returns
    -------
    cx, cv : arrays del mismo tamaño que x, v.
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)

    if center_win <= 0:
        cx = np.full_like(x, np.mean(x))
        cv = np.full_like(v, np.mean(v))
    else:
        w = 2 * center_win + 1
        w = min(w, len(x))   # no mayor que la señal
        try:
            from scipy.ndimage import uniform_filter1d
            cx = uniform_filter1d(x, size=w, mode="nearest")
            cv = uniform_filter1d(v, size=w, mode="nearest")
        except ImportError:
            cx = _moving_average(x, w)
            cv = _moving_average(v, w)

    return cx, cv


def center_trajectory(
    x: np.ndarray,
    v: np.ndarray,
    cx: np.ndarray,
    cv: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Quita el centro lento: xr = x - cx,  vr = v - cv."""
    return np.asarray(x) - np.asarray(cx), np.asarray(v) - np.asarray(cv)


def compute_local_phase(
    xr: np.ndarray,
    vr: np.ndarray,
    t: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fase local phi y su derivada dphi.

    Parameters
    ----------
    xr, vr : trayectoria centrada.
    t      : array de tiempos (opcional). Si se da, dphi = dphi/dt.
             Si no, dphi = dphi/muestra.

    Returns
    -------
    phi  : fase desenvolta [rad].
    dphi : derivada de phi (mismo largo que phi, NaN en el último punto).
    """
    xr = np.asarray(xr, dtype=float)
    vr = np.asarray(vr, dtype=float)

    phi = np.unwrap(np.arctan2(vr, xr))

    dphi = np.full_like(phi, np.nan)
    if t is not None:
        dt = np.diff(np.asarray(t, dtype=float))
        dt[dt == 0] = np.nan
        dphi[:-1] = np.diff(phi) / dt
    else:
        dphi[:-1] = np.diff(phi)

    return phi, dphi


def drift_ratio(
    cx: np.ndarray,
    cv: np.ndarray,
    xr: np.ndarray,
    vr: np.ndarray,
) -> float:
    """Razón deriva / radio local (ρ).

    ρ = ||centro_fin - centro_ini|| / mediana(||xr, vr||)

    ρ ≫ 1  →  caracol alargado; la deriva domina sobre la oscilación.
    ρ ≈ 0  →  espiral compacta centrada; poca deriva.

    Returns
    -------
    rho : float (nan si el radio local es cero).
    """
    cx = np.asarray(cx, dtype=float)
    cv = np.asarray(cv, dtype=float)

    delta_cx = cx[-1] - cx[0]
    delta_cv = cv[-1] - cv[0]
    drift_mag = float(np.sqrt(delta_cx ** 2 + delta_cv ** 2))

    r_local = np.sqrt(xr ** 2 + vr ** 2)
    r_med = float(np.median(r_local[np.isfinite(r_local)]))

    if r_med == 0.0 or np.isnan(r_med):
        return float("nan")
    return drift_mag / r_med
