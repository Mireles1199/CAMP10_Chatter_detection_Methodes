from __future__ import annotations
import numpy as np
from typing import Literal, Optional, Tuple
from scipy.signal import hilbert, savgol_filter, medfilt

def hilbert_features(
    imf: np.ndarray ,
    fs: float,
    phase_diff_mode: Literal["first_diff", "savitzky_golay"] = "first_diff",
    sg_window: int = 11,
    sg_polyorder: int = 2,
    f_inst_smooth_median: Optional[int] = None,
) -> Tuple[np.ndarray , np.ndarray ]:
    """Obtiene amplitud y frecuencia instantánea del IMF mediante Hilbert.

    Args:
        imf (np.ndarray): IMF 1D seleccionado.
        fs (float): Frecuencia de muestreo (Hz).
        phase_diff_mode (Literal["first_diff","savitzky_golay"]): Estrategia de derivación de fase.
        sg_window (int): Ventana de Savitzky-Golay (si `phase_diff_mode` = "savitzky_golay").
        sg_polyorder (int): Orden del polinomio en Savitzky-Golay.
        f_inst_smooth_median (Optional[int]): Tamaño de kernel impar (>1) para mediana sobre f_inst.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (A, f_inst) con amplitud y frecuencia instantánea.

    Notas:
        - La fase se desenvuelve y se deriva; f_inst = dphi * fs / (2*pi).
        - Si `f_inst_smooth_median` se provee, se aplica filtro de mediana a f_inst.
    """
    z: np.ndarray  = hilbert(imf)
    A: np.ndarray  = np.abs(z)
    phase: np.ndarray  = np.unwrap(np.angle(z))

    if phase_diff_mode == "savitzky_golay":
        if sg_window % 2 == 0:
            sg_window += 1
        if sg_window < sg_polyorder + 2:
            sg_window = sg_polyorder + 3 if (sg_polyorder + 3) % 2 == 1 else sg_polyorder + 4
        phase_s: np.ndarray  = savgol_filter(phase, sg_window, sg_polyorder, mode='interp')
        dphi: np.ndarray  = np.gradient(phase_s)
    else:
        dphi = np.diff(phase, prepend=phase[0])

    f_inst: np.ndarray  = (dphi * fs) / (2.0 * np.pi)

    if f_inst_smooth_median is not None and f_inst_smooth_median > 1:
        ksize: int = int(f_inst_smooth_median) | 1
        f_inst = medfilt(f_inst, kernel_size=ksize)

    return A, f_inst


# -----------------------------
# HHS (opcional)
# -----------------------------

def build_hhs(
    A: np.ndarray ,
    f_inst: np.ndarray ,
    fs: float,
    fmax: float = 2000.0,
    fbin_hz: float = 5.0,
    energy_mode: Literal["A2", "A"] = "A2",
) -> Tuple[np.ndarray , np.ndarray ]:
    """Construye el Hilbert-Huang Spectrum (HHS) discreto en un grid de frecuencias.

    Args:
        A (np.ndarray): Amplitud instantánea A(t).
        f_inst (np.ndarray): Frecuencia instantánea f_inst(t) [Hz].
        fs (float): Frecuencia de muestreo, no se usa directamente pero se expone para consistencia.
        fmax (float): Frecuencia máxima del grid [Hz].
        fbin_hz (float): Tamaño del bin en frecuencia [Hz].
        energy_mode (Literal["A2","A"]): Si usar energía A^2 o amplitud A para acumular.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (H, fgrid) con H de tamaño (Fbins, N) y vector de frecuencias.

    Raises:
        ValueError: Si `fbin_hz` no es positivo.

    Notas:
        - Cada muestra t se asigna a un bin k = floor(f_inst/fbin_hz) y se acumula A^2 o A.
        - No es un estimador espectral clásico; es un mapa tiempo-frecuencia tipo HHT.
    """
    if fbin_hz <= 0:
        raise ValueError("fbin_hz debe ser > 0")

    Fbins: int = int(np.ceil(fmax / fbin_hz))
    fgrid: np.ndarray = np.arange(Fbins) * fbin_hz
    H: np.ndarray = np.zeros((Fbins, len(A)), dtype=float)
    E: np.ndarray = A**2 if energy_mode == "A2" else A

    # 1) Recortamos al rango [0, fmax)
    f_inst_clipped = np.clip(f_inst, 0, fmax - 1e-9)

    # 2) Reemplazamos NaN (y, si quieres, infinitos) por algo seguro
    #    - nan → 0.0 (por ejemplo)
    #    - +inf → fmax - 1e-9
    #    - -inf → 0.0
    f_inst_clean = np.nan_to_num(
        f_inst_clipped,
        nan=0.0,
        posinf=fmax - 1e-9,
        neginf=0.0,
    )

    # 3) Pasamos a índices de bin
    idx: np.ndarray = np.floor(f_inst_clean / fbin_hz).astype(np.intp)

    # 4) Clip final de seguridad por si acaso
    idx = np.clip(idx, 0, Fbins - 1)

    cols: np.ndarray = np.arange(len(A), dtype=np.intp)

    H[idx, cols] += E

    return H, fgrid

