"""chatter_hht_pyemd.py
------------------------
Pipeline robusto para identificación temprana de chatter a partir de fuerza F(t) usando:
  - EMD/EEMD/CEEMDAN de **PyEMD** (https://pypi.org/project/PyEMD/)
  - Transformada de Hilbert (scipy.signal.hilbert)
  - Métrica de conteo de energía en banda en el IMF modal (ej. 450–600 Hz)

IMPORTANTE: Este módulo **no implementa** la etapa de decisión/baseline; entrega artefactos intermedios
(energía en banda, HHS, etc.) para que otra capa decida.

Dependencias:
  pip install PyEMD scipy numpy matplotlib
"""

#%%
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Dict, Any

# Comentario: importaciones con manejo de errores para dependencias científicas
try:
    from scipy.signal import hilbert, savgol_filter
except Exception as e:
    raise ImportError("Se requiere scipy: pip install scipy") from e

try:
    # PyEMD: https://github.com/laszukdawid/PyEMD
    from PyEMD import EMD, EEMD, CEEMDAN

except Exception as e:
    raise ImportError("Se requiere PyEMD: pip install PyEMD") from e

# Comentario: utilidades externas usadas en la demo (no esenciales al pipeline)
from signal_chatter import make_chatter_like_signal, amplitude_spectrum
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator, AutoMinorLocator  # noqa: F401 (AutoMinorLocator sin uso directo)
from scipy.signal import savgol_filter  # redundante pero mantiene import original
import numpy.polynomial.chebyshev as cheb

from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit

import PyEMD, inspect

print("PyEMD =", PyEMD)
print("PyEMD.__file__ =", getattr(PyEMD, "__file__", None))
print("EMD =", EEMD)
print("type(EMD) =", type(EEMD))
print("module de EMD =", inspect.getmodule(EEMD))


# -----------------------------
# Utilidades de preprocesado
# -----------------------------
def _maybe_detrend(x: np.ndarray) -> np.ndarray:
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
    n = len(x)
    # Comentario: eje temporal normalizado para estabilidad en la regresión
    t = np.arange(n, dtype=float)
    t = (t - t.mean()) / (t.std() + 1e-12)
    A = np.vstack([t, np.ones_like(t)]).T
    # Comentario: resolución por mínimos cuadrados (pendiente m e intercepto b)
    m, b = np.linalg.lstsq(A, x, rcond=None)[0]
    return x - (m * t + b)


def _maybe_clip(x: np.ndarray, clip: Optional[Tuple[float, float]]) -> np.ndarray:
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


def _maybe_bandlimit(x: np.ndarray, fs: float, band: Optional[Tuple[float, float]]) -> np.ndarray:
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
    # Comentario: asegurar límites dentro del rango físico de Nyquist
    if hi >= fs * 0.5:
        hi = fs * 0.5 * 0.99
    if lo <= 0:
        lo = 1e-6
    try:
        from scipy.signal import butter, filtfilt
    except Exception as e:
        raise ImportError("Se requiere scipy.signal para bandlimit_pre") from e
    # Comentario: diseño Butterworth banda (frecuencias normalizadas por Nyquist)
    b, a = butter(N=4, Wn=[lo/(fs*0.5), hi/(fs*0.5)], btype='band')
    # Comentario: filtfilt para evitar desfase (aplica filtro hacia delante y atrás)
    return filtfilt(b, a, x).astype(x.dtype, copy=False)


# -----------------------------
# Selección de IMF modal
# -----------------------------
def select_imf_near_band(imfs: np.ndarray, fs: float, band: Tuple[float, float]) -> int:
    """Selecciona el índice del IMF con mayor energía en una banda de frecuencia.

    Args:
        imfs (np.ndarray): Matriz (K, N) de IMFs (cada fila es un IMF).
        fs (float): Frecuencia de muestreo (Hz).
        band (Tuple[float, float]): Banda objetivo (lo, hi) en Hz.

    Returns:
        int: Índice k del IMF con mayor energía espectral dentro de la banda.

    Raises:
        ValueError: Si `imfs` no es 2D.

    Notas:
        - Usa ventana Hann y FFT de tamaño `n` (siguiente potencia de 2 >= N) para estimar energía.
        - La energía en banda se calcula como suma de |X|^2 con máscara [lo, hi].
    """
    lo, hi = band
    if imfs.ndim != 2:
        raise ValueError("imfs debe tener forma (K, N)")
    K, N = imfs.shape
    # Comentario: tamaño de FFT como potencia de 2 próxima para eficiencia
    n = int(1 << (N - 1).bit_length())
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    mask = (freqs >= lo) & (freqs <= hi)
    best_k, best_e = 0, -np.inf
    win = np.hanning(N)
    for k in range(K):
        x = imfs[k]
        # Comentario: magnitud espectral con windowing para reducir leakage
        X = np.fft.rfft(x * win, n=n)
        E = np.sum(np.abs(X[mask])**2)  # energía en banda
        if E > best_e:
            best_e, best_k = E, k
    return best_k


# -----------------------------
# Hilbert y frecuencia instantánea
# -----------------------------
def hilbert_features(
    imf: np.ndarray,
    fs: float,
    phase_diff_mode: Literal["first_diff", "savitzky_golay"] = "first_diff",
    sg_window: int = 11,
    sg_polyorder: int = 2,
    f_inst_smooth_median: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
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
    z = hilbert(imf)  # Comentario: señal analítica por transformada de Hilbert
    A = np.abs(z)     # Comentario: amplitud instantánea
    phase = np.unwrap(np.angle(z))  # Comentario: fase desenrollada

    if phase_diff_mode == "savitzky_golay":
        # Comentario: si ventana par, se ajusta a impar; se asegura ventana válida
        if sg_window % 2 == 0:
            sg_window += 1
        if sg_window < sg_polyorder + 2:
            sg_window = sg_polyorder + 3 if (sg_polyorder + 3) % 2 == 1 else sg_polyorder + 4
        phase_s = savgol_filter(phase, sg_window, sg_polyorder, mode='interp')
        dphi = np.gradient(phase_s)  # Comentario: derivada numérica suave
    else:
        dphi = np.diff(phase, prepend=phase[0])  # Comentario: diferencia hacia delante

    f_inst = (dphi * fs) / (2.0 * np.pi)

    if f_inst_smooth_median is not None and f_inst_smooth_median > 1:
        # Comentario: filtro de mediana robusto a outliers; tamaño impar
        from scipy.signal import medfilt
        ksize = int(f_inst_smooth_median) | 1
        f_inst = medfilt(f_inst, kernel_size=ksize)

    return A, f_inst


# -----------------------------
# HHS (opcional)
# -----------------------------
def build_hhs(
    A: np.ndarray,
    f_inst: np.ndarray,
    fs: float,
    fmax: float = 2000.0,
    fbin_hz: float = 5.0,
    energy_mode: Literal["A2", "A"] = "A2",
) -> Tuple[np.ndarray, np.ndarray]:
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
    Fbins = int(np.ceil(fmax / fbin_hz))
    fgrid = np.arange(Fbins) * fbin_hz
    H = np.zeros((Fbins, len(A)), dtype=float)
    E = A**2 if energy_mode == "A2" else A
    # Comentario: proyección directa de f_inst(t) al bin correspondiente
    idx = np.floor(np.clip(f_inst, 0, fmax - 1e-9) / fbin_hz).astype(int)
    H[idx, np.arange(len(A))] += E
    return H, fgrid


# -----------------------------
# Conteo en banda por ventana
# -----------------------------
def band_counts_over_time(
    A: np.ndarray,
    f_inst: np.ndarray,
    band: Tuple[float, float] = (450.0, 600.0),
    energy_mode: Literal["A2", "A"] = "A2",
    thr_mode: Optional[Literal["mad", "percentile", "none"]] = "mad",
    thr_k_mad: float = 3.0,
    thr_percentile: float = 95.0,
    count_win_samples: int = 100,
    count_step_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Cuenta, por ventana, muestras que caen en banda y (opcional) superan un umbral de energía.

    Args:
        A (np.ndarray): Amplitud instantánea A(t).
        f_inst (np.ndarray): Frecuencia instantánea f_inst(t) en Hz.
        band (Tuple[float, float]): Banda (lo, hi) de interés en Hz.
        energy_mode (Literal["A2","A"]): Modo para energía/umbral (A^2 o A).
        thr_mode (Optional[Literal["mad","percentile","none"]]): Estrategia de umbral; None/'none' desactiva.
        thr_k_mad (float): Multiplicador del MAD cuando `thr_mode="mad"`.
        thr_percentile (float): Percentil cuando `thr_mode="percentile"`.
        count_win_samples (int): Tamaño de ventana (muestras) para el conteo.
        count_step_samples (Optional[int]): Paso entre ventanas; por defecto igual a la ventana.

    Returns:
        Tuple[np.ndarray, np.ndarray, Dict[str, Any]]: (counts, centers, debug_meta).

    Detalles:
        - Si `thr_mode` es None/'none', se cuenta solo pertenencia a banda; si no,
          además se exige energía > umbral.
        - `centers` contiene el índice de muestra del centro de cada ventana.
    """
    if count_step_samples is None:
        count_step_samples = count_win_samples

    N = len(A)
    lo, hi = band

    # Comentario: seleccionar magnitud para umbral según `energy_mode`
    E = A**2 if energy_mode == "A2" else A

    # Umbral (opcional)
    thr = None
    if thr_mode in ("mad", "percentile"):
        if thr_mode == "mad":
            med = float(np.median(E))
            mad = float(np.median(np.abs(E - med))) + 1e-12  # Comentario: robustez numérica
            thr = med + thr_k_mad * mad
        else:  # 'percentile'
            thr = float(np.percentile(E, thr_percentile))
    elif thr_mode in ("none", None):
        thr = None
    else:
        raise ValueError("thr_mode debe ser 'mad', 'percentile' o 'none'")

    counts = []
    centers = []
    # Comentario: barrido por ventanas deslizantes
    for start in range(0, N - count_win_samples + 1, count_step_samples):
        sl = slice(start, start + count_win_samples)
        mask_band = (f_inst[sl] >= lo) & (f_inst[sl] <= hi)

        if thr is None:
            # Solo cuenta muestras en la banda
            hits = np.count_nonzero(mask_band)
        else:
            # Banda + energía por encima del umbral
            mask_energy = (E[sl] > thr)
            hits = np.count_nonzero(mask_band & mask_energy)

        counts.append(int(hits))
        centers.append(start + count_win_samples // 2)

    debug = {"thr": thr, "thr_mode": thr_mode, "band": band, "energy_mode": energy_mode}
    return np.asarray(counts), np.asarray(centers), debug


# -----------------------------
# Salida estructurada
# -----------------------------
@dataclass
class ChatterResult:
    """Contenedor de resultados intermedios del pipeline HHT/EMD.

    Atributos:
        imfs (Optional[np.ndarray]): Matriz (K, N) de IMFs o None si no se solicitó.
        k_selected (int): Índice del IMF elegido.
        selected_imf (np.ndarray): IMF seleccionado (o array vacío si no se solicitó).
        A (np.ndarray): Amplitud instantánea (o vacío).
        f_inst (np.ndarray): Frecuencia instantánea (o vacío).
        counts (np.ndarray): Conteos por ventana.
        t_counts_samples (np.ndarray): Centros de ventana (en índices de muestra).
        HHS (Optional[np.ndarray]): Hilbert-Huang Spectrum (o None si deshabilitado).
        fgrid (Optional[np.ndarray]): Grid de frecuencias asociado a HHS.
        meta (Dict[str, Any]): Metadatos del proceso (fs, umbral, etc.).
    """
    imfs: Optional[np.ndarray]
    k_selected: int
    selected_imf: np.ndarray
    A: np.ndarray
    f_inst: np.ndarray
    counts: np.ndarray
    t_counts_samples: np.ndarray
    HHS: Optional[np.ndarray]
    fgrid: Optional[np.ndarray]
    meta: Dict[str, Any]


# -----------------------------
# Función principal (PyEMD)
# -----------------------------
def detect_chatter_from_force(
    F: np.ndarray,
    fs: float,
    # Pre-procesado
    detrend: bool = False,
    clip: Optional[Tuple[float, float]] = None,
    bandlimit_pre: Optional[Tuple[float, float]] = None,
    # EMD/EEMD/CEEMDAN (PyEMD)
    emd_method: Literal["ceemdan", "emd", "eemd"] = "emd",
    emd_max_imfs: Optional[int] = None,      # PyEMD usa max_imf en algunas clases
    emd_max_siftings: Optional[int] = 50,    # límite de sifting por IMF
    ceemdan_noise_strength: float = 0.2,     # std relativo del ruido (CEEMDAN)
    ceemdan_ensemble_size: int = 100,        # realizaciones en el ensamble
    emd_rng_seed: Optional[int] = 42,
    # Selección de IMF
    imf_selection: Literal["auto", "index"] = "auto",
    imf_index: Optional[int] = None,
    band_chatter: Tuple[float, float] = (450.0, 600.0),
    band_selection_margin: float = 0.0,
    # Hilbert
    phase_diff_mode: Literal["first_diff", "savitzky_golay"] = "first_diff",
    sg_window: int = 11,
    sg_polyorder: int = 2,
    f_inst_smooth_median: Optional[int] = None,
    # HHS
    hhs_enable: bool = False,
    hhs_fmax: float = 2000.0,
    hhs_fbin_hz: float = 5.0,
    # Conteo
    count_win_samples: int = 100,
    count_step_samples: Optional[int] = None,
    energy_mode: Literal["A2", "A"] = "A2",
    thr_mode: Literal["mad", "percentile"] = "mad",
    thr_k_mad: float = 3.0,
    thr_percentile: float = 95.0,
    # Salidas
    return_imfs: bool = True,
    return_selected_imf: bool = True,
    return_hilbert: bool = True,
    return_counts: bool = True,
    return_hhs: bool = True,
    return_debug: bool = False,
) -> ChatterResult:
    """Pipeline principal con **PyEMD** (sin decisión/baseline).

    Args:
        F (np.ndarray): Señal de fuerza 1D (real).
        fs (float): Frecuencia de muestreo (Hz).
        detrend (bool): Si True, elimina tendencia lineal previa.
        clip (Optional[Tuple[float,float]]): Rango (vmin, vmax) para clipping duro.
        bandlimit_pre (Optional[Tuple[float,float]]): Filtro pasa-banda previo (Hz).
        emd_method (Literal["ceemdan","emd","eemd"]): Método de descomposición PyEMD.
        emd_max_imfs (Optional[int]): Número máximo de IMFs (si aplica).
        emd_max_siftings (Optional[int]): Máximo de iteraciones de sifting.
        ceemdan_noise_strength (float): Fuerza de ruido relativo en CEEMDAN.
        ceemdan_ensemble_size (int): Número de realizaciones (ensamble) para CEEMDAN/EEMD.
        emd_rng_seed (Optional[int]): Semilla RNG para reproducibilidad.
        imf_selection (Literal["auto","index"]): Estrategia de selección del IMF.
        imf_index (Optional[int]): Índice manual del IMF (si `imf_selection="index"`).
        band_chatter (Tuple[float,float]): Banda objetivo de chatter [Hz].
        band_selection_margin (float): Margen relativo para ampliar banda al seleccionar IMF.
        phase_diff_mode (Literal["first_diff","savitzky_golay"]): Derivada de fase.
        sg_window (int): Ventana Savitzky-Golay.
        sg_polyorder (int): Orden polinómico Savitzky-Golay.
        f_inst_smooth_median (Optional[int]): Kernel de mediana para suavizar f_inst.
        hhs_enable (bool): Si True, construye HHS.
        hhs_fmax (float): Fmáx del grid HHS [Hz].
        hhs_fbin_hz (float): Resolución en frecuencia del grid HHS [Hz].
        count_win_samples (int): Ventana de conteo (muestras).
        count_step_samples (Optional[int]): Paso entre ventanas de conteo.
        energy_mode (Literal["A2","A"]): Magnitud para umbral/conteo (A^2 o A).
        thr_mode (Literal["mad","percentile"]): Tipo de umbral (o "none" para desactivar).
        thr_k_mad (float): Multiplicador del MAD.
        thr_percentile (float): Percentil para umbral.
        return_imfs, return_selected_imf, return_hilbert, return_counts, return_hhs, return_debug:
            Flags para incluir/excluir artefactos en la salida.

    Returns:
        ChatterResult: Contenedor con IMFs, IMF seleccionado, A, f_inst, HHS (opcional), conteos y metadatos.

    Raises:
        ValueError: Si `F` no es 1D o parámetros fuera de rango.
        RuntimeError: Si la descomposición no produce IMFs.

    Notas:
        - La selección de IMF por defecto usa energía en banda objetivo con margen opcional.
        - La etapa de decisión/baseline se deja fuera a propósito.
    """
    if F.ndim != 1:
        raise ValueError("F debe ser un vector 1D")
    x = F.astype(float, copy=True)

    # ---- Preprocesado
    if detrend:
        x = _maybe_detrend(x)
    if clip is not None:
        x = _maybe_clip(x, clip)
    if bandlimit_pre is not None:
        x = _maybe_bandlimit(x, fs, bandlimit_pre)

    # ---- EMD/EEMD/CEEMDAN con PyEMD
    rng = np.random.default_rng(emd_rng_seed) if emd_rng_seed is not None else None  # noqa: F841 (usado para consistencia)
    seed = int(emd_rng_seed) if emd_rng_seed is not None else None

    if emd_method == "ceemdan":
        emd_max_imfs = -1
        decomp = CEEMDAN(random_seed=seed)
        # Comentario: parámetros CEEMDAN específicos
        decomp.noise_seed(seed)
        decomp.trials = int(ceemdan_ensemble_size)
        decomp.noise_strength = float(ceemdan_noise_strength)
        if emd_max_siftings is not None:
            decomp.max_iter = int(emd_max_siftings)
        imfs = decomp.ceemdan(x, max_imf=emd_max_imfs)
    elif emd_method == "eemd":
        decomp = EEMD(trials=ceemdan_ensemble_size, noise_seed=seed)
        if emd_max_siftings is not None:
            # Comentario: EEMD usa internamente EMD; se configura vía `EEMD.EMD`
            base_emd = EMD()
            base_emd.FIXE = int(emd_max_siftings)
            decomp.EMD = base_emd
        imfs = decomp.eemd(x, max_imf=emd_max_imfs)
    else:  # "emd"
        decomp = EMD()
        if emd_max_siftings is not None:
            # Comentario: en PyEMD, FIXE es el nº de iteraciones de sifting
            decomp.FIXE = int(emd_max_siftings)
        imfs = decomp.emd(x, max_imf=emd_max_imfs)

    if imfs.size == 0:
        raise RuntimeError("La descomposición EMD no produjo IMFs válidos.")
    if imfs.ndim == 1:
        imfs = imfs[np.newaxis, :]
    K, N = imfs.shape

    # ---- Selección de IMF
    lo, hi = band_chatter
    if band_selection_margin and band_selection_margin > 0:
        # Comentario: margen relativo para permitir desplazamiento de banda en selección
        bw = (hi - lo) * band_selection_margin
        sel_band = (max(0.0, lo - bw), hi + bw)
    else:
        sel_band = band_chatter

    if imf_selection == "index":
        if imf_index is None:
            raise ValueError("imf_index debe especificarse cuando imf_selection='index'")
        if not (0 <= imf_index < K):
            raise ValueError(f"imf_index fuera de rango (0..{K-1})")
        k_selected = imf_index
    else:
        k_selected = select_imf_near_band(imfs, fs, sel_band)

    selected_imf = imfs[k_selected]

    # ---- Hilbert
    A, f_inst = hilbert_features(
        selected_imf, fs,
        phase_diff_mode=phase_diff_mode,
        sg_window=sg_window,
        sg_polyorder=sg_polyorder,
        f_inst_smooth_median=f_inst_smooth_median,
    )

    # ---- HHS opcional
    HHS = None
    fgrid = None
    if hhs_enable:
        HHS, fgrid = build_hhs(
            A, f_inst, fs,
            fmax=hhs_fmax,
            fbin_hz=hhs_fbin_hz,
            energy_mode=energy_mode,
        )

    # ---- Conteo en banda
    counts, t_centers, debug = band_counts_over_time(
        A, f_inst,
        band=band_chatter,
        energy_mode=energy_mode,
        thr_mode=thr_mode,
        thr_k_mad=thr_k_mad,
        thr_percentile=thr_percentile,
        count_win_samples=count_win_samples,
        count_step_samples=count_step_samples,
    )

    meta: Dict[str, Any] = {
        "fs": fs,
        "K_imfs": K,
        "band_chatter": band_chatter,
        "k_selected": k_selected,
        "threshold": debug.get("thr"),
        "emd_method": emd_method,
        "count_win_samples": count_win_samples,
        "count_step_samples": count_step_samples or count_win_samples,
        "phase_diff_mode": phase_diff_mode,
    }
    if return_debug:
        meta["debug"] = debug

    return ChatterResult(
        imfs=imfs if return_imfs else None,
        k_selected=k_selected,
        selected_imf=selected_imf if return_selected_imf else np.array([]),
        A=A if return_hilbert else np.array([]),
        f_inst=f_inst if return_hilbert else np.array([]),
        counts=counts if return_counts else np.array([]),
        t_counts_samples=t_centers if return_counts else np.array([]),
        HHS=HHS if (hhs_enable and return_hhs) else None,
        fgrid=fgrid if (hhs_enable and return_hhs) else None,
        meta=meta,
    )

# =============================
# Utilidades de graficado (matplotlib)
# =============================
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # Para entornos sin matplotlib


def _time_axis(n: int, fs: float | None):
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


def plot_imfs_separados(imfs: np.ndarray, fs: float | None = None, max_to_plot: int | None = None, show: bool = True):
    """Crea una figura independiente por cada IMF (sin subplots).

    Args:
        imfs (np.ndarray): Matriz (K, N) de IMFs.
        fs (Optional[float]): Frecuencia de muestreo o None.
        max_to_plot (Optional[int]): Máximo de IMFs a graficar. Si None, grafica todos.
        show (bool): Si True, realiza `plt.show()` al final.

    Returns:
        list: Lista de objetos Figure creados.
    """
    if plt is None:
        raise ImportError("Falta matplotlib. Instala con: pip install matplotlib")
    if imfs.ndim != 2:
        raise ValueError("imfs debe tener forma (K, N)")
    K, N = imfs.shape
    if max_to_plot is None or max_to_plot > K:
        max_to_plot = K
    t, xlabel = _time_axis(N, fs)
    figs = []
    for k in range(max_to_plot):
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(t, imfs[k])
        ax.set_title(f"IMF {k+1}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Amplitud")
        figs.append(fig)
    if show:
        plt.show()
    return figs


def plot_imf_seleccionado(selected_imf: np.ndarray, 
                          fs: float | None = None, 
                          plot_spectrum: bool = False,
                          A: np.ndarray | None = None, 
                          f_inst: np.ndarray | None = None, 
                          show: bool = True):
    """Grafica el IMF seleccionado y opcionalmente A(t) y f_inst(t), en figuras separadas.

    Args:
        selected_imf (np.ndarray): IMF 1D a graficar.
        fs (Optional[float]): Frecuencia de muestreo (Hz).
        plot_spectrum (bool): Si True, grafica espectro del IMF en figura adicional.
        A (Optional[np.ndarray]): Amplitud instantánea para trazar (opcional).
        f_inst (Optional[np.ndarray]): Frecuencia instantánea para trazar (opcional).
        show (bool): Si True, hace `plt.show()` al final.

    Returns:
        list: Lista de figuras creadas.
    """
    if plt is None:
        raise ImportError("Falta matplotlib. Instala con: pip install matplotlib")
    N = len(selected_imf)
    t, xlabel = _time_axis(N, fs)

    figs = []

    # Figura 1: IMF seleccionado en el tiempo
    fig1 = plt.figure()
    ax1 = fig1.gca()
    ax1.plot(t, selected_imf)
    ax1.set_title("IMF seleccionado - dominio temporal")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Amplitud")
    figs.append(fig1)

    # Figura 2: Amplitud instantánea (si se provee)
    if A is not None and len(A) == N:
        fig2 = plt.figure()
        ax2 = fig2.gca()
        ax2.plot(t, A)
        ax2.set_title("Amplitud instantánea A(t)")
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("A")
        figs.append(fig2)

    # Figura 3: Frecuencia instantánea (si se provee)
    if f_inst is not None and len(f_inst) == N:
        fig3 = plt.figure()
        ax3 = fig3.gca()
        ax3.plot(t, f_inst)
        ax3.set_title("Frecuencia instantánea f_inst(t)")
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel("Hz")
        figs.append(fig3)

        # Spectrum de cada IMF
    if plot_spectrum:
        fig2, axes_2 = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        axes_2 = np.array(axes_2).reshape(-1)
        ax = axes_2[0]
        f, Pxx = amplitude_spectrum(selected_imf, fs=fs, normalize_to=0.1)
        ax.plot(f, Pxx)
        ax.set_title(f"Espectro IMF seleccionado")
        ax.set_xlabel("Frecuencia (Hz)" if fs is not None else "Frecuencia (muestras)")
        ax.set_ylabel("Amplitud")
        ax.set_xlim(0, 1000 if fs is not None else None)

        ax.xaxis.set_major_locator(MultipleLocator(100))

        fig2.tight_layout()

    if show:
        plt.show()
    return figs


def plot_imfs(imfs: np.ndarray,
                            fs: float | None = None,
                            plot_spectrum: bool = False,
                            max_to_plot: int | None = None, 
                            ncols: int = 1, show: bool = True):
    """Dibuja *todos* los IMFs en **una sola figura** usando subplots.

    Args:
        imfs (np.ndarray): Matriz (K, N) con los IMFs.
        fs (Optional[float]): Frecuencia de muestreo. Si None, el eje x será en muestras.
        plot_spectrum (bool): Si True, crea una segunda figura con espectros de IMFs.
        max_to_plot (Optional[int]): Límite superior de IMFs a graficar. Si None, grafica todos.
        ncols (int): Número de columnas en el mosaico de subplots.
        show (bool): Si True, hace plt.show() al final.

    Returns:
        Tuple[object, np.ndarray]: Figura y arreglo de ejes (subplots).
    """
    if plt is None:
        raise ImportError("Falta matplotlib. Instala con: pip install matplotlib")
    if imfs.ndim != 2:
        raise ValueError("imfs debe tener forma (K, N)")
    K, N = imfs.shape
    if max_to_plot is None or max_to_plot > K:
        max_to_plot = K
    if ncols < 1:
        ncols = 1
    nrows = int(np.ceil(max_to_plot / ncols))

    t, xlabel = _time_axis(N, fs)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(10, max(3, 2*nrows)))
    # Comentario: asegura que axes sea 1D para iterar fácilmente
    axes = np.array(axes).reshape(-1)

    for i in range(max_to_plot):
        ax = axes[i]
        ax.plot(t, imfs[i])
        ax.set_title(f"IMF {i+1}")
        ax.set_ylabel("Amplitud")
    # Oculta subplots vacíos si los hay
    for j in range(max_to_plot, len(axes)):
        axes[j].set_visible(False)

    axes[min(max_to_plot-1, len(axes)-1)].set_xlabel(xlabel)

    for ax in axes:
        # Comentario: asegurar ticks visibles en eje X
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)

    fig.tight_layout()

    # Spectrum de cada IMF
    if plot_spectrum:
        fig2, axes_2 = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(10, max(3, 2*nrows)))
        axes_2 = np.array(axes_2).reshape(-1)
        for i in range(max_to_plot):
            ax = axes_2[i]
            f, Pxx = amplitude_spectrum(imfs[i], fs=fs, normalize_to=0.1)
            ax.plot(f, Pxx)
            ax.set_title(f"Espectro IMF {i+1}")
            # ax.set_xlabel("Frecuencia (Hz)" if fs is not None else "Frecuencia (muestras)")
            ax.set_ylabel("Amplitud")
            ax.set_xlim(0, 1000 if fs is not None else None)

            # Asegurarse de mostrar los ticks del eje X en todos los subgráficos
            ax.tick_params(axis='x', )  # Ticks en el eje x
            ax.tick_params(axis='y', )  # Ticks en el eje y

            ax.xaxis.set_major_locator(MultipleLocator(100))

        for j in range(max_to_plot, len(axes_2)):
            axes_2[j].set_visible(False)

        axes_2[min(max_to_plot-1, len(axes_2)-1)].set_xlabel("Frecuencia (Hz)" if fs is not None else "Frecuencia (muestras)")
        fig2.tight_layout()

        #  Asegurarse de que los ticks del eje X sean visibles en todos los subgráficos
        for ax in axes_2:
            ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)


    if show:
        plt.show()
    return fig, axes


def _smoothstep(z):
    """Función paso-suave: 0 fuera, y 3z^2 - 2z^3 para z en [0,1].

    Args:
        z (np.ndarray | float): Valor(es) de entrada.

    Returns:
        np.ndarray | float: Salida con la misma forma que `z`.
    """
    # 0 fuera, 3z^2 - 2z^3 en [0,1]
    z = np.clip(z, 0.0, 1.0)
    return z*z*(3 - 2*z)


def plot_tendencia(t, y, tipo="lineal", grado=3, ventana=7,
                   ls='--', lw=3, alpha=1.0, color='k',
                   scatter_kwargs=None, trend_kwargs=None, devolver=False, n_bins=10,
                   show: bool = False):
    """Dibuja puntos y línea de tendencia según método elegido.

    Args:
        t (array-like): Eje x (tiempo o índice).
        y (array-like): Serie a suavizar/ajustar.
        tipo (str): 'lineal' | 'polinomial' | 'suavizada' | 'pchip' | 'sigmoide' | 'step_suave'.
        grado (int): Grado polinómico (si aplica).
        ventana (int): Ventana para Savitzky-Golay (si 'suavizada').
        ls (str): Estilo de línea ('--' por defecto).
        lw (float): Grosor de línea.
        alpha (float): Transparencia de la línea.
        color (str): Color de línea.
        scatter_kwargs (dict): kwargs extra para `plt.scatter`.
        trend_kwargs (dict): kwargs extra para `plt.plot` de la tendencia.
        devolver (bool): Si True, devuelve (t, trend).
        n_bins (int): Número de bins para agregación en PCHIP.
        show (bool): Si True, muestra la figura inmediatamente.

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: (t, trend) si `devolver=True`.

    Notas:
        - La rama 'step_suave' ajusta un escalón suavizado (logística/smoothstep) robusto.
        - Se preserva el comportamiento original; sólo se añaden comentarios y docstrings.
    """
    t = np.asarray(t)
    y = np.asarray(y)

    # ---- calcular tendencia
    if tipo == "lineal":
        coef = np.polyfit(t, y, 1)
        trend = np.polyval(coef, t)
        # print(f"y = {coef[0]:.3f} x + {coef[1]:.3f}")
    elif tipo == "polinomial":
        # coef = np.polyfit(t, y, grado)
        # trend = np.polyval(coef, t)

        t_scaled = np.linspace(-1, 1, len(t))
        coef = cheb.chebfit(t_scaled, y, grado)
        trend = cheb.chebval(t_scaled, coef)

    elif tipo == "suavizada":
        if ventana % 2 == 0:
            ventana += 1
        ventana = max(3, min(ventana, len(y) - (len(y)+1)%2))  # asegurar impar y válido
        if ventana >= 3:
            trend = savgol_filter(y, window_length=ventana, polyorder=min(2, grado))
        else:
            trend = y.copy()

    elif tipo == "pchip":
        edges = np.linspace(t.min(), t.max(), n_bins+1)
        idx = np.digitize(t, edges) - 1
        tc, yc = [], []
        for i in range(n_bins):
            m = idx == i
            if np.any(m):
                tc.append(t[m].mean()); yc.append(np.median(y[m]))
        p = PchipInterpolator(np.asarray(tc), np.asarray(yc))
        trend = p(t)

    elif tipo == "sigmoide":
        def logistic(x, L, k, x0, b):  # b + L/(1+e^{-k(x-x0)})
            return b + L/(1.0 + np.exp(-k*(x - x0)))
        L0 = np.percentile(y,95) - np.percentile(y,5)
        k0 = 2.0/(np.ptp(t)+1e-12); x0 = np.median(t); b0 = np.percentile(y,5)
        popt,_ = curve_fit(logistic, t, y, p0=[L0,k0,x0,b0], maxfev=10000)
        trend = logistic(t, *popt)

    elif tipo == "step_suave":
        # Modelo: y = y_lo + (y_hi - y_lo) * smoothstep( (t - t1)/w )
        def model(x, y_lo, y_hi, t1, w):
            return y_lo + (y_hi - y_lo) * _smoothstep((x - t1)/w)

        # Inicializaciones robustas
        y_lo0 = np.median(y[:max(5, len(y)//10)])
        y_hi0 = np.median(y[-max(5, len(y))//10:])
        t1_0  = t[np.argmax(np.gradient(y))]  # aprox del arranque
        w0    = max( (t.max()-t.min())/10, 1e-3 )

        bounds = (
            [y.min()-abs(np.ptp(y)), y.min(), t.min(), 1e-6],   # bajos
            [y.max(),             y.max()+abs(np.ptp(y)), t.max(), np.ptp(t)]  # altos
        )
        try:
            popt,_ = curve_fit(model, t, y, p0=[y_lo0, y_hi0, t1_0, w0],
                               bounds=bounds, maxfev=20000)
            trend = model(t, *popt)
        except Exception:
            # fallback por si no converge: PCHIP
            p = PchipInterpolator(t, y)
            trend = p(t)


    else:
        raise ValueError("tipo must be 'lineal', 'polinomial' or 'suavizada'")

    # ---- trazar
    if scatter_kwargs is None:
        scatter_kwargs = dict(s=20, marker='o', color='b')  # puntos azules
    if trend_kwargs is None:
        # Dashes más 'bonitos' (segmentos 6, espacios 3) y puntas redondeadas
        trend_kwargs = dict(dashes=(6, 3), dash_capstyle='round')

    plt.figure()
    plt.scatter(t, y, **scatter_kwargs)
    plt.plot(t, trend, ls=ls, lw=lw, alpha=alpha, color=color, **trend_kwargs)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Conteo")
    plt.title("Conteo de energía en banda por ventana")
    plt.grid(True)
    plt.tight_layout()
    
    if show:
        plt.show()

    if devolver:
        return t, trend

#%%
# -----------------------------
# Ejemplo mínimo (sintético) bajo __main__
# -----------------------------
# if __name__ == "__main__":
from rms_cv import five_senos, signal_1  # noqa: F401
# --- 1) Señal de ejemplo (sintética). Sustituye por tu F real ---
fs = 2_000.0
T =6.0
t = np.arange(int(T*fs)) / fs

cluster_center =  (20.0, 600.0, 200.0, 300, 400.0,800.0, 950.0)
cluster_offsets = np.array([[-25.0, -12.0, -4.0, 6.0, 14.0, 27.0, 31],
                                           [-25.0, -12.0, -4.0, 6.0, 14.0, 27.0, 31],
                                           [-30.0, -15.0, -5.0, 5.0, 15.0, 25.0, 80.0],
                                           [-30.0, -15.0, -5.0, 5.0, 15.0, 25.0, 80.0],
                                           [-30.0, -15.0, -5.0, 5.0, 15.0, 25.0, 80.0],
                                           [-30.0, -15.0, -5.0, 5.0, 15.0, 25.0, 80.0],
                                           [-30.0, -15.0, -5.0, 5.0, 15.0, 25.0, 80.0]])
cluster_amps = np.array([[0.24, 0.24, 0.20, 0.30, 0.18, 0.26, 0.10],
                                           [0.14, 0.14, 0.10, 0.15, 0.10, 0.16, 0.10],
                                           [0.14, 0.12, 0.10, 0.11, 0.09, 0.13, 0.1],
                                           [0.14, 0.12, 0.10, 0.11, 0.09, 0.13, 0.1],
                                           [0.14, 0.12, 0.10, 0.11, 0.09, 0.13, 0.1],
                                           [0.14, 0.12, 0.10, 0.11, 0.09, 0.13, 0.1],
                                           [0.14, 0.12, 0.10, 0.11, 0.09, 0.13, 0.1]])



am_freqs_chatter=np.array([9.0, 17.0, 26.0])
am_depths_chatter=np.array([0.2, 0.25, 0.07]) 

am_freqs_chatter *= 1.0
am_depths_chatter *= 4.0

f200_hz = 200.0
f200_amp = 1
low_freqs = np.array([8.0, 16.0, 20.0])
low_amps = np.array([0.5, 0.9, 0.5])
mid_freqs = np.array([55.0, 72.0, 95.0, 110.0, 135.0, 150.0])
mid_amps = np.array([0.02, 0.03, 0.02, 0.01, 0.01, 0.01])


cluster_amps[0, :] *= 0.99   #20
cluster_amps[1, :] *= 0.99  #600
cluster_amps[2, :] *= 0.99   #200
cluster_amps[3, :] *= 0.99   #300
cluster_amps[4, :] *= 5   #400
cluster_amps[5, :] *=10    #800
cluster_amps[6, :] *= 7    #950

low_amps *= 30
mid_amps *= 100



# F += (0.05*(1 + 4*np.clip((t-0.8)/0.6, 0, 1))) * np.sin(2*np.pi*520*t)  # modo ~520 Hz que crece
t, F = five_senos(fs, duracion=T, ruido_std=2, fase_aleatoria=True, seed=42)
sig, meta = make_chatter_like_signal(fs=fs, T=T, seed=123, 
                                                        signal_chatter=True,
                                                        f_chatter=500.0,
                                                        t0_chatter=3,
                                                        grow_gain=100.0,
                                                        grow_tau=0.5,

                                                        low_amps= low_amps,
                                                        mid_amps= mid_amps,
                                                        low_freqs= low_freqs,
                                                        mid_freqs= mid_freqs,

                                                        f200_hz=f200_hz,
                                                        f200_amp=f200_amp,

                                                        cluster_center=cluster_center,
                                                        cluster_offsets=cluster_offsets,
                                                        cluster_amps=cluster_amps,

                                                        am_freqs_chatter=am_freqs_chatter,
                                                        am_depths_chatter=am_depths_chatter,
                                                        base_chatter_amp=0.5 ,

                                                        white_noise_sigma=5,
                                                        narrow_noise_sigma=5

                                                        )


t, F = meta['t'], sig
# t, F = t, F

f, A = amplitude_spectrum(F, fs=fs)

# Comentario: bloque de ejemplos/figuras (idéntico al original en lógica)
plt.figure()
plt.plot(t, F)
plt.title("Señal de fuerza F(t) de ejemplo")
plt.xlabel("Tiempo (s)")
plt.ylabel("Fuerza (u.a.)")
plt.show()

plt.figure()
plt.plot(f, A)
plt.title("Espectro de amplitud de F(t)")
plt.xlabel("Frecuencia (Hz)")
plt.xlim(0, 1000)
plt.ylabel("Amplitud (u.a.)")

#%%
# --- 2) Ejecuta el pipeline (sin baseline/decisión) ---
res = detect_chatter_from_force(
    F=F,
    fs=fs,
    bandlimit_pre=(-100, 2000),
    emd_method="ceemdan",          # "emd", "eemd" o "ceemdan" 
    ceemdan_noise_strength=0.2,
    ceemdan_ensemble_size=50,
    band_chatter=(450, 600),
    band_selection_margin=0.15,
    imf_selection="index",
    imf_index=0,
    # Hilbert
    phase_diff_mode = "first_diff",
    sg_window = 11,
    sg_polyorder= 2,
    f_inst_smooth_median = None,
    # HHS
    hhs_enable = True,
    hhs_fmax = 2000.0,
    hhs_fbin_hz = 5.0,
    count_win_samples = 100,
    energy_mode = "A2",
    thr_mode="none",
    thr_k_mad=0.0, # usar MAD sin multiplicador
    thr_percentile=50, # usar percentil 95
)


#%%
# --- 3) Grafica TODOS los IMFs en UNA sola figura ---
#    (puedes limitar con max_to_plot y distribuir en columnas con ncols)
fig, axes = plot_imfs(
    imfs=res.imfs,
    plot_spectrum=True,
    fs=fs,
    max_to_plot=None,   # por ejemplo 8
    ncols=1,            # 2 o 3 si quieres cuadrícula
    show=True
)

#%%
# --- 4) Grafica el IMF seleccionado y, opcionalmente, A(t) y f_inst(t) ---
plot_imf_seleccionado(
    selected_imf=res.selected_imf,
    fs=fs,
    plot_spectrum=True,
    A=res.A,
    f_inst=res.f_inst,
    show=True
)

print("IMFs:", res.imfs.shape if res.imfs is not None else None)
print("IMF seleccionado:", res.k_selected)
print("Counts shape:", res.counts.shape)
print("Umbral usado (energía):", res.meta.get("threshold"))

#%%
HHS =np.where(res.HHS > 0, 1, 0)
cmap = ListedColormap(['blue', 'yellow'])

plt.figure() 
plt.pcolormesh(t, res.fgrid, HHS, shading='nearest', cmap=cmap, vmin=None, vmax=None)
plt.title("Hilbert-Huang Spectrum")
plt.xlabel("Tiempo (s)")
plt.ylabel("Frecuencia (Hz)")
plt.ylim(0, 1000)
plt.show()



#%% ---------- Counts plot ----------
t_counts = res.t_counts_samples / fs
counts = res.counts

# 1) Recta punteada
plot_tendencia(t_counts, counts, tipo="pchip", ls='--', lw=2, color='k')

# 2) Suavizada punteada (queda muy similar a la figura)
plot_tendencia(t_counts, counts, tipo="polinomial", grado=4, ls='--', lw=2, color='k')

# 3) Polinomial punteada (p. ej. grado 2)
plot_tendencia(t_counts, counts, tipo="step_suave", grado=3, ls='--', lw=2, color='k')
