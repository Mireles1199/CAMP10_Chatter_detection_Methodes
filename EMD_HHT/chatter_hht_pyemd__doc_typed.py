# mypy: ignore-errors
"""chatter_hht_pyemd.py (typed)
-------------------------------
Pipeline robusto para identificación temprana de chatter a partir de fuerza F(t) usando:
  - EMD/EEMD/CEEMDAN de **PyEMD**
  - Transformada de Hilbert (scipy.signal.hilbert)
  - Métrica de conteo de energía en banda en el IMF modal (ej. 450–600 Hz)

IMPORTANTE: Este módulo **no implementa** la etapa de decisión/baseline; entrega artefactos intermedios
(energía en banda, HHS, etc.) para que otra capa decida.

Dependencias:
  pip install PyEMD scipy numpy matplotlib
"""
#%%
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np

# Comentario: importaciones con manejo de errores para dependencias científicas
try:
    from scipy.signal import hilbert, savgol_filter  # type: ignore
except Exception as e:  # pragma: no cover - import guard
    raise ImportError("Se requiere scipy: pip install scipy") from e

try:
    # PyEMD: https://github.com/laszukdawid/PyEMD
    from PyEMD import CEEMDAN, EEMD, EMD  # type: ignore
except Exception as e:  # pragma: no cover - import guard
    raise ImportError("Se requiere PyEMD: pip install PyEMD") from e

# Comentario: utilidades externas usadas en la demo (no esenciales al pipeline)
from matplotlib.colors import ListedColormap  # type: ignore
from matplotlib.ticker import AutoMinorLocator, MultipleLocator  # noqa: F401  # type: ignore
from EMD_HHT.signal_chatter import amplitude_spectrum, make_chatter_like_signal  # type: ignore

from scipy.interpolate import PchipInterpolator  # type: ignore
from scipy.optimize import curve_fit  # type: ignore
from scipy.signal import savgol_filter as _savgol_filter  # redundante pero mantiene import original  # type: ignore

import numpy.polynomial.chebyshev as cheb  # type: ignore

# =============================
# Tipos auxiliares
# =============================
ArrayF = np.ndarray  # float arrays (no se fuerza dtype para no alterar lógica)
ArrayB = np.ndarray  # boolean arrays


# -----------------------------
# Utilidades de preprocesado
# -----------------------------
def _maybe_detrend(x: ArrayF) -> ArrayF:
    """Elimina tendencia lineal (detrend) de una señal.

    Args:
        x: Señal 1D original (tipo real).

    Returns:
        Señal 1D sin tendencia lineal (misma longitud que `x`).
    """
    n: int = len(x)
    t: ArrayF = np.arange(n, dtype=float)
    t = (t - t.mean()) / (t.std() + 1e-12)
    A: ArrayF = np.vstack([t, np.ones_like(t)]).T
    m, b = np.linalg.lstsq(A, x, rcond=None)[0]
    return x - (m * t + b)


def _maybe_clip(x: ArrayF, clip: Optional[Tuple[float, float]]) -> ArrayF:
    """Aplica recorte (clip) duro a la señal si se provee un rango."""
    if clip is None:
        return x
    vmin, vmax = clip
    return np.clip(x, vmin, vmax)


def _maybe_bandlimit(x: ArrayF, fs: float, band: Optional[Tuple[float, float]]) -> ArrayF:
    """Filtra band-pass (Butterworth) si se especifica banda válida."""
    if band is None:
        return x
    lo, hi = band
    if hi >= fs * 0.5:
        hi = fs * 0.5 * 0.99
    if lo <= 0:
        lo = 1e-6
    try:
        from scipy.signal import butter, filtfilt  # type: ignore
    except Exception as e:  # pragma: no cover - import guard
        raise ImportError("Se requiere scipy.signal para bandlimit_pre") from e
    b, a = butter(N=4, Wn=[lo/(fs*0.5), hi/(fs*0.5)], btype='band')
    return filtfilt(b, a, x).astype(x.dtype, copy=False)


# -----------------------------
# Selección de IMF modal
# -----------------------------
def select_imf_near_band(imfs: ArrayF, fs: float, band: Tuple[float, float]) -> int:
    """Selecciona el índice del IMF con mayor energía en una banda de frecuencia."""
    lo, hi = band
    if imfs.ndim != 2:
        raise ValueError("imfs debe tener forma (K, N)")
    K, N = imfs.shape
    n: int = int(1 << (N - 1).bit_length())
    freqs: ArrayF = np.fft.rfftfreq(n, d=1.0/fs)
    mask: ArrayB = (freqs >= lo) & (freqs <= hi)
    best_k: int = 0
    best_e: float = -np.inf
    win: ArrayF = np.hanning(N)
    for k in range(K):
        x = imfs[k]
        X: ArrayF = np.fft.rfft(x * win, n=n)
        E: float = float(np.sum(np.abs(X[mask])**2))
        if E > best_e:
            best_e, best_k = E, k
    return best_k


# -----------------------------
# Hilbert y frecuencia instantánea
# -----------------------------
def hilbert_features(
    imf: ArrayF,
    fs: float,
    phase_diff_mode: Literal["first_diff", "savitzky_golay"] = "first_diff",
    sg_window: int = 11,
    sg_polyorder: int = 2,
    f_inst_smooth_median: Optional[int] = None,
) -> Tuple[ArrayF, ArrayF]:
    """Obtiene amplitud y frecuencia instantánea del IMF mediante Hilbert."""
    z: ArrayF = hilbert(imf)
    A: ArrayF = np.abs(z)
    phase: ArrayF = np.unwrap(np.angle(z))

    if phase_diff_mode == "savitzky_golay":
        if sg_window % 2 == 0:
            sg_window += 1
        if sg_window < sg_polyorder + 2:
            sg_window = sg_polyorder + 3 if (sg_polyorder + 3) % 2 == 1 else sg_polyorder + 4
        phase_s: ArrayF = savgol_filter(phase, sg_window, sg_polyorder, mode='interp')
        dphi: ArrayF = np.gradient(phase_s)
    else:
        dphi = np.diff(phase, prepend=phase[0])

    f_inst: ArrayF = (dphi * fs) / (2.0 * np.pi)

    if f_inst_smooth_median is not None and f_inst_smooth_median > 1:
        from scipy.signal import medfilt  # type: ignore
        ksize: int = int(f_inst_smooth_median) | 1
        f_inst = medfilt(f_inst, kernel_size=ksize)

    return A, f_inst


# -----------------------------
# HHS (opcional)
# -----------------------------
def build_hhs(
    A: ArrayF,
    f_inst: ArrayF,
    fs: float,
    fmax: float = 2000.0,
    fbin_hz: float = 5.0,
    energy_mode: Literal["A2", "A"] = "A2",
) -> Tuple[ArrayF, ArrayF]:
    """Construye el Hilbert-Huang Spectrum (HHS) discreto en un grid de frecuencias."""
    if fbin_hz <= 0:
        raise ValueError("fbin_hz debe ser > 0")
    Fbins: int = int(np.ceil(fmax / fbin_hz))
    fgrid: ArrayF = np.arange(Fbins) * fbin_hz
    H: ArrayF = np.zeros((Fbins, len(A)), dtype=float)
    E: ArrayF = A**2 if energy_mode == "A2" else A
    idx: np.ndarray = np.floor(np.clip(f_inst, 0, fmax - 1e-9) / fbin_hz).astype(int)
    H[idx, np.arange(len(A))] += E  # type: ignore[index]
    return H, fgrid


# -----------------------------
# Conteo en banda por ventana
# -----------------------------
def band_counts_over_time(
    A: ArrayF,
    f_inst: ArrayF,
    band: Tuple[float, float] = (450.0, 600.0),
    energy_mode: Literal["A2", "A"] = "A2",
    thr_mode: Optional[Literal["mad", "percentile", "none"]] = "mad",
    thr_k_mad: float = 3.0,
    thr_percentile: float = 95.0,
    count_win_samples: int = 100,
    count_step_samples: Optional[int] = None,
) -> Tuple[ArrayF, ArrayF, Dict[str, Any]]:
    """Cuenta, por ventana, muestras que caen en banda y (opcional) superan un umbral de energía."""
    if count_step_samples is None:
        count_step_samples = count_win_samples

    N: int = len(A)
    lo, hi = band
    E: ArrayF = A**2 if energy_mode == "A2" else A

    thr: Optional[float]
    if thr_mode in ("mad", "percentile"):
        if thr_mode == "mad":
            med: float = float(np.median(E))
            mad: float = float(np.median(np.abs(E - med))) + 1e-12
            thr = med + thr_k_mad * mad
        else:
            thr = float(np.percentile(E, thr_percentile))
    elif thr_mode in ("none", None):
        thr = None
    else:
        raise ValueError("thr_mode debe ser 'mad', 'percentile' o 'none'")

    counts: list[int] = []
    centers: list[int] = []
    for start in range(0, N - count_win_samples + 1, count_step_samples):
        sl = slice(start, start + count_win_samples)
        mask_band: ArrayB = (f_inst[sl] >= lo) & (f_inst[sl] <= hi)

        if thr is None:
            hits: int = int(np.count_nonzero(mask_band))
        else:
            mask_energy: ArrayB = (E[sl] > thr)
            hits = int(np.count_nonzero(mask_band & mask_energy))

        counts.append(hits)
        centers.append(start + count_win_samples // 2)

    debug: Dict[str, Any] = {"thr": thr, "thr_mode": thr_mode, "band": band, "energy_mode": energy_mode}
    return np.asarray(counts), np.asarray(centers), debug


# -----------------------------
# Salida estructurada
# -----------------------------
@dataclass
class ChatterResult:
    """Contenedor de resultados intermedios del pipeline HHT/EMD."""
    imfs: Optional[ArrayF]
    k_selected: int
    selected_imf: ArrayF
    A: ArrayF
    f_inst: ArrayF
    counts: ArrayF
    t_counts_samples: ArrayF
    HHS: Optional[ArrayF]
    fgrid: Optional[ArrayF]
    meta: Dict[str, Any]


# -----------------------------
# Función principal (PyEMD)
# -----------------------------
def detect_chatter_from_force(
    F: ArrayF,
    fs: float,
    # Pre-procesado
    detrend: bool = False,
    clip: Optional[Tuple[float, float]] = None,
    bandlimit_pre: Optional[Tuple[float, float]] = None,
    # EMD/EEMD/CEEMDAN (PyEMD)
    emd_method: Literal["ceemdan", "emd", "eemd"] = "ceemdan",
    emd_max_imfs: Optional[int] = None,
    emd_max_siftings: Optional[int] = 50,
    ceemdan_noise_strength: float = 0.2,
    ceemdan_ensemble_size: int = 100,
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
    """Pipeline principal con **PyEMD** (sin decisión/baseline)."""
    if F.ndim != 1:
        raise ValueError("F debe ser un vector 1D")
    x: ArrayF = F.astype(float, copy=True)

    # ---- Preprocesado
    if detrend:
        x = _maybe_detrend(x)
    if clip is not None:
        x = _maybe_clip(x, clip)
    if bandlimit_pre is not None:
        x = _maybe_bandlimit(x, fs, bandlimit_pre)

    # ---- EMD/EEMD/CEEMDAN con PyEMD
    rng = np.random.default_rng(emd_rng_seed) if emd_rng_seed is not None else None  # noqa: F841
    seed: Optional[int] = int(emd_rng_seed) if emd_rng_seed is not None else None

    if emd_method == "ceemdan":
        decomp = CEEMDAN(random_seed=seed)
        decomp.noise_seed(seed)
        decomp.trials = int(ceemdan_ensemble_size)
        decomp.noise_strength = float(ceemdan_noise_strength)
        if emd_max_siftings is not None:
            decomp.max_iter = int(emd_max_siftings)
        imfs: ArrayF = decomp.ceemdan(x, max_imf=emd_max_imfs)
    elif emd_method == "eemd":
        decomp = EEMD(trials=ceemdan_ensemble_size, noise_seed=seed)
        if emd_max_siftings is not None:
            base_emd = EMD()
            base_emd.FIXE = int(emd_max_siftings)
            decomp.EMD = base_emd
        imfs = decomp.eemd(x, max_imf=emd_max_imfs)
    else:
        decomp = EMD()
        if emd_max_siftings is not None:
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
        bw = (hi - lo) * band_selection_margin
        sel_band: Tuple[float, float] = (max(0.0, lo - bw), hi + bw)
    else:
        sel_band = band_chatter

    if imf_selection == "index":
        if imf_index is None:
            raise ValueError("imf_index debe especificarse cuando imf_selection='index'")
        if not (0 <= imf_index < K):
            raise ValueError(f"imf_index fuera de rango (0..{K-1})")
        k_selected: int = imf_index
    else:
        k_selected = select_imf_near_band(imfs, fs, sel_band)

    selected_imf: ArrayF = imfs[k_selected]

    # ---- Hilbert
    A, f_inst = hilbert_features(
        selected_imf, fs,
        phase_diff_mode=phase_diff_mode,
        sg_window=sg_window,
        sg_polyorder=sg_polyorder,
        f_inst_smooth_median=f_inst_smooth_median,
    )

    # ---- HHS opcional
    HHS: Optional[ArrayF] = None
    fgrid: Optional[ArrayF] = None
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
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - entornos sin GUI
    plt = None  # type: ignore


def _time_axis(n: int, fs: Optional[float]) -> Tuple[ArrayF, str]:
    """Devuelve eje x y etiqueta apropiada según disponibilidad de `fs`."""
    if fs is None or fs <= 0:
        return np.arange(n), "muestras"
    else:
        return np.arange(n) / fs, "s"


def plot_imfs_separados(imfs: ArrayF, fs: Optional[float] = None, max_to_plot: Optional[int] = None, show: bool = True) -> list[Any]:
    """Crea una figura independiente por cada IMF (sin subplots)."""
    if plt is None:
        raise ImportError("Falta matplotlib. Instala con: pip install matplotlib")
    if imfs.ndim != 2:
        raise ValueError("imfs debe tener forma (K, N)")
    K, N = imfs.shape
    if max_to_plot is None or max_to_plot > K:
        max_to_plot = K
    t, xlabel = _time_axis(N, fs)
    figs: list[Any] = []
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


def plot_imf_seleccionado(
    selected_imf: ArrayF,
    fs: Optional[float] = None,
    plot_spectrum: bool = False,
    A: Optional[ArrayF] = None,
    f_inst: Optional[ArrayF] = None,
    show: bool = True
) -> list[Any]:
    """Grafica el IMF seleccionado y opcionalmente A(t) y f_inst(t), en figuras separadas."""
    if plt is None:
        raise ImportError("Falta matplotlib. Instala con: pip install matplotlib")
    N: int = len(selected_imf)
    t, xlabel = _time_axis(N, fs)

    figs: list[Any] = []

    fig1 = plt.figure()
    ax1 = fig1.gca()
    ax1.plot(t, selected_imf)
    ax1.set_title("IMF seleccionado - dominio temporal")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Amplitud")
    figs.append(fig1)

    if A is not None and len(A) == N:
        fig2 = plt.figure()
        ax2 = fig2.gca()
        ax2.plot(t, A)
        ax2.set_title("Amplitud instantánea A(t)")
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("A")
        figs.append(fig2)

    if f_inst is not None and len(f_inst) == N:
        fig3 = plt.figure()
        ax3 = fig3.gca()
        ax3.plot(t, f_inst)
        ax3.set_title("Frecuencia instantánea f_inst(t)")
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel("Hz")
        figs.append(fig3)

    if plot_spectrum:
        fig2, axes_2 = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        axes_2 = np.array(axes_2).reshape(-1)
        ax = axes_2[0]
        f, Pxx = amplitude_spectrum(selected_imf, fs=fs, normalize_to=0.1)  # type: ignore
        ax.plot(f, Pxx)
        ax.set_title("Espectro IMF seleccionado")
        ax.set_xlabel("Frecuencia (Hz)" if fs is not None else "Frecuencia (muestras)")
        ax.set_ylabel("Amplitud")
        ax.set_xlim(0, 1000 if fs is not None else None)
        ax.xaxis.set_major_locator(MultipleLocator(100))  # type: ignore
        fig2.tight_layout()

    if show:
        plt.show()
    return figs


def plot_imfs(
    imfs: ArrayF,
    fs: Optional[float] = None,
    plot_spectrum: bool = False,
    max_to_plot: Optional[int] = None,
    ncols: int = 1,
    show: bool = True
) -> Tuple[Any, np.ndarray]:
    """Dibuja *todos* los IMFs en **una sola figura** usando subplots."""
    if plt is None:
        raise ImportError("Falta matplotlib. Instala con: pip install matplotlib")
    if imfs.ndim != 2:
        raise ValueError("imfs debe tener forma (K, N)")
    K, N = imfs.shape
    if max_to_plot is None or max_to_plot > K:
        max_to_plot = K
    if ncols < 1:
        ncols = 1
    nrows: int = int(np.ceil(max_to_plot / ncols))

    t, xlabel = _time_axis(N, fs)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(10, max(3, 2*nrows)))
    axes = np.array(axes).reshape(-1)

    for i in range(max_to_plot):
        ax = axes[i]
        ax.plot(t, imfs[i])
        ax.set_title(f"IMF {i+1}")
        ax.set_ylabel("Amplitud")
    for j in range(max_to_plot, len(axes)):
        axes[j].set_visible(False)

    axes[min(max_to_plot-1, len(axes)-1)].set_xlabel(xlabel)

    for ax in axes:
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)

    fig.tight_layout()

    if plot_spectrum:
        fig2, axes_2 = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(10, max(3, 2*nrows)))
        axes_2 = np.array(axes_2).reshape(-1)
        for i in range(max_to_plot):
            ax = axes_2[i]
            f, Pxx = amplitude_spectrum(imfs[i], fs=fs, normalize_to=0.1)  # type: ignore
            ax.plot(f, Pxx)
            ax.set_title(f"Espectro IMF {i+1}")
            ax.set_ylabel("Amplitud")
            ax.set_xlim(0, 1000 if fs is not None else None)
            ax.tick_params(axis='x', )
            ax.tick_params(axis='y', )
            ax.xaxis.set_major_locator(MultipleLocator(100))  # type: ignore

        for j in range(max_to_plot, len(axes_2)):
            axes_2[j].set_visible(False)

        axes_2[min(max_to_plot-1, len(axes_2)-1)].set_xlabel("Frecuencia (Hz)" if fs is not None else "Frecuencia (muestras)")
        fig2.tight_layout()
        for ax in axes_2:
            ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)

    if show:
        plt.show()
    return fig, axes


def _smoothstep(z: Union[ArrayF, float]) -> Union[ArrayF, float]:
    """Función paso-suave: 0 fuera, y 3z^2 - 2z^3 para z en [0,1]."""
    z = np.clip(z, 0.0, 1.0)
    return z*z*(3 - 2*z)  # type: ignore[operator]


def plot_tendencia(
    t: Union[Sequence[float], ArrayF],
    y: Union[Sequence[float], ArrayF],
    tipo: str = "lineal",
    grado: int = 3,
    ventana: int = 7,
    ls: str = '--',
    lw: float = 3,
    alpha: float = 1.0,
    color: str = 'k',
    scatter_kwargs: Optional[Mapping[str, Any]] = None,
    trend_kwargs: Optional[Mapping[str, Any]] = None,
    devolver: bool = False,
    n_bins: int = 10
) -> Optional[Tuple[ArrayF, ArrayF]]:
    """Dibuja puntos y línea de tendencia según método elegido."""
    if plt is None:
        raise ImportError("Falta matplotlib. Instala con: pip install matplotlib")

    t = np.asarray(t)
    y = np.asarray(y)

    if tipo == "lineal":
        coef = np.polyfit(t, y, 1)
        trend = np.polyval(coef, t)
    elif tipo == "polinomial":
        t_scaled = np.linspace(-1, 1, len(t))
        coef = cheb.chebfit(t_scaled, y, grado)
        trend = cheb.chebval(t_scaled, coef)
    elif tipo == "suavizada":
        if ventana % 2 == 0:
            ventana += 1
        ventana = max(3, min(ventana, len(y) - (len(y)+1)%2))
        if ventana >= 3:
            trend = savgol_filter(y, window_length=ventana, polyorder=min(2, grado))
        else:
            trend = y.copy()
    elif tipo == "pchip":
        edges = np.linspace(t.min(), t.max(), n_bins+1)
        idx = np.digitize(t, edges) - 1
        tc: list[float] = []
        yc: list[float] = []
        for i in range(n_bins):
            m = idx == i
            if np.any(m):
                tc.append(float(t[m].mean()))
                yc.append(float(np.median(y[m])))
        p = PchipInterpolator(np.asarray(tc), np.asarray(yc))
        trend = p(t)
    elif tipo == "sigmoide":
        def logistic(x: ArrayF, L: float, k: float, x0: float, b: float) -> ArrayF:
            return b + L/(1.0 + np.exp(-k*(x - x0)))
        L0: float = float(np.percentile(y,95) - np.percentile(y,5))
        k0: float = float(2.0/(t.ptp()+1e-12))
        x0: float = float(np.median(t))
        b0: float = float(np.percentile(y,5))
        popt,_ = curve_fit(logistic, t, y, p0=[L0,k0,x0,b0], maxfev=10000)
        trend = logistic(t, *popt)
    elif tipo == "step_suave":
        def model(x: ArrayF, y_lo: float, y_hi: float, t1: float, w: float) -> ArrayF:
            return y_lo + (y_hi - y_lo) * _smoothstep((x - t1)/w)  # type: ignore[operator]

        y_lo0: float = float(np.median(y[:max(5, len(y)//10)]))
        y_hi0: float = float(np.median(y[-max(5, len(y))//10:]))
        t1_0: float  = float(t[np.argmax(np.gradient(y))])
        w0: float    = float(max( (t.max()-t.min())/10, 1e-3 ))

        bounds = (
            [float(y.min()-abs(y.ptp())), float(y.min()), float(t.min()), 1e-6],
            [float(y.max()), float(y.max()+abs(y.ptp())), float(t.max()), float(t.ptp())]
        )
        try:
            popt,_ = curve_fit(model, t, y, p0=[y_lo0, y_hi0, t1_0, w0],
                               bounds=bounds, maxfev=20000)
            trend = model(t, *popt)
        except Exception:
            p = PchipInterpolator(t, y)
            trend = p(t)
    else:
        raise ValueError("tipo must be 'lineal', 'polinomial' or 'suavizada'")

    if scatter_kwargs is None:
        scatter_kwargs = dict(s=20, marker='o', color='b')
    if trend_kwargs is None:
        trend_kwargs = dict(dashes=(6, 3), dash_capstyle='round')

    plt.figure()
    plt.scatter(t, y, **scatter_kwargs)  # type: ignore[arg-type]
    plt.plot(t, trend, ls=ls, lw=lw, alpha=alpha, color=color, **trend_kwargs)  # type: ignore[arg-type]
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Conteo")
    plt.title("Conteo de energía en banda por ventana")
    plt.grid(True)
    plt.show()

    if devolver:
        return t, trend  # type: ignore[return-value]
    return None

#%%
# -----------------------------
# Ejemplo mínimo (sintético) bajo __main__ (se deja intacto)
# -----------------------------
# if __name__ == "__main__":
from rms_cv import five_senos, signal_1  # noqa: F401, F811  # type: ignore

fs: float = 2_000.0
T: float = 6.0
t: ArrayF = np.arange(int(T*fs)) / fs

cluster_center: Tuple[float, float, float, float, float, float, float] =  (20.0, 600.0, 200.0, 300, 400.0,800.0, 950.0)
cluster_offsets: ArrayF = np.array([[-25.0, -12.0, -4.0, 6.0, 14.0, 27.0, 31],
                                    [-25.0, -12.0, -4.0, 6.0, 14.0, 27.0, 31],
                                    [-30.0, -15.0, -5.0, 5.0, 15.0, 25.0, 80.0],
                                    [-30.0, -15.0, -5.0, 5.0, 15.0, 25.0, 80.0],
                                    [-30.0, -15.0, -5.0, 5.0, 15.0, 25.0, 80.0],
                                    [-30.0, -15.0, -5.0, 5.0, 15.0, 25.0, 80.0],
                                    [-30.0, -15.0, -5.0, 5.0, 15.0, 25.0, 80.0]])
cluster_amps: ArrayF = np.array([[0.24, 0.24, 0.20, 0.30, 0.18, 0.26, 0.10],
                                    [0.14, 0.14, 0.10, 0.15, 0.10, 0.16, 0.10],
                                    [0.14, 0.12, 0.10, 0.11, 0.09, 0.13, 0.1],
                                    [0.14, 0.12, 0.10, 0.11, 0.09, 0.13, 0.1],
                                    [0.14, 0.12, 0.10, 0.11, 0.09, 0.13, 0.1],
                                    [0.14, 0.12, 0.10, 0.11, 0.09, 0.13, 0.1],
                                    [0.14, 0.12, 0.10, 0.11, 0.09, 0.13, 0.1]])

am_freqs_chatter: ArrayF = np.array([9.0, 17.0, 26.0])
am_depths_chatter: ArrayF = np.array([0.2, 0.25, 0.07])

am_freqs_chatter *= 1.0
am_depths_chatter *= 4.0

f200_hz: float = 200.0
f200_amp: float = 1
low_freqs: ArrayF = np.array([8.0, 16.0, 20.0])
low_amps: ArrayF = np.array([0.5, 0.9, 0.5])
mid_freqs: ArrayF = np.array([55.0, 72.0, 95.0, 110.0, 135.0, 150.0])
mid_amps: ArrayF = np.array([0.02, 0.03, 0.02, 0.01, 0.01, 0.01])

cluster_amps[0, :] *= 0.99   #20
cluster_amps[1, :] *= 0.99  #600
cluster_amps[2, :] *= 0.99   #200
cluster_amps[3, :] *= 0.99   #300
cluster_amps[4, :] *= 5   #400
cluster_amps[5, :] *=10    #800
cluster_amps[6, :] *= 7    #950

low_amps *= 30
mid_amps *= 100

t, F = five_senos(fs, duracion=T, ruido_std=2, fase_aleatoria=True, seed=42)  # type: ignore
sig, meta = make_chatter_like_signal(  # type: ignore
    fs=fs, T=T, seed=123,
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

t, F = meta['t'], sig  # type: ignore[assignment]
# t, F = t, F
f, A = amplitude_spectrum(F, fs=fs)  # type: ignore



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
res = detect_chatter_from_force(
    F=F,
    fs=fs,
    bandlimit_pre=(-100, 2000),
    emd_method="emd",
    ceemdan_noise_strength=0.2,
    ceemdan_ensemble_size=50,
    band_chatter=(450, 600),
    band_selection_margin=0.15,
    imf_selection="index",
    imf_index=0,
    phase_diff_mode = "first_diff",
    sg_window = 11,
    sg_polyorder= 2,
    f_inst_smooth_median = None,
    hhs_enable = True,
    hhs_fmax = 2000.0,
    hhs_fbin_hz = 5.0,
    count_win_samples = 100,
    energy_mode = "A2",
    thr_mode="none",
    thr_k_mad=0.0,
    thr_percentile=50,
)

#%%
fig, axes = plot_imfs(
    imfs=res.imfs if res.imfs is not None else np.empty((0,0)),
    plot_spectrum=True,
    fs=fs,
    max_to_plot=None,
    ncols=1,
    show=True
)


#%%
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
HHS = np.where(res.HHS > 0, 1, 0) if res.HHS is not None else np.zeros((1,1))
cmap = ListedColormap(['blue', 'yellow'])  # type: ignore

plt.figure()
plt.pcolormesh(t, res.fgrid if res.fgrid is not None else np.arange(HHS.shape[0]), HHS, 
               shading='nearest', cmap=cmap, vmin=None, vmax=None)
plt.title("Hilbert-Huang Spectrum")
plt.xlabel("Tiempo (s)")
plt.ylabel("Frecuencia (Hz)")
plt.ylim(0, 1000)
plt.show()

# %%
t_counts = res.t_counts_samples / fs
counts = res.counts

plot_tendencia(t_counts, counts, tipo="pchip", ls='--', lw=2, color='k')
plot_tendencia(t_counts, counts, tipo="polinomial", grado=4, ls='--', lw=2, color='k')
plot_tendencia(t_counts, counts, tipo="step_suave", grado=3, ls='--', lw=2, color='k')
