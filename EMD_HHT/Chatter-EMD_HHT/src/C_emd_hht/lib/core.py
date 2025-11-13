from __future__ import annotations
import numpy as np
from typing import Any, Dict, Literal, Optional, Tuple, Union
from .datatypes import ChatterResult
from .emd_preproc import _maybe_bandlimit, _maybe_clip, _maybe_detrend
from .selection import select_imf_near_band
from .hht import hilbert_features, build_hhs
from .metrics import band_counts_over_time

from PyEMD import CEEMDAN, EEMD, EMD




def detect_chatter_from_force(
    F: np.ndarray ,
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
    x: np.ndarray  = F.astype(float, copy=True)

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
        if emd_max_imfs is None:
            emd_max_imfs = -1
        decomp = CEEMDAN(random_seed=seed)
        decomp.noise_seed(seed)
        decomp.trials = int(ceemdan_ensemble_size)
        decomp.noise_strength = float(ceemdan_noise_strength)
        if emd_max_siftings is not None:
            decomp.max_iter = int(emd_max_siftings)
        imfs: np.ndarray  = decomp.ceemdan(x, max_imf=emd_max_imfs)
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

    selected_imf: np.ndarray  = imfs[k_selected]

    # ---- Hilbert
    A, f_inst = hilbert_features(
        selected_imf, fs,
        phase_diff_mode=phase_diff_mode,
        sg_window=sg_window,
        sg_polyorder=sg_polyorder,
        f_inst_smooth_median=f_inst_smooth_median,
    )

    # ---- HHS opcional
    HHS: Optional[np.ndarray ] = None
    fgrid: Optional[np.ndarray ] = None
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

