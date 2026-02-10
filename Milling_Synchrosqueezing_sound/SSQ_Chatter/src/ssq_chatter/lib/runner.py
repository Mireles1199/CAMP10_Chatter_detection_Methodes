from __future__ import annotations
from typing import Any, Callable, Dict, List, Sequence, Optional

from collections import defaultdict

from ..utils.types import SignalData, IndicatorResult
from ..lib.pipeline_chatter import ChatterPipeline, PipelineConfig
from ..lib.tf_transformers import SSQ_STFT, STFT
from ..lib.detection_strategies import ThreeSigmaWithLilliefors

import numpy as np

IndicatorFunc = Callable[..., IndicatorResult]

def run_sst_svd(signal: SignalData, INDICATOR_CONFIG: dict ) -> IndicatorResult:
    """
    Ejecuta el indicador SST-SVD con la configuración dada.

    Parameters
    ----------
    signal : SignalData
        Señal de entrada
    INDICATOR_CONFIG : dict
        Configuración específica para el indicador SST-SVD, incluyendo parámetros como n_fft_power, win_length_ms, etc.

    Returns
    -------
    IndicatorResult
        Resultado del indicador
    """
    results: IndicatorResult = None

    func: IndicatorFunc = INDICATOR_CONFIG["func"]
    if func == "Default":
        func = _sst_svd_pipeline

    params: Dict[str, Any] = INDICATOR_CONFIG.get("params", {})

    results = func(signal, **params)

    return results


def _sst_svd_pipeline(
    signal: SignalData,
    n_fft_power: int,
    win_length_ms: float,
    hop_ms: float,
    Ai_length: int,
    mode: str,
    sigma: float,
    frac_stable: float,
    alpha: float,
    z: float,
    fallback_mad: bool,
) -> IndicatorResult:
    """
    Wrapper del indicador basado en SST-SVD (ejemplo).

    Parameters
    ----------
    signal : SignalData
        Señal de entrada (t, x, t_original, x_original, fs, meta).
    n_fft_power : int
        Potencia de 2 para el tamaño de FFT (n_fft = 1024 * 2**n_fft_power).
    win_length_ms : float
        Longitud de ventana en ms.
    hop_ms : float
        Hop size en ms.
    Ai_length : int
        Longitud de la serie temporal de amplitud.
    mode : str
        Modo de SST
    sigma : float
        Parámetro sigma para SST.
    frac_stable : float
        Fracción de estabilidad para la regla de detección.
    alpha : float
        Nivel de significancia para la regla de detección.
    z : float
        Parámetro z para la regla de detección.
    fallback_mad : bool
        Si es True, usa MAD como fallback en la regla de detección.

    Returns
    -------
    IndicatorResult
        Resultado estándar del indicador,
    """

    t = signal.t_analysis
    signal_analysis = signal.signal_analysis
    fs = signal.fs

    # ========= Configuration SSTFT + SSQ ============
    n_fft_power = n_fft_power
    n_fft = 1024*(2**n_fft_power)
    cfg: PipelineConfig = PipelineConfig(
        fs=fs,
        win_length_ms=win_length_ms,
        hop_ms=hop_ms,
        n_fft=n_fft,
        Ai_length=Ai_length,
        mode = mode,
    )

    # Opción A: SSQ-STFT (requiere ssqueezepy)
    hop_length = int(cfg.hop_ms * 1e-3 * cfg.fs)
    tf_strategy = SSQ_STFT(
        win_length=int(cfg.win_length_ms * 1e-3 * cfg.fs),
        hop_length=int(cfg.hop_ms * 1e-3 * cfg.fs),
        n_fft=cfg.n_fft,
        sigma=sigma,
    )

    # regla de detección (Strategy)
    detect_rule = ThreeSigmaWithLilliefors(frac_stable=frac_stable ,
                                        alpha=alpha, z=z,
                                        fallback_mad=fallback_mad,)

    # Comentario: construir tubería (DIP: inyecta estrategias)
    pipe = ChatterPipeline(transformer=tf_strategy, detector=detect_rule, config=cfg)

    # Comentario: ejecutar
    Tsx: np.ndarray
    Sx: np.ndarray
    fs_out: float
    tt: np.ndarray
    A_i: np.ndarray
    t_i: np.ndarray
    D: np.ndarray
    d1: np.ndarray
    res: Dict[str, Any]
    w: np.ndarray
    dWx: np.ndarray

#%%
    # ========= Run pipeline ============
    Tsx, Sx, fs_out, tt, A_i, t_i, D, d1, res, w, dWx = pipe.run(signal_analysis)

    # f = np.linspace(0, fs/2, Sx.shape[0])
    # t = np.arange(Sx.shape[1]) * hop_length / fs

    # Tsx = abs(Tsx)

    # plt.figure(figsize=(7,4))
    # plt.pcolormesh(t, f, Tsx, shading='auto', cmap= 'jet', vmin=None, vmax=None)
    # plt.title("|T_x(μ, ω)| (SSQ STFT)")
    # plt.xlabel("Tiempo [s]")
    # plt.ylabel("Frecuencia [Hz]")
    # plt.ylim(0, 1000)
    # plt.colorbar(label="Magnitud")
    # plt.show()

    chatter_points_mask = np.where(d1 > res['lim_sup'])[0]
    chatter_points_time = t_i[chatter_points_mask] if chatter_points_mask.size > 0 else np.array([])
    chatter_points_values = d1[chatter_points_mask] if chatter_points_mask.size > 0 else np.array([])

    result = IndicatorResult(
        name="SST_SVD",  # nombre del indicador (pon el que quieras)
        t=t_i,
        I_t=d1,
        t_d=chatter_points_time,
        meta={
            "fs_out": fs_out,
            "n_fft_power": n_fft_power,
            "win_length_ms": win_length_ms,
            "hop_ms": hop_ms,
            "Ai_length": Ai_length,
            "mode": mode,
            "sigma": sigma,
            "frac_stable": frac_stable,
            "alpha": alpha,
            "z": z,
            "fallback_mad": fallback_mad,
            "W": w,
            "tt": tt,
            "dWx": dWx,
            "Tsx": Tsx,
            "Sx": Sx,
            "A_i": A_i,
            "D": D,
            "lim_inf": res["lim_inf"],
            "lim_sup": res["lim_sup"],
            "metodo_umbral": res["metodo_umbral"],
            "normal_ok": res["normal_ok"],
            "p_value": res["p_value"],
            "mu": res["mu"],
            "sigma": res["sigma"],
            "chatter": f"{100*res['mask'].mean():.2f}%",
        },
    )

    return result
