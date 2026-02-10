from __future__ import annotations
from typing import Any, Callable, Dict, List, Sequence, Optional, Tuple

from collections import defaultdict
from ..utils.types import SignalData, IndicatorResult
from ..lib.detector import MaxEntSPRTConfig, MaxEntSPRTDetector
from ..lib.entropy import GaussianMaxEntEstimator, EmpiricalHistogramEntropyEstimator, entropy_from_segments
from ..utils.opr import sample_opr

import numpy as np

IndicatorFunc = Callable[..., IndicatorResult]

def run_maxent_sprt(signal: SignalData, INDICATOR_CONFIG: dict ) -> IndicatorResult:
    """
    Ejecuta el indicador RMS-CV con la configuración dada.

    Parameters
    ----------
    signal : SignalData
        Señal de entrada
    INDICATOR_CONFIG : dict
        Configuración específica para el indicador RMS-CV, incluyendo parámetros como n_max, samples_per_window, etc.

    Returns
    -------
    IndicatorResult
        Resultado del indicador
    """

    results: IndicatorResult = None

    func: IndicatorFunc = INDICATOR_CONFIG["func"]
    if func == "Default":
        func = _maxent_sprt_pipeline

    params: Dict[str, Any] = INDICATOR_CONFIG.get("params", {})

    results = func(signal, **params)

    return results

def _cut_signal( t,x , time_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cuts the signal to the specified time range.
    """
    start_time, end_time = time_range
    mask = (t >= start_time) & (t <= end_time)
    return t[mask], x[mask]

def _maxent_sprt_pipeline(
    signal: SignalData,
    rpm: float,
    ratio_sampling: float,
    N_seg: int,
    t_stable_total: float,
    alpha: float,
    beta: float,
    reset_on_H0: bool,
    cut_start_time: Optional[float] = None,
    cut_end_time: Optional[float] = None,

    ) -> IndicatorResult:
    t = signal.t_analysis
    x = signal.signal_analysis
    fs = signal.fs

    fr: float = rpm / 60.0       # Hz, frecuencia de rotación
    t_total = t[-1]-t[0]

    t_stable_total = t_stable_total  # seconds to consider stable
    t_chatter_total = t[-1] - t_stable_total

    t_stable, signal_analysis_stable = _cut_signal( t, x , (cut_start_time, t_stable_total) )
    t_chatter, signal_analysis_chatter = _cut_signal( t, x , (t_stable_total, cut_end_time) )

    print("Signal loaded:")
    print(f" - Samples: {x.size}")
    print(f" - Duration: {t[-1]-t[0]:.2f} s")
    print(f" - Sampling freq.: {fs:.1f} Hz")
    print(f" - Rotation freq.: {fr:.1f} Hz")
    print(f" - Segments of {N_seg} revolutions: {N_seg/fr:.2f} s each")
    print(f" - Total segments available: {int(t_total*fr/N_seg)}")

    print("Generated chatter-free and chatter-included signals.")
    print(f"Size of signal free: {signal_analysis_stable.size} samples.")
    print(f"Size of signal chatter: {signal_analysis_chatter.size} samples.")


    # =========== Fase Ofline : OPR Training ==========
    opr_free, t_opr_free = sample_opr(signal_analysis_stable, t_stable, fs=fs, fr=fr)
    opr_chat, t_opr_chat = sample_opr(signal_analysis_chatter, t_chatter, fs=fs, fr=fr)
    print(f"Sampled OPR: {opr_free.size} samples free, {opr_chat.size} samples chatter.")

    # ============ Offline Phase:END-TO-END GAUSSIAN ===========
    detector_cfg = MaxEntSPRTConfig(alpha=alpha, beta=beta, reset_on_H0=reset_on_H0)
    gaussian_estimator = GaussianMaxEntEstimator()
    detector = MaxEntSPRTDetector(config=detector_cfg, estimator=gaussian_estimator)

    # Entrenamiento offline a partir de OPR
    detector.fit_offline_from_opr(
        opr_free=opr_free,
        opr_t_free=t_opr_free,
        opr_chat=opr_chat,
        opr_t_chat=t_opr_chat,
        N_seg=N_seg,
    )

    models_trained = detector._check_models()
    print("OFFLINE MODEL (Gaussian MaxEnt):")
    print(f"  FREE:  mu0={models_trained.p0.mu:.5f}, sigma0={models_trained.p0.sigma:.5f}")
    print(f"  CHAT:  mu1={models_trained.p1.mu:.5f}, sigma1={models_trained.p1.sigma:.5f}")

    sprt_result, H_seq_online, t_mid_segments = detector.detect_online_from_signal(
        y_online=x,
        t_online=t,
        rpm=rpm,
        ratio_sampling=ratio_sampling,
        N_seg=N_seg,
    )

    #%%
    # ============ Online Phase: Results visualization ===========

    print(f"ONLINE FINAL STATE: {sprt_result.final_state}, decision at segment {sprt_result.decision_index}")

    # =========== Early chatter Results - Points Chatter ==========
    mask = np.where(sprt_result.S_history >= sprt_result.b)[0]
    chatter_points_time = t_mid_segments[mask] if mask.size > 0 else np.array([])
    chatter_points_values = sprt_result.S_history[mask] if mask.size > 0 else np.array([])

    result = IndicatorResult(
        name="MaxEnt_SPRT",  # nombre del indicador (pon el que quieras)
        t=t_mid_segments,
        I_t=sprt_result.S_history,
        t_d=chatter_points_time,
        meta={
            "Samples": x.size,
            "Duration": t_total,
            "fs": fs,
            "Rotational_Frequency_Hz": fr,
            "N_seg": N_seg,
            "alpha": alpha,
            "beta": beta,
            "rpm": rpm,
            "ratio_sampling": ratio_sampling,
            "Total_segments": int(t_total*fr/N_seg),
            "Size_signal_free": signal_analysis_stable.size,
            "Size_signal_chatter": signal_analysis_chatter.size,
            "Sampled OPR free": opr_free.size,
            "Sampled OPR chatter": opr_chat.size,
            "P0_mu": models_trained.p0.mu,
            "P0_sigma": models_trained.p0.sigma,
            "P1_mu": models_trained.p1.mu,
            "P1_sigma": models_trained.p1.sigma,
            "detector": detector,
            "gaussian_estimator": gaussian_estimator,
            "sprt_result": sprt_result,
            "models_trained": models_trained,
            "H_seq_online": H_seq_online,
            "chatter_points_values": chatter_points_values,
            "t_stable": t_stable,
            "signal_analysis_stable": signal_analysis_stable,
            "t_chatter": t_chatter,
            "signal_analysis_chatter": signal_analysis_chatter,
            "t_opr_free": t_opr_free,
            "opr_free": opr_free,
            "t_opr_chat": t_opr_chat,
            "opr_chat": opr_chat,

        },
    )

    return result
