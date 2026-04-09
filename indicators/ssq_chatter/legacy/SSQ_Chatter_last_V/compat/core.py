from __future__ import annotations
# Comentario: compatibilidad para sqq_chatter usando la nueva tubería
from typing import Tuple, Dict, Any
import numpy as np
from ..lib.pipeline_chatter import ChatterPipeline, PipelineConfig
from ..lib.tf_transformers import SSQ_STFT, STFT
from ..lib.detection_strategies import ThreeSigmaWithLilliefors

def sqq_chatter(
    signal: np.ndarray,
    fs: float,
    win_length_ms: float,
    hop_ms: float,
    n_fft: int,
    sigma: float = 4,
    Ai_length: int = 4,
    frac_stable: float = 0.25,
    SSQ: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    # Comentario: construir config y estrategias
    cfg = PipelineConfig(fs=fs, win_length_ms=win_length_ms, hop_ms=hop_ms, n_fft=n_fft, Ai_length=Ai_length)
    if SSQ:
        transformer = SSQ_STFT(
            win_length=int(round(win_length_ms * 1e-3 * fs)),
            hop_length=int(round(hop_ms * 1e-3 * fs)),
            n_fft=n_fft,
            sigma=float(sigma),
        )
    else:
        transformer = STFT(
            win_length=int(round(win_length_ms * 1e-3 * fs)),
            hop_length=int(round(hop_ms * 1e-3 * fs)),
            n_fft=n_fft,
        )
    detector = ThreeSigmaWithLilliefors(frac_stable=frac_stable, alpha=0.05, z=3.0, fallback_mad=True)

    pipe = ChatterPipeline(transformer=transformer, detector=detector, config=cfg)
    Tsx, Sx, fs_out, t, A_i, t_i, D, d1, res = pipe.run(signal, return_TF=True)
    # Comentario: mantener exactamente el orden y tipos devueltos
    return Tsx, Sx, fs_out, t, A_i, t_i, D, d1, res
