
from __future__ import annotations
from typing import Tuple
import numpy as np
from ..utils.opr import segment_opr
from .entropy import entropy_from_segments, EntropyEstimator
from ..models.maxent import fit_maxent_gaussians, MaxEntModels

def offline_train_maxent_sprt(
    opr_free: np.ndarray,
    opr_chat: np.ndarray,
    opr_t_free: np.ndarray,
    opr_t_chat: np.ndarray,
    N_seg: int,
    estimator: EntropyEstimator | None = None,
) -> Tuple[MaxEntModels, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    FASE OFFLINE (prior knowledge) a partir de señales OPR etiquetadas:

    - opr_free: OPR en condición chatter-free
    - opr_chat: OPR en condición early chatter
    - N_seg: número de revoluciones (muestras OPR) por segmento

    Devuelve:
    - models: MaxEntModels con p0(H) y p1(H)
    - H_free: entropías por segmento en estado libre
    - H_chat: entropías por segmento en estado chatter
    - t_mid_free: tiempos medios por segmento en estado libre
    - t_mid_chat: tiempos medios por segmento en estado chatter
    """

    # 1) Segmentación OPR
    segments_free, segments_t_free = segment_opr(opr_free, opr_t_free, N_seg=N_seg)
    segments_chat, segments_t_chat = segment_opr(opr_chat, opr_t_chat, N_seg=N_seg)

    if len(segments_free) == 0 or len(segments_chat) == 0:
        raise ValueError("No hay segmentos suficientes para entrenamiento. Revisa N_seg y longitud de OPR.")

    # 2) Cálculo del indicador (entropía) por segmento
    H_free = entropy_from_segments(segments_free, estimator=estimator)
    H_chat = entropy_from_segments(segments_chat, estimator=estimator)

    t_mid_free = np.array([np.mean(seg_t) for seg_t in segments_t_free])
    t_mid_chat = np.array([np.mean(seg_t) for seg_t in segments_t_chat])

    # 3) Ajuste de pdfs p0(H) y p1(H)
    models = fit_maxent_gaussians(H_free, H_chat)

    return models, H_free, H_chat, t_mid_free, t_mid_chat
