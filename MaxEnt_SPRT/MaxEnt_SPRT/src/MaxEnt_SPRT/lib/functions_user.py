# Funciones originales (sin cambios de nombre)
from .original_imports import *  # noqa: F401,F403

def validate_alpha_beta(func: Callable[P, R]) -> Callable[P, R]:
    """
    Valida que alpha y beta (si se pasan como kwargs) estén en (0, 1).

    Este decorador añade validación sin modificar la lógica interna del
    test estadístico (ejemplo de preocupación transversal).
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        alpha = kwargs.get("alpha", None)
        beta = kwargs.get("beta", None)

        if alpha is not None and beta is not None:
            alpha_f = float(alpha)
            beta_f = float(beta)
            if not (0.0 < alpha_f < 1.0 and 0.0 < beta_f < 1.0):
                raise ValueError("alpha y beta deben estar en el intervalo abierto (0, 1).")

        return func(*args, **kwargs)

    return wrapper

def sample_opr(y: np.ndarray, t: np.ndarray, fs: float, fr: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrae datos once-per-revolution (1 muestra por revolución).

    ratio = fs/fr debe ser entero. Se toma el primer índice de cada revolución.
    """
    ratio = fs / fr
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError("fs/fr debe ser entero para muestreo OPR exacto.")
    step = int(round(ratio))
    return y[::step], t[::step]

def segment_opr(opr: np.ndarray, opr_t: np.ndarray, N_seg: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Divide la señal OPR en segmentos consecutivos de longitud N_seg.
    Se descarta el sobrante final si no completa un segmento.
    """
    n_total = len(opr)
    n_segments = n_total // N_seg
    segments: List[np.ndarray] = []
    segments_t: List[np.ndarray] = []
    for k in range(n_segments):
        start = k * N_seg
        end = start + N_seg
        segments.append(opr[start:end])
        segments_t.append(opr_t[start:end])
    return segments, segments_t

def fit_maxent_gaussians(
    samples_H0: Iterable[float],
    samples_H1: Iterable[float],
    min_sigma: float = 1e-12,
) -> MaxEntModels:
    """
    Ajusta dos gaussianas a:
    - samples_H0: valores MaxEnt en estado chatter-free
    - samples_H1: valores MaxEnt en estado chatter

    Devuelve modelos p0(H) y p1(H) sobre el indicador H.
    """
    g0 = GaussianPDF.from_samples(samples_H0, eps=min_sigma)
    g1 = GaussianPDF.from_samples(samples_H1, eps=min_sigma)
    return MaxEntModels(p0=g0, p1=g1)

def entropy_from_segments(
    segments: Sequence[np.ndarray],
    estimator: EntropyEstimator | None = None,
) -> np.ndarray:
    """
    Calcula el indicador MaxEnt para una colección de segmentos OPR.
    """
    est = estimator or GaussianMaxEntEstimator()
    return est.entropy_from_segments(segments)

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

def sprt_detect_sequence(
    H_seq: Iterable[float],
    models: MaxEntModels,
    alpha: float = 0.01,
    beta: float = 0.01,
    reset_on_H0: bool = True,
) -> Tuple[str, int, np.ndarray, float, float]:
    """
    Wrapper funcional para ejecutar SPRT usando gaussianas sobre el indicador.

    Se mantiene por comodidad, pero internamente usa la clase
    SequentialProbabilityRatioTest y GaussianIndicatorLLR.
    """
    config = SPRTConfig(alpha=alpha, beta=beta, reset_on_H0=reset_on_H0)
    llr_model = GaussianIndicatorLLR(models=models)
    sprt = SequentialProbabilityRatioTest(llr_model=llr_model, config=config)
    result = sprt.run(H_seq=H_seq)

    return (
        result.final_state,
        result.decision_index,
        result.S_history,
        result.a,
        result.b,
    )

def online_maxent_sprt_from_signal(
    y_online: np.ndarray,
    t_online: np.ndarray,
    rpm: float,
    ratio_sampling: float,
    N_seg: int,
    models: MaxEntModels,
    alpha: float = 0.01,
    beta: float = 0.01,
    reset_on_H0: bool = True,
    estimator: EntropyEstimator | None = None,
) -> Tuple[str, int, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    FASE ONLINE completa para una señal nueva:

    1) OPR: y_online -> opr_online
    2) Segmentación: opr_online -> segmentos
    3) Entropía local: segmentos -> H_seq
    4) SPRT: H_seq + models (p0(H), p1(H)) -> decisión chatter/free
    """
    fr = rpm / 60.0      # Hz, frecuencia de rotación
    fs = ratio_sampling * fr  # Hz, frecuencia de muestreo

    # 1) OPR
    opr_online, opr_t_online = sample_opr(y_online, t_online, fs=fs, fr=fr)

    # 2) Segmentación
    segments_online, segments_t_online = segment_opr(opr_online, opr_t_online, N_seg=N_seg)
    if len(segments_online) == 0:
        raise ValueError("No hay segmentos suficientes en la señal online.")

    # 3) Entropía por segmento
    H_seq = entropy_from_segments(segments_online, estimator=estimator)
    t_mid_segments = np.array([np.mean(seg_t) for seg_t in segments_t_online])

    # 4) SPRT (usando wrapper funcional)
    final_state, decision_index, S_hist, a, b = sprt_detect_sequence(
        H_seq=H_seq,
        models=models,
        alpha=alpha,
        beta=beta,
        reset_on_H0=reset_on_H0,
    )

    return final_state, decision_index, H_seq, S_hist, t_mid_segments, a, b