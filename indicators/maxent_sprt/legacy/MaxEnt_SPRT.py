from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Tuple, List, Sequence, Callable, TypeVar, ParamSpec
from abc import ABC, abstractmethod
from functools import wraps

import numpy as np
import math
import matplotlib.pyplot as plt  # type: ignore

from C_emd_hht import signal_chatter_example, sinus_6_C_SNR  # type: ignore


# ==========================================================
# 0. TIPOS GENÉRICOS Y DECORADORES TRANSVERSALES
# ==========================================================

P = ParamSpec("P")
R = TypeVar("R") 


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


# ==========================================================
# 1. UTILIDADES BÁSICAS: OPR Y SEGMENTACIÓN
# ==========================================================

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


# ==========================================================
# 2. MODELO GAUSSIANO Y PDF DEL INDICADOR MaxEnt
# ==========================================================

@dataclass(frozen=True)
class GaussianPDF:
    """
    Representa una distribución normal N(mu, sigma^2) y permite:

    - Evaluar su log-pdf.
    - Calcular su entropía de Shannon cerrada.
    """
    mu: float
    sigma: float  # > 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.mu):
            raise ValueError("mu no es finito.")
        if not (np.isfinite(self.sigma) and self.sigma > 0.0):
            raise ValueError("sigma debe ser finito y > 0.")

    def logpdf(self, x: float) -> float:
        """
        log N(x ; mu, sigma^2)
        """
        z = (x - self.mu) / self.sigma
        return -0.5 * (math.log(2.0 * math.pi) + 2.0 * math.log(self.sigma) + z * z)

    def entropy_shannon(self) -> float:
        """
        Entropía de una normal N(mu, sigma^2):
        H = 0.5 * log(2*pi*e*sigma^2)
        """
        return 0.5 * math.log(2.0 * math.pi * math.e * (self.sigma ** 2))

    @staticmethod
    def from_samples(samples: Iterable[float], eps: float = 1e-12) -> "GaussianPDF":
        """
        Ajusta N(mu, sigma^2) a las muestras dadas.
        """
        x = np.asarray(list(samples), dtype=float)
        if x.size < 2:
            raise ValueError("Se requieren al menos 2 muestras para estimar sigma.")
        mu = float(np.mean(x))
        var = float(np.var(x, ddof=1))
        sigma = math.sqrt(max(var, eps))
        return GaussianPDF(mu=mu, sigma=sigma)


@dataclass
class MaxEntModels:
    """
    Modelos globales de la pdf del indicador MaxEnt (entropía) bajo:
    - H0: chatter-free  -> p0(H)
    - H1: early chatter -> p1(H)
    """
    p0: GaussianPDF
    p1: GaussianPDF


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


# ==========================================================
# 3. ENTROPÍA: ABSTRACCIÓN / HERENCIA / POLIMORFISMO
# ==========================================================

class EntropyEstimator(ABC):
    """
    Clase base abstracta para estimadores de entropía a partir de segmentos OPR.

    Esta abstracción permite enchufar distintos modelos (MaxEnt gaussiano,
    histograma empírico, kernel, etc.) sin cambiar el resto del pipeline.
    """

    @abstractmethod
    def entropy_from_segment(self, seg: np.ndarray) -> float:
        """
        Calcula la entropía para un segmento OPR.
        """

    def entropy_from_segments(self, segments: Sequence[np.ndarray]) -> np.ndarray:
        """
        Calcula la entropía para una secuencia de segmentos OPR.
        """
        return np.array(
            [self.entropy_from_segment(seg) for seg in segments],
            dtype=float,
        )


class GaussianMaxEntEstimator(EntropyEstimator):
    """
    Implementación concreta basada en MaxEnt gaussiano:
    ajusta N(mu, sigma^2) al segmento y usa su entropía analítica.
    """

    def entropy_from_segment(self, seg: np.ndarray) -> float:
        gaussian = GaussianPDF.from_samples(seg)
        return gaussian.entropy_shannon()


class EmpiricalHistogramEntropyEstimator(EntropyEstimator):
    """
    Estimador de entropía empírica usando histograma discreto:

    1) Estima la distribución empírica vía histograma normalizado.
    2) Calcula H = -sum p_i log p_i.

    Esto NO asume modelo paramétrico y permite comparar contra MaxEnt gaussiano.
    """

    def __init__(self, bins: int = 20) -> None:
        if bins <= 0:
            raise ValueError("bins debe ser un entero positivo.")
        self.bins: int = bins

    def entropy_from_segment(self, seg: np.ndarray) -> float:
        """
        Entropía de Shannon discreta a partir de histograma empírico.
        """
        x = np.asarray(seg, dtype=float)
        if x.size == 0:
            raise ValueError("segmento vacío, no se puede calcular entropía.")

        # Histograma de frecuencias (no densidad)
        hist, _ = np.histogram(x, bins=self.bins, density=False)
        total = hist.sum()
        if total == 0:
            return 0.0

        p = hist.astype(float) / float(total)
        mask = p > 0.0
        p_nz = p[mask]
        # Entropía discreta H = -sum p log p (log natural)
        return float(-np.sum(p_nz * np.log(p_nz)))



def entropy_from_segments(
    segments: Sequence[np.ndarray],
    estimator: EntropyEstimator | None = None,
) -> np.ndarray:
    """
    Calcula el indicador MaxEnt para una colección de segmentos OPR.
    """
    est = estimator or GaussianMaxEntEstimator()
    return est.entropy_from_segments(segments)


# ==========================================================
# 4. FASE OFFLINE: ENTRENAR MODELOS p0(H) Y p1(H)
# ==========================================================

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


# ==========================================================
# 5. LLR: MODELOS DE RAZÓN DE VEROSIMILITUD LOGARÍTMICA
# ==========================================================

class LLRModel(ABC):
    """
    Estrategia abstracta para calcular el LLR de una observación H.

    Esto desacopla el motor SPRT de los detalles del modelo (gaussiano,
    kernel, histogramas, etc.) → DIP (Dependency Inversion Principle).
    """

    @abstractmethod
    def llr(self, h_obs: float) -> float:
        """
        Devuelve log( p1(h_obs) / p0(h_obs) ).
        """


@dataclass(frozen=True)
class GaussianIndicatorLLR(LLRModel):
    """
    Implementación concreta de LLR usando dos gaussianas sobre el indicador H.
    """
    models: MaxEntModels

    def llr(self, h_obs: float) -> float:
        return self.models.p1.logpdf(h_obs) - self.models.p0.logpdf(h_obs)


# ==========================================================
# 6. MOTOR SPRT ORIENTADO A OBJETOS
# ==========================================================

@dataclass
class SPRTConfig:
    """
    Configuración del test SPRT:
    - alpha: prob. de falso positivo máxima.
    - beta: prob. de falso negativo máxima.
    - reset_on_H0: si True, resetea S_n a 0 cuando cae por debajo de a.
    """
    alpha: float = 0.01
    beta: float = 0.01
    reset_on_H0: bool = True

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0 and 0.0 < self.beta < 1.0):
            raise ValueError("alpha y beta deben estar en (0, 1).")

    @property
    def a(self) -> float:
        """
        Umbral inferior: región de aceptación de H0 (chatter-free).
        """
        return math.log(self.beta / (1.0 - self.alpha))

    @property
    def b(self) -> float:
        """
        Umbral superior: región de aceptación de H1 (chatter).
        """
        return math.log((1.0 - self.beta) / self.alpha)


@dataclass
class SPRTResult:
    """
    Resultado del test secuencial SPRT sobre una secuencia H_n.
    """
    final_state: str           # "free", "chatter", "indeterminado"
    decision_index: int        # índice del segmento donde decide (-1 si no decide)
    S_history: np.ndarray      # trayectoria de S_n
    a: float                   # umbral inferior
    b: float                   # umbral superior


@dataclass
class SequentialProbabilityRatioTest:
    """
    Motor del SPRT: solo sabe:
    - cómo calcular LLR (objeto LLRModel)
    - qué umbrales usar (SPRTConfig)

    No sabe nada de MaxEnt, ni de señales, ni de PDFs concretas.
    """
    llr_model: LLRModel
    config: SPRTConfig

    def run(self, H_seq: Iterable[float]) -> SPRTResult:
        """
        Ejecuta el SPRT sobre una secuencia H_seq de indicadores H_n.
        """
        H_list = list(H_seq)
        S_hist = np.zeros(len(H_list), dtype=float)
        S = 0.0
        state = "indeterminado"
        idx_decision = -1

        a = self.config.a
        b = self.config.b

        for i, h_obs in enumerate(H_list):
            S += self.llr_model.llr(h_obs)
            S_hist[i] = S

            if S <= a:
                state = "free"
                idx_decision = i
                if self.config.reset_on_H0:
                    S = 0.0

            if S >= b:
                state = "chatter"
                idx_decision = i
                # Se podría parar aquí con break si se quisiera detección temprana.

        return SPRTResult(
            final_state=state,
            decision_index=idx_decision,
            S_history=S_hist,
            a=a,
            b=b,
        )


# ==========================================================
# 7. WRAPPERS FUNCIONALES (OPCIONALES) SOBRE EL MOTOR SPRT
# ==========================================================

@validate_alpha_beta
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


@validate_alpha_beta
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


# ==========================================================
# 8. ENCAPSULACIÓN: DETECTOR MaxEnt + SPRT END-TO-END
# ==========================================================

@dataclass
class MaxEntSPRTConfig:
    """
    Contenedor de configuración para el detector de alto nivel.

    Encapsula parámetros específicos de SPRT (alpha, beta, reset_on_H0).
    """
    alpha: float = 0.01
    beta: float = 0.01
    reset_on_H0: bool = True


@dataclass
class MaxEntSPRTDetector:
    """
    Objeto de alto nivel que encapsula:

    - MaxEntModels (p0, p1) → pdf del indicador H bajo H0/H1.
    - MaxEntSPRTConfig → parámetros del SPRT.
    - EntropyEstimator → cómo se construye el indicador H a partir de segmentos.

    Es un objeto END-TO-END:
    - fit_offline_from_opr / fit_offline_from_signals → entrenamiento offline.
    - detect_from_H_seq → SPRT sobre H ya calculados.
    - detect_online_from_signal → pipeline completa sobre una señal nueva.
    """
    models: MaxEntModels | None = None
    config: MaxEntSPRTConfig = field(default_factory=MaxEntSPRTConfig)
    estimator: EntropyEstimator = field(default_factory=GaussianMaxEntEstimator)

    # Variables de diagnóstico offline (opcionales)
    H_free: np.ndarray | None = field(default=None, init=False)
    H_chat: np.ndarray | None = field(default=None, init=False)
    t_mid_free: np.ndarray | None = field(default=None, init=False)
    t_mid_chat: np.ndarray | None = field(default=None, init=False)

    def _build_sprt_config(self) -> SPRTConfig:
        """
        Construye un SPRTConfig a partir de la configuración de alto nivel.
        """
        return SPRTConfig(
            alpha=self.config.alpha,
            beta=self.config.beta,
            reset_on_H0=self.config.reset_on_H0,
        )

    def _check_models(self) -> MaxEntModels:
        """
        Garantiza que los modelos están entrenados antes de detectar.
        """
        if self.models is None:
            raise RuntimeError("Los modelos MaxEnt no están entrenados. Llama a fit_offline_* primero.")
        return self.models

    # ------------------ OFFLINE ------------------

    def fit_offline_from_opr(
        self,
        opr_free: np.ndarray,
        opr_t_free: np.ndarray,
        opr_chat: np.ndarray,
        opr_t_chat: np.ndarray,
        N_seg: int,
    ) -> "MaxEntSPRTDetector":
        """
        Entrena los modelos MaxEnt (p0(H), p1(H)) directamente a partir de OPR etiquetados.
        """
        models, H_free, H_chat, t_mid_free, t_mid_chat = offline_train_maxent_sprt(
            opr_free=opr_free,
            opr_chat=opr_chat,
            opr_t_free=opr_t_free,
            opr_t_chat=opr_t_chat,
            N_seg=N_seg,
            estimator=self.estimator,
        )
        self.models = models
        self.H_free = H_free
        self.H_chat = H_chat
        self.t_mid_free = t_mid_free
        self.t_mid_chat = t_mid_chat
        return self

    def fit_offline_from_signals(
        self,
        y_free: np.ndarray,
        t_free: np.ndarray,
        y_chat: np.ndarray,
        t_chat: np.ndarray,
        rpm: float,
        ratio_sampling: float,
        N_seg: int,
    ) -> "MaxEntSPRTDetector":
        """
        Wrapper de más alto nivel: parte de señales completas y genera OPR + modelos.
        """
        fr = rpm / 60.0
        fs = ratio_sampling * fr

        opr_free, opr_t_free = sample_opr(y_free, t_free, fs=fs, fr=fr)
        opr_chat, opr_t_chat = sample_opr(y_chat, t_chat, fs=fs, fr=fr)

        return self.fit_offline_from_opr(
            opr_free=opr_free,
            opr_t_free=opr_t_free,
            opr_chat=opr_chat,
            opr_t_chat=opr_t_chat,
            N_seg=N_seg,
        )

    # ------------------ ONLINE / DETECCIÓN ------------------

    def detect_from_H_seq(
        self,
        H_seq: Iterable[float],
    ) -> SPRTResult:
        """
        Ejecuta SPRT sobre una secuencia de H precomputada, usando el motor OO.
        """
        models = self._check_models()
        llr_model = GaussianIndicatorLLR(models=models)
        sprt_config = self._build_sprt_config()
        sprt = SequentialProbabilityRatioTest(llr_model=llr_model, config=sprt_config)
        return sprt.run(H_seq=H_seq)

    def detect_online_from_signal(
        self,
        y_online: np.ndarray,
        t_online: np.ndarray,
        rpm: float,
        ratio_sampling: float,
        N_seg: int,
    ) -> Tuple[SPRTResult, np.ndarray, np.ndarray]:
        """
        Ejecuta la pipeline online completa sobre una señal nueva:

        Devuelve:
            sprt_result: objeto SPRTResult con estado final y S_history.
            H_seq: entropías por segmento.
            t_mid_segments: tiempos medios de cada segmento.
        """
        models = self._check_models()

        fr = rpm / 60.0
        fs = ratio_sampling * fr

        # 1) OPR
        opr_online, opr_t_online = sample_opr(y_online, t_online, fs=fs, fr=fr)

        # 2) Segmentación
        segments_online, segments_t_online = segment_opr(opr_online, opr_t_online, N_seg=N_seg)
        if len(segments_online) == 0:
            raise ValueError("No hay segmentos suficientes en la señal online.")

        # 3) Entropía por segmento
        H_seq = entropy_from_segments(segments_online, estimator=self.estimator)
        t_mid_segments = np.array([np.mean(seg_t) for seg_t in segments_t_online])

        # 4) SPRT con motor OO
        result = self.detect_from_H_seq(H_seq=H_seq)
        return result, H_seq, t_mid_segments


#%%
# ==========================================================
# 9. EJEMPLO DE USO (MAIN) – DEMO INTEGRADA
# ==========================================================

# if __name__ == "__main__":
# --- Parámetros de simulación (ejemplo, puedes adaptarlos) ---
rpm = 15_000.0        # revoluciones por minuto
ratio_sampling = 250  # muestras por revolución
fr = rpm / 60.0       # Hz, frecuencia de rotación
fs = ratio_sampling * fr  # Hz, frecuencia de muestreo
T = 1.0               # s, duración de la señal
N_seg = 20            # nº de revoluciones por segmento
seed = 42

# ------------------- SEÑALES DE ENTRENAMIENTO -------------------
t, y_free = sinus_6_C_SNR(
    fs=fs,
    T=T,
    chatter=False,
    exp=None,
    Amp=5,
    stable_to_chatter=False,
    noise=True,
    SNR_dB=10.0,
    seed=24,
)

t, y_chat = sinus_6_C_SNR(
    fs=fs,
    T=T,
    chatter=True,
    exp=None,
    Amp=5,
    stable_to_chatter=False,
    noise=True,
    SNR_dB=10.0,
    seed=seed,
)

print("Generated chatter-free and chatter-included signals.")
print(f"Size of signal free: {y_free.size} samples.")
print(f"Size of signal chatter: {y_chat.size} samples.")

# Visualización rápida
plt.figure(figsize=(10, 4))
plt.plot(t, y_free, label="Chatter-free")
plt.plot(t, y_chat, label="Chatter-included", alpha=0.7)
plt.legend()
plt.title("Generated signals")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

#%%

# ------------------- OPR DE ENTRENAMIENTO -------------------
opr_free, t_opr_free = sample_opr(y_free, t, fs=fs, fr=fr)
opr_chat, t_opr_chat = sample_opr(y_chat, t, fs=fs, fr=fr)

print(f"Sampled OPR: {opr_free.size} samples free, {opr_chat.size} samples chatter.")

plt.figure(figsize=(10, 4))
plt.plot(t, y_free, label="Chatter-free", alpha=0.7)
plt.scatter(t_opr_free, opr_free, label="OPR free", color="red", alpha=0.7, s=7)
plt.legend()
plt.title("Free OPR samples")
plt.xlabel("Time (s)")
plt.ylabel("OPR Value")

plt.figure(figsize=(10, 4))
plt.plot(t, y_chat, label="Chatter-included", alpha=0.7)
plt.scatter(t_opr_chat, opr_chat, label="OPR chatter", color="red", alpha=0.7, s=7)
plt.legend()
plt.title("Chatter OPR samples")
plt.xlabel("Time (s)")
plt.ylabel("OPR Value")
plt.show()

#%%

# ------------------- DETECTOR END-TO-END GAUSSIANO -------------------
detector_cfg = MaxEntSPRTConfig(alpha=0.01, beta=0.01, reset_on_H0=True)
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

# Histograma de H_free y H_chat
if detector.H_free is not None and detector.H_chat is not None:
    H_free = detector.H_free
    H_chat = detector.H_chat

    plt.figure(figsize=(10, 4))
    plt.hist(H_free, bins=15, alpha=0.5, density=True, label="H free")
    plt.hist(H_chat, bins=15, alpha=0.5, density=True, label="H chatter")
    plt.legend()
    plt.title("Histograms of MaxEnt indicators (Gaussian)")
    plt.xlabel("Entropy H")
    plt.ylabel("Density")
    plt.show()

    xs = np.linspace(
        min(H_free.min(), H_chat.min()) - 0.1,
        max(H_free.max(), H_chat.max()) + 0.1,
        200,
    )
    pdf0 = np.exp([models_trained.p0.logpdf(x) for x in xs])
    pdf1 = np.exp([models_trained.p1.logpdf(x) for x in xs])
    plt.plot(xs, pdf0, label="pdf p0(H) free")
    plt.plot(xs, pdf1, label="pdf p1(H) chatter")
    plt.legend()
    plt.title("MaxEnt indicator PDFs (Gaussian)")
    plt.xlabel("Entropy H")
    plt.ylabel("Probability Density Function (PDF)")
    plt.show()

# ------------------- OPCIONAL: ESTIMADOR EMPÍRICO -------------------
# hist_estimator = EmpiricalHistogramEntropyEstimator(bins=20)
# detector_hist = MaxEntSPRTDetector(config=detector_cfg, estimator=hist_estimator)
# detector_hist.fit_offline_from_opr(
#     opr_free=opr_free,
#     opr_t_free=t_opr_free,
#     opr_chat=opr_chat,
#     opr_t_chat=t_opr_chat,
#     N_seg=N_seg,
# )
# Aquí detector_hist.models contendría modelos p0(H), p1(H) para el indicador empírico.

#%%
# ------------------- FASE ONLINE: SEÑAL NUEVA -------------------
t_on, y_on = sinus_6_C_SNR(
    fs=fs,
    T=T,
    chatter=False,
    exp=False,
    Amp=5,
    stable_to_chatter=False,
    noise=True,
    SNR_dB=10.0,
    seed=seed,
)

plt.figure(figsize=(10, 4))
plt.plot(t_on, y_on, label="Chatter test", alpha=0.7)
plt.legend()
plt.title("Test signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()



sprt_result, H_seq_online, t_mid_segments = detector.detect_online_from_signal(
    y_online=y_on,
    t_online=t_on,
    rpm=rpm,
    ratio_sampling=ratio_sampling,
    N_seg=N_seg,
)

print(f"ONLINE FINAL STATE: {sprt_result.final_state}, decision at segment {sprt_result.decision_index}")

# Visualización de H_seq y S_history
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
ax[0].plot(t_mid_segments, H_seq_online, marker="o")
ax[0].set_ylabel("H (MaxEnt per segment)")
ax[0].set_title("Evolution of MaxEnt indicator")

ax[1].plot(t_mid_segments, sprt_result.S_history, marker="o")
ax[1].axhline(sprt_result.a, linestyle="--", linewidth=0.8)
ax[1].axhline(sprt_result.b, linestyle="--", linewidth=0.8)
ax[1].set_ylabel("S_n (SPRT)")
ax[1].set_xlabel("Time (s)")
ax[1].set_title("Evolution of SPRT statistic")

plt.tight_layout()
plt.show()
