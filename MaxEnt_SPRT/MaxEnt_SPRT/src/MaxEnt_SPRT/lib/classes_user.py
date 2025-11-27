# Clases originales (sin cambios de nombre)
import numpy as np
import math
from typing import Iterable, Tuple, Sequence
from abc import ABC, abstractmethod
from dataclasses import  dataclass, field

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
    b: float

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