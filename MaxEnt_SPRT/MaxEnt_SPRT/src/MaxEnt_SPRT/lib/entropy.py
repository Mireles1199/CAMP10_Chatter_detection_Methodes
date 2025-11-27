
from __future__ import annotations
from typing import Sequence
import numpy as np
from abc import ABC, abstractmethod
from ..models.prob import GaussianPDF


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
