
from __future__ import annotations
from dataclasses import dataclass
import numpy as np, math
from typing import Iterable

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