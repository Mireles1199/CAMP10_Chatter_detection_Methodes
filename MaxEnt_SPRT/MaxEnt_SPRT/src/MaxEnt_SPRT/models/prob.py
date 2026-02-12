
from __future__ import annotations
from dataclasses import dataclass
import numpy as np, math
from typing import Iterable

@dataclass(frozen=True)
class GaussianPDF:
    """
    A class representing a Gaussian (normal) probability distribution.
    This class provides methods to compute log-probability density, entropy,
    and to fit a Gaussian distribution from sample data.
    Attributes:
        mu (float): Mean of the distribution.
        sigma (float): Standard deviation of the distribution. Must be positive.
    Raises:
        ValueError: If mu is not finite or if sigma is not finite and positive.
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
        Calculate the Shannon entropy of the Gaussian distribution.
        The Shannon entropy for a continuous Gaussian distribution N(mu, sigma^2) is given by:
        H(X) = 0.5 * log(2 * pi * e * sigma^2)
        Returns:
            float: The Shannon entropy of the Gaussian distribution.
        """
        return 0.5 * math.log(2.0 * math.pi * math.e * (self.sigma ** 2))

    @staticmethod
    def from_samples(samples: Iterable[float], eps: float = 1e-12) -> "GaussianPDF":
        """
        Create a GaussianPDF instance from a collection of sample data.
        Estimates the mean and standard deviation of a Gaussian distribution
        from the provided samples using maximum likelihood estimation.
        Args:
            samples: An iterable of float values representing the data samples.
            eps: Minimum threshold for variance to prevent numerical issues (default: 1e-12).
                 Ensures sigma is never zero or extremely small.
        Returns:
            GaussianPDF: A new GaussianPDF instance with estimated mu and sigma parameters.
        Raises:
            ValueError: If fewer than 2 samples are provided. At least 2 samples are required
                        to estimate the standard deviation using Bessel's correction (ddof=1).
        Examples:
            >>> samples = [1.0, 2.0, 3.0, 4.0, 5.0]
            >>> pdf = GaussianPDF.from_samples(samples)
            >>> print(pdf.mu, pdf.sigma)
        """
        x = np.asarray(list(samples), dtype=float)
        if x.size < 2:
            raise ValueError("Se requieren al menos 2 muestras para estimar sigma.")
        mu = float(np.mean(x))
        var = float(np.var(x, ddof=1))
        sigma = math.sqrt(max(var, eps))
        return GaussianPDF(mu=mu, sigma=sigma)