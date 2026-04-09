
from __future__ import annotations
from dataclasses import dataclass
import numpy as np, math
from typing import Iterable

@dataclass(frozen=True)
class GaussianPDF:
    """
    Gaussian probability density model.

    The class stores the mean and standard deviation and exposes helpers for
    log-density evaluation, entropy computation, and fitting from samples.

    This lightweight container is the probabilistic core used by MaxEnt-SPRT:
    one instance represents the stable regime (H0) and another the chatter
    regime (H1).
    """
    mu: float
    """Mean of the Gaussian distribution, i.e. the center of the modeled indicator distribution."""
    sigma: float  # > 0
    """Standard deviation. Must be strictly positive; controls the spread/uncertainty around ``mu``."""

    def __post_init__(self) -> None:
        if not np.isfinite(self.mu):
            raise ValueError("mu no es finito.")
        if not (np.isfinite(self.sigma) and self.sigma > 0.0):
            raise ValueError("sigma debe ser finito y > 0.")

    def logpdf(self, x: float) -> float:
        """
        Evaluate the natural logarithm of the Gaussian PDF at ``x``.

        :param x: Scalar observation at which the Gaussian log-density is evaluated.

        Returns:
            float: ``log N(x ; mu, sigma^2)``.

        Notes:
            Using log-density improves numerical stability when values are later
            combined into cumulative LLR statistics.
        """
        z = (x - self.mu) / self.sigma
        return -0.5 * (math.log(2.0 * math.pi) + 2.0 * math.log(self.sigma) + z * z)

    def entropy_shannon(self) -> float:
        """
        Calculate the Shannon entropy of the Gaussian distribution.

        The Shannon entropy for a continuous Gaussian distribution N(mu, sigma^2) is given by:
        H(X) = 0.5 * log(2 * pi * e * sigma^2)

        Returns:
            float: Differential Shannon entropy in nats, fully determined by the
            current value of ``sigma``.
        """
        return 0.5 * math.log(2.0 * math.pi * math.e * (self.sigma ** 2))

    @staticmethod
    def from_samples(samples: Iterable[float], eps: float = 1e-12) -> "GaussianPDF":
        """
        Create a GaussianPDF instance from a collection of sample data.

        Estimates the mean and standard deviation of a Gaussian distribution
        from the provided samples using maximum likelihood estimation.

        :param samples: Collection of scalar observations used to estimate the Gaussian mean and spread.
        :param eps: Minimum variance floor used before taking the square root, ensuring the resulting standard deviation is never zero.

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