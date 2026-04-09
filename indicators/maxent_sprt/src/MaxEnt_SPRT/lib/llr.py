
from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from ..models.maxent import MaxEntModels

class LLRModel(ABC):
    """
    Abstract interface for log-likelihood ratio models.

    Implementations must provide a scalar ``llr`` evaluation for one observed
    indicator value.
    """
    @abstractmethod
    def llr(self, h_obs: float) -> float:
        """
        Evaluate the incremental log-likelihood ratio for one observation.

        :param h_obs: Observed scalar feature value to score, usually an entropy-like indicator extracted from one segment.

        Returns:
            float: Signed evidence contribution added to the cumulative SPRT
            statistic. Positive values favor H1 and negative values favor H0.
        """


@dataclass(frozen=True)
class GaussianIndicatorLLR(LLRModel):
    """
    Gaussian log-likelihood ratio model for chatter detection.

    This implementation compares two Gaussian maximum-entropy models:
    ``p0`` (stable regime) and ``p1`` (chatter regime). For each observed
    indicator value ``h_obs``, it returns the evidence in favor of chatter as:

    ``log p1(h_obs) - log p0(h_obs)``.

    Interpretation of the output is straightforward:

    - Positive values: evidence favoring chatter (H1).
    - Negative values: evidence favoring stable cutting (H0).
    - Values near zero: weak or ambiguous evidence.

    The class is immutable (``frozen=True``), which helps keep SPRT runs
    reproducible once the statistical models are fitted.
    """
    models: MaxEntModels
    """Pair of Gaussian densities where ``p0`` represents the stable regime and ``p1`` represents the chatter regime."""

    def llr(self, h_obs: float) -> float:
        """
        Evaluate the log-likelihood ratio at one observed indicator value.

        :param h_obs: Observed entropy-like indicator value associated with the current segment or analysis window.

        Returns:
            float: Signed evidence score for SPRT updates,
                ``log p1(h_obs) - log p0(h_obs)``.
        """
        return self.models.p1.logpdf(h_obs) - self.models.p0.logpdf(h_obs)