
from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from ..models.maxent import MaxEntModels

class LLRModel(ABC):
    """
    Abstract base class for Log-Likelihood Ratio (LLR) models.
    This class defines the interface for computing the log-likelihood ratio between
    two hypotheses (H1 and H0) for a given observation.
    Methods:
        llr: Compute the log-likelihood ratio for an observation.
    """
    @abstractmethod
    def llr(self, h_obs: float) -> float:
        """
        Calculate the log-likelihood ratio between two hypotheses.

        Computes the natural logarithm of the ratio of probability densities:
        log(p1(h_obs) / p0(h_obs)), where p1 and p0 represent the probability
        density functions under hypothesis H1 and H0 respectively.

        Args:
            h_obs (float): The observed test statistic or measurement value.

        Returns:
            float: The log-likelihood ratio log(p1(h_obs) / p0(h_obs)).
                   Positive values indicate evidence favoring H1,
                   negative values indicate evidence favoring H0.

        Notes:
            This method is typically used in Sequential Probability Ratio Test (SPRT)
            for hypothesis testing.
        """


@dataclass(frozen=True)
class GaussianIndicatorLLR(LLRModel):
    """
    Gaussian Indicator Log-Likelihood Ratio Model.
    A specialized LLR (Log-Likelihood Ratio) model that computes the likelihood ratio
    between two Gaussian distributions for chatter detection.
    Attributes:
        models (MaxEntModels): Container holding the maximum entropy statistical models,
            including p0 (null hypothesis distribution) and p1 (alternative hypothesis distribution).
    Methods:
        llr(h_obs): Computes the log-likelihood ratio given an observed indicator value.
    """
    models: MaxEntModels

    def llr(self, h_obs: float) -> float:
        return self.models.p1.logpdf(h_obs) - self.models.p0.logpdf(h_obs)