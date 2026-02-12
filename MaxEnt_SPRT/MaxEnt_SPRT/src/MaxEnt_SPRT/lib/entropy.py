
from __future__ import annotations
from typing import Sequence
import numpy as np
from abc import ABC, abstractmethod
from ..models.prob import GaussianPDF


class EntropyEstimator(ABC):
    """
    Abstract base class for entropy estimation from signal segments.
    This class defines the interface for calculating entropy metrics from OPR (Operating Point Resident)
    signal segments. Subclasses must implement the entropy calculation for individual segments,
    while this base class provides a vectorized method for multiple segments.
    Methods:
        entropy_from_segment: Abstract method to calculate entropy for a single segment.
        entropy_from_segments: Calculate entropy for multiple segments in sequence.
    """

    @abstractmethod
    def entropy_from_segment(self, seg: np.ndarray) -> float:
        """
        Calculate entropy for a single signal segment.
        This abstract method must be implemented by subclasses to compute the entropy
        metric for an individual OPR (Operating Point Resident) signal segment.
        Args:
            seg (np.ndarray): A one-dimensional numpy array representing a signal segment.
        Returns:
            float: The entropy value calculated for the given segment.
        Raises:
            NotImplementedError: If not implemented by a subclass.
        """

    def entropy_from_segments(self, segments: Sequence[np.ndarray]) -> np.ndarray:
        """
        Calculate entropy values for multiple segments.
        Computes the entropy for each segment in the input sequence using the
        entropy_from_segment method.
        Args:
            segments: A sequence of numpy arrays, where each array represents a segment
                     for which entropy will be calculated.
        Returns:
            np.ndarray: A 1D array of float values containing the entropy for each
                       input segment, in the same order as provided.
        """
        return np.array(
            [self.entropy_from_segment(seg) for seg in segments],
            dtype=float,
        )

class GaussianMaxEntEstimator(EntropyEstimator):
    """
    Gaussian Maximum Entropy Estimator.
    A concrete implementation of EntropyEstimator that computes entropy
    using Gaussian distribution assumptions. This estimator fits a Gaussian
    probability density function to signal segments and calculates the
    Shannon entropy of the resulting distribution.
    Attributes:
        Inherits from EntropyEstimator base class.
    Methods:
        entropy_from_segment: Computes Shannon entropy from a segment of data
                             by fitting a Gaussian distribution to the samples.
    """
    def entropy_from_segment(self, seg: np.ndarray) -> float:
        gaussian = GaussianPDF.from_samples(seg)
        return gaussian.entropy_shannon()

class EmpiricalHistogramEntropyEstimator(EntropyEstimator):
    """
    Empirical Histogram Entropy Estimator:

    1) Estimates the empirical distribution via normalized histogram.
    2) Calculates H = -sum p_i log p_i.

    This does NOT assume a parametric model and allows comparison against Gaussian MaxEnt.

    """

    def __init__(self, bins: int = 20) -> None:
        if bins <= 0:
            raise ValueError("bins debe ser un entero positivo.")
        self.bins: int = bins

    def entropy_from_segment(self, seg: np.ndarray) -> float:
        """
        Calculate the Shannon entropy of a segment of data.

        Computes the discrete entropy of an input array by binning the data into
        a histogram and calculating the information entropy using the formula:
        H = -sum(p * ln(p)), where p is the probability of each bin.

        Parameters
        ----------
        seg : np.ndarray
            Input segment as a numpy array containing numerical values.

        Returns
        -------
        float
            The Shannon entropy value of the segment. Returns 0.0 if the total
            count of histogram values is zero.

        Raises
        ------
        ValueError
            If the input segment is empty.

        Notes
        -----
        - Uses natural logarithm (ln) for entropy calculation
        - Only non-zero probability bins contribute to the sum
        - Binning is based on self.bins attribute
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
    Calculate entropy from a sequence of data segments using a specified estimator.
    This function computes the entropy of multiple data segments using the provided
    entropy estimation method. If no estimator is specified, it defaults to using
    a Gaussian Maximum Entropy estimator.
    Args:
        segments: A sequence of numpy arrays, where each array represents a data segment
                 for which entropy will be calculated.
        estimator: An optional EntropyEstimator instance to use for entropy calculation.
                  If None, defaults to GaussianMaxEntEstimator. Defaults to None.
    Returns:
        np.ndarray: An array containing the entropy values calculated for each segment.
    Example:
        >>> import numpy as np
        >>> segments = [np.random.randn(100), np.random.randn(100)]
        >>> entropies = entropy_from_segments(segments)
    """
    est = estimator or GaussianMaxEntEstimator()
    return est.entropy_from_segments(segments)
