
from __future__ import annotations
from typing import Sequence
import numpy as np
from abc import ABC, abstractmethod
from ..models.prob import GaussianPDF


class EntropyEstimator(ABC):
    """
    Abstract interface for entropy estimation from signal segments.

    Subclasses implement ``entropy_from_segment`` while this base class provides
    a convenience method to process a sequence of segments.
    """

    @abstractmethod
    def entropy_from_segment(self, seg: np.ndarray) -> float:
        """
        Compute one scalar entropy value from a single segment.

        :param seg: One-dimensional segment containing the samples of one analysis window.

        Returns:
            float: Entropy-like scalar feature representing the information
            content or spread of the segment.
        """

    def entropy_from_segments(self, segments: Sequence[np.ndarray]) -> np.ndarray:
        """
        Calculate entropy values for multiple segments.

        Computes the entropy for each segment in the input sequence using the
        entropy_from_segment method.

        :param segments: Ordered collection of 1D signal segments, each processed independently.

        Returns:
            np.ndarray: One-dimensional float array containing one entropy value
            per input segment, preserving input order.
        """
        return np.array(
            [self.entropy_from_segment(seg) for seg in segments],
            dtype=float,
        )

class GaussianMaxEntEstimator(EntropyEstimator):
    """
    Entropy estimator under a Gaussian maximum-entropy assumption.

    For each segment, the method fits a Gaussian distribution from samples and
    returns its Shannon differential entropy. This is the default estimator used
    by the detector because it is compact, fast, and directly compatible with
    the Gaussian likelihood models used in SPRT.
    """
    def entropy_from_segment(self, seg: np.ndarray) -> float:
        """
        Estimate segment entropy under a Gaussian maximum-entropy assumption.

        :param seg: One-dimensional segment whose sample mean and variance define the Gaussian surrogate model.

        Returns:
            float: Shannon differential entropy of the fitted Gaussian model for
            the input segment.
        """
        gaussian = GaussianPDF.from_samples(seg)
        return gaussian.entropy_shannon()

class EmpiricalHistogramEntropyEstimator(EntropyEstimator):
    """
    Histogram-based nonparametric entropy estimator.

    The estimator approximates the empirical distribution of a segment through a
    finite histogram and then computes the discrete Shannon entropy
    ``H = -sum p_i log(p_i)``. Unlike ``GaussianMaxEntEstimator``, it does not
    assume a parametric Gaussian model.
    """

    bins: int
    """Number of histogram bins used to approximate the segment distribution."""

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

        :param seg: One-dimensional array containing the raw sample values of the segment to summarize.

        Returns
        -------
        float
            The Shannon entropy value of the segment. Returns 0.0 if the total
            count of histogram values is zero.

        Raises
        ------
        ValueError
            If the input segment is empty.

        Notes:
            Uses the natural logarithm, ignores zero-probability bins, and bases
            the discretization on the configured number of histogram bins.
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
    Compute entropy values for a sequence of segments with a chosen estimator.

    The function provides a single entry point for batch entropy extraction in
    both offline training and online detection pipelines. If no estimator is
    provided, it defaults to :class:`GaussianMaxEntEstimator`.

    :param segments: Sequence of one-dimensional signal segments to convert into scalar entropy values.
    :param estimator: Estimator implementation used to process each segment. If ``None``, ``GaussianMaxEntEstimator`` is used.

    Returns:
        np.ndarray: One entropy value per input segment, preserving input order.

    Notes:
        The returned values are always ``float`` and can be consumed directly by
        LLR/SPRT routines.
    """
    est = estimator or GaussianMaxEntEstimator()
    return est.entropy_from_segments(segments)
