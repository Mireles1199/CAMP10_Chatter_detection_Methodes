from __future__ import annotations
# Comentario: regla con selección flexible del tramo estable (por índices o por tiempo)
from typing import Dict, Sequence, Optional, Tuple, Union
import numpy as np
from statsmodels.stats.diagnostic import lilliefors
from abc import ABC, abstractmethod

IndexRange = Union[int, Tuple[int, int]]          # Comentario: N  ó  (i0, i1)
TimeRange  = Union[float, Tuple[float, float]]    # Comentario: t_end  ó  (t0, t1)

class DetectionRule(ABC):
    """
    Abstract base class for chatter detection rules.
    This class defines the interface that all detection strategies must implement.
    Subclasses should provide concrete implementations of the detect method to
    identify chatter patterns in vibration or acoustic signals.
    """

    @abstractmethod
    def detect(
        self,
        d1: np.ndarray,
        idx_stable: Optional[Sequence[int]] = None,
        *,
        t: Optional[np.ndarray] = None,
        stable_index: Optional[IndexRange] = None,
        stable_time: Optional[TimeRange] = None,
    ) -> Dict[str, object]:
        raise NotImplementedError

class ThreeSigmaWithLilliefors(DetectionRule):
    """
    ThreeSigmaWithLilliefors Detection Strategy
    A robust anomaly detection method that uses the three-sigma rule with automatic
    fallback to MAD (Median Absolute Deviation) based thresholding when data normality
    is rejected by the Lilliefors test.
    Attributes
    ----------
    frac_stable : float
        Fraction of initial samples to use for stable region estimation (0 to 1).
        Used as fallback when explicit stable indices/times are not provided.
    alpha : float, default=0.05
        Significance level for the Lilliefors normality test.
    z : float, default=3.0
        Number of standard deviations (or MAD multiples) for threshold calculation.
        Typical value is 3.0 for three-sigma rule.
    fallback_mad : bool, default=True
        If True, switches to MAD-based thresholding when Lilliefors test rejects
        normality. If False, uses sigma-based limits regardless of normality.
    stable_index : IndexRange, optional
        Index range specifying the stable region as [i0, i1).
        Takes precedence over stable_time and frac_stable.
    stable_time : TimeRange, optional
        Time range specifying the stable region as [t0, t1].
        Used if stable_index is None. Takes precedence over frac_stable.
    Methods
    -------
    detect(d1, idx_stable=None, *, t=None) -> Dict[str, object]
        Perform chatter detection on the input signal.
        Parameters
        ----------
        d1 : np.ndarray
            1D input signal to analyze.
        idx_stable : Sequence[int], optional
            Explicit indices defining the stable region. Highest priority.
        t : np.ndarray, optional
            Time vector. Required if stable_time is used.
        Returns
        -------
        Dict[str, object]
            Dictionary containing:
            - mask : np.ndarray
                Binary mask (0=normal, 1=anomaly/chatter).
            - mu : float
                Mean or median of stable region.
            - sigma : float
                Standard deviation or MAD-based sigma of stable region.
            - lim_inf, lim_sup : float
                Lower and upper thresholds for anomaly detection.
            - normal_ok : bool
                Lilliefors test result (True if data is normally distributed).
            - p_value : float
                p-value from Lilliefors test.
            - metodo_umbral : str
                Method used ('sigma' or 'MAD').
            - idx_estable_usados : list
                Indices used for stable region estimation.
    Notes
    -----
    Priority for stable region selection:
        1. idx_stable argument (if provided)
        2. self.stable_index (if set)
        3. self.stable_time (if set and time vector t provided)
        4. Fraction-based: first frac_stable*n samples
    If stable region is empty or contains single value, raises ValueError.
    """

    def __init__(self, frac_stable: float , alpha: float = 0.05, z: float = 3.0, fallback_mad: bool = True,
                 stable_time: Optional[TimeRange] = None, stable_index: Optional[IndexRange] = None):
        self.frac_stable = float(frac_stable)
        self.alpha = float(alpha)
        self.z = float(z)
        self.fallback_mad = bool(fallback_mad)
        self.stable_index: Optional[IndexRange] = stable_index
        self.stable_time: Optional[TimeRange] = stable_time

        if stable_time is not None and stable_index is not None:
            self.stable_index = stable_index  # Priority given to stable_index
            self.stable_time = None
            print("Warning: both stable_time and stable_index provided; using stable_index only.")


    def _build_idx_from_ranges(        self,
        n: int,
        *,
        t: Optional[np.ndarray],
        stable_index: Optional[IndexRange],
        stable_time: Optional[TimeRange],
    ) -> Optional[np.ndarray]:
        """
        Build an index array from either index-based or time-based ranges.
        This method constructs a NumPy array of indices that fall within a specified
        range, prioritizing index-based ranges over time-based ranges.
        Parameters
        ----------
        n : int
            The maximum length/size constraint for the index range.
        t : np.ndarray, optional
            Time vector required when stable_time is used. Must be provided if
            stable_time is not None.
        stable_index : int or tuple[int, int], optional
            Index-based range specification. If an int, interpreted as upper bound [0, N).
            If a tuple, interpreted as [lower, upper] bounds (inclusive). Takes priority
            over stable_time.
        stable_time : float or tuple[float, float], optional
            Time-based range specification. If a float, interpreted as upper bound 
            [min(t), value]. If a tuple, interpreted as [lower, upper] bounds (inclusive).
            Only used if stable_index is None. Requires t parameter.
        Returns
        -------
        np.ndarray of int or None
            Array of indices within the specified range, or None if neither stable_index
            nor stable_time is provided.
        Raises
        ------
        ValueError
            If stable_index produces an empty range (i1 < i0).
            If stable_time is misspecified (t1 < t0).
            If stable_time produces an empty range (no indices match the time mask).
            If stable_time is provided but t is None.
        Notes
        -----
        - Priority order: stable_index > stable_time
        - Index bounds are clipped to [0, n-1]
        - Both ranges are inclusive
        """

        if stable_index is not None:
            if isinstance(stable_index, int):
                i0, i1 = 0, int(stable_index)        # Comentario: [0, N)
            else:
                i0, i1 = int(stable_index[0]), int(stable_index[1])
            i0 = max(0, i0)
            i1 = min(n - 1, i1)
            if i1 < i0:
                raise ValueError("stable_index produces empty range")
            return np.arange(i0, i1 + 1, dtype=int)

        if stable_time is not None:
            if t is None:
                raise ValueError("stable_time require vector t")
            tt = np.asarray(t, dtype=float)
            if isinstance(stable_time, (float, int)):
                t0, t1 = float(np.min(tt)), float(stable_time)
            else:
                t0, t1 = float(stable_time[0]), float(stable_time[1])
            if t1 < t0:
                raise ValueError("stable_time invalid (t1 < t0)")
            mask = (tt >= t0) & (tt <= t1)
            idx = np.nonzero(mask)[0]
            if idx.size == 0:
                raise ValueError("stable_time pruduces empty range")
            return idx.astype(int)
        return None

    def detect(        self,
        d1: np.ndarray,
        idx_stable: Optional[Sequence[int]] = None,
        *,
        t: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """
        Detect anomalies or outliers in a 1D signal using statistical thresholding.
        This method identifies values in the input signal that deviate significantly from
        a baseline established using a stable region. It supports both parametric (sigma-based)
        and non-parametric (MAD-based) thresholding methods, with automatic fallback to MAD
        if the stable region fails a normality test.
        Parameters
        ----------
        d1 : np.ndarray
            The 1D input signal to analyze for anomalies.
        idx_stable : Optional[Sequence[int]], default=None
            Explicit indices defining the stable region of the signal. If not provided,
            the stable region is constructed from `stable_index`, `stable_time`, or 
            `frac_stable` attributes. Indices are validated to be within [0, n).
        t : Optional[np.ndarray], default=None
            Time or position array corresponding to d1, used when constructing the stable
            region from time-based ranges. Only applicable if idx_stable is None.
        Returns
        -------
        Dict[str, object]
            A dictionary containing:
            - "mask" (np.ndarray): Binary array (0/1) where 1 indicates an anomaly.
            - "mu" (float): Mean or median of the stable region.
            - "sigma" (float): Standard deviation or robust sigma (MAD) of the stable region.
            - "lim_inf" (float): Lower detection threshold.
            - "lim_sup" (float): Upper detection threshold.
            - "normal_ok" (bool): Result of Lilliefors normality test (True if p >= alpha).
            - "p_value" (float): P-value from the normality test.
            - "metodo_umbral" (str): Thresholding method used ("sigma" or "MAD").
            - "idx_estable_usados" (list): Indices used to define the stable region.
        Raises
        ------
        ValueError
            If d1 is not 1D or if the stable region is empty after validation.
        """

        d1 = np.asarray(d1, dtype=float)
        if d1.ndim != 1:
            raise ValueError("d1 must be 1D")
        n = d1.size

        #idx_estable: explicit stable indices (highest priority) o construido a partir de rangos
        if idx_stable is not None:
            idx_est = np.asarray(list(idx_stable), dtype=int)
        else:
            built = self._build_idx_from_ranges(n, t=t, stable_index=self.stable_index, stable_time=self.stable_time)
            if built is not None:
                idx_est = built
            else:
                m = max(1, int(self.frac_stable * n))
                idx_est = np.arange(0, m, dtype=int)

        # cut indices to valid range
        idx_est = idx_est[(idx_est >= 0) & (idx_est < n)]
        if idx_est.size == 0:
            raise ValueError("idx_estable vacío tras validación")

        d_est = d1[idx_est]

        mu = float(np.mean(d_est))
        sigma = float(np.std(d_est, ddof=1)) if d_est.size > 1 else 0.0

        # Normality test(Lilliefors)
        try:
            _, p_value = lilliefors(d_est, dist="norm")
            normal_ok = bool(p_value >= self.alpha)
        except Exception:
            p_value = 0.0
            normal_ok = False

        z = self.z
        if sigma == 0.0:
            eps = 1e-12 if mu == 0 else 1e-6 * abs(mu)
            lim_inf, lim_sup = mu - z * eps, mu + z * eps
        else:
            lim_inf, lim_sup = mu - z * sigma, mu + z * sigma

        metodo = "sigma"
        if self.fallback_mad and not normal_ok:
            med = float(np.median(d_est))
            mad = float(np.median(np.abs(d_est - med)))
            sigma_rob = 1.4826 * mad
            if sigma_rob == 0.0:
                eps = 1e-12 if med == 0 else 1e-6 * abs(med)
                lim_inf, lim_sup = med - z * eps, med + z * eps
            else:
                lim_inf, lim_sup = med - z * sigma_rob, med + z * sigma_rob
            mu, sigma = med, sigma_rob
            metodo = "MAD"

        mask = ((d1 < lim_inf) | (d1 > lim_sup)).astype(int)
        return {
            "mask": mask,
            "mu": mu,
            "sigma": sigma,
            "lim_inf": lim_inf,
            "lim_sup": lim_sup,
            "normal_ok": normal_ok,
            "p_value": float(p_value),
            "metodo_umbral": metodo,
            "idx_estable_usados": idx_est.tolist(),
        }
