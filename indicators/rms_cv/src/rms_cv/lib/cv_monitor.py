"""Online Coefficient of Variation (CV) monitor for RMS-based chatter detection.

Provides three cooperating objects:

* :class:`CVOnlineConfig` — frozen configuration (window size, thresholds,
  timing).
* :class:`CVOnlineState` — mutable running-statistics state updated each
  step.
* :class:`CVOnlineMonitor` — streaming monitor that ingests one RMS value
  at a time and raises an alert when CV or raw RMS exceeds a threshold.

Algorithm
---------
Statistics are maintained with O(1) running sums so that the window can be
slid efficiently:

.. math::

    \\mu_n = \\frac{\\sum_{i} x_i}{n}, \\quad
    \\sigma_n = \\sqrt{\\frac{\\sum_{i} x_i^2 - (\\sum_{i} x_i)^2 / n}{n - 1}}

When the window is full the oldest sample is removed from the accumulators
before the new value is added, keeping the update O(1) per step.
"""

# Monitor en línea de CV sobre una secuencia de valores RMS
from __future__ import annotations
from dataclasses import dataclass
from typing import Deque, Dict, Any, Optional, Sequence, Tuple, Union
from collections import deque
import numpy as np
from statsmodels.stats.diagnostic import lilliefors

IndexRange = Union[int, Tuple[int, int]]
TimeRange  = Union[float, Tuple[float, float]]

@dataclass
class CVOnlineConfig:
    """
    Configuration class for online coefficient of variation (CV) monitoring.
    Attributes:
        n_max (int): Maximum number of samples to consider in the sliding window.
        use_unbiased_std (bool): Whether to use unbiased standard deviation calculation.
                                 Defaults to True.
        eps (float): Small epsilon value to avoid division by zero. Defaults to 1e-12.
        cv_threshold (Optional[float]): Threshold value for CV alert triggering.
                                        If None, CV monitoring is disabled. Defaults to None.
        rms_threshold (Optional[float]): Threshold value for RMS alert triggering.
                                         If None, RMS monitoring is disabled. Defaults to None.
        n_min_cv (int): Minimum number of samples required before CV calculation begins.
                        Defaults to 2.
        warmup_ignore_alerts (bool): Whether to ignore alerts during the warmup period
                                     (before n_min_cv samples are collected). Defaults to False.
        fs_rms (Optional[float]): Sampling frequency for RMS calculation in Hz.
                                  If None, uses default frequency. Defaults to None.
        dt_rms (Optional[float]): Time step for RMS calculation in seconds.
                                  If None, computed from fs_rms. Defaults to None.
        start_time (float): Initial timestamp reference for monitoring. Defaults to 0.0.
    """
    n_max: int
    use_unbiased_std: bool = True
    eps: float = 1e-12

    cv_threshold: Optional[float] = None
    rms_threshold: Optional[float] = None

    n_min_cv: int = 2
    warmup_ignore_alerts: bool = False

    fs_rms: Optional[float] = None
    dt_rms: Optional[float] = None
    start_time: float = 0.0


@dataclass
class CVOnlineState:
    """
    A class to track the online state of coefficient of variation (CV) calculations.
    This class maintains running statistics for computing the mean, standard deviation,
    and coefficient of variation of a data stream in an online/streaming fashion.
    Attributes:
        n (int): The count of samples processed so far. Defaults to 0.
        sum1 (float): The cumulative sum of all values. Used to calculate the mean. Defaults to 0.0.
        sum2 (float): The cumulative sum of squared values. Used to calculate variance. Defaults to 0.0.
        mu (float): The current mean (average) of the data. Defaults to 0.0.
        sigma (float): The current standard deviation of the data. Defaults to 0.0.
        cv (float): The coefficient of variation (sigma / mu), expressed as a normalized measure of dispersion. Defaults to 0.0.
        t_last (Optional[float]): The timestamp of the last update or measurement. None if no measurement has been taken yet. Defaults to None.
        idx (int): An index or identifier for tracking the current state or sample position. Defaults to 0.
    """

    n: int = 0
    sum1: float = 0.0
    sum2: float = 0.0
    mu: float = 0.0
    sigma: float = 0.0
    cv: float = 0.0
    t_last: Optional[float] = None
    idx: int = 0


class CVOnlineMonitor:
    """Real-time sliding-window CV monitor.

    Processes one RMS value per call to :meth:`update`.  Internally maintains
    a :class:`~collections.deque` of the last ``n_max`` values together with
    their running sum and sum-of-squares so that mean, standard deviation,
    and CV are updated in O(1) time per step.

    Two alert modes, controlled by :class:`CVOnlineConfig`:

    * **Warmup** (``n < n_min_cv``): optionally raises an ``"rms"`` alert
      when the raw value exceeds ``rms_threshold`` (suppressed when
      ``warmup_ignore_alerts`` is ``True``).
    * **Normal** (``n >= n_min_cv``): raises a ``"cv"`` alert when
      :math:`\\sigma / |\\mu| \\geq` ``cv_threshold``.

    Attributes:
        config (CVOnlineConfig): Configuration supplied at construction;
            not modified after initialisation.
        state (CVOnlineState): Mutable statistics updated by each
            :meth:`update` call.  Reset by :meth:`reset`.
        window (collections.deque): Fixed-size deque holding the raw RMS
            values currently in the sliding window
            (``maxlen = config.n_max``).

    Args:
        config (CVOnlineConfig): Monitor configuration.

    Raises:
        ValueError: If ``config.n_max < 1`` or ``config.n_min_cv < 1``.
    """

    def __init__(self, config: CVOnlineConfig) -> None:
        if config.n_max < 1:
            raise ValueError("`n_max` debe ser >= 1.")
        if config.n_min_cv < 1:
            raise ValueError("`n_min_cv` debe ser >= 1.")
        if config.dt_rms is None and config.fs_rms:
            config.dt_rms = 1.0 / float(config.fs_rms)

        self.config: CVOnlineConfig = config
        self.state: CVOnlineState = CVOnlineState()
        self.window: Deque[float] = deque(maxlen=config.n_max)

    def reset(self) -> None:
        """Reset the monitor to its initial state.

        Clears the sliding window and reinitialises :attr:`state` to a fresh
        :class:`CVOnlineState` instance (all counters and accumulators zero).
        """
        self.window.clear()
        self.state = CVOnlineState()

    def update(self, rms_value: float) -> Dict[str, Any]:
        """
        Update the monitor with a new RMS value and determine if an alert should be triggered.
        This method maintains a sliding window of RMS values and computes statistical metrics
        (mean, standard deviation, and coefficient of variation). It checks against configured
        thresholds to determine if alert conditions are met.
        Args:
            rms_value (float): The RMS value to process. If None, returns no alert.
        Returns:
            Dict[str, Any]: A result dictionary containing alert status and reason, generated
                            by _result(). The reason can be "rms" (RMS threshold exceeded during
                            warmup), "cv" (coefficient of variation threshold exceeded), or None.
        Behavior:
            - Maintains a fixed-size sliding window of the last n_max RMS values.
            - Computes running sum and sum of squares for efficient variance calculation.
            - Calculates mean (mu), standard deviation (sigma), and coefficient of variation (cv).
            - During warmup phase (n < n_min_cv): triggers alert if RMS exceeds rms_threshold
              (unless warmup_ignore_alerts is enabled).
            - After warmup phase: triggers alert if CV exceeds cv_threshold (unless warmup_ignore_alerts
              is enabled and still in warmup).
            - Uses unbiased or biased standard deviation estimator based on use_unbiased_std config.
            - Uses eps (epsilon) to prevent division by zero when computing CV.
        """
        cfg = self.config
        st = self.state

        if rms_value is None:
            return self._result(alert=False, reason=None)

        x = float(rms_value)

        if cfg.dt_rms is not None:
            st.t_last = cfg.start_time + st.idx * cfg.dt_rms

        if st.n < cfg.n_max:
            self.window.append(x)
            st.n += 1
            st.sum1 += x
            st.sum2 += x * x
        else:
            oldest = self.window[0]
            self.window.append(x)
            st.sum1 += x - oldest
            st.sum2 += (x * x) - (oldest * oldest)

        n = st.n
        st.mu = st.sum1 / n
        if n >= 2:
            var_num = st.sum2 - (st.sum1 * st.sum1) / n
            denom = (n - 1) if cfg.use_unbiased_std else n
            var_val = max(var_num / denom, 0.0)
            st.sigma = var_val ** 0.5
        else:
            st.sigma = 0.0

        denom_mu = st.mu if abs(st.mu) > cfg.eps else cfg.eps
        st.cv = st.sigma / denom_mu

        alert = False
        reason: Optional[str] = None

        if n < cfg.n_min_cv:
            if cfg.rms_threshold is not None and x > cfg.rms_threshold and not cfg.warmup_ignore_alerts:
                alert, reason = True, "rms"
        else:
            if cfg.cv_threshold is not None and st.cv >= cfg.cv_threshold:
                if not (cfg.warmup_ignore_alerts and n < cfg.n_min_cv):
                    alert, reason = True, "cv"

        st.idx += 1
        return self._result(alert=alert, reason=reason)

    def current_state(self) -> CVOnlineState:
        """Return a reference to the current monitor state without modification.

        Returns:
            CVOnlineState: The live :attr:`state` object; modifying it
            directly will affect subsequent :meth:`update` calls.
        """
        return self.state

    def _result(self, alert: bool, reason: Optional[str]) -> Dict[str, Any]:
        """Package the current state into the standard result dictionary.

        This is an internal helper; callers should use :meth:`update`.

        Args:
            alert (bool): Whether an alert condition was triggered on this
                step.
            reason (Optional[str]): ``"rms"`` or ``"cv"`` when *alert* is
                ``True``, otherwise ``None``.

        Returns:
            Dict[str, Any]: Dictionary with keys ``n, mu, sigma, cv, alert,
            reason, idx, time``.  The ``"time"`` key holds the computed
            timestamp [s] when ``dt_rms`` is configured, otherwise ``None``.
        """
        st = self.state
        cfg = self.config
        time_val: Optional[float] = None
        if cfg.dt_rms is not None:
            time_val = cfg.start_time + (st.idx - 1) * cfg.dt_rms

        return {
            "n": st.n,
            "mu": st.mu,
            "sigma": st.sigma,
            "cv": st.cv,
            "alert": alert,
            "reason": reason,
            "idx": st.idx - 1,
            "time": time_val,
        }


class CVStableRegionDetector:
    """Compute an adaptive CV threshold from a user-specified stable region.

    Mirrors the logic of ``ThreeSigmaWithLilliefors`` used by the SSQ
    indicator, but operates on the **CV series** rather than the first
    singular value.  The threshold is:

    * ``mu + z * sigma``  if the stable-region CV values are normally
      distributed (Lilliefors test, significance ``alpha``).
    * ``median + z * 1.4826 * MAD``  otherwise (when ``fallback_mad=True``).

    Only the **upper** limit matters because CV is always non-negative and
    chatter is declared when CV *rises*.

    Priority for choosing the stable region (highest first):

    1. ``idx_stable`` argument passed to :meth:`detect`
    2. ``stable_index`` constructor parameter
    3. ``stable_time``  constructor parameter (requires ``t`` in :meth:`detect`)
    4. ``frac_stable`` fraction of the first *n* frames

    Parameters
    ----------
    frac_stable : float
        Fallback fraction (0 < frac_stable <= 1) when no explicit range is
        given.
    z : float, optional
        Multiplier for sigma / MAD.  Defaults to ``3.0`` (three-sigma rule).
    alpha : float, optional
        Significance level for the Lilliefors normality test.  Default ``0.05``.
    fallback_mad : bool, optional
        Switch to MAD-based threshold when normality is rejected.  Default ``True``.
    stable_time : float or (float, float), optional
        Time range ``[t0, t1]`` (inclusive) defining the stable region.
    stable_index : int or (int, int), optional
        Index range ``[i0, i1]`` (inclusive) defining the stable region.
    """

    def __init__(
        self,
        frac_stable: float,
        z: float = 3.0,
        alpha: float = 0.05,
        fallback_mad: bool = True,
        stable_time: Optional[TimeRange] = None,
        stable_index: Optional[IndexRange] = None,
    ) -> None:
        self.frac_stable  = float(frac_stable)
        self.z            = float(z)
        self.alpha        = float(alpha)
        self.fallback_mad = bool(fallback_mad)
        self.stable_time  = stable_time
        self.stable_index = stable_index

        if stable_time is not None and stable_index is not None:
            self.stable_time = None  # stable_index wins

    def _build_idx_from_ranges(
        self,
        n: int,
        *,
        t: Optional[np.ndarray],
        stable_index: Optional[IndexRange],
        stable_time: Optional[TimeRange],
    ) -> Optional[np.ndarray]:
        if stable_index is not None:
            if isinstance(stable_index, int):
                i0, i1 = 0, int(stable_index)
            else:
                i0, i1 = int(stable_index[0]), int(stable_index[1])
            i0 = max(0, i0)
            i1 = min(n - 1, i1)
            if i1 < i0:
                raise ValueError("stable_index produces empty range")
            return np.arange(i0, i1 + 1, dtype=int)

        if stable_time is not None:
            if t is None:
                raise ValueError("stable_time requires the time vector t")
            tt = np.asarray(t, dtype=float)
            if isinstance(stable_time, (float, int)):
                t0, t1 = float(np.min(tt)), float(stable_time)
            else:
                t0, t1 = float(stable_time[0]), float(stable_time[1])
            if t1 < t0:
                raise ValueError("stable_time invalid: t1 < t0")
            idx = np.nonzero((tt >= t0) & (tt <= t1))[0]
            if idx.size == 0:
                raise ValueError("stable_time produces an empty index range")
            return idx.astype(int)

        return None

    def detect(
        self,
        cv_series: np.ndarray,
        t: Optional[np.ndarray] = None,
        idx_stable: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Compute adaptive threshold and chatter mask for a CV series.

        Parameters
        ----------
        cv_series : np.ndarray
            1-D array of CV values (output of ``CVOnlineMonitor``).
        t : np.ndarray, optional
            Time vector aligned with *cv_series*.  Required when
            ``stable_time`` is used.
        idx_stable : sequence of int, optional
            Explicit stable-region indices (highest priority).

        Returns
        -------
        dict with keys:

        * ``mask``            — int array (1 = chatter, 0 = normal)
        * ``threshold``       — computed upper threshold
        * ``mu``              — mean / median of stable CV
        * ``sigma``           — sigma / robust-sigma of stable CV
        * ``normal_ok``       — True if Lilliefors test passed
        * ``p_value``         — Lilliefors p-value
        * ``metodo_umbral``   — ``"sigma"`` or ``"MAD"``
        * ``idx_estable_usados`` — list of indices used for estimation
        """
        cv = np.asarray(cv_series, dtype=float)
        if cv.ndim != 1:
            raise ValueError("cv_series must be 1-D")
        n = cv.size

        # ── resolve stable region ───────────────────────────────────────────
        if idx_stable is not None:
            idx_est = np.asarray(list(idx_stable), dtype=int)
        else:
            built = self._build_idx_from_ranges(
                n, t=t,
                stable_index=self.stable_index,
                stable_time=self.stable_time,
            )
            if built is not None:
                idx_est = built
            else:
                m = max(1, int(self.frac_stable * n))
                idx_est = np.arange(0, m, dtype=int)

        idx_est = idx_est[(idx_est >= 0) & (idx_est < n)]
        if idx_est.size == 0:
            raise ValueError("stable region is empty after index validation")

        d_est = cv[idx_est]

        mu    = float(np.mean(d_est))
        sigma = float(np.std(d_est, ddof=1)) if d_est.size > 1 else 0.0

        # ── normality test ─────────────────────────────────────────────────
        try:
            _, p_value = lilliefors(d_est, dist="norm")
            normal_ok  = bool(p_value >= self.alpha)
        except Exception:
            p_value   = 0.0
            normal_ok = False

        z = self.z
        if sigma == 0.0:
            eps = 1e-12 if mu == 0 else 1e-6 * abs(mu)
            threshold = mu + z * eps
        else:
            threshold = mu + z * sigma

        metodo = "sigma"
        if self.fallback_mad and not normal_ok:
            med       = float(np.median(d_est))
            mad       = float(np.median(np.abs(d_est - med)))
            sigma_rob = 1.4826 * mad
            if sigma_rob == 0.0:
                eps       = 1e-12 if med == 0 else 1e-6 * abs(med)
                threshold = med + z * eps
            else:
                threshold = med + z * sigma_rob
            mu, sigma = med, sigma_rob
            metodo    = "MAD"

        mask = (cv > threshold).astype(int)
        return {
            "mask":               mask,
            "threshold":          threshold,
            "mu":                 mu,
            "sigma":              sigma,
            "normal_ok":          normal_ok,
            "p_value":            float(p_value),
            "metodo_umbral":      metodo,
            "idx_estable_usados": idx_est.tolist(),
        }
