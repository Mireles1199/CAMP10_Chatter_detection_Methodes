# Comentario: monitor en línea de CV sobre secuencia RMS
from __future__ import annotations
from dataclasses import dataclass
from typing import Deque, Dict, Any, Optional
from collections import deque

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
    """
    CVOnlineMonitor: Online Coefficient of Variation (CV) Monitoring System
    A real-time monitor that tracks RMS (Root Mean Square) values in a sliding window,
    computes statistical metrics (mean, standard deviation, coefficient of variation),
    and generates alerts based on configurable thresholds.
    Attributes:
        config (CVOnlineConfig): Configuration object containing monitoring parameters.
        state (CVOnlineState): Current state tracking n, mu, sigma, cv, idx, and sums.
        window (Deque[float]): Fixed-size sliding window of RMS values (max size: n_max).
    Methods:
        __init__(config: CVOnlineConfig) -> None:
            Initialize the monitor with configuration. Validates n_max and n_min_cv >= 1.
            Converts fs_rms (sampling frequency) to dt_rms (time step) if needed.
        reset() -> None:
            Clear the window and reset state to initial values.
        update(rms_value: float) -> Dict[str, Any]:
            Process a new RMS value, update sliding window statistics, and evaluate alerts.
            Returns a result dictionary containing current statistics and alert status.
            Alert logic:
            - If n < n_min_cv: triggers alert only if rms_value exceeds rms_threshold
              (unless warmup_ignore_alerts is True).
            - If n >= n_min_cv: triggers alert if CV exceeds cv_threshold
              (unless warmup_ignore_alerts is True and still in warmup phase).
        current_state() -> CVOnlineState:
            Return the current monitoring state without modification.
        _result(alert: bool, reason: Optional[str]) -> Dict[str, Any]:
            Package and return results as a dictionary containing:
            - n: window size
            - mu: mean of window
            - sigma: standard deviation
            - cv: coefficient of variation
            - alert: boolean alert status
            - reason: "rms" or "cv" (alert trigger reason)
            - idx: sample index
            - time: timestamp (if dt_rms is configured)
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
        """
        Reset the monitor to its initial state.
        Clears the display window and reinitializes the internal state
        to a fresh CVOnlineState instance.
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
        """
        Get the current state of the CV monitor.
        Returns:
            CVOnlineState: The current state of the CV monitor.
        """
        return self.state

    def _result(self, alert: bool, reason: Optional[str]) -> Dict[str, Any]:
        """
        Generate a result dictionary containing statistical measurements and alert information.
        Args:
            alert (bool): Flag indicating whether an alert condition has been triggered.
            reason (Optional[str]): Description of the reason for the alert, or None if no alert.
        Returns:
            Dict[str, Any]: A dictionary containing:
                - n (int): Sample count from state.
                - mu (float): Mean value from state.
                - sigma (float): Standard deviation from state.
                - cv (float): Coefficient of variation from state.
                - alert (bool): Alert flag.
                - reason (Optional[str]): Alert reason.
                - idx (int): Current index (adjusted by -1).
                - time (Optional[float]): Timestamp calculated from start_time and current index,
                  or None if dt_rms is not configured.
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
