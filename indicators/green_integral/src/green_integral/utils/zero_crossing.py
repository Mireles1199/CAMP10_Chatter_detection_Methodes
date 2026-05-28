"""Zero-crossing detection hierarchy for the green_integral indicator."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert, butter, filtfilt, correlate


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ZeroCrossing(ABC):
    """Abstract base for zero-crossing detectors."""

    def __init__(self, y_values: np.ndarray, x_values: np.ndarray) -> None:
        self._y_values = np.asarray(y_values, dtype=float)
        self._x_values = np.asarray(x_values, dtype=float)
        self._N = len(y_values)

    @property
    def y_values(self) -> np.ndarray:
        return self._y_values

    @property
    def x_values(self) -> np.ndarray:
        return self._x_values

    @abstractmethod
    def calculate_zero_crossings(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate zero crossings and return (crossings, crossings_id)."""


class ZeroCrossingStrategy(ABC):
    """Pre-processing strategy for :class:`ZeroCrossing_Hilbert`.

    Returns
    -------
    sig_for_hilbert : ndarray — signal fed into the Hilbert transform
    best_imf        : ndarray | None
    imfs            : ndarray | None — shape (n_imfs, N)
    idx_best        : int | None
    """

    @abstractmethod
    def prepare_signal(
        self,
        y_f: np.ndarray,
        t: np.ndarray,
        f0_estimada: Optional[float] = None,
        trim_frac: float = 0.1,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[int]]:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------

class Simple_ZeroCrossing(ZeroCrossing):
    """Sign-change zero-crossing detector (no Hilbert transform)."""

    def calculate_zero_crossings(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(crossings, crossings_id)``.

        crossings    : ndarray (N, 2)  — columns: [x_cruce, 0]
        crossings_id : ndarray (N, 3)  — columns: [x_cruce, 0, idx_segment]
        """
        x = self._x_values
        y = self._y_values

        if x.size < 2 or y.size < 2:
            crossings = np.empty((0, 2), dtype=float)
            crossings_id = np.empty((0, 3), dtype=object)
            return crossings, crossings_id

        # 1) real sign-change crossings (both sides non-zero)
        sign_change = y[:-1] * y[1:] < 0
        idx_sc = np.where(sign_change)[0]

        x0, x1 = x[idx_sc], x[idx_sc + 1]
        y0, y1 = y[idx_sc], y[idx_sc + 1]
        den = y1 - y0
        valid = den != 0
        zero_x_interp = x0[valid] - y0[valid] * (x1[valid] - x0[valid]) / den[valid]
        zero_idx_interp = idx_sc[valid]

        # 2) exact zeros: y[i] == 0 and y[i+1] != 0
        zero_exact_mask = (y[:-1] == 0) & (y[1:] != 0)
        idx_ze = np.where(zero_exact_mask)[0]
        zero_x_exact = x[idx_ze]

        all_zero_x = np.concatenate([zero_x_interp, zero_x_exact])
        all_zero_idx = np.concatenate([zero_idx_interp, idx_ze])

        order = np.argsort(all_zero_x)
        all_zero_x = all_zero_x[order]
        all_zero_idx = all_zero_idx[order]

        crossings = np.column_stack([all_zero_x, np.zeros_like(all_zero_x)])
        crossings_id = np.empty((len(all_zero_x), 3), dtype=object)
        crossings_id[:, :2] = crossings
        crossings_id[:, 2] = all_zero_idx

        return crossings, crossings_id


class HilbertDirectStrategy(ZeroCrossingStrategy):
    """No pre-processing. Passes the signal directly to the Hilbert transform."""

    def prepare_signal(
        self,
        y_f: np.ndarray,
        t: np.ndarray,
        f0_estimada: Optional[float] = None,
        trim_frac: float = 0.1,
    ) -> Tuple[np.ndarray, None, None, None]:
        return y_f, None, None, None

    @property
    def name(self) -> str:
        return "Hilbert"


class ZeroCrossing_Hilbert(ZeroCrossing):
    """Hilbert-transform zero-crossing detector.

    Uses phase unwrapping to identify cycle boundaries.  The pre-processing
    strategy (EMD, direct Hilbert, …) is injected via *strategy*.
    """

    def __init__(
        self,
        y_values: np.ndarray,
        x_values: np.ndarray,
        strategy: Optional[ZeroCrossingStrategy] = None,
        debug_manager=None,
    ) -> None:
        super().__init__(y_values, x_values)
        from .debug import DebugManager  # lazy to avoid circular import

        self._strategy: ZeroCrossingStrategy = (
            strategy if strategy is not None else HilbertDirectStrategy()
        )
        self._dbg = debug_manager if debug_manager is not None else DebugManager(0)

        self.crossing: Optional[np.ndarray] = None
        self.crossings_id: Optional[np.ndarray] = None
        self.analytic_signal: Optional[np.ndarray] = None
        self.phase: Optional[np.ndarray] = None
        self.f0_estimada: Optional[float] = None
        self.raw_phase: Optional[np.ndarray] = None
        self.y_f: Optional[np.ndarray] = None
        self.best_imf: Optional[np.ndarray] = None
        self.imfs: Optional[np.ndarray] = None
        self.idx_best: Optional[int] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _estimate_f0_autocorr(self, max_lag_s: Optional[float] = None) -> float:
        """Estimate fundamental frequency via autocorrelation."""
        t = self._x_values
        y = self._y_values
        dt = float(np.mean(np.diff(t)))
        y0 = y - np.mean(y)
        corr = correlate(y0, y0, mode="full")[len(y0) - 1:]
        lags = np.arange(len(corr)) * dt
        mask = (lags > 0) if max_lag_s is None else (lags > 0) & (lags <= max_lag_s)
        idx_peak = int(np.argmax(corr[mask]))
        T0 = lags[mask][idx_peak] * dt
        return 1.0 / T0

    @staticmethod
    def _smooth_phase(phi: np.ndarray, size: int = 11) -> np.ndarray:
        return uniform_filter1d(phi, size=size)

    # ------------------------------------------------------------------
    # Main method
    # ------------------------------------------------------------------
    def calculate_zero_crossings(
        self,
        f0_estimada: Optional[float] = None,
        min_per: float = 0.0,
        max_lag_s: Optional[float] = None,
        phase_smooth_win: int = 11,
        trim_frac: float = 0.1,
        PI: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect zero-crossings via Hilbert-phase 2π crossings.

        Returns
        -------
        crossing     : ndarray (N, 2)
        crossings_id : ndarray (N, 3)
        keep_s       : ndarray (N,) — sign array
        """
        t = self._x_values
        y = self._y_values

        # 1) f0 estimate
        self.f0_estimada = (
            f0_estimada
            if f0_estimada is not None
            else self._estimate_f0_autocorr(max_lag_s=max_lag_s)
        )

        # 2) pre-processing strategy
        self.y_f = y
        sig_for_hilbert, self.best_imf, self.imfs, self.idx_best = (
            self._strategy.prepare_signal(
                self.y_f, t, f0_estimada=self.f0_estimada, trim_frac=trim_frac
            )
        )
        if self.best_imf is None:
            self.best_imf = sig_for_hilbert

        # 3) analytic phase
        self.raw_phase = np.angle(hilbert(sig_for_hilbert))
        self.y_f = sig_for_hilbert
        self.phase = self._smooth_phase(np.unwrap(self.raw_phase), phase_smooth_win)
        self.analytic_signal = hilbert(self.y_f)

        # 4) 2π crossings
        cycles = np.floor_divide(self.phase, 2 * np.pi).astype(int)
        idx_pre = np.where(np.diff(cycles) > 0)[0]

        if len(idx_pre) == 0:
            self.crossing = np.empty((0, 2))
            self.crossings_id = np.empty((0, 3), dtype=object)
            return self.crossing, self.crossings_id, np.array([], int)

        idx_post = idx_pre + 1
        t_cross = []
        for i in idx_pre:
            phi_b, phi_a = self.phase[i], self.phase[i + 1]
            phi_target = 2 * np.pi * (cycles[i] + 1)
            w = (phi_target - phi_b) / (phi_a - phi_b)
            t_cross.append(t[i] + w * (t[i + 1] - t[i]))
        t_cross = np.asarray(t_cross)

        # 5) signs from phase derivative
        dphi_dt = np.gradient(self.phase, t)
        signs = np.sign(
            interp1d(t, dphi_dt, fill_value="extrapolate")(t_cross)
        )

        # 6) filter nearby crossings
        min_dt = (1.0 / self.f0_estimada) * min_per
        keep = [0] + [
            k for k in range(1, len(t_cross))
            if t_cross[k] - t_cross[k - 1] > min_dt
        ]
        keep_t = t_cross[keep]
        keep_s = signs[keep]
        keep_idx = idx_post[keep]

        # 6b) insert π crossings
        if PI:
            f_phase_to_time = interp1d(
                self.phase, t, bounds_error=False, fill_value="extrapolate"
            )
            f_phase_to_index = interp1d(
                self.phase,
                np.arange(len(self.phase)),
                bounds_error=False,
                fill_value="extrapolate",
            )
            extended_t = list(keep_t)
            extended_s = list(keep_s)
            extended_idx = list(keep_idx)
            for i in range(len(keep_t)):
                phi_mid = 2 * np.pi * i + np.pi
                t_pi = float(f_phase_to_time(phi_mid))
                if len(t_cross) > 0 and t_cross[0] <= t_pi <= t_cross[-1]:
                    extended_t.append(t_pi)
                    extended_s.append(1)
                    extended_idx.append(int(np.round(float(f_phase_to_index(phi_mid)))))
            extended_t = np.asarray(extended_t)
            extended_s = np.asarray(extended_s)
            extended_idx = np.asarray(extended_idx)
            order = np.argsort(extended_t)
            keep_t = extended_t[order]
            keep_s = extended_s[order]
            keep_idx = extended_idx[order]

        # 7) trim edges
        crossing = np.column_stack((keep_t, np.zeros_like(keep_t)))
        crossings_id = np.column_stack((keep_t, np.zeros_like(keep_t), keep_idx))
        n_c = len(crossing)
        cut = int(trim_frac * n_c)
        if trim_frac > 0 and 2 * cut < n_c:
            crossing = crossing[cut:-cut]
            crossings_id = crossings_id[cut:-cut]
            keep_s = keep_s[cut:-cut]

        self.crossing = crossing
        self.crossings_id = crossings_id
        return crossing, crossings_id, keep_s.astype(int)
