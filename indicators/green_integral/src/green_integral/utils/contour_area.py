"""Phase-space contour area calculator via Green's line integral."""

from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.optimize import minimize

from .cycle_grouper import CrossingGrouper

logger = logging.getLogger(__name__)


class Contour_Line_Area:
    """Compute per-cycle phase-space areas via the line integral formulation of Green's theorem.

    Parameters
    ----------
    displacement : q(t) array.
    velocity     : dq/dt array.
    time         : t array.
    dt           : time step between samples.
    T            : base modal period.
    cycles_cluster_points : optional max-distance for :class:`CrossingGrouper`.
    crossing_0_t  : (N, 3) crossing array [t_cross, 0, idx_segment].
    velocity_used_crossing : velocity signal aligned with the crossings (for debug plots).
    config       : :class:`~green_integral.utils.types.GreenIntegralConfig`-like object.
    num_window   : index of the parent window (used by debug checks).
    debug_manager : :class:`~green_integral.utils.debug.DebugManager` instance.
    """

    def __init__(
        self,
        displacement: np.ndarray,
        velocity: np.ndarray,
        time: np.ndarray,
        dt: float,
        T: float,
        cycles_cluster_points: Optional[int] = None,
        crossing_0_t: Optional[np.ndarray] = None,
        velocity_used_crossing: Optional[np.ndarray] = None,
        config: Optional[Any] = None,
        num_window: Optional[int] = None,
        debug_manager: Optional[Any] = None,
    ) -> None:
        from .debug import DebugManager

        self.config = config
        self.x = np.asarray(displacement, dtype=float)
        self.v = np.asarray(velocity, dtype=float)
        self.t = np.asarray(time, dtype=float)
        self.dt = float(dt)
        self.T = float(T)
        self.window_duration: List[float] = [0.0, 0.0]
        self.num_window = num_window
        self._current_window: int = num_window if num_window is not None else 0
        self._dbg: DebugManager = (
            debug_manager if debug_manager is not None else DebugManager(0)
        )

        # Results
        self.centers_x: List[float] = []
        self.centers_v: List[float] = []
        self.window_times: List[float] = []
        self.instantaneous_window_area: List[float] = []

        if cycles_cluster_points is not None and crossing_0_t is not None:
            self.crossing_0_t = crossing_0_t

            x_idx = self._convertir_a_estructurado(self.x)
            v_idx = self._convertir_a_estructurado(self.v)

            cluster = CrossingGrouper(
                max_distance=cycles_cluster_points,
                selection_strategy="center",
            )
            clustered = cluster.cluster(crossing_0_t[:, -1].astype(int).tolist())

            mask = np.isin(crossing_0_t[:, -1], clustered)
            self.crossing_0_t = crossing_0_t[mask]
            mask = np.isin(x_idx[:, -1], clustered)
            self.crossing_0_x = x_idx[mask]
            mask = np.isin(v_idx[:, -1], clustered)
            self.crossing_0_v = v_idx[mask]

            if (
                self._dbg.is_window_in_debug_range(num_window)
                and self.crossing_0_t.shape[0] > 0
            ):
                cluster.plot(
                    original_data_t=self.t,
                    original_data_v=self.v,
                    original_data_crossing=crossing_0_t,
                    velocity_used_crossing=velocity_used_crossing,
                    num_window=num_window,
                )

            self.points_cicles_t = self.crossing_0_t.astype(float)[::2, :2]
            self.points_cicles_x = self.crossing_0_x.astype(float)[::2, :2]
            self.points_cicles_v = self.crossing_0_v.astype(float)[::2, :2]
            self.points_semicircles_t = self.crossing_0_t.astype(float)[::1, :2]
            self.points_semicircles_x = self.crossing_0_x.astype(float)[::1, :2]
            self.points_semicircles_v = self.crossing_0_v.astype(float)[::1, :2]

        else:
            # No clustering — build points_cicles directly from raw crossing data.
            # crossing_0_t is the (N, 3) array [time, 0, segment_idx] from Simple_ZeroCrossing.
            if crossing_0_t is not None and len(crossing_0_t) > 0:
                self.crossing_0_t = crossing_0_t

                x_idx = self._convertir_a_estructurado(self.x)
                v_idx = self._convertir_a_estructurado(self.v)

                crossing_indices = crossing_0_t[:, -1].astype(int)
                mask_x = np.isin(x_idx[:, -1].astype(int), crossing_indices)
                self.crossing_0_x = x_idx[mask_x]
                mask_v = np.isin(v_idx[:, -1].astype(int), crossing_indices)
                self.crossing_0_v = v_idx[mask_v]

                self.points_cicles_t = self.crossing_0_t.astype(float)[::2, :2]
                self.points_cicles_x = self.crossing_0_x.astype(float)[::2, :2]
                self.points_cicles_v = self.crossing_0_v.astype(float)[::2, :2]
                self.points_semicircles_t = self.crossing_0_t.astype(float)[::1, :2]
                self.points_semicircles_x = self.crossing_0_x.astype(float)[::1, :2]
                self.points_semicircles_v = self.crossing_0_v.astype(float)[::1, :2]
            else:
                # No crossing data at all — initialize to empty so analysis returns no cycles.
                empty2 = np.empty((0, 2), dtype=float)
                self.crossing_0_t = np.empty((0, 3), dtype=float)
                self.crossing_0_x = np.empty((0, 2), dtype=object)
                self.crossing_0_v = np.empty((0, 2), dtype=object)
                self.points_cicles_t = empty2
                self.points_cicles_x = empty2.copy()
                self.points_cicles_v = empty2.copy()
                self.points_semicircles_t = empty2.copy()
                self.points_semicircles_x = empty2.copy()
                self.points_semicircles_v = empty2.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _convertir_a_estructurado(arr: np.ndarray) -> np.ndarray:
        """Return (N, 2) object array: [value, original_index]."""
        arr = arr.ravel()
        result = np.empty((arr.shape[0], 2), dtype=object)
        result[:, 0] = arr
        result[:, 1] = np.arange(arr.shape[0])
        return result

    def _cost_function(
        self,
        center: Tuple[float, float],
        x_window: np.ndarray,
        v_window: np.ndarray,
    ) -> float:
        return float(
            np.sum((x_window - center[0]) ** 2 + (v_window - center[1]) ** 2)
        )

    @staticmethod
    def _compute_contour_area(x: np.ndarray, v: np.ndarray) -> float:
        """Shoelace formula (Green's theorem)."""
        return 0.5 * float(
            np.abs(np.dot(x, np.roll(v, -1)) - np.dot(v, np.roll(x, -1)))
        )

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------
    def analyze_contour_area_interpolate(self) -> None:
        """Compute per-cycle areas using zero-crossing boundaries."""

        max_iterations = self.config.num_T

        x_window_total: List[np.ndarray] = []
        v_window_total: List[np.ndarray] = []

        for i in range(min(max_iterations, self.points_cicles_t.shape[0] - 1)):
            if i == 0:
                self.window_duration[0] = float(self.points_cicles_t[0, 0])
                self.window_duration[1] = float(self.points_cicles_t[1, 0])
            else:
                self.window_duration[1] = float(self.points_cicles_t[i + 1, 0])

            mask = (self.t >= self.points_cicles_t[i, 0]) & (
                self.t <= self.points_cicles_t[i + 1, 0]
            )
            t_window = self.t[mask]
            x_window = self.x[mask]
            v_window = self.v[mask]

            # append and prepend crossing boundary points
            t_window = np.append(t_window, self.points_cicles_t[i + 1, 0])
            x_window = np.append(x_window, self.points_cicles_x[i + 1, 0])
            v_window = np.append(v_window, self.points_cicles_t[i + 1, 1])

            t_window = np.insert(t_window, 0, self.points_cicles_t[i, 0])
            x_window = np.insert(x_window, 0, self.points_cicles_x[i, 0])
            v_window = np.insert(v_window, 0, self.points_cicles_t[i, 1])

            v_window_pivot = v_window.copy()

            # close the loop
            x_window = np.append(x_window, x_window[0])
            v_window = np.append(v_window, v_window[0])

            x_window_total.append(x_window)
            v_window_total.append(v_window)

            # oscillation centre via Nelder-Mead
            initial_center = [float(np.mean(x_window)), float(np.mean(v_window))]
            result = minimize(
                self._cost_function,
                initial_center,
                args=(x_window, v_window),
                method="Nelder-Mead",
            )
            x_opt = float(result.x[0])
            v_opt = float(result.x[1])

            if self._dbg.is_window_in_debug_range(self.num_window):
                logger.debug(
                    "Window %d | Cycle %d | centre=(%.3e, %.3e)",
                    self.num_window, i, x_opt, v_opt,
                )

            inst_area = self._compute_contour_area(x_window, v_window)
            self.instantaneous_window_area.append(inst_area)
            self.centers_x.append(x_opt)
            self.centers_v.append(v_opt)
            self.window_times.append(float(t_window[0]))

            if self._dbg.is_window_in_debug_range(self.num_window):
                logger.debug(
                    "  Área ciclo %d : %.4e", i, inst_area
                )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_results(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (centers_x, centers_v, window_times, instantaneous_window_area)."""
        return (
            np.asarray(self.centers_x),
            np.asarray(self.centers_v),
            np.asarray(self.window_times),
            np.asarray(self.instantaneous_window_area),
        )

    def get_window_times(self) -> Tuple[float, float]:
        return self.window_duration[0], self.window_duration[1]
