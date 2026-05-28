"""Per-window signal processing for the green_integral indicator."""

from __future__ import annotations

import logging
import statistics
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt

from ..utils.types import GreenIntegralConfig
from ..utils.debug import DebugManager
from ..utils.zero_crossing import Simple_ZeroCrossing, ZeroCrossing_Hilbert, HilbertDirectStrategy
from ..utils.contour_area import Contour_Line_Area
from ..utils.signal_filter import filter_window_signals, moving_average
from .delta_n import compute_delta_n, LOG_CTC

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _process_one_window(
    t_win: np.ndarray,
    q_win: np.ndarray,
    q_o_win: np.ndarray,
    start: float,
    end: float,
    num_window: int,
    t_full: np.ndarray,
    config: GreenIntegralConfig,
    dbg: DebugManager,
) -> tuple:
    """Process a single time window and return raw arrays.

    Returns
    -------
    (t_areas, areas, crossing_0_t, crossing_0_t_id,
     start_w, end_w, velocity_crossing, q_o_filtered,
     centers_x, centers_v, instantaneous_area_time, instantaneous_area)
    """
    if config.data_filtrated:
        q_filtered, q_o_filtered = filter_window_signals(q_win, q_o_win)
        velocity_used_crossing = q_o_filtered
        displacement_used_crossing = q_filtered
    else:
        q_filtered = q_win.copy()
        q_o_filtered = q_o_win.copy()
        velocity_used_crossing = q_o_filtered
        displacement_used_crossing = q_filtered

    # --- zero-crossing detection ---
    if config.hilbert:
        data_used_for_crossing = displacement_used_crossing
        _strategy = HilbertDirectStrategy()
        zc = ZeroCrossing_Hilbert(
            y_values=data_used_for_crossing,
            x_values=t_win,
            strategy=_strategy,
            debug_manager=dbg,
        )
        crossing_0_t, crossing_0_t_id, _signs = zc.calculate_zero_crossings(
            f0_estimada=config.f_modal,
            trim_frac=0.0,
            PI=True,
        )
    else:
        data_used_for_crossing = velocity_used_crossing
        zc_simple = Simple_ZeroCrossing(
            y_values=data_used_for_crossing,
            x_values=t_win,
        )
        crossing_0_t, crossing_0_t_id = zc_simple.calculate_zero_crossings()

    # --- per-cycle areas via Green's theorem ---
    displacement = q_filtered
    velocity = q_o_filtered
    dt = float(t_full[1] - t_full[0])

    contour = Contour_Line_Area(
        displacement=displacement,
        velocity=velocity,
        time=t_win,
        T=config.T_modal,
        dt=dt,
        cycles_cluster_points=config.cycles_cluster_points,
        velocity_used_crossing=data_used_for_crossing,
        crossing_0_t=crossing_0_t_id,
        config=config,
        num_window=num_window,
        debug_manager=dbg,
    )
    contour.analyze_contour_area_interpolate()
    centers_x, centers_v, instantaneous_area_time, instantaneous_area = contour.get_results()
    start_w, end_w = contour.get_window_times()

    return (
        instantaneous_area_time, instantaneous_area,
        crossing_0_t, crossing_0_t_id,
        start_w, end_w,
        velocity_used_crossing, q_o_filtered,
        centers_x, centers_v,
        instantaneous_area_time, instantaneous_area,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_windows_serial(
    t: np.ndarray,
    q: np.ndarray,
    q_o: np.ndarray,
    config: GreenIntegralConfig,
    dbg: DebugManager,
) -> List[Dict[str, Any]]:
    """Slide a window over the signal and compute per-window indicators.

    Parameters
    ----------
    t       : time array.
    q       : displacement array.
    q_o     : velocity array.
    config  : :class:`~green_integral.utils.types.GreenIntegralConfig`.
    dbg     : :class:`~green_integral.utils.debug.DebugManager`.

    Returns
    -------
    List of per-window result dicts (see :data:`~green_integral.utils.types.WindowResult`).
    """
    results: List[Dict[str, Any]] = []
    num_window = 0
    window_insufficient = 0
    T_total_window = config.num_T
    T_window = config.T_modal / 1 * (T_total_window)

    for t0 in np.arange(float(t[0]), float(t[-1]), config.dt):
        start = float(t0)
        end = start + T_window

        if end > float(t[-1]):
            break

        mask = (t >= start) & (t < end)
        t_win = t[mask]
        q_win = q[mask]
        q_o_win = q_o[mask]

        if dbg.is_window_in_debug_range(num_window):
            logger.debug(
                "Debugging window %d | start=%.5f", num_window, start
            )

        try:
            (
                t_areas, areas,
                crossing_0_t, crossing_0_t_id,
                start_w, end_w,
                velocity_crossing, q_o_filtered,
                centers_x, centers_v, _1, _2,
            ) = _process_one_window(
                t_win=t_win, q_win=q_win, q_o_win=q_o_win,
                start=start, end=end, num_window=num_window,
                t_full=t, config=config, dbg=dbg,
            )

            # extend window if not enough cycles
            if config.while_loop_extend:
                while len(t_areas) < config.num_T:
                    window_insufficient += 1
                    end += T_window
                    if end > float(t[-1]):
                        break
                    mask_ext = (t >= start) & (t < end)
                    t_win_ext = t[mask_ext]
                    q_win_ext = q[mask_ext]
                    q_o_win_ext = q_o[mask_ext]
                    (
                        t_areas, areas,
                        crossing_0_t, crossing_0_t_id,
                        start_w, end_w,
                        velocity_crossing, q_o_filtered,
                        centers_x, centers_v, _1, _2,
                    ) = _process_one_window(
                        t_win=t_win_ext, q_win=q_win_ext, q_o_win=q_o_win_ext,
                        start=start, end=end, num_window=num_window,
                        t_full=t, config=config, dbg=dbg,
                    )

            if not config.while_loop_extend and len(t_areas) < config.num_T:
                logger.warning(
                    "Window %d skipped — insufficient cycles: %d < %d",
                    num_window, len(t_areas), config.num_T,
                )
                num_window += 1
                continue

            # --- indicator ---
            A = np.asarray(areas, dtype=float)
            t_n = np.asarray(t_areas, dtype=float)
            ind = compute_delta_n(A, t_n, theil_sen=config.thein_sen)
            t_n = ind["t_n"]
            if config.thein_sen:
                valid_mask = np.asarray(areas, dtype=float) > 0
                A = np.asarray(areas, dtype=float)[valid_mask]
            else:
                A = np.asarray(areas, dtype=float)

            indicadores = {
                "t_n": float(t_n[0]) if len(t_n) > 0 else float("nan"),
                "delta_n": ind["delta_n"],
            }

            # --- window signal slice ---
            w_mask = (t >= start_w) & (t < end_w)
            q_signal_in_window = q[w_mask]
            q_o_signal_in_window = q_o[w_mask]
            time_in_window = t[w_mask]

            cycle_key_agrupation = np.floor(t_n / config.dt).astype(int)
            time_key_agrupation = cycle_key_agrupation * config.dt

            # --- area statistics for the window ---
            n_a = len(A)
            median_area = float(np.median(A)) if n_a > 0 else float("nan")
            positive = A[A > 0]
            gmean_area = (
                float(np.exp(np.mean(np.log(positive)))) if len(positive) > 0 else float("nan")
            )

            t_idx = np.arange(1, n_a + 1, dtype=float)
            t_bar = float(np.mean(t_idx))
            A_bar = float(np.mean(A))
            denom = float(np.sum((t_idx - t_bar) ** 2))
            if denom != 0.0:
                beta = float(np.sum((t_idx - t_bar) * (A - A_bar))) / denom
                alpha = A_bar - beta * t_bar
                t_centro = (n_a + 1) / 2.0
                A_centro = alpha + beta * t_centro
            else:
                A_centro = A_bar

            t_centre = float(np.mean(t_n)) if len(t_n) > 0 else float("nan")

            # --- debug output ---
            if dbg.is_window_in_debug_range(num_window):
                logger.debug(
                    "Window %d | start_w=%.5f end_w=%.5f | cycles=%d",
                    num_window, start_w, end_w, len(A),
                )
                for i_a, a_val in enumerate(A):
                    logger.debug("  Área %d: %.4e", i_a, a_val)
                logger.debug("  Indicador delta_n: %s", indicadores["delta_n"])
                logger.debug(
                    "  Median Area: %.4e | Geometric Mean: %.4e | Center: %.4e",
                    median_area, gmean_area, A_centro,
                )
                if not config.thein_sen and ind["r_n"] is not None:
                    r_n = ind["r_n"]
                    delta_n_arr = -LOG_CTC * r_n
                    for i_r, (r_v, dn_v) in enumerate(zip(r_n, delta_n_arr)):
                        logger.debug("  r_n %d: %.4e | delta_n %d: %.4e", i_r, r_v, i_r, dn_v)
                    logger.debug("  Indicador delta_n mediano: %s", indicadores["delta_n"])

            results.append(
                {
                    "window_number": num_window,
                    "start_time": start_w,
                    "end_time": end_w,
                    "window_duration": end_w - start_w,
                    "num_processed_data": len(t_n),
                    "exp_fit_times": t_n.tolist(),
                    "exp_fit_values": A.tolist(),
                    "window_times": time_in_window.tolist(),
                    "window_q_signal": q_signal_in_window.tolist(),
                    "window_q_o_signal": q_o_signal_in_window.tolist(),
                    "cycle_key_agrupation": cycle_key_agrupation.tolist(),
                    "time_key_agrupation": time_key_agrupation.tolist(),
                    "indicadores": indicadores,
                    "centers_x": centers_x.tolist(),
                    "centers_v": centers_v.tolist(),
                    "median_area": median_area,
                    "geometric_mean_area": gmean_area,
                    "center_area_value": A_centro,
                    "t_centre": t_centre,
                }
            )

            num_window += 1

        except Exception as exc:
            logger.error(
                "Error in window %d (start=%.5f): %s",
                num_window, start, exc,
                exc_info=True,
            )
            num_window += 1
            continue

    logger.info("Total windows processed: %d", num_window)
    logger.info("Insufficient windows (skipped): %d", window_insufficient)

    return results
