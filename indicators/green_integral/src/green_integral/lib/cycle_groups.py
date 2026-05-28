"""Cross-window cycle accumulation (grouping) for the green_integral indicator."""

from __future__ import annotations

import logging
import statistics
from collections import Counter
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def build_cycle_groups(
    windows_results: List[Dict[str, Any]],
) -> Tuple[Dict[int, Any], float]:
    """Group per-window cycle data into a cross-window accumulation dict.

    Iterates over all completed window results — each containing
    ``cycle_key_agrupation``, ``time_key_agrupation``, ``exp_fit_times``,
    ``exp_fit_values``, and ``indicadores['delta_n']`` — and accumulates
    area / time values per cycle key.  Per-cycle statistics (mean time, mean
    area, median area) are computed at the end.

    Parameters
    ----------
    windows_results : list of per-window result dicts produced by
        :func:`~green_integral.lib.window_processor.process_windows_serial`.

    Returns
    -------
    resultado_agrupamiento : dict
        ``cycle_key -> {count, times_window, values_window, area_acumulada,
        time_agrupation, window, promedio_tiempo_window, promedio_area_window,
        mediana_area_window}``
    delta_mediana : float
        Median of the per-window ``delta_n`` indicator values.
    """
    resultado_agrupamiento: Dict[int, Any] = {}
    delta_list: List[float] = []

    for idx, item in enumerate(windows_results):
        cycles = item["cycle_key_agrupation"]
        times = item["time_key_agrupation"]
        values_in_window = item["exp_fit_values"]
        t_in_window = item["exp_fit_times"]
        delta_in_window = item["indicadores"]["delta_n"]

        delta_list.append(delta_in_window)

        for c, t, t_win, v_win in zip(cycles, times, t_in_window, values_in_window):
            if c in resultado_agrupamiento:
                resultado_agrupamiento[c]["count"] += 1
                resultado_agrupamiento[c]["times_window"].append(t_win)
                resultado_agrupamiento[c]["values_window"].append(v_win)
                resultado_agrupamiento[c]["area_acumulada"] += v_win
                if idx not in resultado_agrupamiento[c]["window"]:
                    resultado_agrupamiento[c]["window"].append(idx)
            else:
                resultado_agrupamiento[c] = {
                    "times_window": [t_win],
                    "count": 1,
                    "time_agrupation": t,
                    "values_window": [v_win],
                    "area_acumulada": v_win,
                    "window": [idx],
                }

    # Finalise per-cycle statistics
    for c in resultado_agrupamiento:
        count = resultado_agrupamiento[c]["count"]
        resultado_agrupamiento[c]["promedio_tiempo_window"] = (
            sum(resultado_agrupamiento[c]["times_window"]) / count
        )
        resultado_agrupamiento[c]["promedio_area_window"] = (
            resultado_agrupamiento[c]["area_acumulada"] / count
        )
        resultado_agrupamiento[c]["mediana_area_window"] = statistics.median(
            resultado_agrupamiento[c]["values_window"]
        )

    # Log distribution of cycle repetition counts
    freq = Counter(d["count"] for d in resultado_agrupamiento.values())
    logger.info("Cycle repetition distribution:")
    for reps, n_cycles in sorted(freq.items()):
        logger.info("  %d repetition(s)  →  %d cycle(s)", reps, n_cycles)

    delta_mediana = statistics.median(delta_list) if delta_list else float("nan")
    return resultado_agrupamiento, delta_mediana
