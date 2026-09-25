"""Top-level dispatcher for the green_integral indicator."""

from __future__ import annotations

import logging
import statistics
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..utils.types import SignalData, GreenIntegralConfig, GreenIntegralResult
from ..utils.debug import DebugManager
from .window_processor import process_windows_serial
from .cycle_groups import build_cycle_groups
from .delta_n import LOG_CTC

logger = logging.getLogger(__name__)

# Level constant (registered in __init__.py)
INFO_PLUS_LEVEL = 15

# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: Dict[str, Any] = {
    "num_T": 6,
    "dt": 1e-2,
    "data_filtrated": True,
    "hilbert": False,
    "while_loop_extend": False,
    "cycles_cluster_points": None,
    "thein_sen": False,
    "use_area_threshold": False,
    "training_intervals": None,
    "frac_stable": 0.30,
    "stable_time": None,
    "z_sigma": 3.0,
    "reference_signal": None,
    "debug_level": 0,
    "debug_window_range": (0, None),
    "save_figures_windows": False,
    "work_space": None,
}

INDICATOR_CONFIG: Dict[str, Any] = {
    "func": "Default",
    "params": _DEFAULT_PARAMS,
}


def _resolve_config(
    f_modal: float,
    params: Dict[str, Any],
) -> GreenIntegralConfig:
    """Merge *params* on top of defaults and return a :class:`GreenIntegralConfig`."""
    merged = {**_DEFAULT_PARAMS, **params}
    return GreenIntegralConfig(f_modal=f_modal, **{
        k: merged[k] for k in merged if k != "f_modal"
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_stable_mask(
    t_wins: np.ndarray,
    training_intervals: Optional[List[Tuple[float, float, str]]],
    stable_time: Optional[Tuple[float, float]],
    frac_stable: float,
) -> np.ndarray:
    """Boolean mask of windows belonging to the stable training region."""
    if training_intervals is not None:
        mask = np.zeros(len(t_wins), dtype=bool)
        for t0, t1, label in training_intervals:
            if str(label).startswith("stable"):
                mask |= (t_wins >= t0) & (t_wins <= t1)
    elif stable_time is not None:
        mask = (t_wins >= stable_time[0]) & (t_wins <= stable_time[1])
    else:
        n_stable = max(3, int(len(t_wins) * frac_stable))
        mask = np.zeros(len(t_wins), dtype=bool)
        mask[:n_stable] = True
    return mask


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _green_integral_pipeline(
    signal: SignalData,
    config: GreenIntegralConfig,
) -> GreenIntegralResult:
    """Core execution pipeline."""
    dbg = DebugManager(
        debug_level=config.debug_level,
        window_range=config.debug_window_range,
        save_figures=config.save_figures_windows,
    )

    logger.info("=" * 60)
    logger.info("Green Integral Method  |  signal: %s", signal.name)
    logger.info("  f_modal   = %.2f Hz  |  T_modal = %.4e s", config.f_modal, config.T_modal)
    logger.info("  num_T     = %d", config.num_T)
    logger.info("  dt        = %.4e s", config.dt)
    logger.info("  filtered  = %s  |  hilbert = %s", config.data_filtrated, config.hilbert)
    logger.info("  theil_sen = %s", config.thein_sen)
    logger.info("=" * 60)

    t = np.asarray(signal.t, dtype=float)
    q = np.asarray(signal.displacement, dtype=float)
    q_o = np.asarray(signal.velocity, dtype=float)

    windows_results = process_windows_serial(t, q, q_o, config, dbg)

    logger.log(
        INFO_PLUS_LEVEL,
        "Delta_n data_window len: %d",
        len(windows_results),
    )

    agrupamiento, delta_mediana = build_cycle_groups(windows_results)

    training_source = "external_reference" if config.reference_signal is not None else "internal"

    global_data = {
        "q_signal": q.tolist(),
        "q_o_signal": q_o.tolist(),
        "t": t.tolist(),
        "type_signal": "Area",
        "type_method": "GreenIntegral",
        "use_area_threshold": bool(config.use_area_threshold),
        "training_source": training_source,
    }

    # ---- mu +- 3*sigma area threshold (optional) --------------------------
    t_d_detected: Optional[float] = None
    # Only compute μ±zσ area threshold when the user explicitly enabled
    # `use_area_threshold` AND provided a training source (`reference_signal`
    # or `training_intervals`).  If neither is given we skip the branch to
    # avoid inferring a training region automatically.
    has_training_source = config.reference_signal is not None or config.training_intervals is not None
    if config.use_area_threshold and has_training_source and len(windows_results) >= 3:
        raw_areas = np.array(
            [dw.get("center_area_value") or dw.get("median_area") or np.nan
             for dw in windows_results],
            dtype=float,
        )
        t_wins_gi = np.array(
            [dw["indicadores"]["t_n"] for dw in windows_results],
            dtype=float,
        )
        valid_mask = np.isfinite(raw_areas) & (raw_areas > 0)

        if config.reference_signal is not None:
            if config.training_intervals is not None or config.stable_time is not None:
                logger.warning(
                    "Both 'reference_signal' and internal training selectors "
                    "(training_intervals/stable_time) were provided; "
                    "reference_signal takes priority."
                )
            # Run the same windowing pipeline over the external reference
            # signal and use ALL of its windows as the training population.
            ref_config = replace(
                config, reference_signal=None, use_area_threshold=False, debug_level=0,
            )
            ref_result = _green_integral_pipeline(config.reference_signal, ref_config)
            raw_areas_ref = np.array(
                [dw.get("center_area_value") or dw.get("median_area") or np.nan
                 for dw in ref_result.data_window],
                dtype=float,
            )
            t_wins_gi_ref = np.array(
                [dw["indicadores"]["t_n"] for dw in ref_result.data_window],
                dtype=float,
            )
            valid_ref = np.isfinite(raw_areas_ref) & (raw_areas_ref > 0)
            train_areas = raw_areas_ref[valid_ref]
            train_t_wins = t_wins_gi_ref[valid_ref]
            det_mask = valid_mask
        else:
            stab = _select_stable_mask(
                t_wins_gi, config.training_intervals,
                config.stable_time, config.frac_stable,
            )
            train_areas = raw_areas[stab & valid_mask]
            train_t_wins = t_wins_gi[stab & valid_mask]
            det_mask = ~stab & valid_mask

        if train_areas.size >= 3:
            mu = float(np.mean(train_areas))
            sigma_v = float(np.std(train_areas, ddof=1))
            upper = mu + config.z_sigma * sigma_v
            lower = max(0.0, mu - config.z_sigma * sigma_v)
            global_data["area_mu_3sigma"] = {
                "mu": mu, "sigma": sigma_v,
                "upper": upper, "lower": lower, "z": config.z_sigma,
            }
            # Exact population that trained the threshold above — used by
            # plots.plot_training_distribution() so the histogram/curve always
            # matches training_source, instead of being re-derived (possibly
            # against the wrong signal) from training_intervals at plot time.
            global_data["training_areas"] = train_areas
            global_data["training_t_wins"] = train_t_wins
            det_idx = np.where(det_mask & (raw_areas > upper))[0]
            if det_idx.size > 0:
                t_d_detected = float(t_wins_gi[det_idx[0]])
            logger.info(
                "Area threshold (%s): mu=%.4g, sigma=%.4g, upper=%.4g | t_d=%s",
                training_source, mu, sigma_v, upper, t_d_detected,
            )
        else:
            logger.warning(
                "Area threshold: not enough training windows (%d < 3), skipped.",
                train_areas.size,
            )

    logger.info("-" * 60)
    logger.info("Mediana delta_n = %.4f", delta_mediana)
    logger.info(
        "Interpretation: %s",
        "UNSTABLE (chatter)" if delta_mediana < 0 else "STABLE",
    )
    logger.info("-" * 60)

    return GreenIntegralResult(
        data_window=windows_results,
        agrupamiento=agrupamiento,
        Mediana_delta_n=delta_mediana,
        global_data=global_data,
        Name=signal.name,
        t_d=t_d_detected,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_green_integral(
    signal: SignalData,
    config: Union[Dict[str, Any], GreenIntegralConfig],
) -> GreenIntegralResult:
    """Run the Green Integral chatter detection indicator.

    Parameters
    ----------
    signal : :class:`~green_integral.utils.types.SignalData`
        Input signal container with time, displacement, and velocity arrays.
    config : dict or :class:`~green_integral.utils.types.GreenIntegralConfig`
        Indicator configuration.  When a dict is provided it must contain
        ``"func"`` (e.g. ``"Default"``) and ``"params"`` (sub-dict).  The
        ``"params"`` dict must include ``"f_modal"`` (Hz).

        Example::

            config = {
                "func": "Default",
                "params": {
                    "f_modal": 150.0,
                    "num_T": 6,
                    "dt": 0.005,
                    "data_filtrated": True,
                }
            }

    Returns
    -------
    :class:`~green_integral.utils.types.GreenIntegralResult`
    """
    if isinstance(config, GreenIntegralConfig):
        cfg = config
    elif isinstance(config, dict):
        params = config.get("params", {})
        f_modal = params.pop("f_modal")  # required
        cfg = _resolve_config(f_modal=f_modal, params=params)
    else:
        raise TypeError(
            f"config must be a dict or GreenIntegralConfig, got {type(config)}"
        )

    return _green_integral_pipeline(signal, cfg)
