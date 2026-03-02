"""
pareto_stage1.py
================

Stage 1 multi-objective Pareto optimisation for chatter indicator window parameters.

Objectives to **minimise** simultaneously:
    - window_cost
    - delay
    - false_alarm_rate
    - flipping
    - miss
    - stable_I_var

Usage example
-------------
>>> from pareto_stage1 import run_pareto_stage1
>>> from MaxEnt_SPRT.src.MaxEnt_SPRT.lib.runner import run_MaxEnt_SPRT
>>>
>>> SEARCH_SPACE = {
...     "N_seg":         [5, 8, 10, 12, 15, 20],
...     "t_stable_total":[2, 3, 4, 5, 6],
...     "alpha":         [0.01, 0.05, 0.1],
... }
>>> BASE_CONFIG = {
...     "id":     "MaxEnt_SPRT",
...     "func":   "Default",
...     "params": { ... },          # your working baseline params
... }
>>> df_all, df_pareto = run_pareto_stage1(
...     sig=my_signal,
...     run_fn=run_MaxEnt_SPRT,
...     base_config=BASE_CONFIG,
...     search_space_for_indicator=SEARCH_SPACE,
...     t_star=t_chatter_onset,
...     method="random",
...     n=2000,
...     seed=0,
... )
"""

from __future__ import annotations

import copy
import itertools
import json
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import os

import numpy as np
import pandas as pd

from MaxEnt_SPRT import SignalData
from MaxEnt_SPRT import HDF5Reader
from MaxEnt_SPRT import run_maxent_sprt, plots_maxent_sprt

from ssq_chatter import run_sst_svd
from ssq_chatter import plots_sst_svd

from rms_cv import run_rms_cv

logger = logging.getLogger(__name__)

INFO_PLUS_LEVEL = 15
logging.addLevelName(INFO_PLUS_LEVEL, "INFO_PLUS")

def info_plus(self, message, *args, **kwargs):
    if self.isEnabledFor(INFO_PLUS_LEVEL):
        self._log(INFO_PLUS_LEVEL, message, args, **kwargs)

logging.Logger.info_plus = info_plus

logging.basicConfig(
    level=logging.INFO,   # DEBUG para ver todo
    format="%(asctime)s | %(levelname)s | %(message)s"
)




# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
# NOTE: These are the canonical implementations used by this module.
#       If you already have them in another module you may replace these
#       with: from your_module import D_from_times, far_time_fraction


def D_from_times(
    t: np.ndarray,
    t_chatter: np.ndarray,
    tol: Optional[float] = None,
) -> np.ndarray:
    """Build a binary detection signal *D* from a list of chatter event times.

    Parameters
    ----------
    t:
        Time axis of the signal, shape ``(N,)``.
    t_chatter:
        Detection times reported by the indicator (``result.t_d``).
        May be empty.
    tol:
        Matching tolerance in time units.
        Defaults to ``0.51 * median(diff(t))``.

    Returns
    -------
    np.ndarray of int, shape ``(N,)``
        ``1`` at the indices closest to each detection time, ``0`` elsewhere.
    """
    t = np.asarray(t, dtype=float)
    t_chatter = np.asarray(t_chatter, dtype=float)

    if len(t_chatter) == 0:
        return np.zeros(len(t), dtype=int)

    dt_med = float(np.median(np.diff(t)))
    if tol is None:
        tol = 0.51 * dt_med

    D = np.zeros(len(t), dtype=int)

    idx = np.searchsorted(t, t_chatter)
    idx = np.clip(idx, 0, len(t) - 1)

    left = np.clip(idx - 1, 0, len(t) - 1)
    pick_left = np.abs(t[left] - t_chatter) < np.abs(t[idx] - t_chatter)
    idx[pick_left] = left[pick_left]

    ok = np.abs(t[idx] - t_chatter) <= tol
    D[idx[ok]] = 1

    return D


def far_time_fraction(
    t: np.ndarray,
    D: np.ndarray,
    t_E: float,
) -> float:
    """False-alarm rate as fraction of **stable** time where ``D == 1``.

    Uses non-uniform ``dt`` weights so it works for any sampling rate.

    Parameters
    ----------
    t:
        Time axis.
    D:
        Binary detection signal (output of :func:`D_from_times`).
    t_E:
        Chatter onset time. Only ``t < t_E`` (stable region) is considered.

    Returns
    -------
    float
        FAR in [0, 1], or ``nan`` if fewer than 2 pre-event samples exist.
    """
    t = np.asarray(t, dtype=float)
    D = np.asarray(D, dtype=int)

    st = t < t_E
    if st.sum() < 2:
        return float("nan")

    dt = np.diff(t, prepend=t[0])
    dt[0] = t[1] - t[0]

    num = float(np.sum(D[st] * dt[st]))
    den = float(np.sum(dt[st]))
    return num / den


# ---------------------------------------------------------------------------
# Window-cost functions  (one per indicator)
# ---------------------------------------------------------------------------
def window_cost_rms_cv(params, result, sig) -> float:
    """Effective time support (window cost) for RMS–CV, in ms.

    Tw ≈ T_RMS + (n_max - 1) * Δt_RMS
    where Δt_RMS = T_RMS * (1 - overlap)
    """
    N = int(params["samples_per_window"])
    overlap = float(params.get("overlap_pct", 0.0))
    n_max = int(params.get("n_max", 0))

    # if overlap is given in percent (e.g., 50), uncomment:
    # if overlap > 1.0:
    #     overlap /= 100.0

    fs = float(sig.fs)

    T_rms = N / fs
    dt_rms = T_rms * (1.0 - overlap)

    T_w = T_rms + max(n_max - 1, 0) * dt_rms
    return 1000.0 * T_w

def window_cost_maxent(
    params: Dict[str, Any],
    result: Any,
    sig: Any,
) -> float:
    """Window cost for **MaxEnt_SPRT**.

    Uses ``N_seg`` as the primary cost driver (number of segments per
    analysis window).  Returns the raw value so units are comparable across
    parameter sweeps.
    """
    N_seg = float(params.get("N_seg", None))
    rpm = rpm = result.meta.get("rpm")
    f_r = rpm / 60.0
    T_rev = 1.0 / f_r

    return T_rev * N_seg*1000  # ms

def window_cost_sst(
    params: Dict[str, Any],
    result: Any,
    sig: Any,
) -> float:
    w = float(params.get("win_length_ms", 1.0))
    h = float(params.get("hop_ms", 1.0))
    a = float(params.get("Ai_length", 1.0))

    # Ventana efectiva temporal (ms)
    if a <= 1:
        return w
    return w + (a - 1.0) * h



def window_cost_default(
    params: Dict[str, Any],
    result: Any,
    sig: Any,
) -> float:
    """Default window cost when no indicator-specific function is registered.

    Tries keys ``N_seg``, ``window``, ``w`` in that order.
    Falls back to ``1.0`` when none are present.
    """
    print("DEFAULT WINDOW COST FUNCTION CALLED. CHECK IF THIS IS INTENDED.")
    for key in ("N_seg", "window", "w"):
        if key in params:
            return float(params[key])
    return 1.0


# Registry ─ add a new entry here for every indicator that needs a custom cost
WINDOW_COST_FUNCTIONS: Dict[str, Callable[[Dict[str, Any], Any, Any], float]] = {
    "MaxEnt_SPRT": window_cost_maxent,
    "RMS_CV":      window_cost_rms_cv,   # ← add custom functions here
    "SST_SVD":    window_cost_sst,
    # "EMD_HHT":     window_cost_emd,
}


def get_window_cost_fn(
    indicator_id: str,
) -> Callable[[Dict[str, Any], Any, Any], float]:
    """Return the window-cost function registered for *indicator_id*.

    Falls back to :func:`window_cost_default` for unknown indicators.
    """
    return WINDOW_COST_FUNCTIONS.get(indicator_id, window_cost_default)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

#: Objective column names (all to be minimised).
OBJECTIVE_COLS: List[str] = [
    "window_cost",
    "delay",
    "false_alarm_rate",
    "stable_I_var",
    "miss",
]

def _compute_metrics(
    t: np.ndarray,
    t_d: np.ndarray,
    t_star: float,
    params: Dict[str, Any],
    result: Any,
    sig: Any,
    indicator_id: str,
) -> Dict[str, Any]:
    """Compute all five objectives for one indicator evaluation.

    Parameters
    ----------
    t:
        Time vector from ``result.t``.
    t_d:
        Detection times from ``result.t_d``.
    t_star:
        Ground-truth chatter onset time.
    params:
        The parameter combination used in this run.
    result:
        Full ``IndicatorResult`` object (passed to window-cost function).
    sig:
        ``SignalData`` object passed directly to the window-cost function.
    indicator_id:
        Indicator name for window-cost dispatch.

    Returns
    -------
    dict
        Keys: ``window_cost``, ``delay``, ``false_alarm_rate``, ``miss`",
        ``t_hat``, ``detections_count``.
    """
    n_det = len(t_d)

    # separar detecciones pre y post onset
    post = t_d[t_d >= t_star]

    if len(post) > 0:
        t_hat = float(np.min(post))
        delay = float(t_hat - t_star)
        miss = 0.0
    else:
        t_hat = float("nan")
        delay = float(t[-1] - t_star)
        miss = 1.0

    # ── false alarm rate ────────────────────────────────────────────────────
    D = D_from_times(t, t_d, tol=None)
    far = far_time_fraction(t, D, t_star)

    # ── flipping (retriggers as instability proxy) ──────────────────────────
    # flipping = float(max(0, n_det - 1))

    t_first = float(np.min(t_d)) if len(t_d) else float("nan")
    early = float(max(0.0, t_star - t_first)) if np.isfinite(t_first) else float("nan")

    # ── window cost ─────────────────────────────────────────────────────────
    wc_fn = get_window_cost_fn(indicator_id)
    wc = wc_fn(params, result, sig)

    # ---- stable indicator variance objective (t < t_star) ----
    stable_I_var = float("nan")
    try:
        t_I = np.asarray(result.t)
        I_t = np.asarray(result.I_t)

        if t_I.shape == I_t.shape:
            m_stable = t_I < t_star
            I_stable = I_t[m_stable]

            if I_stable.size >= 2:
                # Recomendado: log(1+var) para evitar que domine por escala
                stable_I_var = float(np.log1p(np.var(I_stable, ddof=0)))
    except Exception:
        stable_I_var = float("nan")

    metrics = {
        "window_cost":       wc,
        "delay":             delay,
        "false_alarm_rate":  far,
        "stable_I_var":      stable_I_var,
        "miss":              miss,
        "t_hat":             t_hat,
        "detections_count":  n_det,
        "early":            early,
    }

    return metrics


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def generate_candidates(
    search_space: Dict[str, List[Any]],
    method: str = "random",
    n: int = 2000,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Generate parameter-combination candidates from *search_space*.

    Parameters
    ----------
    search_space:
        Mapping ``{param_name: [value1, value2, …]}``.
        Only the keys present here will be varied; all other params keep
        their values from ``base_config["params"]``.
    method:
        ``"grid"``   — full Cartesian product (ignores *n*).
        ``"random"`` — *n* combinations sampled i.i.d. per parameter.
    n:
        Number of random candidates (``method="random"`` only).
    seed:
        NumPy random seed for reproducibility.

    Returns
    -------
    list of dict
        Each dict maps parameter names to concrete values.

    Raises
    ------
    ValueError
        If *method* is not ``"grid"`` or ``"random"``.
    """
    if method not in ("grid", "random"):
        raise ValueError(f"method must be 'grid' or 'random', got '{method!r}'")

    keys = list(search_space.keys())
    value_lists = [list(search_space[k]) for k in keys]

    def pack(keys, combo) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in zip(keys, combo):
            if isinstance(k, tuple):
                # k = ("win_length_ms","hop_ms"), v = (win_ms, hop_ms)
                if not isinstance(v, (tuple, list)) or len(v) != len(k):
                    raise ValueError(f"Composite key {k} expects {len(k)} values, got {v}")
                for kk, vv in zip(k, v):
                    out[kk] = vv
            else:
                out[k] = v
        return out

    if method == "grid":
        candidates = [pack(keys, combo) for combo in itertools.product(*value_lists)]
        logger.info(
            "generate_candidates | grid | %d candidates from %d parameters.",
            len(candidates), len(keys),
        )
        return candidates

    # random (si lo tienes más abajo en tu código, aplica pack() igual)
    import numpy as np
    rng = np.random.default_rng(seed)
    candidates = []
    for _ in range(n):
        combo = [rng.choice(vs) for vs in value_lists]
        candidates.append(pack(keys, combo))
    logger.info(
        "generate_candidates | random | %d candidates from %d parameters.",
        len(candidates), len(keys),
    )
    return candidates


# ---------------------------------------------------------------------------
# Single evaluation
# ---------------------------------------------------------------------------

def evaluate_one(
    sig: Any,
    run_fn: Callable,
    base_config: Dict[str, Any],
    params_override: Dict[str, Any],
    t_star: float,
) -> Optional[Dict[str, Any]]:
    """Evaluate a **single** parameter combination.

    The function deep-copies *base_config*, applies *params_override* on top
    of ``base_config["params"]``, calls *run_fn*, and computes the objective
    metrics.

    Parameters
    ----------
    sig:
        ``SignalData`` object passed directly to *run_fn* unchanged.
    run_fn:
        Callable with signature ``run_fn(sig, config) -> IndicatorResult``.
        Example: ``run_MaxEnt_SPRT``.
    base_config:
        Dict with at least ``{"id": …, "params": {…}}``.
        **Not mutated** — a deep copy is used internally.
    params_override:
        Parameters to override in ``base_config["params"]`` for this run.
    t_star:
        Ground-truth chatter onset time.

    Returns
    -------
    dict or None
        Metrics dict (including ``params_json``), or ``None`` if the
        indicator raised an exception or returned invalid output.
    """
    config = copy.deepcopy(base_config)
    config["params"].update(params_override)

    indicator_id: str = str(config.get("id", "unknown"))

    # ── run indicator ────────────────────────────────────────────────────────
    try:
        result = run_fn(sig, config)

        logger.info_plus(f"\n Result for {indicator_id} with params {params_override}:")
        logger.info_plus(f"time detected: {result.t_d[0]}\n")
        # plots_maxent_sprt(signal=sig, result=result, show_signal=True,
        #     zoom_x=None, zoom_y=None, vlines=None, hlines=None,)
    except Exception as exc:
        logger.info_plus(
            "run_fn raised | indicator=%s | params=%s | %s: %s",
            indicator_id, params_override, type(exc).__name__, exc,
        )
        return None

    if result is None:
        logger.warning(
            "run_fn returned None | indicator=%s | params=%s",
            indicator_id, params_override,
        )
        return None

    # ── extract t ────────────────────────────────────────────────────────────
    t_raw = getattr(result, "t", None)
    if t_raw is None or len(t_raw) == 0:
        logger.warning(
            "result.t missing or empty | indicator=%s | params=%s",
            indicator_id, params_override,
        )
        return None

    t_arr = np.asarray(t_raw, dtype=float)

    # ── extract t_d ──────────────────────────────────────────────────────────
    t_d_raw = getattr(result, "t_d", None)
    t_d_arr = np.asarray([] if t_d_raw is None else t_d_raw, dtype=float)

    # ── compute metrics ───────────────────────────────────────────────────────
    metrics = _compute_metrics(
        t=t_arr,
        t_d=t_d_arr,
        t_star=t_star,
        params=params_override,
        result=result,
        sig=sig,
        indicator_id=indicator_id,
    )

    metrics["params_json"] = json.dumps(params_override, default=str)
    return metrics


# ---------------------------------------------------------------------------
# Pareto dominance filter
# ---------------------------------------------------------------------------

def pareto_filter(
    df: pd.DataFrame,
    cols: List[str],
) -> pd.DataFrame:
    """Return the Pareto-non-dominated subset of *df*.

    **Dominance definition** (all objectives are minimised):
    Solution A dominates B if

    * ``A[obj] <= B[obj]`` for **all** objectives in *cols*, **and**
    * ``A[obj] <  B[obj]`` for **at least one** objective.

    The algorithm is O(N²) which is practical for N ≤ 5 000.

    Parameters
    ----------
    df:
        Results DataFrame (any columns may be present).
    cols:
        List of column names to treat as objectives (must all be numeric).

    Returns
    -------
    pd.DataFrame
        Non-dominated solutions with a fresh integer index.
        All original columns are preserved.
    """
    if df.empty:
        return df.iloc[0:0].copy()

    vals = df[cols].to_numpy(dtype=float)
    n = len(vals)
    dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            # Skip j if we already know it is dominated:
            # by transitivity, if j is dominated by some k, and j would
            # dominate i, then k also dominates i (Pareto is transitive),
            # so i will be caught when we check k vs i.
            if dominated[j]:
                continue
            # Does j dominate i?
            if np.all(vals[j] <= vals[i]) and np.any(vals[j] < vals[i]):
                dominated[i] = True
                break

    return df[~dominated].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_pareto_stage1(
    sig: SignalData,
    run_fn: Callable,
    base_config: Dict[str, Any],
    search_space_for_indicator: Dict[str, List[Any]],
    t_star: float,
    method: str = "random",
    n: int = 2000,
    seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run Stage 1 Pareto multi-objective optimisation.

    Evaluates all candidate parameter combinations produced by
    :func:`generate_candidates`, computes the five objectives for each, and
    returns the full results table together with the Pareto-optimal subset.

    Parameters
    ----------
    sig:
        ``SignalData`` object passed directly to *run_fn* unchanged.
    run_fn:
        ``run_<indicator>(sig, config) -> IndicatorResult``.
        The function is called once per candidate — do **not** modify its
        internal detection logic.
    base_config:
        Base indicator configuration::

            {
                "id":     "MaxEnt_SPRT",
                "func":   "Default",
                "params": { … }          # baseline params; will be overridden
            }

        A deep copy is made for every evaluation; *base_config* is never
        mutated.
    search_space_for_indicator:
        Search space dictionary for **this indicator's** parameters only.
        Keys must match keys inside ``base_config["params"]``::

            {
                "N_seg":          [5, 8, 10, 12, 15, 20],
                "t_stable_total": [2, 3, 4, 5, 6],
                "alpha":          [0.01, 0.05, 0.1],
            }

        Only the listed keys will be varied; all other params stay fixed.
    t_star:
        Ground-truth chatter onset time used to compute *delay* and *FAR*.
    method:
        ``"random"`` (default) or ``"grid"``.
    n:
        Number of random candidates when ``method="random"``.
    seed:
        NumPy random seed for reproducibility.

    Returns
    -------
    df_results : pd.DataFrame
        All evaluated candidates. Columns:

        ``params_json``, ``window_cost``, ``delay``, ``false_alarm_rate`",
        ``flipping``, ``miss``, ``t_hat``, ``detections_count``.

        Rows where any objective is ``NaN`` (e.g. not enough pre-event
        samples for FAR) are kept here but excluded from the Pareto front.

    df_pareto : pd.DataFrame
        Non-dominated solutions from the valid (no-NaN) subset, same columns.

    Notes
    -----
    * Logs progress at DEBUG level every 100 candidates.
    * Returns two empty DataFrames with the correct columns if no evaluation
      succeeds.
    """
    indicator_id: str = str(base_config.get("id", "unknown"))

    logger.info(
        "Pareto Stage 1 | indicator=%s | method=%s | n=%d | seed=%d | t_star=%.6g",
        indicator_id, method, n, seed, t_star,
    )

    candidates = generate_candidates(
        search_space=search_space_for_indicator,
        method=method,
        n=n,
        seed=seed,
    )

    total = len(candidates)
    logger.info("Evaluating %d candidates …", total)

    records: List[Dict[str, Any]] = []
    n_errors = 0

    for idx, params in enumerate(candidates, start=1):
        row = evaluate_one(
            sig=sig,
            run_fn=run_fn,
            base_config=base_config,
            params_override=params,
            t_star=t_star,
        )
        if row is None:
            n_errors += 1
        else:
            records.append(row)

        if idx % 100 == 0:
            logger.debug("  %d / %d evaluated (%d errors so far).", idx, total, n_errors)

    logger.info(
        "\n" + "Evaluation complete | valid=%d | errors=%d | total=%d",
        len(records), n_errors, total,
    )

    # ── assemble DataFrame ───────────────────────────────────────────────────
    col_order = ["params_json"] + OBJECTIVE_COLS + ["early"]  + ["t_hat", "detections_count"]

    if not records:
        logger.error("No valid evaluations. Returning empty DataFrames.")
        empty = pd.DataFrame(columns=col_order)
        return empty, empty.copy()

    df_results = pd.DataFrame(records)[col_order].copy()

    # ── Pareto filter on rows with finite objectives ─────────────────────────
    df_no_nan = df_results.dropna(subset=OBJECTIVE_COLS)
    n_dropped_nan = len(df_results) - len(df_no_nan)

    df_valid = df_no_nan[df_no_nan["early"] <= 0.0]
    n_dropped_early = len(df_no_nan) - len(df_valid)

    if n_dropped_nan > 0:
        logger.warning("%d row(s) dropped due to NaN objective(s).", n_dropped_nan)

    if n_dropped_early > 0:
        logger.warning("%d row(s) dropped due to early > 0.0 restriction.", n_dropped_early)

    df_pareto = pareto_filter(df_valid, OBJECTIVE_COLS)

    logger.info(
        "Pareto front | %d solutions from %d valid evaluations.",
        len(df_pareto), len(df_valid),
    )

    return df_results, df_pareto

def _cut_signal( t,x , time_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a portion of a signal within a specified time range.
    Parameters
    ----------
    t : np.ndarray
        Time array containing timestamp values.
    x : np.ndarray
        Signal array containing corresponding signal values.
    time_range : Tuple[float, float]
        A tuple containing (start_time, end_time) defining the time window
        for extraction. Both boundaries are inclusive.
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - t_cut : np.ndarray
            Time values within the specified range.
        - x_cut : np.ndarray
            Signal values corresponding to the time range.
    Examples
    --------
    >>> t = np.array([0, 1, 2, 3, 4, 5])
    >>> x = np.array([10, 20, 30, 40, 50, 60])
    >>> t_cut, x_cut = _cut_signal(t, x, (1, 4))
    >>> t_cut
    array([1, 2, 3, 4])
    >>> x_cut
    array([20, 30, 40, 50])
    """

    start_time, end_time = time_range
    mask = (t >= start_time) & (t <= end_time)
    return t[mask], x[mask]



# >>> SEARCH_SPACE = {
# ...     "N_seg":         [5, 8, 10, 12, 15, 20],
# ...     "t_stable_total":[2, 3, 4, 5, 6],
# ...     "alpha":         [0.01, 0.05, 0.1],
# ... }
# >>> BASE_CONFIG = {
# ...     "id":     "MaxEnt_SPRT",
# ...     "func":   "Default",
# ...     "params": { ... },          # your working baseline params
# ... }
# >>> df_all, df_pareto = run_pareto_stage1(
# ...     sig=my_signal,
# ...     run_fn=run_MaxEnt_SPRT,
# ...     base_config=BASE_CONFIG,
# ...     search_space_for_indicator=SEARCH_SPACE,
# ...     t_star=t_chatter_onset,
# ...     method="random",
# ...     n=2000,
# ...     seed=0,
# ... )


dir_cono =  r'D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\2DOF_Cono\1DOF_150Hz'
dir_path_use = dir_cono

data_dir = os.path.abspath(os.path.join(dir_path_use, 'out.hdf5' ))
data = HDF5Reader(data_dir)

tool_dyn = data.get_element('tool_dyn/data',)
t = tool_dyn[:,0]
tool_dyn = tool_dyn[:,1]
tool_dyn_vel = data.get_element('tool_dyn_o/data',)[:,1]
force_N = data.get_element('res_R_p/data',)[:,1] #Newtons

t = t
v = tool_dyn_vel
fs = 1.0 / (t[1]-t[0])
curt_range: Tuple[float, float] = (0.05, 15)

t_cut, v_cut = _cut_signal( t, v , curt_range )
_ , x_cut = _cut_signal( t, tool_dyn , curt_range )
_ , force_cut = _cut_signal( t, force_N , curt_range )

BASE_INDICATOR_CONFIG_MAXENT = {
    "id": "MaxEnt_SPRT",                  # internal identifier (optional)
    "func": "Default",                    # indicator wrapper
    "params": {                           # default parameters for this benchmark
        "rpm": 12_000.0,
        "N_seg": 10,
        "ratio_sampling": 100.0,
        "t_stable_total": 5.365770208787228,
        "alpha": 0.05,
        "beta": 0.05,
        "reset_on_H0": True,
        "cut_start_time": 0.05,
        "cut_end_time": 14,
    },
}

BASE_INDICATOR_CONFIG_SST_SVD ={
        "id": "SST_SVD",
        "func": "Default",
        "params": {
            "n_fft_power": 3,
            "win_length_ms": 50.0,
            "hop_ms": 30.0,
            "Ai_length": 4,
            "mode": "causal_inclusive",
            "sigma": 6.0,
            "frac_stable": 0.36052,
            "alpha": 0.05,
            "z": 3.0,
            "fallback_mad": False,
        },
    }


BASE_INDICATOR_CONFIG_RMS_CV = {
    "id": "RMS_CV",
    "func": "Default",
    "params": {
        "n_max": 20,
        "samples_per_window": 400,
        "overlap_pct": 0.0,
        "detrend": False,
        "pad_mode": "none",
        "use_unbiased_std": True,
        "eps": 1e-12,
        "cv_threshold": 1.05,
        "rms_threshold": 0.9,
        "n_min_cv": 2,
        "warmup_ignore_alerts": False,
        "start_time": 0.05,
    },
}


sig = SignalData(
    t_cut=t_cut,
    v_cut=v_cut,
    x_cut = x_cut,
    force_cut = force_cut,
    t_original=t,
    x_original=tool_dyn,
    v_original=v,
    t_analysis=t_cut,
    signal_analysis=v_cut,
    force_original=force_N,
    path=data_dir,
    fs=fs,
    meta={"AP": "5mm-15mm",
            "RPM": 12_000,}
)

f_modal = 150.0 #Hz
T_modal = 1.0 / f_modal #s
Num_cycles_max = 30

# ============ RMS_CV search space ===========
samples_per_window = [math.ceil(n * fs / f_modal) for n in range(1, Num_cycles_max + 1)]

SEARCH_SPACE_RMS_CV = {
     "n_max": list(range(15,35)),  # 15 to 20 inclusive
     "samples_per_window": samples_per_window ,
    #  "overlap_pct": [0.0, 0.25, 0.33, 0.5],
    #  "n_min_cv": list(range(1, 5))
}


# =========== MaxEnt_SPRT search space ===========
rpm = BASE_INDICATOR_CONFIG_MAXENT["params"]["rpm"]
N_seg = [math.ceil(n * T_modal / (60.0 / rpm)) for n in range(1, Num_cycles_max+1)]

t_stable = float(BASE_INDICATOR_CONFIG_MAXENT["params"]["t_stable_total"])
t_chatter =  float(sig.t_analysis[-1] - t_stable)

Num_porcentiles = 20
cut_start_time = np.linspace(0.0, 0.75, Num_porcentiles + 1) * t_stable
cut_end_time = t_stable + np.linspace(0.25, 1.0, Num_porcentiles + 1) * t_chatter



SEARCH_SPACE_MAXENT = {
     "N_seg": N_seg,  # 5 to 20 inclusive
    #  "cut_start_time": cut_start_time,
     "cut_end_time": cut_end_time,
}

# =========== SST_SVD search space with composite keys for win/hop pairs ===========
hop_fracs = [0.25,0.33,  0.5]


# win_length_ms = [n*T_modal*1000 for n in range(1, Num_cycles_max+1)]

win_hop_pairs = [
    (w * T_modal * 1000.0, (w * T_modal * 1000.0) * f)  # (win_ms, hop_ms)
    for w in range(1, Num_cycles_max + 1)
    for f in hop_fracs
]

SEARCH_SPACE_SST_SVD = {
    ("win_length_ms", "hop_ms"): win_hop_pairs,
    "Ai_length": list(range(1, 10)),  # 1 to 10 inclusive
    "sigma": [3.0, 6.0, 9.0],
}
# ===========================================================================




df_all, df_pareto = run_pareto_stage1(
    sig=sig,
    run_fn=run_sst_svd,
    base_config=BASE_INDICATOR_CONFIG_SST_SVD,
    search_space_for_indicator=SEARCH_SPACE_SST_SVD,
    t_star= 5.365770208787228,
    method="grid",
    n=2000,
    seed=0,
 )
# ================== DF all results =======================
if df_all.empty:
    logger.warning("All results DataFrame is empty. No parameters to convert.")
else:
    # Copiamos el dataframe original
    df_all_print = df_all.copy()
    params = df_all_print["params_json"].apply(json.loads)
    df_params = params.apply(pd.Series)

    if "N_seg" in df_params.columns:
        rpm = BASE_INDICATOR_CONFIG_MAXENT["params"]["rpm"]
        f_r = rpm / 60.0
        T_rev = 1.0 / f_r
        df_params["Num_period_modal"] = df_params["N_seg"] * T_rev/ T_modal

    if "win_length_ms" in df_params.columns:
        df_params["Window_period_modal"] = df_params["win_length_ms"] / (T_modal * 1000)
    if "hop_ms" in df_params.columns:
        df_params["hop_period_modal"] = df_params["hop_ms"] / (T_modal * 1000)

    if "samples_per_window" in df_params.columns:
        df_params["Window_period_modal"] = df_params["samples_per_window"] / (fs * T_modal)

    df_all_print_c = pd.concat(
        [df_params, df_all_print.drop(columns=["params_json"])],
        axis=1
    )

# ================ DF Pareto Convert =======================
if df_pareto.empty:
    logger.warning("Pareto front is empty. No parameters to convert.")
else:
    # Copiamos el dataframe original
    df_pareto_print = df_pareto.copy()

    # Convertimos params_json a dict
    params = df_pareto_print["params_json"].apply(json.loads)

    # 1) Separar TODAS las llaves del dict en columnas
    df_params = params.apply(pd.Series)   # o: pd.json_normalize(params)

    # 2) Modificar solo las llaves/columnas específicas (si existen)
    if "win_length_ms" in df_params.columns:
        df_params["win_length_c"] = df_params["win_length_ms"] / (T_modal * 1000)

    if "hop_ms" in df_params.columns:
        df_params["hop_c"] = df_params["hop_ms"] / (T_modal * 1000)

    if "N_seg" in df_params.columns:
        rpm = BASE_INDICATOR_CONFIG_MAXENT["params"]["rpm"]
        f_r = rpm / 60.0
        T_rev = 1.0 / f_r
        df_params["Num_period_modal"] = df_params["N_seg"] * T_rev/ T_modal

    if "samples_per_window" in df_params.columns:
        df_params["Window_period_modal"] = df_params["samples_per_window"] / (fs * T_modal)

    # (opcional) Si no quieres conservar las originales en ms:
    # df_params = df_params.drop(columns=[c for c in ["win_length_ms", "hop_ms"] if c in df_params.columns])

    # 3) Unir con el resto del dataframe (quitando params_json)
    df_pareto_print_c = pd.concat(
        [df_params, df_pareto_print.drop(columns=["params_json"])],
        axis=1
    )
#=========================================================


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.5f}".format)

logger.info("===== TODOS LOS RESULTADOS =====")
if df_all.empty:
    logger.warning("All results DataFrame is empty. No solutions to display.")
else:
    logger.info(f"Total evaluados: {len(df_all_print_c)}")
    logger.info("\n\n" + df_all_print_c.to_string(index=True) +"\n\n")

logger.info("===== FRENTE DE PARETO =====")

if df_pareto.empty:
    logger.warning("Pareto front is empty. No solutions to display.")
else:
    logger.info(f"Soluciones no dominadas: {len(df_pareto_print)}")
    logger.info("\n\n" + df_pareto_print_c.to_string(index=True)+"\n")

logger.info("Finished Pareto Stage 1.")

