"""
sweep/enumerator.py
===================
Enumerate all feasible integer ``(N_win, step)`` combinations for a given
``K_total`` (total number of physical cycles) per indicator.

Feasibility conditions
----------------------
RMS-CV and SST-SVD share the same sliding-window structure:
    - 1 ≤ N_win ≤ K_total - 1
    - 1 ≤ step ≤ N_win
    - (K_total - N_win) % step == 0          (exact coverage)
    - n_accum = (K_total - N_win) // step + 1  (number of accumulated frames)

MaxEnt-SPRT uses N_seg = K_total (segment size fixed by K_total):
    - N_seg = K_total  (fixed — not swept)
    - 1 ≤ step ≤ K_total                      (hop in OPR units between segments)
    - n_accum is not applicable (MaxEnt processes all available segments online)

SweepMode filters
-----------------
    FREE_ALL     — all feasible combinations
    FIX_WIN(N)   — only combos with N_win == N   (kwarg: n_win=N)
    FIX_STEP(s)  — only combos with step == s    (kwarg: step=s)
    FIX_N_ACCUM(n) — only combos with n_accum == n  (kwarg: n_accum=n)

Usage
-----
    from sweep.enumerator import SweepMode, enumerate_feasible

    combos = enumerate_feasible(K_total=8, indicator="rms_cv",
                                sweep_mode=SweepMode.FREE_ALL)
    # Each entry is a dict: {K_total, N_win, step, n_accum, overlap_frac,
    #                        n_combos_valid, ...}
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

__all__ = ["SweepMode", "enumerate_feasible"]

_SLIDING_INDICATORS = {"rms_cv", "sst_svd"}
_MAXENT_INDICATORS  = {"maxent", "maxent_sprt"}


class SweepMode(Enum):
    """Sweep filter mode for :func:`enumerate_feasible`."""

    FREE_ALL    = "FREE_ALL"
    FIX_WIN     = "FIX_WIN"      # kwarg: n_win=<int>
    FIX_STEP    = "FIX_STEP"     # kwarg: step=<int>
    FIX_N_ACCUM = "FIX_N_ACCUM"  # kwarg: n_accum=<int>  (not applicable to MaxEnt)


def enumerate_feasible(
    K_total: int,
    indicator: str,
    sweep_mode: SweepMode = SweepMode.FREE_ALL,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Return a list of all feasible ``(N_win, step)`` combinations for
    the given ``K_total`` and ``indicator``.

    Parameters
    ----------
    K_total : int
        Total number of physical cycles (integer).  Must be ≥ 2 for sliding
        indicators and ≥ 1 for MaxEnt.
    indicator : str
        Indicator identifier.  Recognised values:
        ``"rms_cv"``, ``"sst_svd"``,
        ``"maxent"``, ``"maxent_sprt"``.
    sweep_mode : SweepMode
        Filter to apply (see module docstring).
    **kwargs
        Filter parameters depending on ``sweep_mode``:
        - ``n_win``   (int) — required for ``FIX_WIN``
        - ``step``    (int) — required for ``FIX_STEP``
        - ``n_accum`` (int) — required for ``FIX_N_ACCUM``

    Returns
    -------
    List[Dict[str, Any]]
        Each entry is a dict with keys:
        ``K_total``, ``N_win`` (or None for MaxEnt), ``step``, ``n_accum``
        (or None for MaxEnt), ``overlap_frac`` (or None for MaxEnt),
        ``n_combos_valid``.

    Raises
    ------
    ValueError
        On invalid ``K_total``, unsupported ``indicator``, or missing kwargs.
    """
    ind = indicator.lower()

    if ind in _SLIDING_INDICATORS:
        return _enumerate_sliding(K_total, sweep_mode, **kwargs)
    elif ind in _MAXENT_INDICATORS:
        return _enumerate_maxent(K_total, sweep_mode, **kwargs)
    else:
        raise ValueError(
            f"Unknown indicator {indicator!r}. "
            f"Supported: {sorted(_SLIDING_INDICATORS | _MAXENT_INDICATORS)}"
        )


# ── sliding-window indicators (RMS-CV, SST-SVD) ──────────────────────────────

def _enumerate_sliding(
    K_total: int,
    sweep_mode: SweepMode,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Enumerate feasible (N_win, step) for sliding-window indicators."""
    if K_total < 2:
        raise ValueError(
            f"K_total must be >= 2 for sliding-window indicators, got {K_total}."
        )

    # -- apply FIX_WIN / FIX_STEP / FIX_N_ACCUM ranges ---------------------
    if sweep_mode == SweepMode.FIX_WIN:
        n_win_fixed: Optional[int] = int(kwargs["n_win"])
        n_win_range = range(n_win_fixed, n_win_fixed + 1)
    else:
        n_win_range = range(1, K_total)  # 1 ≤ N_win ≤ K_total - 1

    if sweep_mode == SweepMode.FIX_STEP:
        step_fixed: Optional[int] = int(kwargs["step"])
    else:
        step_fixed = None

    if sweep_mode == SweepMode.FIX_N_ACCUM:
        n_accum_fixed: Optional[int] = int(kwargs["n_accum"])
    else:
        n_accum_fixed = None

    combos: List[Dict[str, Any]] = []

    for N_win in n_win_range:
        remainder = K_total - N_win   # must be divisible by step

        if remainder <= 0:
            # N_win == K_total would give remainder=0, n_accum=1 — excluded
            # (N_win < K_total is enforced by n_win_range upper bound)
            continue

        # iterate over valid step values
        step_lo = step_fixed if step_fixed is not None else 1
        step_hi = step_fixed if step_fixed is not None else N_win

        for step in range(step_lo, step_hi + 1):
            if remainder % step != 0:
                continue

            n_accum = remainder // step + 1

            if n_accum_fixed is not None and n_accum != n_accum_fixed:
                continue

            if n_accum < 1:
                continue

            overlap_frac = 1.0 - step / N_win

            combos.append({
                "K_total":      K_total,
                "N_win":        N_win,
                "step":         step,
                "n_accum":      n_accum,
                "overlap_frac": overlap_frac,
            })

    # attach total count to each entry
    n_valid = len(combos)
    for c in combos:
        c["n_combos_valid"] = n_valid

    return combos


# ── MaxEnt-SPRT ───────────────────────────────────────────────────────────────

def _enumerate_maxent(
    K_total: int,
    sweep_mode: SweepMode,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Enumerate feasible step values for MaxEnt-SPRT.

    For MaxEnt, N_seg = K_total (fixed).  The only free parameter is
    ``step`` (hop between segments).  Any integer in [1, K_total] is valid.
    """
    if K_total < 1:
        raise ValueError(
            f"K_total must be >= 1 for MaxEnt, got {K_total}."
        )

    if sweep_mode == SweepMode.FIX_WIN:
        raise ValueError(
            "FIX_WIN is not applicable to MaxEnt-SPRT "
            "(N_seg is always equal to K_total)."
        )

    if sweep_mode == SweepMode.FIX_N_ACCUM:
        raise ValueError(
            "FIX_N_ACCUM is not applicable to MaxEnt-SPRT."
        )

    if sweep_mode == SweepMode.FIX_STEP:
        step_fixed: Optional[int] = int(kwargs["step"])
        step_range = range(step_fixed, step_fixed + 1)
    else:
        step_range = range(1, K_total + 1)  # 1 ≤ step ≤ K_total

    combos: List[Dict[str, Any]] = []

    for step in step_range:
        if not (1 <= step <= K_total):
            continue
        combos.append({
            "K_total":      K_total,
            "N_win":        None,          # N_seg = K_total (not a "window" in the same sense)
            "step":         step,
            "n_accum":      None,          # MaxEnt processes all available segments
            "overlap_frac": None,          # not applicable
        })

    n_valid = len(combos)
    for c in combos:
        c["n_combos_valid"] = n_valid

    return combos
