"""
sweep/config_builder.py
=======================
Build the ``INDICATOR_CONFIG`` dict consumed by each indicator's
``run_*`` API from a ``(indicator_id, basis, combo, base_params)`` tuple.

Each indicator has a distinct physical-parameter layout:

RMS-CV  (by_modal)
    ``T_modal``, ``N_modal_window``, ``step_modal``,
    ``n_max_mode="frames"``, ``n_max_modal``

RMS-CV  (by_revolution)
    ``T_rev``, ``N_rev_window``, ``step_rev``,
    ``n_max_mode="frames"``, ``n_max_rev``

SST-SVD  (by_modal)
    ``T_modal``, ``N_modal_window``, ``step_modal``,
    ``Ai_length_mode="frames"``, ``Ai_length_modal``

SST-SVD  (by_revolution)
    ``T_rev``, ``N_rev_window``, ``step_rev``,
    ``Ai_length_mode="frames"``, ``Ai_length_rev``

MaxEnt  (by_modal)
    ``T_rev`` (required quirk — does not affect T_total),
    ``T_modal``, ``N_modal_per_seg``, ``step_modal``

MaxEnt  (by_revolution)
    ``T_rev``, ``N_rev_per_seg``, ``step_rev``,
    + ``segmentation="raw"`` automatically injected when
    ``not basis.maxent_opr_valid`` (OPR blind to chatter frequency).

Usage
-----
    from sweep.basis import StudyBasis
    from sweep.config_builder import build_indicator_config

    basis  = StudyBasis("by_modal", f_modal=150.0, rpm=12_000.0)
    combo  = {"K_total": 8, "N_win": 4, "step": 2, "n_accum": 3}
    config = build_indicator_config("rms_cv", basis, combo, base_params={...})
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .basis import StudyBasis

__all__ = ["build_indicator_config"]

# ── indicator id normalisation ───────────────────────────────────────────────
_RMS_CV_IDS   = {"rms_cv"}
_SST_SVD_IDS  = {"sst_svd"}
_MAXENT_IDS   = {"maxent", "maxent_sprt"}


def build_indicator_config(
    indicator_id: str,
    basis: StudyBasis,
    combo: Dict[str, Any],
    base_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the ``INDICATOR_CONFIG`` dict for one run.

    Parameters
    ----------
    indicator_id : str
        Indicator name: ``"rms_cv"``, ``"sst_svd"``, ``"maxent"``
        (or ``"maxent_sprt"``).
    basis : StudyBasis
        Physical basis for the study (supplies T_unit, T_rev, T_modal, mode).
    combo : dict
        Single combo dict produced by :func:`~sweep.enumerator.enumerate_feasible`.
        Keys: ``K_total``, ``N_win`` (None for MaxEnt), ``step``,
        ``n_accum`` (None for MaxEnt), ``overlap_frac``.
    base_params : dict
        Non-swept indicator-specific parameters (thresholds, flags, etc.).
        These are merged into ``params_physical`` and will be passed through
        by the indicator runner to the internal pipeline.

    Returns
    -------
    dict
        A complete ``INDICATOR_CONFIG`` ready for ``run_rms_cv``,
        ``run_sst_svd``, or ``run_maxent_sprt``.

    Raises
    ------
    ValueError
        On unknown ``indicator_id``.
    """
    ind = indicator_id.lower()

    if ind in _RMS_CV_IDS:
        return _build_rms_cv(basis, combo, base_params)
    elif ind in _SST_SVD_IDS:
        return _build_sst_svd(basis, combo, base_params)
    elif ind in _MAXENT_IDS:
        return _build_maxent(basis, combo, base_params)
    else:
        raise ValueError(
            f"Unknown indicator_id {indicator_id!r}. "
            f"Supported: rms_cv, sst_svd, maxent, maxent_sprt."
        )


# ── RMS-CV ────────────────────────────────────────────────────────────────────

def _build_rms_cv(
    basis: StudyBasis,
    combo: Dict[str, Any],
    base_params: Dict[str, Any],
) -> Dict[str, Any]:
    N_win   = int(combo["N_win"])
    step    = int(combo["step"])
    n_accum = int(combo["n_accum"])

    if basis.mode == "by_modal":
        phys: Dict[str, Any] = {
            "T_modal":        basis.T_modal,
            "N_modal_window": N_win,
            "step_modal":     step,
            "n_max_mode":     "frames",
            "n_max_modal":    n_accum,
        }
    else:  # by_revolution
        phys = {
            "T_rev":        basis.T_rev,
            "N_rev_window": N_win,
            "step_rev":     step,
            "n_max_mode":   "frames",
            "n_max_rev":    n_accum,
        }

    phys.update(base_params)

    return {
        "id":              "RMS_CV",
        "func":            "Default",
        "param_mode":      basis.mode,
        "params_physical": phys,
    }


# ── SST-SVD ───────────────────────────────────────────────────────────────────

def _build_sst_svd(
    basis: StudyBasis,
    combo: Dict[str, Any],
    base_params: Dict[str, Any],
) -> Dict[str, Any]:
    N_win   = int(combo["N_win"])
    step    = int(combo["step"])
    n_accum = int(combo["n_accum"])

    if basis.mode == "by_modal":
        phys: Dict[str, Any] = {
            "T_modal":         basis.T_modal,
            "N_modal_window":  N_win,
            "step_modal":      step,
            "Ai_length_mode":  "frames",
            "Ai_length_modal": n_accum,
        }
    else:  # by_revolution
        phys = {
            "T_rev":         basis.T_rev,
            "N_rev_window":  N_win,
            "step_rev":      step,
            "Ai_length_mode": "frames",
            "Ai_length_rev": n_accum,
        }

    phys.update(base_params)

    return {
        "id":              "SST_SVD",
        "func":            "Default",
        "param_mode":      basis.mode,
        "params_physical": phys,
    }


# ── MaxEnt-SPRT ───────────────────────────────────────────────────────────────

def _build_maxent(
    basis: StudyBasis,
    combo: Dict[str, Any],
    base_params: Dict[str, Any],
) -> Dict[str, Any]:
    K_total = int(combo["K_total"])
    step    = int(combo["step"])

    if basis.mode == "by_modal":
        # T_rev is always required by the MaxEnt by_modal API even though
        # it does not affect T_total (API quirk documented in runner.py).
        phys: Dict[str, Any] = {
            "T_rev":           basis.T_rev,      # required quirk
            "T_modal":         basis.T_modal,
            "N_modal_per_seg": K_total,           # N_seg = K_total
            "step_modal":      step,
        }
    else:  # by_revolution
        phys = {
            "T_rev":          basis.T_rev,
            "N_rev_per_seg":  K_total,            # N_seg = K_total
            "step_rev":       step,
        }
        # Inject raw segmentation when OPR cannot resolve chatter frequency
        if not basis.maxent_opr_valid:
            phys["segmentation"] = "raw"

    phys.update(base_params)

    return {
        "id":              "MaxEnt_SPRT",
        "func":            "Default",
        "param_mode":      basis.mode,
        "params_physical": phys,
    }
