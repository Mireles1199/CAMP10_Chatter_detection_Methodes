"""
noise_study_runner.py
=====================

Monte Carlo robustness study for your indicators under MEASUREMENT noise (AWGN),
using your existing interfaces:

- SignalData (your dataclass/class)
- run_indicator(sig, INDICATOR_CONFIG) -> IndicatorResult
- IndicatorResult.t_d is a LIST of detection times (candidates)
- For transition cases with known t_E: VALID detection = first t_d >= t_E

You can adjust:
- SNR levels (snr_grid_db)
- base seed (seed_base)
- number of Monte Carlo runs (n_mc)

This script is designed so that VS Code Copilot/Claude only needs to "polish"
imports and connect to your real pipeline functions.

Requirements: numpy. (Optional: pandas for CSV convenience)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import math
import numpy as np
import os
import logging

try:
    import pandas as pd  # optional
except Exception:
    pd = None

from MaxEnt_SPRT import SignalData
from MaxEnt_SPRT import HDF5Reader
from MaxEnt_SPRT import run_maxent_sprt

# ============================================================
# 1) USER INTERFACES (you already have these classes)
# ============================================================
# - SignalData: your object that carries t_analysis, signal_analysis, fs, meta...
# - IndicatorResult: your object with at least .name, .t, .I_t, .t_d, .meta

IndicatorFn = Callable[[Any, Dict[str, Any]], Any]  # (sig, config) -> IndicatorResult

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


# ============================================================
# 2) CONFIG
# ============================================================

@dataclass(frozen=True)
class StudySettings:
    snr_grid_db: Tuple[float, ...] = (40.0, 30.0, 20.0, 10.0, 5.0)
    n_mc: int = 50
    seed_base: int = 12345
    eps_after_tE: float = 0.0  # if you want strict > tE, set tiny eps

    # Power reference segment used to define SNR (in seconds, in t_analysis coordinates)
    t_ref0: float = 0.0
    t_ref1: float = 1.0

    # If True, compute Px on (signal_analysis) as-is. If you want "post-filter" power,
    # pass a pre-filtered signal_analysis in sig (or modify compute_power_ref).
    compute_power_on_signal_analysis: bool = True


@dataclass(frozen=True)
class CaseDef:
    """
    One test case.
    kind:
      - "A1_stable_only": stable for whole duration (no valid t_d; only false alarms)
      - "A2_unstable_only": unstable from the start (t_E = t_start conceptually)
      - "B_transition": stable then unstable, must supply t_E
      - "C_nonstationary": ap ramp etc.; may or may not supply t_E
    """
    name: str
    kind: str
    t_E: Optional[float] = None  # only meaningful for transition-like cases


# ============================================================
# 3) CORE HELPERS: power, SNR->sigma, noise, t_d filtering
# ============================================================

def compute_power_ref(sig: Any, t0: float, t1: float) -> float:
    """
    Px = mean(x^2) over t in [t0, t1), using sig.t_analysis and sig.signal_analysis.
    """
    t = np.asarray(sig.t_analysis, dtype=float)
    x = np.asarray(sig.signal_analysis, dtype=float)

    mask = (t >= t0) & (t < t1)
    if not np.any(mask):
        raise ValueError(
            f"Power reference segment empty. Check t_ref0={t0}, t_ref1={t1} "
            f"against t_analysis range [{t.min()}, {t.max()}]."
        )
    x_ref = x[mask]
    Px = float(np.mean(x_ref**2))
    if Px <= 0 or not np.isfinite(Px):
        raise ValueError(f"Invalid Px={Px}. Cannot define SNR.")
    return Px


def sigma_from_snr(Px: float, snr_db: float) -> float:
    """
    SNR_dB = 10 log10(Px / Pv), with Pv = sigma^2 (for AWGN).
    """
    Pv = Px * (10.0 ** (-snr_db / 10.0))
    sigma = float(np.sqrt(Pv))
    return sigma


def add_awgn(rng: np.random.Generator, x: np.ndarray, sigma: float) -> np.ndarray:
    return x + rng.normal(loc=0.0, scale=sigma, size=x.shape)


def make_noisy_signaldata(sig_clean: Any, rng: np.random.Generator, sigma: float) -> Any:
    """
    Return a copy of sig_clean with ONLY signal_analysis contaminated.
    You said other fields are for plotting only.
    """
    noisy = add_awgn(rng, np.asarray(sig_clean.signal_analysis, dtype=float), sigma)

    # Most robust: create a shallow copy by mutating a clone if available
    # If SignalData is a dataclass, it often supports "dataclasses.replace".
    try:
        from dataclasses import replace
        return replace(sig_clean, signal_analysis=noisy)
    except Exception:
        # fallback: try .copy()
        if hasattr(sig_clean, "copy"):
            s2 = sig_clean.copy()
            s2.signal_analysis = noisy
            return s2
        # last resort: try constructing a new object (you will adapt this)
        raise TypeError(
            "Cannot copy SignalData. Please make SignalData a dataclass or add .copy()."
        )


def first_alarm(t_d_list: Any) -> Optional[float]:
    """
    First alarm among candidate times (could be false).
    t_d_list is expected to be list/array-like, but we handle scalars too.
    """
    if t_d_list is None:
        return None
    if isinstance(t_d_list, (float, int, np.floating, np.integer)):
        val = float(t_d_list)
        return None if not np.isfinite(val) else val

    try:
        arr = np.asarray(list(t_d_list), dtype=float)
    except Exception:
        return None

    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.min(arr))


def first_valid_after(t_d_list: Any, t_E: float, eps: float = 0.0) -> Optional[float]:
    """
    First detection candidate >= t_E (+ eps). If none, returns None (i.e., no valid detection).
    """
    if t_d_list is None:
        return None
    try:
        arr = np.asarray(list(t_d_list), dtype=float)
    except Exception:
        # scalar?
        if isinstance(t_d_list, (float, int, np.floating, np.integer)):
            val = float(t_d_list)
            return val if (np.isfinite(val) and val >= t_E + eps) else None
        return None

    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None

    valid = arr[arr >= (t_E + eps)]
    if valid.size == 0:
        return None
    return float(np.min(valid))


# ============================================================
# 4) ONE MONTE CARLO RUN (one indicator, one noise realization)
# ============================================================

def run_one(
    *,
    sig_clean: Any,
    indicator_fn: IndicatorFn,
    indicator_config: Dict[str, Any],
    sigma: float,
    seed: int,
    case: CaseDef,
    eps_after_tE: float,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    sig_noisy = make_noisy_signaldata(sig_clean, rng, sigma)

    diff = sig_noisy.signal_analysis - sig_clean.signal_analysis
    logger.debug("noise_rms", float(np.sqrt(np.mean(diff**2))), "noise_max", float(np.max(np.abs(diff))))

    res = indicator_fn(sig_noisy, indicator_config)  # -> IndicatorResult
    t_d_list = getattr(res, "t_d", None)

    t_first_any = first_alarm(t_d_list)

    # Valid detection time depends on case kind
    t_valid: Optional[float] = None
    delay: Optional[float] = None
    premature_alarm: Optional[bool] = None

    if case.kind in ("B_transition", "C_nonstationary") and (case.t_E is not None):
        # "valid detection" must be AFTER t_E
        t_valid = first_valid_after(t_d_list, case.t_E, eps=eps_after_tE)
        delay = (t_valid - case.t_E) if (t_valid is not None) else None
        premature_alarm = (t_first_any is not None and t_first_any < case.t_E)
    elif case.kind == "A2_unstable_only":
        # unstable from the start: first alarm is considered valid detection
        t_valid = t_first_any
        delay = None
        premature_alarm = None
    elif case.kind == "A1_stable_only":
        # stable only: there is NO valid detection, only false alarms
        t_valid = None
        delay = None
        premature_alarm = None
    else:
        # generic: treat first alarm as "activation"
        t_valid = t_first_any

    return {
        "t_first_any": t_first_any,
        "t_valid": t_valid,
        "delay": delay,
        "premature_alarm": premature_alarm,
        "seed": seed,
        "sigma": sigma,
        "indicator_name": getattr(res, "name", "unknown"),
        # Keep raw list if you want debugging:
        # "t_d_list": t_d_list,
    }


# ============================================================
# 5) MONTE CARLO LOOP: per indicator, per case, per SNR
# ============================================================

def seed_for(snr_db: float, mc_id: int, seed_base: int) -> int:
    """
    Reproducible seeds:
      base + 100000*round(snr) + mc_id
    """
    return int(seed_base + 100000 * int(round(snr_db)) + mc_id)


def summarize_case_rows(rows: List[Dict[str, Any]], case: CaseDef) -> Dict[str, Any]:
    """
    Compute metrics from Monte Carlo results at one (case, indicator, SNR).
    """
    n = len(rows)

    # First-alarm metrics
    t_first = np.array([r["t_first_any"] for r in rows if r["t_first_any"] is not None], dtype=float)
    p_any_alarm = float(t_first.size / max(n, 1))

    # Valid detection metrics
    t_valid = np.array([r["t_valid"] for r in rows if r["t_valid"] is not None], dtype=float)
    p_fail_valid = float(1.0 - (t_valid.size / max(n, 1)))

    out: Dict[str, Any] = {
        "case": case.name,
        "kind": case.kind,
        "n_mc": n,
        "p_any_alarm": p_any_alarm,            # includes false alarms
        "p_fail_valid": p_fail_valid,          # fails to provide valid detection
        "t_first_mean": float(np.mean(t_first)) if t_first.size else math.nan,
        "t_first_std": float(np.std(t_first, ddof=1)) if t_first.size > 1 else (0.0 if t_first.size == 1 else math.nan),
        "t_valid_mean": float(np.mean(t_valid)) if t_valid.size else math.nan,
        "t_valid_std": float(np.std(t_valid, ddof=1)) if t_valid.size > 1 else (0.0 if t_valid.size == 1 else math.nan),
    }

    # Delay metrics for transition cases
    if case.t_E is not None:
        delays = np.array([r["delay"] for r in rows if r["delay"] is not None], dtype=float)
        prem = np.array([r["premature_alarm"] for r in rows if r["premature_alarm"] is not None], dtype=bool)
        out.update({
            "t_E": float(case.t_E),
            "delay_mean": float(np.mean(delays)) if delays.size else math.nan,
            "delay_std": float(np.std(delays, ddof=1)) if delays.size > 1 else (0.0 if delays.size == 1 else math.nan),
            "p_premature_alarm": float(np.mean(prem)) if prem.size else math.nan,
        })

    return out


def run_study(
    *,
    sig_by_case: Dict[str, Any],                      # case.name -> SignalData (clean)
    indicators: Dict[str, Tuple[IndicatorFn, Dict[str, Any]]],  # indicator_name -> (fn, config)
    cases: List[CaseDef],
    settings: StudySettings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
      run_rows: list of per-run records
      summary_rows: list of per-(case,indicator,SNR) summaries
    """
    run_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for case in cases:
        sig_clean = sig_by_case[case.name]

        # Compute Px once per case (based on clean signal + reference segment)
        Px = compute_power_ref(sig_clean, settings.t_ref0, settings.t_ref1)

        for ind_name, (ind_fn, ind_cfg) in indicators.items():
            for snr_db in settings.snr_grid_db:
                sigma = sigma_from_snr(Px, float(snr_db))

                rows_this: List[Dict[str, Any]] = []
                for mc_id in range(settings.n_mc):
                    seed = seed_for(snr_db, mc_id, settings.seed_base)

                    out = run_one(
                        sig_clean=sig_clean,
                        indicator_fn=ind_fn,
                        indicator_config=ind_cfg,
                        sigma=sigma,
                        seed=seed,
                        case=case,
                        eps_after_tE=settings.eps_after_tE,
                    )

                    rec = {
                        "case": case.name,
                        "kind": case.kind,
                        "indicator": ind_name,
                        "snr_db": float(snr_db),
                        "Px_ref": Px,
                        "sigma": sigma,
                        "mc_id": mc_id,
                        **out,
                    }
                    run_rows.append(rec)
                    rows_this.append(rec)

                # Summarize
                summ = summarize_case_rows(rows_this, case)
                summ.update({
                    "indicator": ind_name,
                    "snr_db": float(snr_db),
                    "Px_ref": Px,
                    "sigma": sigma,
                })
                summary_rows.append(summ)

    return run_rows, summary_rows


# ============================================================
# 6) EXPORT HELPERS
# ============================================================

def save_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if pd is not None:
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        import csv
        if len(rows) == 0:
            raise ValueError("No rows to save.")
        keys = sorted(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)


# ============================================================
# 7) HOW YOU CONNECT YOUR REAL INDICATORS
# ============================================================
# Example:
#   indicators = {
#       "MaxEnt_SPRT": (run_maxent_sprt, INDICATOR_CONFIG_MAXENT),
#       "Areas": (run_areas, INDICATOR_CONFIG_AREAS),
#   }




# ============================================================
# 8) MAIN EXAMPLE (you will adapt to your real signals)
# ============================================================

def main():

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

    t_stable = 5.365770208787228

    CFG_MAXENT = {
        "id": "MaxEnt_SPRT",                  # internal identifier (optional)
        "func": "Default",                    # indicator wrapper
        "params": {                           # default parameters for this benchmark
            "rpm": 12_000.0,
            "ratio_sampling": 50.0,
            "N_seg": 2,
            "t_stable_total": t_stable,
            "alpha": 0.05,
            "beta": 0.05,
            "reset_on_H0": True,
            "cut_start_time":  1.00608,
            "cut_end_time":10.30330 ,
        },
    }

    sig = SignalData(
        # t_cut=t_cut,
        # v_cut=v_cut,
        # x_cut = x_cut,
        # force_cut = force_cut,
        # t_original=t,
        # x_original=tool_dyn,
        # v_original=v,
        t_analysis=t_cut,
        signal_analysis=v_cut,
        # force_original=force_N,
        path=data_dir,
        fs=fs,
        meta={"AP": "5mm-15mm",
                "RPM": 12_000,}
    )


    # --------- Adjust settings here ----------
    settings = StudySettings(
        snr_grid_db=(20.0, 10.0, 5.0, 0.0),
        n_mc=50,
        seed_base=20260302,
        t_ref0=0.0,
        t_ref1=t_stable,
        eps_after_tE=0.0,   # set small >0 if you require strictly after t_E
    )

    # --------- Define cases ----------
    # A1: stable only
    # A2: unstable only
    # B: transition (must give t_E)
    # C: nonstationary (optional t_E)
    cases = [
        # CaseDef(name="A1_stable_only", kind="A1_stable_only", t_E=None),
        # CaseDef(name="A2_unstable_only", kind="A2_unstable_only", t_E=None),
        CaseDef(name="Case_Test", kind="B_transition", t_E=t_stable),
        # CaseDef(name="C_nonstationary", kind="C_nonstationary", t_E=2.0),
    ]

    sig_by_case = {
        # "A1_stable_only": FakeSig(t, x_stable, fs),
        # "A2_unstable_only": FakeSig(t, x_unstable, fs),
        "Case_Test": sig,
        # "C_nonstationary": FakeSig(t, x_transition, fs),  # replace with ap-ramp signal
    }

    # --------- Register indicators ----------
    indicators = {
          "MaxEnt_SPRT": (run_maxent_sprt, CFG_MAXENT),
        #   "Areas": (run_areas, cfg_areas),
    }

    # --------- Run study ----------
    run_rows, summary_rows = run_study(
        sig_by_case=sig_by_case,
        indicators=indicators,
        cases=cases,
        settings=settings,
    )

    # --------- Save results ----------
    save_csv("noise_mc_runs.csv", run_rows)
    save_csv("noise_mc_summary.csv", summary_rows)

    print("Saved:")
    print(" - noise_mc_runs.csv (per Monte Carlo run)")
    print(" - noise_mc_summary.csv (per case/indicator/SNR summary)")
    print(f"Total runs: {len(run_rows)} | Total summaries: {len(summary_rows)}")


if __name__ == "__main__":
    main()