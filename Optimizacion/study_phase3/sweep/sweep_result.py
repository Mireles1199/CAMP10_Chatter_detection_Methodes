"""
sweep/sweep_result.py
=====================
Aggregate all ``RunResult`` objects from a sweep into a pandas DataFrame
and provide convenience analysis methods.

``SweepResult`` stores:
- ``df``     — flat pandas DataFrame with all scalar fields per run
- ``arrays`` — dict keyed by ``run_id`` → ``{"t_indicator", "I_t", "t_d_array"}``

Analysis methods
----------------
pareto(indicator, K_total)
    Returns the Pareto-optimal front on (delta_t_d, N_fa) for valid runs.
best(indicator, K_total)
    Returns the single run with the lowest ``score`` for valid runs.
sensitivity()
    For each indicator, returns mean delta_t_d and N_fa as a function of K_total.
gap_curve()
    Returns ``delta_T_total_vs_des`` statistics per (indicator, K_total).

DataFrame schema
----------------
basis_mode, K_total, T_des_s, indicator, N_win, step, n_accum,
overlap_frac, T_win_s, T_hop_s, T_total_actual_s, K_total_actual,
delta_K, delta_T_total_vs_des, lower_bound_delta_td, t_d_first,
delta_t_d, N_fa, P_det, score, n_pts_indicator, run_ok, error_str,
n_combos_valid

Usage
-----
    records = [run.to_record() for run in results]
    arrays  = {run.run_id: run.arrays for run in results}
    sweep   = SweepResult(records, arrays)
    print(sweep.df.head())
    print(sweep.best("rms_cv", K_total=8))
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .run_one import RunResult

__all__ = ["SweepResult"]

# ── columns in order ─────────────────────────────────────────────────────────
_DF_COLUMNS = [
    "run_id",
    "basis_mode",
    "indicator",
    "N_cyc_total",
    "T_des_s",
    "N_cyc",
    "step_cyc",
    "N_fen",
    "overlap_frac",
    "T_win_s",
    "T_hop_s",
    "T_total_actual_s",
    "N_cyc_total_actual",
    "delta_K",
    "delta_T_total_vs_des",
    "lower_bound_delta_td",
    "t_d_first",
    "t_d_first_true",
    "delta_t_d",
    "N_fa",
    "P_det",
    "score",
    "score_lb",
    "n_pts_indicator",
    "run_ok",
    "error_str",
    "n_combos_valid",
]


def _run_to_record(run: RunResult) -> Dict[str, Any]:
    """Convert a :class:`RunResult` to a flat dict for DataFrame construction."""
    return {
        "run_id":                run.run_id,
        "basis_mode":            run.basis_mode,
        "indicator":             run.indicator,
        "N_cyc_total":           run.K_total,
        "T_des_s":               run.T_des_s,
        "N_cyc":                 run.N_win,
        "step_cyc":              run.step,
        "N_fen":                 run.n_accum,
        "overlap_frac":          run.overlap_frac,
        "T_win_s":               run.T_win_s,
        "T_hop_s":               run.T_hop_s,
        "T_total_actual_s":      run.T_total_actual_s,
        "N_cyc_total_actual":    run.K_total_actual,
        "delta_K":               run.delta_K,
        "delta_T_total_vs_des":  run.delta_T_total_vs_des,
        "lower_bound_delta_td":  run.lower_bound_delta_td,
        "t_d_first":             run.t_d_first,
        "t_d_first_true":        run.t_d_first_true,
        "delta_t_d":             run.delta_t_d,
        "N_fa":                  run.N_fa,
        "P_det":                 run.P_det,
        "score":                 run.score,
        "score_lb":              run.score_lb,
        "n_pts_indicator":       run.n_pts_indicator,
        "run_ok":                run.run_ok,
        "error_str":             run.error_str,
        "n_combos_valid":        run.n_combos_valid,
    }


class SweepResult:
    """
    Aggregated results of a discrete-parameter sweep study.

    Parameters
    ----------
    runs : list of RunResult
        All completed run results (succeeded and failed).

    Attributes
    ----------
    df : pd.DataFrame
        Flat DataFrame with one row per run.
    arrays : dict
        ``{run_id: {"t_indicator": ..., "I_t": ..., "t_d_array": ...}}``
    """

    def __init__(self, runs: List[RunResult]) -> None:
        self._runs   = runs
        records      = [_run_to_record(r) for r in runs]
        self._df     = pd.DataFrame(records, columns=_DF_COLUMNS)
        self._arrays = {r.run_id: r.arrays for r in runs}

    # ── public properties ─────────────────────────────────────────────────────

    @property
    def df(self) -> pd.DataFrame:
        """Full results DataFrame."""
        return self._df

    @property
    def arrays(self) -> Dict[str, Any]:
        """Signal arrays keyed by run_id."""
        return self._arrays

    # ── analysis methods ──────────────────────────────────────────────────────

    def valid_runs(
        self,
        indicator: Optional[str] = None,
        K_total: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Return subset of ``df`` for runs that succeeded and detected chatter.

        Parameters
        ----------
        indicator : str, optional
            Filter by indicator name.
        K_total : int, optional
            Filter by K_total value.
        """
        mask = self._df["run_ok"] & self._df["P_det"].astype(bool)
        if indicator is not None:
            mask &= self._df["indicator"] == indicator.lower()
        if K_total is not None:
            mask &= self._df["N_cyc_total"] == K_total
        return self._df[mask].copy()

    def failed_runs(self) -> pd.DataFrame:
        """Return subset of ``df`` for runs that raised an exception."""
        return self._df[~self._df["run_ok"]].copy()

    def pareto(
        self,
        indicator: str,
        K_total: int,
    ) -> pd.DataFrame:
        """
        Return the Pareto-optimal front on (delta_t_d, N_fa).

        A run dominates another if it has ≤ delta_t_d AND ≤ N_fa, with at
        least one strict inequality.

        Parameters
        ----------
        indicator : str
        K_total : int

        Returns
        -------
        pd.DataFrame sorted by delta_t_d ascending.
        """
        sub = self.valid_runs(indicator=indicator, K_total=K_total)
        if sub.empty:
            return sub

        sub = sub.sort_values(["delta_t_d", "N_fa"]).reset_index(drop=True)

        # Greedy Pareto filter
        pareto_rows = []
        min_N_fa    = math.inf
        for _, row in sub.iterrows():
            if row["N_fa"] < min_N_fa:
                pareto_rows.append(row)
                min_N_fa = row["N_fa"]

        return pd.DataFrame(pareto_rows).reset_index(drop=True)

    def best(
        self,
        indicator: str,
        K_total: int,
    ) -> pd.Series:
        """
        Return the single run with the lowest composite ``score``.

        Parameters
        ----------
        indicator : str
        K_total : int

        Returns
        -------
        pd.Series (one row) or empty Series if no valid runs exist.
        """
        sub = self.valid_runs(indicator=indicator, K_total=K_total)
        if sub.empty:
            return pd.Series(dtype=float)
        sub_finite = sub[sub["score"].notna() & (sub["score"] >= 0)]
        if sub_finite.empty:
            return pd.Series(dtype=float)
        idx = sub_finite["score"].idxmin()
        return sub_finite.loc[idx]

    def sensitivity(self) -> pd.DataFrame:
        """
        Sensitivity of delta_t_d and N_fa to K_total per indicator.

        Returns a DataFrame indexed by (indicator, K_total) with columns:
        ``mean_delta_t_d``, ``std_delta_t_d``, ``min_delta_t_d``,
        ``mean_N_fa``, ``P_det_rate``, ``n_valid``.
        """
        sub = self._df[self._df["run_ok"]].copy()
        if sub.empty:
            return pd.DataFrame()

        rows = []
        for (ind, K), group in sub.groupby(["indicator", "N_cyc_total"]):
            det     = group[group["P_det"].astype(bool)]
            n_valid = len(det)
            rows.append({
                "indicator":       ind,
                "N_cyc_total":     K,
                "n_valid":         n_valid,
                "P_det_rate":      n_valid / max(len(group), 1),
                "mean_t_d_first":       det["t_d_first"].mean()      if n_valid > 0 else math.nan,
                "min_t_d_first":        det["t_d_first"].min()       if n_valid > 0 else math.nan,
                "mean_t_d_first_true":  det["t_d_first_true"].mean() if n_valid > 0 else math.nan,
                "min_t_d_first_true":   det["t_d_first_true"].min()  if n_valid > 0 else math.nan,
                "mean_delta_t_d":  det["delta_t_d"].mean() if n_valid > 0 else math.nan,
                "std_delta_t_d":   det["delta_t_d"].std()  if n_valid > 1 else math.nan,
                "min_delta_t_d":   det["delta_t_d"].min()  if n_valid > 0 else math.nan,
                "mean_N_fa":       det["N_fa"].mean()       if n_valid > 0 else math.nan,
            })

        return pd.DataFrame(rows).set_index(["indicator", "N_cyc_total"])

    def gap_curve(self) -> pd.DataFrame:
        """
        T_total quantisation gap statistics per (indicator, K_total).

        Returns a DataFrame indexed by (indicator, K_total) with columns:
        ``mean_delta_T``, ``max_delta_T``, ``mean_delta_K``.
        """
        sub = self._df[self._df["run_ok"] & self._df["delta_T_total_vs_des"].notna()].copy()
        if sub.empty:
            return pd.DataFrame()

        rows = []
        for (ind, K), group in sub.groupby(["indicator", "N_cyc_total"]):
            rows.append({
                "indicator":    ind,
                "N_cyc_total":  K,
                "mean_delta_T": group["delta_T_total_vs_des"].mean(),
                "max_delta_T":  group["delta_T_total_vs_des"].abs().max(),
                "mean_delta_K": group["delta_K"].mean(),
            })

        return pd.DataFrame(rows).set_index(["indicator", "N_cyc_total"])

    # ── convenience ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a structured text summary of the sweep results."""
        total    = len(self._df)
        ok       = int(self._df["run_ok"].sum())
        failed   = total - ok
        detected = int((self._df["run_ok"] & self._df["P_det"].astype(bool)).sum())
        header   = f"SweepResult --- {total} runs  (OK={ok} | Failed={failed} | Detected={detected})"
        sub = self._df[self._df["run_ok"]].copy()
        if sub.empty:
            return header
        sub["t_d [ms]"]      = (sub["t_d_first"]      * 1e3).round(1)
        sub["t_d_true [ms]"] = (sub["t_d_first_true"] * 1e3).round(1)
        sub["Dt_d [ms]"]     = (sub["delta_t_d"]      * 1e3).round(1)
        sub["score"]         = sub["score"].round(4)
        disp = sub[["indicator", "N_cyc_total", "N_cyc", "step_cyc", "N_fen",
                    "t_d [ms]", "t_d_true [ms]", "Dt_d [ms]", "N_fa", "score"]].copy()
        disp = disp.rename(columns={"N_cyc_total": "K", "N_cyc": "N_cyc", "N_fen": "N_fen"})
        disp = disp.sort_values(["indicator", "K"]).reset_index(drop=True)
        disp = disp.set_index(["indicator", "K"])
        return header + "\n" + disp.to_string()

    def best_table(self) -> pd.DataFrame:
        """
        Return a DataFrame with the best run per (indicator, K_total).

        One row per combination, indexed by (indicator, K).  Columns:
        N_win, step, N_acc, t_d [ms], t_d_true [ms], Dt_d [ms], N_fa, score.
        """
        rows = []
        for ind in sorted(self._df["indicator"].unique()):
            for K in sorted(self._df["N_cyc_total"].unique()):
                b = self.best(ind, K)
                row: Dict[str, Any] = {"indicator": ind, "K": K}
                if b.empty:
                    row.update({
                        "N_cyc": pd.NA, "step_cyc": pd.NA, "N_fen": pd.NA,
                        "t_d [ms]": float("nan"), "t_d_true [ms]": float("nan"),
                        "Dt_d [ms]": float("nan"), "N_fa": pd.NA, "score": float("nan"),
                    })
                else:
                    row.update({
                        "N_cyc":         pd.NA if pd.isna(b["N_cyc"])  else int(b["N_cyc"]),
                        "step_cyc":      int(b["step_cyc"]),
                        "N_fen":         pd.NA if pd.isna(b["N_fen"])  else int(b["N_fen"]),
                        "t_d [ms]":      round(b["t_d_first"]      * 1e3, 1),
                        "t_d_true [ms]": round(b["t_d_first_true"] * 1e3, 1),
                        "Dt_d [ms]":     round(b["delta_t_d"]      * 1e3, 1),
                        "N_fa":          int(b["N_fa"]),
                        "score":         round(b["score"], 4),
                    })
                rows.append(row)
        return pd.DataFrame(rows).set_index(["indicator", "K"])

    # ── convergence & sensitivity analysis ───────────────────────────────────

    def convergence_vs_k(
        self,
        score_penalty: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Best-score and mean-score convergence curves as K_total increases.

        Parameters
        ----------
        score_penalty : float, optional
            Score assigned to non-detecting runs when computing
            ``mean_score_pen``.  If None, defaults to 3 x T_des_s(max K).

        Columns
        -------
        best_score       : min score among detecting runs
        mean_score       : mean score among detecting runs only
        std_score        : std  score among detecting runs only
        mean_score_pen   : penalised mean — non-detectors get score_penalty:
                           P_det_rate * mean_score + (1-P_det_rate) * penalty
        marginal_gain    : best_score(K-1) - best_score(K)  [positive = improvement]
        best_Dtd_ms      : delta_t_d [ms] of the best-score run
        best_Nfa         : N_fa of the best-score run
        n_combos         : total runs for this K
        n_detected       : runs that detected chatter
        P_det_rate       : n_detected / n_combos
        pct_good_05      : % runs with score < 0.05
        """
        rows = []
        for ind in sorted(self._df["indicator"].unique()):
            ind_df    = self._df[self._df["indicator"] == ind]
            prev_best: float = math.nan
            for K in sorted(ind_df["N_cyc_total"].unique()):
                sub   = ind_df[ind_df["N_cyc_total"] == K]
                valid = sub[
                    sub["run_ok"] & sub["P_det"].astype(bool)
                    & (sub["score"] >= 0) & sub["score"].notna()
                ]
                n_c   = len(sub)
                n_det = len(valid)
                if n_det > 0:
                    bs    = float(valid["score"].min())
                    ms    = float(valid["score"].mean())
                    ss    = float(valid["score"].std()) if n_det > 1 else math.nan
                    ibest = valid["score"].idxmin()
                    bd_ms = round(float(valid.loc[ibest, "delta_t_d"]) * 1e3, 1)
                    bn_fa = int(valid.loc[ibest, "N_fa"])
                    marg  = round(prev_best - bs, 6) if not math.isnan(prev_best) else math.nan
                    pct_g = round(len(valid[valid["score"] < 0.05]) / n_c * 100, 1)
                    prev_best = bs
                    _pen   = score_penalty if score_penalty is not None else (
                        float(sub["T_des_s"].max()) * 3.0
                    )
                    p_det  = n_det / n_c
                    ms_pen = p_det * ms + (1.0 - p_det) * _pen
                else:
                    bs = ms = ss = bd_ms = bn_fa = marg = pct_g = ms_pen = math.nan
                rows.append({
                    "indicator":      ind,
                    "N_cyc_total":    K,
                    "best_score":     round(bs,    6) if not math.isnan(bs)     else math.nan,
                    "mean_score":     round(ms,    6) if not math.isnan(ms)     else math.nan,
                    "std_score":      round(ss,    6) if not math.isnan(ss)     else math.nan,
                    "mean_score_pen": round(ms_pen,6) if not math.isnan(ms_pen) else math.nan,
                    "marginal_gain":  marg,
                    "best_Dtd_ms":    bd_ms,
                    "best_Nfa":       bn_fa,
                    "n_combos":       n_c,
                    "n_detected":     n_det,
                    "P_det_rate":     round(n_det / max(n_c, 1), 3),
                    "pct_good_05":    pct_g,
                })
        return pd.DataFrame(rows).set_index(["indicator", "N_cyc_total"])

    def param_sensitivity(
        self,
        indicator: Optional[str] = None,
        fixed_K: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Marginal effect of each swept parameter on score, delta_t_d, and N_fa.

        Parameters
        ----------
        indicator : str, optional  — filter to one indicator (None = all pooled)
        fixed_K   : int, optional  — restrict to K_total == fixed_K

        Returns
        -------
        dict keyed by 'step', 'N_win', 'n_accum', 'overlap_frac'.
        Each value: DataFrame indexed by param value.
        Columns: n_runs, P_det_rate, mean_score, std_score, min_score,
                 mean_Dtd_ms, std_Dtd_ms, min_Dtd_ms, mean_Nfa, var_ratio.
        """
        sub = self._df[self._df["run_ok"]].copy()
        if indicator is not None:
            sub = sub[sub["indicator"] == indicator.lower()]
        if fixed_K is not None:
            sub = sub[sub["N_cyc_total"] == fixed_K]
        if sub.empty:
            return {}
        valid = sub[
            sub["P_det"].astype(bool) & (sub["score"] >= 0) & sub["score"].notna()
        ].copy()
        total_var = float(valid["score"].var()) if len(valid) > 1 else math.nan
        result: Dict[str, pd.DataFrame] = {}
        for param in ("step_cyc", "N_cyc", "N_fen",):
            if param not in valid.columns or valid[param].dropna().nunique() < 2:
                continue
            v2 = valid[valid[param].notna()].copy()
            rows_p = []
            for val, grp in v2.groupby(param):
                n_all = int(sub[sub[param] == val].shape[0])
                n_v   = len(grp)
                sc    = grp["score"]
                dt    = grp["delta_t_d"] * 1e3
                nfa   = grp["N_fa"]
                rows_p.append({
                    param:           val,
                    "n_runs":        n_all,
                    "P_det_rate":    round(n_v / max(n_all, 1), 3),
                    "mean_score":    round(float(sc.mean()),  6),
                    "std_score":     round(float(sc.std()),   6) if n_v > 1 else math.nan,
                    "min_score":     round(float(sc.min()),   6),
                    "mean_Dtd_ms":   round(float(dt.mean()),  2),
                    "std_Dtd_ms":    round(float(dt.std()),   2) if n_v > 1 else math.nan,
                    "min_Dtd_ms":    round(float(dt.min()),   2),
                    "mean_Nfa":      round(float(nfa.mean()), 3),
                })
            df_p = pd.DataFrame(rows_p).set_index(param).sort_index()
            if not math.isnan(total_var) and total_var > 0:
                group_means = v2.groupby(param)["score"].mean()
                weights     = v2.groupby(param)["score"].count()
                w_mean      = float((group_means * weights).sum() / weights.sum())
                between_var = float(
                    ((group_means - w_mean) ** 2 * weights).sum() / weights.sum()
                )
                df_p["var_ratio"] = round(between_var / total_var, 4)
            else:
                df_p["var_ratio"] = math.nan
            result[param] = df_p
        return result

    def importance_ranking(
        self,
        indicator: Optional[str] = None,
        fixed_K: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Rank parameters by fraction of score variance they explain (var_ratio).

        Returns DataFrame indexed by parameter with columns:
        var_ratio, n_unique_values, score_range (max - min group means).
        """
        ps = self.param_sensitivity(indicator=indicator, fixed_K=fixed_K)
        if not ps:
            return pd.DataFrame()
        rows = []
        for param, df_p in ps.items():
            vr = float(df_p["var_ratio"].iloc[0]) if "var_ratio" in df_p.columns else math.nan
            rows.append({
                "parameter":       param,
                "var_ratio":       round(vr, 4),
                "n_unique_values": len(df_p),
                "score_range":     round(
                    float(df_p["mean_score"].max() - df_p["mean_score"].min()), 6
                ),
            })
        return (
            pd.DataFrame(rows)
            .set_index("parameter")
            .sort_values("var_ratio", ascending=False)
        )

    def tradeoff_table(
        self,
        indicator: Optional[str] = None,
        param: str = "step_cyc",
    ) -> pd.DataFrame:
        """
        Trade-off delta_t_d vs N_fa as a function of one parameter, per indicator.

        Returns DataFrame indexed by (indicator, <param>) with columns:
        mean_Dtd_ms, mean_Nfa, min_Dtd_ms, min_Nfa, n_runs.
        """
        sub = self._df[self._df["run_ok"]].copy()
        if indicator is not None:
            sub = sub[sub["indicator"] == indicator.lower()]
        valid = sub[
            sub["P_det"].astype(bool) & (sub["score"] >= 0) & sub["score"].notna()
        ].copy()
        if valid.empty or param not in valid.columns:
            return pd.DataFrame()
        valid = valid[valid[param].notna()].copy()
        rows = []
        for (ind, val), grp in valid.groupby(["indicator", param]):
            dt  = grp["delta_t_d"] * 1e3
            nfa = grp["N_fa"]
            rows.append({
                "indicator":   ind,
                param:         val,
                "mean_Dtd_ms": round(float(dt.mean()),  2),
                "mean_Nfa":    round(float(nfa.mean()), 3),
                "min_Dtd_ms":  round(float(dt.min()),   2),
                "min_Nfa":     int(nfa.min()),
                "n_runs":      len(grp),
            })
        return pd.DataFrame(rows).set_index(["indicator", param]).sort_index()

    def feasible_space_quality(self, score_threshold: float = 0.05) -> pd.DataFrame:
        """
        Fraction of combos with score < score_threshold per (indicator, K_total).

        Reveals whether the optimum is isolated (low %) or robust/broad (high %).
        Returns DataFrame indexed by (indicator, K_total) with columns:
        n_total, n_valid, n_good, pct_good, score_threshold.
        """
        sub = self._df[self._df["run_ok"]].copy()
        rows = []
        for (ind, K), grp in sub.groupby(["indicator", "N_cyc_total"]):
            valid  = grp[
                grp["P_det"].astype(bool) & (grp["score"] >= 0) & grp["score"].notna()
            ]
            n_good = int((valid["score"] < score_threshold).sum())
            rows.append({
                "indicator":       ind,
                "N_cyc_total":     K,
                "n_total":         len(grp),
                "n_valid":         len(valid),
                "n_good":          n_good,
                "pct_good":        round(n_good / max(len(grp), 1) * 100, 1),
                "score_threshold": score_threshold,
            })
        return pd.DataFrame(rows).set_index(["indicator", "N_cyc_total"])

    def __repr__(self) -> str:
        return f"SweepResult({len(self._df)} runs, {len(self._df['indicator'].unique())} indicators)"
