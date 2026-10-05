#!/usr/bin/env python
# coding: utf-8
"""validate_indicators.py — Score the indicators of a validation DOE against its amplitude labels.

Inputs
  --ind_results  doe_indicator_results.h5   (doe_indicators.py, run on the VALIDATION doe_results.h5, with the
                                             training reference_dataset as reference)
  --labels       reference_dataset*.h5      (reference_dataset.py build on the VALIDATION doe_results.h5; the
                                             stable/unstable/gray labels are the ground truth)
  --reference    (optional) training reference_dataset*.h5, stored in /training for plotting

Output  doe_validation_results.h5  — same layout as doe_indicator_results.h5 (case_NNN/<run_name>/{t, I_t, t_d}),
so doe_unified_selector.py opens it as-is (table columns, SLD, I_t curves), plus the validation data:
  case_NNN attrs : (copy of the case attrs) + $kappa$ $truth$ $t_onset_amp$ and one $outcome_<run>$ per indicator
                   (only attrs written as $name$ become columns of the viewer table)
  case_NNN       : datasets truth_t0, truth_t1, truth_label (datasets, not a group: the viewer reads every
                   sub-group of a case as an indicator run)
  case_NNN/<run> : pred, truth_w (per window: 0 stable, 1 unstable, -1 gray/unlabelled) + attrs outcome,
                   first_detection_t, delay_start_s, delay_onset_s and the window counts TP FP TN FN tpr tnr
  schema 4 (was 3): no early_tol (no TP_early outcome, no early_alarm/anticipated metrics, no early_tol_s attr)
  /summary/<run> : one row per case (case, kappa, ap_mm, spin_rpm, truth, outcome, first_detection_t,
                   delay_start_s, t_onset_amp, delay_onset_s, tpr, tnr)
  /metrics/<run> : attrs with the per-case metrics below (+ alarm quality and ROC/AUC)
  /ranking       : the runs ordered by balanced accuracy, then MCC, then AUC (+ <out>_metrics.csv with every metric)
  /pairwise      : exact McNemar test between every pair of indicators over the scored cases (run_a, run_b, a_only, b_only,
                   p_value); written when there are two or more indicators
  /roc/<run>/{high,low} : fpr, tpr, thr of the case-level ROC for each orientation of I_t
  /training      : case, kappa, ap_mm, spin_rpm, label of the training dataset

Labels are per CASE (the amplitude labelling gives each constant case one interval over the whole signal), so the
headline metrics are per case and there is NO time tolerance: an unstable case is TP if the indicator flags anything
(its first detection t_det exists), else FN; a stable case is FP if it flags anything, else TN; gray cases are ignored.
This is what the case-level ROC measures too (a detection = I_t above a threshold anywhere). t_onset = t_onset_amp below
(first sample over the limit) for constant cases; it is used only to report the delay, never to decide the outcome.
RAMPS of Ap (PLAN_ramps.md): a ramp whose truth goes from stable to unstable ('mixed', group 'ramp') has a label in time
(stable stretch, then unstable), so its first detection t_det is compared with t_onset = start of its first unstable
window (no theoretical crossing time anywhere): t_det >= t_onset -> TP; t_det < t_onset -> FA (false alarm in the stable
stretch, nothing is tolerated); no detection -> FN. These ramps do NOT enter the global metrics, the ranking nor the ROC
(group 'global' = constant cases + ramps whose truth does not change) and get their own ramp_* metrics in /metrics/<run>
(see ramp_metrics).
  TPR = TP/(TP+FN)   TNR = TN/(TN+FP)   accuracy = (TP+TN)/(TP+TN+FP+FN)   balanced accuracy = (TPR+TNR)/2
  F1 = 2TP/(2TP+FP+FN)   MCC = (TP*TN - FP*FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
  TPR, TNR and accuracy come with a 95% Wilson interval (_lo/_hi): with few cases a bare proportion is very coarse.
Detection times, over the hits (TP; median, with p25 / p75 of the signed delay):
  delay_start_s = first flagged window inside the unstable part - its start (time since the signal starts)
  delay_onset_s = first detection - t_onset, signed (negative: the alarm came before the amplitude of the truth reached the limit)
  t_ratio = t_det / t_onset (per hit, median_t_ratio): < 1 = alarm before the amplitude limit is reached, independent of kappa
  t_onset_amp = first time |labeling_signal| exceeds labeling_lim_sup_pct % of the base (labeling_base_attr *
  labeling_base_scale): the same threshold that makes the amplitude labelling call a case unstable. NaN if the signal
  or the base is not in the validation file.
GRAY cases (a case whose whole label is gray: amplitude between the two limits) are handled by --gray (GRAY_MODES):
  ignore    (default) not scored: outcome 'n/a', left out of the counts, the metrics, the ranking and the ROC
  stable    pessimistic: scored as stable cases (alarm -> FP, none -> TN), they enter every metric and the ROC
  unstable  optimistic: scored as unstable cases (alarm -> TP, none -> FN), they enter every metric and the ROC
The mode is stored as the root attr gray_mode, the case attr $gray$ and the summary column is_gray. A non-default mode goes
to its own file (gray_suffix: doe_validation_results_gray-stable.h5, ...). Whatever the mode, gray_metrics reports apart
n_gray, n_gray_alarm, gray_alarm_rate and the bounds gray_as_stable_* / gray_as_unstable_* (TPR, TNR, balanced_accuracy,
MCC) computed on the cases that are not gray: so the omission is visible and the two scenarios can be compared.
Intervals: TPR, TNR, accuracy have Wilson 95%; balanced_accuracy and MCC have a 95% interval by simulation (ci_bal_mcc: TPR and
TNR drawn from Beta(k+0.5, n-k+0.5), the metric recomputed at each draw, fixed seed); AUC has Hanley-McNeil.
Alarm quality (window level, but on whole-signal labels so it is clean where it is used):
  alarm_fraction = flagged windows / windows, in STABLE cases (0 = never alarms); mean_alarm_fraction_stable
  persistence    = flagged windows / windows from the first detection on, in TP cases (1 = keeps the alarm on);
                   mean_persistence
ROC / AUC (case level): the case score is the max of I_t over the signal (a detection = I_t above a threshold anywhere,
so thresholding the max reproduces "detects anywhere"). Positives = unstable cases, negatives = stable cases.
Orientation (does chatter mean HIGH or LOW I_t?) differs between indicators; it is taken from the indicator's own
decisions (mean I_t of the flagged windows vs the unflagged ones, pooled over all cases), never from the labels. Both
orientations are stored (AUC_if_high_is_chatter / AUC_if_low_is_chatter); AUC is the one in `roc_direction`.
AUC has a Hanley-McNeil 95% interval (AUC_lo/AUC_hi); with few cases it is wide. The operating point of the indicator
(FPR = 1 - TNR, TPR) is a point on that ROC at its own threshold.
Window-level counts (TP FP TN FN tpr tnr per case) are also stored as a diagnostic: with a whole-signal label the
initial transient of an unstable case counts as unstable, so do not use them as the main score. A window is
"chatter" when its time is in t_d.

Usage (with the entorno_CAMP10 Python):
    python validate_indicators.py --ind_results X/doe_indicator_results.h5 --labels X/reference_dataset_amp.h5
    python validate_indicators.py --ind_results X --labels Y --gray stable      (pessimistic; also: ignore, unstable)
    python validate_indicators.py --selftest
"""
import argparse
import datetime
import os
import sys

import h5py
import numpy as np

CODE = {"stable": 0, "unstable": 1}   # anything else (gray) -> -1
EARLY_TOL_S = 0.0   # DEPRECATED, no effect: only so that doe_indicators / experiment.py (--early-tol) still import and call
OUTCOME_TEXT = {"TP": "OK", "FA": "MAL: false alarm", "FN": "MAL: missed", "TN": "OK", "FP": "MAL: false alarm"}


def detection_outcome(t_det: float, t_onset: float, early_tol=None) -> str:
    """Outcome of the unstable part of a RAMP from its FIRST detection t_det and the onset of the truth t_onset: TP if
    t_det >= t_onset, FA (false alarm in the stable stretch) before it, FN without detection. Without an onset, any
    detection is TP. early_tol is DEPRECATED and ignored."""
    if not np.isfinite(t_det):
        return "FN"
    return "TP" if not np.isfinite(t_onset) or t_det >= t_onset else "FA"
STR = h5py.string_dtype()
GRAY_MODES = ("ignore", "stable", "unstable")


def gray_suffix(mode: str) -> str:
    """Suffix of the output files for a gray mode: '' for the default (ignore), '_gray-stable', '_gray-unstable'."""
    return "" if mode == "ignore" else f"_gray-{mode}"


SUMMARY_FLOATS = ("is_gray", "t_ratio", "kappa", "ap_mm", "ap_end_mm", "kappa_start", "kappa_end", "spin_rpm", "first_detection_t",
                  "delay_start_s", "t_onset_amp", "t_onset", "delay_onset_s", "delay_det_s", "alarm_fraction", "persistence",
                  "hit_in_unstable", "score_max", "score_min", "tpr", "tnr")
RANK_BY = ("balanced_accuracy", "MCC", "AUC")


# ============================================================================== logic
def read_intervals(labels_h5: str, channel: str) -> dict:
    """{case: [(t0, t1, label)]} sorted by t0, taken from the pieces of one channel (all channels share intervals)."""
    out = {}
    with h5py.File(labels_h5, "r") as f:
        for label in f:
            for case in f[label]:
                for piece in f[label][case].values():
                    if str(piece.attrs.get("channel", "")) == channel:
                        out.setdefault(case, []).append((float(piece.attrs["t0"]), float(piece.attrs["t1"]), label))
    return {c: sorted(v) for c, v in out.items()}


def truth_windows(t, intervals) -> np.ndarray:
    """int8 per window: 0 stable, 1 unstable, -1 gray / outside every interval."""
    w = np.full(t.shape, -1, dtype=np.int8)
    for t0, t1, label in intervals:
        w[(t >= t0) & (t <= t1)] = CODE.get(label, -1)
    return w


def pred_windows(t, t_d) -> np.ndarray:
    """uint8 per window: 1 where the indicator flags chatter (nearest window to each t_d, within half a step)."""
    pred = np.zeros(t.shape, dtype=np.uint8)
    if t.size < 2 or t_d.size == 0:
        return pred
    j = np.clip(np.searchsorted(t, t_d), 1, t.size - 1)
    j = np.where(np.abs(t[j - 1] - t_d) <= np.abs(t[j] - t_d), j - 1, j)
    ok = np.abs(t[j] - t_d) <= 0.5 * np.median(np.diff(t)) * (1 + 1e-9)
    pred[j[ok]] = 1
    return pred


def score(t, pred, truth_w, t_start, t_onset=np.nan, ramp: bool = False) -> dict:
    """Window counts + case outcome + the detection times. A constant case with an unstable part is TP with any
    detection, else FN (no tolerance); a ramp is scored from its FIRST detection against t_onset (detection_outcome:
    TP, FA, FN; no t_onset -> t_start); a stable case is FP with any detection, else TN. delay_onset_s = first
    detection - t_onset, signed, for TP; delay_start_s = first flagged window inside the unstable part - t_start."""
    p, s, u = pred == 1, truth_w == 0, truth_w == 1
    tp, fp, tn, fn = int((p & u).sum()), int((p & s).sum()), int((~p & s).sum()), int((~p & u).sum())
    t_det = float(t[p][0]) if p.any() else np.nan
    onset = t_onset if np.isfinite(t_onset) else t_start
    if u.any():
        outcome = detection_outcome(t_det, onset) if ramp else ("TP" if np.isfinite(t_det) else "FN")
    elif s.any():
        outcome = "FP" if fp > 0 else "TN"
    else:
        outcome = "n/a"
    t_hit = t[p & u]
    hit = t_hit.size > 0
    return dict(TP=tp, FP=fp, TN=tn, FN=fn,
                alarm_fraction=fp / int(s.sum()) if s.any() else np.nan,
                persistence=float(p[u & (t >= t_hit[0])].mean()) if hit else np.nan,
                tpr=tp / (tp + fn) if tp + fn else np.nan, tnr=tn / (tn + fp) if tn + fp else np.nan,
                first_detection_t=t_det, n_fp_windows=fp, outcome=outcome, t_onset=float(onset),
                hit_in_unstable=float(hit),   # 0 for a 'TP' whose detections all fall outside the unstable part
                delay_start_s=float(t_hit[0] - t_start) if hit and np.isfinite(t_start) else np.nan,
                delay_onset_s=float(t_det - onset) if outcome == "TP" and np.isfinite(onset) else np.nan,
                # diagnostic: the same signed delay for ANY first detection (also an early alarm's)
                delay_det_s=float(t_det - onset) if u.any() and np.isfinite(onset) else np.nan,
                # t_det / t_onset of a hit (< 1: alarm before the amplitude limit is reached); only with a real onset
                t_ratio=float(t_det / t_onset) if outcome == "TP" and np.isfinite(t_onset) and t_onset > 0 else np.nan)


def case_truth(intervals) -> tuple:
    """(truth, t_start): truth in stable / unstable / mixed / unlabelled; t_start = start of the first unstable interval."""
    labels = {lab for _, _, lab in intervals}
    t_start = min((t0 for t0, _, lab in intervals if lab == "unstable"), default=np.nan)
    if "unstable" in labels:
        return ("mixed" if "stable" in labels else "unstable"), t_start
    return ("stable" if "stable" in labels else "unlabelled"), t_start


def amplitude_onset(sg, params: dict, t_start: float) -> float:
    """First time |signal| > lim_sup% of the base (the amplitude-labelling threshold), from t_start + warmup on.
    NaN when the case has no unstable interval, the signal is not in the case group or the base is missing."""
    if not np.isfinite(t_start):
        return np.nan
    try:
        thr = float(params["labeling_lim_sup_pct"]) / 100.0 * float(sg.attrs[params["labeling_base_attr"]]) \
            * float(params["labeling_base_scale"])
        g = sg[str(params["labeling_signal"])]
        t, y = g["time"][()], g["values"][()]
    except (KeyError, TypeError, ValueError):
        return np.nan
    hit = (np.abs(y) > thr) & (t >= t_start + float(params.get("labeling_warmup", 0.0)))
    return float(t[hit][0]) if thr > 0 and hit.any() else np.nan


def _attr_float(attrs, *keys) -> float:
    for k in keys:
        if k in attrs:
            try:
                return float(attrs[k])
            except (TypeError, ValueError):
                pass
    return np.nan


def write_training(out, ref_h5: str) -> None:
    rows = {}
    with h5py.File(ref_h5, "r") as f:
        for label in f:
            for case, cg in f[label].items():
                piece = next(iter(cg.values()), None)
                if piece is None:
                    continue
                a = piece.attrs
                r = rows.setdefault(case, dict(kappa=_attr_float(a, "kappa"), ap=_attr_float(a, "$Ap_start$") * 1e3,
                                               spin=_attr_float(a, "$spin_rate$"), labels=set()))
                r["labels"].add(label)
    g = out.create_group("training")
    names = sorted(rows)
    g.create_dataset("case", data=np.array(names, dtype=object), dtype=STR)
    g.create_dataset("kappa", data=[rows[n]["kappa"] for n in names])
    g.create_dataset("ap_mm", data=[rows[n]["ap"] for n in names])
    g.create_dataset("spin_rpm", data=[rows[n]["spin"] for n in names])
    g.create_dataset("label", data=np.array([",".join(sorted(rows[n]["labels"])) for n in names], dtype=object), dtype=STR)


def is_ramp(attrs) -> bool:
    """Case with a ramp of Ap ($Ap_end$ != $Ap_start$; experiment.is_ramp)."""
    a0, a1 = _attr_float(attrs, "$Ap_start$"), _attr_float(attrs, "$Ap_end$")
    return np.isfinite(a0) and np.isfinite(a1) and abs(a1 - a0) > 1e-9


def _first_piece_attrs(labels_h5: str) -> dict:
    """attrs of the first piece of a reference_dataset*.h5, whatever label group holds it (gray can be empty)."""
    with h5py.File(labels_h5, "r") as lf:
        for lab in lf:
            for case in lf[lab].values():
                for piece in case.values():
                    return dict(piece.attrs)
    return {}


def validate(ind_h5: str, labels_h5: str, out_h5: str, channel: str = "Axial_disp", reference_h5: str = None,
             gray: str = "ignore") -> dict:
    """Write out_h5 and return {run_name: [row dict per case]}. Each row has 'group': 'ramp' for a ramp of Ap whose
    truth changes from stable to unstable (scored apart: ramp_* metrics), 'global' otherwise (constant cases and
    ramps whose truth does not change: the usual metrics, ranking and ROC)."""
    intervals = read_intervals(labels_h5, channel)
    if not intervals:
        raise ValueError(f"no pieces of channel '{channel}' in {labels_h5}")
    if gray not in GRAY_MODES:
        raise ValueError(f"gray must be one of {GRAY_MODES}, not {gray!r}")
    summary = {}
    os.makedirs(os.path.dirname(os.path.abspath(out_h5)), exist_ok=True)
    if os.path.exists(out_h5):
        os.remove(out_h5)
    params = {k: v for k, v in _first_piece_attrs(labels_h5).items() if k.startswith("labeling_")}   # same for all
    with h5py.File(ind_h5, "r") as src, h5py.File(out_h5, "w") as out:
        out.attrs.update(schema="doe_validation_results/4", created=datetime.datetime.now().isoformat(timespec="seconds"),
                         indicator_results_file=os.path.basename(ind_h5), labels_file=os.path.basename(labels_h5),
                         channel=channel, gray_mode=gray)
        out.attrs.update(params)
        if reference_h5:
            out.attrs["reference_h5"] = os.path.basename(reference_h5)
            write_training(out, reference_h5)

        for case in sorted(k for k in src if k.startswith("case_")):
            if case not in intervals:
                print(f"  [skip] {case}: no labels")
                continue
            sg, cg = src[case], out.create_group(case)
            for k, v in sg.attrs.items():
                cg.attrs[k] = v
            for sig in ("Axial_disp", "Axial_vel", "Axial_acc"):   # signals, so the viewer shows them too
                if sig in sg:
                    src.copy(sg[sig], cg, name=sig)
            truth, t_start = case_truth(intervals[case])
            is_gray = truth == "unlabelled"
            if is_gray and gray != "ignore":   # the whole case counts as stable / unstable (--gray)
                truth = gray
                t_start = intervals[case][0][0] if gray == "unstable" else t_start
            ramp = is_ramp(sg.attrs)
            group = "ramp" if ramp and truth == "mixed" else "global"
            t_on = amplitude_onset(sg, params, t_start)
            # onset of the rule: constant cases (and ramps whose truth does not change) as before, the first sample
            # over the limit; a ramp that crosses, the start of its first unstable window (no theoretical time)
            onset = t_start if group == "ramp" else t_on
            kappa = np.nan if ramp else _attr_float(sg.attrs, "kappa")
            k0, k1 = (_attr_float(sg.attrs, "kappa_start"), _attr_float(sg.attrs, "kappa_end")) if ramp else (kappa, kappa)
            ap_mm, spin = _attr_float(sg.attrs, "$Ap_start$", "Ap_start") * 1e3, _attr_float(sg.attrs, "$spin_rate$", "spin_rate")
            ap_end = _attr_float(sg.attrs, "$Ap_end$") * 1e3 if ramp else ap_mm
            cg.attrs.update({"$kappa$": kappa, "$truth$": truth, "$gray$": int(is_gray), "$t_onset_amp$": t_on, "$group$": group,
                             "$t_onset$": onset if np.isfinite(onset) else t_start})   # NaN for a stable case
            if ramp:
                cg.attrs.update({"$kappa_start$": k0, "$kappa_end$": k1, "$Ap_end_mm$": ap_end})
            cg.create_dataset("truth_t0", data=[i[0] for i in intervals[case]])
            cg.create_dataset("truth_t1", data=[i[1] for i in intervals[case]])
            cg.create_dataset("truth_label", data=np.array([i[2] for i in intervals[case]], dtype=object), dtype=STR)

            for run in sg:
                rg = sg[run]
                if not isinstance(rg, h5py.Group) or "t" not in rg or "I_t" not in rg:
                    continue
                t, i_t = rg["t"][()], rg["I_t"][()]
                t_d = rg["t_d"][()] if "t_d" in rg else np.array([])
                pred, tw = pred_windows(t, t_d), truth_windows(t, intervals[case])
                if is_gray and gray != "ignore":
                    tw[:] = 0 if gray == "stable" else 1
                m = score(t, pred, tw, t_start, onset, group == "ramp")
                fin, fl = np.isfinite(i_t), pred == 1
                m.update(score_max=float(np.max(i_t[fin])) if fin.any() else np.nan,
                         score_min=float(np.min(i_t[fin])) if fin.any() else np.nan)
                og = cg.create_group(run)
                for k, v in rg.attrs.items():
                    og.attrs[k] = v
                og.create_dataset("t", data=t, compression="gzip")
                og.create_dataset("I_t", data=i_t, compression="gzip")
                if t_d.size:
                    og.create_dataset("t_d", data=t_d, compression="gzip")
                og.create_dataset("pred", data=pred, compression="gzip")
                og.create_dataset("truth_w", data=tw, compression="gzip")
                og.attrs.update(m)
                cg.attrs[f"$outcome_{run}$"] = m["outcome"]
                summary.setdefault(run, []).append(dict(
                    case=case, kappa=kappa, ap_mm=ap_mm, ap_end_mm=ap_end, kappa_start=k0, kappa_end=k1,
                    spin_rpm=spin, truth=truth, group=group, is_gray=float(is_gray), t_onset_amp=t_on, **m,
                    it_flag_sum=float(i_t[fl & fin].sum()), it_flag_n=int((fl & fin).sum()),     # for roc_direction
                    it_unflag_sum=float(i_t[~fl & fin].sum()), it_unflag_n=int((~fl & fin).sum())))

        sg, mg = out.create_group("summary"), out.create_group("metrics")
        for run, rows in summary.items():
            g = sg.create_group(run)
            for col in ("case", "truth", "outcome", "group"):
                g.create_dataset(col, data=np.array([r[col] for r in rows], dtype=object), dtype=STR)
            for col in SUMMARY_FLOATS:
                g.create_dataset(col, data=np.array([r[col] for r in rows], dtype=float))
        metrics = {run: run_metrics(rows) for run, rows in summary.items()}
        for run, (m, curves) in metrics.items():
            mg.create_group(run).attrs.update(m)
            for orient, (fpr, tpr, thr) in curves.items():
                rg = out.require_group(f"roc/{run}/{orient}")
                rg.create_dataset("fpr", data=fpr)
                rg.create_dataset("tpr", data=tpr)
                rg.create_dataset("thr", data=thr)
        order = rank_runs({r: m for r, (m, _) in metrics.items()})
        rk = out.create_group("ranking")
        rk.create_dataset("run", data=np.array(order, dtype=object), dtype=STR)
        rk.create_dataset("rank", data=np.arange(1, len(order) + 1))
        for col in ("balanced_accuracy", "MCC", "F1", "AUC", "TPR", "TNR", "accuracy"):
            rk.create_dataset(col, data=np.array([metrics[r][0][col] for r in order], dtype=float))
        pw = mcnemar_pairs(summary)
        if pw:
            g = out.create_group("pairwise")
            for col in ("run_a", "run_b"):
                g.create_dataset(col, data=np.array([q[col] for q in pw], dtype=object), dtype=STR)
            for col in ("a_only", "b_only"):
                g.create_dataset(col, data=np.array([q[col] for q in pw], dtype=int))
            g.create_dataset("p_value", data=np.array([q["p_value"] for q in pw], dtype=float))
    write_csv(os.path.splitext(out_h5)[0] + "_metrics.csv", {r: m for r, (m, _) in metrics.items()}, order)
    return summary


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson score interval for k successes out of n (nan, nan if n == 0)."""
    if n == 0:
        return np.nan, np.nan
    p, den = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / den, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(c - h, 0.0)), float(min(c + h, 1.0))


HIT = ("TP",)


def _mean(v) -> float:
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def case_metrics(rows) -> dict:
    """Per-case confusion for one indicator: counts, TPR/TNR/accuracy (+ Wilson 95%), balanced accuracy, F1, MCC and the
    median of the two detection times over the hits. TP / FN from the unstable cases (detected or not), TN / FP from
    the stable ones; there is no time tolerance."""
    c = {o: sum(r["outcome"] == o for r in rows) for o in ("TP", "FN", "TN", "FP")}
    tp, fn, tn, fp = c["TP"], c["FN"], c["TN"], c["FP"]
    n = dict(TP=tp, FN=fn, TN=tn, FP=fp)
    div = lambda a, b: a / b if b else float("nan")
    tpr, tnr = div(tp, tp + fn), div(tn, tn + fp)
    mcc_den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    med = lambda key: float(np.median(v)) if (v := [r[key] for r in rows if r["outcome"] in HIT and np.isfinite(r[key])]) \
        else float("nan")
    hits = lambda key: [r[key] for r in rows if r["outcome"] in HIT and np.isfinite(r.get(key, np.nan))]   # noqa: E731
    pct = lambda key, q: float(np.percentile(hits(key), q)) if hits(key) else float("nan")   # noqa: E731
    out = dict(**n, TPR=tpr, TNR=tnr, accuracy=div(tp + tn, sum(n.values())), balanced_accuracy=(tpr + tnr) / 2,
               F1=div(2 * tp, 2 * tp + fp + fn), MCC=float((tp * tn - fp * fn) / mcc_den) if mcc_den else float("nan"),
               median_delay_start_s=med("delay_start_s"), median_delay_onset_s=med("delay_onset_s"),
               delay_onset_p25_s=pct("delay_onset_s", 25), delay_onset_p75_s=pct("delay_onset_s", 75),
               median_t_ratio=pct("t_ratio", 50))
    for name, k, m in (("TPR", tp, tp + fn), ("TNR", tn, tn + fp), ("accuracy", tp + tn, sum(n.values()))):
        out[f"{name}_lo"], out[f"{name}_hi"] = wilson(k, m)
    af = [r.get("alarm_fraction", np.nan) for r in rows if r["outcome"] in ("TN", "FP")]
    pe = [r.get("persistence", np.nan) for r in rows if r["outcome"] in HIT]
    out.update(mean_alarm_fraction_stable=_mean(af), mean_persistence=_mean(pe))
    return out


def ramp_metrics(rows) -> dict:
    """Metrics of the ramps whose truth crosses from stable to unstable (group 'ramp'), apart from the global ones,
    ramp_n, the rates of detection (TP: first detection at or after the crossing), early alarm (first detection before
    it, in the stable stretch) and miss (+ Wilson 95%), the signed delay to the crossing of the truth (median, p25,
    p75; over the TP), the alarm fraction in the stable stretch of the ramps and the persistence in their unstable
    stretch. {} without such ramps."""
    rows = [r for r in rows if r.get("group") == "ramp"]
    if not rows:
        return {}
    n = len(rows)
    k = {o: sum(r["outcome"] == o for r in rows) for o in ("TP", "FA", "FN")}
    out = {"ramp_n": n}
    for name, cnt in (("detection", k["TP"]), ("early_alarm", k["FA"]), ("miss", k["FN"])):
        out[f"ramp_{name}_rate"] = cnt / n
        out[f"ramp_{name}_rate_lo"], out[f"ramp_{name}_rate_hi"] = wilson(cnt, n)
    d = [r["delay_onset_s"] for r in rows if r["outcome"] in HIT and np.isfinite(r["delay_onset_s"])]
    q = np.percentile(d, [25, 50, 75]) if d else [np.nan] * 3
    out.update(ramp_median_delay_s=float(q[1]), ramp_delay_p25_s=float(q[0]), ramp_delay_p75_s=float(q[2]),
               ramp_alarm_fraction_stable=_mean([r.get("alarm_fraction", np.nan) for r in rows]),
               ramp_persistence=_mean([r.get("persistence", np.nan) for r in rows if r["outcome"] in HIT]))
    return out


def auc_mw(pos, neg) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie) (Mann-Whitney); nan if either class is empty."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def roc_curve(pos, neg) -> tuple:
    """(fpr, tpr, thr): predict 'unstable' when score >= thr; starts at (0, 0) with thr = inf."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    thr = np.unique(np.concatenate([pos, neg]))[::-1]
    tpr = np.array([0.0] + [float((pos >= x).mean()) if pos.size else np.nan for x in thr])
    fpr = np.array([0.0] + [float((neg >= x).mean()) if neg.size else np.nan for x in thr])
    return fpr, tpr, np.concatenate([[np.inf], thr])


def hanley_mcneil(auc: float, n_pos: int, n_neg: int) -> tuple:
    """Approximate 95% interval of an AUC (Hanley & McNeil 1982); nan if a class is empty."""
    if not np.isfinite(auc) or n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc * auc) + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    h = 1.96 * np.sqrt(max(var, 0.0))
    return float(max(auc - h, 0.0)), float(min(auc + h, 1.0))


def roc_metrics(rows) -> tuple:
    """(dict of AUC fields, {orientation: (fpr, tpr, thr)}) for one indicator. See the module docstring."""
    cls = [(r["truth"], r["score_max"], r["score_min"]) for r in rows
           if r["truth"] in ("stable", "unstable") and np.isfinite(r["score_max"])]
    n_flag, n_unf = sum(r.get("it_flag_n", 0) for r in rows), sum(r.get("it_unflag_n", 0) for r in rows)
    if n_flag and n_unf:
        flagged = sum(r["it_flag_sum"] for r in rows) / n_flag
        direction = 1 if flagged > sum(r["it_unflag_sum"] for r in rows) / n_unf else -1
        source = "from the indicator's detections"
    else:
        direction, source = 1, "assumed high=chatter (no detections to tell)"
    sc = {"high": ([x for k, x, _ in cls if k == "unstable"], [x for k, x, _ in cls if k == "stable"]),
          "low": ([-x for k, _, x in cls if k == "unstable"], [-x for k, _, x in cls if k == "stable"])}
    a = {o: auc_mw(*v) for o, v in sc.items()}
    main = a["high"] if direction == 1 else a["low"]
    n_pos, n_neg = len(sc["high"][0]), len(sc["high"][1])
    lo, hi = hanley_mcneil(main, n_pos, n_neg)
    out = dict(AUC=main, AUC_lo=lo, AUC_hi=hi, AUC_if_high_is_chatter=a["high"], AUC_if_low_is_chatter=a["low"],
               roc_direction=direction, roc_direction_source=source, n_pos=n_pos, n_neg=n_neg)
    return out, {o: roc_curve(*v) for o, v in sc.items()}


def ci_bal_mcc(tp, fn, tn, fp, n: int = 4000, seed: int = 0) -> dict:
    """95% interval of balanced accuracy and MCC by simulation: TPR ~ Beta(tp+.5, fn+.5) and TNR ~ Beta(tn+.5, fp+.5)
    (Jeffreys, so it does not collapse at 0 or 1 like a plain bootstrap), the metric recomputed at each draw. Fixed seed:
    the same counts always give the same interval. NaN if a class is empty."""
    nan = float("nan")
    out = dict(balanced_accuracy_lo=nan, balanced_accuracy_hi=nan, MCC_lo=nan, MCC_hi=nan)
    npos, nneg = tp + fn, tn + fp
    if npos == 0 or nneg == 0:
        return out
    rng = np.random.default_rng(seed)
    tpr, tnr = rng.beta(tp + 0.5, fn + 0.5, n), rng.beta(tn + 0.5, fp + 0.5, n)
    a, b, c, d = tpr * npos, tnr * nneg, (1 - tnr) * nneg, (1 - tpr) * npos          # expected tp, tn, fp, fn
    mcc = (a * b - c * d) / np.sqrt((a + c) * (a + d) * (b + c) * (b + d))
    (out["balanced_accuracy_lo"], out["balanced_accuracy_hi"]) = np.percentile((tpr + tnr) / 2, [2.5, 97.5])
    (out["MCC_lo"], out["MCC_hi"]) = np.percentile(mcc, [2.5, 97.5])
    den = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    for k, est in (("balanced_accuracy", (tp / npos + tn / nneg) / 2), ("MCC", (tp * tn - fp * fn) / np.sqrt(den) if den else nan)):
        if np.isfinite(est):   # the interval always contains the estimate (at 11/11 the draws never reach exactly 1)
            out[k + "_lo"], out[k + "_hi"] = min(out[k + "_lo"], est), max(out[k + "_hi"], est)
    return {k: float(v) for k, v in out.items()}


def mcnemar_pairs(summary: dict) -> list:
    """Exact McNemar test between every pair of indicators over the cases both scored (constant cases): a_only = cases A
    got right and B wrong, b_only = the other way; p = two-sided binomial p-value of the discordant pairs (1.0 without
    any). It says whether a difference between two indicators is more than chance on THESE cases."""
    from math import comb
    ok = {run: {r["case"]: r["outcome"] in ("TP", "TN") for r in rows
                if r.get("group", "global") == "global" and r["outcome"] in ("TP", "FN", "TN", "FP")}
          for run, rows in summary.items()}
    out = []
    runs = list(ok)
    for i, ra in enumerate(runs):
        for rb in runs[i + 1:]:
            cases = ok[ra].keys() & ok[rb].keys()
            a = sum(ok[ra][c] and not ok[rb][c] for c in cases)
            b = sum(ok[rb][c] and not ok[ra][c] for c in cases)
            n = a + b
            p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(a, b) + 1)) / 2 ** n)
            out.append(dict(run_a=ra, run_b=rb, a_only=a, b_only=b, p_value=float(p)))
    return out


def gray_metrics(rows) -> dict:
    """Say how many gray cases there are, how many alarm and the metrics if all of them counted as stable (alarm -> FP,
    none -> TN) or as unstable (alarm -> TP, none -> FN), over the cases that are not gray: it does not depend on --gray."""
    isg = lambda r: r.get("is_gray") == 1.0 or r["truth"] == "unlabelled"   # noqa: E731  (original gray, any --gray mode)
    gray = [r for r in rows if isg(r)]
    alarm = [bool(np.isfinite(r["first_detection_t"])) for r in gray]
    scored = [r for r in rows if r["outcome"] in ("TP", "FN", "TN", "FP") and not isg(r)]
    out = dict(n_gray=len(gray), n_gray_alarm=sum(alarm), gray_alarm_rate=sum(alarm) / len(gray) if gray else float("nan"))
    for name, hit, miss in (("stable", "FP", "TN"), ("unstable", "TP", "FN")):
        m = case_metrics(scored + [dict(r, outcome=hit if a else miss) for r, a in zip(gray, alarm)])
        out.update({f"gray_as_{name}_{k}": m[k] for k in ("TPR", "TNR", "balanced_accuracy", "MCC")})
    return out


def run_metrics(rows) -> tuple:
    """(all metrics of one indicator as a dict, ROC curves). The global metrics, ranking and ROC use the 'global'
    rows only (constant cases and ramps whose truth does not change); the ramps that cross give the ramp_* ones."""
    glob = [r for r in rows if r.get("group", "global") == "global"]
    roc, curves = roc_metrics(glob)
    m = case_metrics(glob)
    return {**m, **ci_bal_mcc(m["TP"], m["FN"], m["TN"], m["FP"]), **roc, **gray_metrics(glob), **ramp_metrics(rows)}, curves


def rank_runs(metrics: dict) -> list:
    """Run names ordered best first by RANK_BY (balanced accuracy, then MCC, then AUC); NaN counts as worst."""
    return sorted(metrics, key=lambda r: tuple(-np.nan_to_num(metrics[r][k], nan=-1e9) for k in RANK_BY))


def write_csv(path: str, metrics: dict, order: list) -> None:
    import csv
    cols = sorted({k for m in metrics.values() for k in m})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "run", *cols])
        for i, r in enumerate(order, 1):
            w.writerow([i, r, *[metrics[r].get(c, "") for c in cols]])


def print_summary(summary: dict) -> None:
    metrics = {run: run_metrics(rows)[0] for run, rows in summary.items()}
    order = rank_runs(metrics)
    for run in order:
        m = metrics[run]
        ci = lambda k: f"{m[k]:.2f} [{m[k + '_lo']:.2f}-{m[k + '_hi']:.2f}]"
        chatter_is = "chatter" if m["roc_direction"] == 1 else "stable"
        print(f"{run}\n  cases TP={m['TP']} FN={m['FN']} TN={m['TN']} FP={m['FP']}"
              f" | TPR {ci('TPR')} TNR {ci('TNR')} acc {ci('accuracy')}"
              f"\n  balanced acc {ci('balanced_accuracy')} F1={m['F1']:.2f} MCC {ci('MCC')}"
              f" | AUC {ci('AUC')} (high I_t = {chatter_is}; {m['roc_direction_source']})"
              f"\n  alarm in stable cases={m['mean_alarm_fraction_stable']:.3f} of windows | persistence after detection="
              f"{m['mean_persistence']:.2f} | delay vs amplitude onset {m['median_delay_onset_s']:+.3f} s "
              f"[{m['delay_onset_p25_s']:+.3f}, {m['delay_onset_p75_s']:+.3f}] | t_det/t_onset {m['median_t_ratio']:.2f}")
        if m.get("n_gray"):
            print(f"  gray cases: {m['n_gray']}, {m['n_gray_alarm']} alarm | if all stable: TNR "
                  f"{m['gray_as_stable_TNR']:.2f}, bal.acc {m['gray_as_stable_balanced_accuracy']:.2f} | if all unstable: TPR "
                  f"{m['gray_as_unstable_TPR']:.2f}, bal.acc {m['gray_as_unstable_balanced_accuracy']:.2f}")
        if m.get("ramp_n"):
            rc = lambda k: f"{m[k]:.2f} [{m[k + '_lo']:.2f}-{m[k + '_hi']:.2f}]"   # noqa: E731
            print(f"  RAMPS that cross ({m['ramp_n']}): detected {rc('ramp_detection_rate')}, early alarm "
                  f"{rc('ramp_early_alarm_rate')}, missed "
                  f"{rc('ramp_miss_rate')} | delay to the crossing {m['ramp_median_delay_s']:+.3f} s "
                  f"[{m['ramp_delay_p25_s']:+.3f}, {m['ramp_delay_p75_s']:+.3f}] | alarm in their stable stretch "
                  f"{m['ramp_alarm_fraction_stable']:.3f} | persistence {m['ramp_persistence']:.2f}")
    print("\nRanking (balanced accuracy, MCC, AUC):")
    for i, run in enumerate(order, 1):
        m = metrics[run]
        print(f"  {i}. {run:40s} bal.acc={m['balanced_accuracy']:.2f} MCC={m['MCC']:.2f} AUC={m['AUC']:.2f}")


# ============================================================================== self-check
def _selftest():
    import tempfile
    t = np.arange(0.0, 10.0, 0.1)
    assert pred_windows(t, np.array([1.0, 5.04]))[[10, 50]].tolist() == [1, 1] and pred_windows(t, np.array([1.0, 5.04])).sum() == 2
    assert pred_windows(t, np.array([5.0 + 0.2])).sum() == 1 and pred_windows(t, np.array([])).sum() == 0
    iv = [(0.0, 4.95, "stable"), (4.95, 5.55, "gray"), (5.55, 10.0, "unstable")]
    tw = truth_windows(t, iv)
    assert tw[0] == 0 and tw[50] == -1 and tw[60] == 1 and case_truth(iv) == ("mixed", 5.55)
    # constant case: any detection is TP (no tolerance); the delay to the onset is only reported
    ok = score(t, pred_windows(t, t[tw == 1][:3]), tw, 5.55, 5.8)          # 5.6 s: 0.2 s before the onset 5.8 s
    assert ok["outcome"] == "TP" and ok["FP"] == 0 and ok["TN"] == (tw == 0).sum()
    assert abs(ok["delay_start_s"] - 0.05) < 1e-9 and abs(ok["delay_onset_s"] - (-0.2)) < 1e-9   # 5.6 - 5.55 | 5.6 - 5.8
    late = score(t, pred_windows(t, np.array([6.0, 6.1])), tw, 5.55, 5.8)
    assert late["outcome"] == "TP" and abs(late["delay_onset_s"] - 0.2) < 1e-9 and late["t_onset"] == 5.8
    early = score(t, pred_windows(t, np.array([1.0, 6.0])), tw, 5.55, ramp=True)   # ramp: alarm at 1.0 s, before the onset
    assert early["outcome"] == "FA" and early["FP"] == 1 and early["n_fp_windows"] == 1 and np.isnan(early["delay_onset_s"])
    assert early["t_onset"] == 5.55                                     # no amplitude onset: the unstable interval start
    assert abs(early["delay_det_s"] - (1.0 - 5.55)) < 1e-9 and np.isnan(miss_d := score(t, pred_windows(t, np.array([])), tw, 5.55)["delay_det_s"])
    assert score(t, pred_windows(t, np.array([1.0, 6.0])), tw, 5.55)["outcome"] == "TP"   # the same detections, constant case
    assert score(t, pred_windows(t, np.array([5.2])), tw, 5.55, ramp=True)["outcome"] == "FA"   # nothing tolerated on a ramp
    assert score(t, pred_windows(t, np.array([5.2])), tw, 5.55)["outcome"] == "TP"
    miss = score(t, pred_windows(t, np.array([])), tw, 5.55)
    assert miss["outcome"] == "FN" and miss["tpr"] == 0 and miss["tnr"] == 1 and np.isnan(miss["delay_start_s"])
    assert [detection_outcome(x, 10.0) for x in (10.0, 9.99, 9.5, np.nan)] == ["TP", "FA", "FA", "FN"]
    assert detection_outcome(3.0, np.nan) == "TP"                       # no onset known: any detection
    st = [(0.0, 10.0, "stable")]
    stw = truth_windows(t, st)
    assert score(t, pred_windows(t, np.array([3.0])), stw, np.nan)["outcome"] == "FP"
    assert score(t, pred_windows(t, np.array([])), stw, np.nan)["outcome"] == "TN"
    # amplitude onset: |y| ramps 0..1e-4 over 10 s, base = f_tooth 0.05 * 1e-3 = 5e-5, 40 % -> 2e-5 -> crossing at t = 2.0
    d = tempfile.mkdtemp()
    with h5py.File(os.path.join(d, "a.h5"), "w") as f:
        g = f.create_group("c")
        g.attrs["$f_tooth$"] = 0.05
        g["Axial_disp/time"], g["Axial_disp/values"] = t, -1e-5 * t
        P = {"labeling_lim_sup_pct": 40.0, "labeling_base_attr": "$f_tooth$", "labeling_base_scale": 1e-3, "labeling_signal": "Axial_disp"}
        assert abs(amplitude_onset(g, P, 0.0) - 2.1) < 1e-9, amplitude_onset(g, P, 0.0)   # first sample with |y| > 2e-5
        assert abs(amplitude_onset(g, {**P, "labeling_warmup": 5.0}, 0.0) - 5.0) < 1e-9
        assert np.isnan(amplitude_onset(g, P, np.nan)) and np.isnan(amplitude_onset(g, {**P, "labeling_signal": "nope"}, 0.0))
        assert np.isnan(amplitude_onset(g, {**P, "labeling_base_attr": "$nope$"}, 0.0))
    # Wilson (8/10 -> 0.490-0.943, the textbook values) and the derived metrics
    lo, hi = wilson(8, 10)
    assert abs(lo - 0.490) < 1e-3 and abs(hi - 0.943) < 1e-3 and np.isnan(wilson(0, 0)[0]) and wilson(10, 10)[1] == 1.0
    m = case_metrics([dict(outcome=o, delay_start_s=a, delay_onset_s=b) for o, a, b in
                      (("TP", 1.0, -0.5), ("TP", 3.0, 0.5), ("FN", np.nan, np.nan), ("TN", np.nan, np.nan),
                       ("FP", np.nan, np.nan), ("TN", np.nan, np.nan))])
    assert (m["TP"], m["FN"], m["TN"], m["FP"]) == (2, 1, 2, 1) and abs(m["TPR"] - 2 / 3) < 1e-12 and abs(m["TNR"] - 2 / 3) < 1e-12
    assert abs(m["accuracy"] - 4 / 6) < 1e-12 and abs(m["balanced_accuracy"] - 2 / 3) < 1e-12 and abs(m["F1"] - 2 / 3) < 1e-12
    assert abs(m["MCC"] - 1 / 3) < 1e-12 and m["median_delay_start_s"] == 2.0 and m["median_delay_onset_s"] == 0.0
    assert m["TPR_lo"] < m["TPR"] < m["TPR_hi"] and np.isnan(case_metrics([])["TPR"]) and np.isnan(case_metrics([])["MCC"])
    # the delay is the median over the hits (signed: negative = before the onset)
    m2 = case_metrics([dict(outcome=o, delay_start_s=1.0, delay_onset_s=d) for o, d in
                       (("TP", -0.3), ("TP", 0.5), ("FN", np.nan), ("TN", np.nan))])
    assert (m2["TP"], m2["FN"], m2["TN"], m2["FP"]) == (2, 1, 1, 0) and abs(m2["median_delay_onset_s"] - 0.1) < 1e-12
    assert not {"n_anticipated", "n_early_alarm", "early_alarm_rate"} & set(m2)
    # quartiles of the delay, median t_det / t_onset (hits only)
    mq = case_metrics([dict(outcome="TP", delay_start_s=1.0, delay_onset_s=dl, t_ratio=tr) for dl, tr in
                       ((-2.0, 0.5), (-1.0, 0.75), (0.0, 1.0), (1.0, 1.25))] + [dict(outcome="TN", delay_start_s=np.nan, delay_onset_s=np.nan)])
    assert abs(mq["delay_onset_p25_s"] + 1.25) < 1e-12 and abs(mq["delay_onset_p75_s"] - 0.25) < 1e-12 and abs(mq["median_t_ratio"] - 0.875) < 1e-12
    assert np.isnan(case_metrics([dict(outcome="TN", delay_start_s=np.nan, delay_onset_s=np.nan)])["median_t_ratio"])
    sc = score(t, pred_windows(t, np.array([2.0])), np.ones(t.size, dtype=np.int8), 0.0, 4.0)
    assert sc["outcome"] == "TP" and abs(sc["t_ratio"] - 0.5) < 1e-12 and np.isnan(score(t, pred_windows(t, np.array([2.0])), np.ones(t.size, dtype=np.int8), 0.0)["t_ratio"])
    # interval of balanced accuracy and MCC by simulation: contains the estimate, deterministic, NaN without a class
    ci = ci_bal_mcc(11, 0, 7, 1)
    assert ci["balanced_accuracy_lo"] < 0.9375 < ci["balanced_accuracy_hi"] <= 1.0 and ci["MCC_lo"] < 0.8956 < ci["MCC_hi"] <= 1.0
    assert ci == ci_bal_mcc(11, 0, 7, 1) and np.isnan(ci_bal_mcc(0, 0, 5, 5)["MCC_lo"])
    assert ci_bal_mcc(2, 2, 2, 2)["balanced_accuracy_hi"] - ci_bal_mcc(20, 20, 20, 20)["balanced_accuracy_hi"] > 0.1   # fewer cases: wider
    # exact McNemar: 5 cases only A gets right -> p = 2 / 2**5; ties -> 1.0
    rw = lambda outs: [dict(case=f"c{i}", group="global", outcome=o) for i, o in enumerate(outs)]   # noqa: E731
    pw = mcnemar_pairs({"a": rw(["TP"] * 5 + ["TN"]), "b": rw(["FN"] * 5 + ["TN"]), "c": rw(["TP"] * 5 + ["TN"])})
    assert [(q["run_a"], q["run_b"], q["a_only"], q["b_only"]) for q in pw] == [("a", "b", 5, 0), ("a", "c", 0, 0), ("b", "c", 0, 5)]
    assert abs(pw[0]["p_value"] - 2 / 32) < 1e-12 and pw[1]["p_value"] == 1.0 and abs(pw[2]["p_value"] - 2 / 32) < 1e-12
    assert mcnemar_pairs({"a": rw(["TP", "FN"]), "b": rw(["FN", "TP"])})[0]["p_value"] == 1.0
    # end to end on synthetic files
    ind, lab, out = (os.path.join(d, n) for n in ("ind.h5", "lab.h5", "out.h5"))
    with h5py.File(ind, "w") as f:
        # case_002-004: ramps 5 -> 15 mm whose truth turns unstable at 5.0 s: early alarm, late 0.5 s, late 1.0 s
        for name, kap, td in (("case_000", 0.6, []), ("case_001", 1.5, [6.0, 6.1, 7.0]), ("case_002", 0.0, [2.0, 6.0]),
                              ("case_003", 0.0, [5.5]), ("case_004", 0.0, [6.0]), ("case_005", 1.1, [7.0])):
            g = f.create_group(name)
            if kap:
                g.attrs.update({"$spin_rate$": 12000.0, "$Ap_start$": 0.005 * kap, "kappa": kap, "$f_tooth$": 0.05})
            else:
                g.attrs.update({"$spin_rate$": 12000.0, "$Ap_start$": 0.005, "$Ap_end$": 0.015, "kappa": 0.58,
                                "kappa_start": 0.58, "kappa_end": 1.74, "$f_tooth$": 0.05})
            g["Axial_disp/time"], g["Axial_disp/values"] = t, 1e-5 * t * (kap or 1.0)
            r = g.create_group("fake_run")
            r["t"], r["I_t"] = t, np.sin(t)
            if td:
                r["t_d"] = td
    with h5py.File(lab, "w") as f:
        f.create_group("aaa_empty")   # an empty label group first (as an empty 'gray'): the parameters are still found
        for label, case, t0, t1 in (("stable", "case_000", 0, 10), ("gray", "case_005", 0, 10), ("stable", "case_001", 0, 4.95),
                                    ("gray", "case_001", 4.95, 5.55), ("unstable", "case_001", 5.55, 10),
                                    *((lab_, c, a, b) for c in ("case_002", "case_003", "case_004")
                                      for lab_, a, b in (("stable", 0, 5.0), ("unstable", 5.0, 10)))):
            p = f.require_group(f"{label}/{case}").create_dataset("Axial_disp__000", data=[0.0])
            p.attrs.update(channel="Axial_disp", t0=t0, t1=t1, labeling_strategy="amplitude", labeling_lim_sup_pct=40.0,
                           labeling_base_attr="$f_tooth$", labeling_base_scale=1e-3, labeling_signal="Axial_disp")
    s = validate(ind, lab, out)
    assert [r["outcome"] for r in s["fake_run"]] == ["TN", "TP", "FA", "TP", "TP", "n/a"], s
    assert [r["group"] for r in s["fake_run"]] == ["global", "global", "ramp", "ramp", "ramp", "global"]
    with h5py.File(out, "r") as f:
        assert f.attrs["schema"].startswith("doe_validation") and f.attrs["labeling_strategy"] == "amplitude"
        assert "early_tol_s" not in f.attrs
        c1 = f["case_001"]
        assert c1.attrs["$truth$"] == "mixed" and "$zone$" not in c1.attrs and np.isnan(f["case_000"].attrs["$t_onset_amp$"])
        assert c1.attrs["$outcome_fake_run$"] == "TP" and list(c1["truth_label"].asstr()[()])[-1] == "unstable"
        assert abs(c1.attrs["$t_onset_amp$"] - 5.6) < 1e-9, c1.attrs["$t_onset_amp$"]   # 1e-5*1.5*t > 2e-5 -> t > 1.33 ... from 5.55 on: 5.6
        assert abs(c1["fake_run"].attrs["delay_onset_s"] - (6.0 - 5.6)) < 1e-9 and abs(c1["fake_run"].attrs["delay_start_s"] - 0.45) < 1e-9
        assert list(f["summary/fake_run/outcome"].asstr()[()])[:2] == ["TN", "TP"] and "zone" not in f["summary/fake_run"]
        mt = f["metrics/fake_run"].attrs   # global: the 2 constant cases only (the ramps that cross apart)
        assert mt["TP"] == 1 and mt["TN"] == 1 and mt["accuracy"] == 1.0 and mt["n_neg"] == 1   # ramps not in ROC
        assert c1["fake_run/pred"][()].sum() == 3
        # the ramps: onset = start of the first unstable window (5.0 s), no theoretical time; their metrics apart
        c3 = f["case_003"].attrs
        assert c3["$group$"] == "ramp" and c3["$t_onset$"] == 5.0 and np.isnan(c3["$kappa$"]) and c3["$kappa_end$"] == 1.74
        assert c3["$Ap_end_mm$"] == 15.0 and abs(f["case_003/fake_run"].attrs["delay_onset_s"] - 0.5) < 1e-9
        assert mt["ramp_n"] == 3 and abs(mt["ramp_detection_rate"] - 2 / 3) < 1e-12 and abs(mt["ramp_early_alarm_rate"] - 1 / 3) < 1e-12
        assert mt["ramp_miss_rate"] == 0.0 and "ramp_anticipated_rate" not in mt
        assert abs(mt["ramp_median_delay_s"] - 0.75) < 1e-9 and mt["ramp_delay_p25_s"] < mt["ramp_median_delay_s"] < mt["ramp_delay_p75_s"]
        assert mt["ramp_detection_rate_lo"] < 2 / 3 < mt["ramp_detection_rate_hi"]
        # 50 stable windows per ramp (5.0 s is unstable); one detection (2.0 s) in one of them
        assert abs(mt["ramp_alarm_fraction_stable"] - (1 / 50 + 0 + 0) / 3) < 1e-9, mt["ramp_alarm_fraction_stable"]
        assert list(f["summary/fake_run/group"].asstr()[()]) == ["global", "global", "ramp", "ramp", "ramp", "global"]
        assert f.attrs["gray_mode"] == "ignore" and f["case_005"].attrs["$gray$"] == 1 and f["case_005"].attrs["$truth$"] == "unlabelled"
        assert list(f["summary/fake_run/is_gray"][()]) == [0, 0, 0, 0, 0, 1]
        assert (mt["n_gray"], mt["n_gray_alarm"], mt["gray_alarm_rate"]) == (1, 1, 1.0)      # ignore: left out of the counts
        assert mt["balanced_accuracy_lo"] <= mt["balanced_accuracy"] <= mt["balanced_accuracy_hi"] and "pairwise" not in f   # one run
        assert abs(f["summary/fake_run/t_ratio"][1] - 6.0 / 5.6) < 1e-9, f["summary/fake_run/t_ratio"][()]   # case_001: t_det 6.0, onset 5.6
        assert mt["gray_as_stable_TNR"] == 0.5 and mt["gray_as_unstable_TPR"] == 1.0 and mt["gray_as_unstable_TNR"] == 1.0
        assert abs(f["summary/fake_run/ap_end_mm"][2] - 15.0) < 1e-9 and f["summary/fake_run/t_onset"][2] == 5.0
    # the three gray modes on the same files: ignore (above), stable (pessimistic), unstable (optimistic)
    assert gray_suffix("ignore") == "" and gray_suffix("stable") == "_gray-stable"
    for mode, outcome, tp, fn, tn, fp, npos, nneg in (("stable", "FP", 1, 0, 1, 1, 0, 2), ("unstable", "TP", 2, 0, 1, 0, 1, 1)):
        o = os.path.join(d, f"out{gray_suffix(mode)}.h5")
        sm = validate(ind, lab, o, gray=mode)
        assert [r["outcome"] for r in sm["fake_run"]][-1] == outcome and [r["truth"] for r in sm["fake_run"]][-1] == mode
        with h5py.File(o, "r") as f:
            m3 = f["metrics/fake_run"].attrs
            assert f.attrs["gray_mode"] == mode and f["case_005"].attrs["$gray$"] == 1 and f["case_005"].attrs["$truth$"] == mode
            assert (m3["TP"], m3["FN"], m3["TN"], m3["FP"]) == (tp, fn, tn, fp) and (m3["n_pos"], m3["n_neg"]) == (npos, nneg), dict(m3)
            assert m3["n_gray"] == 1 and m3["gray_as_stable_TNR"] == 0.5 and m3["gray_as_unstable_TPR"] == 1.0   # same bounds, any mode
            assert f["case_005/fake_run"].attrs["outcome"] == outcome
    try:
        validate(ind, lab, os.path.join(d, "x.h5"), gray="maybe")
        raise SystemExit("gray='maybe' should fail")
    except ValueError:
        pass
    # gray cases: left out of the counts, reported apart (bounds if all stable / all unstable)
    gr = [dict(truth=t, outcome=o, first_detection_t=d, delay_start_s=np.nan, delay_onset_s=np.nan) for t, o, d in
          (("unstable", "TP", 1.0), ("unstable", "TP", 2.0), ("stable", "TN", np.nan), ("stable", "FP", 3.0),
           ("unlabelled", "n/a", 4.0), ("unlabelled", "n/a", np.nan))]
    g = gray_metrics(gr)
    assert (g["n_gray"], g["n_gray_alarm"], g["gray_alarm_rate"]) == (2, 1, 0.5)
    assert g["gray_as_stable_TNR"] == 0.5 and g["gray_as_stable_TPR"] == 1.0          # alarm -> FP, none -> TN: TN 2, FP 2
    assert g["gray_as_unstable_TPR"] == 0.75 and g["gray_as_unstable_TNR"] == 0.5      # alarm -> TP, none -> FN: TP 3, FN 1
    assert case_metrics([r for r in gr if r["truth"] != "unlabelled"])["TN"] == 1      # the headline counts do not move
    assert gray_metrics(gr[:4])["n_gray"] == 0 and np.isnan(gray_metrics(gr[:4])["gray_alarm_rate"])
    # alarm quality: 3 flagged windows from 5.6 on, 44 windows from 5.6 to 9.9 inside the unstable interval
    assert ok["alarm_fraction"] == 0.0 and abs(ok["persistence"] - 3 / 44) < 1e-12
    fa = score(t, pred_windows(t, np.array([3.0])), stw, np.nan)
    assert abs(fa["alarm_fraction"] - 1 / 100) < 1e-12 and np.isnan(fa["persistence"])
    # AUC / ROC / Hanley-McNeil
    assert abs(auc_mw([0.8, 0.6, 0.4], [0.5, 0.3]) - 5 / 6) < 1e-12 and auc_mw([3, 4], [1, 2]) == 1.0
    assert auc_mw([1, 2], [3, 4]) == 0.0 and auc_mw([1.0], [1.0]) == 0.5 and np.isnan(auc_mw([], [1.0]))
    fpr, tpr, thr = roc_curve([0.8, 0.6, 0.4], [0.5, 0.3])
    assert (fpr[0], tpr[0], fpr[-1], tpr[-1]) == (0.0, 0.0, 1.0, 1.0) and np.isinf(thr[0]) and np.all(np.diff(tpr) >= 0)
    lo, hi = hanley_mcneil(0.8, 10, 10)
    assert 0.5 < lo < 0.8 < hi <= 1.0 and np.isnan(hanley_mcneil(0.8, 0, 5)[0])
    mk = lambda truth, mx, mn, fs, fn_, us, un: dict(truth=truth, score_max=mx, score_min=mn, it_flag_sum=fs, it_flag_n=fn_,
                                                    it_unflag_sum=us, it_unflag_n=un, outcome="TP")
    hi_rows = [mk("unstable", 9, 5, 18, 2, 10, 10), mk("unstable", 8, 6, 16, 2, 10, 10), mk("stable", 2, 0, 0, 0, 10, 10),
               mk("stable", 3, 1, 0, 0, 10, 10)]       # flagged windows have high I_t -> chatter is high; separable
    r_hi, _ = roc_metrics(hi_rows)
    assert r_hi["roc_direction"] == 1 and r_hi["AUC"] == 1.0 and r_hi["AUC_if_low_is_chatter"] == 0.0 and (r_hi["n_pos"], r_hi["n_neg"]) == (2, 2)
    lo_rows = [mk("unstable", 0, -9, -18, 2, -10, 10), mk("unstable", 1, -8, -16, 2, -10, 10), mk("stable", 4, 2, 0, 0, 10, 10),
               mk("stable", 5, 3, 0, 0, 10, 10)]        # flagged windows have low I_t -> chatter is low
    r_lo, _ = roc_metrics(lo_rows)
    assert r_lo["roc_direction"] == -1 and r_lo["AUC"] == 1.0 and r_lo["AUC_if_high_is_chatter"] == 0.0
    assert roc_metrics([mk("stable", 1, 1, 0, 0, 5, 5)])[0]["roc_direction"] == 1       # no detections anywhere -> assumed
    # ranking: balanced accuracy first, MCC breaks ties, NaN last
    assert rank_runs({"a": dict(balanced_accuracy=.8, MCC=.5, AUC=.9), "b": dict(balanced_accuracy=.9, MCC=.1, AUC=.1),
                      "c": dict(balanced_accuracy=.8, MCC=.7, AUC=.2), "d": dict(balanced_accuracy=np.nan, MCC=1, AUC=1)}) == ["b", "c", "a", "d"]
    with h5py.File(out, "r") as f:   # the end-to-end file written above
        assert list(f["ranking/run"].asstr()[()]) == ["fake_run"] and "high" in f["roc/fake_run"] and "AUC" in f["metrics/fake_run"].attrs
        assert f["case_001/fake_run"].attrs["persistence"] > 0 and f["case_000/fake_run"].attrs["alarm_fraction"] == 0.0
    csv_rows = open(os.path.splitext(out)[0] + "_metrics.csv", encoding="utf-8").read().splitlines()
    assert csv_rows[0].startswith("rank,run,") and csv_rows[1].startswith("1,fake_run,") and len(csv_rows) == 2
    print("selftest OK")


def main():
    p = argparse.ArgumentParser(description="Score the indicators of a validation DOE against its amplitude labels.")
    p.add_argument("--ind_results", metavar="PATH", help="doe_indicator_results.h5 of the validation DOE")
    p.add_argument("--labels", metavar="PATH", help="reference_dataset*.h5 built on the validation doe_results.h5")
    p.add_argument("--reference", metavar="PATH", default=None, help="training reference_dataset*.h5 (stored in /training)")
    p.add_argument("--out", metavar="PATH", default=None,
                   help="default: doe_validation_results[_gray-<mode>].h5 next to --ind_results")
    p.add_argument("--gray", choices=GRAY_MODES, default="ignore",
                   help="gray cases: ignore (not scored, default) | stable (pessimistic) | unstable (optimistic)")
    p.add_argument("--channel", default="Axial_disp", help="channel whose pieces give the intervals (default Axial_disp)")
    p.add_argument("--early-tol", type=float, default=None, help=argparse.SUPPRESS)   # DEPRECATED: accepted, ignored
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return _selftest()
    if not (a.ind_results and a.labels):
        p.error("--ind_results and --labels are required")
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.ind_results)),
                                    f"doe_validation_results{gray_suffix(a.gray)}.h5")
    print_summary(validate(a.ind_results, a.labels, out, a.channel, a.reference, a.gray))
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
