#!/usr/bin/env python
# coding: utf-8
"""validate_noise.py — Score the indicators run on NOISY copies of the validation cases against the truth of the
CLEAN cases (DOE_utils/PLAN_noise_validation.md §4.3, §5.3).

Inputs
  --noise_ind  doe_noise_indicator_results.h5 (doe_indicators.py on doe_noise_multi_results.h5): one group per noisy
               copy, snr_{SNR:06.2f}__{case}__r{K:02d}, attrs snr_db / case_source / realization; root attrs of the
               noise file (snr_ref_case, snr_levels, ...)
  --clean      doe_validation_results*.h5 of the same experiment (validate_indicators.py): the truth intervals of
               every case, $t_onset_amp$, gray_mode and the clean metrics
Output  doe_noise_validation_results[_gray-<mode>].h5 (+ <out>_by_snr.csv)
  attrs           schema doe_noise_validation_results/1, gray_mode, clean_results, noise_indicator_results,
                  breakdown_drop, and the root attrs of the noise file
  /summary/<run>  one row per noisy copy: copy, case, snr_db, realization, truth, is_gray, outcome, kappa,
                  first_detection_t, t_ratio, delay_det_s, score_max, alarm_fraction
  /metrics/<run>  1-D datasets aligned, one row per (snr_db, realization): METRICS  (the clean file keeps the same
                  names as attrs: same names, another format)
  /by_snr/<run>   1-D datasets aligned, one row per level (clean to noisy): snr_db and <m>_mean, <m>_min, <m>_max
                  over the realizations; attr snr_breakdown_db
  /clean/<run>    attrs: the clean metrics of that indicator over the SAME cases as the noisy copies (their clean
                  runs rescored with the same rules and gray mode: the comparable "no noise" baseline), and <m>_all =
                  the clean metric over every case of the clean file (NaN when the indicator is not in it)
The truth is the clean case's (the noise does not change it), with the gray mode of the clean validation. Ramps are
left out (deferred). Each realization is a full validation of the cases at one level: the realizations are summarised
(mean / min / max), never pooled as cases. snr_breakdown_db = highest SNR whose mean balanced accuracy falls more than
BREAKDOWN_DROP below the clean one of the same cases (NaN if it never does or there is no clean value).

Usage (entorno_CAMP10 Python):
    python validate_noise.py --noise_ind X/doe_noise_indicator_results.h5 --clean X/doe_validation_results.h5
    python validate_noise.py --selftest
"""
import argparse
import csv
import datetime
import os
import sys
import warnings

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_indicators as vi  # noqa: E402

METRICS = ("TP", "FN", "TN", "FP", "TPR", "TNR", "balanced_accuracy", "MCC", "AUC", "mean_alarm_fraction_stable",
           "median_t_ratio", "n_gray")
BREAKDOWN_DROP = 0.05
SUMMARY_STR = ("copy", "case", "truth", "outcome")
SUMMARY_FLOATS = ("snr_db", "realization", "is_gray", "kappa", "first_detection_t", "t_ratio", "delay_det_s",
                  "score_max", "alarm_fraction")
NOISE_ATTRS = ("noise_layout", "snr_mode", "snr_ref_case", "snr_levels", "realizations", "seed", "cases")
NAN = float("nan")


def load_clean(path: str) -> tuple:
    """(gray_mode, {case: {intervals, t_onset_amp, kappa, ramp}}, {run: clean metrics}) of a clean validation file."""
    with h5py.File(path, "r") as f:
        cases = {}
        for c, g in f.items():
            if c.startswith("case_") and "truth_t0" in g:
                iv = sorted(zip(map(float, g["truth_t0"][()]), map(float, g["truth_t1"][()]), g["truth_label"].asstr()[()]))
                cases[c] = dict(intervals=iv, t_onset_amp=float(g.attrs.get("$t_onset_amp$", NAN)),
                                kappa=float(g.attrs.get("$kappa$", NAN)), ramp=str(g.attrs.get("$group$", "global")) == "ramp")
        metrics = {r: dict(g.attrs) for r, g in f["metrics"].items()} if "metrics" in f else {}
        return str(f.attrs.get("gray_mode", "ignore")), cases, metrics


def _score(rg, c: dict, gray: str) -> dict:
    """Row of one indicator run (t, I_t, t_d) of a case scored against its clean truth c (same rules for the noisy
    copies and the clean baseline); None if the group is not a run."""
    if not isinstance(rg, h5py.Group) or "t" not in rg or "I_t" not in rg:
        return None
    truth, t_start, is_gray = vi.effective_truth(c["intervals"], gray)
    t_d = rg["t_d"][()] if "t_d" in rg else np.array([])
    m, _, _, roc_sums = vi.score_run(rg["t"][()], rg["I_t"][()], t_d, c["intervals"], t_start, c["t_onset_amp"], False,
                                     gray if is_gray and gray != "ignore" else None)
    return dict(truth=truth, is_gray=float(is_gray), group="global", kappa=c["kappa"], **m, **roc_sums)


def clean_on(clean: str, gray: str, cases: dict, subset) -> dict:
    """{run: clean metrics over `subset`}: the clean runs of those cases, rescored like the noisy copies."""
    rows = {}
    with h5py.File(clean, "r") as f:
        for case in sorted(subset):
            for run, rg in f[case].items():
                row = _score(rg, cases[case], gray)
                if row is not None:
                    rows.setdefault(run, []).append(dict(row, case=case))
    return {run: vi.run_metrics(rr)[0] for run, rr in rows.items()}


def score_noise(noise_ind: str, gray: str, cases: dict, realizations=None) -> tuple:
    """({root noise attrs}, {run: [row per noisy copy]}): every copy scored against its clean case's truth
    (only the realizations K listed in `realizations`, if given)."""
    rows, skipped = {}, set()
    with h5py.File(noise_ind, "r") as f:
        nattrs = {k: f.attrs[k] for k in f.attrs if k in NOISE_ATTRS or k.startswith("snr_ref_power_")}
        for name in sorted(f):
            g = f[name]
            case = str(g.attrs.get("case_source", ""))
            c = cases.get(case)
            if c is None or c["ramp"] or "snr_db" not in g.attrs:
                skipped.add(case or name)
                continue
            if realizations is not None and int(g.attrs.get("realization", 0)) not in realizations:
                continue
            for run, rg in g.items():
                row = _score(rg, c, gray)
                if row is not None:
                    rows.setdefault(run, []).append(dict(row, copy=name, case=case, snr_db=float(g.attrs["snr_db"]),
                                                         realization=float(g.attrs.get("realization", 0))))
    if skipped:
        print(f"  [skip] not a constant case of the clean validation: {sorted(skipped)}")
    return nattrs, rows


def level_table(rows: list) -> list:
    """[(snr_db, realization, {metric: value})], clean to noisy: vi.run_metrics over the rows of each level x realization."""
    keys = sorted({(r["snr_db"], r["realization"]) for r in rows}, key=lambda x: (-x[0], x[1]))
    out = []
    for snr, k in keys:
        m = vi.run_metrics([r for r in rows if r["snr_db"] == snr and r["realization"] == k])[0]
        out.append((snr, k, {name: float(m.get(name, NAN)) for name in METRICS}))
    return out


def by_snr(table: list, clean_bal: float) -> tuple:
    """({snr_db, <m>_mean, <m>_min, <m>_max}, snr_breakdown_db) over the realizations of each level."""
    levels = sorted({s for s, _, _ in table}, reverse=True)
    out = {"snr_db": levels}
    with warnings.catch_warnings():   # a metric NaN in every realization (e.g. MCC without TN) stays NaN, quietly
        warnings.simplefilter("ignore", RuntimeWarning)
        for name in METRICS:
            vals = [np.array([d[name] for s, _, d in table if s == lv], float) for lv in levels]
            for stat, fn in (("mean", np.nanmean), ("min", np.nanmin), ("max", np.nanmax)):
                out[f"{name}_{stat}"] = [float(fn(v)) for v in vals]
    bal = out["balanced_accuracy_mean"]
    brk = next((lv for lv, b in zip(levels, bal) if np.isfinite(clean_bal) and b < clean_bal - BREAKDOWN_DROP), NAN)
    return out, float(brk)


def validate_noise(noise_ind: str, clean: str, out_h5: str = None, realizations=None) -> dict:
    """Write out_h5 (+ _by_snr.csv) and return {run: (level table, by_snr dict, snr_breakdown_db, clean bal. acc.)}.
    `realizations`: indices K to score (None = every realization in noise_ind)."""
    gray, cases, clean_m = load_clean(clean)
    nattrs, rows = score_noise(noise_ind, gray, cases, None if realizations is None else {int(k) for k in realizations})
    if not rows:
        raise ValueError(f"no noisy copy of a constant case of {clean} in {noise_ind}"
                         + ("" if realizations is None else f" for realizations {list(realizations)}"))
    scored = sorted({int(r["realization"]) for rr in rows.values() for r in rr})
    clean_sub = clean_on(clean, gray, cases, {r["case"] for rr in rows.values() for r in rr})
    out_h5 = out_h5 or os.path.join(os.path.dirname(os.path.abspath(noise_ind)),
                                    f"doe_noise_validation_results{vi.gray_suffix(gray)}.h5")
    os.makedirs(os.path.dirname(os.path.abspath(out_h5)), exist_ok=True)
    result = {}
    with h5py.File(out_h5, "w") as out:
        out.attrs.update(schema="doe_noise_validation_results/1", created=datetime.datetime.now().isoformat(timespec="seconds"),
                         gray_mode=gray, clean_results=os.path.basename(clean),
                         noise_indicator_results=os.path.basename(noise_ind), breakdown_drop=BREAKDOWN_DROP,
                         realizations_scored=np.array(scored), **nattrs)
        for run, rr in rows.items():
            g = out.create_group(f"summary/{run}")
            for col in SUMMARY_STR:
                g.create_dataset(col, data=np.array([r[col] for r in rr], dtype=object), dtype=vi.STR)
            for col in SUMMARY_FLOATS:
                g.create_dataset(col, data=np.array([r.get(col, NAN) for r in rr], float))
            table = level_table(rr)
            mg = out.create_group(f"metrics/{run}")
            mg.create_dataset("snr_db", data=np.array([s for s, _, _ in table], float))
            mg.create_dataset("realization", data=np.array([k for _, k, _ in table], float))
            for name in METRICS:
                mg.create_dataset(name, data=np.array([d[name] for _, _, d in table], float))
            cm = {k: v for k, v in clean_sub.get(run, {}).items() if np.isscalar(v)} or {name: NAN for name in METRICS}
            cm.update({f"{name}_all": float(clean_m.get(run, {}).get(name, NAN)) for name in METRICS})
            bs, brk = by_snr(table, float(cm.get("balanced_accuracy", NAN)))
            bg = out.create_group(f"by_snr/{run}")
            for k, v in bs.items():
                bg.create_dataset(k, data=np.array(v, float))
            bg.attrs["snr_breakdown_db"] = brk
            out.create_group(f"clean/{run}").attrs.update(cm)
            result[run] = (table, bs, brk, float(cm.get("balanced_accuracy", NAN)))
    with open(os.path.splitext(out_h5)[0] + "_by_snr.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = [k for k in next(iter(result.values()))[1] if k != "snr_db"]
        w.writerow(["run", "snr_db", *cols, "snr_breakdown_db"])
        for run, (_, bs, brk, _) in result.items():
            for i, lv in enumerate(bs["snr_db"]):
                w.writerow([run, lv, *[bs[c][i] for c in cols], brk])
    return result


def print_summary(result: dict) -> None:
    for run, (_, bs, brk, b0) in result.items():
        levels = " | ".join(f"{lv:g} dB {m:.2f} [{lo:.2f}-{hi:.2f}]" for lv, m, lo, hi in
                            zip(bs["snr_db"], bs["balanced_accuracy_mean"], bs["balanced_accuracy_min"], bs["balanced_accuracy_max"]))
        print(f"{run}\n  balanced accuracy: clean {b0:.2f} | {levels}\n  breakdown SNR: {brk:g} dB")


# ============================================================================== self-check
def _selftest():
    import tempfile
    d = tempfile.mkdtemp()
    t = np.arange(0.0, 10.0, 0.1)
    ind, lab, ref = (os.path.join(d, n) for n in ("ind.h5", "lab.h5", "ref.h5"))
    # clean: case_000 stable (FALSE alarm 3.0 s), case_001 unstable (alarm 6.0 s), case_002 gray (alarm 7.0 s),
    # case_003 stable (no alarm, and NO noisy copy): over every case bal. acc. 0.75, over the noisy cases 0.5
    with h5py.File(ind, "w") as f, h5py.File(lab, "w") as fl:
        for i, (label, kap, td) in enumerate((("stable", 0.6, [3.0]), ("unstable", 1.5, [6.0]), ("gray", 1.05, [7.0]),
                                               ("stable", 0.7, []))):
            g = f.create_group(f"case_{i:03d}")
            g.attrs.update({"$spin_rate$": 12000.0, "$Ap_start$": 0.005 * kap, "kappa": kap, "$f_tooth$": 0.05})
            g["Axial_disp/time"], g["Axial_disp/values"] = t, 1e-5 * t * kap
            r = g.create_group("ind_a")
            r["t"], r["I_t"] = t, np.sin(t) + kap
            if td:
                r["t_d"] = td
            p = fl.require_group(f"{label}/case_{i:03d}").create_dataset("Axial_disp__000", data=[0.0])
            p.attrs.update(channel="Axial_disp", t0=0.0, t1=10.0, labeling_strategy="amplitude", labeling_lim_sup_pct=40.0,
                           labeling_base_attr="$f_tooth$", labeling_base_scale=1e-3, labeling_signal="Axial_disp")
    clean = os.path.join(d, "clean.h5")
    vi.validate(ind, lab, clean)
    # noisy copies: 40 dB like the clean one; 10 dB r00 a false alarm on the stable case, r01 the unstable one missed
    det = {(40.0, 0): {"case_001": [6.0], "case_002": [7.0]}, (40.0, 1): {"case_001": [6.2], "case_002": [7.0]},
           (10.0, 0): {"case_000": [3.0], "case_001": [5.0], "case_002": [2.0]}, (10.0, 1): {"case_002": [2.0]}}
    nind = os.path.join(d, "noise_ind.h5")
    with h5py.File(nind, "w") as f:
        f.attrs.update(noise_layout="multi", snr_mode="absolute", snr_ref_case="case_001", snr_levels=[40.0, 10.0], realizations=2)
        for (snr, k), tds in det.items():
            for c in ("case_000", "case_001", "case_002", "case_009"):   # case_009: not in the clean file -> skipped
                g = f.create_group(f"snr_{snr:06.2f}__{c}__r{k:02d}")
                g.attrs.update(snr_db=snr, case_source=c, realization=k)
                for run in ("ind_a", "ind_b"):   # ind_b: not in the clean validation -> /clean NaN
                    r = g.create_group(run)
                    r["t"], r["I_t"] = t, np.sin(t) + (2.0 if tds.get(c) else 0.0)
                    if tds.get(c):
                        r["t_d"] = tds[c]
    out = os.path.join(d, "nv.h5")
    res = validate_noise(nind, clean, out)
    table, bs, brk, b0 = res["ind_a"]
    assert [(s, k) for s, k, _ in table] == [(40.0, 0), (40.0, 1), (10.0, 0), (10.0, 1)]
    m = {(s, k): dd for s, k, dd in table}
    assert (m[40.0, 0]["TP"], m[40.0, 0]["TN"], m[40.0, 0]["FP"], m[40.0, 0]["FN"]) == (1, 1, 0, 0) and m[40.0, 0]["n_gray"] == 1
    assert (m[10.0, 0]["TP"], m[10.0, 0]["FP"]) == (1, 1) and (m[10.0, 1]["FN"], m[10.0, 1]["TN"]) == (1, 1)
    assert bs["snr_db"] == [40.0, 10.0] and bs["balanced_accuracy_mean"] == [1.0, 0.5]
    assert (bs["TPR_mean"][1], bs["TPR_min"][1], bs["TPR_max"][1]) == (0.5, 0.0, 1.0)   # realizations summarised, not pooled
    with h5py.File(out, "r") as f:
        assert list(f.attrs["realizations_scored"]) == [0, 1] and f.attrs["realizations"] == 2
    # --realizations 0: only r00 scored, the rest of the file ignored; the attr says which
    out0 = os.path.join(d, "nv0.h5")
    r0 = validate_noise(nind, clean, out0, realizations=[0])
    assert [(s_, k) for s_, k, _ in r0["ind_a"][0]] == [(40.0, 0), (10.0, 0)] and r0["ind_a"][1]["TPR_min"] == r0["ind_a"][1]["TPR_max"]
    with h5py.File(out0, "r") as f:
        assert list(f.attrs["realizations_scored"]) == [0] and f.attrs["realizations"] == 2 and len(f["summary/ind_a/copy"]) == 6
    try:
        validate_noise(nind, clean, out0, realizations=[7])
        raise SystemExit("realization 7 should have failed")
    except ValueError:
        pass
    # baseline on the SAME cases: clean 0.5 there (0.75 over every case) -> no false breakdown at 10 dB (0.5)
    assert b0 == 0.5 and np.isnan(brk)
    assert np.isnan(res["ind_b"][2])                    # no clean value -> no breakdown
    with h5py.File(out, "r") as f:
        assert f.attrs["schema"] == "doe_noise_validation_results/1" and f.attrs["gray_mode"] == "ignore"
        assert f.attrs["snr_ref_case"] == "case_001" and f.attrs["clean_results"] == "clean.h5"
        s = f["summary/ind_a"]
        assert len(s["copy"]) == 12 and "case_009" not in set(s["case"].asstr()[()])
        gray_rows = s["is_gray"][()] == 1
        assert set(s["outcome"].asstr()[()][gray_rows]) == {"n/a"} and set(s["truth"].asstr()[()][gray_rows]) == {"unlabelled"}
        assert list(f["metrics/ind_a/snr_db"][()]) == [40.0, 40.0, 10.0, 10.0] and f["metrics/ind_a/TP"].ndim == 1
        ca = f["clean/ind_a"].attrs
        assert np.isnan(f["by_snr/ind_a"].attrs["snr_breakdown_db"]) and ca["balanced_accuracy"] == 0.5
        assert ca["balanced_accuracy_all"] == 0.75 and (ca["TP"], ca["FP"], ca["FP_all"]) == (1, 1, 1) and ca["TN_all"] == 1
        assert np.isnan(f["clean/ind_b"].attrs["balanced_accuracy"]) and np.isnan(f["clean/ind_b"].attrs["balanced_accuracy_all"])
    rows_csv = open(os.path.splitext(out)[0] + "_by_snr.csv", encoding="utf-8").read().splitlines()
    assert rows_csv[0].startswith("run,snr_db,") and len(rows_csv) == 1 + 2 * 2
    # the gray mode comes from the clean file: gray = stable -> the gray case's alarms are false alarms
    clean_s = os.path.join(d, "clean_gs.h5")
    vi.validate(ind, lab, clean_s, gray="stable")
    res_s = validate_noise(nind, clean_s)
    ms = {(s_, k): dd for s_, k, dd in res_s["ind_a"][0]}
    assert (ms[40.0, 0]["TP"], ms[40.0, 0]["TN"], ms[40.0, 0]["FP"]) == (1, 1, 1) and ms[40.0, 0]["TNR"] == 0.5
    assert os.path.isfile(os.path.join(d, "doe_noise_validation_results_gray-stable.h5"))   # default name follows the mode
    print("validate_noise selftest OK")


def main():
    p = argparse.ArgumentParser(description="Score the indicators on noisy copies against the truth of the clean cases.")
    p.add_argument("--noise_ind", metavar="PATH", help="doe_noise_indicator_results.h5 (multi-case noise)")
    p.add_argument("--clean", metavar="PATH", help="doe_validation_results*.h5 of the same experiment")
    p.add_argument("--out", metavar="PATH", default=None,
                   help="default: doe_noise_validation_results[_gray-<mode>].h5 next to --noise_ind")
    p.add_argument("--realizations", nargs="+", type=int, default=None, metavar="K",
                   help="realization indices to score (suffix __rKK); default: all in --noise_ind. Written to the attr "
                        "realizations_scored; attr realizations stays the count of the noise file")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return _selftest()
    if not (a.noise_ind and a.clean):
        p.error("--noise_ind and --clean are required")
    print_summary(validate_noise(a.noise_ind, a.clean, a.out, a.realizations))


if __name__ == "__main__":
    main()
