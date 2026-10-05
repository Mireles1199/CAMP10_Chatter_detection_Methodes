#!/usr/bin/env python
# coding: utf-8
"""validation_figures.py — figures of a doe_validation_results.h5 (read only; see validate_indicators.py).

    python validation_figures.py --results X/doe_validation_results.h5 --out-dir validacion_figs/n12000

One PNG per figure (name -> h5 data it reads):
  ranking.png          bal. accuracy / MCC / AUC per run (+ Wilson / Hanley-McNeil 95%)   /ranking, /metrics/<run>
  roc.png              case-level ROC, orientation of roc_direction, operating point        /roc/<run>/{high,low}, /metrics
  case_matrix.png      run x case grid of outcomes, cases ordered by kappa (ramps: by Ap)    /summary/<run>
  detection_time.png   first detection vs kappa, against the amplitude onset t_onset         /summary/<run>
  delay_vs_kappa.png   signed delay of the first detection vs kappa, with the early_tol band /summary/<run>, attrs early_tol_s
  detection_amp.png    amplitude (% of the labelling base) when each indicator first alarms  case_NNN/Axial_disp, /summary
  score_vs_kappa.png   max(I_t) per case vs kappa (what the ROC thresholds)                 /summary/<run>.score_max
"""
import argparse
import os
import sys

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import ARTICLE_RCPARAMS  # noqa: E402

STYLE = {**ARTICLE_RCPARAMS, "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 11, "xtick.labelsize": 9,
         "ytick.labelsize": 9, "legend.fontsize": 8, "lines.markersize": 5, "xtick.direction": "out", "ytick.direction": "out", "savefig.transparent": False, "savefig.dpi": 160}
OUT_COLOR = {"TP": "#2a9d8f", "TP_early": "#8ecae6", "FA": "#e76f51", "FN": "#9b2226", "TN": "#bfd8bd",
             "FP": "#f4a261", "n/a": "#dddddd"}
RUN_COLOR = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def short(run: str) -> str:
    return run.split("_revo")[0].split("_aux")[0]


def load(path):
    """(attrs, {run: {col: array}}, ranking order, metrics {run: attrs}) from the h5."""
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        summ = {r: {c: (g[c].asstr()[()] if g[c].dtype.kind == "O" else g[c][()]) for c in g} for r, g in f["summary"].items()}
        order = list(f["ranking/run"].asstr()[()])
        met = {r: dict(g.attrs) for r, g in f["metrics"].items()}
    return attrs, summ, order, met


def _save(fig, out_dir, name):
    fig.savefig(os.path.join(out_dir, name))
    plt.close(fig)
    print("  ", name)


def fig_ranking(met, order, out_dir):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w, x = 0.26, np.arange(len(order))
    for j, (k, lab) in enumerate((("balanced_accuracy", "balanced accuracy"), ("MCC", "MCC"), ("AUC", "AUC"))):
        v = np.array([met[r][k] for r in order], float)
        err = None
        if k == "AUC":
            err = np.array([[met[r][k] - met[r]["AUC_lo"], met[r]["AUC_hi"] - met[r][k]] for r in order]).T
        ax.bar(x + (j - 1) * w, np.nan_to_num(v), w, yerr=err, capsize=2, label=lab, color=RUN_COLOR[j])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x, [f"{i + 1}. {short(r)}" for i, r in enumerate(order)], rotation=15)
    ax.set_ylabel("score"); ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.set_title("ranking: balanced accuracy, then MCC, then AUC", pad=22)
    _save(fig, out_dir, "ranking.png")


def fig_roc(path, met, order, out_dir):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    with h5py.File(path, "r") as f:
        for i, r in enumerate(order):
            m = met[r]
            g = f[f"roc/{r}/{'high' if m['roc_direction'] == 1 else 'low'}"]
            ax.step(g["fpr"][()], g["tpr"][()], where="post", color=RUN_COLOR[i],
                    label=f"{short(r)} AUC={m['AUC']:.2f}" + ("" if m["roc_direction"] == 1 else " (low=chatter)"))
            ax.plot(1 - m["TNR"], m["TPR"], "o", color=RUN_COLOR[i], mfc="none", ms=8)
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set(xlabel="FPR (stable cases that alarm)", ylabel="TPR", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02),
           title="case-level ROC of max(I_t); o = operating point of the indicator\n(its own threshold, first-detection rule)")
    ax.title.set_fontsize(8)
    ax.legend(loc="lower right")
    _save(fig, out_dir, "roc.png")


def _xkey(s):
    """x position of each case: kappa, or Ap start for ramps (kappa is NaN there). Returns (x, is_ramp)."""
    ramp = np.array([g == "ramp" for g in s["group"]])
    return np.where(ramp, s["ap_mm"], s["kappa"]), ramp


def fig_matrix(summ, order, out_dir):
    s0 = summ[order[0]]
    idx = np.argsort(np.where(s0["group"] == "ramp", 1e3 + s0["ap_mm"], s0["kappa"]), kind="stable")
    outs = list(OUT_COLOR)
    grid = np.array([[outs.index(summ[r]["outcome"][i]) if summ[r]["outcome"][i] in outs else len(outs) - 1 for i in idx]
                     for r in order])
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(idx) + 2), 0.55 * len(order) + 1.6))
    cmap = matplotlib.colors.ListedColormap(list(OUT_COLOR.values()))
    ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=len(outs) - 0.5, aspect="auto")
    tr = {"stable": "S", "unstable": "U", "mixed": "M"}
    lab = [(f"{s0['kappa'][i]:.2f}" if s0["group"][i] != "ramp" else f"{s0['ap_mm'][i]:.0f}->{s0['ap_end_mm'][i]:.0f}mm")
           + f" {tr.get(s0['truth'][i], 'g')}" for i in idx]
    ax.set_xticks(range(len(idx)), lab, rotation=90)
    ax.set_yticks(range(len(order)), [short(r) for r in order])
    ax.set_xlabel("case: kappa and truth  [S stable, U unstable, M ramp crossing, g gray]")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=o) for o, c in OUT_COLOR.items()], ncol=7,
              loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7)
    _save(fig, out_dir, "case_matrix.png")


def fig_detection_time(summ, order, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    s0 = summ[order[0]]
    x, ramp = _xkey(s0)
    m = ~ramp & np.isfinite(s0["t_onset_amp"])
    o = np.argsort(x[m])
    ax.plot(x[m][o], s0["t_onset_amp"][m][o], "k-s", ms=4, label="t_onset (amplitude > lim_sup)")
    for i, r in enumerate(order):
        s = summ[r]
        ok = ~ramp & np.isfinite(s["first_detection_t"])
        st = s["truth"] == "stable"
        ax.plot(x[ok & ~st], s["first_detection_t"][ok & ~st], "o", color=RUN_COLOR[i], label=f"{short(r)} (unstable)")
        ax.plot(x[ok & st], s["first_detection_t"][ok & st], "x", color=RUN_COLOR[i], label=f"{short(r)} (stable-labelled case)")
    ax.axvline(1, color="grey", ls=":"); ax.set_yscale("log")
    ax.set(xlabel="kappa = ap / ap_lim", ylabel="time of first detection [s]",
           title="first detection vs the amplitude onset of the truth")
    ax.legend(fontsize=6, ncol=2)
    _save(fig, out_dir, "detection_time.png")


def fig_delay(summ, order, early_tol, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, r in enumerate(order):
        s = summ[r]
        x, ramp = _xkey(s)
        ok = ~ramp & np.isfinite(s["delay_det_s"])
        ax.plot(x[ok], s["delay_det_s"][ok], "o-", color=RUN_COLOR[i], ms=4, label=short(r))
    ax.axhspan(-early_tol, 0, color="#8ecae6", alpha=0.4, label=f"anticipated hit (early_tol={early_tol:g} s)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xlabel="kappa", ylabel="first detection - t_onset [s]", yscale="symlog", yticks=[-10, -3, -1, -0.5, 0, 1],
           title="signed delay (negative = alarm before the amplitude onset)")
    ax.set_yticklabels(["-10", "-3", "-1", "-0.5", "0", "1"])
    ax.legend(fontsize=7)
    _save(fig, out_dir, "delay_vs_kappa.png")


def fig_detection_amp(path, summ, attrs, order, out_dir):
    """Amplitude of Axial_disp (max |y| in the 0.1 s before the alarm) as % of the labelling base."""
    with h5py.File(path, "r") as f:
        fig, ax = plt.subplots(figsize=(6, 4))
        s0 = summ[order[0]]
        x, ramp = _xkey(s0)
        base_attr, scale = str(attrs.get("labeling_base_attr", "$f_tooth$")), float(attrs.get("labeling_base_scale", 1e-3))
        for i, r in enumerate(order):
            xs, ys = [], []
            for k, case in enumerate(s0["case"]):
                td = summ[r]["first_detection_t"][k]
                if ramp[k] or s0["truth"][k] != "unstable" or not np.isfinite(td) or "Axial_disp" not in f[case]:
                    continue
                t, y = f[case]["Axial_disp/time"][()], f[case]["Axial_disp/values"][()]
                msk = (t <= td) & (t > td - 0.1)
                if msk.any() and base_attr in f[case].attrs:
                    xs.append(x[k]); ys.append(100 * np.abs(y[msk]).max() / (float(f[case].attrs[base_attr]) * scale))
            ax.plot(xs, ys, "o-", color=RUN_COLOR[i], ms=4, label=short(r))
    ax.axhline(float(attrs.get("labeling_lim_sup_pct", 40)), color="k", ls="--", label="lim_sup (truth onset)")
    ax.axhline(float(attrs.get("labeling_lim_inf_pct", 10)), color="grey", ls=":", label="lim_inf")
    ax.set(xlabel="kappa", ylabel="|Axial_disp| at first alarm [% of base]", yscale="log",
           title="how much chatter is visible when the indicator alarms (unstable cases)")
    ax.legend(fontsize=7)
    _save(fig, out_dir, "detection_amp.png")


def fig_score(summ, met, order, out_dir):
    fig, axs = plt.subplots(1, len(order), figsize=(2.6 * len(order), 3.4), squeeze=False, sharex=True)
    for ax, r in zip(axs[0], order):
        s = summ[r]
        x, ramp = _xkey(s)
        sc = s["score_max"] if met[r]["roc_direction"] == 1 else s["score_min"]
        for truth, c in (("stable", "#2a9d8f"), ("unstable", "#e76f51")):
            k = ~ramp & (s["truth"] == truth)
            ax.plot(x[k], sc[k], "o", color=c, ms=4, label=truth)
        ax.axvline(1, color="grey", ls=":")
        ax.set(title=f"{short(r)}\nAUC={met[r]['AUC']:.2f}", xlabel="kappa", yscale="symlog" if np.nanmin(sc) <= 0 else "log")
    axs[0][0].set_ylabel("case score (max I_t, or min if low=chatter)"); axs[0][0].legend(fontsize=7)
    _save(fig, out_dir, "score_vs_kappa.png")


def make_all(path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    attrs, summ, order, met = load(path)
    jobs = (lambda: fig_ranking(met, order, out_dir), lambda: fig_roc(path, met, order, out_dir),
            lambda: fig_matrix(summ, order, out_dir), lambda: fig_detection_time(summ, order, out_dir),
            lambda: fig_delay(summ, order, float(attrs.get("early_tol_s", 0.5)), out_dir),
            lambda: fig_detection_amp(path, summ, attrs, order, out_dir), lambda: fig_score(summ, met, order, out_dir))
    with plt.rc_context(STYLE):
        for job in jobs:   # a figure without data (e.g. only ramps: no kappa) is skipped, not fatal
            try:
                job()
            except ValueError as e:
                plt.close("all")
                print("   [skip]", e)

def main():
    p = argparse.ArgumentParser(description="Figures of a doe_validation_results.h5 (read only).")
    p.add_argument("--results", required=True, metavar="PATH", help="doe_validation_results.h5")
    p.add_argument("--out-dir", required=True, metavar="DIR")
    a = p.parse_args()
    make_all(a.results, a.out_dir)


if __name__ == "__main__":
    main()
