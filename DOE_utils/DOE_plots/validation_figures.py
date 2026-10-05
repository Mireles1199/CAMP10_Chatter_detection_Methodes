#!/usr/bin/env python
# coding: utf-8
"""validation_figures.py — figures of a doe_validation_results.h5 (read only; see validate_indicators.py).

article-plot-style (plot_style.py): FIGSIZE_* x FIGSCALE, constrained_layout, EN/FR text, Okabe-Ito colors.
Constant-Ap cases only (rows with group == 'global'); ramps are left out until their rule is settled.

API (what doe_unified_selector.py / launcher.py consume):
    FIGURES                          {name: fig_<name>}, every one  fig_<name>(h5_path, out_dir=None) -> Figure
    fig_compare(h5_a, h5_b, out_dir=None)   A/B of two validation files (not in FIGURES: it needs two files)
    make_all(h5_path, out_dir=None)  every figure of FIGURES; out_dir default = <folder of the h5>/figs_validation
A figure is saved as <out_dir>/<name>.png (300 dpi) only when out_dir is given; fig._keep_size holds its size.

CLI:  python validation_figures.py --results X/doe_validation_results.h5 [--out-dir D] [--lang EN|FR|both] [--scale 1.5]
      python validation_figures.py --selftest

name (-> h5 data it reads)
  ranking            bal. accuracy / MCC / AUC (+ Hanley-McNeil)       /ranking, /metrics/<run>
  roc                case-level ROC + operating point                   /roc/<run>/{high,low}, /metrics
  tpr_tnr            TPR and TNR with Wilson 95%                        /metrics/<run> TPR TNR (_lo, _hi)
  confusion          2x2 per indicator                                  /metrics/<run> TP FN TN FP
  case_matrix        indicator x case outcomes, cases ordered by kappa  /summary/<run>
  detection_time     first detection vs the amplitude onset             /summary/<run>
  delay_vs_kappa     signed delay of the first detection vs kappa       /summary/<run>
  detection_amp      |Axial_disp| (% of base) at the first alarm        case_NNN/Axial_disp, /summary/<run>
  score_vs_kappa     max(I_t) per case vs kappa                         /summary/<run>
  score_dist         score of stable vs unstable cases                  /summary/<run>
  alarm_quality      alarm fraction in stable cases, persistence        /metrics/<run>
  training_coverage  training cases vs validated cases (kappa, rpm)     /training, /summary/<run>
"""
import argparse
import functools
import os
import sys
from types import SimpleNamespace

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_style as ps  # noqa: E402
from sld_model import OUTCOMES  # noqa: E402  (same TP/TN/FN/FP colors as the SLD of the viewer)

LANGUAGE = "EN"   # "EN" | "FR" | "both"
FIGSCALE = 1.5    # multiplier of the plot_style presets (same criterion as sld_model.py)
RUN_COLOR = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]   # Okabe-Ito, one per indicator
FIGURES = {}


def T(en, fr=None, sep="\n"):
    return ps.lang_text(en, fr or en, LANGUAGE, sep)


def short(run):
    return run.split("_revo")[0].split("_aux")[0]


# ============================================================================== data
def load(path):
    """Constant-Ap rows of /summary (group 'global'), run order of /ranking, /metrics attrs and the root attrs."""
    with h5py.File(path, "r") as f:
        summ = {}
        for r, g in f["summary"].items():
            cols = {c: (g[c].asstr()[()] if g[c].dtype.kind == "O" else g[c][()]) for c in g}
            keep = cols["group"] == "global" if "group" in cols else np.ones(len(cols["case"]), bool)
            summ[r] = {c: v[keep] for c, v in cols.items()}
        return SimpleNamespace(path=path, attrs=dict(f.attrs), summ=summ, order=list(f["ranking/run"].asstr()[()]),
                               met={r: dict(g.attrs) for r, g in f["metrics"].items()})


# ============================================================================== figure plumbing
def _fig(base=ps.FIGSIZE_SIMPLE, ncols=1):
    fig, axs = plt.subplots(1, ncols, figsize=ps.figsize_from_scale(base, FIGSCALE), constrained_layout=True, squeeze=False)
    return fig, (axs[0, 0] if ncols == 1 else list(axs[0]))


def _grid(n):
    """n panels in a grid of at most 2 columns, each one a FIGSIZE_SIMPLE (plot_style.figsize_grid)."""
    nc = min(n, 2)
    nr = -(-n // nc)
    fig, axs = plt.subplots(nr, nc, figsize=ps.figsize_from_scale(ps.figsize_grid(nc, nr), FIGSCALE),
                            constrained_layout=True, squeeze=False)
    flat = list(axs.ravel())
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def _done(fig, out_dir, name):
    fig._keep_size = tuple(fig.get_size_inches())   # the viewer neither resizes nor saves it at another size
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, name + ".png"), dpi=300, bbox_inches="tight")
    return fig


def _figure(fn):
    """Registers fig_<name>(D) as FIGURES[name](h5_path, out_dir=None) -> Figure, drawn with ARTICLE_RCPARAMS."""
    name = fn.__name__[len("fig_"):]

    @functools.wraps(fn)
    def run(h5_path, out_dir=None):
        with plt.rc_context(ps.ARTICLE_RCPARAMS):
            fig = fn(load(h5_path))
        return _done(fig, out_dir, name)
    FIGURES[name] = run
    return run


def _yscale(v):
    return "log" if np.nanmin(v) > 0 else "symlog"


def _kappa(s):
    return s["kappa"]


# ============================================================================== figures
@_figure
def fig_ranking(D):
    fig, ax = _fig()
    w, x = 0.26, np.arange(len(D.order))
    for j, (k, lab) in enumerate((("balanced_accuracy", T("balanced accuracy", "exactitude équilibrée", " / ")),
                                  ("MCC", "MCC"), ("AUC", "AUC"))):
        err = None
        if k == "AUC":
            err = np.nan_to_num(np.array([[D.met[r]["AUC"] - D.met[r]["AUC_lo"], D.met[r]["AUC_hi"] - D.met[r]["AUC"]]
                                          for r in D.order], float).T)
        ax.bar(x + (j - 1) * w, np.nan_to_num([D.met[r][k] for r in D.order]), w, yerr=err, capsize=2, color=RUN_COLOR[j],
               label=lab)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [f"{i + 1}. {short(r)}" for i, r in enumerate(D.order)], rotation=15)
    ax.set_ylabel(T("score", "score"))
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=1 if LANGUAGE == "both" else 3)
    return fig


@_figure
def fig_roc(D):
    fig, ax = _fig()
    with h5py.File(D.path, "r") as f:
        for i, r in enumerate(D.order):
            m = D.met[r]
            g = f[f"roc/{r}/{'high' if m['roc_direction'] == 1 else 'low'}"]
            ax.step(g["fpr"][()], g["tpr"][()], where="post", color=RUN_COLOR[i],
                    label=f"{short(r)} AUC={m['AUC']:.2f}" + ("" if m["roc_direction"] == 1 else " (low)"))
            ax.plot(1 - m["TNR"], m["TPR"], "o", ms=7, mfc="none", color=RUN_COLOR[i])   # operating point of the indicator
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set(xlabel=T("FPR (stable cases that alarm)", "FPR (cas stables avec alarme)"), ylabel="TPR",
           xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.legend(loc="lower right")
    return fig


@_figure
def fig_tpr_tnr(D):
    fig, ax = _fig()
    x = np.arange(len(D.order))
    for off, k, col, lab in ((-0.15, "TPR", ps.COLOR_UNSTABLE, T("TPR (unstable cases)", "TPR (cas instables)", " / ")),
                             (0.15, "TNR", ps.COLOR_STABLE, T("TNR (stable cases)", "TNR (cas stables)", " / "))):
        v = np.array([D.met[r][k] for r in D.order], float)
        err = np.array([[D.met[r][k] - D.met[r][k + "_lo"], D.met[r][k + "_hi"] - D.met[r][k]] for r in D.order], float).T
        ax.errorbar(x + off, v, yerr=np.nan_to_num(err), fmt="o", ms=6, capsize=3, color=col, label=lab)
    ax.set_xticks(x, [short(r) for r in D.order], rotation=15)
    ax.set(ylim=(-0.05, 1.05), ylabel=T("rate (Wilson 95%)", "taux (Wilson 95 %)"))
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=1 if LANGUAGE == "both" else 2)
    return fig


@_figure
def fig_confusion(D):
    fig, axs = _grid(len(D.order))
    cmap = ListedColormap([OUTCOMES[k][0] for k in ("TP", "FN", "FP", "TN")])
    for ax, r in zip(axs, D.order):
        m = D.met[r]
        ax.imshow([[0, 1], [2, 3]], cmap=cmap, vmin=-0.5, vmax=3.5, alpha=0.8)
        for (i, j), k in zip(((0, 0), (0, 1), (1, 0), (1, 1)), ("TP", "FN", "FP", "TN")):
            ax.text(j, i, f"{k}\n{int(m[k])}", ha="center", va="center", fontsize=14)
        ax.set_xticks([0, 1], [T("alarm", "alarme"), T("no alarm", "pas d'alarme")])
        ax.set_yticks([0, 1], [T("unstable", "instable"), T("stable", "stable")], rotation=90, va="center")
        ax.set_title(short(r))
        ax.tick_params(length=0)
    return fig


@_figure
def fig_case_matrix(D):
    s0 = D.summ[D.order[0]]
    idx = np.argsort(_kappa(s0), kind="stable")
    keys = ["TP", "TN", "FN", "FP", "n/a"]
    colors = [OUTCOMES[k][0] for k in keys[:4]] + [ps.COLOR_GRAY]
    grid = np.array([[keys.index(D.summ[r]["outcome"][i]) if D.summ[r]["outcome"][i] in keys else 4 for i in idx]
                     for r in D.order])
    fig, ax = _fig(ps.FIGSIZE_WIDE)
    ax.imshow(grid, cmap=ListedColormap(colors), vmin=-0.5, vmax=4.5, aspect="auto")
    tr = {"stable": "S", "unstable": "U"}
    ax.set_xticks(range(len(idx)), [f"{s0['kappa'][i]:.2f} {tr.get(s0['truth'][i], 'g')}" for i in idx], rotation=90)
    ax.set_yticks(range(len(D.order)), [short(r) for r in D.order])
    ax.set_xlabel(T(r"case: $\kappa$ and truth (S stable, U unstable, g gray)",
                    r"cas : $\kappa$ et vérité (S stable, U instable, g gris)"))
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=k) for k, c in zip(keys, colors)], ncol=5,
               loc="outside upper center")
    return fig


@_figure
def fig_detection_time(D):
    fig, ax = _fig()
    s0 = D.summ[D.order[0]]
    m = np.isfinite(s0["t_onset_amp"])
    o = np.argsort(_kappa(s0)[m])
    ax.plot(_kappa(s0)[m][o], s0["t_onset_amp"][m][o], "k-s", ms=4, label=T("amplitude onset $t_{onset}$", "début en amplitude $t_{onset}$", " / "))
    for i, r in enumerate(D.order):
        s = D.summ[r]
        ok, st, un = np.isfinite(s["first_detection_t"]), s["truth"] == "stable", s["truth"] == "unstable"
        gr = ok & (s["truth"] == "unlabelled")   # gray: hollow, not scored
        ax.plot(_kappa(s)[gr], s["first_detection_t"][gr], "o", ms=5, mfc="none", color=RUN_COLOR[i])
        ax.plot(_kappa(s)[ok & un], s["first_detection_t"][ok & un], "o", ms=5, color=RUN_COLOR[i], label=short(r))
        ax.plot(_kappa(s)[ok & st], s["first_detection_t"][ok & st], "x", ms=6, color=RUN_COLOR[i])
    ax.plot([], [], "o", mfc="none", color="k", label=T("gray (not scored)", "gris (non noté)", " / "))
    ax.axvline(1, color="grey", ls=":")
    ax.set(xlabel=r"$\kappa$", ylabel=T("first detection [s]", "première détection [s]"), yscale="log")
    ax.legend(fontsize=8, ncol=2)
    return fig


@_figure
def fig_delay_vs_kappa(D):
    fig, ax = _fig()
    for i, r in enumerate(D.order):
        s = D.summ[r]
        ok = np.isfinite(s["delay_det_s"])
        o = np.argsort(_kappa(s)[ok])
        ax.plot(_kappa(s)[ok][o], s["delay_det_s"][ok][o], "o-", ms=4, color=RUN_COLOR[i], label=short(r))
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xlabel=r"$\kappa$", ylabel=T("first detection $-$ $t_{onset}$ [s]", "première détection $-$ $t_{onset}$ [s]"),
           yscale="symlog")
    ax.legend(fontsize=8)
    return fig


@_figure
def fig_detection_amp(D):
    """|Axial_disp| (max in the 0.1 s before the alarm) as % of the labelling base, unstable cases."""
    fig, ax = _fig()
    base_attr, scale = str(D.attrs.get("labeling_base_attr", "$f_tooth$")), float(D.attrs.get("labeling_base_scale", 1e-3))
    s0 = D.summ[D.order[0]]
    with h5py.File(D.path, "r") as f:
        for i, r in enumerate(D.order):
            xs, ys = [], []
            for k, case in enumerate(s0["case"]):
                td = D.summ[r]["first_detection_t"][k]
                if s0["truth"][k] != "unstable" or not np.isfinite(td) or "Axial_disp" not in f[case]:
                    continue
                t, y = f[case]["Axial_disp/time"][()], f[case]["Axial_disp/values"][()]
                msk = (t <= td) & (t > td - 0.1)
                if msk.any() and base_attr in f[case].attrs:
                    xs.append(s0["kappa"][k]); ys.append(100 * np.abs(y[msk]).max() / (float(f[case].attrs[base_attr]) * scale))
            ax.plot(xs, ys, "o-", ms=4, color=RUN_COLOR[i], label=short(r))
    ax.axhline(float(D.attrs.get("labeling_lim_sup_pct", 40)), color="k", ls="--", label="lim_sup")
    ax.axhline(float(D.attrs.get("labeling_lim_inf_pct", 10)), color="grey", ls=":", label="lim_inf")
    ax.set(xlabel=r"$\kappa$", ylabel=T("$|$Axial_disp$|$ at first alarm [% of base]", "$|$Axial_disp$|$ à la 1re alarme [% base]"),
           yscale="log")
    ax.legend(fontsize=8)
    return fig


def _case_score(D, r):
    s = D.summ[r]
    return s["score_max"] if D.met[r]["roc_direction"] == 1 else s["score_min"]


@_figure
def fig_score_vs_kappa(D):
    fig, axs = _grid(len(D.order))
    for ax, r in zip(axs, D.order):
        s, sc = D.summ[r], _case_score(D, r)
        for truth, col in (("stable", ps.COLOR_STABLE), ("unstable", ps.COLOR_UNSTABLE)):
            k = s["truth"] == truth
            ax.plot(_kappa(s)[k], sc[k], "o", ms=5, color=col, label=T(truth, {"stable": "stable", "unstable": "instable"}[truth], " / "))
        g = (s["truth"] == "unlabelled") & np.isfinite(sc)
        if g.any():
            ax.plot(_kappa(s)[g], sc[g], "o", ms=5, mfc="none", color=ps.COLOR_GRAY, label=T("gray (not scored)", "gris (non noté)", " / "))
        ax.axvline(1, color="grey", ls=":")
        ax.set(title=f"{short(r)}  AUC={D.met[r]['AUC']:.2f}", xlabel=r"$\kappa$", yscale=_yscale(sc))
        ax.set_ylabel("max $I_t$" if D.met[r]["roc_direction"] == 1 else "min $I_t$")
    axs[0].legend(fontsize=8)
    return fig


@_figure
def fig_score_dist(D):
    fig, axs = _grid(len(D.order))
    for ax, r in zip(axs, D.order):
        s, sc = D.summ[r], _case_score(D, r)
        for x0, truth, col in ((0, "stable", ps.COLOR_STABLE), (1, "unstable", ps.COLOR_UNSTABLE), (2, "unlabelled", ps.COLOR_GRAY)):
            v = sc[s["truth"] == truth]
            ax.plot(x0 + np.linspace(-0.15, 0.15, len(v)), v, "o", ms=5, color=col, mfc="none" if truth == "unlabelled" else col)
        ax.set_xticks([0, 1, 2], [T("stable", "stable"), T("unstable", "instable"), T("gray", "gris")])
        ax.set(title=f"{short(r)}  AUC={D.met[r]['AUC']:.2f}", yscale=_yscale(sc), xlim=(-0.5, 2.5))
        ax.set_ylabel(T("case score", "score du cas"))
    return fig


@_figure
def fig_alarm_quality(D):
    fig, axs = _fig(ps.FIGSIZE_WIDE, ncols=2)
    x = np.arange(len(D.order))
    for ax, k, lab in ((axs[0], "mean_alarm_fraction_stable", T("alarm fraction, stable cases", "fraction d'alarme, cas stables")),
                       (axs[1], "mean_persistence", T("persistence after detection", "persistance après détection"))):
        ax.bar(x, np.nan_to_num([D.met[r][k] for r in D.order]), color=RUN_COLOR[:len(x)])
        ax.set_xticks(x, [short(r) for r in D.order], rotation=15)
        ax.set(ylabel=lab, ylim=(0, 1.05))
    return fig


@_figure
def fig_training_coverage(D):
    fig, ax = _fig()
    with h5py.File(D.path, "r") as f:
        if "training" not in f:
            raise ValueError("no /training in this file (run validate with --reference)")
        g = f["training"]
        k, rpm, lab = g["kappa"][()], g["spin_rpm"][()], g["label"].asstr()[()]
    col = {"stable": ps.COLOR_STABLE, "unstable": ps.COLOR_UNSTABLE}
    for name in ("stable", "unstable", "gray"):
        m = np.array([name in str(x).split(",") for x in lab])
        if m.any():
            ax.plot(k[m], rpm[m], "o", ms=4, color=col.get(name, ps.COLOR_GRAY), label=T(f"training {name}", f"entraînement {name}", " / "))
    s0 = D.summ[D.order[0]]
    ax.plot(_kappa(s0), s0["spin_rpm"], "kx", ms=7, label=T("validation", "validation"))
    ax.set(xlabel=r"$\kappa$", ylabel=T("spin [rpm]", "rotation [tr/min]"))
    ax.legend(fontsize=8)
    return fig


def fig_compare(h5_a, h5_b, out_dir=None):
    """A/B of balanced accuracy, MCC and AUC per indicator (dumbbell A -> B) for two validation files."""
    A, B = load(h5_a), load(h5_b)
    runs = [r for r in A.order if r in B.order]
    if not runs:
        raise ValueError("the two files have no indicator in common")
    name = lambda D: str(D.attrs.get("experiment", os.path.basename(os.path.dirname(os.path.abspath(D.path)))))   # noqa: E731
    with plt.rc_context(ps.ARTICLE_RCPARAMS):
        fig, axs = _fig(ps.FIGSIZE_WIDE, ncols=3)
        y = np.arange(len(runs))
        for ax, k in zip(axs, ("balanced_accuracy", "MCC", "AUC")):
            a, b = [np.array([D.met[r][k] for r in runs], float) for D in (A, B)]
            ax.hlines(y, a, b, color="grey", lw=1)
            ax.plot(a, y, "o", ms=6, mfc="none", color=ps.COLOR_UNSTABLE, label="A: " + name(A))
            ax.plot(b, y, "o", ms=6, color=ps.COLOR_STABLE, label="B: " + name(B))
            ax.set_yticks(y, [short(r) for r in runs] if ax is axs[0] else [])
            ax.set(xlabel=k.replace("balanced_accuracy", "bal. accuracy"))
        fig.legend(*axs[0].get_legend_handles_labels(), loc="outside upper center", ncol=2, fontsize=8)
    return _done(fig, out_dir, f"compare_{name(A)}_vs_{name(B)}")


# ============================================================================== run
def make_all(h5_path, out_dir=None):
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(h5_path)), "figs_validation")
    for name, fn in FIGURES.items():
        try:
            plt.close(fn(h5_path, out_dir))
            print("  ", name)
        except Exception as e:   # a figure without data (e.g. only ramps, no /training) is skipped, not fatal
            plt.close("all")
            print("   [skip]", name, "-", e)


def _selftest():
    import tempfile
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DOE_analisis"))
    import validate_indicators as vi
    d = tempfile.mkdtemp()
    t = np.arange(0.0, 10.0, 0.1)
    kap = [0.5, 0.7, 0.9, 1.2, 1.5, 1.9]
    ind, lab, out = (os.path.join(d, n) for n in ("ind.h5", "lab.h5", "out.h5"))
    with h5py.File(ind, "w") as f, h5py.File(lab, "w") as fl:
        for i, k in enumerate(kap):
            g = f.create_group(f"case_{i:03d}")
            g.attrs.update({"$spin_rate$": 12000.0, "$Ap_start$": 0.005 * k, "kappa": k, "$f_tooth$": 0.05})
            g["Axial_disp/time"], g["Axial_disp/values"] = t, 1e-5 * t * k
            for run, it in (("ind_a", np.sin(t) * k), ("ind_b", np.cos(3 * t) + k)):
                r = g.create_group(run)
                r["t"], r["I_t"] = t, it
                if k > 1.0:
                    r["t_d"] = [6.0 - i * 0.5, 6.1 - i * 0.5]
            p = fl.require_group(f"{'stable' if k < 1.0 else 'unstable'}/case_{i:03d}").create_dataset("Axial_disp__000", data=[0.0])
            p.attrs.update(channel="Axial_disp", t0=0.0, t1=10.0, labeling_strategy="amplitude", labeling_lim_sup_pct=40.0,
                           labeling_lim_inf_pct=10.0, labeling_base_attr="$f_tooth$", labeling_base_scale=1e-3,
                           labeling_signal="Axial_disp", kappa=k, **{"$Ap_start$": 0.005 * k, "$spin_rate$": 12000.0})
    vi.validate(ind, lab, out, reference_h5=lab)
    figs = os.path.join(d, "figs")
    make_all(out, figs)
    assert sorted(os.listdir(figs)) == sorted(n + ".png" for n in FIGURES), os.listdir(figs)
    assert len(FIGURES) == 12
    fig = FIGURES["roc"](out)
    assert fig._keep_size == tuple(fig.get_size_inches()) and np.allclose(fig._keep_size, ps.figsize_from_scale(ps.FIGSIZE_SIMPLE, FIGSCALE))
    fc = fig_compare(out, out, figs)
    assert fc._keep_size and any(n.startswith("compare_") for n in os.listdir(figs))
    print("validation_figures selftest OK")


def main():
    global LANGUAGE, FIGSCALE
    p = argparse.ArgumentParser(description="Figures of a doe_validation_results.h5 (read only).")
    p.add_argument("--results", metavar="PATH", help="doe_validation_results.h5")
    p.add_argument("--out-dir", metavar="DIR", help="default: <folder of the h5>/figs_validation")
    p.add_argument("--lang", choices=("EN", "FR", "both"), default=LANGUAGE)
    p.add_argument("--scale", type=float, default=FIGSCALE, help="multiplier of the plot_style presets")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    LANGUAGE, FIGSCALE = a.lang, a.scale
    plt.switch_backend("Agg")
    if a.selftest:
        return _selftest()
    if not a.results:
        p.error("--results is required")
    make_all(a.results, a.out_dir)


if __name__ == "__main__":
    main()
