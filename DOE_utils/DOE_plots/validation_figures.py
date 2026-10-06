#!/usr/bin/env python
# coding: utf-8
"""validation_figures.py — figures of a doe_validation_results.h5 (read only; see validate_indicators.py).

article-plot-style (plot_style.py): FIGSIZE_* x FIGSCALE, constrained_layout, EN/FR text, Okabe-Ito colors.
Constant-Ap cases only (rows with group == 'global'); ramps are left out until their rule is settled.

API (what doe_unified_selector.py / launcher.py consume):
    FIGURES                          {name: fig_<name>}, every one  fig_<name>(h5_path, out_dir=None) -> Figure
    fig_compare(h5_a, h5_b, out_dir=None)   A/B of two validation files (not in FIGURES: it needs two files)
    make_all(h5_path, out_dir=None)  every figure of FIGURES; out_dir default = figs_dir(h5_path)
    figs_dir(h5_path)                <folder of the h5>/figs_validation, + _gray-<mode> when the file was made with --gray stable|unstable;
                                     figs_noise_validation[_gray-<mode>] for a noise validation file
    NOISE_FIGURES                    same API, for doe_noise_validation_results*.h5 (validate_noise.py); make_all picks the
                                     registry from the schema of the file
Gray cases (--gray of validate_indicators.py): mode 'ignore' leaves them out of the metrics; 'stable' / 'unstable' score them
as such. Either way they are drawn as hollow gray points, and a note on the figure says the mode when it is not 'ignore'.
A figure is saved as <out_dir>/<name>.png (300 dpi) only when out_dir is given; fig._keep_size holds its size.

CLI:  python validation_figures.py --results X/doe_validation_results.h5 [--out-dir D] [--lang EN|FR|both] [--scale 1.5]
      python validation_figures.py --selftest

name (-> h5 data it reads)
  ranking            bal. accuracy / MCC / AUC with their 95% intervals  /ranking, /metrics/<run> (*_lo, *_hi)
  roc                case-level ROC + operating point                   /roc/<run>/{high,low}, /metrics
  tpr_tnr            TPR and TNR with Wilson 95%                        /metrics/<run> TPR TNR (_lo, _hi)
  confusion          2x2 per indicator                                  /metrics/<run> TP FN TN FP
  case_matrix        indicator x case outcomes, cases ordered by kappa  /summary/<run>
  detection_time     first detection vs the amplitude onset             /summary/<run>
  delay_vs_kappa     signed delay of the first detection vs kappa       /summary/<run>
  detection_amp      |Axial_disp| (% of base) at the first alarm        case_NNN/Axial_disp, /summary/<run>
  score_vs_kappa     max(I_t) per case vs kappa                         /summary/<run>
  score_dist         score of stable vs unstable cases                  /summary/<run>
  anticipation       t_det / t_onset of the hits vs kappa               /summary/<run>.t_ratio
  pairwise_test      exact McNemar p-value between indicators           /pairwise
  gray_bounds        bal. accuracy / MCC if gray = stable / unstable    /metrics/<run> gray_as_*
  alarm_quality      alarm fraction in stable cases, persistence        /metrics/<run>
  training_coverage  training cases vs validated cases (kappa, rpm)     /training, /summary/<run>
NOISE_FIGURES (doe_noise_validation_results.h5; PLAN_noise_validation.md §6)
  noise_metrics      bal. accuracy, TPR, TNR, alarm fraction vs SNR      /by_snr/<run> (mean, min, max), /clean/<run>
  noise_case_matrix  fraction of realizations right, case x SNR          /summary/<run>
  noise_anticipation median t_det / t_onset vs SNR                       /by_snr/<run> median_t_ratio_*, /clean/<run>
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
NOISE_FIGURES = {}


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


def _figure(fn=None, *, registry=None, loader=None):
    """Registers fig_<name>(D) as registry[name](h5_path, out_dir=None) -> Figure (FIGURES and load by default),
    drawn with ARTICLE_RCPARAMS."""
    if fn is None:
        return functools.partial(_figure, registry=registry, loader=loader)
    registry, loader = (FIGURES if registry is None else registry), (loader or load)
    name = fn.__name__[len("fig_"):]

    @functools.wraps(fn)
    def run(h5_path, out_dir=None):
        with plt.rc_context(ps.ARTICLE_RCPARAMS):
            D = loader(h5_path)
            fig = fn(D)
            mode = str(D.attrs.get("gray_mode", "ignore"))
            if mode != "ignore":   # a figure of a non-default mode must say so
                fig.supxlabel(T(f"gray cases counted as {mode}", f"cas gris comptés comme {'stable' if mode == 'stable' else 'instable'}"),
                              x=0.99, ha="right", fontsize=7, color="grey")   # supxlabel: constrained_layout leaves room for it
        return _done(fig, out_dir, name)
    registry[name] = run
    return run


def _err(D, k):
    """[[v - lo], [hi - v]] of metric k over the runs of the ranking, None if the file has no interval for it."""
    try:
        return np.nan_to_num(np.array([[D.met[r][k] - D.met[r][k + "_lo"], D.met[r][k + "_hi"] - D.met[r][k]] for r in D.order], float).T)
    except KeyError:
        return None


def _yscale(v):
    return "log" if np.nanmin(v) > 0 else "symlog"


def _gray(s):
    """Cases whose label was gray (whatever --gray did with them)."""
    return s["is_gray"] == 1 if "is_gray" in s else s["truth"] == "unlabelled"


def _kappa(s):
    return s["kappa"]


# ============================================================================== figures
@_figure
def fig_ranking(D):
    fig, ax = _fig()
    w, x = 0.26, np.arange(len(D.order))
    for j, (k, lab) in enumerate((("balanced_accuracy", T("balanced accuracy", "exactitude équilibrée", " / ")),
                                  ("MCC", "MCC"), ("AUC", "AUC"))):
        err = _err(D, k)
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
            mk = "osD^v<"[i % 6]   # one marker per indicator: identical points do not hide each other
            ax.step(g["fpr"][()], g["tpr"][()], where="post", color=RUN_COLOR[i])
            ax.plot(1 - m["TNR"], m["TPR"], mk, ms=9, mfc="none", mew=1.5, color=RUN_COLOR[i])   # operating point of the indicator
            ax.plot([], [], color=RUN_COLOR[i], marker=mk, mfc="none", ms=8,
                    label=f"{short(r)} AUC={m['AUC']:.2f}" + ("" if m["roc_direction"] == 1 else " (low)"))
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
        ax.errorbar(x + off, v, yerr=_err(D, k), fmt="o", ms=6, capsize=3, color=col, label=lab)
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
    g0 = _gray(s0)
    ax.set_xticks(range(len(idx)), [f"{s0['kappa'][i]:.2f} {'g' if g0[i] else tr.get(s0['truth'][i], 'g')}" for i in idx], rotation=90)
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
        g = _gray(s)
        ok, st, un = np.isfinite(s["first_detection_t"]), (s["truth"] == "stable") & ~g, (s["truth"] == "unstable") & ~g
        gr = ok & g   # gray: hollow
        ax.plot(_kappa(s)[gr], s["first_detection_t"][gr], "o", ms=5, mfc="none", color=RUN_COLOR[i])
        ax.plot(_kappa(s)[ok & un], s["first_detection_t"][ok & un], "o", ms=5, color=RUN_COLOR[i], label=short(r))
        ax.plot(_kappa(s)[ok & st], s["first_detection_t"][ok & st], "x", ms=6, color=RUN_COLOR[i])
    ax.plot([], [], "o", mfc="none", color="k", label=T("gray (not scored)", "gris (non noté)", " / "))
    ax.axvline(1, color="grey", ls=":")
    ax.set(xlabel=r"$\kappa$", ylabel=T("first detection [s]", "première détection [s]"), yscale="log")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=3, fontsize=8)
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
    g0 = _gray(s0)
    with h5py.File(D.path, "r") as f:
        for i, r in enumerate(D.order):
            xs, ys = [], []
            for k, case in enumerate(s0["case"]):
                td = D.summ[r]["first_detection_t"][k]
                if s0["truth"][k] != "unstable" or g0[k] or not np.isfinite(td) or "Axial_disp" not in f[case]:
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
            k = (s["truth"] == truth) & ~_gray(s)
            ax.plot(_kappa(s)[k], sc[k], "o", ms=5, color=col, label=T(truth, {"stable": "stable", "unstable": "instable"}[truth], " / "))
        g = _gray(s) & np.isfinite(sc)
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
        g = _gray(s)
        for x0, k, col in ((0, (s["truth"] == "stable") & ~g, ps.COLOR_STABLE), (1, (s["truth"] == "unstable") & ~g, ps.COLOR_UNSTABLE),
                           (2, g, ps.COLOR_GRAY)):
            v = sc[k]
            ax.plot(x0 + np.linspace(-0.15, 0.15, len(v)), v, "o", ms=5, color=col, mfc="none" if x0 == 2 else col)
        ax.set_xticks([0, 1, 2], [T("stable", "stable"), T("unstable", "instable"), T("gray", "gris")])
        ax.set(title=f"{short(r)}  AUC={D.met[r]['AUC']:.2f}", yscale=_yscale(sc), xlim=(-0.5, 2.5))
        ax.set_ylabel(T("case score", "score du cas"))
    return fig


@_figure
def fig_anticipation(D):
    """t_det / t_onset of the hits: 1 = the alarm comes when the amplitude reaches the limit, 0.5 = at half of that time."""
    fig, ax = _fig()
    for i, r in enumerate(D.order):
        s = D.summ[r]
        if "t_ratio" not in s:
            raise ValueError("no t_ratio in this file (run validate again)")
        ok = np.isfinite(s["t_ratio"]) & ~_gray(s)
        o = np.argsort(_kappa(s)[ok])
        ax.plot(_kappa(s)[ok][o], s["t_ratio"][ok][o], "o-", ms=4, color=RUN_COLOR[i], label=short(r))
    ax.axhline(1, color="k", ls="--", lw=0.8)
    ax.set(xlabel=r"$\kappa$", ylabel=T("$t_{det}\,/\,t_{onset}$", "$t_{det}\,/\,t_{onset}$"), ylim=(0, 1.1))
    ax.text(0.02, 0.97, T("1 = alarm when the amplitude reaches the limit", "1 = alarme quand l'amplitude atteint la limite"),
            transform=ax.transAxes, va="top", fontsize=8)
    ax.legend(fontsize=8, loc="lower right")
    return fig


@_figure
def fig_pairwise_test(D):
    """Exact McNemar p-value for every pair of indicators over the cases both scored: p < 0.05 = more than chance."""
    with h5py.File(D.path, "r") as f:
        if "pairwise" not in f:
            raise ValueError("no /pairwise in this file (needs two or more indicators)")
        g = f["pairwise"]
        pa, pb, ao, bo, pv = (g[c].asstr()[()] if c.startswith("run") else g[c][()] for c in ("run_a", "run_b", "a_only", "b_only", "p_value"))
    n = len(D.order)
    pos = {r: i for i, r in enumerate(D.order)}
    grid = np.full((n, n), -1)
    fig, ax = _fig()
    for a, b, x, y, p in zip(pa, pb, ao, bo, pv):
        for i, j, u, v in ((pos[a], pos[b], x, y), (pos[b], pos[a], y, x)):
            grid[i, j] = int(p < 0.05)
            ax.text(j, i, f"p={p:.3f}\n{u} | {v}", ha="center", va="center", fontsize=9)
    ax.imshow(grid, cmap=ListedColormap(["#f2f2f2", "#dddddd", ps.COLOR_UNSTABLE]), vmin=-1, vmax=1)
    ax.set_xticks(range(n), [short(r) for r in D.order], rotation=15)
    ax.set_yticks(range(n), [short(r) for r in D.order])
    ax.set_xlabel(T("orange: p < 0.05 (more than chance)\ncell: row right, column wrong | the reverse",
                    "orange : p < 0,05 (plus que le hasard)\ncase : ligne juste, colonne fausse | l'inverse"), fontsize=8)
    ax.tick_params(length=0)
    return fig


@_figure
def fig_gray_bounds(D):
    """Balanced accuracy and MCC if the gray cases counted as stable (pessimistic) or as unstable (optimistic)."""
    if not any(D.met[r].get("n_gray", 0) for r in D.order):
        raise ValueError("no gray cases in this file")
    fig, axs = _fig(ps.FIGSIZE_WIDE, ncols=2)
    y = np.arange(len(D.order))
    for ax, k, lab in ((axs[0], "balanced_accuracy", T("balanced accuracy", "exactitude équilibrée")), (axs[1], "MCC", "MCC")):
        pes = np.array([D.met[r]["gray_as_stable_" + k] for r in D.order], float)
        opt = np.array([D.met[r]["gray_as_unstable_" + k] for r in D.order], float)
        cur = np.array([D.met[r][k] for r in D.order], float)
        ax.hlines(y, np.fmin(pes, opt), np.fmax(pes, opt), color="grey", lw=2)
        ax.plot(pes, y, "s", ms=7, mfc="none", mew=1.5, color=ps.COLOR_STABLE, label=T("gray = stable (pessimistic)", "gris = stable (pessimiste)", " / "))
        ax.plot(opt, y, "^", ms=8, mfc="none", mew=1.5, color=ps.COLOR_UNSTABLE, label=T("gray = unstable (optimistic)", "gris = instable (optimiste)", " / "))
        ax.plot(cur, y, "o", ms=5, color="k", label=T("this file", "ce fichier", " / "))
        ax.set_yticks(y, [short(r) for r in D.order] if ax is axs[0] else [])
        ax.set(xlabel=lab)
        ax.invert_yaxis()
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside upper center", ncol=3, fontsize=8)
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


# ============================================================================== noise validation figures
def load_noise(path):
    """A doe_noise_validation_results.h5: /by_snr and /summary per run, /clean attrs, runs ordered by clean balanced accuracy."""
    with h5py.File(path, "r") as f:
        if not str(f.attrs.get("schema", "")).startswith("doe_noise_validation"):
            raise ValueError("not a noise validation file (validate_noise.py)")
        col = lambda g: {c: (g[c].asstr()[()] if g[c].dtype.kind == "O" else g[c][()]) for c in g}   # noqa: E731
        by = {r: dict(col(g), snr_breakdown_db=float(g.attrs.get("snr_breakdown_db", np.nan))) for r, g in f["by_snr"].items()}
        clean = {r: dict(g.attrs) for r, g in f["clean"].items()}
        summ = {r: col(g) for r, g in f["summary"].items()}
        order = sorted(by, key=lambda r: -np.nan_to_num(float(clean.get(r, {}).get("balanced_accuracy", np.nan)), nan=-1e9))
        return SimpleNamespace(path=path, attrs=dict(f.attrs), by=by, clean=clean, summ=summ, order=order)


def _snr_axis(ax, levels):
    """SNR on x, from clean (left) to noisy (right), with a 'clean' tick before the highest level. Returns its x."""
    levels = sorted(levels, reverse=True)
    x_clean = levels[0] + max(10.0, 0.15 * (levels[0] - levels[-1]))
    ax.set_xticks([x_clean, *levels], [T("clean", "propre"), *[f"{v:g}" for v in levels]])
    ax.set_xlim(x_clean + 5, levels[-1] - 5)   # inverted: clean on the left
    ax.set_xlabel(T("SNR [dB] (absolute)", "SNR [dB] (absolu)"))
    return x_clean


@_figure(registry=NOISE_FIGURES, loader=load_noise)
def fig_noise_metrics(D):
    """Balanced accuracy, TPR, TNR and alarm fraction in stable cases vs SNR: line = mean, band = min-max over the
    realizations, marker on the left = clean, triangle above the balanced accuracy panel = breakdown SNR of each
    indicator (own color, one row per indicator so they never overlap, labelled with its dB)."""
    fig, axs = _grid(4)
    panels = (("balanced_accuracy", T("balanced accuracy", "exactitude équilibrée")), ("TPR", "TPR"), ("TNR", "TNR"),
              ("mean_alarm_fraction_stable", T("alarm fraction, stable cases", "fraction d'alarme, cas stables")))
    for ax, (k, lab) in zip(axs, panels):
        for i, r in enumerate(D.order):
            b, c = D.by[r], RUN_COLOR[i % len(RUN_COLOR)]
            x_clean = _snr_axis(ax, b["snr_db"])
            ax.plot(b["snr_db"], b[k + "_mean"], "o-", ms=4, color=c, label=short(r))
            ax.fill_between(b["snr_db"], b[k + "_min"], b[k + "_max"], color=c, alpha=0.15, lw=0)
            ax.plot([x_clean], [float(D.clean.get(r, {}).get(k, np.nan))], "D", ms=6, mfc="none", mew=1.5, color=c)
            if k == "balanced_accuracy" and np.isfinite(b["snr_breakdown_db"]):
                y = 1.05 + 0.08 * (i + 0.5)   # a row above the data per indicator: same SNR, no overlap
                ax.plot([b["snr_breakdown_db"]], [y], "v", ms=6, color=c, clip_on=False)
                ax.annotate(f"{b['snr_breakdown_db']:g} dB", (b["snr_breakdown_db"], y), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=7, color=c, annotation_clip=False)
        if k == "balanced_accuracy":
            ax.set(ylim=(-0.05, 1.05 + 0.08 * len(D.order)), yticks=np.arange(0, 1.01, 0.2))
        else:
            ax.set(ylim=(-0.05, 1.05))
        ax.set_ylabel(lab, y=0.4 if k == "balanced_accuracy" else 0.5)
    h, l = axs[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], marker="v", ls="", color="grey", ms=6))
    l.append(T("breakdown SNR", "SNR de rupture"))
    fig.legend(h, l, loc="outside upper center", ncol=5, fontsize=8)
    return fig


@_figure(registry=NOISE_FIGURES, loader=load_noise)
def fig_noise_case_matrix(D):
    """Per indicator: cases (rows, by kappa) x SNR levels (columns); color = fraction of the realizations whose outcome
    is right (TP or TN); gray cell = not scored (gray case in mode 'ignore')."""
    fig, axs = _grid(len(D.order))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")
    im = None
    for ax, r in zip(axs, D.order):
        s = D.summ[r]
        levels = sorted(set(s["snr_db"]), reverse=True)
        cases = sorted(set(s["case"]), key=lambda c: (float(np.nanmax(np.where(s["case"] == c, s["kappa"], np.nan))), c))
        grid = np.full((len(cases), len(levels)), np.nan)
        for i, c in enumerate(cases):
            for j, lv in enumerate(levels):
                o = s["outcome"][(s["case"] == c) & (s["snr_db"] == lv)]
                scored = o[np.isin(o, ("TP", "TN", "FP", "FN"))]
                grid[i, j] = np.isin(scored, ("TP", "TN")).mean() if scored.size else np.nan
        im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        tag = {}
        for c in cases:
            m = s["case"] == c
            tag[c] = "g" if np.any(s["is_gray"][m] == 1) else {"stable": "S", "unstable": "U"}.get(s["truth"][m][0], "?")
        kap = {c: float(np.nanmax(np.where(s["case"] == c, s["kappa"], np.nan))) for c in cases}
        ax.set_yticks(range(len(cases)), [f"{kap[c]:.2f} {tag[c]}" for c in cases], fontsize=8)
        ax.set_xticks(range(len(levels)), [f"{v:g}" for v in levels])
        ax.set(title=short(r), xlabel=T("SNR [dB]", "SNR [dB]"))
        ax.tick_params(length=0)
    axs[0].set_ylabel(T(r"case: $\kappa$ and truth", r"cas : $\kappa$ et vérité"))
    fig.colorbar(im, ax=axs, shrink=0.8, label=T("fraction of realizations right", "fraction de réalisations justes"))
    return fig


@_figure(registry=NOISE_FIGURES, loader=load_noise)
def fig_noise_anticipation(D):
    """Median t_det / t_onset of the hits vs SNR (band = min-max over the realizations); 1 = alarm when the amplitude
    reaches the limit. Does the noise cost anticipation?"""
    fig, ax = _fig()
    for i, r in enumerate(D.order):
        b, c = D.by[r], RUN_COLOR[i % len(RUN_COLOR)]
        x_clean = _snr_axis(ax, b["snr_db"])
        ax.plot(b["snr_db"], b["median_t_ratio_mean"], "o-", ms=4, color=c, label=short(r))
        ax.fill_between(b["snr_db"], b["median_t_ratio_min"], b["median_t_ratio_max"], color=c, alpha=0.15, lw=0)
        ax.plot([x_clean], [float(D.clean.get(r, {}).get("median_t_ratio", np.nan))], "D", ms=6, mfc="none", mew=1.5, color=c)
    ax.axhline(1, color="k", ls="--", lw=0.8)
    ax.set(ylabel=T("median $t_{det}\\,/\\,t_{onset}$", "médiane $t_{det}\\,/\\,t_{onset}$"), ylim=(0, 1.1))
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=4, fontsize=8)
    return fig


# ============================================================================== run
def _is_noise(h5_path):
    with h5py.File(h5_path, "r") as f:
        return str(f.attrs.get("schema", "")).startswith("doe_noise_validation"), str(f.attrs.get("gray_mode", "ignore"))


def figs_dir(h5_path):
    """Where the figures of a validation file go: next to it, in figs_validation (+ _gray-<mode> for --gray
    stable|unstable); figs_noise_validation[...] for a noise validation file."""
    noise, mode = _is_noise(h5_path)
    return os.path.join(os.path.dirname(os.path.abspath(h5_path)),
                        ("figs_noise_validation" if noise else "figs_validation") + ("" if mode == "ignore" else f"_gray-{mode}"))


def make_all(h5_path, out_dir=None):
    out_dir = out_dir or figs_dir(h5_path)
    for name, fn in (NOISE_FIGURES if _is_noise(h5_path)[0] else FIGURES).items():
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
    kap = [0.5, 0.7, 0.9, 1.05, 1.2, 1.5, 1.9]   # 1.05: gray
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
            p = fl.require_group(f"{'stable' if k < 1.0 else 'gray' if k == 1.05 else 'unstable'}/case_{i:03d}").create_dataset("Axial_disp__000", data=[0.0])
            p.attrs.update(channel="Axial_disp", t0=0.0, t1=10.0, labeling_strategy="amplitude", labeling_lim_sup_pct=40.0,
                           labeling_lim_inf_pct=10.0, labeling_base_attr="$f_tooth$", labeling_base_scale=1e-3,
                           labeling_signal="Axial_disp", kappa=k, **{"$Ap_start$": 0.005 * k, "$spin_rate$": 12000.0})
    vi.validate(ind, lab, out, reference_h5=lab)
    figs = os.path.join(d, "figs")
    make_all(out, figs)
    assert sorted(os.listdir(figs)) == sorted(n + ".png" for n in FIGURES), os.listdir(figs)
    assert len(FIGURES) == 15 and figs_dir(out) == os.path.join(d, "figs_validation")
    for mode in ("stable", "unstable"):   # a non-default --gray mode: its own file, its own folder, a note on the figure
        o = os.path.join(d, f"out{vi.gray_suffix(mode)}.h5")
        vi.validate(ind, lab, o, reference_h5=lab, gray=mode)
        assert figs_dir(o) == os.path.join(d, f"figs_validation_gray-{mode}")
        make_all(o, os.path.join(d, f"f_{mode}"))
        assert sorted(os.listdir(os.path.join(d, f"f_{mode}"))) == sorted(n + ".png" for n in FIGURES)
        assert FIGURES["ranking"](o)._supxlabel.get_text().startswith("gray cases counted as")
    assert FIGURES["ranking"](out)._supxlabel is None   # the default mode carries no note
    fig = FIGURES["roc"](out)
    assert fig._keep_size == tuple(fig.get_size_inches()) and np.allclose(fig._keep_size, ps.figsize_from_scale(ps.FIGSIZE_SIMPLE, FIGSCALE))
    fc = fig_compare(out, out, figs)
    assert fc._keep_size and any(n.startswith("compare_") for n in os.listdir(figs))
    # noise validation figures: noisy copies of the clean cases at 2 levels x 2 realizations
    import validate_noise as vn
    nind = os.path.join(d, "noise_ind.h5")
    with h5py.File(ind, "r") as src, h5py.File(nind, "w") as f:
        f.attrs.update(noise_layout="multi", snr_ref_case="case_004")
        for snr in (40.0, 10.0):
            for k in (0, 1):
                for c in src:
                    g = f.create_group(f"snr_{snr:06.2f}__{c}__r{k:02d}")
                    g.attrs.update(snr_db=snr, case_source=c, realization=k)
                    for run in ("ind_a", "ind_b"):
                        r = g.create_group(run)
                        r["t"], r["I_t"] = t, src[c][run]["I_t"][()]
                        td = src[c][run]["t_d"][()] if "t_d" in src[c][run] else ([1.0] if snr == 10.0 and k == 0 else [])
                        if len(td):
                            r["t_d"] = td
    nv = os.path.join(d, "noise_val.h5")
    vn.validate_noise(nind, out, nv)
    assert figs_dir(nv) == os.path.join(d, "figs_noise_validation") and len(NOISE_FIGURES) == 3
    make_all(nv, os.path.join(d, "fn"))
    assert sorted(os.listdir(os.path.join(d, "fn"))) == sorted(n + ".png" for n in NOISE_FIGURES)
    fig = NOISE_FIGURES["noise_metrics"](nv)
    assert fig._keep_size and len(fig.axes) >= 4
    try:
        NOISE_FIGURES["noise_metrics"](out)
        raise SystemExit("a clean validation file is not a noise one")
    except ValueError:
        pass
    print("validation_figures selftest OK")


def main():
    global LANGUAGE, FIGSCALE
    p = argparse.ArgumentParser(description="Figures of a doe_validation_results.h5 (read only).")
    p.add_argument("--results", metavar="PATH", help="doe_validation_results.h5")
    p.add_argument("--out-dir", metavar="DIR", help="default: figs_dir(h5): <folder of the h5>/figs_validation[_gray-<mode>]")
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
