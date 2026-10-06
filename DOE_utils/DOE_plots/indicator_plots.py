"""indicator_plots.py — the own figures of an indicator package for ONE case and ONE variant of an experiment.

Re-runs the variant on the case exactly as doe_indicators.py does (same cut, same T_rev / T_modal resolution, same trained
reference) and calls the package's plotting function (plots_maxent_sprt, plots_rms_cv, plots_sst_svd, plots_green_integral /
plots_lyapunov). The packages only draw with pyplot and end with plt.show(): here show is a no-op and the open figures are
collected, then pickled (for the export window of the viewer) and / or saved as PNG. A separate process on purpose: it keeps
the heavy imports and the memory of the runs out of the viewer. The texts of these figures are English (the packages have no
language option).

    python indicator_plots.py --experiment E --case case_007 --variant maxent_revo_dec7_1step --pickle-dir D
    python indicator_plots.py --experiment E --case case_007 --variant V --save-dir figs_indicator/case_007/V --dpi 300
    python indicator_plots.py --experiment E --case case_007 --variant V --show       # figures in windows

prints "[indicator_plots] ..." progress lines; the last one is "[indicator_plots] DONE n figures".
"""
import argparse
import json
import os
import pickle
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UTILS = os.path.dirname(HERE)
for p in (UTILS, os.path.join(UTILS, "DOE_analisis")):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib   # noqa: E402

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np   # noqa: E402

_show = plt.show
plt.show = lambda *args, **kwargs: None   # the packages end with plt.show(): here only after everything is drawn (--show)


def say(msg: str) -> None:
    print(f"[indicator_plots] {msg}", flush=True)


def prepare(exp, case: str, variant: str, t_end: float = 0.0):
    """(ind_id, runner, sig, config, info) of a variant on a case of the experiment data: what doe_indicators.prepare_run
    prepares before calling the runner (same cut, spin of the case, trained reference), so the figures come from exactly the
    run of the stage. t_end > 0 analyses only up to that time (quick tests)."""
    import doe_indicators as di
    import experiment as ex
    start, end = ex.analysis_cut()
    end = float("inf") if end is None else end
    run = next((r for r in ex.indicator_runs(exp) if r["name"] == variant), None)
    if run is None:
        raise ValueError(f"variant '{variant}' is not in experiment {exp.name}")
    settings = {"label_key": "kappa", "cut": (start, min(end, t_end) if t_end else end), "reference_h5": exp.reference,
                "spin_fallback": None}
    prep = di.prepare_run(exp.data_h5, case, run, settings)
    if isinstance(prep, dict):   # the stage would have left an empty result here too
        raise ValueError(f"cannot prepare {variant} on {case}: {prep.get('error') or prep.get('reason') or prep}")
    ind_id, runner, sig, config = prep
    return ind_id, runner, sig, config, {"cfg": config}


def config_mismatch(info: dict, ind_h5: str, case: str, variant: str) -> list:
    """Parameters of the variant now (resolved for this case) that differ from the ones the results file was made with
    (its pp_* attributes): the YAML changed after the run."""
    import h5py
    if not ind_h5 or not os.path.isfile(ind_h5):
        return []
    with h5py.File(ind_h5, "r") as f:
        if case not in f or variant not in f[case]:
            return []
        attrs = dict(f[case][variant].attrs)
    bad = []
    for k, v in info["cfg"]["params_physical"].items():
        old = attrs.get(f"pp_{k}")
        if v is None and (old is None or str(old) in ("None", "nan")):   # an attribute set to None is stored as text
            continue
        try:
            same = old is not None and v is not None and bool(np.allclose(np.asarray(old, float), np.asarray(v, float)))
        except (TypeError, ValueError):
            same = str(old) == str(v)
        if not same:
            bad.append(f"{k}: file {old!r}, YAML now {v!r}")
    return bad


def modal_frequencies(f_modal) -> list:
    """f_modal of the experiment (a number, a list, or None) as a list of positive frequencies [Hz]."""
    v = f_modal if isinstance(f_modal, (list, tuple)) else [f_modal]
    return [float(x) for x in v if x is not None and float(x) > 0]


def sst_frequency_axes(f_modal, fs: float):
    """(f_max, f_slice) of the SST-SVD figures F1-F2c from the modal frequency of the experiment: the top of the axes is
    twice the largest mode (not above the Nyquist frequency of the signal), the slice is at the first mode. (None, None)
    when the experiment has no f_modal: the package keeps its own 250 / 150 Hz."""
    fm = modal_frequencies(f_modal)
    if not fm:
        return None, None
    return min(2.0 * max(fm), fs / 2.0), fm[0]


def draw(ind_id: str, sig, result, config, case: str, scale: float, t_gt, spectrograms: bool = False,
         waterfall: str = "time", f_max=None, f_slice=None):
    """Calls the plotting function of the package; the figures stay open in pyplot."""
    kw = dict(scale=scale, figsize_simple=_presets()[0], figsize_wide=_presets()[1], show=False)
    sig.meta["signal_id"] = case   # the package tags its figures with it
    if ind_id == "MaxEnt_SPRT":
        from MaxEnt_SPRT import plots_maxent_sprt
        plots_maxent_sprt(signal=sig, result=result, show_signal=True, t_gt=t_gt, **kw)
    elif ind_id == "RMS_CV":
        from rms_cv import plots_rms_cv
        plots_rms_cv(signal=sig, result=result, show_signal=True, t_gt=t_gt, **kw)
    elif ind_id == "SST_SVD":
        from ssq_chatter import plots_sst_svd
        plots_sst_svd(signal=sig, result=result, show_signal=True, reference_signal=config.get("reference_signal"),
                      show_spectrograms=spectrograms, waterfall_lines=waterfall,
                      **({"f_max": f_max} if f_max else {}), **({"f_slice": f_slice} if f_slice else {}), **kw)
    elif ind_id == "Green_Integral":
        from green_integral import SignalData as GreenSignal, plots_green_integral, plots_lyapunov
        raw = result.meta["raw_result"]
        x = np.asarray(sig.signal_analysis, float)
        t = np.asarray(sig.t_analysis, float)
        v = sig.meta.get("velocity")
        gsig = GreenSignal(t=t, displacement=x, velocity=np.gradient(x, t) if v is None else np.asarray(v, float), name=case)
        if config.get("func", "Default") == "Lyapunov":
            plots_lyapunov(signal=gsig, result=raw, training_intervals=config["params_physical"].get("training_intervals", []),
                           reference_signal=config.get("reference_signal"), **kw)
        else:
            plots_green_integral(signal=gsig, result=raw, scale=scale, figsize_wide=kw["figsize_wide"], show=False)
    else:
        raise ValueError(f"no plotting function for indicator {ind_id}")


def _presets() -> tuple:
    """(FIGSIZE_SIMPLE, FIGSIZE_WIDE) of article-plot-style (the same presets as the export window)."""
    import plot_style as ps
    return ps.FIGSIZE_SIMPLE, ps.FIGSIZE_WIDE


def collect(case: str) -> list:
    """[(name, Figure)] of every figure the package left open, in creation order; the case tag is dropped from the name."""
    out = []
    for n in plt.get_fignums():
        fig = plt.figure(n)
        name = fig.get_label() or str(n)
        name = re.sub(rf"\s*—\s*{re.escape(case)}$", "", name) or str(n)
        if name.isdigit():   # a figure the package did not name: its title
            sup = fig._suptitle.get_text() if getattr(fig, "_suptitle", None) is not None else ""
            title = sup or (fig.axes[0].get_title() if fig.axes else "")
            name = f"{title} (figure {name})" if title else f"Figure {name}"
        out.append((name, fig))
    return out


def _selftest():
    """The names of the collected figures and the comparison with the parameters of a results file (no indicator is run)."""
    import tempfile
    import h5py
    plt.close("all")
    f1 = plt.figure("F0a — Training — case_007")
    f1.add_subplot(111)
    f2 = plt.figure()
    f2.add_subplot(111).set_title("Tool Velocity")
    names = [n for n, _ in collect("case_007")]
    assert names == ["F0a — Training", "Tool Velocity (figure %d)" % f2.number], names
    plt.close("all")
    path = os.path.join(tempfile.mkdtemp(prefix="indplots_"), "ind.h5")
    with h5py.File(path, "w") as h:
        g = h.create_group("case_000/v1")
        g.attrs.update(pp_N_rev_window=7, pp_T_rev=0.0049, pp_cv_threshold="None")
    info = {"cfg": {"params_physical": {"N_rev_window": 7, "T_rev": 0.0049, "cv_threshold": None}}}
    assert config_mismatch(info, path, "case_000", "v1") == []
    info["cfg"]["params_physical"]["N_rev_window"] = 4
    assert len(config_mismatch(info, path, "case_000", "v1")) == 1 and config_mismatch(info, "", "case_000", "v1") == []
    assert sst_frequency_axes(150.0, 1000.0) == (300.0, 150.0) and sst_frequency_axes([150, 250], 1000.0) == (500.0, 150.0)
    assert sst_frequency_axes(250.0, 600.0) == (300.0, 250.0)                      # not above the Nyquist frequency
    assert sst_frequency_axes(None, 1000.0) == (None, None) and sst_frequency_axes([], 1000.0) == (None, None)
    print("indicator_plots selftest OK")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--scale", type=float, default=1.5, help="article-plot-style scale (FIGSCALE of the export window)")
    ap.add_argument("--ind-h5", default="", help="indicator results of the run: warns when the YAML no longer matches")
    ap.add_argument("--pickle-dir", default="", help="one .pkl per figure + index.json (for the export window)")
    ap.add_argument("--save-dir", default="", help="save the figures as PNG here")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--show", action="store_true", help="open the figures in windows")
    ap.add_argument("--spectrograms", action="store_true",
                    help="SST_SVD only: also the STFT / SST spectrograms, slices and 3D waterfalls (F1-F2c; heavy)")
    ap.add_argument("--f-max", type=float, default=0.0, help="SST_SVD only: top of the frequency axes of F1-F2c [Hz] (default: "
                    "2 x the modal frequency of the experiment, not above the Nyquist frequency)")
    ap.add_argument("--f-slice", type=float, default=0.0, help="SST_SVD only: frequency of the slices F1b / F2b [Hz] (default: "
                    "the modal frequency of the experiment)")
    ap.add_argument("--waterfall", default="time", choices=("time", "freq", "both", "surface", "wire"),
                    help="SST_SVD only: lines of the 3D waterfalls (with --spectrograms)")
    ap.add_argument("--t-end", type=float, default=0.0, help="analyse the signal only up to this time [s] (quick tests; 0 = all)")
    if "--selftest" in (argv if argv is not None else sys.argv):
        return _selftest() or 0
    a = ap.parse_args(argv)

    import experiment as ex
    exp = ex.load(a.experiment)
    if a.variant not in exp.indicators["specs"]:
        say(f"ERROR variant '{a.variant}' is not in experiment {exp.name}")
        return 2
    say(f"{exp.name} / {a.case} / {a.variant}: preparing")
    ind_id, runner, sig, config, info = prepare(exp, a.case, a.variant, a.t_end)
    for line in config_mismatch(info, a.ind_h5, a.case, a.variant):
        say(f"WARNING the experiment changed since the results were made: {line}")
    say(f"running {ind_id} ({len(sig.signal_analysis)} samples)")
    result = runner(sig, config)
    onset = ex.label_info(exp.label["out"]).get(a.case, {}).get("t_onset")
    t_gt = float(onset) if onset is not None and np.isfinite(onset) else None
    say("drawing")
    plt.close("all")
    f_max, f_slice = sst_frequency_axes(exp.indicators.get("f_modal"), float(sig.fs)) if ind_id == "SST_SVD" else (None, None)
    if a.f_max:
        f_max = min(a.f_max, float(sig.fs) / 2.0)
    f_slice = a.f_slice or f_slice
    if ind_id == "SST_SVD" and a.spectrograms:
        say(f"spectrogram axes: up to {f_max or 250:g} Hz, slices at {f_slice or 150:g} Hz")
    draw(ind_id, sig, result, config, a.case, a.scale, t_gt, a.spectrograms, a.waterfall, f_max, f_slice)
    figs = collect(a.case)
    say(f"{len(figs)} figures")
    names = []
    for i, (name, fig) in enumerate(figs):
        stem = f"{i:02d}_" + re.sub(r"[^\w\-]+", "_", name).strip("_")
        if a.save_dir:
            os.makedirs(a.save_dir, exist_ok=True)
            fig.savefig(os.path.join(a.save_dir, stem + ".png"), dpi=a.dpi, bbox_inches="tight")
        if a.pickle_dir:
            os.makedirs(a.pickle_dir, exist_ok=True)
            try:
                with open(os.path.join(a.pickle_dir, stem + ".pkl"), "wb") as fh:
                    pickle.dump(fig, fh)
                names.append({"name": name, "file": stem + ".pkl"})
            except Exception as exc:   # e.g. a local function inside the figure: the PNG is still there
                say(f"WARNING {name}: cannot be pickled ({type(exc).__name__}: {exc})")
        if not a.show:
            plt.close(fig)   # the grids of a package are big: free each one as soon as it is stored
    if a.pickle_dir:
        with open(os.path.join(a.pickle_dir, "index.json"), "w", encoding="utf-8") as fh:
            json.dump({"case": a.case, "variant": a.variant, "indicator": ind_id, "figures": names}, fh)
    say(f"DONE {len(figs)} figures")
    if a.show:
        _show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
