"""Check of the app forms (v2) on a temp copy of the experiments: new experiment from scratch (a single Ap, η
from a model, an Ap = inf refused), settings, copy, simulation form (a config run written in full), indicator
table, import, standardize, dry-run, log viewer, mark up to date, compare. Real data are only read; captures in
%TEMP%/app_dialog_shots for a visual check."""
import os
import shutil
import sys
import tempfile

import h5py

if hasattr(sys.stdout, "reconfigure"):   # the messages say η: a cp1252 console would refuse it
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DOE = os.path.dirname(os.path.abspath(__file__))
SHOTS = os.path.join(tempfile.gettempdir(), "app_dialog_shots")   # captures for a visual check
os.makedirs(SHOTS, exist_ok=True)
sys.path.insert(0, DOE)
import experiment as ex  # noqa: E402
import launcher as L  # noqa: E402

TR, VA, N9 = "train_1DOF150_n12098_k0.5-2", "val_1DOF150_n12098_k0.51-1.99", "train_1DOF150_n9000_k1-1.5"
tmp = tempfile.mkdtemp(prefix="app_dialogs_")
os.makedirs(os.path.join(tmp, "experiments"))   # empty: only the presets (the user's experiments are not touched)
shutil.copy(ex.VARIANTS_FILE, os.path.join(tmp, "experiments"))
dr = ex._doe_runner()
cfg_tmp = os.path.join(tmp, "configs")
shutil.copytree(ex.CONFIGS_DIR, cfg_tmp)
real_cfg = ex.CONFIGS_DIR
ex.set_root(os.path.join(tmp, "experiments"))
ex.CONFIGS_DIR = dr.CONFIGS_DIR = cfg_tmp
base = os.path.join(tmp, "data")                       # a case folder for new simulations (never simulated here)
os.makedirs(os.path.join(base, "1DOF_150Hz", "p"))
launched = []
L.launch = lambda kind, script, args: launched.append((script, args)) or (None, None)   # no real tools
L.open_console = lambda argv, cwd=None, close=False: launched.append(("console", argv, close))
# the experiments of the check: N9 reads an older-style config of configs/; TR and VA are made below by importing
# the real training / validation folders (only read; records go to the temp copy)
with open(os.path.join(cfg_tmp, "cfg_n9000.yaml"), "w") as fh:
    fh.write(f"base_dir: {base.replace(chr(92), '/')}\ncase: 1DOF_150Hz\ndoe_name: DOE_n9000\nmode: sweep\nnb_proc: 2\n"
             "sweep:\n  $Ap_start$: [0.008056]\n  $Ap_end$: [0.008056]\n  $spin_rate$: 9000.0\n"
             "  $f_tooth$: 0.05\n  $dxl_size$: 0.0002\n  $nb_dt_rev$: 200\n")
ex.create_experiment(N9, [{"config": "cfg_n9000"}], stages_on=ex.FLOWS["Indicators"])
import numpy as np  # noqa: E402


def fake_doe(folder, kappas, labelled):
    """An already extracted DOE folder (temp): case folders, doe_results.h5 and, if labelled, an amplitude
    reference_dataset (the check never depends on where the user keeps the real data)."""
    t = np.linspace(0, 1, 50)
    with h5py.File(os.path.join(_mk(folder), "doe_results.h5"), "w") as h:
        for i, k in enumerate(kappas):
            os.makedirs(os.path.join(folder, str(i), "1DOF_150Hz"))
            open(os.path.join(folder, str(i), "1DOF_150Hz", "sens_out.hdf5"), "w").close()
            g = h.create_group(f"case_{i:03d}")
            g.attrs.update({"$Ap_start$": k * 0.0086, "$Ap_end$": k * 0.0086, "$spin_rate$": 12098.28,
                            "$dxl_size$": 2e-4, "$nb_dt_rev$": 200.0, "$f_tooth$": 0.05, "kappa": k})
            for s in ("Axial_disp", "Axial_vel"):
                g.create_dataset(f"{s}/time", data=t)
                g.create_dataset(f"{s}/values", data=np.sin(20 * t) * k)
    if labelled:
        with h5py.File(os.path.join(folder, "reference_dataset_amp.h5"), "w") as h:
            for i, k in enumerate(kappas):
                p = h.create_group(f"{'stable' if k < 1 else 'unstable'}/case_{i:03d}").create_dataset(
                    "Axial_disp__000", data=[0.0])
                p.attrs.update(labeling_strategy="amplitude", labeling_signal="Axial_disp", kappa=k)


def _mk(d):
    os.makedirs(d, exist_ok=True)
    return d


TR_DIR, VA_DIR = os.path.join(base, "TRAIN"), os.path.join(base, "VALID")
fake_doe(TR_DIR, [0.6, 0.8, 1.0, 1.2, 1.4], True)
fake_doe(VA_DIR, [0.7, 0.9, 1.1], False)

import tkinter as tk  # noqa: E402
from tkinter import messagebox  # noqa: E402
os.environ["DOE_VIEWER_LOCK"] = os.path.join(tmp, "viewer_lock.json")   # never talk to a viewer the user has open
from PIL import ImageGrab  # noqa: E402
import ctypes  # noqa: E402
F = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100.0
messagebox.askyesno = lambda *a, **k: True                       # confirmations accepted
errors = []
L.App._msg = lambda self, title, text, kind="info": errors.append((title, text, kind))


def shot(win, name):
    win.attributes("-topmost", True)
    win.lift()
    win.update()
    win.after(400)
    win.update()
    x, y, w, h = win.winfo_rootx(), win.winfo_rooty(), win.winfo_width(), win.winfo_height()
    ImageGrab.grab(bbox=tuple(int(v * F) for v in (x, y, x + w, y + h))).save(os.path.join(SHOTS, name))


def check_noise():
    """Noise validation (PLAN_noise_validation.md): the Noise form (old mode untouched, cases picked from the list, the
    keys written), the Noise validation form (only out) and the new box in the diagram, on the fake labelled training."""
    e = ex.load(TR)
    stages0 = ex.own_yaml(TR)["stages"]
    f = L.NoiseForm(app, e)
    assert not f.multi.get() and f.levels.get() == "80, 60, 40, 30, 20, 10" and f.count.cget("text") == ""
    f._ok()
    assert not ex.own_yaml(TR).get("noise") and not ex.noise_multi(ex.load(TR))        # saving the old mode changes nothing
    f = L.NoiseForm(app, ex.load(TR))
    f.multi.set(True)
    f.levels.set("40, 20")
    f.nreal.set("2")
    f._pick()                                                                         # list of cases with kappa and label
    lb = f.picker.winfo_children()[0]
    assert lb.size() == 5 and "η 0.600" in lb.get(0) and "stable" in lb.get(0), lb.get(0)
    lb.selection_clear(0, "end")
    lb.selection_set(1)
    lb.selection_set(3)
    f._pick_ok()
    assert f.cases.get() == "case_001, case_003" and "8 noisy copies" in f.count.cget("text"), f.count.cget("text")
    shot(f.win, "noise_form.png")
    f.nreal.set("0")
    f._ok()                                                                           # refused: the form stays, says why
    assert errors.pop()[2] == "error" and ex.own_yaml(TR).get("noise") is None
    f.nreal.set("2")
    f.seed.set("7")
    f._ok()
    assert ex.own_yaml(TR)["noise"] == {"cases": ["case_001", "case_003"], "snr_list": [40, 20], "realizations": 2,
                                        "seed": 7}, ex.own_yaml(TR)["noise"]
    assert ex.noise_multi(ex.load(TR)) and not ex._noise_problems(ex.load(TR), [])
    f = L.NoiseForm(app, ex.load(TR))                                                 # reopened: the keys come back
    assert f.multi.get() and f.cases.get() == "case_001, case_003" and f.levels.get() == "40, 20" and f.nreal.get() == "2"
    f.multi.set(False)
    f._ok()
    assert "cases" not in ex.own_yaml(TR)["noise"] and ex.own_yaml(TR)["noise"]["seed"] == 7   # back to the old mode
    ex.save_section(TR, "noise", {"cases": "all", "snr_list": [30, 10]})
    # Noise validation: only out; the box is in the diagram and its card says there is nothing yet
    d = ex.own_yaml(TR)
    d["stages"] = stages0 + ["noise", "noise_indicators", "noise_validate"]
    ex.yaml_save(d, ex.exp_path(TR))
    ex.reload()
    nv = L.NoiseValidateForm(app, ex.load(TR))
    nv.out.set(os.path.join(tmp, "nv.h5"))
    nv._ok()
    assert ex.stages(ex.load(TR))["noise_validate"].outputs == [os.path.normpath(os.path.join(tmp, "nv.h5"))]
    # realizations_run in both forms and resume in Noise indicators: written only when filled, read back, the command follows
    nv = L.NoiseValidateForm(app, ex.load(TR))
    assert nv.rr.get() == "" and ex.own_yaml(TR)["noise_validate"] == {"out": os.path.normpath(os.path.join(tmp, "nv.h5")).replace("\\", "/")}
    nv.rr.set("0")
    nv._ok()
    assert ex.own_yaml(TR)["noise_validate"]["realizations_run"] == [0] and "out" in ex.own_yaml(TR)["noise_validate"]
    assert ex.stages(ex.load(TR))["noise_validate"].cmds[0][-2:] == ["--realizations", "0"]
    ni = L.NoiseIndicatorsForm(app, ex.load(TR))                                      # its own form now (the variants: a button)
    assert ni.rr.get() == "" and not ni.resume.get()
    ni.rr.set("0, 2")
    ni.resume.set(True)
    shot(ni.win, "noise_indicators_form.png")
    ni._ok()
    assert ex.own_yaml(TR)["noise_indicators"] == {"realizations_run": [0, 2], "resume": True}, ex.own_yaml(TR)["noise_indicators"]
    c = ex.stages(ex.load(TR))["noise_indicators"].cmds[0]
    assert c[-5:] == ["--no-signals", "--realizations", "0", "2", "--resume"], c
    ni = L.NoiseIndicatorsForm(app, ex.load(TR))
    assert ni.rr.get() == "0 2" and ni.resume.get()
    ni.rr.set("a")
    ni._ok()                                                                          # refused: the form stays, says why
    assert errors.pop()[2] == "error" and ex.own_yaml(TR)["noise_indicators"]["realizations_run"] == [0, 2]
    ni.rr.set("")
    ni.resume.set(False)
    ni._ok()
    assert ex.own_yaml(TR).get("noise_indicators") is None                            # emptied: the section leaves the YAML
    ex.save_section(TR, "noise_validate", None)
    lay = L.diagram_layout(list(ex.stages(ex.load(TR))), False)
    assert lay["noise_validate"][0] > lay["noise_indicators"][0] and lay["noise_validate"][1] == lay["noise"][1]
    app.select(TR, "noise_validate")
    assert ex.stage_summary(ex.load(TR), "noise_validate") == [("no results yet", None)]
    shot(root, "noise_main.png")
    # the arrows of the noise stages say what travels along them, and the note of noise_indicators is in its card
    assert L.label_anchor([0, 0, 10, 0, 10, 40]) == (14, 27, "w") and L.label_anchor([0, 0, 30, 0]) == (15, -8, "s")
    app.redraw(True)
    texts = [app.canvas.itemcget(i, "text") for i in app.canvas.find_all() if app.canvas.type(i) == "text"]
    for lab in ("thresholds", "noisy signals", "I(t) on noise", "clean truth"):
        assert lab in texts, (lab, texts)
    assert "does not need the Indicators results" in ex.STAGE_INFO["noise_indicators"][0]
    assert "does not need" in L.BOX_NOTES["noise_indicators"]
    d["stages"] = stages0                                                             # leave the experiment as it was
    ex.yaml_save(d, ex.exp_path(TR))
    ex.save_section(TR, "noise", None)
    print("noise forms OK")


def check_run_only():
    """'Run only…' (indicators / noise_indicators): the button follows the state of the stage, the form asks for variants, a
    partial run needs an up-to-date stage and starts the console with --only."""
    import time
    e = ex.load(N9)
    st = ex.stages(e)["indicators"]
    out = st.outputs[0]
    made = []                                                                         # inputs the stage needs: fake, removed at the end
    for p in [x for x in st.inputs if x and not os.path.exists(x)]:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with h5py.File(p, "w") as fh:
            if p == st.inputs[0]:
                fh.create_group("case_000")                                           # one case: 'Run missing' has work
        made.append(p)
    time.sleep(1.2)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    h5py.File(out, "w").close()
    ex.write_record(e, "indicators", {"stage": "indicators", "status": "done", "start": time.time(), "end": time.time(),
                                      "hash": st.hash})
    ex.reload()
    app.select(N9, "indicators")
    app.root.update()
    assert not app.stage_btns["only"].instate(["disabled"]), ex.status(ex.load(N9))["indicators"]
    # 'Run missing' (PLAN_ronda3 T5): the tasks not in the results file; a record without its configuration -> it asks
    if st.inputs[0] in made:
        nv = len(ex.load(N9).indicators["variants"])
        assert ex.missing_tasks(ex.load(N9), "indicators") == (nv, nv) and ex.extension_problems(ex.load(N9), "indicators") is None
        assert not app.stage_btns["missing"].instate(["disabled"])
        app.run_missing()
        assert launched[-1][0] == "console" and launched[-1][1][3:5] == [N9, "indicators"] and "--resume" in launched[-1][1]
        assert any(t.startswith(f"missing: {nv} of {nv} tasks") for t, _ in ex.stage_summary(ex.load(N9), "indicators"))
        print("run missing OK")
    app.select(N9, "label_build")
    app.root.update()
    assert app.stage_btns["only"].instate(["disabled"]) and app.stage_btns["missing"].instate(["disabled"])   # indicator stages only
    f = L.RunOnlyForm(app, ex.load(N9), "indicators")
    assert list(f.vars) == list(ex.load(N9).indicators["variants"])
    shot(f.win, "run_only_form.png")
    n = len(launched)
    f._ok()                                                                           # nothing ticked: refused, says why
    assert errors.pop()[2] == "error" and len(launched) == n
    first = next(iter(f.vars))
    f.vars[first].set(True)
    f._ok()
    argv = launched[-1][1]
    assert launched[-1][0] == "console" and argv[argv.index("--only") + 1] == first and argv[3:5] == [N9, "indicators"], argv
    # the code notice: shown on the card, 'Mark up to date' (enabled even if the stage is done) dismisses it
    pkgs = sorted({ex.CODE_PKG[sp["indicator"]] for sp in ex.load(N9).indicators["specs"].values()})
    ex.write_record(ex.load(N9), "indicators", {"stage": "indicators", "status": "done", "start": time.time(), "end": time.time(),
                                                "hash": st.hash, "code": {p: 0.0 for p in pkgs}})
    assert sorted(ex.indicator_code_changed(ex.load(N9), "indicators")) == pkgs
    card = [t for t, _ in ex.stage_summary(ex.load(N9), "indicators")]
    assert card[0].startswith("indicator code changed since this run") and "Mark up to date" in card[0], card
    app.select(N9, "indicators")
    app.root.update()
    assert not app.stage_btns["accept"].instate(["disabled"]) and ex.status(ex.load(N9))["indicators"][0] == "done"
    h = ex.stages(ex.load(N9))["indicators"].hash
    app.mark_uptodate()
    assert ex.indicator_code_changed(ex.load(N9), "indicators") == [] and ex.status(ex.load(N9))["indicators"][0] == "done"
    assert ex.stages(ex.load(N9))["indicators"].hash == h
    app.root.update()
    assert app.stage_btns["accept"].instate(["disabled"])                              # nothing left to accept
    ex.write_record(ex.load(N9), "indicators", {"stage": "indicators", "status": "failed", "start": time.time(), "exit_code": 1})
    f = L.RunOnlyForm(app, ex.load(N9), "indicators")                                 # not up to date: refused
    f.vars[first].set(True)
    n = len(launched)
    f._ok()
    assert errors.pop()[2] == "error" and len(launched) == n
    os.remove(ex.record_path(ex.load(N9), "indicators"))
    for p in [out] + made:
        os.remove(p)
    print("run only OK")


def check_ramps():
    """Ramps of Ap (PLAN_ramps.md): create (mm, η, mixed with constant cases), SLD picker in ramps mode, edit,
    copy, dry-run, a decreasing ramp on a one-way workpiece, import / standardize of an .h5 with ramps."""
    # new experiment: Ap in mm, a ramp + a constant case in the same run
    nd = L.NewExperimentDialog(app)
    fr = nd.frame
    nd.flow.set("Validation against a reference")
    nd.ref.set(TR)
    fr.base_dir.set(base)
    fr.case.set("1DOF_150Hz")
    fr.spins.set("12098.28")
    fr.depths.set("5, 9")
    fr.depths_end.set("15, 9")
    fr.ap_mode.set("model_at_spin")
    fr.ap_model.set("1DOF_150")
    nd.name.set("ramps_mm")
    fr.doe_name.set("ramps_mm")
    assert fr.preview(), fr.out.get("1.0", "end")
    txt = fr.out.get("1.0", "end")
    assert "Ap end" in txt and " ramp " in txt and "const" in txt and "1 ramp case(s): no repeat check" in txt, txt
    shot(nd.win, "ramps_new_mm.png")
    nd._ok()
    e = ex.load("ramps_mm")
    sw = e.runs[0].cfg["sweep"]
    assert sw["$Ap_start$"] == [0.005, 0.009] and sw["$Ap_end$"] == [0.015, 0.009], sw
    assert any("ramp Ap 5.0000 -> 15.0000 mm" in t for t, _ in ex.dry_run(e)), ex.dry_run(e)
    assert ex.summary(e)["ramps"] == 1
    # new experiment in kappa: one ramp kappa 0.6 -> 1.6 (Ap = kappa x limit at n)
    nd = L.NewExperimentDialog(app)
    fr = nd.frame
    fr.base_dir.set(base)
    fr.case.set("1DOF_150Hz")
    fr.spins.set("12098.28")
    fr.unit.set("η")
    fr.depths.set("0.6")
    fr.depths_end.set("1.6")
    fr.ap_mode.set("model_at_spin")
    fr.ap_model.set("1DOF_150")
    nd.flow.set("Simulation only")
    nd._propose()
    assert nd.name.get().startswith("sim_1DOF150_n12098_k0.6-1.6"), nd.name.get()
    assert fr.preview(), fr.out.get("1.0", "end")
    nd._ok()
    k = ex.load(nd.name.get())
    lim = ex._sld().ap_lim("1DOF_150", 12098.28) * 1e-3
    s2 = k.runs[0].cfg["sweep"]
    assert abs(s2["$Ap_start$"][0] - 0.6 * lim) < 1e-8 and abs(s2["$Ap_end$"][0] - 1.6 * lim) < 1e-8, s2
    rows = ex.case_rows(k.runs[0].cfg)
    assert rows[0]["ramp"] and abs(rows[0]["kappa_start"] - 0.6) < 1e-6 and abs(rows[0]["kappa_end"] - 1.6) < 1e-6
    # the simulation form loads the ramp back (Ap mm + ends), and 'load values from' too
    sf = L.SimulationForm(app, e)
    assert sf.frame.depths.get() == "5, 9" and sf.frame.depths_end.get() == "15, 9", (sf.frame.depths.get(),
                                                                                     sf.frame.depths_end.get())
    shot(sf.win, "ramps_simulation_form.png")
    sf.win.destroy()
    nd = L.NewExperimentDialog(app)
    nd.src.set("experiment: ramps_mm")
    assert nd.frame.depths_end.get() == "15, 9"
    # SLD picker, ramps mode: two clicks = a ramp, right click removes it, Fill / Propose / Accept, Use
    nd.frame.base_dir.set(base)
    pk = L.SldPicker(app, nd.frame)
    assert pk.mode.get() == "ramps" and pk.ramps == [(5.0, 15.0)] and pk.aps == [9.0], (pk.ramps, pk.aps)
    pk.model.set("1DOF_150")
    pk.r_unit.set("Ap [mm]")
    pk.new_view()

    class Ev:
        def __init__(self, y, b=1):
            self.inaxes, self.ydata, self.button = pk.ax, y, b
    pk.ax.set_xlim(11000, 13000)                      # a zoom: kept while ramps are added or removed
    pk.on_click(Ev(12.0))
    assert pk._pending == 12.0 and pk.ax.get_xlim() == (11000, 13000)
    pk.on_click(Ev(6.0))
    assert (12.0, 6.0) in pk.ramps and pk._pending is None and pk.ax.get_xlim() == (11000, 13000), pk.ramps
    pk.on_click(Ev(7.0, 3))                           # right click: the ramp 12 -> 6 contains 7
    assert (12.0, 6.0) not in pk.ramps and (5.0, 15.0) in pk.ramps
    pk.new_view()
    pk.r_unit.set("η")
    pk.new_view()
    pk.r_from.set("0.6"), pk.r_to.set("0.9"), pk.r_n.set("2"), pk.span.set("0.8")
    pk.fill()
    lim_mm = pk.limit()
    new = [r for r in pk.ramps if r != (5.0, 15.0)]
    assert len(new) == 2 and abs(new[0][0] - round(0.6 * lim_mm, 4)) < 1e-9 and abs(new[0][1] - round(1.4 * lim_mm, 4)) < 1e-9, new
    pk.span.set("-0.5")
    pk.r_from.set("1.2"), pk.r_to.set("1.6"), pk.r_n.set("3")
    pk.propose()
    assert len(pk.r_proposed) == 3 and all(r[1] < r[0] for r in pk.r_proposed), pk.r_proposed
    first = list(pk.r_proposed)
    pk.seed.set("2")
    pk.propose()                                      # a new proposal replaces the previous one
    assert len(pk.r_proposed) == 3 and not set(first) & set(pk.ramps) - set(pk.r_proposed)
    pk.accept_proposed()
    assert len(pk.r_accepted) == 3 and not pk.r_proposed
    lines = pk.lst.get(0, "end")
    assert any("[accepted] ramp Ap" in x and "η" in x for x in lines), lines
    assert any("crosses 1" in x for x in lines), lines
    shot(pk.win, "ramps_sld_picker.png")
    pk.lst.selection_clear(0, "end")
    pk.lst.selection_set(len(pk.aps))                 # the first ramp of the list
    gone = pk._items[len(pk.aps)]
    pk.remove_selected()
    assert gone[1] not in pk.ramps
    n_ramps = len(pk.ramps)
    pk.use()
    st, en = ex._values(nd.frame.depths.get()), ex._values(nd.frame.depths_end.get())
    assert len(st) == len(en) == 1 + n_ramps and st[0] == en[0] == 9.0, (st, en)
    nd.win.destroy()
    # copy of a ramp experiment keeps the ramps; the copy's dry-run shows them
    ex.copy_experiment("ramps_mm", "ramps_copy", doe_suffix="_c")
    assert ex.load("ramps_copy").runs[0].cfg["sweep"]["$Ap_end$"] == [0.015, 0.009]
    # a decreasing ramp on a case whose workpiece only makes Ap grow: refused with the reason
    os.makedirs(os.path.join(base, "1DOF_150Hz", "in"), exist_ok=True)
    with open(os.path.join(base, "1DOF_150Hz", "in", "1DOF_150Hz-db_def.py"), "w") as fh:
        fh.write("def h_max_truncado(u, v): pass\n")
    nd = L.NewExperimentDialog(app)
    fr = nd.frame
    fr.base_dir.set(base)
    fr.case.set("1DOF_150Hz")
    fr.doe_name.set("down")
    fr.spins.set("12098.28")
    fr.depths.set("12")
    fr.depths_end.set("5")
    assert not fr.preview() and "only makes Ap grow" in fr.out.get("1.0", "end"), fr.out.get("1.0", "end")
    shot(nd.win, "ramps_decreasing_refused.png")
    with open(os.path.join(base, "1DOF_150Hz", "in", "1DOF_150Hz-db_def.py"), "w") as fh:
        fh.write(ex.RAMP_MARK + "\n")
    assert fr.preview(), fr.out.get("1.0", "end")
    nd.win.destroy()
    os.remove(os.path.join(base, "1DOF_150Hz", "in", "1DOF_150Hz-db_def.py"))
    # import of an extracted folder with a ramp (its 'kappa' = start is ignored) and standardize of an .h5 with one
    rd = os.path.join(base, "RAMPS")
    fake_doe(rd, [0.7], False)
    with h5py.File(os.path.join(rd, "doe_results.h5"), "a") as h:
        g = h.create_group("case_001")
        g.attrs.update({"$Ap_start$": 0.005, "$Ap_end$": 0.015, "$spin_rate$": 12098.28, "$dxl_size$": 2e-4,
                        "$nb_dt_rev$": 200.0, "$f_tooth$": 0.05, "kappa": 0.58, "kappa_start": 0.58, "kappa_end": 1.74})
        for s_ in ("Axial_disp", "Axial_vel"):
            g.create_dataset(f"{s_}/time", data=np.linspace(0, 1, 50))
            g.create_dataset(f"{s_}/values", data=np.zeros(50))
    os.makedirs(os.path.join(rd, "1", "1DOF_150Hz"))
    open(os.path.join(rd, "1", "1DOF_150Hz", "sens_out.hdf5"), "w").close()
    from tkinter import filedialog
    filedialog.askdirectory = lambda **k_: rd
    im = L.ImportDialog(app)
    txt = im.prev.get("1.0", "end")
    assert "2 cases (1 ramps)" in txt and "η 0.58-1.74" in txt and "ramp case_001: Ap 5 -> 15 mm" in txt, txt
    shot(im.win, "ramps_import.png")
    im.name.set("ramps_imported")
    im._ok()
    ri = ex.load("ramps_imported")
    assert ex.summary(ri)["ramps"] == 1 and "(1 ramps)" in ri.cfg["description"]
    assert any("ramp case_001" in t for t, _ in ex.stage_summary(ri, "extract")), ex.stage_summary(ri, "extract")
    with h5py.File(os.path.join(rd, "doe_results.h5"), "a") as h:
        for k_ in ("kappa", "kappa_start", "kappa_end"):
            del h["case_001"].attrs[k_]
    filedialog.askopenfilename = lambda **k_: os.path.join(rd, "doe_results.h5")
    sd = L.StandardizeDialog(app)
    assert sd.rep["missing"]["kappa"] == ["case_001"] and sd.rep["ramps"] == ["case_001"], sd.rep
    sd.ap_mode.set("manual")
    sd.ap_manual.set("8.6")
    sd.create.set(False)
    shot(sd.win, "ramps_standardize.png")
    sd._ok()
    with h5py.File(os.path.join(rd, "doe_results.h5"), "r") as h:
        a = h["case_001"].attrs
        assert abs(a["eta_start"] - 5 / 8.6) < 1e-9 and abs(a["eta_end"] - 15 / 8.6) < 1e-9 and "eta" not in a and "kappa" not in a
    print("ramps OK: new (mm, η, mixed), SLD picker ramps mode, form, copy, decreasing refused, import, standardize")


root = tk.Tk()
root.geometry("1300x860+10+10")
app = L.App(root, auto_refresh=False)
try:
    # ---- viewer / planner get their arguments as a list: a path with spaces arrives without quotes
    assert L.split_args('--h5 "D:/a b/c.h5"') == ["--h5", "D:/a b/c.h5"]
    app.view_h5("D:/x y/doe_results.h5")
    assert launched[-1] == ("DOE_plots/doe_unified_selector.py", ["--h5", "D:/x y/doe_results.h5"]), launched[-1]
    print("viewer arguments OK")
    # one viewer window: with one open the file is sent to it (no new process); without, a new viewer is started
    sent = []
    L.send_to_viewer = lambda paths: sent.append(list(paths)) or True
    n_launched = len(launched)
    app.view_h5("D:/x y/doe_results.h5")
    assert sent == [["D:/x y/doe_results.h5"]] and len(launched) == n_launched and "viewer window" in app.status_msg.get()
    L.send_to_viewer = lambda paths: False
    app.view_h5("D:/x y/doe_results.h5")
    assert len(launched) == n_launched + 1
    print("viewer reuse OK")
    # ---- import: preview of the folder before OK, name = folder, choice of the labelled dataset
    from tkinter import filedialog
    filedialog.askdirectory = lambda **k: TR_DIR
    im = L.ImportDialog(app)
    assert im.name.get() == os.path.basename(TR_DIR), im.name.get()
    im.lab.set("reference_dataset_amp.h5")
    txt = im.prev.get("1.0", "end")
    assert "5 case folders" in txt and "chosen labelled dataset reference_dataset_amp.h5" in txt and "Label build" in txt, txt
    shot(im.win, "v2_import_preview.png")
    im.name.set(TR)
    im._ok()
    t = ex.load(TR)
    assert t.label["out"].endswith("reference_dataset_amp.h5") and ex.status(t)["label_build"][0] == "done"
    ex.import_dir(VA, VA_DIR, reference=TR)
    print("import with preview OK:", TR, "+", VA)
    # ---- import a folder simulated but never extracted (no .h5): Extract runs from the app
    raw = os.path.join(base, "RAW")
    for i, ap in enumerate((0.004, 0.012)):
        c = os.path.join(raw, str(i), "1DOF_150Hz")
        os.makedirs(c)
        open(os.path.join(c, "sens_out.hdf5"), "w").close()
        with open(os.path.join(c, "var_val.py"), "w") as fh:
            fh.write(f"var_val = {{'$Ap_start$': {ap}, '$Ap_end$': {ap}, '$spin_rate$': 12000.0}}\n")
    filedialog.askdirectory = lambda **k: raw
    im = L.ImportDialog(app)
    assert im.h5.get() == L.NOT_EXTRACTED
    im.ap_mode.set("manual")
    im.ap_manual.set("8")
    txt = im.prev.get("1.0", "end")
    assert "not extracted yet: 2 cases" in txt and "η 0.5-1.5" in txt, txt
    shot(im.win, "v2_import_not_extracted.png")
    im._ok()
    rw = ex.load("RAW")
    assert ex.status(rw)["extract"][0] == "pending" and not ex.run_blockers(rw, "extract")
    assert "would delete" in ex.run_blockers(rw, "simulate")[0]
    print("import not extracted OK: Extract ready, Simulate blocked")
    # ---- run to goal: one console with the chain
    app.select(N9)
    app.run_to_goal()
    assert launched[-1][0] == "console" and "chain" in launched[-1][1], launched[-1]
    print("run to goal OK:", ex.chain_stages(ex.load(N9)))
    # ---- new experiment from scratch: a single Ap
    nd = L.NewExperimentDialog(app)
    nd.flow.set("Simulation only")
    fr = nd.frame
    fr.base_dir.set(base)
    fr.case.set("1DOF_150Hz")
    fr.spins.set("9000")
    fr.depths.set("8.056")
    nd._propose()
    assert nd.name.get() == "sim_1DOF150_n9000" and fr.doe_name.get() == nd.name.get(), nd.name.get()
    assert fr.preview(), fr.out.get("1.0", "end")
    shot(nd.win, "v2_new_single_ap.png")
    nd._ok()
    e = ex.load("sim_1DOF150_n9000")
    assert e.runs[0].n_cases == 1 and list(ex.stages(e)) == ["simulate", "extract"]
    assert e.runs[0].cfg["sweep"]["$Ap_start$"] == [0.008056] and "simulation" in ex.own_yaml(e.name)["runs"][0]
    print("new experiment, single Ap OK:", e.name)
    # ---- kappa x SLD limit at an n inside a pocket of the lobes -> refused with the reason (never Ap = inf)
    nd = L.NewExperimentDialog(app)
    fr = nd.frame
    fr.base_dir.set(base)
    fr.case.set("1DOF_150Hz")
    fr.doe_name.set("pocket")
    fr.unit.set("η")
    fr.depths.set("1.0")
    fr.ap_mode.set("model_at_spin")
    fr.ap_model.set("1DOF_150")
    pocket = next((n for n in range(8000, 16000, 50) if ex._sld().ap_lim("1DOF_150", n) == float("inf")), None)
    if pocket:
        fr.spins.set(str(pocket))
        assert not fr.preview() and "no finite stability limit" in fr.out.get("1.0", "end"), fr.out.get("1.0", "end")
        shot(nd.win, "v2_new_kappa_pocket.png")
        print(f"η at n = {pocket} rpm (pocket) refused OK")
    fr.spins.set("12098.28")
    nd.flow.set("Validation against a reference")
    nd.ref.set(TR)
    nd.name.set("val_from_dialog")
    fr.doe_name.set("val_from_dialog")
    fr.depths.set("0.8:1.2:0.2")
    assert fr.preview(), fr.out.get("1.0", "end")
    txt = fr.out.get("1.0", "end")
    assert "already in the reference" in txt and "time:" in txt, txt     # the training has kappa 0.8, 1.0, 1.2
    shot(nd.win, "v2_new_validation_preview.png")
    nd._ok()
    v = ex.load("val_from_dialog")
    assert v.ref.name == TR and "validate" in ex.stages(v) and v.runs[0].cfg["ap_ref"]["mode"] == "model_at_spin"
    assert len(v.runs[0].cfg["sweep"]["$Ap_start$"]) == 3
    print("new validation with η x SLD limit OK; Ap =", v.runs[0].cfg["sweep"]["$Ap_start$"])
    # ---- load values from an existing experiment (pre-fill) keeps every value
    nd = L.NewExperimentDialog(app)
    assert nd.frame.n2m.get().endswith("nessy2m/n2m.bat"), nd.frame.n2m.get()   # from base.yaml, shown in full
    nd.src.set(f"experiment: {N9}")
    assert nd.frame.n2m.get().endswith("nessy2m/n2m.bat"), nd.frame.n2m.get()   # kept when loading values
    assert nd.frame.spins.get() == "9000" and nd.frame.depths.get() == "8.056", (nd.frame.spins.get(), nd.frame.depths.get())
    shot(nd.win, "v2_new_load_values.png")
    nd.win.destroy()
    # ---- load values from an IMPORTED experiment (no config: rebuilt from its .h5) + pick the Ap on the SLD
    nd = L.NewExperimentDialog(app)
    nd.ref.set(TR)
    assert f"experiment: {TR}" in L._value_sources()
    nd.src.set(f"experiment: {TR}")
    assert nd.frame.spins.get() == "12098.28" and len(nd.frame.depths.get().split(",")) == 5, nd.frame.depths.get()
    nd.name.set("val_picked")
    nd.frame.doe_name.set("val_picked")
    nd.frame.base_dir.set(base)
    pk = L.SldPicker(app, nd.frame)
    assert len(pk.aps) == 5 and pk.lst.size() == 5          # the loaded depths are on the plot
    pk.aps.clear()
    pk.model.set("1DOF_150")
    pk.r_from.set("0.9"), pk.r_to.set("1.1"), pk.r_n.set("3"), pk.r_unit.set("η")
    pk.fill()
    assert len(pk.aps) == 3 and "η  1.000" in pk.lst.get(1), pk.lst.get(0, "end")

    # the button: n goes to the bottom of the lobe that the current n is in (not to another lobe)
    pk.n.set("13000")
    pk.draw()
    r13 = pk.min_limit_rpm()
    assert 12000 < r13 < 12200, r13                          # the lobe of 12098 rpm: its bottom at ~12091 rpm
    pk.set_min_n()
    assert pk.n.get() == f"{r13:.1f}" and abs(pk.limit() - ex._sld().ap_crit("1DOF_150")) < 1e-6
    pk.n.set("5200")                                          # another lobe: its own bottom, not the global one
    pk.draw()
    assert 5000 < pk.min_limit_rpm() < 5400, pk.min_limit_rpm()
    pk.n.set("12098.28")
    pk.draw()
    # two modes (2DOF_150_250): the scopes are the lobe of n, the whole SLD and the lowest point of each mode
    pk.model.set("2DOF_150_250")
    pk.draw()
    sc = pk.scopes()
    assert sc[0] == pk.LOBE and sc[1] == pk.ALL and len(sc) == 4, sc
    lb2, fp2 = ex._sld().lobes("2DOF_150_250")
    whole = pk.min_limit_rpm(pk.ALL)
    assert abs(whole - ex._sld().ap_crit("2DOF_150_250")) < 1e-6 or abs(
        pk._lowest_point(lb2, pk._segments(pk.ALL)) - whole) < 1e-9
    per = [pk.min_limit_rpm(s) for s in sc[2:]]
    assert len(set(round(v) for v in per)) == 2 and whole in per, (per, whole)   # the global minimum is one mode's
    pk.scope.set(sc[2])
    pk.set_min_n()
    assert pk.n.get() == f"{per[0]:.1f}", (pk.n.get(), per)
    # the intersection of the two modes: enabled for a model with 2 modes, n goes to the one nearest to the current n
    assert pk.btn_x.instate(["!disabled"]) and len(pk._intersections()) >= 2
    pk.win.update()
    right = pk.btn_x.winfo_rootx() + pk.btn_x.winfo_width()
    assert right <= pk.win.winfo_rootx() + pk.win.winfo_width(), ("the button does not fit in the picker window", right,
                                                                  pk.win.winfo_rootx() + pk.win.winfo_width())
    pk.n.set("12098.28")
    pk.set_intersection_n()
    assert pk.n.get() == "9989.1" and abs(pk.limit() - 15.557) < 0.02, (pk.n.get(), pk.limit())   # the usual crossing
    pk.n.set("3500")
    assert abs(pk.intersection_rpm()[0] - 3666.3) < 0.2          # 3666 is nearer to 3500 than 3324
    pk.set_intersection_n()
    assert pk.n.get() == "3666.3"
    pk.scope.set(sc[3])                                       # the 250 Hz mode: not a mode of the 1-mode model
    pk.model.set("1DOF_150")
    pk.draw()
    assert pk.scope.get() == pk.LOBE, pk.scope.get()          # the scope is reset when it does not exist in the model
    assert pk.btn_x.instate(["disabled"]) and pk._intersections() == []   # one mode: its lobes do not cross
    n_err = len(errors)
    pk.set_intersection_n()
    assert errors[n_err][2] == "warn" and "one mode" in errors[n_err][1], errors[n_err:]
    del errors[n_err:]
    pk.n.set("12098.28")
    pk.draw()
    lim = pk.limit()
    assert "η" in pk.ax.get_ylabel() and abs(pk.div - lim) < 1e-12      # kappa chosen -> the y axis is kappa

    class _Ev:   # a left click at kappa = 0.5 -> Ap = 0.5 x limit
        inaxes, ydata, button = pk.ax, 0.5, 1
    pk.on_click(_Ev)
    assert any(abs(a - 0.5 * lim) < 1e-3 for a in pk.aps) and len(pk.aps) == 4, pk.aps
    shot(pk.win, "v2_sld_picker_kappa.png")
    pk.r_unit.set("Ap [mm]")
    pk.draw()
    assert pk.ax.get_ylabel() == "Ap [mm]" and pk.div == 1.0
    shot(pk.win, "v2_sld_picker.png")
    pk.use()
    assert nd.frame.ap_mode.get() == "model_at_spin" and abs(float(nd.frame.depths.get().split(",")[0]) - 0.5 * lim) < 1e-3
    assert nd.frame.preview(), nd.frame.out.get("1.0", "end")
    nd._ok()
    vp = ex.load("val_picked")
    assert len(vp.runs[0].cfg["sweep"]["$Ap_start$"]) == 4 and vp.ref.name == TR
    print("load values from an imported experiment + SLD picker OK:", [round(a * 1e3, 3) for a in vp.runs[0].cfg["sweep"]["$Ap_start$"]])
    # ---- an imported folder can be opened in the planner (its simulation is rebuilt)
    p = ex.planner_config(ex.load(TR))
    assert os.path.isfile(p) and dr.load_config(p)["doe_name"] == os.path.basename(TR_DIR)
    # ---- simulation form on an experiment that reads configs/: saving writes it in full in the experiment
    sf = L.SimulationForm(app, ex.load(N9))
    sf.frame.timed.set(True)
    shot(sf.win, "v2_simulation_form.png")
    sf._ok()
    n9 = ex.load(N9)
    raw = ex.own_yaml(N9)
    assert "simulation" in raw["runs"][0] and raw["runs"][0]["simulation"]["nb_proc"] == 2, raw["runs"][0]
    assert raw["simulate"] == {"timed": True} and "--timed" in ex.stages(n9)["simulate"].cmds[0]
    print("simulation form OK: written in full, --timed on")
    # ---- settings: stages, reference, output folder
    sd = L.SettingsDialog(app, ex.load("sim_1DOF150_n9000"))
    sd.flow.set("Labelled dataset (training)")
    sd.out.set(os.path.join(tmp, "outs"))
    shot(sd.win, "v2_settings.png")
    sd._ok()
    s = ex.load("sim_1DOF150_n9000")
    assert list(ex.stages(s)) == ex.FLOWS["Labelled dataset (training)"] and s.out_dir == os.path.join(tmp, "outs")
    assert s.label["out"].startswith(os.path.join(tmp, "outs"))
    print("settings OK: stages", list(ex.stages(s)))
    # ---- copy with another DOE folder
    cd = L.CopyDialog(app, s)
    cd.name.set("sim_copy")
    cd.suffix.set("_b")
    cd._ok()
    c = ex.load("sim_copy")
    assert c.runs[0].doe_name == "sim_1DOF150_n9000_b" and "out_dir" not in ex.own_yaml("sim_copy")
    print("copy OK")
    # ---- indicator table: add from a preset, edit a row (window 6 -> new name), remove one, save
    app.select(N9, "indicators")
    f = L.IndicatorsForm(app, ex.load(N9))
    n0 = len(f.specs)
    f.preset.set("maxent_revo_dec7_1step")
    f.add()
    f.tree.selection_set("maxent_revo_dec7_1step")
    r = L.RowEditor(app, f, "maxent_revo_dec7_1step")
    r.win_n.set("6")
    r.name.set(ex.propose_variant_name(r._spec(), [n for n in f.specs if n != "maxent_revo_dec7_1step"]))
    assert r.name.get() == "maxent_revo_dec6_1step", r.name.get()
    shot(r.win, "v2_row_editor.png")
    r._ok()
    f.tree.selection_set("ssq_revo_aux4_n_aux4_dec7_1step")
    f.remove()
    shot(f.win, "v2_indicators_table.png")
    f._ok()
    specs = ex.load(N9).indicators["specs"]
    assert specs["maxent_revo_dec6_1step"]["params_physical"]["N_rev_window"] == 6 and len(specs) == n0, specs.keys()
    assert "t_theorical" not in specs["maxent_revo_dec6_1step"]["params_physical"]
    print("indicator table OK:", list(specs))
    # ---- dry-run, log viewer
    tv = L.TextView(app, "dry-run", ex.dry_run(ex.load(N9)))
    assert "case   0: Ap 8.0560 mm" in tv.text.get("1.0", "end")
    shot(tv.win, "v2_dry_run.png")
    tv.win.destroy()
    log = os.path.join(tmp, "x.log")
    with open(log, "w", encoding="utf-8") as fh:
        fh.write("$ python doe_indicators.py\n[3/34] completado: case_002 / maxent\nTraceback (most recent call last):\n"
                 "-- case_002  η 0.6  verdad: stable (amplitude)\n")
    lv = L.LogViewer(app, log, "test")
    lv.q.set("case_002")
    lv.find()
    assert lv.text.tag_ranges("hit") and lv.text.tag_ranges("bad") and lv.text.tag_ranges("run")
    shot(lv.win, "v2_log_viewer.png")
    lv.win.destroy()
    print("dry-run and log viewer OK")
    # ---- run with 'close the console' -> cmd /c
    app.close_console.set(True)
    app.select("sim_1DOF150_n9000", "simulate")
    app.run_stage()
    assert launched[-1][0] == "console" and launched[-1][2] is True
    print("run with close console OK")
    # ---- mark up to date: a stage stale only because of its configuration fingerprint
    t = ex.load(TR)
    rec = ex.read_record(t, "extract")
    ex.write_record(t, "extract", dict(rec, hash="old"))
    app.select(TR, "extract")
    assert ex.status(ex.load(TR))["extract"][0] == "stale"
    app.mark_uptodate()
    assert ex.status(ex.load(TR))["extract"][0] in ("done", "stale") and ex.read_record(t, "extract")["hash"] != "old"
    print("mark up to date OK")
    # ---- standardize an external .h5 (copy of a small fake file) and create its experiment
    ext = os.path.join(tmp, "ext_doe")
    os.makedirs(os.path.join(ext, "0", "1DOF_150Hz"))
    with h5py.File(os.path.join(ext, "old_results.h5"), "w") as h:
        for i, ap in enumerate((0.006, 0.010)):
            g = h.create_group(f"case_{i:03d}")
            g.attrs["$Ap_start$"] = ap
            g.create_dataset("Axial_disp/time", data=[0.0, 1.0])
            g.create_dataset("Axial_disp/values", data=[0.0, 1.0])
    from tkinter import filedialog
    filedialog.askopenfilename = lambda **k: os.path.join(ext, "old_results.h5")
    sd = L.StandardizeDialog(app)
    sd.sim_model.set("1DOF_150")
    sd.spin.set("12098.28")
    sd.ap_mode.set("manual")
    sd.ap_manual.set("8.60396")
    sd.name.set("ext_std")
    shot(sd.win, "v2_standardize.png")
    sd._ok()
    rep = ex.inspect_h5(os.path.join(ext, "old_results.h5"))
    assert not rep["missing"]["kappa"] and not rep["missing"]["sim_model"] and rep["values"]["sim_case"][0] == "1DOF_150Hz"
    st = ex.load("ext_std")
    assert st.data_h5.endswith("old_results.h5") and abs(ex.summary(st)["kappa"][0] - 0.006 / 0.00860396) < 1e-9
    print("standardize OK: η", ex.summary(st)["kappa"])
    n_before = len(ex.list_experiments())               # the same file again: it already has its experiment
    sd = L.StandardizeDialog(app)
    assert sd.users == ["ext_std"] and not sd.create.get()
    sd.sim_model.set("2DOF_150_250")
    sd._ok()
    assert len(ex.list_experiments()) == n_before
    assert ex.inspect_h5(os.path.join(ext, "old_results.h5"))["values"]["sim_model"][0] == "2DOF_150_250"
    assert ex.own_yaml("ext_std")["runs"][0]["model"] == "2DOF_150_250" and ex.load("ext_std").runs[0].model == "2DOF_150_250"
    sd = L.StandardizeDialog(app)                      # the values in the file are shown, nothing is re-written
    assert sd.sim_model.get() == "2DOF_150_250" and sd.sim_case.get()
    mt = os.path.getmtime(os.path.join(ext, "old_results.h5"))
    sd._ok()
    assert os.path.getmtime(os.path.join(ext, "old_results.h5")) == mt
    print("standardize of a file that has its experiment: only attributes, no new experiment OK")
    check_ramps()
    check_noise()
    check_run_only()
    # ---- compare: two validations with fabricated metrics
    for n, ba in ((VA, 0.9), ("val_from_dialog", 0.7)):
        out = os.path.join(tmp, f"{n}_val.h5")
        d = ex.own_yaml(n)
        d["validate"] = {"channel": "Axial_disp", "out": out.replace("\\", "/")}
        ex.yaml_save(d, ex.exp_path(n))
        with h5py.File(out, "w") as h:
            h.create_group("metrics/maxent_revo_dec7_1step").attrs.update(
                balanced_accuracy=ba, MCC=ba - 0.1, **({"ramp_n": 2, "ramp_detection_rate": 0.5, "n_gray": 2,
                                                         "n_gray_alarm": 1, "gray_alarm_rate": 0.5,
                                                         "gray_as_stable_TNR": 0.6, "gray_as_unstable_TPR": 0.8,
                                                         "balanced_accuracy_lo": 0.8, "median_t_ratio": 0.5}
                                                        if n == VA else {}))
    ex.reload()
    app.refresh(True)
    assert set(app.cmp_cb_a["values"]) >= {VA, "val_from_dialog"}, app.cmp_cb_a["values"]
    app.cmp_a.set(VA)
    app.cmp_b.set("val_from_dialog")
    app.compare()
    rows = [app.cmp_tree.item(i, "values") for i in app.cmp_tree.get_children()]
    assert rows[0][1] == "0.900" and rows[0][2] == "0.700", rows
    j = 1 + 2 * ex.METRIC_COLUMNS.index("ramp_detection_rate")
    assert rows[0][j] == "0.500" and rows[0][j + 1] == "", rows          # the ramp metrics: A has them, B not
    for k, v in (("balanced_accuracy_lo", "0.800"), ("median_t_ratio", "0.500")):   # intervals / ratio (new files) vs old files
        j = 1 + 2 * ex.METRIC_COLUMNS.index(k)
        assert rows[0][j] == v and rows[0][j + 1] == "", (k, rows[0][j], rows[0][j + 1])
    for k, v in (("n_gray", "2"), ("gray_alarm_rate", "0.500"), ("gray_as_stable_TNR", "0.600"),
                 ("gray_as_unstable_TPR", "0.800")):                     # gray cases: A (new files) has them, B not
        j = 1 + 2 * ex.METRIC_COLUMNS.index(k)
        assert rows[0][j] == v and rows[0][j + 1] == "", (k, rows[0][j], rows[0][j + 1])
    card = [t for t, _ in ex.stage_summary(ex.load(VA), "validate")]
    assert any("gray cases (not scored): 2, 1 alarm" in t and "TNR 0.60" in t and "TPR 0.80" in t for t in card), card
    assert not any("gray cases (" in t for t, _ in ex.stage_summary(ex.load("val_from_dialog"), "validate"))   # no n_gray
    assert ex.stage_summary(ex.load("val_from_dialog"), "validate")[0][0] == "gray cases: ignore (not scored)"
    print("compare OK (with the ramp columns)")
    # Compare > Plot: a side without validation results is a warning, not a crash (the figure itself is checked with a
    # real validation file by validation_figures.py --selftest)
    app.cmp_b.set(N9)
    app.plot_compare()
    assert errors[-1][0] == "Compare" and errors[-1][2] == "warn", errors[-1]
    errors.pop()
    app.cmp_b.set("val_from_dialog")
    # with both results: the A/B figure goes to the export window (nothing is saved until Save); these fabricated
    # files have no /ranking, so the figure itself fails and the window says so instead of crashing
    app.plot_compare()
    cw = app._cmp_win
    cw.draw()
    assert cw.labels == [f"compare_{VA}_vs_val_from_dialog"] and cw.fig is None and "Error" in cw.status.get(), cw.status.get()
    assert not os.path.isdir(cw.folder.get()) or not os.listdir(cw.folder.get())
    cw.win.destroy()
    print("compare plot window OK")
    # validate form: only the channel (the rule has no tolerance any more); an old early_tol_s in the YAML is
    # ignored by the stage command and dropped when the form is saved
    ex.save_section(VA, "validate", {"channel": "Axial_disp", "early_tol_s": 0.3})
    assert "--early-tol" not in ex.stages(ex.load(VA))["validate"].cmds[0]
    vf = L.ValidateForm(app, ex.load(VA))
    assert not hasattr(vf, "tol")
    shot(vf.win, "ramps_validate_form.png")
    vf._ok()
    va = ex.load(VA)
    assert "early_tol_s" not in va.section("validate") and "--early-tol" not in ex.stages(va)["validate"].cmds[0]
    # gray cases: three modes, each with its own results file; absent = ignore keeps the old fingerprint
    h0, c0 = ex.stages(va)["validate"].hash, ex.stages(va)["validate"].cmds[0]
    assert c0[-2:] == ["--gray", "ignore"] and ex.gray_mode(va) == "ignore"
    assert os.path.basename(ex.validation_path(va)) == "doe_validation_results.h5"
    vf = L.ValidateForm(app, va)
    assert vf.gray.get() == ex.GRAY_LABELS["ignore"]
    vf.gray.set(ex.GRAY_LABELS["stable"])
    vf._ok()
    va = ex.load(VA)
    sv = ex.stages(va)["validate"]
    assert va.section("validate")["gray"] == "stable" and ex.gray_mode(va) == "stable"
    assert os.path.basename(sv.outputs[0]) == "doe_validation_results_gray-stable.h5" and sv.cmds[0][-2:] == ["--gray", "stable"]
    assert sv.cmds[0][sv.cmds[0].index("--out") + 1] == sv.outputs[0] and sv.hash != h0      # changing the mode: Validate stale
    ex.save_section(VA, "validate", {"channel": "Axial_disp", "gray": "unstable"})
    assert os.path.basename(ex.stages(ex.load(VA))["validate"].outputs[0]) == "doe_validation_results_gray-unstable.h5"
    ex.save_section(VA, "validate", {"channel": "Axial_disp", "gray": "stable", "out": os.path.join(tmp, "mine.h5")})
    assert ex.validation_path(ex.load(VA)) == os.path.normpath(os.path.join(tmp, "mine.h5"))   # an explicit out is respected
    ex.save_section(VA, "validate", {"channel": "Axial_disp", "gray": "sometimes"})
    assert any("validate.gray" in x for x in ex.check(ex.load(VA))[0]) and ex.gray_mode(ex.load(VA)) == "ignore"
    vf = L.ValidateForm(app, ex.load(N9))                         # back to ignore: the key leaves the YAML
    ex.save_section(N9, "validate", {"channel": "Axial_disp", "gray": "stable"})
    vf = L.ValidateForm(app, ex.load(N9))
    assert vf.gray.get() == ex.GRAY_LABELS["stable"]
    vf.gray.set(ex.GRAY_LABELS["ignore"])
    vf._ok()
    assert "gray" not in ex.load(N9).section("validate")
    ex.save_section(N9, "validate", {"channel": "Axial_disp", "gray": "stable"})   # metrics of the mode of the experiment
    n9 = ex.load(N9)
    os.makedirs(os.path.dirname(ex.validation_path(n9)), exist_ok=True)
    with h5py.File(ex.validation_path(n9), "w") as h:
        h.create_group("metrics/run_x").attrs.update(balanced_accuracy=0.8, n_gray=3, n_gray_alarm=1)
    assert ex.validation_metrics(n9)["run_x"]["n_gray"] == 3 and not ex.validation_metrics(ex.load(VA))
    mine = os.path.join(tmp, "mine_gray.h5")                       # the card of an experiment with the Validate stage
    ex.save_section(VA, "validate", {"channel": "Axial_disp", "gray": "stable", "out": mine})
    with h5py.File(mine, "w") as h:
        h.create_group("metrics/run_x").attrs.update(balanced_accuracy=0.8, TP=1, FN=0, TN=1, FP=0, n_gray=3, n_gray_alarm=1,
                                                      balanced_accuracy_lo=0.6, balanced_accuracy_hi=0.9, MCC=0.5, MCC_lo=0.2,
                                                      MCC_hi=0.8, median_t_ratio=0.45)
    card = [t for t, _ in ex.stage_summary(ex.load(VA), "validate")]
    assert card[0].startswith("gray cases: stable (pessimistic)") and any("scored as stable): 3, 1 alarm" in t for t in card), card
    assert any("bal.acc 0.80 [0.60-0.90]" in t and "MCC 0.50 [0.20-0.80]" in t and "t_det/t_onset 0.45" in t for t in card), card
    ex.save_section(VA, "validate", {"channel": "Axial_disp"})
    print("gray modes OK")
    app.notebook.select(app.tab_exp)
    app.select(VA, "validate")
    shot(root, "v2_main_validation.png")
    app.select(TR, "label_template")
    shot(root, "v2_main_training.png")
    # Label grid button: only on Label build with its dataset present; it launches label_grid.py on that .h5
    assert app.stage_btns["grid"].instate(["disabled"])
    app.select(TR, "label_build")
    assert app.stage_btns["grid"].instate(["!disabled"])
    app.open_grid()
    assert launched[-1][0] == "label_grid.py" and launched[-1][1][1] == ex.load(TR).label["out"], launched[-1]
    print("label grid button OK")
    assert not [x for x in errors if x[2] == "error"], errors
finally:
    root.destroy()
    ex.CONFIGS_DIR = dr.CONFIGS_DIR = real_cfg
    shutil.rmtree(tmp, ignore_errors=True)
print("dialogs OK; shots in", SHOTS)
