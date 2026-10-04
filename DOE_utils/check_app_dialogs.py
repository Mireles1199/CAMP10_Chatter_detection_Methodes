"""Check of the app forms (v2) on a temp copy of the experiments: new experiment from scratch (a single Ap, kappa
from a model, an Ap = inf refused), settings, copy, simulation form (a config run written in full), indicator
table, import, standardize, dry-run, log viewer, mark up to date, compare. Real data are only read; captures in
%TEMP%/app_dialog_shots for a visual check."""
import os
import shutil
import sys
import tempfile

import h5py

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


root = tk.Tk()
root.geometry("1300x860+10+10")
app = L.App(root, auto_refresh=False)
try:
    # ---- viewer / planner get their arguments as a list: a path with spaces arrives without quotes
    assert L.split_args('--h5 "D:/a b/c.h5"') == ["--h5", "D:/a b/c.h5"]
    app.view_h5("D:/x y/doe_results.h5")
    assert launched[-1] == ("DOE_plots/doe_unified_selector.py", ["--h5", "D:/x y/doe_results.h5"]), launched[-1]
    print("viewer arguments OK")
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
    assert "not extracted yet: 2 cases" in txt and "kappa 0.5-1.5" in txt, txt
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
    fr.unit.set("kappa")
    fr.depths.set("1.0")
    fr.ap_mode.set("model_at_spin")
    fr.ap_model.set("1DOF_150")
    pocket = next((n for n in range(8000, 16000, 50) if ex._sld().ap_lim("1DOF_150", n) == float("inf")), None)
    if pocket:
        fr.spins.set(str(pocket))
        assert not fr.preview() and "no finite stability limit" in fr.out.get("1.0", "end"), fr.out.get("1.0", "end")
        shot(nd.win, "v2_new_kappa_pocket.png")
        print(f"kappa at n = {pocket} rpm (pocket) refused OK")
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
    print("new validation with kappa x SLD limit OK; Ap =", v.runs[0].cfg["sweep"]["$Ap_start$"])
    # ---- load values from an existing experiment (pre-fill) keeps every value
    nd = L.NewExperimentDialog(app)
    nd.src.set(f"experiment: {N9}")
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
    pk.r_from.set("0.9"), pk.r_to.set("1.1"), pk.r_n.set("3"), pk.r_unit.set("kappa")
    pk.fill()
    assert len(pk.aps) == 3 and "kappa  1.000" in pk.lst.get(1), pk.lst.get(0, "end")

    class _Ev:   # a left click at Ap = 5 mm
        inaxes, ydata, button = pk.ax, 5.0, 1
    pk.on_click(_Ev)
    assert 5.0 in pk.aps and len(pk.aps) == 4
    shot(pk.win, "v2_sld_picker.png")
    pk.use()
    assert nd.frame.ap_mode.get() == "model_at_spin" and nd.frame.depths.get().startswith("5")
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
                 "-- case_002  kappa 0.6  verdad: stable (amplitude)\n")
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
    print("standardize OK: kappa", ex.summary(st)["kappa"])
    n_before = len(ex.list_experiments())               # the same file again: it already has its experiment
    sd = L.StandardizeDialog(app)
    assert sd.users == ["ext_std"] and not sd.create.get()
    sd.sim_model.set("2DOF_150_250")
    sd._ok()
    assert len(ex.list_experiments()) == n_before
    assert ex.inspect_h5(os.path.join(ext, "old_results.h5"))["values"]["sim_model"][0] == "2DOF_150_250"
    print("standardize of a file that has its experiment: only attributes, no new experiment OK")
    # ---- compare: two validations with fabricated metrics
    for n, ba in ((VA, 0.9), ("val_from_dialog", 0.7)):
        out = os.path.join(tmp, f"{n}_val.h5")
        d = ex.own_yaml(n)
        d["validate"] = {"channel": "Axial_disp", "out": out.replace("\\", "/")}
        ex.yaml_save(d, ex.exp_path(n))
        with h5py.File(out, "w") as h:
            h.create_group("metrics/maxent_revo_dec7_1step").attrs.update(balanced_accuracy=ba, MCC=ba - 0.1)
    ex.reload()
    app.refresh(True)
    assert set(app.cmp_cb_a["values"]) >= {VA, "val_from_dialog"}, app.cmp_cb_a["values"]
    app.cmp_a.set(VA)
    app.cmp_b.set("val_from_dialog")
    app.compare()
    rows = [app.cmp_tree.item(i, "values") for i in app.cmp_tree.get_children()]
    assert rows[0][1] == "0.900" and rows[0][2] == "0.700", rows
    print("compare OK")
    app.notebook.select(app.tab_exp)
    app.select(VA, "validate")
    shot(root, "v2_main_validation.png")
    app.select(TR, "label_template")
    shot(root, "v2_main_training.png")
    assert not [x for x in errors if x[2] == "error"], errors
finally:
    root.destroy()
    ex.CONFIGS_DIR = dr.CONFIGS_DIR = real_cfg
    shutil.rmtree(tmp, ignore_errors=True)
print("dialogs OK; shots in", SHOTS)
