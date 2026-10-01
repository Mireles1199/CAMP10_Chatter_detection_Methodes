"""Check of the app dialogs (forms, variants, new DOE, derive, new validation, compare) of the app on a temp copy of the experiments (real data only read)."""
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

TR, VA = "train_1DOF150_n12098_k0.5-2.0", "val_1DOF150_n12098_k0.53-1.91"
tmp = tempfile.mkdtemp(prefix="app_dialogs_")
shutil.copytree(ex.EXP_DIR, os.path.join(tmp, "experiments"))
dr = ex._doe_runner()
cfg_tmp = os.path.join(tmp, "configs")
os.makedirs(cfg_tmp)
for f in ("base.yaml", "test_validaicon.yaml", "tube_ap_sweep.yaml"):
    shutil.copy(os.path.join(ex.CONFIGS_DIR, f), cfg_tmp)
real_cfg = ex.CONFIGS_DIR
ex.set_root(os.path.join(tmp, "experiments"))
ex.CONFIGS_DIR = dr.CONFIGS_DIR = cfg_tmp
launched = []
L.launch = lambda kind, script, args: launched.append((script, args)) or (None, None)   # no real tools

import tkinter as tk  # noqa: E402
from PIL import ImageGrab  # noqa: E402
import ctypes  # noqa: E402
F = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100.0


def shot(win, name):
    win.attributes("-topmost", True)
    win.lift()
    win.update()
    win.after(400)
    win.update()
    x, y, w, h = win.winfo_rootx(), win.winfo_rooty(), win.winfo_width(), win.winfo_height()
    ImageGrab.grab(bbox=tuple(int(v * F) for v in (x, y, x + w, y + h))).save(os.path.join(SHOTS, name))


root = tk.Tk()
root.geometry("1300x800+10+10")
app = L.App(root, auto_refresh=False)
try:
    # ---- F6 indicators form on the validation: inherit -> custom with 2 variants
    app.select(VA, "indicators")
    f = L.IndicatorsForm(app, ex.load(VA))
    shot(f.win, "f6_indicators_val.png")
    f.inherit.set(False)
    for n, (v, _) in f.checks.items():
        v.set(n in ("maxent_revo_dec4_1step", "ssq_revo_aux4_n_aux4_dec7_1step"))
    f._ok()
    assert ex.own_yaml(VA)["indicators"]["variants"] == ["maxent_revo_dec4_1step", "ssq_revo_aux4_n_aux4_dec7_1step"]
    print("F6 indicators form OK (custom variants saved)")
    # ---- variant editor: duplicate maxent with a 6-revolution window
    f = L.IndicatorsForm(app, ex.load(TR))
    f.focus.set("maxent_revo_dec4_1step")
    v = L.VariantEditor(app, f, "maxent_revo_dec4_1step", True)
    assert v.name.get() == "maxent_revo_dec4_1step_v2", v.name.get()
    v.name.set("maxent_revo_dec6_1step")
    v.text.delete("1.0", "end")
    import yaml
    spec = dict(ex.variants_library()["variants"]["maxent_revo_dec4_1step"])
    spec["params_physical"] = dict(spec["params_physical"], N_rev_window=6)
    v.text.insert("1.0", yaml.safe_dump(spec, sort_keys=False))
    shot(v.win, "f6_variant_editor.png")
    v._ok()
    assert ex.variants_library()["variants"]["maxent_revo_dec6_1step"]["params_physical"]["N_rev_window"] == 6
    print("F6 variant duplicated OK")
    root.update()
    for w in root.winfo_children():
        if isinstance(w, tk.Toplevel):
            w.destroy()
    # ---- resolved config
    r = L.ResolvedView(app, ex.load(TR), "maxent_revo_dec4_1step", ["case_000", "case_009"])
    r.case.set("case_009")
    txt = r.text.get("1.0", "end")
    assert '"T_rev": 0.0049593' in txt and "reference_dataset_amp.h5" in txt, txt[:400]
    shot(r.win, "f6_resolved.png")
    r.win.destroy()
    print("F6 resolved config OK")
    # ---- label form on the training: change lim_sup_pct -> label stages become stale
    lf = L.LabelForm(app, ex.load(TR))
    shot(lf.win, "f6_label_train.png")
    lf.vars["lim_sup_pct"].set("45")
    lf._ok()
    lab = ex.own_yaml(TR)["label"]
    assert lab["lim_sup_pct"] == 45.0 and lab["out"].endswith("reference_dataset_amp.h5")
    st = ex.status(ex.load(TR))
    assert st["label_template"][0] == "stale" and st["label_build"][0] == "stale", st
    print("F6 label form OK -> label stages stale:", st["label_build"][1])
    lv = L.LabelForm(app, ex.load(VA))
    assert str(lv.vars["strategy"]._name) and lv.body.winfo_children()
    shot(lv.win, "f6_label_val_locked.png")
    lv.win.destroy()
    # ---- F7 new DOE from a template (configs redirected to the temp copy)
    nd = L.NewDoeDialog(app)
    nd.template.set("test_validaicon")
    nd.spin.set("10000")
    nd.kappa.set("0.5:1.0:0.25")
    nd._propose()
    shot(nd.win, "f7_new_doe.png")
    name, doe = nd.name.get(), nd.doe.get()
    nd._ok()
    e = ex.load(name)
    assert e.runs[0].doe_name == doe and e.runs[0].n_cases == 3 and e.kind == "training"
    assert e.runs[0].cfg["sweep"]["$spin_rate$"][0] == 10000.0
    print("F7 new DOE OK:", name, "->", doe, "Ap", e.runs[0].cfg["sweep"]["$Ap_start$"])
    assert not os.path.exists(os.path.join(real_cfg, doe + ".yaml"))
    # ---- derive at another n
    dd = L.DeriveDialog(app, e)
    dd.spin.set("11000")
    shot(dd.win, "f7_derive.png")
    dname = dd.name.get()
    dd._ok()
    de = ex.load(dname)
    assert de.runs[0].cfg["sweep"]["$spin_rate$"][0] == 11000.0 and de.cfg.get("description")
    print("F7 derive OK:", dname)
    # ---- new validation (planner launch is recorded, not opened)
    nv = L.NewValidationDialog(app)
    nv.training.set(TR)
    shot(nv.win, "f7_new_validation.png")
    vname = nv.name.get()
    nv._ok()
    assert ex.load(vname).training.name == TR and launched and "doe_val_planner" in launched[-1][0]
    st = ex.status(ex.load(vname))
    print("F7 new validation OK:", vname, "| before the planner saves:", ex.check(ex.load(vname))[0][:1])
    # ---- F8 compare: two validations with fabricated metrics in temp files
    for n, ba in ((VA, 0.9), (vname, 0.7)):
        out = os.path.join(tmp, f"{n}_val.h5")
        d = ex.own_yaml(n)
        d["validate"] = {"channel": "Axial_disp", "out": out.replace("\\", "/")}
        ex.yaml_save(d, ex.exp_path(n))
        with h5py.File(out, "w") as h:
            for run in ("maxent_revo_dec4_1step", "ssq_revo_aux4_n_aux4_dec7_1step"):
                h.create_group(f"metrics/{run}").attrs.update(balanced_accuracy=ba, MCC=ba - 0.1, AUC=ba + 0.05,
                                                              TPR=1.0, TNR=ba)
    ex.reload()
    app.refresh(True)
    app.cmp_a.set(VA)
    app.cmp_b.set(vname)
    app.compare()
    rows = [app.cmp_tree.item(i, "values") for i in app.cmp_tree.get_children()]
    assert len(rows) == 2 and rows[0][1] == "0.900" and rows[0][2] == "0.700", rows
    app.root.nametowidget(app.tab_cmp.winfo_parent()).select(app.tab_cmp)
    shot(root, "f8_compare.png")
    print("F8 compare OK:", rows[0][:5])
    # ---- the main window with the stale training
    app.root.nametowidget(app.tab_exp.winfo_parent()).select(app.tab_exp)
    app.select(TR, "label_build")
    shot(root, "app_stale_training.png")
finally:
    root.destroy()
    ex.CONFIGS_DIR = dr.CONFIGS_DIR = real_cfg
    shutil.rmtree(tmp, ignore_errors=True)
print("dialogs OK; shots in", SHOTS)
