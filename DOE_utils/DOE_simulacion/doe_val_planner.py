#!/usr/bin/env python
# coding: utf-8
"""doe_val_planner.py — Build the YAML of a VALIDATION DOE from the training dataset.

Open a labelled reference_dataset*.h5; it provides what must stay IDENTICAL (spin_rate, dxl_size, f_tooth,
nb_dt_rev, ap_ref = Ap/kappa) and the kappa values already used for training. Define zones of kappa and how many
new cases each one gets. Each zone is split into N equal strata with one case per stratum, placed with a jitter
and kept away from the training kappa values. Nothing is simulated: the tool writes a YAML (mode: sweep, only Ap
changes) to open with doe_planner.py and launch with doe_runner.py.

The ground-truth label of the new cases does NOT come from kappa: it is obtained afterwards with
reference_dataset.py using the same labeling_* settings as the training dataset (the kappa zones only decide
WHAT to simulate).

Usage (with the entorno_CAMP10 Python):
    python doe_val_planner.py [reference_dataset_amp.h5]   # no argument: asks which file to open
    python doe_val_planner.py --selftest
"""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # DOE_utils/
import eta_compat  # noqa: E402  (kappa -> eta: reads both names)
CONFIGS = os.path.join(HERE, "configs")
# (name, kappa_min, kappa_max, n_cases). Where the real transition is shows in the colours of the figure.
ZONES = [("far stable", 0.50, 0.90, 4), ("transition", 0.90, 1.10, 6),
         ("near unstable", 1.10, 1.30, 3), ("far unstable", 1.30, 2.00, 4)]
# Note: training may already cover the transition densely (e.g. every 0.01); there a new case is almost an
# interpolation (it falls between two training kappa, at >= tol from both).
SCALARS = ("$spin_rate$", "$f_tooth$", "$dxl_size$", "$nb_dt_rev$")   # must match the training dataset
JITTER_HELP = ("Jitter (0-1): how far a case may move from the centre of its stratum.\n"
               "0 = exactly at the centre (regular spacing); 1 = anywhere inside the stratum.\n"
               "Same seed + same inputs = same cases.")


# ============================================================================== logic (no GUI)
def read_training(path: str) -> dict:
    """{'cases': {group: {kappa, ap, labels:set}}, 'scalars': {...}, 'ap_ref': m, 'base_dir': folder}."""
    import h5py
    cases, scal = {}, {}
    with h5py.File(path, "r") as f:
        for lab in f:
            for case in f[lab]:
                pieces = list(f[lab][case].values())
                if not pieces:
                    continue
                a = pieces[0].attrs
                if abs(float(a.get("$Ap_end$", a["$Ap_start$"])) - float(a["$Ap_start$"])) > 1e-9:
                    continue   # a ramp has no single kappa (training is made of constant cases)
                c = cases.setdefault(case, dict(kappa=float(eta_compat.col(a, "eta")), ap=float(a["$Ap_start$"]), labels=set()))
                c["labels"].add(lab)
                for k in SCALARS:
                    v = float(a[k])
                    if scal.setdefault(k, v) != v:
                        raise ValueError(f"{k} is not the same in every case of the dataset ({scal[k]} vs {v})")
    if not cases:
        raise ValueError("the .h5 has no constant-Ap cases")
    refs = sorted(c["ap"] / c["kappa"] for c in cases.values() if c["kappa"] > 0)
    return dict(cases=cases, scalars=scal, ap_ref=refs[len(refs) // 2],
                base_dir=os.path.dirname(os.path.dirname(os.path.abspath(path))).replace("\\", "/"))


def sample_zones(zones, used, tol, jitter, seed) -> list:
    """[(zone, kappa)]. One case per stratum (centre + jitter*half-width), at >= tol from `used` and from the
    ones already chosen. If a stratum has no room, raises ValueError."""
    rng, taken, out = random.Random(seed), list(used), []
    for name, k0, k1, n in zones:
        w = (k1 - k0) / n
        for i in range(n):
            c = k0 + (i + 0.5) * w
            for _ in range(200):
                k = c + rng.uniform(-1, 1) * jitter * w / 2
                if all(abs(k - t) >= tol for t in taken):
                    break
            else:
                raise ValueError(f"zone '{name}': stratum {i + 1}/{n} has no room at {tol} from other kappa "
                                 f"(lower the tolerance or N, or widen the zone)")
            taken.append(k)
            out.append((name, k))
    return out


def extends_ref(dest_dir: str) -> str:
    """Value of `extends:` for a YAML saved in dest_dir (resolved relative to the declaring file)."""
    if os.path.normcase(os.path.abspath(dest_dir)) == os.path.normcase(CONFIGS):
        return "base"
    base = os.path.join(CONFIGS, "base.yaml")
    try:
        return os.path.relpath(base, dest_dir).replace("\\", "/")
    except ValueError:   # other drive: no relative path exists
        return base.replace("\\", "/")


def yaml_text(name: str, tr: dict, kappas: list, base_dir: str, case: str, src: str, extends: str = "base") -> str:
    ap_ref, s = tr["ap_ref"], tr["scalars"]
    aps = [round(k * ap_ref, 7) for k in kappas]
    lst = "\n".join(f"    - {a:.7g}".replace("e-0", "e-") + f"     # kappa {k:.3f}" for a, k in zip(aps, kappas))
    return f"""# VALIDATION DOE generated by doe_val_planner.py from:
#   {src}
# Only Ap changes (fixed depth per case); spin and discretisation = training dataset.
# Ground-truth labels are obtained afterwards with reference_dataset.py (same labeling_* as training).
extends: {extends}

base_dir: {base_dir}
case: {case}
doe_name: {name}
mode: sweep
sweep:
  $Ap_start$: &ap_val          # [m]
{lst}
  $Ap_end$: *ap_val
  $spin_rate$: {s['$spin_rate$']!r}
  $f_tooth$: {s['$f_tooth$']!r}
  $dxl_size$: {s['$dxl_size$']!r}
  $nb_dt_rev$: {int(s['$nb_dt_rev$'])}

ap_ref:                        # same as training: kappa = Ap / ap_ref
  mode: manual
  manual: {ap_ref:.6g}
"""


# ============================================================================== GUI
def run_gui(h5_path: str, doe_name: str = ""):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    root = tk.Tk()
    root.title("DOE val planner")
    root.minsize(1000, 600)
    st = dict(tr=None, path=None, pick=[])
    COL = {"stable": "#1976d2", "unstable": "#e65100", "gray": "#9e9e9e"}

    left, right = ttk.Frame(root, padding=6), ttk.Frame(root, padding=6)
    left.pack(side=tk.LEFT, fill=tk.Y)
    right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    info = tk.StringVar()
    ttk.Label(left, textvariable=info, justify="left", font=("Consolas", 9)).pack(anchor="w")

    def ask():
        load(filedialog.askopenfilename(title="Training dataset (.h5)", filetypes=[("HDF5", "*.h5")]))
    ttk.Button(left, text="Open dataset .h5…", command=ask).pack(anchor="w", pady=4)

    zf = ttk.LabelFrame(left, text="Zones  (name · kappa min · kappa max · N)", padding=4)
    zf.pack(fill=tk.X, pady=4)
    rows = []

    def add_zone(z=("zone", 1.0, 1.1, 2)):
        r = [tk.StringVar(value=str(v)) for v in z]
        fr = ttk.Frame(zf)
        fr.pack(fill=tk.X)
        for v, w in zip(r, (14, 6, 6, 4)):
            ttk.Entry(fr, textvariable=v, width=w).pack(side=tk.LEFT, padx=1)

        def rm():
            fr.destroy()
            rows.remove(r)
        ttk.Button(fr, text="✕", width=2, command=rm).pack(side=tk.LEFT)
        rows.append(r)
    for z in ZONES:
        add_zone(z)
    ttk.Button(zf, text="+ zone", command=add_zone).pack(anchor="w", pady=2)

    pf = ttk.Frame(left)
    pf.pack(fill=tk.X, pady=4)
    jit, tol, seed = tk.StringVar(value="0.6"), tk.StringVar(value="0.004"), tk.StringVar(value="1")
    name, case = tk.StringVar(value=doe_name), tk.StringVar(value="1DOF_150Hz")
    for i, (lbl, v) in enumerate((("jitter (0-1)", jit), ("min kappa gap to training", tol), ("seed", seed),
                                  ("doe_name", name), ("case", case))):
        ttk.Label(pf, text=lbl).grid(row=i, column=0, sticky="w")
        ttk.Entry(pf, textvariable=v, width=30 if lbl in ("doe_name", "case") else 8).grid(row=i, column=1, sticky="w")
    ttk.Label(left, text=JITTER_HELP, justify="left", foreground="#555", font=("Segoe UI", 8)).pack(anchor="w")

    out = tk.Text(left, height=14, width=44, font=("Consolas", 9), state="disabled")
    fig = Figure(figsize=(7, 3), dpi=100)
    ax = fig.add_subplot(111)

    def export():   # Export… (DOE_plots/figures_window.py): the figure on screen, as a copy
        sys.path.insert(0, os.path.join(HERE, "..", "DOE_plots"))
        from figures_window import FiguresWindow, Item
        FiguresWindow(root, "Export — doe_val_planner", [Item("val_planner_kappa", lambda: fig, live=True)],
                      out_dir=os.path.dirname(st["path"]) if st["path"] else "")
    ttk.Button(right, text="💾  Export…", command=export).pack(side=tk.TOP, anchor="e")
    cv = FigureCanvasTkAgg(fig, master=right)
    tb = NavigationToolbar2Tk(cv, right)   # zoom / pan / home / save PNG
    tb.pack(side=tk.BOTTOM, fill=tk.X)
    cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def zones():
        return [(r[0].get(), float(r[1].get()), float(r[2].get()), int(r[3].get())) for r in rows]

    def draw():
        ax.clear()
        tr = st["tr"]
        if tr:
            for lab in COL:
                ks = [c["kappa"] for c in tr["cases"].values() if lab in c["labels"]]
                ax.plot(ks, [1] * len(ks), "o", color=COL[lab], label=f"training: {lab}")
        try:
            for i, (nm, k0, k1, _) in enumerate(zones()):
                ax.axvspan(k0, k1, color="#000", alpha=0.04 if i % 2 else 0.09)
        except ValueError:   # a zone cell is half-typed
            pass
        if st["pick"]:
            ax.plot([k for _, k in st["pick"]], [0] * len(st["pick"]), "s", color="#2e7d32", label="validation (new)")
        ax.set_yticks([])
        ax.set_ylim(-1, 2)
        ax.set_xlabel("kappa = Ap / ap_ref")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=8)
        cv.draw_idle()
        tb.update()   # after a redraw, "home" becomes the new view

    def say(txt):
        out.configure(state="normal")
        out.delete("1.0", tk.END)
        out.insert("1.0", txt)
        out.configure(state="disabled")

    def load(path):
        if not path:
            return
        try:
            st["tr"], st["path"] = read_training(path), path
        except Exception as exc:
            messagebox.showerror("dataset", str(exc))
            return
        tr = st["tr"]
        info.set(f"{os.path.basename(path)}\n{len(tr['cases'])} cases · spin {tr['scalars']['$spin_rate$']:.2f} rpm\n"
                 f"ap_ref {tr['ap_ref'] * 1e3:.4f} mm")
        st["pick"] = []
        say("")
        draw()

    def generate():
        if not st["tr"]:
            return
        try:
            st["pick"] = sample_zones(zones(), [c["kappa"] for c in st["tr"]["cases"].values()],
                                      float(tol.get()), float(jit.get()), int(seed.get()))
        except Exception as exc:
            messagebox.showerror("generate", str(exc))
            return
        ref = st["tr"]["ap_ref"]
        say("\n".join(f"{n[:16]:16s} kappa {k:.3f}  Ap {k * ref * 1e3:7.3f} mm" for n, k in st["pick"]) +
            f"\n\nTotal: {len(st['pick'])} cases")
        draw()

    def save():
        if not st["pick"] or not name.get().strip():
            messagebox.showinfo("save", "Generate the cases and type a doe_name first.")
            return
        dest = filedialog.asksaveasfilename(title="Save validation YAML", initialdir=CONFIGS, defaultextension=".yaml",
                                            initialfile=name.get().strip() + ".yaml", filetypes=[("YAML", "*.yaml")])
        if not dest:
            return
        ks = sorted(k for _, k in st["pick"])
        with open(dest, "w", encoding="utf-8") as f:
            f.write(yaml_text(name.get().strip(), st["tr"], ks, st["tr"]["base_dir"], case.get(), st["path"],
                              extends_ref(os.path.dirname(os.path.abspath(dest)))))
        say(f"Saved:\n{dest}\n\nOpen it in doe_planner.py (\"Other YAML…\" if it is not in configs/).")

    bf = ttk.Frame(left)
    bf.pack(fill=tk.X)
    ttk.Button(bf, text="Generate cases", command=generate).pack(side=tk.LEFT)
    ttk.Button(bf, text="Save YAML…", command=save).pack(side=tk.LEFT, padx=4)
    out.pack(fill=tk.BOTH, expand=True, pady=4)
    draw()
    if h5_path:
        load(h5_path)
    else:
        root.after(200, ask)
    root.mainloop()


# ============================================================================== self-check
def _selftest():
    used = [0.55, 1.5, 1.9]
    a = sample_zones(ZONES, used, 0.02, 0.6, 1)
    assert a == sample_zones(ZONES, used, 0.02, 0.6, 1), "same seed = same result"
    assert len(a) == sum(z[3] for z in ZONES)
    ks = [k for _, k in a]
    assert all(abs(k - u) >= 0.02 for k in ks for u in used) and len({round(k, 6) for k in ks}) == len(ks)
    for nm, k0, k1, n in ZONES:   # every case falls inside its zone
        assert all(k0 <= k <= k1 for z, k in a if z == nm)
    assert [k for _, k in sample_zones([("z", 1, 2, 4)], [], 0.0, 0.0, 0)] == [1.125, 1.375, 1.625, 1.875]
    try:
        sample_zones([("z", 1.0, 1.01, 5)], [], 0.05, 0.5, 0)
        raise AssertionError("should have failed: zone without room")
    except ValueError:
        pass
    assert extends_ref(CONFIGS) == "base"
    assert extends_ref(os.path.join(HERE, "other")) == "../configs/base.yaml"
    import h5py
    import tempfile
    d = tempfile.mkdtemp()
    trained = {}
    for key in ("kappa", "eta"):   # kappa -> eta: a training dataset of either age gives the same cases
        p = os.path.join(d, key, "ds", "reference_dataset.h5")
        os.makedirs(os.path.dirname(p))
        with h5py.File(p, "w") as f:
            for lab, case, e in (("stable", "case_000", 0.5), ("unstable", "case_001", 1.5)):
                pc = f.create_group(f"{lab}/{case}").create_dataset("Axial_disp__000", data=[0.0])
                pc.attrs.update({key: e, "$Ap_start$": 0.005 * e, "$spin_rate$": 1.0, "$f_tooth$": 0.05, "$dxl_size$": 1e-4, "$nb_dt_rev$": 200.0})
        trained[key] = read_training(p)
    assert {c: v["kappa"] for c, v in trained["kappa"]["cases"].items()} == {c: v["kappa"] for c, v in trained["eta"]["cases"].items()} ==         {"case_000": 0.5, "case_001": 1.5} and trained["kappa"]["ap_ref"] == trained["eta"]["ap_ref"] == 0.005
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        import argparse
        ap = argparse.ArgumentParser(description="Validation DOE planner")
        ap.add_argument("h5", nargs="?", default=None, help="training reference_dataset*.h5 (asks if missing)")
        ap.add_argument("--name", default="", help="doe_name already filled in (the experiments app passes it)")
        a = ap.parse_args()
        run_gui(a.h5, a.name)
