"""Check (slow, ~10 min, real data):  library + per-case physics reproduce the CONFIG of doe_indicators exactly."""
import copy
import os
import subprocess
import sys

import h5py
import numpy as np

DOE = r"D:/Thesis/03-Code_Storage/02-Altintlas_Nessy2m_Storage/Chatter-Criteria/CAMP10_Chatter_detection_Methodes/DOE_utils"
TMP = __import__("tempfile").mkdtemp(prefix="f3_")
sys.path[:0] = [os.path.join(DOE, "DOE_analisis"), DOE]
import doe_indicators as di  # noqa: E402
import experiment as ex  # noqa: E402

EXP = "train_1DOF150_n12098_k0.5-2.0"
e = ex.load(EXP)
H5 = e.data_h5
CASE = "case_009"

# ---- (a) capture CONFIG RUNS (all 10) and compare the dicts at the CONFIG rpm ------------------------------
cap = {}
orig_run_all = di.run_all
di.run_all = lambda **kw: 0
sys.argv = ["doe_indicators.py", "--doe_results", H5, "--label_key", "$Ap_start$", "--dry_run"]
sys.setprofile(lambda f, ev, a: cap.update(f.f_locals) if ev == "return" and f.f_code.co_name == "main"
               and f.f_globals.get("__name__") == di.__name__ else None)
try:
    di.main()
finally:
    sys.setprofile(None)
    di.run_all = orig_run_all
lib = ex.variants_library()["variants"]
rpm, fmod = 60.0 / cap["_T_REV"], 1.0 / cap["_T_MODAL"]
for r in cap["RUNS"]:
    name = di._run_name(r)
    want = r["indicator_config"]
    got = ex.indicator_config(lib[name], rpm, fmod)
    assert got == want, (name, {k: (got["params_physical"].get(k), v) for k, v in want["params_physical"].items()
                                if got["params_physical"].get(k) != v})
print(f"(a) OK: {len(cap['RUNS'])} variants == CONFIG dicts (rpm {rpm:g}, f_modal {fmod:g})")

# ---- (b) real case: CONFIG path (T_rev of the case) vs experiment path -------------------------------------
with h5py.File(H5, "r") as f:
    spin = float(f[CASE].attrs["$spin_rate$"])
settings_cfg = {"cut": tuple(cap["_CUT_START"] for _ in (0,)) + (cap["_CUT_END"],), "label_key": "kappa",
                "reference_h5": e.reference, "spin_fallback": None}
settings_exp = dict(settings_cfg, cut=(ex.analysis_cut()[0], float("inf")))
results_b = {}
for r_exp in ex.indicator_runs(e):
    name = r_exp["name"]
    r_cfg = copy.deepcopy(next(r for r in cap["RUNS"] if di._run_name(r) == name))
    pp = r_cfg["indicator_config"]["params_physical"]
    if "T_rev" in pp:
        pp["T_rev"] = 60.0 / spin
    if "G_memory" in pp:
        pp["G_memory"] = (60.0 / spin) * 10
    a = di._run_one(H5, CASE, r_cfg, settings_cfg)
    b = di._run_one(H5, CASE, r_exp, settings_exp)
    assert "error" not in a["meta"] and "error" not in b["meta"], (a["meta"].get("error"), b["meta"].get("error"))
    for k in ("t", "I_t", "t_d"):
        assert np.array_equal(a[k], b[k], equal_nan=True), (name, k, a[k][:5], b[k][:5])
    assert a["run_name"] == b["run_name"] == name
    results_b[name] = b
    print(f"(b) OK: {name}: t/I_t/t_d identical ({b['t'].size} windows, first t_d "
          f"{b['t_d'][0] if b['t_d'].size else None})")

# ---- (c) the CLI with --experiment writes the same arrays ---------------------------------------------------
out = os.path.join(TMP, "f3_cli.h5")
if os.path.exists(out):
    os.remove(out)
cmd = [sys.executable, os.path.join(DOE, "DOE_analisis", "doe_indicators.py"), "--experiment", EXP,
       "--cases", CASE, "--workers", "1", "--out", out]
r = subprocess.run(cmd, capture_output=True, text=True)
assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
with h5py.File(out, "r") as f:
    g = f[CASE]
    assert sorted(k for k in g if "I_t" in g[k]) == sorted(results_b), list(g)
    for name, b in results_b.items():
        for k in ("t", "I_t", "t_d"):
            got = g[name][k][()] if k in g[name] else np.array([])
            assert np.array_equal(got, b[k], equal_nan=True), (name, k)
print("(c) OK: doe_indicators --experiment wrote identical groups:", sorted(results_b))
