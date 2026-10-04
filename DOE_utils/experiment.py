#!/usr/bin/env python
# coding: utf-8
"""experiment.py — DOE experiments: one YAML per experiment, stage registry, status, goals and checks.

An experiment (experiments/<name>.yaml) is one DOE (or a single case) and the stages the user turns on: its
simulation written in full inside the file (or an already simulated folder), the labelling, the indicator variants
and, when it points to a reference experiment, the validation against it. Every stage knows what it needs, what it
writes and the command that runs it; the status of a stage comes from its output files plus the run records in
experiments/.runs/<experiment>/<stage>.json. See PLAN_app_experimentos.md and PLAN_app_v2.md.

CLI (entorno_CAMP10 Python):
    python experiment.py status [EXP]           stage table + next step (all experiments without EXP)
    python experiment.py check EXP              configuration errors / warnings
    python experiment.py dryrun EXP             what would run: cases of each run, folders, commands (nothing runs)
    python experiment.py accept EXP [STAGE ...] mark stale stages as up to date (the change does not alter them)
    python experiment.py import NAME DIR [--reference EXP] [--label-out FILE] [--h5 FILE]
    python experiment.py selftest
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "DOE_simulacion")
ANA = os.path.join(HERE, "DOE_analisis")
PLOTS = os.path.join(HERE, "DOE_plots")
EXP_DIR = os.path.join(HERE, "experiments")
RUNS_DIR = os.path.join(EXP_DIR, ".runs")
VARIANTS_FILE = os.path.join(EXP_DIR, "indicator_variants.yaml")


def set_root(path: str) -> None:
    """Point the module at another experiments folder (used by the selftest)."""
    global EXP_DIR, RUNS_DIR, VARIANTS_FILE
    EXP_DIR, RUNS_DIR = path, os.path.join(path, ".runs")
    VARIANTS_FILE = os.path.join(path, "indicator_variants.yaml")
    _LOADED.clear()


# stage order = topological order; main path first, optional branches after
STAGES = ("simulate", "extract", "merge", "label_template", "label_build", "indicators", "validate",
          "static_deflection", "noise", "noise_indicators", "model_snr")
OPTIONAL = {"static_deflection", "noise", "noise_indicators", "model_snr"}
TITLES = {"simulate": "Simulate", "extract": "Extract", "merge": "Merge runs", "label_template": "Label template",
          "label_build": "Label build", "indicators": "Indicators", "validate": "Validate",
          "static_deflection": "Static deflection", "noise": "Noise", "noise_indicators": "Noise indicators",
          "model_snr": "Model SNR"}
MAIN = ("simulate", "extract", "label_template", "label_build", "indicators", "validate")
GOALS = {"Simulated data": ["@data"], "Labelled dataset": ["label_build"], "Indicators computed": ["indicators"],
         "Indicator validation": ["validate"], "Noise robustness": ["noise_indicators"], "Model SNR": ["model_snr"],
         "Static deflection": ["static_deflection"]}
# flow templates: which stages a new experiment turns on (training / validation are just two of them)
FLOWS = {"Simulation only": ["simulate", "extract"],
         "Labelled dataset (training)": ["simulate", "extract", "label_template", "label_build"],
         "Indicators": ["simulate", "extract", "label_template", "label_build", "indicators"],
         "Validation against a reference": ["simulate", "extract", "label_template", "label_build", "indicators",
                                            "validate"]}
# what each signal channel of doe_results.h5 is (Nessy2m sensors of the tool, frame of the tool)
CHANNELS = {"Axial_disp": "tool displacement along its axis (z of the tool frame) [m]",
            "Axial_vel": "tool velocity along its axis [m/s]",
            "Axial_acc": "tool acceleration along its axis [m/s2]",
            "res_R_p": "resultant cutting force [N]",
            "Axial_disp_out_deflex": "axial displacement minus the static deflection [m]"}
LABEL_PARAMS = ("strategy", "amp_signal", "base_attr", "base_scale", "lim_inf_pct", "lim_sup_pct", "warmup",
                "kappa_threshold", "t_start", "t_end", "channels", "window_mode", "window_N", "window_step", "f_modal")
LABEL_FLAGS = {"amp_signal": "--amp-signal", "base_attr": "--base-attr", "base_scale": "--base-scale",
               "lim_inf_pct": "--lim-inf-pct", "lim_sup_pct": "--lim-sup-pct", "warmup": "--warmup",
               "kappa_threshold": "--kappa-threshold", "t_start": "--t-start", "t_end": "--t-end",
               "window_mode": "--window-mode", "window_N": "--window-N", "window_step": "--window-step",
               "f_modal": "--f-modal"}
# the window of the amplitude rule on RAMP cases (PLAN_ramps.md): resolved per case like an indicator's window
WINDOW_KEYS = ("window_mode", "window_N", "window_step")
STRATEGY_SHORT = {"amplitude": "amp", "kappa": "kappa", "manual": "manual"}
DISCRETISATION = ("$dxl_size$", "$nb_dt_rev$", "$f_tooth$")


# ============================================================================== yaml
def yaml_load(path: str) -> dict:
    """safe_load that refuses duplicate keys (plain YAML silently keeps the last one)."""
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    def mapping(loader, node, deep=False):
        seen = set()
        for k_node, _ in node.value:
            k = loader.construct_object(k_node, deep=deep)
            if k in seen:
                raise ValueError(f"{path}: duplicate key {k!r} (line {k_node.start_mark.line + 1})")
            seen.add(k)
        return loader.construct_mapping(node, deep)

    Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.load(f, Loader=Loader) or {}
    if not isinstance(d, dict):
        raise ValueError(f"{path}: expected a 'key: value' map")
    return d


def yaml_save(d: dict, path: str) -> None:
    import yaml
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, sort_keys=False, allow_unicode=True)


def _merge(a: dict, b: dict) -> dict:
    """b over a; dict sections are merged one level deep, everything else is replaced."""
    out = dict(a)
    for k, v in b.items():
        out[k] = {**a[k], **v} if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return out


def exp_path(name: str) -> str:
    if name.endswith((".yaml", ".yml")) and os.path.isfile(name):
        return os.path.abspath(name)
    return os.path.join(EXP_DIR, name + ".yaml")


def load_raw(name: str, seen: tuple = ()) -> dict:
    path = exp_path(name)
    if path in seen:
        raise ValueError(f"circular extends: {' -> '.join(seen + (path,))}")
    if not os.path.isfile(path):
        raise ValueError(f"experiment '{name}' not found ({path})")
    d = yaml_load(path)
    parent = d.pop("extends", None)
    d.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    return _merge({k: v for k, v in load_raw(parent, seen + (path,)).items() if k != "name"}, d) if parent else d


def list_experiments() -> list:
    if not os.path.isdir(EXP_DIR):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(EXP_DIR)
                  if f.endswith(".yaml") and f != os.path.basename(VARIANTS_FILE))


def variants_library() -> dict:
    """{'cut': {...}, 'variants': {...}} or {} when the presets file does not exist yet."""
    return yaml_load(VARIANTS_FILE) if os.path.isfile(VARIANTS_FILE) else {}


def presets() -> dict:
    """{name: spec} of indicator_variants.yaml: the presets a new row of the indicator table starts from."""
    return variants_library().get("variants") or {}


# keys that never change a result (parallelism, launcher, text): left out of the configuration fingerprint
HASH_IGNORE = {"nb_proc", "n2m_bat", "workers", "description", "timed", "auto_extract"}


def _canon(o):
    """Fingerprint form: paths compared case- and separator-insensitively, HASH_IGNORE keys dropped."""
    if isinstance(o, dict):
        return {str(k): _canon(v) for k, v in o.items() if k not in HASH_IGNORE}
    if isinstance(o, (list, tuple)):
        return [_canon(v) for v in o]
    if isinstance(o, str) and (o[1:3] in (":\\", ":/") or o.startswith(("\\\\", "//"))):
        return os.path.normcase(os.path.normpath(o))
    return o


def _hash(obj) -> str:
    return hashlib.sha1(json.dumps(_canon(obj), sort_keys=True, default=str).encode()).hexdigest()[:12]


def _mtime(p: str) -> float:
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0.0


# ============================================================================== ramps of Ap (PLAN_ramps.md)
RAMP_TOL = 1e-9   # [m] |Ap_end - Ap_start| above this = a ramp inside the case


def _attr(a, *keys):
    for k in keys:
        if k in a and a[k] is not None:
            try:
                return float(a[k])
            except (TypeError, ValueError):
                pass
    return None


def is_ramp(a) -> bool:
    """True when the case (attrs of an .h5, var_val or a case_rows row) has Ap_end != Ap_start."""
    a0, a1 = _attr(a, "$Ap_start$", "Ap_start"), _attr(a, "$Ap_end$", "Ap_end")
    return a0 is not None and a1 is not None and abs(a1 - a0) > RAMP_TOL


def ap_of_t(a, t, t_range):
    """Ap [m] at time t of a case: linear from Ap_start at t_range[0] to Ap_end at t_range[1] (the signal span;
    checked in F0 of PLAN_ramps on the cone 5 -> 15 mm: the force grows as Ap, the signal lasts L / Vf)."""
    import numpy as np
    a0 = _attr(a, "$Ap_start$", "Ap_start")
    a1 = _attr(a, "$Ap_end$", "Ap_end")
    a1 = a0 if a1 is None else a1
    t0, t1 = float(t_range[0]), float(t_range[1])
    x = np.clip((np.asarray(t, float) - t0) / (t1 - t0), 0.0, 1.0) if t1 > t0 else np.zeros_like(np.asarray(t, float))
    out = a0 + (a1 - a0) * x
    return float(out) if np.ndim(out) == 0 else out


def case_kappa(a) -> tuple:
    """(kappa, kappa_start, kappa_end) of a case: a constant case has kappa only; a ramp has kappa_start /
    kappa_end and its 'kappa' (if any, = the start) is ignored. None where missing."""
    if is_ramp(a):
        return None, _attr(a, "kappa_start"), _attr(a, "kappa_end")
    return _attr(a, "kappa"), None, None


def kappa_text(a, fmt: str = ".3f") -> str:
    """'1.03' or '0.58 -> 1.74' (ramp; '?' where missing) or '-'."""
    k, k0, k1 = case_kappa(a)
    if is_ramp(a):
        f = lambda v: "?" if v is None else format(v, fmt)   # noqa: E731
        return f"{f(k0)} -> {f(k1)}"
    return "-" if k is None else format(k, fmt)


def ap_text(a, fmt: str = ".4f") -> str:
    """'8.8000 mm' or '5.0000 -> 15.0000 mm' (Ap in m in the attrs)."""
    a0, a1 = _attr(a, "$Ap_start$", "Ap_start"), _attr(a, "$Ap_end$", "Ap_end")
    if a0 is None:
        return "Ap ?"
    if is_ramp(a):
        return f"{format(a0 * 1e3, fmt)} -> {format(a1 * 1e3, fmt)} mm"
    return f"{format(a0 * 1e3, fmt)} mm"


def kappa_span(info) -> tuple | None:
    """(min, max) kappa of a doe_results.h5 (h5_info) counting the ramps (their start and end), or None."""
    ks = list(info["kappa"]) + [k for r in info.get("kappa_ramps", []) for k in r if k is not None]
    return (min(ks), max(ks)) if ks else None


@lru_cache(maxsize=256)
def _h5_info(path: str, mtime: float) -> dict:
    """Cases, first-case attrs and kappa list of a doe_results.h5 (cached by mtime). kappa = the constant cases;
    kappa_ramps = (kappa_start, kappa_end) of each ramp case; ramps = their number."""
    import h5py
    with h5py.File(path, "r") as f:
        cases = sorted(k for k in f if k.startswith("case_"))
        first = {k: (v.item() if hasattr(v, "item") else v) for k, v in f[cases[0]].attrs.items()} if cases else {}
        kappa, kappa_ramps, ramp_text, ramp_cases = [], [], [], []
        for c in cases:
            k, k0, k1 = case_kappa(f[c].attrs)
            if is_ramp(f[c].attrs):
                kappa_ramps.append((k0, k1))
                ramp_cases.append(c)
                ramp_text.append(f"{c}: Ap {ap_text(f[c].attrs, '.3g')}, kappa {kappa_text(f[c].attrs)}")
            elif k is not None:
                kappa.append(k)
        deflex = bool(cases) and "Out_Deflex" in f[cases[0]]
        signals, duration = [], None
        if cases:
            g = f[cases[0]]
            signals = [s for s in g if isinstance(g[s], h5py.Group) and "time" in g[s]]
            if signals:
                duration = float(g[signals[0]]["time"][-1])
        groups = sorted(k for k in f if not k.startswith("case_"))   # control / snr_* in noise files
    return dict(cases=cases, first=first, kappa=kappa, deflex=deflex, signals=signals, duration=duration,
                groups=groups, kappa_ramps=kappa_ramps, ramps=len(kappa_ramps), ramp_text=ramp_text,
                ramp_cases=ramp_cases)


def h5_info(path: str):
    """_h5_info, or None when the file does not exist or cannot be read yet (a stage is writing it: HDF5 locks
    a file while it is open for writing)."""
    if not os.path.isfile(path):
        return None
    try:
        return _h5_info(path, _mtime(path))
    except (OSError, KeyError, ValueError):
        return None


def _doe_runner():
    if SIM not in sys.path:
        sys.path.insert(0, SIM)
    import doe_runner
    return doe_runner


def variant_window(spec: dict) -> tuple:
    """(mode, N, step) of the decision window of an indicator variant, in T_rev or T_modal: the window, or for
    RMS-CV / SST-SVD (aux windows) window + (n_aux - 1) x step, as in the variant names (dec7)."""
    pp, mode = spec.get("params_physical") or {}, spec.get("mode", "by_revolution")
    u = "rev" if mode == "by_revolution" else "modal"
    win, step = float(pp.get(f"N_{u}_window", 0) or 0), float(pp.get(f"step_{u}", 1) or 1)
    n_aux = ((pp.get(f"n_max_{u}") or pp.get(f"Ai_length_{u}"))
             if spec.get("indicator") in ("RMS_CV", "SST_SVD") else None)
    return mode, (win + (float(n_aux) - 1) * step if n_aux else win), step


DEFAULT_WINDOW = {"window_mode": "by_revolution", "window_N": 7.0, "window_step": 1.0}


def default_window(specs: dict, spin=None, f_modal=None) -> dict:
    """Labelling window of the ramps from the indicator variants: theirs when they agree, else the longest (in
    seconds at the n of the first case and f_modal; without them, the longest N of the commonest mode). No
    variants: 7 revolutions, step 1."""
    ws = [variant_window(s) for s in (specs or {}).values() if isinstance(s, dict)]
    ws = [w for w in ws if w[1] > 0]
    if not ws:
        return dict(DEFAULT_WINDOW)
    t = {"by_revolution": 60.0 / float(spin) if spin else None, "by_modal": 1.0 / float(f_modal) if f_modal else None}
    if all(t[m] for m, _, _ in ws):
        mode, n, step = max(ws, key=lambda w: (w[1] * t[w[0]], -w[2]))
    else:
        modes = [m for m, _, _ in ws]
        common = max(set(modes), key=modes.count)
        mode, n, step = max((w for w in ws if w[0] == common), key=lambda w: (w[1], -w[2]))
    out = {"window_mode": mode, "window_N": float(n), "window_step": float(step)}
    if mode == "by_modal" and f_modal:
        out["f_modal"] = float(f_modal)
    return out


# ============================================================================== experiment
def cfg_table(cfg: dict) -> tuple:
    """(variables, rows) of the cases of a loaded doe_runner config."""
    dr = _doe_runner()
    return dr._with_config(cfg, lambda: dr.build_doe_cases(dr.DOE_MODE))


SIM_KEYS = ("base_dir", "case", "n2m_bat", "doe_name", "nb_proc", "mode", "factorial", "sweep", "manual", "ap_ref",
            "extract_signals", "force_signal")   # doe_runner's YAML keys, in the order the app writes them


def _write_if_changed(path: str, text: str) -> None:
    try:
        with open(path, encoding="utf-8") as f:
            if f.read() == text:
                return
    except OSError:
        pass
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


class Run:
    """One DOE run of an experiment:
      simulation: {...}  the doe_runner configuration written in full inside the experiment (the app writes the
                         file doe_runner reads to .runs/<experiment>/sim_<doe_name>.yaml)
      config: NAME       a YAML of DOE_simulacion/configs/ (older experiments; may inherit from base.yaml)
      dir: FOLDER        an already simulated folder (read-only; h5: FILE when it is not doe_results.h5)"""

    def __init__(self, entry: dict, exp_name: str = ""):
        self.entry, self.config, self.cfg, self.source = entry, None, None, ""
        dr = _doe_runner()
        if "simulation" in entry:
            sim = dict(entry["simulation"] or {})
            if "extends" in sim:
                raise ValueError("an explicit simulation cannot use 'extends': write every key")
            import yaml
            self.config = os.path.join(RUNS_DIR, exp_name or "_", f"sim_{sim.get('doe_name', 'doe')}.yaml")
            _write_if_changed(self.config, "# Written by the experiments app from the experiment YAML: edit the "
                                           "experiment, not this file.\n" + yaml.safe_dump(sim, sort_keys=False))
            self.source = "simulation written in the experiment"
        elif "config" in entry:
            self.config = dr.find_config(str(entry["config"]))
            short = self.config[len(SIM) + 1:] if self.config.lower().startswith(SIM.lower()) else self.config
            self.source = f"config {short}" + (" (inherits from base.yaml)" if "extends" in yaml_load(self.config) else "")
        if self.config:
            self.cfg = dr.load_config(self.config)
            self.base_dir, self.doe_name = self.cfg["base_dir"], self.cfg["doe_name"]
            self.case = self.cfg.get("case") or "1DOF_150Hz"
            self.n_cases = len(self.table()[1])
        elif "dir" in entry:
            d = os.path.normpath(str(entry["dir"]))
            self.base_dir, self.doe_name = os.path.dirname(d), os.path.basename(d)
            self.case = entry.get("case") or detect_case(d)
            self.n_cases = None
            self.source = f"imported folder {d}"
        else:
            raise ValueError(f"run entry needs 'simulation', 'config' or 'dir': {entry}")
        self.doe_dir = os.path.join(self.base_dir, self.doe_name)
        self.h5 = os.path.join(self.doe_dir, str(entry.get("h5") or "doe_results.h5"))
        self.imported = self.config is None
        # existing: simulated outside the app, no .h5 yet. The simulation is rebuilt from the folder (var_val.py)
        # so that Extract can run; Simulate stays done and blocked (doe_runner would delete the folder)
        self.existing = bool(entry.get("existing")) and self.config is not None
        # model: the SLD preset of the simulated machine (run entry 'model', else the ap_ref model)
        self.model = entry.get("model") or ((self.cfg or {}).get("ap_ref") or {}).get("model")

    def table(self, cfg: dict | None = None) -> tuple:
        """(variables, rows) of the cases of this run's config (or of cfg)."""
        return cfg_table(cfg or self.cfg)

    def folder_config(self):
        """The config doe_runner copied into the DOE folder when it simulated it (doe_config.yaml), or None."""
        p = os.path.join(self.doe_dir, "doe_config.yaml")
        try:
            return _doe_runner().load_config(p) if os.path.isfile(p) else None
        except Exception:
            return None

    def signature(self) -> dict:
        """What decides the simulated / extracted data: folder, case table, ap_ref, signals. An imported folder
        uses the config doe_runner left in it, so pointing at the same folder through a config or as an import
        gives the same fingerprint."""
        cfg = self.cfg or self.folder_config()
        if not cfg:
            return {"doe_dir": self.doe_dir}
        lst, val = self.table(cfg)
        return {"doe_dir": self.doe_dir, "case": cfg.get("case") or self.case, "vars": lst, "cases": val,
                "ap_ref": cfg.get("ap_ref") or {"mode": "none"},
                "extract_signals": cfg.get("extract_signals"), "force_signal": cfg.get("force_signal")}

    def simulated(self) -> int:
        """Number of index folders with a finished simulation (sens_out.hdf5)."""
        if not os.path.isdir(self.doe_dir):
            return 0
        return sum(os.path.isfile(os.path.join(self.doe_dir, i, self.case, "sens_out.hdf5"))
                   for i in os.listdir(self.doe_dir) if i.isdigit())


def detect_case(doe_dir: str) -> str:
    """Case sub-folder name inside the first index folder (e.g. 0/1DOF_150Hz)."""
    for i in sorted((i for i in os.listdir(doe_dir) if i.isdigit()), key=int) if os.path.isdir(doe_dir) else []:
        for c in os.listdir(os.path.join(doe_dir, i)):
            if os.path.isdir(os.path.join(doe_dir, i, c)):
                return c
    return "1DOF_150Hz"


_LOADED: dict = {}


def _deps_mtime(path: str, seen: tuple = ()) -> float:
    """Newest modification time of everything an experiment is built from besides its own file: the experiments it
    extends / trains on, the simulation configs (configs/*.yaml) and the variant library. A change in any of them
    must reload the experiment (hand edits included)."""
    files = [VARIANTS_FILE]
    if os.path.isdir(CONFIGS_DIR):
        files += [os.path.join(CONFIGS_DIR, f) for f in os.listdir(CONFIGS_DIR) if f.endswith(".yaml")]
    t = max(_mtime(f) for f in files)
    if path in seen:
        return t
    try:
        d = yaml_load(path)
    except Exception:
        return t
    for k in ("extends", "training", "reference"):
        if isinstance(d.get(k), str) and os.path.isfile(exp_path(d[k])):
            t = max(t, _mtime(exp_path(d[k])), _deps_mtime(exp_path(d[k]), seen + (path,)))
    return t


def load(name: str) -> "Exp":
    """Load (cached per process by the mtime of the experiment file and of what it is built from)."""
    path = exp_path(name)
    key = (path, _mtime(path), _deps_mtime(path))
    if key not in _LOADED:
        _LOADED[key] = Exp(name)
    return _LOADED[key]


def reload() -> None:
    _LOADED.clear()


class Exp:
    def __init__(self, name: str):
        self.path = exp_path(name)
        self.cfg = load_raw(name)
        self.name = self.cfg["name"]
        self.errors: list = []
        self.runs = []
        for r in self.cfg.get("runs") or []:
            try:
                self.runs.append(Run(r, self.name))
            except Exception as exc:   # bad config: reported by check(), the rest still loads
                self.errors.append(f"run {r}: {exc}")
        # reference = the experiment whose labelled dataset trains the indicators ('training' in older files)
        ref = self.cfg.get("reference") or self.cfg.get("training")
        self.ref = None
        if ref:
            try:
                self.ref = load(ref)
            except Exception as exc:
                self.errors.append(f"reference '{ref}': {exc}")
        elif self.cfg.get("kind") == "validation":
            self.errors.append("validation experiment without 'reference'")
        # stages the user turned on (older files: the whole main path, validate when validation, all optional)
        st = self.cfg.get("stages")
        if st is None:
            st = [k for k in STAGES if k != "validate" or self.cfg.get("kind") == "validation" or self.ref]
        bad = [k for k in st if k not in STAGES]
        if bad:
            self.errors.append(f"unknown stages {bad}; valid: {list(STAGES)}")
        self.enabled = [k for k in STAGES if k in st]
        merge = self.cfg.get("merge") or {}
        self.merge = merge if len(self.runs) > 1 else {}
        if len(self.runs) > 1 and not merge.get("into"):
            self.errors.append("several runs need 'merge: {into: <new folder name>}'")
        if self.runs:
            r0 = self.runs[0]
            self.data_dir = os.path.join(r0.base_dir, self.merge["into"]) if self.merge.get("into") else r0.doe_dir
            self.data_h5 = os.path.join(self.data_dir, "doe_results.h5") if self.merge.get("into") else r0.h5
        else:
            self.data_dir = self.data_h5 = ""
            self.errors.append("no runs")
        od = self.cfg.get("out_dir")
        self.out_dir = os.path.normpath(str(od)) if od else (os.path.join(self.data_dir, self.name) if self.data_dir else "")
        self.indicators = self._indicators()   # first: the labelling window of the ramps comes from the variants
        self.label = self._label()

    def has_ramps(self) -> bool:
        """True when a case of the data (or, before Extract, of the planned simulation) is a ramp of Ap."""
        info = h5_info(self.data_h5) if self.data_h5 else None
        if info:
            return info["ramps"] > 0
        return any(_planned_ramps(r) for r in self.runs)

    def first_spin(self):
        """n [rpm] of the first case (data, else the planned simulation), or None."""
        info = h5_info(self.data_h5) if self.data_h5 else None
        if info and info["first"].get("$spin_rate$") is not None:
            return float(info["first"]["$spin_rate$"])
        for r in self.runs:
            try:
                lst, val = r.table() if r.cfg else ([], [])
            except Exception:
                continue
            if val and "$spin_rate$" in lst:
                return float(val[0][lst.index("$spin_rate$")])
        return None

    @property
    def flow(self) -> str:
        """Name of the flow template the enabled stages match, else 'custom'."""
        main = [k for k in self.enabled if k in MAIN]
        return next((f for f, ks in FLOWS.items() if ks == main), "custom")

    # ------------------------------------------------------------------ resolved sections
    def section(self, name: str) -> dict:
        return dict(self.cfg.get(name) or {})

    def out(self, section: str, key: str, default: str, shared: bool = False) -> str:
        v = self.section(section).get(key)
        if v and os.path.isabs(str(v)):
            return os.path.normpath(str(v))
        return os.path.join(self.data_dir if shared else self.out_dir, str(v or default))

    def _label(self) -> dict:
        own = self.section("label")
        if self.ref is None:
            lab = {k: own[k] for k in LABEL_PARAMS if k in own}
        else:   # the ground truth is labelled exactly like the reference: parameters come from it
            lab = {k: self.ref.label[k] for k in LABEL_PARAMS if k in self.ref.label}
            diff = [k for k in LABEL_PARAMS if k in own and own[k] != lab.get(k)]
            if diff:
                self.errors.append(f"label {diff} differs from reference '{self.ref.name}' "
                                   f"(labels compared with the reference must use its parameters)")
        lab.setdefault("strategy", "amplitude")
        # ramps labelled by windows: the window written in the experiment (or its reference's); when none is
        # written, the window of the experiment's indicator variants. Only for data with ramps, so the labelling
        # of the constant experiments (and their fingerprint) does not change.
        if lab["strategy"] == "amplitude" and any(k not in lab for k in WINDOW_KEYS) and self.has_ramps():
            for k, v in default_window(self.indicators["specs"], self.first_spin(),
                                       self.indicators.get("f_modal")).items():
                lab.setdefault(k, v)
            lab["window_from"] = "the indicator variants (no window written in the label section)"
        short = STRATEGY_SHORT.get(lab["strategy"], lab["strategy"])
        lab["labels_yaml"] = self.out("label", "labels_yaml", "reference_labels.yaml")
        lab["out"] = self.out("label", "out", f"reference_dataset_{short}.h5")
        return lab

    def _indicators(self) -> dict:
        """variants: {name: spec} written in the experiment, 'inherit' (those of the reference) or, in older
        files, a list of preset names of indicator_variants.yaml. Resolved to ind['specs'] {name: spec}."""
        ind = self.section("indicators")
        v = ind.get("variants", "inherit" if self.ref is not None else [])
        if self.ref is not None:
            for k in ("f_modal", "workers"):
                ind.setdefault(k, self.ref.indicators.get(k))
        if v == "inherit":
            specs = dict(self.ref.indicators["specs"]) if self.ref is not None else {}
        elif isinstance(v, dict):
            specs = dict(v)
        else:
            lib = presets()
            specs = {n: lib[n] for n in v or [] if n in lib}
            miss = [n for n in v or [] if n not in lib]
            if miss:
                self.errors.append(f"indicator variants not in the presets: {miss}")
        for n, s in specs.items():
            if not isinstance(s, dict) or any(k not in s for k in ("indicator", "mode", "signal", "params_physical")):
                self.errors.append(f"variant '{n}' needs indicator, mode, signal and params_physical")
        ind["specs"] = specs
        ind["variants"] = list(specs)
        ind.setdefault("cases", "all")
        ind["out"] = self.out("indicators", "out", "doe_indicator_results.h5")
        return ind

    @property
    def reference(self) -> str:
        """Labelled dataset the indicators learn from (the reference experiment's, else its own)."""
        return (self.ref or self).label["out"]

    @property
    def data_stage(self) -> str:
        return "merge" if self.merge else "extract"

    def runs_dir(self) -> str:
        return os.path.join(RUNS_DIR, self.name)


# ============================================================================== stages
class Stage:
    def __init__(self, exp: Exp, key: str, deps, inputs, outputs, cmds, section, runnable=True, why_not="",
                 present=None, writes=None, roles=None):
        self.exp, self.key, self.title = exp, key, TITLES[key]
        self.deps = deps                    # [(Exp, stage key)]
        self.inputs, self.outputs = inputs, outputs
        self.cmds = cmds                    # [[argv...]] run in order, cwd = folder of the script
        self.hash = _hash(section)
        self.runnable, self.why_not = runnable, why_not
        self.present = present or (lambda: bool(outputs) and all(os.path.exists(p) for p in outputs))
        self.writes = writes if writes is not None else list(outputs)
        self.optional = key in OPTIONAL
        # what each input / output is, for the panel: (in roles, out roles), one text per path
        self.roles = roles or ([""] * len(inputs), [""] * len(outputs))


def _py(script: str, *args) -> list:
    return [script, *[str(a) for a in args]]


def stages(exp: Exp) -> dict:
    """{key: Stage} for the stages the experiment turned on (plus the ones they need), in topological order."""
    S = _all_stages(exp)
    keep, todo = set(), [k for k in exp.enabled if k in S]
    while todo:   # an enabled stage brings the stages of this experiment it depends on
        k = todo.pop()
        if k not in keep:
            keep.add(k)
            todo += [dk for de, dk in S[k].deps if de is exp and dk in S]
    return {k: S[k] for k in STAGES if k in keep}


def _all_stages(exp: Exp) -> dict:
    S, dr_script = {}, os.path.join(SIM, "doe_runner.py")
    runs, data, label, ind = exp.runs, exp.data_h5, exp.label, exp.indicators
    ref_exp = exp.ref or exp
    if not runs:
        return S
    cfg_runs = [r for r in runs if not r.imported]
    imported_why = "imported run (already simulated folder, no simulation config): it cannot be re-run from the app"
    sim_opts = exp.section("simulate")
    flags = (["--timed"] if sim_opts.get("timed") else []) + (["--auto-extract"] if sim_opts.get("auto_extract") else [])
    sig = [r.signature() for r in runs]

    # inputs compared by date: only a config file the user edits (the file generated from the experiment is
    # rewritten by the app; its content is covered by the fingerprint)
    sim_in = [r.config for r in cfg_runs if "simulation" not in r.entry]
    S["simulate"] = Stage(exp, "simulate", [], sim_in, [r.doe_dir for r in runs],
                          [_py(dr_script, "--config", r.config, "--yes", "--command", "n2m_sch", *flags) for r in cfg_runs],
                          sig, runnable=not any(r.imported or r.existing for r in runs),
                          why_not=imported_why if any(r.imported for r in runs) else
                          "simulated outside the app (imported without .h5): re-simulating would delete that folder; "
                          "run Extract",
                          present=lambda: all(r.imported and os.path.isdir(r.doe_dir) or
                                              (r.n_cases or 0) > 0 and r.simulated() >= r.n_cases for r in runs),
                          roles=(["simulation config"] * len(sim_in), ["DOE folder (one sub-folder per case)"] * len(runs)))
    S["extract"] = Stage(exp, "extract", [(exp, "simulate")], [], [r.h5 for r in runs],
                         [_py(dr_script, "--config", r.config, "--yes", "--command", "extract") for r in cfg_runs],
                         sig, runnable=not any(r.imported for r in runs),
                         why_not=imported_why, roles=([], ["signals of every case (doe_results.h5)"] * len(runs)))
    if exp.merge:
        r0, into = runs[0], exp.merge["into"]
        base_flags = ["--config", r0.config] if r0.config else ["--case_dir", os.path.join(r0.base_dir, r0.case)]
        cmds = [_py(dr_script, *base_flags, "--yes", "--command", "merge", "--doe_name", r0.doe_name,
                    "--merge_from", runs[1].doe_name, "--merge_out", into)]
        cmds += [_py(dr_script, *base_flags, "--yes", "--command", "merge", "--doe_name", into,
                     "--merge_from", r.doe_name) for r in runs[2:]]   # later runs go into our own new folder
        S["merge"] = Stage(exp, "merge", [(exp, "extract")], [r.h5 for r in runs], [data], cmds, exp.merge,
                           roles=([f"doe_results.h5 of {r.doe_name}" for r in runs], ["merged doe_results.h5"]))
    d = exp.data_stage
    lab_params = {k: v for k, v in label.items() if k in LABEL_PARAMS}
    tpl = [os.path.join(SIM, "reference_dataset.py"), "template", data, label["labels_yaml"],
           "--strategy", label["strategy"]]
    for k, flag in LABEL_FLAGS.items():
        if label.get(k) is not None:
            tpl += [flag, str(label[k])]
    S["label_template"] = Stage(exp, "label_template", [(exp, d)], [data], [label["labels_yaml"]], [tpl],
                                {**lab_params, "data": data},
                                roles=(["data"], ["labels YAML: one label per case (review it before building)"]))
    build = [os.path.join(SIM, "reference_dataset.py"), "build", data, label["labels_yaml"], label["out"]]
    if label.get("channels"):
        build += ["--channels", *map(str, label["channels"])]
    for k in ("t_start", "t_end"):
        if label.get(k) is not None:
            build += [LABEL_FLAGS[k], str(label[k])]
    S["label_build"] = Stage(exp, "label_build", [(exp, "label_template")], [data, label["labels_yaml"]],
                             [label["out"]], [build], {**lab_params, "data": data},
                             roles=(["data", "labels YAML"],
                                    ["labelled dataset = ground truth" + ("" if exp.ref else
                                                                          "; the indicators learn from it")]))
    ind_section = {"f_modal": ind.get("f_modal"), "cases": ind.get("cases"), "reference": exp.reference,
                   "specs": ind["specs"], "cut": list(analysis_cut())}
    S["indicators"] = Stage(exp, "indicators", [(exp, d), (ref_exp, "label_build")], [data, exp.reference],
                            [ind["out"]], [_py(os.path.join(ANA, "doe_indicators.py"), "--experiment", exp.path)],
                            ind_section,
                            roles=(["data", f"labelled dataset the indicators learn from (experiment {ref_exp.name})"],
                                   ["indicator results: I(t) and detections per case and variant"]))
    val = exp.section("validate")
    # the tolerance of the early-detection rule is always part of the fingerprint: validations scored before the
    # rule existed (any alarm = hit) turn stale, and re-running Validate takes seconds
    val.setdefault("early_tol_s", EARLY_TOL_S)
    val_out = exp.out("validate", "out", "doe_validation_results.h5")
    cmd = _py(os.path.join(ANA, "validate_indicators.py"), "--ind_results", ind["out"], "--labels", label["out"],
              "--reference", exp.reference, "--out", val_out, "--channel", val.get("channel", "Axial_disp"),
              "--early-tol", val["early_tol_s"])
    S["validate"] = Stage(exp, "validate", [(exp, "label_build"), (exp, "indicators")],
                          [ind["out"], label["out"]], [val_out], [cmd], {**val, "out": val_out},
                          roles=(["indicator results", "ground truth (this experiment's labelled dataset)"],
                                 ["validation results: metrics per variant (+ _metrics.csv)"]))
    # optional branches (configured by their YAML section only)
    sd = exp.section("static_deflection")
    S["static_deflection"] = Stage(exp, "static_deflection", [(exp, d)], [], [data],
                                   [_py(os.path.join(SIM, "static_deflection.py"), data, "--experiment", exp.path)],
                                   sd, present=lambda: bool((h5_info(data) or {}).get("deflex")), writes=[data],
                                   roles=([], ["data (adds the Out_Deflex group inside it)"]))
    noise_out = exp.out("noise", "out", "doe_noise_results.h5", shared=True)
    S["noise"] = Stage(exp, "noise", [(exp, d)], [data], [noise_out],
                       [_py(os.path.join(SIM, "doe_noise.py"), "--doe_results", data, "--out", noise_out,
                            "--experiment", exp.path)], exp.section("noise"),
                       roles=(["data"], ["noisy signals of a control case"]))
    ni_out = exp.out("noise_indicators", "out", "doe_noise_indicator_results.h5")
    S["noise_indicators"] = Stage(exp, "noise_indicators", [(exp, "noise"), (ref_exp, "label_build")],
                                  [noise_out, exp.reference], [ni_out],
                                  [_py(os.path.join(ANA, "doe_indicators.py"), "--experiment", exp.path,
                                       "--doe_results", noise_out, "--out", ni_out)],
                                  {**ind_section, **exp.section("noise_indicators")},
                                  roles=(["noisy signals", f"labelled dataset (experiment {ref_exp.name})"],
                                         ["indicator results on the noisy signals"]))
    snr_out = exp.out("model_snr", "out", "doe_model_snr_results.h5", shared=True)
    r0 = runs[0]
    S["model_snr"] = Stage(exp, "model_snr", [(exp, "simulate")], [], [snr_out],
                           [_py(os.path.join(ANA, "doe_model_snr.py"), "--experiment", exp.path,
                                "--doe_name", r0.doe_name, "--out", snr_out)], exp.section("model_snr"),
                           roles=([], ["model SNR per case"]))
    return {k: S[k] for k in STAGES if k in S}


# ============================================================================== run records + status
def record_path(exp: Exp, key: str) -> str:
    return os.path.join(exp.runs_dir(), f"{key}.json")


def read_record(exp: Exp, key: str):
    try:
        with open(record_path(exp, key), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_record(exp: Exp, key: str, rec: dict) -> None:
    os.makedirs(exp.runs_dir(), exist_ok=True)
    tmp = record_path(exp, key) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, record_path(exp, key))


def pid_alive(pid) -> bool:
    """True if a process with this PID is running. Windows: OpenProcess/GetExitCodeProcess (os.kill would KILL it)."""
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, int(pid))   # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259          # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _fmt_time(ts) -> str:
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "?"


def status(exp: Exp, _memo=None) -> dict:
    """{key: (state, reason)}; state in done / stale / running / failed / pending / blocked / skipped.
    skipped = never run, but a later stage of this experiment already has its result (e.g. an imported labelled
    dataset made from another labels YAML): nothing to do."""
    memo = {} if _memo is None else _memo
    if exp.name in memo:
        return memo[exp.name]
    memo[exp.name] = out = {}
    S = stages(exp)
    for key, st in S.items():
        out[key] = _stage_state(exp, st, S, out, memo)
    covered = {}
    for key in reversed(list(S)):
        if out[key][0] in ("done", "stale") or key in covered:
            for de, dk in S[key].deps:
                if de is exp:
                    covered.setdefault(dk, covered.get(key, key))
    for key, by in covered.items():
        if out[key][0] == "pending":
            out[key] = ("skipped", f"not run: '{TITLES[by]}' already has its result without it (e.g. imported)")
    for key, (state, reason) in out.items():
        if state == "done" and "earlier step missing" in reason:
            out[key] = (state, "outputs exist, made without the earlier step (imported)")
    return out


def _stage_state(exp, st, S, own, memo):
    dep_states = []
    for dexp, dkey in st.deps:
        states = own if dexp is exp else status(dexp, memo)
        if dkey not in states:
            return "blocked", f"needs '{dkey}'" + ("" if dexp is exp else f" of experiment '{dexp.name}'")
        dep_states.append((dexp, dkey, states[dkey][0]))
    rec = read_record(exp, st.key)
    if rec and rec.get("status") == "running":
        if pid_alive(rec.get("pid")):
            return "running", f"started {_fmt_time(rec.get('start'))}"
        if not rec.get("end"):
            rec = dict(rec, status="failed", exit_code="?")   # process gone without writing its end
    missing = ""
    for dexp, dkey, s in dep_states:
        if s in ("pending", "blocked", "failed", "running", "skipped"):
            where = "" if dexp is exp else f" of experiment '{dexp.name}'"
            verb = "waits for" if s == "running" else "needs"
            missing = f"{verb} '{dkey}'{where} ({s})"
            break
    present = st.present()
    if missing and (not present or missing.startswith("waits for")):
        return "blocked", missing
    if rec and rec.get("status") == "failed":
        return "failed", f"exit code {rec.get('exit_code')} at {_fmt_time(rec.get('end') or rec.get('start'))}"
    if not present:
        return "pending", "ready to run" if st.runnable else st.why_not
    ref = max(rec.get("start") or 0, rec.get("accepted") or 0) if rec else min((_mtime(p) for p in st.outputs if os.path.exists(p)), default=0)
    if rec and rec.get("hash") and rec["hash"] != st.hash:
        return "stale", "configuration changed since the last run"
    newer = [p for p in st.inputs if p and _mtime(p) > (ref or 0) + 1]
    if newer:
        return "stale", f"input changed after the run: {os.path.basename(newer[0])}"
    up = [dkey for _, dkey, s in dep_states if s == "stale"]
    if up:
        return "stale", f"upstream '{up[0]}' is stale"
    if missing:   # output already there (e.g. imported) although an earlier step is missing: keep it usable
        return "done", f"outputs exist; earlier step missing: {missing}"
    if rec is None:
        return "done", "outputs exist (no run record: made outside the app)"
    return "done", ("imported " if rec.get("status") == "imported" else "") + _fmt_time(rec.get("end") or rec.get("start"))


# ============================================================================== goals
def goal_chain(exp: Exp, goal: str) -> list:
    """[(Exp, key)] needed for the goal, in run order (dependencies first)."""
    S = stages(exp)
    targets = [exp.data_stage if t == "@data" else t for t in GOALS[goal]]
    out, seen = [], set()

    def visit(e, k):
        if (e.name, k) in seen:
            return
        seen.add((e.name, k))
        st = stages(e).get(k)
        if st is None:
            return
        for de, dk in st.deps:
            visit(de, dk)
        out.append((e, k))

    for t in targets:
        if t in S:
            visit(exp, t)
    return out


def goals(exp: Exp) -> list:
    """Goals whose stages this experiment has."""
    S = stages(exp)
    return [g for g, ts in GOALS.items() if all((exp.data_stage if t == "@data" else t) in S for t in ts)]


def default_goal(exp: Exp) -> str:
    """The furthest stage of the main path the experiment turned on."""
    for g, k in (("Indicator validation", "validate"), ("Indicators computed", "indicators"),
                 ("Labelled dataset", "label_build")):
        if k in exp.enabled:
            return g
    return "Simulated data"


def next_step(exp: Exp, goal: str | None = None):
    """(Exp, key, state, reason) of the first stage of the goal that is not done; None when the goal is reached."""
    todo = todo_stages(exp, goal)
    return todo[0] if todo else None


def todo_stages(exp: Exp, goal: str | None = None) -> list:
    """[(Exp, key, state, reason)] still to do for the goal, in run order: this experiment's stages and those of
    the experiments it needs (its reference: simulation, labelled dataset)."""
    goal = goal if goal in GOALS else default_goal(exp)
    memo: dict = {}
    S = stages(exp)
    todo, seen = [], set()

    def need(e, k):   # post-order from the targets; a stage that is done covers everything before it
        if (e.name, k) in seen or k not in stages(e):
            return
        seen.add((e.name, k))
        state, reason = status(e, memo)[k]
        if state in ("done", "skipped"):
            return
        for de, dk in stages(e)[k].deps:
            need(de, dk)
        todo.append((e, k, state, reason))

    for t in GOALS[goal]:
        t = exp.data_stage if t == "@data" else t
        if t in S:
            need(exp, t)
    # the reference's stages first (the training before the test): they never depend on this experiment
    return [x for x in todo if x[0] is not exp] + [x for x in todo if x[0] is exp]


# ============================================================================== checks
def check(exp: Exp) -> tuple:
    """(errors, warnings) of the whole experiment configuration."""
    errs, warns = list(exp.errors), []
    for r in exp.runs:
        if r.imported and not os.path.isdir(r.doe_dir):
            errs.append(f"imported run folder not found: {r.doe_dir}")
    if len({r.base_dir for r in exp.runs}) > 1 and exp.merge:
        errs.append("merged runs must live in the same base_dir (doe_runner merges sibling folders)")
    for r in exp.runs:
        if r.cfg:
            e, w = sim_problems(r.cfg)
            errs += [f"run {r.doe_name}: {x}" for x in e]
            warns += [f"run {r.doe_name}: {x}" for x in w]
    names = [r.doe_dir.lower() for r in exp.runs]
    if len(set(names)) < len(names):
        errs.append("two runs write the same DOE folder (change doe_name)")
    inherit = (exp.cfg.get("indicators") or {}).get("variants", "inherit" if exp.ref else []) == "inherit"
    if "indicators" in exp.enabled and not exp.indicators["specs"]:
        errs.append(f"the reference '{exp.ref.name}' has no indicator variants to inherit: choose them in Indicators "
                    "> Edit config (untick 'Same variants as the reference')" if inherit and exp.ref is not None
                    else "indicators stage on, but no indicator variant chosen (Indicators > Edit config)")
    if exp.ref is not None:   # training and test must come from the same simulated machine
        mine = {r.model for r in exp.runs if r.model}
        theirs = {r.model for r in exp.ref.runs if r.model}
        if mine and theirs and mine != theirs:
            errs.append(f"simulated model {sorted(mine)} differs from the reference '{exp.ref.name}' "
                        f"{sorted(theirs)}: the indicators would be tested on another machine")
    if exp.ref is not None and "label_build" not in stages(exp.ref):
        errs.append(f"the reference '{exp.ref.name}' has no labelled dataset (Label build is off there): turn on its "
                    "label stages in 'Experiment settings…' of that experiment (flow 'Labelled dataset'), or import "
                    "it with its reference_dataset*.h5")
    if "validate" in exp.enabled and exp.ref is None:
        warns.append("validate without a reference: the indicators are scored on the same cases they learned from")
    if exp.ref is not None and exp.ref.name == exp.name:
        errs.append("an experiment cannot be its own reference")
    if exp.ref is not None:
        mine, theirs = h5_info(exp.data_h5), h5_info(exp.ref.data_h5)
        if mine and theirs:
            diff = [k for k in DISCRETISATION if k in mine["first"] and k in theirs["first"]
                    and abs(float(mine["first"][k]) - float(theirs["first"][k])) > 1e-12]
            if diff:
                errs.append(f"discretisation differs from the reference: {diff}")
            n1, n2 = mine["first"].get("$spin_rate$"), theirs["first"].get("$spin_rate$")
            if n1 is not None and n2 is not None and abs(float(n1) - float(n2)) > 1e-6:
                warns.append(f"spin differs from the reference ({n1} vs {n2} rpm): generalisation test")
            rep = sorted({round(k, 6) for k in mine["kappa"]} & {round(k, 6) for k in theirs["kappa"]})
            if rep:
                warns.append(f"{len(rep)} kappa also in the reference: {rep[:5]}")
    gray = _gray_fraction(exp.label["out"])
    if gray is not None and gray > 0.25:
        warns.append(f"{gray:.0%} of the cases are 'gray' in {os.path.basename(exp.label['out'])}")
    return errs, warns


def _gray_fraction(path: str):
    if not os.path.isfile(path):
        return None
    import h5py
    with h5py.File(path, "r") as f:
        n = {g: len(f[g]) for g in f if g in ("stable", "unstable", "gray")}
    tot = sum(n.values())
    return n.get("gray", 0) / tot if tot else None


def summary(exp: Exp) -> dict:
    """What the experiments list shows: n, cases, kappa range, main-path progress."""
    info = h5_info(exp.data_h5) if exp.data_h5 else None
    spin = None
    if info:
        spin = info["first"].get("$spin_rate$")
    elif exp.runs and exp.runs[0].cfg:
        spin = (exp.runs[0].cfg.get("sweep") or exp.runs[0].cfg.get("factorial") or {}).get("$spin_rate$", [None])[0]
    st = status(exp)
    main = [k for k in st if k not in OPTIONAL]
    if info:
        span, ramps = kappa_span(info), info["ramps"]
    else:   # not extracted yet: the ramps of the planned cases (no kappa: it needs the SLD)
        span, ramps = None, sum(_planned_ramps(r) for r in exp.runs)
    return dict(spin=spin, cases=len(info["cases"]) if info else sum(r.n_cases or 0 for r in exp.runs),
                kappa=span, ramps=ramps, done=sum(st[k][0] in ("done", "skipped") for k in main), total=len(main))


def _planned_ramps(r: "Run") -> int:
    if not r.cfg:
        return 0
    try:
        lst, val = r.table()
    except Exception:
        return 0
    return sum(is_ramp(dict(zip(lst, row))) for row in val)


# ============================================================================== indicator variants
def resolve_physics(value, T_rev, T_modal):
    """'T_rev' / 'T_modal' / '<k>*T_rev' / 'T_rev*<k>' -> number for this case; anything else unchanged."""
    if not isinstance(value, str):
        return value
    s = value.replace(" ", "")
    for name, val in (("T_rev", T_rev), ("T_modal", T_modal)):
        if name not in s:
            continue
        if val is None:
            raise ValueError(f"'{value}' needs {name} ({'$spin_rate$ of the case' if name == 'T_rev' else 'f_modal'})")
        if s == name:
            return val
        if s.endswith("*" + name):
            return float(s[:-len(name) - 1]) * val
        if s.startswith(name + "*"):
            return val * float(s[len(name) + 1:])
    return value


def indicator_config(variant: dict, spin, f_modal) -> dict:
    """INDICATOR_CONFIG dict of a variant for one case (what doe_indicators passes to the runner)."""
    t_rev = 60.0 / float(spin) if spin else None
    t_modal = 1.0 / float(f_modal) if f_modal else None
    pp = {k: resolve_physics(v, t_rev, t_modal) for k, v in variant["params_physical"].items()}
    if variant["indicator"] == "MaxEnt_SPRT":   # its signature asks for it; unused with the external reference
        pp.setdefault("t_stable_total", None)
    return {"id": variant["indicator"], "func": variant.get("func", "Default"), "param_mode": variant["mode"],
            "params_physical": pp}


def indicator_runs(exp: Exp) -> list:
    """RUNS for doe_indicators: one entry per variant of the experiment, physics resolved per case ('variant')."""
    if exp.errors:
        raise ValueError("; ".join(exp.errors))
    specs = exp.indicators["specs"]
    return [{"enabled": True, "name": v, "signal": s["signal"], "variant": s,
             "f_modal": exp.indicators.get("f_modal")} for v, s in specs.items()]


def analysis_cut() -> tuple:
    """(start, end) of the analysed signal; end None = end of each case."""
    c = variants_library().get("cut") or {}
    return float(c.get("start", 0.0)), (None if c.get("end") is None else float(c["end"]))


def resolve(exp: Exp, variant: str, case: str, h5: str | None = None) -> dict:
    """Final config of a variant for one case of the experiment data (experiment.py resolve / app preview)."""
    import h5py
    specs = exp.indicators["specs"]
    if variant not in specs:
        raise ValueError(f"variant '{variant}' is not in the experiment")
    with h5py.File(h5 or exp.data_h5, "r") as f:
        if case not in f:
            raise ValueError(f"case '{case}' not in {h5 or exp.data_h5}")
        spin = f[case].attrs.get("$spin_rate$")
    cfg = indicator_config(specs[variant], spin, exp.indicators.get("f_modal"))
    return {"variant": variant, "case": case, "signal": specs[variant]["signal"], "spin_rate": spin,
            "reference": exp.reference, "cut": list(analysis_cut()), "indicator_config": cfg}


def section_overrides(name: str, section: str, allowed) -> tuple:
    """({CONSTANT: value}, Exp) from the `section` of an experiment, for scripts whose CONFIG are module
    constants: the YAML keys are those names in lower case (snr_range -> SNR_RANGE). Unknown keys raise;
    'out' is not a constant (the app passes --out)."""
    exp = load(name)
    sec = {k: v for k, v in exp.section(section).items() if k != "out"}
    bad = [k for k in sec if k.upper() not in allowed]
    if bad:
        raise ValueError(f"{exp.path} [{section}]: unknown keys {bad}; valid: {sorted(a.lower() for a in allowed)}")
    return {k.upper(): v for k, v in sec.items()}, exp


# ============================================================================== what each stage is / what it produced
STAGE_INFO = {
    "simulate": ("Runs Nessy2m for every case of the run configs: one folder per case with sens_out.hdf5.",
                 "All cases simulated. Long stage: about the time per case x cases / nb_proc."),
    "extract": ("Collects every simulated case into doe_results.h5 (signals + DOE attributes and kappa).",
                "Number of cases, kappa range and spin are the planned ones."),
    "merge": ("Joins the doe_results.h5 of several runs into a new folder (the cases of the next runs are renumbered).",
              "The merged file has the cases of every run."),
    "label_template": ("Pre-fills reference_labels.yaml: one stable / gray / unstable label per case (amplitude "
                       "criterion: max|signal| against a % of the feed per tooth); a RAMP of Ap gets the same rule "
                       "window by window (several intervals: where its truth turns unstable).",
                       "Review the YAML before building: these labels are the ground truth."),
    "label_build": ("Cuts the labelled signals into reference_dataset*.h5: the ground truth of this experiment. "
                    "Without a reference experiment it is also what the indicators learn from.",
                    "Where the stable/unstable boundary falls and how many cases are gray."),
    "indicators": ("Runs the indicator variants on every case. Each variant learns its threshold from the stable "
                   "(and, MaxEnt, unstable) pieces of a labelled dataset: the reference experiment's, or this "
                   "experiment's own. That is why it needs Label build first. T_rev comes from the spin of each case.",
                   "Every case x variant done, no errors; how many stable / unstable cases each variant flags."),
    "validate": ("Scores each variant against the validation labels: TP/FN/TN/FP per case (an alarm earlier "
                 "than early_tol_s before the onset of the truth is an early alarm, counted as FN), balanced "
                 "accuracy, MCC, AUC and detection times; ramps that cross are scored apart (ramp_* metrics).",
                 "Ranking of the variants; the Compare tab puts two validations side by side."),
    "static_deflection": ("Adds the theoretical static deflection (group Out_Deflex) to doe_results.h5.",
                          "Its section of the experiment YAML sets f_tooth_mm, k_cut, k_sys, alpha_deg, theta_deg."),
    "noise": ("Builds doe_noise_results.h5: one control case with Gaussian noise at several SNR levels.",
              "Its section sets control_case_idx, snr_list / snr_range, seed, signals."),
    "noise_indicators": ("Runs the indicator variants on the noisy signals (robustness to noise).",
                         "Every SNR level x variant done."),
    "model_snr": ("Model SNR of every case against a control case, from the simulation folders.",
                  "Its section sets control_idx."),
}


def _fmt_dur(sec) -> str:
    if sec is None:
        return "?"
    sec = float(sec)
    return f"{sec:.0f} s" if sec < 90 else (f"{sec / 60:.0f} min" if sec < 5400 else f"{sec / 3600:.1f} h")


def _krange(ks) -> str:
    return (f"{min(ks):g}" if min(ks) == max(ks) else f"{min(ks):g}-{max(ks):g}") if ks else "-"


def _wall_times(r: "Run") -> list:
    out = []
    if os.path.isdir(r.doe_dir):
        for i in os.listdir(r.doe_dir):
            p = os.path.join(r.doe_dir, i, "wall_time_s.txt")
            if i.isdigit() and os.path.isfile(p):
                try:
                    out.append(float(open(p).read().strip()))
                except ValueError:
                    pass
    return out


def case_label(intervals) -> str:
    """One label for a case from its intervals [(t0, t1, label)]: 'mixed' when it has stable AND unstable parts
    (a ramp that crosses), else unstable / stable / gray (gray only when nothing else), 'none' without intervals.
    Same rule as validate_indicators.case_truth."""
    labs = {iv[2] for iv in intervals}
    if "unstable" in labs:
        return "mixed" if "stable" in labs else "unstable"
    return "stable" if "stable" in labs else ("gray" if "gray" in labs else "none")


def t_onset(intervals):
    """Start of the first unstable interval (the crossing of the ground truth of a ramp), or None."""
    return min((float(iv[0]) for iv in intervals if iv[2] == "unstable"), default=None)


def transitions_text(intervals, n: int = 4) -> str:
    """'stable -> unstable at 10.66 s' (the label changes of a case, the first n)."""
    iv = sorted(intervals, key=lambda x: float(x[0]))
    if not iv:
        return "no labels"
    if len(iv) == 1:
        return f"{iv[0][2]} all along"
    parts = [iv[0][2]] + [f"{b[2]} at {float(b[0]):.2f} s" for b in iv[1:n + 1]]
    return " -> ".join(parts) + (f" (+{len(iv) - n - 1} changes)" if len(iv) > n + 1 else "")


def label_info(path: str) -> dict:
    """{case: {label, kappa, intervals, t_onset, ramp}} of a reference_dataset*.h5: label from every piece of the
    case (case_label), kappa of a constant case (NaN for a ramp), intervals of its labelling channel."""
    import h5py
    out = {}
    if os.path.isfile(path):
        with h5py.File(path, "r") as f:
            for lab in f:
                for case, g in f[lab].items():
                    d = out.setdefault(case, {"pieces": [], "kappa": float("nan"), "ramp": False, "ch": None})
                    for piece in g.values():
                        a = piece.attrs
                        if d["ch"] is None:
                            d["ch"] = str(a.get("channel", ""))
                            k = case_kappa(a)[0]
                            d["kappa"], d["ramp"] = (float("nan") if k is None else k), is_ramp(a)
                        if str(a.get("channel", "")) == d["ch"] and "t0" in a:
                            d["pieces"].append((float(a["t0"]), float(a["t1"]), lab))
                        elif "t0" not in a:
                            d["pieces"].append((0.0, 0.0, lab))
    for d in out.values():
        iv = sorted(set(d.pop("pieces")))
        d.update(intervals=iv, label=case_label(iv), t_onset=t_onset(iv))
        d.pop("ch")
    return out


def _label_cases(path: str) -> dict:
    """{case: (label, kappa)} of a reference_dataset*.h5: label stable / gray / unstable / mixed (a ramp with
    stable and unstable parts); kappa of a constant case (NaN for a ramp: it has no single kappa)."""
    return {c: (d["label"], d["kappa"]) for c, d in label_info(path).items()}


def yaml_label_info(path: str) -> dict:
    """{case: {label, intervals, t_onset}} of a reference_labels.yaml."""
    if not os.path.isfile(path):
        return {}
    d = yaml_load(path).get("cases") or {}
    out = {}
    for c, iv in d.items():
        iv = [tuple(x) for x in iv or []]
        out[c] = {"label": case_label(iv), "intervals": iv, "t_onset": t_onset(iv)}
    return out


def _yaml_labels(path: str) -> dict:
    """{case: label} of a reference_labels.yaml (case_label of its intervals: 'none' without any)."""
    return {c: d["label"] for c, d in yaml_label_info(path).items()}


def stage_progress(exp: Exp, key: str):
    """(done, total) while a stage advances, else None: simulated cases, the cases extracted so far ('Casos
    encontrados: N' / 'Caso k extraido' of doe_runner's log) or '[k/N] completado' of the indicators' log."""
    if key == "simulate":
        tot = sum(r.n_cases or 0 for r in exp.runs if not r.imported)
        return (sum(min(r.simulated(), r.n_cases or 0) for r in exp.runs if not r.imported), tot) if tot else None
    rec = read_record(exp, key)
    log = rec.get("log") if rec else None
    if key == "extract" and log and os.path.isfile(log):
        import re
        with open(log, encoding="utf-8", errors="replace") as f:
            txt = f.read()
        tot = sum(int(n) for n in re.findall(r"Casos encontrados: (\d+)", txt))
        return (len(re.findall(r"Caso \d+ extraido", txt)), tot) if tot else None
    if key in ("indicators", "noise_indicators") and log and os.path.isfile(log):
        import re
        with open(log, encoding="utf-8", errors="replace") as f:
            hits = re.findall(r"\[(\d+)/(\d+)\] completado", f.read())
        if hits:
            return int(hits[-1][0]), int(hits[-1][1])
    return None


def run_timing(exp: Exp, key: str) -> dict:
    """{'elapsed', 'duration', 'eta'} in seconds from the run record (+ progress for the ETA)."""
    rec = read_record(exp, key) or {}
    out = {}
    if rec.get("start") and rec.get("status") in ("done", "failed") and rec.get("end"):
        out["duration"] = rec["end"] - rec["start"]
    if rec.get("status") == "running" and rec.get("start"):
        out["elapsed"] = time.time() - rec["start"]
        pr = stage_progress(exp, key)
        if pr and pr[0] > 0 and pr[1]:
            out["eta"] = out["elapsed"] * (pr[1] - pr[0]) / pr[0]
    return out


def stage_summary(exp: Exp, key: str) -> list:
    """What the outputs of the stage contain: [(text, tag)], tag in ok / warn / bad / None. Never raises."""
    rec = read_record(exp, key)
    if key != "simulate" and rec and rec.get("status") == "running" and pid_alive(rec.get("pid")):
        return [("being written by the running stage: its content appears when it ends (progress below)", None)]
    try:
        return _stage_summary(exp, key)
    except Exception as exc:   # an unreadable / half-written file must not break the panel
        return [(f"could not read the outputs: {exc}", "warn")]


def _stage_summary(exp: Exp, key: str) -> list:
    S = stages(exp)
    if key not in S:
        return []
    out = []
    if key == "simulate":
        for r in exp.runs:
            wt = _wall_times(r)
            if r.imported:
                n = len((h5_info(r.h5) or {}).get("cases", []))
                out.append((f"{r.doe_name}: imported folder, {n} cases in doe_results.h5", "ok"))
            else:
                done = r.simulated()
                tag = "ok" if done >= (r.n_cases or 0) else ("warn" if done else None)
                txt = f"{r.doe_name}: {done}/{r.n_cases} cases simulated"
                if wt:
                    left = max((r.n_cases or 0) - done, 0)
                    nb = (r.cfg or {}).get("nb_proc") or 1
                    txt += f", {_fmt_dur(sum(wt) / len(wt))} per case" + (
                        f", ~{_fmt_dur(sum(wt) / len(wt) * left / nb)} left" if left else "")
                out.append((txt, tag))
            if wt and r.imported:
                out.append((f"  {_fmt_dur(sum(wt) / len(wt))} per case on average ({len(wt)} cases)", None))
        return out
    if key in ("extract", "merge", "static_deflection"):
        files = [r.h5 for r in exp.runs] if key == "extract" else [exp.data_h5]
        for p in files:
            info = h5_info(p)
            if not info:
                out.append((f"{os.path.basename(os.path.dirname(p))}: no doe_results.h5 yet", None))
                continue
            a = info["first"]
            if key == "static_deflection":
                out.append(("Out_Deflex present" if info["deflex"] else "no Out_Deflex yet", "ok" if info["deflex"] else None))
                if "deflex_theoric_m" in a:
                    out.append((f"theoretical deflection (case 0): {float(a['deflex_theoric_m']):.3e} m", None))
                continue
            out.append((f"{len(info['cases'])} cases, kappa {_krange(info['kappa'])}, n = "
                        f"{float(a.get('$spin_rate$', float('nan'))):.6g} rpm", "ok") if not info["ramps"] else
                       (f"{len(info['cases'])} cases ({info['ramps']} ramps), kappa of the constant cases "
                        f"{_krange(info['kappa'])}, n = {float(a.get('$spin_rate$', float('nan'))):.6g} rpm", "ok"))
            out += [(f"  ramp {t}", None) for t in info["ramp_text"][:8]]
            if info["ramps"] > 8:
                out.append((f"  … {info['ramps'] - 8} more ramps", None))
            out.append((f"simulated model: case {a.get('sim_case', '?')}, SLD model {a.get('sim_model', '?')}"
                        + ("" if a.get("sim_model") else "  (unknown: Standardize an .h5… or the run's 'model')"),
                        None if a.get("sim_model") else "warn"))
            out.append((f"signals {', '.join(info['signals']) or '-'}; {info['duration']:.2f} s per case" if
                        info["duration"] else f"signals {', '.join(info['signals']) or '-'}", None))
            disc = ", ".join(f"{k.strip('$')} {a[k]:g}" for k in DISCRETISATION if k in a)
            if disc:
                out.append((disc, None))
        return out
    if key in ("label_template", "label_build"):
        binfo, yinfo = label_info(exp.label["out"]), yaml_label_info(exp.label["labels_yaml"])
        built = {c: (d["label"], d["kappa"]) for c, d in binfo.items()}
        yml = {c: d["label"] for c, d in yinfo.items()}
        data = h5_info(exp.data_h5) if exp.data_h5 else None
        ramp_set = set((data or {}).get("ramp_cases", [])) | {c for c, d in binfo.items() if d["ramp"]}
        src = built if key == "label_build" else {c: (lab, built.get(c, (None, float("nan")))[1]) for c, lab in yml.items()}
        if not src:
            return [("nothing yet", None)]
        ivs = {c: (binfo[c]["intervals"] if key == "label_build" else yinfo[c]["intervals"]) for c in src}
        ramps = sorted(c for c in src if c in ramp_set)
        src = {c: v for c, v in src.items() if c not in ramp_set}   # constants: counts, kappa, boundary
        by = {}
        for c, (lab, k) in src.items():
            by.setdefault(lab, []).append(k)
        if ramps:   # the ramps apart: where the ground truth changes along the cut
            ons = [t_onset(ivs[c]) for c in ramps]
            ons = [t for t in ons if t is not None]
            out.append((f"ramps {len(ramps)}" + (f": crosses at t ≈ {', '.join(f'{t:.2f}' for t in ons[:6])} s"
                                                 if ons else ": none crosses into unstable"), "ok"))
            out += [(f"  ramp {c}: {transitions_text(ivs[c])}", None) for c in ramps[:8]]
            if len(ramps) > 8:
                out.append((f"  … {len(ramps) - 8} more ramps", None))
            p = exp.label
            if all(k in p for k in WINDOW_KEYS):
                out.append((f"  ramps labelled window by window: {p['window_N']:g} x "
                            f"{'T_rev' if p['window_mode'] == 'by_revolution' else 'T_modal'}, step {p['window_step']:g}"
                            + (f" (window of {p['window_from']})" if p.get("window_from") else ""), None))
            if not src:
                out.append(("(no constant cases)", None))
        order = [lab for lab in ("stable", "gray", "unstable") if lab in by] + [l for l in by if l not in ("stable", "gray", "unstable")]
        if order:
            out.append(("  ·  ".join(f"{lab} {len(by[lab])}" for lab in order)
                        + ("   (constant cases)" if ramps else ""), "ok"))
        for lab in order:
            ks = [k for k in by[lab] if k == k]
            out.append((f"  {lab:<8s} kappa {_krange(ks)}", None))
        st, un = [k for k in by.get("stable", []) if k == k], [k for k in by.get("unstable", []) if k == k]
        if st and un:
            gap = f"boundary between kappa {max(st):g} (last stable) and {min(un):g} (first unstable)"
            out.append((gap, "warn" if max(st) > min(un) else None))
        if built and yml:
            diff = sorted(c for c in yml if c in built and yml[c] != built[c][0])
            if diff:
                out.append((f"labels YAML and dataset disagree on {len(diff)} cases (e.g. {', '.join(diff[:3])}): "
                            f"the dataset was not built from this YAML", "bad"))
        if key == "label_template" and exp.label.get("strategy"):
            p = exp.label
            out.append((f"strategy {p.get('strategy')}: stable < {p.get('lim_inf_pct', '?')} %, unstable > "
                        f"{p.get('lim_sup_pct', '?')} % of {p.get('base_attr', '?')} x {p.get('base_scale', '?')} "
                        f"({p.get('amp_signal', '?')})", None))
        return out
    if key in ("indicators", "noise_indicators"):
        import h5py
        path = S[key].outputs[0]
        if not os.path.isfile(path):
            return [("no results yet", None)]
        truth = {c: lab for c, (lab, _) in _label_cases(exp.label["out"]).items()} if key == "indicators" else {}
        with h5py.File(path, "r") as f:
            groups = [g for g in f if isinstance(f[g], h5py.Group) and g not in ("summary", "metrics", "training")]
            variants = sorted({v for g in groups for v in f[g] if isinstance(f[g][v], h5py.Group) and "t" in f[g][v]})
            out.append((f"{len(variants)} variants x {len(groups)} {'cases' if key == 'indicators' else 'groups'}", "ok"))
            for v in variants:
                done = [g for g in groups if v in f[g]]
                errs = [g for g in done if "meta_error" in f[g][v].attrs]
                flag = [g for g in done if "t_d" in f[g][v] and f[g][v]["t_d"].size > 0]
                txt = f"  {v}: {len(done)}/{len(groups)} done"
                if errs:
                    txt += f", {len(errs)} errors"
                if truth:
                    fs = sum(truth.get(g) == "stable" for g in flag)
                    fu = sum(truth.get(g) == "unstable" for g in flag)
                    ns, nu = sum(t == "stable" for t in truth.values()), sum(t == "unstable" for t in truth.values())
                    txt += f"; flags {fu}/{nu} unstable, {fs}/{ns} stable"
                    nm = sum(t == "mixed" for t in truth.values())
                    if nm:
                        txt += f", {sum(truth.get(g) == 'mixed' for g in flag)}/{nm} ramps that cross"
                else:
                    txt += f"; flags {len(flag)}"
                out.append((txt, "bad" if errs else None))
        if truth:
            out.append(("(flags = cases with at least one detection; labels of this experiment's dataset)", None))
        return out
    if key == "validate":
        m = validation_metrics(exp)
        if not m:
            return [("no validation results yet", None)]
        f2 = lambda x: "-" if x is None or x != x else f"{x:.2f}"
        rank = sorted(m, key=lambda r: -(m[r].get("balanced_accuracy") or -1))
        for i, r in enumerate(rank, 1):
            d = m[r]
            out.append((f"{i}. {r}: bal.acc {f2(d.get('balanced_accuracy'))}  MCC {f2(d.get('MCC'))}  "
                        f"AUC {f2(d.get('AUC'))}  TP {d.get('TP')} FN {d.get('FN')} TN {d.get('TN')} FP {d.get('FP')}"
                        + (f"  (early alarms {d['n_early_alarm']}, anticipated {d['n_anticipated']})"
                           if d.get("n_early_alarm") or d.get("n_anticipated") else ""),
                        "ok" if i == 1 else None))
        ramps = [r for r in rank if m[r].get("ramp_n")]
        if ramps:
            out.append((f"ramps that cross ({m[ramps[0]]['ramp_n']}, apart from the ranking): detected / anticipated / "
                        f"early alarm / missed, median delay to the crossing of the truth", None))
            for r in ramps:
                d = m[r]
                out.append((f"  {r}: {f2(d.get('ramp_detection_rate'))} / {f2(d.get('ramp_anticipated_rate'))} / "
                            f"{f2(d.get('ramp_early_alarm_rate'))} / {f2(d.get('ramp_miss_rate'))}, delay "
                            + ("-" if d.get("ramp_median_delay_s") is None or d["ramp_median_delay_s"] != d["ramp_median_delay_s"]
                               else f"{d['ramp_median_delay_s']:+.3f} s"), None))
        tol = exp.section("validate").get("early_tol_s", EARLY_TOL_S)
        out.append((f"rule: a first detection more than {tol:g} s before the onset of the truth is an early alarm "
                    f"(counted as FN); within {tol:g} s, an anticipated hit", None))
        return out
    if key == "noise":
        info = h5_info(S[key].outputs[0])
        if not info:
            return [("no noise file yet", None)]
        snr = [g for g in info["groups"] if g.startswith("snr_")]
        return [(f"control + {len(snr)} SNR levels" + (f" ({snr[0][4:]} ... {snr[-1][4:]} dB)" if snr else ""), "ok")]
    if key == "model_snr":
        p = S[key].outputs[0]
        info = h5_info(p) if os.path.isfile(p) else None
        return [(f"{len(info['cases'])} cases with model SNR", "ok")] if info else [("no results yet", None)]
    return out


def stage_badge(exp: Exp, key: str) -> str:
    """Short text for the diagram box (what the output holds, or the progress)."""
    st = status(exp).get(key, ("", ""))[0]
    if st == "running":
        pr = stage_progress(exp, key)
        return f"{100 * pr[0] / pr[1]:.0f} %" if pr and pr[1] else "running…"
    if st not in ("done", "stale"):
        return ""
    try:
        if key == "simulate":
            n = sum(len((h5_info(r.h5) or {}).get("cases", [])) if r.imported else r.simulated() for r in exp.runs)
            return f"{n} cases"
        if key in ("extract", "merge"):
            info = h5_info(exp.data_h5)
            return f"{len(info['cases'])} cases" if info else ""
        if key == "label_build":
            by = {}
            for lab, _ in _label_cases(exp.label["out"]).values():
                by[lab] = by.get(lab, 0) + 1
            return " · ".join(f"{lab[0].upper()} {by[lab]}" for lab in ("stable", "gray", "unstable", "mixed")
                              if lab in by)
        if key == "indicators":
            return f"{len(exp.indicators['variants'])} variants"
        if key == "validate":
            m = validation_metrics(exp)
            best = max((d.get("balanced_accuracy") or 0 for d in m.values()), default=None)
            return f"best bal.acc {best:.2f}" if best is not None else ""
    except Exception:
        return ""
    return ""


# ============================================================================== editing (used by the app)
CONFIGS_DIR = os.path.join(SIM, "configs")
_PREFIX = {("MaxEnt_SPRT", "Default"): "maxent", ("RMS_CV", "Default"): "rms_cv", ("SST_SVD", "Default"): "ssq",
           ("Green_Integral", "Default"): "green_default", ("Green_Integral", "Lyapunov"): "green_fixed"}


def own_yaml(name: str) -> dict:
    """The experiment file as written (without extends resolved)."""
    return yaml_load(exp_path(name))


def save_section(name: str, section: str, data) -> None:
    """Replace one section of the experiment's own YAML (None removes it)."""
    d = own_yaml(name)
    if data is None:
        d.pop(section, None)
    else:
        d[section] = data
    yaml_save(d, exp_path(name))
    reload()


def propose_variant_name(spec: dict, taken=()) -> str:
    """Same pattern as doe_indicators._run_name, made unique against `taken` (the other rows of the table)."""
    pp, mode = spec["params_physical"], spec["mode"]
    u = "rev" if mode == "by_revolution" else "modal"
    ind = _PREFIX.get((spec["indicator"], spec.get("func", "Default")), spec["indicator"].lower())
    win, step = int(float(pp.get(f"N_{u}_window", 0))), int(float(pp.get(f"step_{u}", 1)))
    short = "revo" if mode == "by_revolution" else "modal"
    if ind in ("rms_cv", "ssq"):
        n_aux = int(float(pp.get(f"n_max_{u}") or pp.get(f"Ai_length_{u}") or 0))
        base = f"{ind}_{short}_aux{win}_n_aux{n_aux}_dec{win + (n_aux - 1) * step}_{step}step"
    else:
        base = f"{ind}_{short}_dec{win}_{step}step"
    names = set(taken)
    name, i = base, 2
    while name in names:
        name, i = f"{base}_v{i}", i + 1
    return name


FLOW_PREFIX = {"Simulation only": "sim", "Labelled dataset (training)": "train", "Indicators": "ind",
               "Validation against a reference": "val"}


def propose_name(flow: str, case: str, spin, kappas=None) -> str:
    """<sim|train|ind|val|exp>_<machine>_n<rpm>_k<kappa or min-max>, unique among the experiments."""
    machine = "".join(ch for ch in (case or "doe") if ch.isalnum()).replace("Hz", "")
    parts = [FLOW_PREFIX.get(flow, "exp"), machine]
    if spin:
        parts.append(f"n{float(spin):.0f}")
    if kappas:
        lo, hi = round(min(kappas), 2), round(max(kappas), 2)
        parts.append(f"k{lo:g}" if lo == hi else f"k{lo:g}-{hi:g}")
    base, existing = "_".join(parts), set(list_experiments())
    name, i = base, 2
    while name in existing:
        name, i = f"{base}_{i}", i + 1
    return name


def list_configs() -> list:
    return sorted(os.path.splitext(f)[0] for f in os.listdir(CONFIGS_DIR)
                  if f.endswith(".yaml") and f != "base.yaml") if os.path.isdir(CONFIGS_DIR) else []


def _sld():
    if PLOTS not in sys.path:
        sys.path.insert(0, PLOTS)
    import sld_model
    return sld_model


def sld_models() -> list:
    try:
        return list(_sld().MODELS)
    except Exception:   # sld_tools missing: the model modes are just not offered
        return []


def ap_ref_value(ap_ref, spin=None):
    """Reference depth [m] of an ap_ref section at this n (None for mode none). Raises, with what to do, when it
    is not a finite positive number (e.g. n in a pocket between lobes of the SLD: infinite limit)."""
    import math
    a = ap_ref or {}
    mode = a.get("mode") or "none"
    if mode == "none":
        return None
    if mode == "manual":
        v = float(a.get("manual") or 0)
        if not (math.isfinite(v) and v > 0):
            raise ValueError(f"ap_ref manual must be a depth > 0 in metres (got {a.get('manual')!r})")
        return v
    if mode == "model":
        v = _sld().ap_crit(a.get("model")) * 1e-3
    elif mode == "model_at_spin":
        if spin is None:
            raise ValueError("ap_ref model_at_spin needs n ($spin_rate$)")
        v = _sld().ap_lim(a.get("model"), float(spin)) * 1e-3
    else:
        raise ValueError(f"ap_ref mode {mode!r}: use none, manual, model or model_at_spin")
    if not (math.isfinite(v) and v > 0):
        raise ValueError(f"the SLD of model '{a.get('model')}' has no finite stability limit at n = {spin} rpm "
                         f"(a pocket between lobes): choose another n, or ap_ref 'model' / 'manual'")
    return v


ROW_EXTRA = ("case", "Ap_start", "Ap_end", "spin_rate", "kappa", "kappa_error", "kappa_start", "kappa_end", "ramp")


def case_rows(cfg: dict) -> list:
    """One dict per case of a doe_runner config: index, Ap [m] (start/end), n, kappa (None if not computable) and
    every other variable; 'kappa_error' says why kappa is missing. A ramp (Ap_end != Ap_start) has ramp True,
    kappa_start / kappa_end and kappa None."""
    lst, val = cfg_table(cfg)
    rows = []
    for i, r in enumerate(val):
        d = {"case": i, **{k.strip("$"): v for k, v in zip(lst, r)}}
        ap, ap1 = d.get("Ap_start"), d.get("Ap_end")
        d["ramp"] = is_ramp(d)
        try:
            ref = ap_ref_value(cfg.get("ap_ref"), d.get("spin_rate"))
            k = None if ref is None or ap is None else ap / ref
            if d["ramp"]:
                d["kappa"], d["kappa_start"], d["kappa_end"] = None, k, (None if ref is None else ap1 / ref)
            else:
                d["kappa"] = k
        except Exception as exc:
            d["kappa"], d["kappa_error"] = None, str(exc)
        rows.append(d)
    return rows


RAMP_MARK = "# ramps: both directions"   # in the case's *db_def.py: its workpiece also makes Ap decrease


def ramp_template_ok(base_dir: str, case: str):
    """True / False when the case's workpiece (in/*db_def.py) can / cannot make Ap DECREASE along the cut (the
    1DOF_150Hz template only grows a cone: a decreasing ramp there is simulated as a constant Ap_end); None when
    the file cannot be read."""
    import glob
    files = glob.glob(os.path.join(base_dir or "", case or "", "in", "*db_def.py"))
    if not files:
        return None
    try:
        with open(files[0], encoding="utf-8", errors="replace") as f:
            return RAMP_MARK in f.read()
    except OSError:
        return None


def sim_problems(cfg: dict) -> tuple:
    """(errors, warnings) of the cases of a loaded doe_runner config: Ap finite and > 0, units, duplicates,
    ramps the case cannot simulate (decreasing Ap on a one-way workpiece) and ramps that do not cross kappa = 1."""
    import math
    errs, warns = [], []
    rows = case_rows(cfg)
    if not rows:
        errs.append("no cases")
    for d in rows:
        for k in ("Ap_start", "Ap_end"):
            v = d.get(k)
            if v is not None and not (math.isfinite(v) and v > 0):
                errs.append(f"case {d['case']}: {k} = {v} (must be a finite depth > 0; an infinite value comes "
                            f"from kappa x an SLD limit that is infinite at that n)")
            elif v is not None and v > 0.1:
                warns.append(f"case {d['case']}: {k} = {v:g} m (Ap is in metres: is it mm by mistake?)")
        if d.get("kappa_error") and (cfg.get("ap_ref") or {}).get("mode") not in (None, "none"):
            warns.append(f"case {d['case']}: no kappa ({d['kappa_error']})")
        k0, k1 = d.get("kappa_start"), d.get("kappa_end")
        if d.get("ramp") and k0 is not None and k1 is not None and (min(k0, k1) >= 1 or max(k0, k1) < 1):
            warns.append(f"case {d['case']}: ramp kappa {k0:.3f} -> {k1:.3f} does not cross kappa = 1 (fine; it is "
                         f"scored with the constant cases if its labels do not change along the cut)")
    down = [d["case"] for d in rows if d.get("ramp") and d["Ap_end"] < d["Ap_start"]]
    if down and ramp_template_ok(cfg.get("base_dir"), cfg.get("case")) is False:
        errs.append(f"cases {down[:6]}: Ap decreases along the cut, but the workpiece of case '{cfg.get('case')}' "
                    f"(in/*db_def.py) only makes Ap grow: it would be simulated as a constant Ap_end. Use a case "
                    f"whose db_def supports both directions (marked '{RAMP_MARK}', e.g. "
                    f"Data/1DOF_150_Ramp_check/1DOF_150Hz)")
    seen = {}
    for d in rows:
        key = tuple(sorted((k, v) for k, v in d.items() if k not in ("case", "kappa", "kappa_error", "kappa_start",
                                                                       "kappa_end", "ramp")))
        if key in seen:
            warns.append(f"cases {seen[key]} and {d['case']} are identical")
        seen.setdefault(key, d["case"])
    return errs, warns


def _values(text) -> list:
    """'0.5, 0.6' / '0.5 0.6' / 'start:stop:step' (stop included) / a number / a list -> floats."""
    if isinstance(text, (int, float)):
        return [float(text)]
    if isinstance(text, (list, tuple)):
        return [float(x) for x in text]
    text = str(text).strip()
    if not text:
        return []
    if ":" in text:
        a, b, s = (float(x) for x in text.split(":"))
        if s <= 0 or b < a:
            raise ValueError(f"'{text}': start:stop:step needs step > 0 and stop >= start")
        return [round(a + i * s, 10) for i in range(int(round((b - a) / s)) + 1)]
    return [float(x) for x in text.replace(";", ",").replace(" ", ",").split(",") if x]


def build_simulation(base_dir: str, case: str, doe_name: str, depths, depth_unit: str = "mm", spins=(),
                     ap_ref=None, variables=None, combine: bool = False, nb_proc: int = 1, n2m_bat: str = "",
                     extract_signals=None, force_signal: str = "res_R_p", depths_end=None) -> dict:
    """Explicit doe_runner configuration (mode sweep: one row per case, nothing inherited).
    depths: Ap values in mm (depth_unit 'mm') or kappa values (depth_unit 'kappa': Ap = kappa x ap_ref at the n
    of the case). depths_end: empty = every case at a constant Ap; else the Ap (or kappa) at the end of the cut,
    one per depth (a single value repeats): a row whose end differs from its start is a ramp, an equal one stays
    constant. spins: n [rpm]. variables: {'$f_tooth$': values, ...}. combine=True: every combination
    (factorial; each depth keeps its own end), else row by row (lists of the same length; a single value
    repeats). Raises with every problem."""
    import itertools
    errs = []
    depths, spins = _values(depths), _values(spins)
    ends = _values(depths_end) if depths_end is not None else []
    if ends and len(ends) not in (1, len(depths)):
        errs.append(f"depths end: one value per depth ({len(depths)}) or a single one, got {len(ends)}")
    pairs = list(zip(depths, ends if len(ends) > 1 else ends * len(depths))) if ends else [(d, d) for d in depths]
    var = {k if k.startswith("$") else f"${k}$": _values(v) for k, v in (variables or {}).items()}
    var = {k: v for k, v in var.items() if v}
    if not depths:
        errs.append("give at least one Ap (or kappa)")
    if not spins:
        errs.append("give n [rpm]")
    if not doe_name or any(c in doe_name for c in "/\\:"):
        errs.append("doe_name: a folder name, without path separators")
    if not base_dir or not os.path.isdir(base_dir):
        errs.append(f"base_dir: folder not found ({base_dir})")
    elif case and not os.path.isdir(os.path.join(base_dir, case)):
        errs.append(f"case: no sub-folder '{case}' in {base_dir}")
    if errs:
        raise ValueError("\n".join(errs))
    cols = [pairs, spins, *var.values()]
    if combine:
        rows = [list(r) for r in itertools.product(*cols)]
    else:
        n = max(len(c) for c in cols)
        if any(len(c) not in (1, n) for c in cols):
            raise ValueError(f"row by row: every list needs {n} values or a single one "
                             f"(lengths {[len(c) for c in cols]}); or tick 'every combination'")
        rows = [[c[i] if len(c) > 1 else c[0] for c in cols] for i in range(n)]
    aps, aps_end = [], []
    for r in rows:
        if depth_unit == "kappa":
            ref = ap_ref_value(ap_ref, r[1])
            if ref is None:
                raise ValueError("kappa needs an ap_ref (manual, model or model_at_spin)")
            scale = ref
        else:
            scale = 1e-3
        aps.append(round(r[0][0] * scale, 9))
        aps_end.append(round(r[0][1] * scale, 9))
    sweep = {"$Ap_start$": aps, "$Ap_end$": aps_end, "$spin_rate$": [r[1] for r in rows]}
    for j, k in enumerate(var):
        sweep[k] = [r[2 + j] for r in rows]
    sim = {"base_dir": os.path.normpath(base_dir).replace("\\", "/"), "case": case, "n2m_bat": n2m_bat or None,
           "doe_name": doe_name, "nb_proc": int(nb_proc), "mode": "sweep", "sweep": sweep,
           "ap_ref": dict(ap_ref or {"mode": "none"}),
           "extract_signals": list(extract_signals or ["Axial_disp", "Axial_vel", "Axial_acc"]),
           "force_signal": force_signal or "res_R_p"}
    sim = {k: v for k, v in sim.items() if v is not None}
    e, _ = sim_problems(_doe_runner().load_config(_tmp_config(sim)))
    if e:
        raise ValueError("\n".join(e))
    return sim


def _tmp_config(sim: dict) -> str:
    """sim written to a temp file doe_runner can load (validation of a simulation that is not saved yet)."""
    import yaml
    p = os.path.join(tempfile.gettempdir(), "experiments_app_check.yaml")
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(sim, f, sort_keys=False)
    return p


def check_simulation(sim: dict) -> tuple:
    """(rows, errors, warnings) of an explicit simulation dict, without saving it (the 'dry-run' of the form)."""
    try:
        cfg = _doe_runner().load_config(_tmp_config(sim))
    except Exception as exc:
        return [], [str(exc)], []
    e, w = sim_problems(cfg)
    d = os.path.join(cfg["base_dir"], cfg["doe_name"])
    if os.path.isdir(d):
        w.append(f"the DOE folder {d} already exists: Simulate replaces it")
    return case_rows(cfg), e, w


def explicit_simulation(cfg: dict) -> dict:
    """Explicit form of a loaded doe_runner config (base.yaml merged in): every key written, sweep lists full."""
    out = {}
    for k in SIM_KEYS:
        if k in cfg:
            out[k] = cfg[k]
    if "base_dir" in out:
        out["base_dir"] = os.path.normpath(out["base_dir"]).replace("\\", "/")
    out.setdefault("ap_ref", {"mode": "none"})
    return out


def base_yaml() -> dict:
    """configs/base.yaml as written (not validated: its base_dir may point to a folder that no longer exists)."""
    p = os.path.join(CONFIGS_DIR, "base.yaml")
    return yaml_load(p) if os.path.isfile(p) else {}


def simulation_form(sim: dict) -> dict:
    """Form values (depths in mm, n, other variables, row by row) that rebuild exactly this simulation."""
    cfg = _doe_runner().load_config(_tmp_config(sim))
    lst, val = cfg_table(cfg)
    cols = {k: [r[i] for r in val] for i, k in enumerate(lst)}
    one = lambda v: v[:1] if len(set(v)) == 1 else v   # noqa: E731
    out = {"base_dir": cfg["base_dir"].replace("\\", "/"), "case": cfg.get("case", ""),
           "doe_name": cfg["doe_name"], "nb_proc": cfg.get("nb_proc", 1), "n2m_bat": cfg.get("n2m_bat") or base_yaml().get("n2m_bat", ""),
           "ap_ref": cfg.get("ap_ref") or {"mode": "none"},
           "extract_signals": cfg.get("extract_signals") or ["Axial_disp", "Axial_vel", "Axial_acc"],
           "force_signal": cfg.get("force_signal", "res_R_p"),
           "depths": [round(a * 1e3, 9) for a in cols.pop("$Ap_start$", [])],
           "spins": one(cols.pop("$spin_rate$", []))}
    ends = cols.pop("$Ap_end$", None)
    ends = [round(a * 1e3, 9) for a in ends] if ends is not None else []
    out["depths_end"] = ends if any(abs(a - b) > RAMP_TOL * 1e3 for a, b in zip(out["depths"], ends)) else []
    out["variables"] = {k: one(v) for k, v in cols.items()}
    return out


# what a new experiment starts with (same values as the current training dataset)
LABEL_DEFAULTS = {"strategy": "amplitude", "amp_signal": "Axial_disp", "base_attr": "$f_tooth$", "base_scale": 1.0e-3,
                  "lim_inf_pct": 10.0, "lim_sup_pct": 40.0, "warmup": 0.0}
INDICATOR_PRESETS_DEFAULT = ("maxent_revo_dec7_1step", "rms_cv_revo_aux4_n_aux4_dec7_1step",
                             "ssq_revo_aux4_n_aux4_dec7_1step", "green_fixed_revo_dec7_1step")
METRIC_COLUMNS = ("balanced_accuracy", "MCC", "AUC", "TPR", "TNR", "F1", "accuracy", "median_delay_onset_s",
                  "mean_alarm_fraction_stable", "mean_persistence", "early_alarm_rate", "ramp_n", "ramp_detection_rate",
                  "ramp_anticipated_rate", "ramp_early_alarm_rate", "ramp_miss_rate", "ramp_median_delay_s")
EARLY_TOL_S = 0.5   # [s] validate: a first detection up to this before the onset of the truth is an anticipated hit


def label_defaults(indicators_section=None) -> dict:
    """label section of a new experiment: LABEL_DEFAULTS + the labelling window of the ramps, written explicitly
    (the window of the indicator variants of the section given, else of the default presets)."""
    sec = indicators_section or default_indicators()
    v = sec.get("variants")
    specs = v if isinstance(v, dict) else ({n: presets()[n] for n in v if n in presets()} if isinstance(v, list) else {})
    return {**LABEL_DEFAULTS, **default_window(specs, None, sec.get("f_modal"))}


def default_indicators() -> dict:
    lib = presets()
    return {"f_modal": 150.0, "cases": "all", "workers": 6,
            "variants": {n: lib[n] for n in INDICATOR_PRESETS_DEFAULT if n in lib}}


def indicators_for(reference: str | None) -> dict:
    """indicators section of a new experiment: the reference's variants when it has some, else the defaults."""
    try:
        has = bool(load(reference).indicators["specs"]) if reference else False
    except Exception:
        has = False
    return {"variants": "inherit"} if has else default_indicators()


def create_experiment(name: str, runs: list, stages_on=None, reference: str | None = None, description: str = "",
                      out_dir: str | None = None, **sections) -> str:
    """Write experiments/<name>.yaml (sections written explicitly; label/indicators defaults when their stages
    are on and nothing is given)."""
    name = name.strip()
    if not name or any(c in name for c in '/\\:*?"<>|'):
        raise ValueError("experiment name: letters, digits, _ - . only")
    path = exp_path(name)
    if os.path.exists(path):
        raise ValueError(f"experiment '{name}' already exists")
    st = [k for k in STAGES if k in (stages_on or FLOWS["Labelled dataset (training)"])]
    d = {"name": name, "description": description, "stages": st}
    if reference:
        d["reference"] = reference
    if out_dir:
        d["out_dir"] = os.path.normpath(out_dir).replace("\\", "/")
    d["runs"] = runs
    if "indicators" in st and "indicators" not in sections:
        sections["indicators"] = indicators_for(reference)
    if "label_template" in st and not reference and "label" not in sections:
        sections["label"] = label_defaults(sections.get("indicators"))
    if "validate" in st and "validate" not in sections:
        sections["validate"] = {"channel": "Axial_disp", "early_tol_s": EARLY_TOL_S}
    d.update({k: v for k, v in sections.items() if v})
    yaml_save(d, path)
    reload()
    return path


def resolved_yaml(name: str) -> dict:
    """The experiment as one explicit file: extends merged, config runs written in full, variants as a table."""
    e = load(name)
    d = {k: v for k, v in load_raw(name).items() if k not in ("kind", "training")}
    if e.ref is not None:
        d["reference"] = e.ref.name
    d["stages"] = [k for k in e.enabled if k != "merge"]   # merge is automatic with several runs
    d["runs"] = [({"simulation": explicit_simulation(r.cfg), **({"existing": True} if r.existing else {}),
                   **({"model": r.entry["model"]} if r.entry.get("model") else {})}
                  if r.cfg else dict(r.entry)) for r in e.runs]
    if "indicators" in d or e.indicators["specs"]:
        ind = dict(d.get("indicators") or {})
        if ind.get("variants") != "inherit":
            ind["variants"] = dict(e.indicators["specs"])
        d["indicators"] = ind
    order = ["name", "description", "stages", "reference", "out_dir", "runs", "merge", "simulate", "label",
             "indicators", "validate"]
    return {**{k: d[k] for k in order if k in d}, **{k: v for k, v in d.items() if k not in order}}


def make_explicit(name: str) -> None:
    """Rewrite an experiment as one explicit file (see resolved_yaml). Outputs and run records do not move."""
    d = resolved_yaml(name)
    yaml_save(d, exp_path(name))
    reload()


def copy_experiment(src: str, name: str, description: str = "", doe_suffix: str = "") -> str:
    """Explicit copy under another name (its outputs go to its own folder). doe_suffix renames the DOE folder of
    every simulated run so the copy can be simulated with other values without touching the original's data."""
    if os.path.exists(exp_path(name)):
        raise ValueError(f"experiment '{name}' already exists")
    d = resolved_yaml(src)
    d["name"] = name
    d["description"] = description or f"copy of {src}"
    for k in ("label", "indicators", "validate", "noise_indicators"):   # own outputs: never the source's files
        if isinstance(d.get(k), dict):
            d[k] = {a: b for a, b in d[k].items() if a not in ("out", "labels_yaml")}
    if doe_suffix:
        for r in d["runs"]:
            if "simulation" in r:
                r["simulation"]["doe_name"] += doe_suffix
                r.pop("existing", None)   # another folder: the copy is simulated by the app
    d.pop("out_dir", None)
    yaml_save(d, exp_path(name))
    reload()
    return exp_path(name)


def set_simulation(name: str, i: int, sim: dict) -> None:
    """Replace run i of the experiment by an explicit simulation (append it when i is past the end)."""
    d = own_yaml(name)
    runs = list(d.get("runs") or [])
    if i >= len(runs):
        runs.append({"simulation": sim})
    else:   # a run simulated outside the app keeps its flag (Simulate stays blocked) and its model
        runs[i] = {"simulation": sim, **{k: runs[i][k] for k in ("existing", "model") if runs[i].get(k)}}
    d["runs"] = runs
    yaml_save(d, exp_path(name))
    reload()


def set_run_model(name: str, h5: str, model) -> bool:
    """Write 'model' (SLD preset of the simulated machine; None removes it) in the run(s) of the experiment whose
    signals file is h5. True if something changed."""
    e, d = load(name), own_yaml(name)
    runs, changed = list(d.get("runs") or []), False
    for i, r in enumerate(e.runs):
        if i < len(runs) and _norm(r.h5) == _norm(h5) and runs[i].get("model") != model:
            if model:
                runs[i]["model"] = model
            else:
                runs[i].pop("model", None)
            changed = True
    if changed:
        d["runs"] = runs
        yaml_save(d, exp_path(name))
        reload()
    return changed


def stamp_model(exp: Exp) -> None:
    """After Extract: write each run's model (sim_model) into its doe_results.h5, so the model saved in the
    experiment survives a new extraction (sim_case is written by doe_runner extract itself)."""
    for r in exp.runs:
        if os.path.isfile(r.h5) and r.model:
            add_case_attrs(r.h5, {"sim_model": r.model})


def validation_metrics(exp: Exp) -> dict:
    """{variant: {metric: value}} from the /metrics group of the experiment's doe_validation_results.h5."""
    import h5py
    path = exp.out("validate", "out", "doe_validation_results.h5")
    if not os.path.isfile(path):
        return {}
    with h5py.File(path, "r") as f:
        if "metrics" not in f:
            return {}
        return {run: {k: (v.item() if hasattr(v, "item") else v) for k, v in g.attrs.items()}
                for run, g in f["metrics"].items()}


def dependents(name: str) -> list:
    """Experiments that extend this one or use it as reference."""
    out = []
    for n in list_experiments():
        if n == name:
            continue
        try:
            d = own_yaml(n)
        except Exception:
            continue
        if name in (d.get("extends"), d.get("training"), d.get("reference")):
            out.append(n)
    return out


def delete(name: str) -> None:
    """Remove the experiment YAML and its run records (.runs/<name>: logs, records, generated simulation files).
    Never touches data (.h5, DOE folders, configs)."""
    deps = dependents(name)
    if deps:
        raise ValueError(f"'{name}' is used by {deps}: delete or change those first")
    os.remove(exp_path(name))
    shutil.rmtree(os.path.join(RUNS_DIR, name), ignore_errors=True)
    reload()


def orphan_runs() -> list:
    """Folders of .runs/ without an experiment file (left when a YAML is renamed or removed by hand)."""
    have = set(list_experiments())
    return sorted(d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d)) and d not in have) \
        if os.path.isdir(RUNS_DIR) else []


# ============================================================================== import
_LABEL_ATTRS = {"labeling_strategy": "strategy", "labeling_signal": "amp_signal", "labeling_base_attr": "base_attr",
                "labeling_base_scale": "base_scale", "labeling_lim_inf_pct": "lim_inf_pct",
                "labeling_lim_sup_pct": "lim_sup_pct", "labeling_warmup": "warmup",
                "labeling_window_mode": "window_mode", "labeling_window_N": "window_N",
                "labeling_window_step": "window_step", "labeling_f_modal": "f_modal"}


def _label_attrs(path: str) -> dict:
    """labeling_* attrs of the first piece of a reference_dataset*.h5 -> label params."""
    import h5py
    with h5py.File(path, "r") as f:
        for lab in f:
            for case in f[lab].values():
                for piece in case.values():
                    a = piece.attrs
                    return {v: (a[k].item() if hasattr(a[k], "item") else a[k]) for k, v in _LABEL_ATTRS.items()
                            if k in a}
    return {}


def simulated_cases(doe_dir: str) -> tuple:
    """(case, [(index, var_val)]) of the simulated cases of a DOE folder (index folders with sens_out.hdf5 and
    var_val.py), sorted by index."""
    dr, case = _doe_runner(), detect_case(doe_dir)
    out = []
    for i in sorted((i for i in os.listdir(doe_dir) if i.isdigit()), key=int) if os.path.isdir(doe_dir) else []:
        c = os.path.join(doe_dir, i, case)
        if os.path.isfile(os.path.join(c, "sens_out.hdf5")) and os.path.isfile(os.path.join(c, "var_val.py")):
            out.append((int(i), dr.read_var_val(os.path.join(c, "var_val.py"))))
    return case, out


def simulation_from_folder(doe_dir: str, ap_ref=None) -> dict:
    """Explicit simulation of a DOE folder simulated outside the app, from the var_val.py of its cases: what
    doe_runner extract needs (folder, case, ap_ref for kappa, signals). Raises if it cannot be rebuilt."""
    case, cases = simulated_cases(doe_dir)
    if not cases:
        raise ValueError(f"{doe_dir}: no simulated case (index folders with {case}/sens_out.hdf5 and var_val.py)")
    keys = list(cases[0][1])
    bad = [i for i, v in cases if list(v) != keys]
    if bad:
        raise ValueError(f"cases {bad[:5]} have other variables than case {cases[0][0]}: {keys}")
    if [i for i, _ in cases] != list(range(len(cases))):
        raise ValueError(f"case folders are not 0..{len(cases) - 1} (some missing or not simulated): "
                         f"{[i for i, _ in cases][:10]}")
    base = os.path.dirname(os.path.normpath(doe_dir))
    if not os.path.isdir(os.path.join(base, case)):
        raise ValueError(f"doe_runner extract needs the case folder {os.path.join(base, case)} next to the DOE folder")
    sim = {"base_dir": base.replace("\\", "/"), "case": case, "doe_name": os.path.basename(os.path.normpath(doe_dir)),
           "nb_proc": 1, "mode": "sweep", "sweep": {k: [float(v[k]) for _, v in cases] for k in keys},
           "ap_ref": dict(ap_ref or {"mode": "none"}), "extract_signals": ["Axial_disp", "Axial_vel", "Axial_acc"],
           "force_signal": "res_R_p"}
    _doe_runner().load_config(_tmp_config(sim))   # same validation as any simulation
    return sim


def ap_ref_of_h5(path: str):
    """ap_ref section written by doe_runner extract in a doe_results.h5 (root attrs), or None."""
    import h5py
    try:
        with h5py.File(path, "r") as f:
            a = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
    except OSError:
        return None
    mode = str(a.get("ap_ref_mode", "none"))
    if mode in ("model", "model_at_spin") and a.get("ap_ref_model"):
        return {"mode": mode, "model": str(a["ap_ref_model"])}
    if mode == "manual" and a.get("ap_ref_m"):
        return {"mode": "manual", "manual": float(a["ap_ref_m"])}
    return {"mode": "none"}


def simulation_of_run(r: Run) -> dict:
    """Explicit simulation of any run, also of an imported folder (for 'load values from' and the planner):
    its config, else the doe_config.yaml doe_runner left in the folder, else rebuilt from the var_val.py of the
    cases, else from the case attributes of its .h5 (ap_ref from the .h5 root attributes)."""
    import h5py
    if r.cfg:
        return explicit_simulation(r.cfg)
    fc = r.folder_config()
    if fc:
        return explicit_simulation(fc)
    ap = ap_ref_of_h5(r.h5) if os.path.isfile(r.h5) else None
    try:
        return simulation_from_folder(r.doe_dir, ap)
    except ValueError:
        pass
    with h5py.File(r.h5, "r") as f:
        cases = sorted(c for c in f if c.startswith("case_"))
        keys = [k for k in f[cases[0]].attrs if str(k).startswith("$")] if cases else []
        sweep = {k: [float(f[c].attrs[k]) for c in cases] for k in keys if all(k in f[c].attrs for c in cases)}
    if not sweep:
        raise ValueError(f"{r.h5}: no $variable$ attributes to rebuild the simulation from")
    sim = {"base_dir": os.path.normpath(r.base_dir).replace("\\", "/"), "case": r.case, "doe_name": r.doe_name,
           "nb_proc": 1, "mode": "sweep", "sweep": sweep, "ap_ref": ap or {"mode": "none"},
           "extract_signals": ["Axial_disp", "Axial_vel", "Axial_acc"], "force_signal": "res_R_p"}
    _doe_runner().load_config(_tmp_config(sim))
    return sim


def planner_config(exp: Exp, i: int = 0) -> str:
    """A doe_runner YAML of run i that doe_planner can open (also for an imported folder)."""
    import yaml
    p = os.path.join(exp.runs_dir(), f"planner_{exp.runs[i].doe_name}.yaml")
    _write_if_changed(p, "# For doe_planner only (written by the experiments app)\n"
                      + yaml.safe_dump(simulation_of_run(exp.runs[i]), sort_keys=False))
    return p


def case_points_of(exp) -> list:
    """[(n rpm, Ap mm, label, Ap end mm)] of the cases of an experiment's data, label from its labelled dataset
    ('' if none); Ap end = Ap for a constant case (a ramp is the segment Ap -> Ap end)."""
    import h5py
    if exp is None or not exp.data_h5 or not os.path.isfile(exp.data_h5):
        return []
    lab = {c: lb for c, (lb, _) in _label_cases(exp.label["out"]).items()}
    out = []
    try:
        with h5py.File(exp.data_h5, "r") as f:
            for c in f:
                a = f[c].attrs if c.startswith("case_") else {}
                if "$spin_rate$" in a and "$Ap_start$" in a:
                    ap0 = float(a["$Ap_start$"]) * 1e3
                    out.append((float(a["$spin_rate$"]), ap0, lab.get(c, ""),
                                float(a["$Ap_end$"]) * 1e3 if is_ramp(a) else ap0))
    except OSError:
        return []
    return out


def import_dir(name: str, doe_dir: str, reference: str | None = None, label_out: str | None = None,
               description: str | None = None, h5: str = "doe_results.h5", ap_ref=None) -> str:
    """Create experiments/<name>.yaml from an already simulated DOE folder (h5 = its signals file); the stages
    whose results are found are turned on, plus validate when a reference is given. Existing outputs get an
    'imported' baseline record so nothing shows as stale because of a history the app does not know.
    h5 empty (the folder was simulated but never extracted): the simulation is rebuilt from the folder
    (simulation_from_folder, ap_ref for kappa) so Extract can run from the app; Simulate stays blocked."""
    doe_dir = os.path.normpath(os.path.abspath(doe_dir))
    path = exp_path(name)
    if os.path.exists(path):
        raise ValueError(f"experiment '{name}' already exists ({path})")
    if not h5:
        run = {"simulation": simulation_from_folder(doe_dir, ap_ref), "existing": True}
    elif not os.path.isfile(os.path.join(doe_dir, h5)):
        raise ValueError(f"{doe_dir} has no {h5}")
    else:
        run = {"dir": doe_dir.replace("\\", "/"), "case": detect_case(doe_dir)}
        if h5 != "doe_results.h5":
            run["h5"] = h5
        m = inspect_h5(os.path.join(doe_dir, h5))["values"].get("sim_model") or []
        m = next((str(v) for v in m if v is not None), None) or (ap_ref_of_h5(os.path.join(doe_dir, h5)) or {}).get("model")
        if m:
            run["model"] = m   # the simulated machine, kept in the experiment file
    d = {"name": name, "description": description or "", "stages": ["simulate", "extract"]}
    if reference:
        d["reference"] = reference
    d["runs"] = [run]
    refs = sorted(f for f in os.listdir(doe_dir) if f.startswith("reference_dataset") and f.endswith(".h5"))
    if label_out == "-":   # the user chose not to link any labelled dataset
        refs = []
    elif label_out:
        refs = [label_out]
    elif len(refs) > 1:   # prefer the one that says how it was labelled
        refs = [f for f in refs if _label_attrs(os.path.join(doe_dir, f)).get("strategy")][:1] or refs[:1]
    if refs and not reference:
        label = _label_attrs(os.path.join(doe_dir, refs[0]))
        label["out"] = os.path.join(doe_dir, refs[0]).replace("\\", "/")
        yml = os.path.join(doe_dir, "reference_labels.yaml")
        if os.path.isfile(yml):   # linked only if the dataset was built from it (same label for every case)
            built, ylab = _label_cases(os.path.join(doe_dir, refs[0])), _yaml_labels(yml)
            if built and all(ylab.get(c) == lab for c, (lab, _) in built.items()):
                label["labels_yaml"] = yml.replace("\\", "/")
        d["label"] = label
        d["stages"] += ["label_template", "label_build"]
    ind_h5 = os.path.join(doe_dir, "doe_indicator_results.h5")
    if os.path.isfile(ind_h5) and not reference:
        import h5py
        with h5py.File(ind_h5, "r") as f:
            c0 = next((f[k] for k in sorted(f) if k.startswith("case_")), None)
            names = sorted(k for k in c0 if "t" in c0[k] and "I_t" in c0[k]) if c0 is not None else []
        lib = presets()
        d["indicators"] = {"variants": {n: lib[n] for n in names if n in lib}, "out": ind_h5.replace("\\", "/")}
        d["stages"].append("indicators")
    if reference:
        d["stages"] += ["label_template", "label_build", "indicators", "validate"]
        d["indicators"] = indicators_for(reference)
        d["validate"] = {"channel": "Axial_disp", "early_tol_s": EARLY_TOL_S}
    if not d["description"] and not h5:
        sw = run["simulation"]["sweep"]
        n = sorted(set(sw.get("$spin_rate$", [])))
        d["description"] = (f"imported from {os.path.basename(doe_dir)} (simulated, not extracted): "
                            f"{len(next(iter(sw.values())))} cases" + (f", n = {n[0]:.10g} rpm" if len(n) == 1 else ""))
    if not d["description"]:
        info = h5_info(os.path.join(doe_dir, h5))
        spin = info["first"].get("$spin_rate$")
        kap = kappa_span(info)
        d["description"] = (f"imported from {os.path.basename(doe_dir)}: {len(info['cases'])} cases"
                            + (f" ({info['ramps']} ramps)" if info["ramps"] else "")
                            + (f", n = {float(spin):.10g} rpm" if spin is not None else "")
                            + (f", kappa {kap[0]:.2f}-{kap[1]:.2f}" if kap else ""))
    yaml_save(d, path)
    reload()
    exp = load(name)
    now = time.time()
    for key, st in stages(exp).items():
        if st.present():
            write_record(exp, key, {"stage": key, "status": "imported", "start": now, "end": now,
                                    "exit_code": 0, "hash": st.hash})
    return path


# ============================================================================== standardise an external .h5
STD_ATTRS = ("$Ap_start$", "$Ap_end$", "$spin_rate$", "kappa", "sim_case", "sim_model")


def inspect_h5(path: str) -> dict:
    """What an .h5 made outside the app has, compared with what the app needs (doe_results.h5 layout: case_*
    groups with <signal>/time + values and DOE attributes). {'ok', 'problems', 'cases', 'signals',
    'missing': {attr: [cases without it]}, 'values': {attr: [value per case]}}."""
    import h5py
    out = {"ok": False, "problems": [], "cases": [], "signals": [], "missing": {}, "values": {}, "ramps": []}
    with h5py.File(path, "r") as f:
        cases = sorted(k for k in f if k.startswith("case_") and isinstance(f[k], h5py.Group))
        out["cases"] = cases
        if not cases:
            out["problems"].append(f"no case_* groups (found: {sorted(f)[:8]}); the app expects case_000, case_001 …")
            return out
        g = f[cases[0]]
        out["signals"] = [s for s in g if isinstance(g[s], h5py.Group) and "time" in g[s] and "values" in g[s]]
        if not out["signals"]:
            out["problems"].append(f"{cases[0]} has no <signal>/time + values groups (found: {list(g)[:8]})")
        for a in STD_ATTRS:
            if a == "kappa":   # a ramp needs kappa_start and kappa_end instead (its 'kappa' is ignored)
                vals = [kappa_text(f[c].attrs, "g") if is_ramp(f[c].attrs) else f[c].attrs.get(a) for c in cases]
                out["missing"][a] = [c for c, v in zip(cases, vals) if v is None or "?" in str(v)]
            else:
                vals = [f[c].attrs.get(a) for c in cases]
                out["missing"][a] = [c for c, v in zip(cases, vals) if v is None]
            out["values"][a] = [v.item() if hasattr(v, "item") else v for v in vals]
        out["ramps"] = [c for c in cases if is_ramp(f[c].attrs)]
    out["ok"] = not out["problems"]
    return out


def experiments_using(h5: str) -> list:
    """Experiments whose data (a run's signals file or the merged data) is this .h5."""
    target, out = _norm(h5), []
    for n in list_experiments():
        try:
            e = load(n)
        except Exception:
            continue
        if any(_norm(p) == target for p in [e.data_h5] + [r.h5 for r in e.runs] if p):
            out.append(n)
    return out


def add_case_attrs(path: str, values: dict) -> list:
    """Write attributes to every case of an .h5: {attr: value or [value per case]}; signals are not touched.
    With 'ap_ref' (an ap_ref section) also kappa = Ap / ap_ref of each case. Returns what was written.
    Only metadata (sim_case, sim_model): the file keeps its date, so the stages that read it do not turn stale."""
    import h5py
    values = dict(values)
    ap_ref = values.pop("ap_ref", None)
    keep_date = not ap_ref and set(values) <= {"sim_case", "sim_model"}
    st = os.stat(path)
    done = []
    _write_case_attrs(path, values, ap_ref, done)
    for k in ("sim_case", "sim_model"):   # also at the root, as doe_runner extract writes them
        if values.get(k) and not isinstance(values[k], (list, tuple)):
            with h5py.File(path, "a") as f:
                f.attrs[k] = values[k]
    if keep_date:
        os.utime(path, (st.st_atime, st.st_mtime))
    _h5_info.cache_clear()
    return done


def _write_case_attrs(path: str, values: dict, ap_ref, done: list) -> None:
    import h5py
    with h5py.File(path, "a") as f:
        cases = sorted(k for k in f if k.startswith("case_"))
        for i, c in enumerate(cases):
            a = f[c].attrs
            for k, v in values.items():
                v = v[i] if isinstance(v, (list, tuple)) else v
                if v is None or v == "":
                    continue
                a[k] = v
                done.append(f"{c}.{k} = {v}")
            if ap_ref and "$Ap_start$" in a:
                ref = ap_ref_value(ap_ref, a.get("$spin_rate$"))
                if ref and is_ramp(a):   # a ramp: kappa at both ends (an old 'kappa' is left as it is, ignored)
                    a["kappa_start"], a["kappa_end"] = float(a["$Ap_start$"]) / ref, float(a["$Ap_end$"]) / ref
                    a["ap_ref_m"] = ref
                    done.append(f"{c}.kappa_start = {a['kappa_start']:.4g}, kappa_end = {a['kappa_end']:.4g}")
                elif ref:
                    a["kappa"] = float(a["$Ap_start$"]) / ref
                    a["ap_ref_m"] = ref
                    done.append(f"{c}.kappa = {a['kappa']:.4g}")
        if ap_ref:
            f.attrs["ap_ref_mode"] = ap_ref.get("mode", "none")
            if ap_ref.get("model"):
                f.attrs["ap_ref_model"] = ap_ref["model"]


# ============================================================================== run (the wrapper)
BACKUP_BEFORE_RUN = {"label_template"}   # its script refuses to overwrite: the old file is kept as .bak-<time>


def _norm(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))


def _overlap(a: str, b: str) -> bool:
    """Same file, or one is a folder that contains the other."""
    a, b = _norm(a), _norm(b)
    return a == b or a.startswith(b + os.sep) or b.startswith(a + os.sep)


def write_conflicts(exp: Exp, key: str) -> list:
    """Stages running now (any experiment) that write a file this stage reads or writes."""
    st = stages(exp)[key]
    mine = [p for p in st.inputs + st.writes if p]
    out = []
    for name in list_experiments():
        try:
            other = load(name)
        except Exception:
            continue
        for k, o in stages(other).items():
            if other.name == exp.name and k == key:
                continue
            rec = read_record(other, k)
            if not (rec and rec.get("status") == "running" and pid_alive(rec.get("pid"))):
                continue
            hit = [w for w in o.writes for p in mine if _overlap(w, p)]
            if hit:
                out.append(f"'{k}' of experiment '{other.name}' is running and writes {os.path.basename(hit[0])}")
    return out


def run_blockers(exp: Exp, key: str) -> list:
    """Reasons why the stage must not start now (empty list = it can run)."""
    S = stages(exp)
    if key not in S:
        return [f"stage '{key}' does not apply to experiment '{exp.name}'"]
    st, out = S[key], []
    if not st.runnable:
        out.append(st.why_not)
    missing = [p for p in st.inputs if p and not os.path.exists(p)]
    if missing and status(exp)[key][0] != "blocked":   # when blocked, the missing step already says it
        out.append(f"input missing: {os.path.basename(missing[0])}" + (f" (+{len(missing) - 1})" if len(missing) > 1 else ""))
    errs, _ = check(exp)
    out += [f"configuration error: {e}" for e in errs]
    state, reason = status(exp)[key]
    if state == "running":
        out.append("already running")
    elif state == "blocked":
        out.append(reason)
    out += write_conflicts(exp, key)
    return out


def existing_outputs(exp: Exp, key: str) -> list:
    """Outputs that a run would replace (asked before running)."""
    st = stages(exp)[key]
    return [p for p in st.outputs if os.path.isfile(p)] if st.present() else []


def accept(exp: Exp, keys=None) -> list:
    """Mark stale stages as up to date: their record takes the current configuration fingerprint and an
    'accepted' time later than its inputs (for a change that does not alter the result: the same folder written
    another way, attributes added to an input...). Only stages stale because of their configuration or of a
    newer input; 'upstream is stale' clears once the upstream is accepted. Returns the stages accepted."""
    done = []
    for k, (state, reason) in status(exp).items():
        if (keys and k not in keys) or state != "stale" or not (
                "configuration changed" in reason or "input changed after the run" in reason):
            continue
        rec = read_record(exp, k) or {"stage": k, "status": "imported", "start": time.time(), "exit_code": 0}
        st = stages(exp)[k]
        rec.update(hash=st.hash, accepted=max([time.time()] + [_mtime(p) for p in st.inputs if p]))
        write_record(exp, k, rec)
        done.append(k)
    return done


def dry_run(exp: Exp) -> list:
    """What running the experiment would do, without running anything: [(text, tag)], tag in ok / warn / bad /
    None. Cases of every simulated run (Ap, kappa, n), DOE folders, stages and their commands, checks."""
    out = [(f"Dry-run of {exp.name}  (nothing is run or written)", "head")]
    errs, warns = check(exp)
    for r in exp.runs:
        out.append((f"Run {r.doe_name}: {r.source}", "head"))
        out.append((f"  DOE folder {r.doe_dir}" + ("  (exists)" if os.path.isdir(r.doe_dir) else "  (new)"), None))
        if r.cfg:
            ap = r.cfg.get("ap_ref") or {"mode": "none"}
            out.append((f"  case {r.case}, nb_proc {r.cfg.get('nb_proc', 1)}, ap_ref {ap.get('mode')}"
                        + (f" {ap.get('model') or ap.get('manual')}" if ap.get("mode") != "none" else ""), None))
            for d in case_rows(r.cfg):
                extra = ", ".join(f"{a} {b:g}" for a, b in d.items() if a not in ROW_EXTRA)
                out.append((f"  case {d['case']:3d}: {'ramp ' if d['ramp'] else ''}Ap {ap_text(d)}"
                            f"  kappa {kappa_text(d)}  n {d.get('spin_rate', float('nan')):g} rpm"
                            + (f"  {extra}" if extra else ""), None))
        else:
            info = h5_info(r.h5)
            out.append((f"  {len(info['cases'])} cases in {os.path.basename(r.h5)}" if info else
                        f"  {r.h5} not found", "ok" if info else "bad"))
    out.append(("Stages", "head"))
    st = status(exp)
    for k, s in stages(exp).items():
        out.append((f"  {TITLES[k]:<18s} {st[k][0]:<8s} {st[k][1]}", None))
        for c in s.cmds:
            out.append(("      $ python " + " ".join(os.path.basename(c[0]) if i == 0 else str(a)
                                                for i, a in enumerate(c)), "hint"))
    out.append(("Checks", "head"))
    out += [(f"  ERROR {e}", "bad") for e in errs] + [(f"  warning {w}", "warn") for w in warns]
    if not errs and not warns:
        out.append(("  no problem found", "ok"))
    return out


def link_truth(exp: Exp) -> int:
    """Write into the experiment's indicator results, case by case, the true label of its labelled dataset
    (true_label, label_strategy), Ap_mm and the files used (attributes only, no recomputation; the file keeps its
    date so nothing downstream turns stale). Returns the number of cases labelled."""
    import h5py
    out, lab = exp.indicators["out"], exp.label["out"]
    if not (os.path.isfile(out) and os.path.isfile(lab)):
        return 0
    info = label_info(lab)
    mt = os.stat(out)
    n = 0
    with h5py.File(out, "a") as f:
        f.attrs.update(reference_dataset=exp.reference, label_dataset=lab,
                       label_strategy=str(exp.label.get("strategy", "")))
        for c in f:
            g = f[c]
            if not (isinstance(g, h5py.Group) and c.startswith("case_")):
                continue
            if "$Ap_start$" in g.attrs:
                g.attrs["Ap_mm"] = float(g.attrs["$Ap_start$"]) * 1e3   # a ramp: Ap at the start, Ap_end_mm too
                if "$Ap_end$" in g.attrs:
                    g.attrs["Ap_end_mm"] = float(g.attrs["$Ap_end$"]) * 1e3
            if c in info:
                d = info[c]
                g.attrs.update(true_label=d["label"], label_strategy=str(exp.label.get("strategy", "")))
                if d["ramp"] and d["t_onset"] is not None:   # where the ground truth of the ramp turns unstable
                    g.attrs["t_onset"] = d["t_onset"]
                elif "t_onset" in g.attrs:
                    del g.attrs["t_onset"]
                n += 1
    os.utime(out, (mt.st_atime, mt.st_mtime))
    _h5_info.cache_clear()
    return n


def stamp_outputs(exp: Exp, st: Stage) -> None:
    """Traceability attrs on the .h5 outputs of a successful run (with what the file is, for the viewer)."""
    import h5py
    for p, role in zip(st.outputs, st.roles[1] or [""] * len(st.outputs)):
        if p.endswith(".h5") and os.path.isfile(p):
            try:
                with h5py.File(p, "a") as f:
                    f.attrs.update(experiment=exp.name, experiment_stage=st.key, experiment_hash=st.hash,
                                   experiment_role=role, experiment_reference=exp.reference,
                                   experiment_date=datetime.datetime.now().isoformat(timespec="seconds"))
            except OSError as exc:   # traceability must never fail a finished run
                print(f"[experiment] could not stamp {p}: {exc}")


def run_stage(name: str, key: str, yes: bool = False, cmds=None, notify_end: bool = True) -> int:
    """Run one stage: checks, record 'running', tee output to the console and the log, record the result.
    cmds overrides the stage commands (selftest only). Returns the exit code."""
    import subprocess
    exp = load(name)
    blockers = run_blockers(exp, key)
    if blockers:
        print("[experiment] cannot run:\n  - " + "\n  - ".join(blockers))
        return 2
    st = stages(exp)[key]
    old = existing_outputs(exp, key)
    if old and not yes:
        ans = input(f"[experiment] this run replaces {[os.path.basename(p) for p in old]}. Continue? [y/N] ")
        if ans.strip().lower() not in ("y", "yes", "s", "si"):
            print("[experiment] cancelled")
            return 1
    if key in BACKUP_BEFORE_RUN:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for p in old:
            os.replace(p, f"{p}.bak-{stamp}")
    for p in st.outputs:
        if p.endswith((".h5", ".yaml")):
            os.makedirs(os.path.dirname(p), exist_ok=True)
    os.makedirs(exp.runs_dir(), exist_ok=True)
    log_path = os.path.join(exp.runs_dir(), f"{key}.log")
    cmds = cmds if cmds is not None else st.cmds
    rec = {"stage": key, "status": "running", "start": time.time(), "pid": os.getpid(), "hash": st.hash,
           "cmds": [[sys.executable, *c] for c in cmds], "log": log_path}
    write_record(exp, key, rec)
    code = 0
    with open(log_path, "w", encoding="utf-8") as log:
        try:
            for c in cmds:
                line = f"$ {sys.executable} {' '.join(c)}"
                print(line)
                log.write(line + "\n")
                p = subprocess.Popen([sys.executable, "-u", *c], cwd=os.path.dirname(c[0]) or None,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace",
                                     env=dict(os.environ, PYTHONIOENCODING="utf-8"))   # scripts print non-cp1252 chars
                for out_line in p.stdout:
                    sys.stdout.write(out_line)
                    log.write(out_line)
                    log.flush()
                code = p.wait()
                if code != 0:
                    break
        except KeyboardInterrupt:
            code = -2
            log.write("\n[experiment] interrupted\n")
    rec.update(status="done" if code == 0 else "failed", end=time.time(), exit_code=code)
    if notify_end and cmds is None and rec["end"] - rec["start"] > NOTIFY_AFTER_S:
        notify(f"{exp.name}: {TITLES[key]} " + ("done" if code == 0 else "FAILED"),
               f"took {_fmt_dur(rec['end'] - rec['start'])}" + ("" if code == 0 else f", exit code {code}: see Log"))
    write_record(exp, key, rec)
    if code == 0:
        stamp_outputs(exp, st)
        if key == "extract":
            try:
                stamp_model(exp)
            except OSError as exc:
                print(f"[experiment] could not write the model into doe_results.h5: {exc}")
        if key in ("label_build", "indicators"):   # the indicator results always carry the current true labels
            try:
                link_truth(exp)
            except OSError as exc:
                print(f"[experiment] could not write the true labels into the indicator results: {exc}")
    print(f"[experiment] {key}: {'done' if code == 0 else f'FAILED (exit {code})'} - log: {log_path}")
    return code


def _selftest_edit(root: str, train_cfg: str) -> None:
    """Editing helpers on the selftest experiments; configs/ redirected to a temp folder."""
    global CONFIGS_DIR, _sld
    dr = _doe_runner()
    old = (CONFIGS_DIR, dr.CONFIGS_DIR)
    CONFIGS_DIR = dr.CONFIGS_DIR = os.path.join(root, "configs")
    os.makedirs(CONFIGS_DIR)
    base = os.path.join(root, "data")
    try:
        save_section("train", "validate", {"channel": "Axial_vel"})
        assert own_yaml("train")["validate"] == {"channel": "Axial_vel"}
        save_section("train", "validate", None)
        assert "validate" not in own_yaml("train")
        # variant names: same pattern as doe_indicators, unique among the rows of the table
        spec = {"indicator": "MaxEnt_SPRT", "func": "Default", "mode": "by_revolution", "signal": "Axial_vel",
                "params_physical": {"T_rev": "T_rev", "N_rev_window": 4, "step_rev": 1}}
        assert propose_variant_name(spec) == "maxent_revo_dec4_1step"
        assert propose_variant_name(spec, {"maxent_revo_dec4_1step"}) == "maxent_revo_dec4_1step_v2"
        # names: no forced prefix, a single kappa is written once
        n1 = propose_name("Validation against a reference", "1DOF_150Hz", 12098.28, [0.528, 1.906])
        assert n1.startswith("val_1DOF150_n12098_k0.53-1.91"), n1
        assert propose_name("Simulation only", "1DOF_150Hz", 9000, [1.0]) == "sim_1DOF150_n9000_k1"
        # explicit simulation from the form: Ap in mm, kappa x manual ap_ref, a single Ap, every combination
        man = {"mode": "manual", "manual": 0.008}
        sim = build_simulation(base, "1DOF_150Hz", "DOE_F", "4, 12", "mm", "12000", man,
                               {"$f_tooth$": "0.05", "$dxl_size$": "2e-4"})
        assert sim["sweep"]["$Ap_start$"] == [0.004, 0.012] and sim["sweep"]["$f_tooth$"] == [0.05, 0.05]
        assert build_simulation(base, "1DOF_150Hz", "D", "0.5, 1.5", "kappa", "12000", man)["sweep"]["$Ap_start$"] == [0.004, 0.012]
        one = build_simulation(base, "1DOF_150Hz", "D1", "8", "mm", "9000", man)
        assert one["sweep"]["$Ap_start$"] == [0.008] and len(check_simulation(one)[0]) == 1   # a single Ap
        four = build_simulation(base, "1DOF_150Hz", "D4", "4 8", "mm", "9000 12000", man, combine=True)
        assert len(four["sweep"]["$Ap_start$"]) == 4
        try:
            build_simulation(base, "1DOF_150Hz", "D", "4 8", "mm", "9000 10000 12000", man)
            raise AssertionError("lists of different lengths accepted")
        except ValueError as exc:
            assert "row by row" in str(exc)
        form = simulation_form(sim)
        again = build_simulation(form["base_dir"], form["case"], form["doe_name"], form["depths"], "mm",
                                 form["spins"], form["ap_ref"], form["variables"])
        assert again["sweep"] == sim["sweep"], (again["sweep"], sim["sweep"])
        assert form["depths_end"] == []                                       # constant: no ends in the form
        # ramps: an end per depth (equal = constant), in mm or kappa; the form rebuilds them
        rmp = build_simulation(base, "1DOF_150Hz", "D_R", "4, 12, 6", "mm", "12000", man, depths_end="12, 4, 6")
        assert rmp["sweep"]["$Ap_end$"] == [0.012, 0.004, 0.006] and rmp["sweep"]["$Ap_start$"] == [0.004, 0.012, 0.006]
        rows = case_rows(_doe_runner().load_config(_tmp_config(rmp)))
        assert [d["ramp"] for d in rows] == [True, True, False] and rows[0]["kappa"] is None
        assert abs(rows[0]["kappa_start"] - 0.5) < 1e-12 and abs(rows[0]["kappa_end"] - 1.5) < 1e-12 and rows[2]["kappa"] == 0.75
        assert kappa_text(rows[0]) == "0.500 -> 1.500" and ap_text(rows[1]) == "12.0000 -> 4.0000 mm"
        fr2 = simulation_form(rmp)
        assert fr2["depths_end"] == [12.0, 4.0, 6.0]
        again = build_simulation(base, "1DOF_150Hz", "D_R", fr2["depths"], "mm", fr2["spins"], man, fr2["variables"],
                                 depths_end=fr2["depths_end"])
        assert again["sweep"] == rmp["sweep"]
        k_r = build_simulation(base, "1DOF_150Hz", "D", "0.5", "kappa", "12000", man, depths_end="1.5")
        assert k_r["sweep"]["$Ap_start$"] == [0.004] and k_r["sweep"]["$Ap_end$"] == [0.012]
        fac = build_simulation(base, "1DOF_150Hz", "D", "4 8", "mm", "9000 12000", man, combine=True, depths_end="10 8")
        assert list(zip(fac["sweep"]["$Ap_start$"], fac["sweep"]["$Ap_end$"])) == [(0.004, 0.01)] * 2 + [(0.008, 0.008)] * 2
        try:
            build_simulation(base, "1DOF_150Hz", "D", "4 8 9", "mm", "9000", man, depths_end="10 8")
            raise AssertionError("ends of another length accepted")
        except ValueError as exc:
            assert "depths end" in str(exc)
        assert any("does not cross kappa = 1" in w for w in sim_problems(_doe_runner().load_config(_tmp_config(
            build_simulation(base, "1DOF_150Hz", "D", "2", "mm", "12000", man, depths_end="4"))))[1])
        w = kappa_overlap(rows, load("train"))     # training kappa 0.5 / 1.5: the ramps are not compared
        assert w == ["2 ramp case(s): no repeat check (a ramp covers a range of kappa)"], w
        # a decreasing ramp needs a workpiece that can make Ap decrease (marker in the case's db_def)
        dbd = os.path.join(base, "1DOF_150Hz", "in")
        os.makedirs(dbd, exist_ok=True)
        cfg_r = _doe_runner().load_config(_tmp_config(rmp))
        assert ramp_template_ok(base, "1DOF_150Hz") is None and not sim_problems(cfg_r)[0]   # unknown: no error
        with open(os.path.join(dbd, "1DOF_150Hz-db_def.py"), "w") as f:
            f.write("def h_max_truncado(u, v): pass\n")
        assert any("only makes Ap grow" in x for x in sim_problems(cfg_r)[0])
        with open(os.path.join(dbd, "1DOF_150Hz-db_def.py"), "w") as f:
            f.write(RAMP_MARK + "\n")
        assert not sim_problems(cfg_r)[0]
        shutil.rmtree(dbd)
        # kappa at an n where the SLD has no limit (pocket between lobes) -> clear error, never Ap = inf
        real_sld = _sld

        class _FakeSld:
            MODELS = {"1DOF_150": None}

            @staticmethod
            def ap_lim(model, spin):
                return float("inf") if spin == 9000 else 8.0

            @staticmethod
            def ap_crit(model):
                return 8.0
        _sld = lambda: _FakeSld   # noqa: E731
        try:
            at_spin = {"mode": "model_at_spin", "model": "1DOF_150"}
            assert build_simulation(base, "1DOF_150Hz", "D", "1", "kappa", "12000", at_spin)["sweep"]["$Ap_start$"] == [0.008]
            try:
                build_simulation(base, "1DOF_150Hz", "D", "1", "kappa", "9000", at_spin)
                raise AssertionError("infinite ap_ref accepted")
            except ValueError as exc:
                assert "no finite stability limit" in str(exc)
        finally:
            _sld = real_sld
        # an Ap = inf already written in a config is an error of the experiment (blocks Run), not a silent run
        bad = os.path.join(CONFIGS_DIR, "bad_inf.yaml")
        with open(bad, "w") as f:
            f.write(f"base_dir: {base}\ncase: 1DOF_150Hz\ndoe_name: DOE_INF\nmode: sweep\n"
                    "sweep:\n  $Ap_start$: [.inf]\n  $Ap_end$: [.inf]\n  $spin_rate$: 9000.0\n")
        create_experiment("inf", [{"config": "bad_inf"}], stages_on=FLOWS["Simulation only"])
        errs = check(load("inf"))[0]
        assert any("Ap_start = inf" in x for x in errs) and run_blockers(load("inf"), "simulate"), errs
        delete("inf")
        # create from scratch: explicit simulation, chosen stages; the generated doe_runner file lives in .runs
        create_experiment("fresh", [{"simulation": sim}], stages_on=FLOWS["Indicators"], description="x")
        fr = load("fresh")
        assert fr.runs[0].n_cases == 2 and fr.runs[0].config.startswith(os.path.join(RUNS_DIR, "fresh"))
        assert list(stages(fr)) == ["simulate", "extract", "label_template", "label_build", "indicators"]
        assert fr.flow == "Indicators" and default_goal(fr) == "Indicators computed"
        assert any("no indicator variant" in x for x in check(fr)[0])          # the selftest presets have no defaults
        assert any("case   1: Ap 12.0000 mm" in t for t, _ in dry_run(fr)), dry_run(fr)
        w = kappa_overlap([{"kappa": k, "spin_rate": 12000.0} for k in (0.5, 0.505, 1.0)], load("train"))
        assert "1 kappa already in the reference" in w[0] and "within" in w[1], w   # training: kappa 0.5 and 1.5, 12000
        w = kappa_overlap([{"kappa": 0.5, "spin_rate": 9000.0}], load("train"))     # another n: no comparison
        assert len(w) == 1 and "different n" in w[0] and "already" not in w[0], w
        assert [k for _, k in chain_stages(fr)] == ["simulate", "extract", "label_template", "label_build", "indicators"]
        # a test linked to a training that has not run yet: the training's stages come first, then the test's
        create_experiment("te_new", [{"simulation": dict(sim, doe_name="D_TE")}],
                          stages_on=FLOWS["Validation against a reference"], reference="fresh")
        ch = chain_stages(load("te_new"))
        assert ch[:4] == [("fresh", "simulate"), ("fresh", "extract"), ("fresh", "label_template"),
                          ("fresh", "label_build")], ch
        assert ch[-1] == ("te_new", "validate") and ("te_new", "simulate") in ch[4:], ch
        global run_stage, notify   # run_chain from the test runs the training's stage (fake run that fails at once)
        real = (run_stage, notify)
        calls = []
        run_stage = lambda n, k, **kw: calls.append((n, k)) or 1   # noqa: E731
        notify = lambda *a: None   # noqa: E731
        try:
            assert run_chain("te_new") == 1 and calls == [("fresh", "simulate")], calls
        finally:
            run_stage, notify = real
        delete("te_new")
        assert estimate_time(sim).startswith("time:")
        # only the stages turned on (plus what they need)
        create_experiment("simonly", [{"simulation": one}], stages_on=["extract"])
        assert list(stages(load("simonly"))) == ["simulate", "extract"]
        # a validation whose reference is simulation-only: clear error, defaults instead of an empty inheritance
        create_experiment("val_on_sim", [{"simulation": dict(one, doe_name="D2")}],
                          stages_on=FLOWS["Validation against a reference"], reference="simonly")
        vs = load("val_on_sim")
        assert own_yaml("val_on_sim")["indicators"]["variants"] != "inherit"
        assert any("has no labelled dataset" in x for x in check(vs)[0]), check(vs)
        assert status(vs)["indicators"][0] == "blocked"
        delete("val_on_sim")
        # copy: explicit, its own outputs, another DOE folder
        copy_experiment("fresh", "fresh_copy", doe_suffix="_b")
        cp = load("fresh_copy")
        assert cp.runs[0].doe_name == "DOE_F_b" and cp.out_dir.endswith("fresh_copy") and "extends" not in own_yaml("fresh_copy")
        delete("fresh_copy")
        assert "fresh_copy" not in list_experiments() and not os.path.isdir(os.path.join(RUNS_DIR, "fresh_copy"))
        # a config run (may inherit) written in full inside the experiment
        make_explicit("train")
        t = own_yaml("train")
        assert "simulation" in t["runs"][0] and t["runs"][0]["simulation"]["doe_name"] == "DOE_T"
        assert load("train").runs[0].n_cases == 2
        try:
            delete("train")
            raise AssertionError("deleted an experiment others depend on")
        except ValueError as exc:
            assert "used by" in str(exc)
    finally:
        CONFIGS_DIR, dr.CONFIGS_DIR = old


# ============================================================================== extras: kappa overlap, time, notify, chain
def kappa_overlap(rows, ref, tol: float = 0.01) -> list:
    """Warnings for the cases of a new DOE (rows: case_rows of the simulation) whose kappa repeats (or is closer
    than tol to) a kappa of the reference experiment: they test nothing new. kappa = Ap / limit at the case's n,
    so it is only comparable at the SAME n: at another n the same kappa is another depth, and the check says so.
    Ramp cases cover a range of kappa: they are not compared point by point."""
    info = h5_info(ref.data_h5) if ref is not None and ref.data_h5 else None
    if not info or not info["kappa"]:
        return []
    n_ramps = sum(bool(d.get("ramp")) for d in rows)
    if n_ramps:
        note = [f"{n_ramps} ramp case(s): no repeat check (a ramp covers a range of kappa)"]
        rest = [d for d in rows if not d.get("ramp")]
        return note + (kappa_overlap(rest, ref, tol) if rest else [])
    rn = info["first"].get("$spin_rate$")
    rk = sorted(info["kappa"])
    mine = [(float(d["kappa"]), float(d["spin_rate"])) for d in rows
            if d.get("kappa") is not None and d.get("spin_rate") is not None]
    if not mine:
        return []
    if rn is not None:
        other = sorted({round(n, 1) for _, n in mine if abs(n - float(rn)) > 1.0})
        if other:
            same_n = [(k, n) for k, n in mine if abs(n - float(rn)) <= 1.0]
            note = (f"the new cases are at n {', '.join(f'{n:g}' for n in other)} rpm, the reference at "
                    f"{float(rn):g} rpm (different n): no repeat check against the reference")
            if not same_n:
                return [note]
            mine = same_n
            ks = [k for k, _ in mine]
            return [note] + kappa_overlap([{"kappa": k, "spin_rate": rn} for k in ks], ref, tol)
    ks = [k for k, _ in mine]
    same = sorted({round(k, 3) for k in ks if min(abs(k - r) for r in rk) < 1e-3})
    near = sorted({round(k, 3) for k in ks if 1e-3 <= min(abs(k - r) for r in rk) < tol})
    out = []
    if same:
        out.append(f"{len(same)} kappa already in the reference '{ref.name}': {same[:8]} (they test nothing new)")
    if near:
        out.append(f"{len(near)} kappa within {tol:g} of a reference kappa: {near[:8]}")
    return out


@lru_cache(maxsize=32)
def _case_times(base_dir: str, case: str, stamp: float) -> list:
    """[(dxl_size, nb_dt_rev, seconds per case, exact)] of every DOE already simulated in base_dir: wall_time_s.txt
    (written by --timed: exact) or, without it, the spacing of the sens_out.hdf5 dates of that DOE (approximate:
    it already includes the parallelism of that run)."""
    dr = _doe_runner()
    out = []
    for doe in os.listdir(base_dir) if os.path.isdir(base_dir) else []:
        d = os.path.join(base_dir, doe)
        idx = [i for i in os.listdir(d) if i.isdigit()] if os.path.isdir(d) else []
        rows = []
        for i in idx:
            vv = os.path.join(d, i, case, "var_val.py")
            so = os.path.join(d, i, case, "sens_out.hdf5")
            if not os.path.isfile(so):
                continue
            try:
                v = dr.read_var_val(vv) if os.path.isfile(vv) else {}
            except Exception:
                v = {}
            wt = os.path.join(d, i, "wall_time_s.txt")
            try:
                sec = float(open(wt).read().strip()) if os.path.isfile(wt) else None
            except ValueError:
                sec = None
            rows.append((v.get("$dxl_size$"), v.get("$nb_dt_rev$"), sec, _mtime(so)))
        exact = [r for r in rows if r[2] is not None]
        out += [(r[0], r[1], r[2], True) for r in exact]
        if not exact and len(rows) >= 3:
            ts = sorted(r[3] for r in rows)
            out += [(rows[0][0], rows[0][1], (ts[-1] - ts[0]) / (len(ts) - 1), False)]
    return out


def estimate_time(sim: dict) -> str:
    """Rough duration of a simulation from the DOEs already simulated in its base_dir with the same
    discretisation (dxl_size, nb_dt_rev). Text for the preview."""
    import statistics
    try:
        rows = case_rows(_doe_runner().load_config(_tmp_config(sim)))
    except Exception:
        return ""
    base, case = sim.get("base_dir", ""), sim.get("case", "")
    known = _case_times(base, case, _mtime(base))
    if not known:
        return "time: no earlier simulation in this base_dir to estimate it (run once with --timed to measure)"
    nb = max(1, int(sim.get("nb_proc") or 1))
    per, kinds = [], set()
    for d in rows:
        same = [t for t in known if t[0] is not None and t[1] is not None and d.get("dxl_size") is not None
                and abs(float(t[0]) - float(d["dxl_size"])) < 1e-12 and abs(float(t[1]) - float(d.get("nb_dt_rev", -1))) < 1e-9]
        pool = [t for t in same if t[3]] or same or known
        kinds.add("exact" if pool[0][3] and pool is not known else ("approx." if pool is not known else "other discretisation"))
        per.append(statistics.median(t[2] for t in pool))
    total = sum(per) / nb
    return (f"time: ~{_fmt_dur(statistics.median(per))} per case, ~{_fmt_dur(total)} for {len(rows)} case(s) with "
            f"nb_proc {nb} ({', '.join(sorted(kinds))}, from {len(known)} earlier cases in this base_dir)")


NOTIFY_AFTER_S = 60   # a Windows notification when a stage that took longer than this ends


def notify(title: str, msg: str) -> None:
    """Windows notification (toast), best effort, never blocks nor fails the caller."""
    if os.name != "nt":
        return
    import subprocess
    ps = ("[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null;"
          "$x = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent("
          "[Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
          "$t = $x.GetElementsByTagName('text');"
          "$t.Item(0).AppendChild($x.CreateTextNode($env:EXP_N_TITLE)) > $null;"
          "$t.Item(1).AppendChild($x.CreateTextNode($env:EXP_N_MSG)) > $null;"
          "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("
          r"'{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe')"
          ".Show([Windows.UI.Notifications.ToastNotification]::new($x))")
    try:
        subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                         env=dict(os.environ, EXP_N_TITLE=title, EXP_N_MSG=msg), creationflags=0x08000000,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass


def chain_stages(exp: Exp, goal: str | None = None) -> list:
    """[(experiment name, stage)] that 'Run to goal' would run, in order: also the stages of the reference
    experiment the goal needs (e.g. a test runs the simulation and labelled dataset of its training first)."""
    return [(e.name, k) for e, k, _s, _r in todo_stages(exp, goal)]


def run_chain(name: str, goal: str | None = None, review: bool = True) -> int:
    """Run the next steps of the goal one after the other in this console, in this experiment and in the ones
    it needs (its reference). Stops on an error, on a stage that is running or blocked, and (review=True) after
    a Label template so the labels can be checked."""
    ran = []
    while True:
        reload()
        exp = load(name)
        g = goal if goal in GOALS else default_goal(exp)
        nxt = next_step(exp, g)
        done = ", ".join(f"{TITLES[k]}" + ("" if n == name else f" ({n})") for n, k in ran) or "nothing to run"
        if nxt is None:
            print(f"[experiment] goal '{g}' reached: {done}")
            notify(f"{name}: goal reached", f"{g}: {done}")
            return 0
        e, k, state, reason = nxt
        where = "" if e is exp else f" of experiment '{e.name}'"
        if (e.name, k) in ran or state in ("blocked", "running"):
            print(f"[experiment] stop at '{k}'{where} ({state}: {reason})")
            return 2
        print(f"[experiment] ===== {TITLES[k]}{where} =====")
        code = run_stage(e.name, k, yes=True, notify_end=False)
        if code:
            notify(f"{name}: {TITLES[k]}{where} FAILED", f"exit code {code}; the chain stopped (see Log)")
            return code
        ran.append((e.name, k))
        if review and k == "label_template":
            print(f"[experiment] labels proposed{where}: review the labels YAML ('Labels YAML' in the app), then "
                  "'Run to goal' again: it continues from Label build.")
            notify(f"{e.name}: review the labels", "Label template done; check the labels YAML, then Run to goal again")
            return 0


def chain_command(name: str, goal: str, python: str | None = None) -> list:
    """argv the app uses to open a console that runs the experiment up to its goal."""
    return [python or sys.executable, os.path.abspath(__file__), "chain", name, "--goal", goal]


def run_command(name: str, key: str, python: str | None = None, yes: bool = True) -> list:
    """argv the app uses to open a console with the wrapper."""
    return [python or sys.executable, os.path.abspath(__file__), "run", name, key] + (["--yes"] if yes else [])


def _selftest_run(e: Exp) -> None:
    """Wrapper checks on the selftest training experiment (its label_build is done at this point)."""
    import subprocess
    import h5py
    tmp = tempfile.mkdtemp(prefix="exp_run_")
    ok, bad, tpl = (os.path.join(tmp, n) for n in ("ok.py", "bad.py", "tpl.py"))
    with open(ok, "w") as f:
        f.write("import sys, h5py\nprint('hello from the stage')\nh5py.File(sys.argv[1], 'w').close()\n")
    with open(bad, "w") as f:
        f.write("print('about to fail')\nraise SystemExit(3)\n")
    with open(tpl, "w") as f:
        f.write("import sys\nopen(sys.argv[1], 'w').write('new')\n")
    # what the panel shows: label counts + boundary, badge, data summary
    summ = " | ".join(t for t, _ in stage_summary(e, "label_build"))
    assert "stable 1" in summ and "gray 1" in summ, summ
    assert stage_badge(e, "label_build") == "S 1 · G 1" and "2 cases" in stage_summary(e, "extract")[0][0]
    assert stage_summary(e, "indicators") == [("no results yet", None)] and stage_progress(e, "simulate") == (2, 2)
    # a done later stage covers its missing earlier ones: no "next step" back to the template
    lab = e.label["labels_yaml"]
    os.rename(lab, lab + ".hide")
    assert status(e)["label_build"][0] == "done" and next_step(e, "Labelled dataset") is None
    assert status(e)["label_template"][0] == "skipped"
    assert any(b.startswith("input missing") for b in run_blockers(e, "label_build"))
    os.rename(lab + ".hide", lab)
    out = e.indicators["out"]
    assert run_stage(e.name, "indicators", yes=True, cmds=[[ok, out]]) == 0
    rec = read_record(e, "indicators")
    assert rec["status"] == "done" and rec["exit_code"] == 0 and "hello from the stage" in open(rec["log"]).read()
    with h5py.File(out, "r") as f:
        assert f.attrs["experiment"] == e.name and f.attrs["experiment_stage"] == "indicators"
    assert status(e)["indicators"][0] == "done" and existing_outputs(e, "indicators") == [out]
    with h5py.File(out, "a") as f:   # true labels written into existing indicator results, date kept
        f.create_group("case_000").attrs["$Ap_start$"] = 0.004
    mt = os.path.getmtime(out)
    assert link_truth(e) == 1 and os.path.getmtime(out) == mt and status(e)["indicators"][0] == "done"
    with h5py.File(out, "r") as f:
        assert f["case_000"].attrs["true_label"] == "stable" and abs(f["case_000"].attrs["Ap_mm"] - 4.0) < 1e-9
    assert run_stage(e.name, "indicators", yes=True, cmds=[[bad]]) == 3
    assert status(e)["indicators"][0] == "failed" and "exit code 3" in status(e)["indicators"][1]
    # label_template keeps the old file as .bak-<time> (its script refuses to overwrite)
    assert run_stage(e.name, "label_template", yes=True, cmds=[[tpl, e.label["labels_yaml"]]]) == 0
    assert any(f.startswith("reference_labels.yaml.bak-") for f in os.listdir(os.path.dirname(e.label["labels_yaml"])))
    # a running stage that writes doe_results.h5 blocks the stages that read it, in any experiment
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        write_record(e, "static_deflection", {"status": "running", "start": time.time(), "pid": sleeper.pid})
        blk = run_blockers(e, "label_template")
        assert any("is running and writes doe_results.h5" in b for b in blk), blk
        assert run_stage(e.name, "label_template", yes=True, cmds=[[ok, "x.h5"]]) == 2
        assert run_blockers(e, "static_deflection") == ["already running"] or "already running" in run_blockers(e, "static_deflection")
    finally:
        sleeper.kill()
        sleeper.wait()
    assert not write_conflicts(e, "label_template")                 # process gone: nothing blocks
    os.remove(record_path(e, "static_deflection"))
    assert run_command(e.name, "extract")[2:5] == ["run", e.name, "extract"]
    shutil.rmtree(tmp, ignore_errors=True)


# ============================================================================== CLI text
SYMBOL = {"done": "OK ", "stale": "OLD", "running": "RUN", "failed": "ERR", "pending": " . ", "blocked": " x ",
          "skipped": " - "}


def status_text(exp: Exp, goal: str | None = None) -> str:
    sm, st = summary(exp), status(exp)
    head = (f"{exp.name}  [{exp.flow}" + (f", reference {exp.ref.name}" if exp.ref else "") + "]  "
            + (f"n={float(sm['spin']):.10g} rpm  " if sm["spin"] is not None else "")
            + f"{sm['cases']} cases" + (f"  kappa {sm['kappa'][0]:.2f}-{sm['kappa'][1]:.2f}" if sm["kappa"] else "")
            + (f"  ramps {sm['ramps']} of {sm['cases']}" if sm.get("ramps") else "")
            + f"  progress {sm['done']}/{sm['total']}")
    lines = [head]
    for key, (state, reason) in st.items():
        lines.append(f"  [{SYMBOL[state]}] {key:<18s} {state:<8s} {reason}" + ("   (optional)" if key in OPTIONAL else ""))
    goal = goal if goal in goals(exp) else default_goal(exp)
    nxt = next_step(exp, goal)
    lines.append(f"  goal: {goal} -> " + ("reached" if nxt is None else
                 f"next: {nxt[1]}" + ("" if nxt[0] is exp else f" of '{nxt[0].name}'") + f" ({nxt[2]}: {nxt[3]})"))
    errs, warns = check(exp)
    lines += [f"  ERROR: {e}" for e in errs] + [f"  warning: {w}" for w in warns]
    return "\n".join(lines)


# ============================================================================== selftest
def _selftest():
    import h5py
    root = tempfile.mkdtemp(prefix="exp_selftest_")
    old = (EXP_DIR, RUNS_DIR, VARIANTS_FILE)
    try:
        set_root(os.path.join(root, "experiments"))
        base = os.path.join(root, "data")
        os.makedirs(os.path.join(base, "1DOF_150Hz"))
        cfg = os.path.join(root, "train_cfg.yaml")
        with open(cfg, "w") as f:
            f.write(f"base_dir: {base}\ncase: 1DOF_150Hz\ndoe_name: DOE_T\nmode: sweep\n"
                    "sweep:\n  $Ap_start$: [0.004, 0.012]\n  $spin_rate$: 12000.0\n"
                    "ap_ref: {mode: manual, manual: 0.008}\n")
        yaml_save({"kind": "training", "runs": [{"config": cfg}],
                   "label": {"strategy": "amplitude", "lim_inf_pct": 10, "lim_sup_pct": 40},
                   "indicators": {"variants": ["v1"], "f_modal": 150.0}}, exp_path("train"))
        yaml_save({"variants": {"v1": {"indicator": "MaxEnt_SPRT", "mode": "by_revolution", "signal": "Axial_vel",
                                       "params_physical": {"T_rev": "T_rev", "N_rev_window": 4, "step_rev": 1}}}},
                  VARIANTS_FILE)
        e = load("train")   # an older file: kind + preset names, no 'stages' (the whole main path + optional)
        assert e.flow == "Indicators" and e.runs[0].n_cases == 2 and e.data_h5.endswith(os.path.join("DOE_T", "doe_results.h5"))
        assert e.out_dir == os.path.join(base, "DOE_T", "train") and e.label["out"].endswith("reference_dataset_amp.h5")
        st = status(e)
        assert st["simulate"][0] == "pending" and st["extract"][0] == "blocked" and "merge" not in st
        assert "validate" not in st and next_step(e)[1] == "simulate"
        # simulate both cases
        for i in range(2):
            p = os.path.join(base, "DOE_T", str(i), "1DOF_150Hz")
            os.makedirs(p)
            open(os.path.join(p, "sens_out.hdf5"), "w").close()
        reload()
        e = load("train")
        assert status(e)["simulate"][0] == "done" and status(e)["extract"][0] == "pending"
        with h5py.File(e.data_h5, "w") as f:
            for i, k in enumerate((0.5, 1.5)):
                g = f.create_group(f"case_{i:03d}")
                g.attrs.update({"$spin_rate$": 12000.0, "$dxl_size$": 2e-4, "$nb_dt_rev$": 200.0, "$f_tooth$": 0.05,
                                "kappa": k})
        st = status(e)
        assert st["extract"][0] == "done" and st["label_template"][0] == "pending", st
        assert "no run record" in st["extract"][1]
        tpl = stages(e)["label_template"].cmds[0]
        assert "--lim-sup-pct" in tpl and tpl[tpl.index("--lim-sup-pct") + 1] == "40"
        # label outputs + a record whose hash is old -> stale, and its dependents too
        os.makedirs(e.out_dir)
        open(e.label["labels_yaml"], "w").close()
        with h5py.File(e.label["out"], "w") as f:
            f.create_group("stable/case_000").create_dataset("Axial_disp__000", data=[0.0])
            f.create_group("gray/case_001").create_dataset("Axial_disp__000", data=[0.0])
        assert status(e)["label_build"][0] == "done" and next_step(e, "Labelled dataset") is None
        write_record(e, "label_template", {"status": "done", "start": time.time(), "hash": "old"})
        st = status(e)
        assert st["label_template"][0] == "stale" and st["label_build"][0] == "stale" and "upstream" in st["label_build"][1]
        assert accept(e) == ["label_template"] and status(e)["label_build"][0] == "done"   # "mark up to date"
        t_old, t_new = os.path.getmtime(e.data_h5), time.time() + 30   # an input rewritten later (attributes added)
        os.utime(e.data_h5, (t_new, t_new))
        assert "input changed" in status(e)["label_template"][1]
        assert set(accept(e)) == {"label_template", "label_build"} and status(e)["label_build"][0] == "done", status(e)
        os.utime(e.data_h5, (t_old, t_old))
        # the fingerprint ignores how a path is written and what does not change a result
        assert _hash({"p": "D:/Data/Run", "nb_proc": 2, "workers": 6}) == _hash({"p": "d:\\data\\run"})
        write_record(e, "label_template", {"status": "failed", "start": time.time(), "end": time.time(), "exit_code": 2})
        assert status(e)["label_template"][0] == "failed"
        write_record(e, "label_template", {"status": "running", "start": time.time(), "pid": os.getpid()})
        st = status(e)
        assert st["label_template"][0] == "running" and st["label_build"][0] == "blocked" and "waits for" in st["label_build"][1]
        write_record(e, "label_template", {"status": "running", "start": time.time(), "pid": 999999})
        assert status(e)["label_template"][0] == "failed"           # dead process without an end
        os.remove(record_path(e, "label_template"))
        assert pid_alive(os.getpid()) and not pid_alive(999999)
        assert any("gray" in w for w in check(e)[1])                # 1 of 2 cases gray
        # validation linked to the training
        os.makedirs(os.path.join(base, "DOE_V", "0", "1DOF_150Hz"))
        with h5py.File(os.path.join(base, "DOE_V", "doe_results.h5"), "w") as f:
            g = f.create_group("case_000")
            g.attrs.update({"$spin_rate$": 12000.0, "$dxl_size$": 4e-4, "$nb_dt_rev$": 200.0, "$f_tooth$": 0.05,
                            "kappa": 1.5})
        yaml_save({"kind": "validation", "training": "train", "runs": [{"dir": os.path.join(base, "DOE_V")}],
                   "indicators": {"variants": "inherit"}}, exp_path("val"))
        v = load("val")   # older file: kind validation + training
        assert "validate" in stages(v) and v.ref.name == "train" and v.label["lim_sup_pct"] == 40 and v.label["out"].startswith(os.path.join(base, "DOE_V", "val"))
        assert v.reference == e.label["out"] and v.indicators["variants"] == ["v1"] and v.indicators["f_modal"] == 150.0
        st = status(v)
        assert st["simulate"][0] == "done" and st["extract"][0] == "done" and not stages(v)["extract"].runnable
        assert st["validate"][0] == "blocked" and next_step(v)[1] == "label_template", (st, next_step(v))
        errs, warns = check(v)
        assert any("discretisation" in x for x in errs) and any("kappa also in the reference" in w for w in warns)
        yaml_save({"reference": "train", "stages": FLOWS["Validation against a reference"],
                   "runs": [{"dir": os.path.join(base, "DOE_V")}]}, exp_path("val2"))
        v2 = load("val2")
        assert v2.flow == "Validation against a reference" and default_goal(v2) == "Indicator validation"
        assert list(stages(v2)) == FLOWS["Validation against a reference"] and v2.indicators["variants"] == ["v1"]
        # the training dataset disappears -> the validation indicators say what is missing and where
        os.rename(e.label["out"], e.label["out"] + ".bak")
        st = status(v)
        assert st["indicators"][0] == "blocked" and "of experiment 'train'" in st["indicators"][1]
        os.rename(e.label["out"] + ".bak", e.label["out"])
        # a validation that edits the label parameters is an error
        yaml_save({"kind": "validation", "training": "train", "runs": [{"dir": os.path.join(base, "DOE_V")}],
                   "label": {"lim_sup_pct": 50}}, exp_path("val_bad"))
        assert any("differs from reference" in x for x in check(load("val_bad"))[0])
        # duplicate keys refused, extends merges sections one level deep
        with open(exp_path("dup"), "w") as f:
            f.write("kind: training\nkind: validation\n")
        try:
            load("dup")
            raise AssertionError("duplicate key accepted")
        except ValueError as exc:
            assert "duplicate key" in str(exc)
        os.remove(exp_path("dup"))
        yaml_save({"extends": "train", "indicators": {"variants": ["v2"]}}, exp_path("child"))
        c = load("child")
        assert c.name == "child" and c.indicators["f_modal"] == 150.0
        assert any("not in the presets: ['v2']" in x for x in check(c)[0])
        # import: label params from the h5 attrs, baseline records -> nothing stale, extract read-only
        imp = os.path.join(base, "DOE_I")
        os.makedirs(os.path.join(imp, "0", "1DOF_150Hz"))
        with h5py.File(os.path.join(imp, "doe_results.h5"), "w") as f:
            f.create_group("case_000").attrs.update({"$spin_rate$": 12098.28, "kappa": 0.5})
        with h5py.File(os.path.join(imp, "reference_dataset_amp.h5"), "w") as f:
            p = f.create_group("stable/case_000").create_dataset("Axial_disp__000", data=[0.0])
            p.attrs.update(labeling_strategy="amplitude", labeling_lim_sup_pct=40.0, labeling_signal="Axial_disp")
        import_dir("imp", imp)
        i = load("imp")
        assert i.label["strategy"] == "amplitude" and i.label["lim_sup_pct"] == 40.0 and i.runs[0].case == "1DOF_150Hz"
        st = status(i)
        assert st["extract"][0] == "done" and st["label_build"][0] == "done" and "made without the earlier step" in st["label_build"][1], st
        assert st["label_template"][0] == "skipped" and "Label build" in st["label_template"][1], st
        assert "12098.28 rpm" in i.cfg["description"] and summary(i)["spin"] == 12098.28
        assert "progress" in status_text(i)
        try:
            import_dir("imp", imp)
            raise AssertionError("import over an existing experiment")
        except ValueError:
            pass
        # a folder simulated outside the app but never extracted: Simulate done + blocked, Extract runnable
        raw = os.path.join(base, "DOE_RAW")
        for i, ap in enumerate((0.004, 0.012)):
            c = os.path.join(raw, str(i), "1DOF_150Hz")
            os.makedirs(c)
            open(os.path.join(c, "sens_out.hdf5"), "w").close()
            with open(os.path.join(c, "var_val.py"), "w") as f:
                f.write(f"var_val = {{'$Ap_start$': {ap}, '$Ap_end$': {ap}, '$spin_rate$': 12000.0}}\n")
        try:
            import_dir("raw", raw, h5="doe_results.h5")
            raise AssertionError("import of a folder without .h5 accepted as extracted")
        except ValueError:
            pass
        import_dir("raw", raw, h5="", ap_ref={"mode": "manual", "manual": 0.008})
        rw = load("raw")
        st = status(rw)
        assert rw.runs[0].existing and rw.runs[0].n_cases == 2 and st["simulate"][0] == "done", st
        assert st["extract"][0] == "pending" and not run_blockers(rw, "extract"), (st, run_blockers(rw, "extract"))
        assert run_blockers(rw, "simulate") and "would delete" in run_blockers(rw, "simulate")[0]
        assert "existing" in resolved_yaml("raw")["runs"][0]
        # model: kept in the experiment, a metadata-only attribute keeps the file date, mismatch with the reference
        import h5py as _h5
        with _h5.File(os.path.join(raw, "doe_results.h5"), "w") as f:
            f.create_group("case_000").attrs["$Ap_start$"] = 0.004
        mt = os.path.getmtime(os.path.join(raw, "doe_results.h5"))
        add_case_attrs(os.path.join(raw, "doe_results.h5"), {"sim_model": "1DOF_150"})
        assert os.path.getmtime(os.path.join(raw, "doe_results.h5")) == mt
        assert set_run_model("raw", os.path.join(raw, "doe_results.h5"), "1DOF_150") and load("raw").runs[0].model == "1DOF_150"
        assert resolved_yaml("raw")["runs"][0]["model"] == "1DOF_150"
        os.remove(os.path.join(raw, "doe_results.h5"))
        yaml_save(dict(own_yaml("val2"), runs=[{"dir": os.path.join(base, "DOE_V"), "model": "2DOF_150_250"}]),
                  exp_path("val2"))
        set_run_model("train", load("train").runs[0].h5, "1DOF_150")
        assert any("simulated model" in x for x in check(load("val2"))[0])
        set_run_model("train", load("train").runs[0].h5, None)
        # Extract running: progress from doe_runner's log, the panel does not try to read the half-written file
        log = os.path.join(rw.runs_dir(), "extract.log")
        with open(log, "w") as f:
            f.write("[INFO] Casos encontrados: 2 | DOE dir: x\n[INFO] Caso 000 extraido | var_val={}\n")
        write_record(rw, "extract", {"status": "running", "start": time.time(), "pid": os.getpid(), "log": log})
        with open(rw.runs[0].h5, "wb") as f:
            f.write(b"not finished")                               # what a reader sees while it is written
        assert stage_progress(rw, "extract") == (1, 2) and stage_badge(rw, "extract") == "50 %"
        assert "being written" in stage_summary(rw, "extract")[0][0] and h5_info(rw.runs[0].h5) is None
        assert summary(rw)["cases"] == 2                            # the list does not fail either
        os.remove(record_path(rw, "extract"))
        os.remove(rw.runs[0].h5)
        import subprocess   # doe_runner itself finds the 2 cases with the generated config (dry-run: nothing written)
        r = subprocess.run([sys.executable, os.path.join(SIM, "doe_runner.py"), "--config", rw.runs[0].config,
                            "--command", "extract", "--dry-run"], capture_output=True, text=True)
        assert r.returncode == 0 and "Casos encontrados: 2" in r.stdout + r.stderr, r.stdout + r.stderr
        # ramps in a doe_results.h5: their 'kappa' (= the start, written by older extracts) is ignored
        rh5 = os.path.join(base, "ramps.h5")
        with h5py.File(rh5, "w") as f:
            f.create_group("case_000").attrs.update({"$Ap_start$": 0.005, "$Ap_end$": 0.015, "$spin_rate$": 12000.0,
                                                     "kappa": 0.58, "kappa_start": 0.58, "kappa_end": 1.74})
            f.create_group("case_001").attrs.update({"$Ap_start$": 0.008, "$Ap_end$": 0.008, "$spin_rate$": 12000.0,
                                                     "kappa": 0.93})
            f.create_group("case_002").attrs.update({"$Ap_start$": 0.012, "$Ap_end$": 0.006, "$spin_rate$": 12000.0})
        info = h5_info(rh5)
        assert info["kappa"] == [0.93] and info["ramps"] == 2 and kappa_span(info) == (0.58, 1.74), info
        assert "case_000: Ap 5 -> 15 mm, kappa 0.580 -> 1.740" in info["ramp_text"][0], info["ramp_text"]
        assert abs(ap_of_t(f_attrs := {"$Ap_start$": 0.005, "$Ap_end$": 0.015}, 7.5, (0.0, 15.0)) - 0.010) < 1e-12
        assert ap_of_t(f_attrs, [-1.0, 15.0, 20.0], (0.0, 15.0)).tolist() == [0.005, 0.015, 0.015]
        rep = inspect_h5(rh5)
        assert rep["ramps"] == ["case_000", "case_002"] and rep["missing"]["kappa"] == ["case_002"], rep["missing"]
        add_case_attrs(rh5, {"ap_ref": {"mode": "manual", "manual": 0.01}})
        with h5py.File(rh5, "r") as f:
            a2 = f["case_002"].attrs
            assert abs(a2["kappa_start"] - 1.2) < 1e-12 and abs(a2["kappa_end"] - 0.6) < 1e-12 and "kappa" not in a2
            assert abs(f["case_001"].attrs["kappa"] - 0.8) < 1e-12
        assert not inspect_h5(rh5)["missing"]["kappa"]
        os.remove(rh5)
        # labelling of the ramps: the window comes from the variants (only when the data have ramps: the constant
        # experiments keep their commands and fingerprints), the labels per case (mixed, t_onset), the summary
        assert variant_window({"indicator": "RMS_CV", "mode": "by_revolution", "params_physical": {
            "N_rev_window": 4, "step_rev": 1, "n_max_rev": 4}}) == ("by_revolution", 7.0, 1.0)
        assert default_window({}) == DEFAULT_WINDOW and default_window(
            {"a": {"indicator": "MaxEnt_SPRT", "mode": "by_modal", "params_physical": {"N_modal_window": 2, "step_modal": 1}},
             "b": {"indicator": "MaxEnt_SPRT", "mode": "by_revolution", "params_physical": {"N_rev_window": 7, "step_rev": 1}}},
            12000, 150) == {"window_mode": "by_revolution", "window_N": 7.0, "window_step": 1.0}   # 35 ms > 13 ms
        rdir = os.path.join(base, "DOE_RMP")
        os.makedirs(os.path.join(rdir, "0", "1DOF_150Hz"))
        with h5py.File(os.path.join(rdir, "doe_results.h5"), "w") as f:
            f.create_group("case_000").attrs.update({"$Ap_start$": 0.004, "$Ap_end$": 0.012, "$spin_rate$": 12000.0,
                                                     "$dxl_size$": 2e-4, "$nb_dt_rev$": 200.0, "$f_tooth$": 0.05,
                                                     "kappa_start": 0.5, "kappa_end": 1.5})
            f.create_group("case_001").attrs.update({"$Ap_start$": 0.004, "$Ap_end$": 0.004, "$spin_rate$": 12000.0,
                                                     "$dxl_size$": 2e-4, "$nb_dt_rev$": 200.0, "$f_tooth$": 0.05,
                                                     "kappa": 0.5})
        import_dir("rmp", rdir, reference="train")
        rp = load("rmp")
        assert rp.has_ramps() and not load("train").has_ramps()
        assert rp.label["window_N"] == 4.0 and rp.label["window_step"] == 1.0 and "window_from" in rp.label, rp.label
        tpl_r = stages(rp)["label_template"].cmds[0]
        assert tpl_r[tpl_r.index("--window-N") + 1] == "4.0" and "--window-N" not in stages(load("train"))["label_template"].cmds[0]
        os.makedirs(os.path.dirname(rp.label["out"]), exist_ok=True)
        with h5py.File(rp.label["out"], "w") as f:
            for lab, case, t0, t1 in (("stable", "case_000", 0.0, 5.0), ("gray", "case_000", 5.0, 5.5),
                                      ("unstable", "case_000", 5.5, 10.0), ("stable", "case_001", 0.0, 10.0)):
                p = f.require_group(f"{lab}/{case}").create_dataset("Axial_disp__000", data=[0.0])
                p.attrs.update({"channel": "Axial_disp", "t0": t0, "t1": t1, "$Ap_start$": 0.004, "kappa": 0.5,
                                "$Ap_end$": 0.012 if case == "case_000" else 0.004})
        li = label_info(rp.label["out"])
        assert li["case_000"]["label"] == "mixed" and li["case_000"]["t_onset"] == 5.5 and li["case_000"]["ramp"]
        assert li["case_000"]["kappa"] != li["case_000"]["kappa"] and li["case_001"]["label"] == "stable"
        assert li["case_001"]["kappa"] == 0.5 and _label_cases(rp.label["out"])["case_000"][0] == "mixed"
        summ = [t for t, _ in stage_summary(rp, "label_build")]
        assert summ[0] == "ramps 1: crosses at t ≈ 5.50 s" and "ramp case_000: stable -> gray at 5.00 s -> unstable at 5.50 s" in summ[1], summ
        assert any(t.startswith("stable 1   (constant cases)") for t in summ) and not any("boundary" in t for t in summ), summ
        assert stage_badge(rp, "label_build") in ("", "S 1 · M 1")
        assert case_label([(0, 1, "gray")]) == "gray" and case_label([]) == "none" and transitions_text([(0, 9, "stable")]) == "stable all along"
        delete("rmp")
        _selftest_run(load("train"))
        _selftest_edit(root, cfg)
        print("experiment selftest OK")
    finally:
        set_root(old[0])
        shutil.rmtree(root, ignore_errors=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="DOE experiments: status, checks, import.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status")
    s.add_argument("exp", nargs="?")
    s.add_argument("--goal", choices=list(GOALS))
    sub.add_parser("check").add_argument("exp")
    sub.add_parser("dryrun", help="what would run (cases, folders, commands); nothing runs").add_argument("exp")
    ac = sub.add_parser("accept", help="mark stale stages as up to date (configuration change without effect)")
    ac.add_argument("exp")
    ac.add_argument("stages", nargs="*", choices=STAGES + ("",), default=[])
    im = sub.add_parser("import")
    im.add_argument("name")
    im.add_argument("dir")
    im.add_argument("--reference", help="experiment whose labelled dataset trains the indicators (turns on validate)")
    im.add_argument("--label-out")
    im.add_argument("--h5", default="doe_results.h5", help="signals file of the folder")
    im.add_argument("--not-extracted", action="store_true",
                    help="the folder was simulated but has no .h5 yet: rebuild its simulation so Extract can run")
    im.add_argument("--ap-ref", default=None, metavar="MODE[:VALUE]",
                    help="with --not-extracted, for kappa: manual:<m>, model:<preset> or model_at_spin:<preset>")
    im.add_argument("--description")
    r = sub.add_parser("run", help="run one stage in this console (what the app opens)")
    r.add_argument("exp")
    r.add_argument("stage", choices=STAGES)
    r.add_argument("--yes", action="store_true", help="do not ask before replacing existing outputs")
    r.add_argument("--pause-on-error", action="store_true", help="wait for Enter if the stage fails (console closes otherwise)")
    ch = sub.add_parser("chain", help="run the next steps of the goal one after the other (what 'Run to goal' opens)")
    ch.add_argument("exp")
    ch.add_argument("--goal", choices=list(GOALS))
    ch.add_argument("--no-review", action="store_true", help="do not stop after Label template")
    ch.add_argument("--pause-on-error", action="store_true")
    sub.add_parser("selftest")
    a = p.parse_args()
    if a.cmd == "selftest":
        return _selftest()
    if a.cmd == "chain":
        code = run_chain(a.exp, a.goal, not a.no_review)
        if code and a.pause_on_error:
            input("\n[experiment] the chain stopped: read the messages above, then press Enter to close ")
        sys.exit(code)
    if a.cmd == "run":
        code = run_stage(a.exp, a.stage, a.yes)
        if code and a.pause_on_error:
            input("\n[experiment] the stage failed: read the messages above, then press Enter to close ")
        sys.exit(code)
    if a.cmd == "status":
        names = [a.exp] if a.exp else list_experiments()
        print("\n\n".join(status_text(load(n), a.goal) for n in names) or "no experiments in " + EXP_DIR)
    elif a.cmd == "check":
        errs, warns = check(load(a.exp))
        print("\n".join([f"ERROR: {x}" for x in errs] + [f"warning: {x}" for x in warns]) or "OK")
        sys.exit(1 if errs else 0)
    elif a.cmd == "dryrun":
        print("\n".join(t for t, _ in dry_run(load(a.exp))))
    elif a.cmd == "accept":
        print("marked up to date:", accept(load(a.exp), a.stages or None) or "nothing (no stale configuration)")
    elif a.cmd == "import":
        ap = None
        if a.ap_ref:
            mode, _, val = a.ap_ref.partition(":")
            ap = {"mode": mode, **({"manual": float(val)} if mode == "manual" else {"model": val} if val else {})}
        print("written:", import_dir(a.name, a.dir, a.reference, a.label_out, a.description,
                                     "" if a.not_extracted else a.h5, ap))


if __name__ == "__main__":
    main()
