#!/usr/bin/env python
# coding: utf-8
"""experiment.py — DOE experiments: one YAML per experiment, stage registry, status, goals and checks.

An experiment (experiments/<name>.yaml) ties together the DOE runs (configs/*.yaml to simulate, or already
simulated folders), the labelling, the indicator variants and the validation. Every stage knows what it needs,
what it writes and the command that runs it; the status of a stage comes from its output files plus the run
records in experiments/.runs/<experiment>/<stage>.json. See PLAN_app_experimentos.md.

CLI (entorno_CAMP10 Python):
    python experiment.py status [EXP]           stage table + next step (all experiments without EXP)
    python experiment.py check EXP              configuration errors / warnings
    python experiment.py import NAME DIR [--kind training|validation] [--training EXP] [--label-out FILE]
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
GOALS = {"Simulated data": ["@data"], "Training dataset": ["label_build"], "Indicators computed": ["indicators"],
         "Indicator validation": ["validate"], "Noise robustness": ["noise_indicators"], "Model SNR": ["model_snr"],
         "Static deflection": ["static_deflection"]}
DEFAULT_GOAL = {"training": "Training dataset", "validation": "Indicator validation"}
LABEL_PARAMS = ("strategy", "amp_signal", "base_attr", "base_scale", "lim_inf_pct", "lim_sup_pct", "warmup",
                "kappa_threshold", "t_start", "t_end", "channels")
LABEL_FLAGS = {"amp_signal": "--amp-signal", "base_attr": "--base-attr", "base_scale": "--base-scale",
               "lim_inf_pct": "--lim-inf-pct", "lim_sup_pct": "--lim-sup-pct", "warmup": "--warmup",
               "kappa_threshold": "--kappa-threshold", "t_start": "--t-start", "t_end": "--t-end"}
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
    """{'defaults': {...}, 'variants': {...}, ...} or {} when the library file does not exist yet."""
    return yaml_load(VARIANTS_FILE) if os.path.isfile(VARIANTS_FILE) else {}


def _hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _mtime(p: str) -> float:
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0.0


@lru_cache(maxsize=256)
def _h5_info(path: str, mtime: float) -> dict:
    """Cases, first-case attrs and kappa list of a doe_results.h5 (cached by mtime)."""
    import h5py
    with h5py.File(path, "r") as f:
        cases = sorted(k for k in f if k.startswith("case_"))
        first = {k: (v.item() if hasattr(v, "item") else v) for k, v in f[cases[0]].attrs.items()} if cases else {}
        kappa = [float(f[c].attrs["kappa"]) for c in cases if "kappa" in f[c].attrs]
        deflex = bool(cases) and "Out_Deflex" in f[cases[0]]
        signals, duration = [], None
        if cases:
            g = f[cases[0]]
            signals = [s for s in g if isinstance(g[s], h5py.Group) and "time" in g[s]]
            if signals:
                duration = float(g[signals[0]]["time"][-1])
        groups = sorted(k for k in f if not k.startswith("case_"))   # control / snr_* in noise files
    return dict(cases=cases, first=first, kappa=kappa, deflex=deflex, signals=signals, duration=duration,
                groups=groups)


def h5_info(path: str):
    return _h5_info(path, _mtime(path)) if os.path.isfile(path) else None


def _doe_runner():
    if SIM not in sys.path:
        sys.path.insert(0, SIM)
    import doe_runner
    return doe_runner


# ============================================================================== experiment
class Run:
    """One DOE run of an experiment: a config of configs/ (can be simulated) or an imported folder (read-only)."""

    def __init__(self, entry: dict):
        self.entry, self.config, self.cfg = entry, None, None
        if "config" in entry:
            dr = _doe_runner()
            self.config = dr.find_config(str(entry["config"]))
            self.cfg = dr.load_config(self.config)
            self.base_dir, self.doe_name = self.cfg["base_dir"], self.cfg["doe_name"]
            self.case = self.cfg.get("case") or "1DOF_150Hz"
            self.n_cases = len(dr._with_config(self.cfg, lambda: dr.build_doe_cases(dr.DOE_MODE))[1])
        elif "dir" in entry:
            d = os.path.normpath(str(entry["dir"]))
            self.base_dir, self.doe_name = os.path.dirname(d), os.path.basename(d)
            self.case = entry.get("case") or detect_case(d)
            self.n_cases = None
        else:
            raise ValueError(f"run entry needs 'config' or 'dir': {entry}")
        self.doe_dir = os.path.join(self.base_dir, self.doe_name)
        self.h5 = os.path.join(self.doe_dir, "doe_results.h5")
        self.imported = self.config is None

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


def load(name: str) -> "Exp":
    """Load (cached per process; call reload() after editing a YAML)."""
    path = exp_path(name)
    key = (path, _mtime(path))
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
        self.kind = self.cfg.get("kind", "training")
        self.errors: list = []
        self.runs = []
        for r in self.cfg.get("runs") or []:
            try:
                self.runs.append(Run(r))
            except Exception as exc:   # bad config: reported by check(), the rest still loads
                self.errors.append(f"run {r}: {exc}")
        self.training = None
        if self.kind == "validation":
            if not self.cfg.get("training"):
                self.errors.append("validation experiment without 'training'")
            else:
                try:
                    self.training = load(self.cfg["training"])
                except Exception as exc:
                    self.errors.append(f"training '{self.cfg['training']}': {exc}")
        merge = self.cfg.get("merge") or {}
        self.merge = merge if len(self.runs) > 1 else {}
        if len(self.runs) > 1 and not merge.get("into"):
            self.errors.append("several runs need 'merge: {into: <new folder name>}'")
        if self.runs:
            r0 = self.runs[0]
            self.data_dir = os.path.join(r0.base_dir, self.merge["into"]) if self.merge.get("into") else r0.doe_dir
        else:
            self.data_dir = ""
            self.errors.append("no runs")
        self.data_h5 = os.path.join(self.data_dir, "doe_results.h5") if self.data_dir else ""
        self.out_dir = os.path.join(self.data_dir, self.name) if self.data_dir else ""
        self.label = self._label()
        self.indicators = self._indicators()

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
        if self.training is None:
            lab = {k: own[k] for k in LABEL_PARAMS if k in own}
        else:   # validation: parameters come from the training, never edited here
            lab = {k: self.training.label[k] for k in LABEL_PARAMS if k in self.training.label}
            diff = [k for k in LABEL_PARAMS if k in own and own[k] != lab.get(k)]
            if diff:
                self.errors.append(f"label {diff} differs from training '{self.training.name}' "
                                   f"(validation labels must use the training parameters)")
        lab.setdefault("strategy", "amplitude")
        short = STRATEGY_SHORT.get(lab["strategy"], lab["strategy"])
        lab["labels_yaml"] = self.out("label", "labels_yaml", "reference_labels.yaml")
        lab["out"] = self.out("label", "out", f"reference_dataset_{short}.h5")
        return lab

    def _indicators(self) -> dict:
        ind = self.section("indicators")
        if self.training is not None:
            base = self.training.indicators
            if ind.get("variants", "inherit") == "inherit":
                ind["variants"] = list(base.get("variants") or [])
            for k in ("f_modal", "workers"):
                ind.setdefault(k, base.get(k))
        ind.setdefault("variants", [])
        ind.setdefault("cases", "all")
        ind["out"] = self.out("indicators", "out", "doe_indicator_results.h5")
        return ind

    @property
    def reference(self) -> str:
        """Training dataset the indicators learn from (own for training, the linked one for validation)."""
        return (self.training or self).label["out"]

    @property
    def data_stage(self) -> str:
        return "merge" if self.merge else "extract"

    def runs_dir(self) -> str:
        return os.path.join(RUNS_DIR, self.name)


# ============================================================================== stages
class Stage:
    def __init__(self, exp: Exp, key: str, deps, inputs, outputs, cmds, section, runnable=True, why_not="",
                 present=None, writes=None):
        self.exp, self.key, self.title = exp, key, TITLES[key]
        self.deps = deps                    # [(Exp, stage key)]
        self.inputs, self.outputs = inputs, outputs
        self.cmds = cmds                    # [[argv...]] run in order, cwd = folder of the script
        self.hash = _hash(section)
        self.runnable, self.why_not = runnable, why_not
        self.present = present or (lambda: bool(outputs) and all(os.path.exists(p) for p in outputs))
        self.writes = writes if writes is not None else list(outputs)
        self.optional = key in OPTIONAL


def _py(script: str, *args) -> list:
    return [script, *[str(a) for a in args]]


def stages(exp: Exp) -> dict:
    """{key: Stage} for the stages that apply to this experiment, in topological order."""
    S, dr_script = {}, os.path.join(SIM, "doe_runner.py")
    runs, data, label, ind = exp.runs, exp.data_h5, exp.label, exp.indicators
    ref_exp = exp.training or exp
    if not runs:
        return S
    cfg_runs = [r for r in runs if not r.imported]
    imported_why = "imported run (no config): re-running it from the app is not possible"

    S["simulate"] = Stage(exp, "simulate", [], [r.config for r in cfg_runs], [r.doe_dir for r in runs],
                          [_py(dr_script, "--config", r.config, "--yes", "--command", "n2m_sch") for r in cfg_runs],
                          [r.cfg or r.entry for r in runs], runnable=not any(r.imported for r in runs),
                          why_not=imported_why,
                          present=lambda: all(r.imported and os.path.isdir(r.doe_dir) or
                                              (r.n_cases or 0) > 0 and r.simulated() >= r.n_cases for r in runs))
    S["extract"] = Stage(exp, "extract", [(exp, "simulate")], [], [r.h5 for r in runs],
                         [_py(dr_script, "--config", r.config, "--yes", "--command", "extract") for r in cfg_runs],
                         [r.cfg or r.entry for r in runs], runnable=not any(r.imported for r in runs),
                         why_not=imported_why)
    if exp.merge:
        r0, into = runs[0], exp.merge["into"]
        base_flags = ["--config", r0.config] if r0.config else ["--case_dir", os.path.join(r0.base_dir, r0.case)]
        cmds = [_py(dr_script, *base_flags, "--yes", "--command", "merge", "--doe_name", r0.doe_name,
                    "--merge_from", runs[1].doe_name, "--merge_out", into)]
        cmds += [_py(dr_script, *base_flags, "--yes", "--command", "merge", "--doe_name", into,
                     "--merge_from", r.doe_name) for r in runs[2:]]   # later runs go into our own new folder
        S["merge"] = Stage(exp, "merge", [(exp, "extract")], [r.h5 for r in runs], [data], cmds, exp.merge)
    d = exp.data_stage
    lab_params = {k: v for k, v in label.items() if k in LABEL_PARAMS}
    tpl = [os.path.join(SIM, "reference_dataset.py"), "template", data, label["labels_yaml"],
           "--strategy", label["strategy"]]
    for k, flag in LABEL_FLAGS.items():
        if label.get(k) is not None:
            tpl += [flag, str(label[k])]
    S["label_template"] = Stage(exp, "label_template", [(exp, d)], [data], [label["labels_yaml"]], [tpl],
                                {**lab_params, "data": data})
    build = [os.path.join(SIM, "reference_dataset.py"), "build", data, label["labels_yaml"], label["out"]]
    if label.get("channels"):
        build += ["--channels", *map(str, label["channels"])]
    for k in ("t_start", "t_end"):
        if label.get(k) is not None:
            build += [LABEL_FLAGS[k], str(label[k])]
    S["label_build"] = Stage(exp, "label_build", [(exp, "label_template")], [data, label["labels_yaml"]],
                             [label["out"]], [build], {**lab_params, "data": data})
    lib = variants_library().get("variants") or {}
    ind_section = {**{k: v for k, v in ind.items() if k != "out"}, "reference": exp.reference,
                   "variants_def": {v: lib.get(v) for v in ind["variants"]}}
    S["indicators"] = Stage(exp, "indicators", [(exp, d), (ref_exp, "label_build")], [data, exp.reference],
                            [ind["out"]], [_py(os.path.join(ANA, "doe_indicators.py"), "--experiment", exp.path)],
                            ind_section)
    if exp.kind == "validation":
        val = exp.section("validate")
        val_out = exp.out("validate", "out", "doe_validation_results.h5")
        cmd = _py(os.path.join(ANA, "validate_indicators.py"), "--ind_results", ind["out"], "--labels", label["out"],
                  "--reference", exp.reference, "--out", val_out, "--channel", val.get("channel", "Axial_disp"))
        S["validate"] = Stage(exp, "validate", [(exp, "label_build"), (exp, "indicators")],
                              [ind["out"], label["out"]], [val_out], [cmd], {**val, "out": val_out})
    # optional branches (configured by their YAML section only)
    sd = exp.section("static_deflection")
    S["static_deflection"] = Stage(exp, "static_deflection", [(exp, d)], [], [data],
                                   [_py(os.path.join(SIM, "static_deflection.py"), data, "--experiment", exp.path)],
                                   sd, present=lambda: bool((h5_info(data) or {}).get("deflex")), writes=[data])
    noise_out = exp.out("noise", "out", "doe_noise_results.h5", shared=True)
    S["noise"] = Stage(exp, "noise", [(exp, d)], [data], [noise_out],
                       [_py(os.path.join(SIM, "doe_noise.py"), "--doe_results", data, "--out", noise_out,
                            "--experiment", exp.path)], exp.section("noise"))
    ni_out = exp.out("noise_indicators", "out", "doe_noise_indicator_results.h5")
    S["noise_indicators"] = Stage(exp, "noise_indicators", [(exp, "noise"), (ref_exp, "label_build")],
                                  [noise_out, exp.reference], [ni_out],
                                  [_py(os.path.join(ANA, "doe_indicators.py"), "--experiment", exp.path,
                                       "--doe_results", noise_out, "--out", ni_out)],
                                  {**ind_section, **exp.section("noise_indicators")})
    snr_out = exp.out("model_snr", "out", "doe_model_snr_results.h5", shared=True)
    r0 = runs[0]
    S["model_snr"] = Stage(exp, "model_snr", [(exp, "simulate")], [], [snr_out],
                           [_py(os.path.join(ANA, "doe_model_snr.py"), "--experiment", exp.path,
                                "--doe_name", r0.doe_name, "--out", snr_out)], exp.section("model_snr"))
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
    """{key: (state, reason)}; state in done / stale / running / failed / pending / blocked."""
    memo = {} if _memo is None else _memo
    if exp.name in memo:
        return memo[exp.name]
    memo[exp.name] = out = {}
    S = stages(exp)
    for key, st in S.items():
        out[key] = _stage_state(exp, st, S, out, memo)
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
        if s in ("pending", "blocked", "failed", "running"):
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
    ref = rec.get("start") if rec else min((_mtime(p) for p in st.outputs if os.path.exists(p)), default=0)
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


def next_step(exp: Exp, goal: str | None = None):
    """(Exp, key, state, reason) of the first stage of the goal that is not done; None when the goal is reached."""
    goal = goal or DEFAULT_GOAL.get(exp.kind, "Training dataset")
    memo: dict = {}
    S = stages(exp)
    todo, seen = [], set()

    def need(e, k):   # post-order from the targets; a stage that is done covers everything before it
        if (e.name, k) in seen or k not in stages(e):
            return
        seen.add((e.name, k))
        state, reason = status(e, memo)[k]
        if state == "done":
            return
        for de, dk in stages(e)[k].deps:
            need(de, dk)
        todo.append((e, k, state, reason))

    for t in GOALS[goal]:
        t = exp.data_stage if t == "@data" else t
        if t in S:
            need(exp, t)
    return todo[0] if todo else None


# ============================================================================== checks
def check(exp: Exp) -> tuple:
    """(errors, warnings) of the whole experiment configuration."""
    errs, warns = list(exp.errors), []
    for r in exp.runs:
        if r.imported and not os.path.isdir(r.doe_dir):
            errs.append(f"imported run folder not found: {r.doe_dir}")
    if len({r.base_dir for r in exp.runs}) > 1 and exp.merge:
        errs.append("merged runs must live in the same base_dir (doe_runner merges sibling folders)")
    lib_vars = None
    try:
        lib_vars = variants_library().get("variants") or {}
    except Exception as exc:
        errs.append(f"indicator_variants.yaml: {exc}")
    if lib_vars is not None:
        missing = [v for v in exp.indicators["variants"] if v not in lib_vars]
        if missing and lib_vars:
            errs.append(f"indicator variants not in the library: {missing}")
        elif missing:
            warns.append("indicator_variants.yaml does not exist yet")
    if exp.training is not None:
        mine, theirs = h5_info(exp.data_h5), h5_info(exp.training.data_h5)
        if mine and theirs:
            diff = [k for k in DISCRETISATION if k in mine["first"] and k in theirs["first"]
                    and abs(float(mine["first"][k]) - float(theirs["first"][k])) > 1e-12]
            if diff:
                errs.append(f"discretisation differs from training: {diff}")
            n1, n2 = mine["first"].get("$spin_rate$"), theirs["first"].get("$spin_rate$")
            if n1 is not None and n2 is not None and abs(float(n1) - float(n2)) > 1e-6:
                warns.append(f"spin differs from training ({n1} vs {n2} rpm): generalisation test")
            rep = sorted({round(k, 6) for k in mine["kappa"]} & {round(k, 6) for k in theirs["kappa"]})
            if rep:
                warns.append(f"{len(rep)} kappa also in training: {rep[:5]}")
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
    kap = info["kappa"] if info else []
    return dict(spin=spin, cases=len(info["cases"]) if info else sum(r.n_cases or 0 for r in exp.runs),
                kappa=(min(kap), max(kap)) if kap else None, done=sum(st[k][0] == "done" for k in main),
                total=len(main))


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
    """INDICATOR_CONFIG dict of a library variant for one case (what doe_indicators passes to the runner)."""
    t_rev = 60.0 / float(spin) if spin else None
    t_modal = 1.0 / float(f_modal) if f_modal else None
    pp = {k: resolve_physics(v, t_rev, t_modal) for k, v in variant["params_physical"].items()}
    return {"id": variant["indicator"], "func": variant.get("func", "Default"), "param_mode": variant["mode"],
            "params_physical": pp}


def indicator_runs(exp: Exp) -> list:
    """RUNS for doe_indicators: one entry per selected variant, physics resolved per case (key 'variant')."""
    lib = variants_library().get("variants") or {}
    missing = [v for v in exp.indicators["variants"] if v not in lib]
    if missing:
        raise ValueError(f"variants not in {VARIANTS_FILE}: {missing}")
    return [{"enabled": True, "name": v, "signal": lib[v]["signal"], "variant": lib[v],
             "f_modal": exp.indicators.get("f_modal")} for v in exp.indicators["variants"]]


def analysis_cut() -> tuple:
    """(start, end) of the analysed signal; end None = end of each case."""
    c = variants_library().get("cut") or {}
    return float(c.get("start", 0.0)), (None if c.get("end") is None else float(c["end"]))


def resolve(exp: Exp, variant: str, case: str, h5: str | None = None) -> dict:
    """Final config of a variant for one case of the experiment data (experiment.py resolve / app preview)."""
    import h5py
    lib = variants_library().get("variants") or {}
    if variant not in lib:
        raise ValueError(f"variant '{variant}' not in the library")
    with h5py.File(h5 or exp.data_h5, "r") as f:
        if case not in f:
            raise ValueError(f"case '{case}' not in {h5 or exp.data_h5}")
        spin = f[case].attrs.get("$spin_rate$")
    cfg = indicator_config(lib[variant], spin, exp.indicators.get("f_modal"))
    return {"variant": variant, "case": case, "signal": lib[variant]["signal"], "spin_rate": spin,
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
                       "criterion: max|signal| against a % of the feed per tooth).",
                       "Review the YAML before building: these labels are the ground truth."),
    "label_build": ("Cuts the labelled signals into reference_dataset*.h5. For a training it is what the indicators "
                    "learn from; for a validation it is the ground truth.",
                    "Where the stable/unstable boundary falls and how many cases are gray."),
    "indicators": ("Runs the selected indicator variants on every case, trained on the training dataset "
                   "(T_rev from the spin of each case).",
                   "Every case x variant done, no errors; how many stable / unstable cases each variant flags."),
    "validate": ("Scores each variant against the validation labels: TP/FN/TN/FP per case, balanced accuracy, "
                 "MCC, AUC and detection times.",
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


def _label_cases(path: str) -> dict:
    """{case: (label, kappa)} of a reference_dataset*.h5 (one label per case: the first piece)."""
    import h5py
    out = {}
    if os.path.isfile(path):
        with h5py.File(path, "r") as f:
            for lab in f:
                for case, g in f[lab].items():
                    piece = next(iter(g.values()), None)
                    k = float(piece.attrs["kappa"]) if piece is not None and "kappa" in piece.attrs else float("nan")
                    out.setdefault(case, (lab, k))
    return out


def _yaml_labels(path: str) -> dict:
    """{case: label} of a reference_labels.yaml (label of its first interval)."""
    if not os.path.isfile(path):
        return {}
    d = yaml_load(path).get("cases") or {}
    return {c: (iv[0][2] if iv else "none") for c, iv in d.items()}


def stage_progress(exp: Exp, key: str):
    """(done, total) while a stage advances, else None: simulated cases, or '[k/N] completado' of the log."""
    if key == "simulate":
        tot = sum(r.n_cases or 0 for r in exp.runs if not r.imported)
        return (sum(min(r.simulated(), r.n_cases or 0) for r in exp.runs if not r.imported), tot) if tot else None
    rec = read_record(exp, key)
    log = rec.get("log") if rec else None
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
                        f"{float(a.get('$spin_rate$', float('nan'))):.6g} rpm", "ok"))
            out.append((f"signals {', '.join(info['signals']) or '-'}; {info['duration']:.2f} s per case" if
                        info["duration"] else f"signals {', '.join(info['signals']) or '-'}", None))
            disc = ", ".join(f"{k.strip('$')} {a[k]:g}" for k in DISCRETISATION if k in a)
            if disc:
                out.append((disc, None))
        return out
    if key in ("label_template", "label_build"):
        built, yml = _label_cases(exp.label["out"]), _yaml_labels(exp.label["labels_yaml"])
        src = built if key == "label_build" else {c: (lab, built.get(c, (None, float("nan")))[1]) for c, lab in yml.items()}
        if not src:
            return [("nothing yet", None)]
        by = {}
        for c, (lab, k) in src.items():
            by.setdefault(lab, []).append(k)
        order = [lab for lab in ("stable", "gray", "unstable") if lab in by] + [l for l in by if l not in ("stable", "gray", "unstable")]
        out.append(("  ·  ".join(f"{lab} {len(by[lab])}" for lab in order), "ok"))
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
                        f"AUC {f2(d.get('AUC'))}  TP {d.get('TP')} FN {d.get('FN')} TN {d.get('TN')} FP {d.get('FP')}",
                        "ok" if i == 1 else None))
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
            return " · ".join(f"{lab[0].upper()} {by[lab]}" for lab in ("stable", "gray", "unstable") if lab in by)
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


def _save_library(lib: dict) -> None:
    """Write the variant library keeping its header comment."""
    import yaml
    head = []
    if os.path.isfile(VARIANTS_FILE):
        with open(VARIANTS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.startswith("#"):
                    break
                head.append(line)
    tmp = VARIANTS_FILE + ".tmp"
    os.makedirs(os.path.dirname(VARIANTS_FILE), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("".join(head) + yaml.safe_dump(lib, sort_keys=False, allow_unicode=True))
    os.replace(tmp, VARIANTS_FILE)


def variant_used(variant: str) -> list:
    """Experiments whose indicator results already contain this variant (then it must not be edited)."""
    import h5py
    out = []
    for name in list_experiments():
        try:
            e = load(name)
        except Exception:
            continue
        for path in (e.indicators["out"], e.out("noise_indicators", "out", "doe_noise_indicator_results.h5")):
            if not os.path.isfile(path):
                continue
            with h5py.File(path, "r") as f:
                if any(isinstance(f[c], h5py.Group) and variant in f[c] for c in f):
                    out.append(name)
                    break
    return out


def propose_variant_name(spec: dict) -> str:
    """Same pattern as doe_indicators._run_name, made unique against the library."""
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
    names = set((variants_library().get("variants") or {}))
    name, i = base, 2
    while name in names:
        name, i = f"{base}_v{i}", i + 1
    return name


def save_variant(name: str, spec: dict, new: bool) -> None:
    """Add a variant (new=True) or edit one that has no results yet. Used variants are duplicated, never edited."""
    for k in ("indicator", "mode", "signal", "params_physical"):
        if k not in spec:
            raise ValueError(f"variant needs '{k}'")
    lib = variants_library() or {"cut": {"start": 0.05, "end": None}, "variants": {}}
    lib.setdefault("variants", {})
    if new and name in lib["variants"]:
        raise ValueError(f"variant '{name}' already exists: choose another name")
    if not new:
        if name not in lib["variants"]:
            raise ValueError(f"variant '{name}' does not exist")
        used = variant_used(name)
        if used:
            raise ValueError(f"variant '{name}' already has results in {used}: duplicate it instead of editing")
    lib["variants"][name] = spec
    _save_library(lib)


def propose_name(kind: str, case: str, spin, kappas) -> str:
    """<train|val>_<machine>_n<rpm>_k<min>-<max>, unique among the experiments."""
    machine = "".join(ch for ch in (case or "doe") if ch.isalnum()).replace("Hz", "")
    parts = ["train" if kind == "training" else "val", machine]
    if spin:
        parts.append(f"n{float(spin):.0f}")
    if kappas:
        parts.append(f"k{round(min(kappas), 2):g}-{round(max(kappas), 2):g}")
    base, existing = "_".join(parts), set(list_experiments())
    name, i = base, 2
    while name in existing:
        name, i = f"{base}_{i}", i + 1
    return name


def list_configs() -> list:
    return sorted(os.path.splitext(f)[0] for f in os.listdir(CONFIGS_DIR)
                  if f.endswith(".yaml") and f != "base.yaml") if os.path.isdir(CONFIGS_DIR) else []


def new_doe_config(template: str, doe_name: str, spin: float, kappas=None, aps=None, ap_ref: float | None = None,
                   header: str = "") -> str:
    """configs/<doe_name>.yaml from a template config: only Ap (from kappa x ap_ref, or given), spin and doe_name
    change. Validated with doe_runner before returning; an invalid file is removed."""
    dr = _doe_runner()
    tpl_path = dr.find_config(template)
    raw = yaml_load(tpl_path)
    full = dr.load_config(tpl_path)
    out = os.path.join(CONFIGS_DIR, doe_name + ".yaml")
    if os.path.exists(out):
        raise ValueError(f"config '{doe_name}' already exists")
    if ap_ref is None:
        a = full.get("ap_ref") or {}
        ap_ref = float(a["manual"]) if a.get("mode") == "manual" else None
    if aps is None:
        if not kappas or not ap_ref:
            raise ValueError("give the Ap list, or kappa + ap_ref (the template has no manual ap_ref)")
        aps = [round(float(k) * ap_ref, 9) for k in kappas]
    table = full.get("sweep") or full.get("factorial") or {}
    sweep = {"$Ap_start$": [float(a) for a in aps], "$Ap_end$": [float(a) for a in aps],
             "$spin_rate$": float(spin)}
    for k, v in table.items():   # other scalars of the template (f_tooth, dxl_size, nb_dt_rev...) stay
        if k not in sweep and len(set(v)) == 1:
            sweep[k] = v[0]
    d = {k: v for k, v in raw.items() if k not in ("sweep", "factorial", "manual", "mode", "doe_name", "ap_ref")}
    d.update(doe_name=doe_name, mode="sweep", sweep=sweep)
    if ap_ref:
        d["ap_ref"] = {"mode": "manual", "manual": float(ap_ref)}
    if "base_dir" not in raw and "base_dir" in full:
        d.setdefault("base_dir", full["base_dir"].replace("\\", "/"))
    import yaml
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# Generated by the experiments app from '{template}'{header}\n"
                + yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
    try:
        dr.load_config(out)
    except Exception:
        os.remove(out)
        raise
    return out


def create_experiment(name: str, kind: str, runs: list, training: str | None = None, description: str = "",
                      **sections) -> str:
    path = exp_path(name)
    if os.path.exists(path):
        raise ValueError(f"experiment '{name}' already exists")
    d = {"name": name, "kind": kind, "description": description, "runs": runs}
    if training:
        d["training"] = training
    d.update({k: v for k, v in sections.items() if v})
    yaml_save(d, path)
    reload()
    return path


def derive(parent: str, name: str, spin: float | None = None, variants=None, ap_ref_preset: str | None = None,
           description: str = "") -> str:
    """Child experiment (extends parent). With spin: a copy of each config run at that n (same kappa list;
    Ap recomputed with the SLD limit at that n when ap_ref_preset is given, else same Ap)."""
    p = load(parent)
    d = {"extends": parent, "name": name, "description": description or f"derived from {parent}"}
    if spin is not None:
        if any(r.imported for r in p.runs):
            raise ValueError("the parent has imported runs (no config): create a New DOE instead")
        runs = []
        for r in p.runs:
            doe = f"{r.doe_name}_n{float(spin):.0f}"
            ap_ref = None
            if ap_ref_preset:
                if PLOTS not in sys.path:
                    sys.path.insert(0, PLOTS)
                import sld_model
                ap_ref = sld_model.ap_lim(ap_ref_preset, float(spin)) / 1e3
            src = (r.cfg.get("sweep") or {}).get("$Ap_start$")
            a = r.cfg.get("ap_ref") or {}
            old_ref = float(a["manual"]) if a.get("mode") == "manual" else None
            kappas = [x / old_ref for x in src] if (ap_ref and old_ref and src) else None
            new_doe_config(r.config, doe, spin, kappas=kappas, aps=None if kappas else src, ap_ref=ap_ref,
                           header=f" (derived experiment {name}, n = {float(spin):g} rpm)")
            runs.append({"config": doe})
        d["runs"] = runs
    if variants is not None:
        d["indicators"] = {"variants": list(variants)}
    if os.path.exists(exp_path(name)):
        raise ValueError(f"experiment '{name}' already exists")
    yaml_save(d, exp_path(name))
    reload()
    return exp_path(name)


def duplicate(src: str, name: str) -> str:
    if os.path.exists(exp_path(name)):
        raise ValueError(f"experiment '{name}' already exists")
    d = own_yaml(src)
    d["name"] = name
    yaml_save(d, exp_path(name))
    reload()
    return exp_path(name)


# what a new training experiment starts with (same values as the current training dataset)
LABEL_DEFAULTS = {"strategy": "amplitude", "amp_signal": "Axial_disp", "base_attr": "$f_tooth$", "base_scale": 1.0e-3,
                  "lim_inf_pct": 10.0, "lim_sup_pct": 40.0, "warmup": 0.0}
INDICATOR_DEFAULTS = {"variants": ["maxent_revo_dec4_1step", "rms_cv_revo_aux4_n_aux4_dec7_1step",
                                   "ssq_revo_aux4_n_aux4_dec7_1step", "green_fixed_revo_dec4_1step"],
                      "f_modal": 150.0, "cases": "all", "workers": 6}
METRIC_COLUMNS = ("balanced_accuracy", "MCC", "AUC", "TPR", "TNR", "F1", "accuracy", "median_delay_onset_s",
                  "mean_alarm_fraction_stable", "mean_persistence")


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
    """Experiments that extend this one or use it as training."""
    out = []
    for n in list_experiments():
        if n == name:
            continue
        try:
            d = own_yaml(n)
        except Exception:
            continue
        if d.get("extends") == name or d.get("training") == name:
            out.append(n)
    return out


def delete(name: str) -> None:
    """Remove the experiment YAML and its run records. Never touches data (.h5, DOE folders, configs)."""
    deps = dependents(name)
    if deps:
        raise ValueError(f"'{name}' is used by {deps}: delete or change those first")
    os.remove(exp_path(name))
    shutil.rmtree(os.path.join(RUNS_DIR, name), ignore_errors=True)
    reload()


# ============================================================================== import
_LABEL_ATTRS = {"labeling_strategy": "strategy", "labeling_signal": "amp_signal", "labeling_base_attr": "base_attr",
                "labeling_base_scale": "base_scale", "labeling_lim_inf_pct": "lim_inf_pct",
                "labeling_lim_sup_pct": "lim_sup_pct", "labeling_warmup": "warmup"}


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


def import_dir(name: str, doe_dir: str, kind: str = "training", training: str | None = None,
               label_out: str | None = None, description: str | None = None) -> str:
    """Create experiments/<name>.yaml from an already simulated DOE folder; existing outputs get an 'imported'
    baseline record so nothing shows as stale because of a history the app does not know."""
    doe_dir = os.path.normpath(os.path.abspath(doe_dir))
    path = exp_path(name)
    if os.path.exists(path):
        raise ValueError(f"experiment '{name}' already exists ({path})")
    if not os.path.isfile(os.path.join(doe_dir, "doe_results.h5")):
        raise ValueError(f"{doe_dir} has no doe_results.h5")
    d = {"name": name, "kind": kind, "description": description or "",
         "runs": [{"dir": doe_dir.replace("\\", "/"), "case": detect_case(doe_dir)}]}
    if training:
        d["training"] = training
    refs = sorted(f for f in os.listdir(doe_dir) if f.startswith("reference_dataset") and f.endswith(".h5"))
    if label_out:
        refs = [label_out]
    elif len(refs) > 1:   # prefer the one that says how it was labelled
        refs = [f for f in refs if _label_attrs(os.path.join(doe_dir, f)).get("strategy")][:1] or refs[:1]
    label = {}
    if refs and kind == "training":
        label = _label_attrs(os.path.join(doe_dir, refs[0]))
        label["out"] = os.path.join(doe_dir, refs[0]).replace("\\", "/")
        yml = os.path.join(doe_dir, "reference_labels.yaml")
        if os.path.isfile(yml):   # linked only if the dataset was built from it (same label for every case)
            built, ylab = _label_cases(os.path.join(doe_dir, refs[0])), _yaml_labels(yml)
            if built and all(ylab.get(c) == lab for c, (lab, _) in built.items()):
                label["labels_yaml"] = yml.replace("\\", "/")
        d["label"] = label
    ind_h5 = os.path.join(doe_dir, "doe_indicator_results.h5")
    if os.path.isfile(ind_h5):
        import h5py
        with h5py.File(ind_h5, "r") as f:
            c0 = next((f[k] for k in sorted(f) if k.startswith("case_")), None)
            variants = sorted(k for k in c0 if "t" in c0[k] and "I_t" in c0[k]) if c0 is not None else []
        d["indicators"] = {"variants": variants, "out": ind_h5.replace("\\", "/")}
    if not d["description"]:
        info = h5_info(os.path.join(doe_dir, "doe_results.h5"))
        spin = info["first"].get("$spin_rate$")
        kap = info["kappa"]
        d["description"] = (f"{kind}, imported from {os.path.basename(doe_dir)}: {len(info['cases'])} cases"
                            + (f", n = {float(spin):.10g} rpm" if spin is not None else "")
                            + (f", kappa {min(kap):.2f}-{max(kap):.2f}" if kap else ""))
    yaml_save(d, path)
    reload()
    exp = load(name)
    now = time.time()
    for key, st in stages(exp).items():
        if st.present():
            write_record(exp, key, {"stage": key, "status": "imported", "start": now, "end": now,
                                    "exit_code": 0, "hash": st.hash})
    return path


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


def stamp_outputs(exp: Exp, st: Stage) -> None:
    """Traceability attrs on the .h5 outputs of a successful run."""
    import h5py
    for p in st.outputs:
        if p.endswith(".h5") and os.path.isfile(p):
            try:
                with h5py.File(p, "a") as f:
                    f.attrs.update(experiment=exp.name, experiment_stage=st.key, experiment_hash=st.hash,
                                   experiment_date=datetime.datetime.now().isoformat(timespec="seconds"))
            except OSError as exc:   # traceability must never fail a finished run
                print(f"[experiment] could not stamp {p}: {exc}")


def run_stage(name: str, key: str, yes: bool = False, cmds=None) -> int:
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
    write_record(exp, key, rec)
    if code == 0:
        stamp_outputs(exp, st)
    print(f"[experiment] {key}: {'done' if code == 0 else f'FAILED (exit {code})'} - log: {log_path}")
    return code


def _selftest_edit(root: str, train_cfg: str) -> None:
    """Editing helpers on the selftest experiments; configs/ redirected to a temp folder."""
    global CONFIGS_DIR
    import h5py
    dr = _doe_runner()
    old = (CONFIGS_DIR, dr.CONFIGS_DIR)
    CONFIGS_DIR = dr.CONFIGS_DIR = os.path.join(root, "configs")
    os.makedirs(CONFIGS_DIR)
    try:
        save_section("train", "validate", {"channel": "Axial_vel"})
        assert own_yaml("train")["validate"] == {"channel": "Axial_vel"}
        save_section("train", "validate", None)
        assert "validate" not in own_yaml("train")
        # variants: proposed name, no duplicates, used ones are not edited
        spec = {"indicator": "MaxEnt_SPRT", "func": "Default", "mode": "by_revolution", "signal": "Axial_vel",
                "params_physical": {"T_rev": "T_rev", "N_rev_window": 4, "step_rev": 1}}
        name = propose_variant_name(spec)
        assert name == "maxent_revo_dec4_1step", name
        save_variant(name, spec, new=True)
        assert propose_variant_name(spec) == "maxent_revo_dec4_1step_v2"
        try:
            save_variant(name, spec, new=True)
            raise AssertionError("duplicate variant accepted")
        except ValueError:
            pass
        save_variant(name, dict(spec, signal="Axial_disp"), new=False)          # unused: editable
        e = load("train")
        with h5py.File(e.indicators["out"], "a") as f:
            f.require_group(f"case_000/{name}")
        assert variant_used(name) == ["train"]
        try:
            save_variant(name, spec, new=False)
            raise AssertionError("used variant edited")
        except ValueError as exc:
            assert "duplicate it" in str(exc)
        with open(VARIANTS_FILE, encoding="utf-8") as f:
            lib_text = f.read()
        assert "maxent_revo_dec4_1step" in lib_text
        # names
        n1 = propose_name("validation", "1DOF_150Hz", 12098.28, [0.528, 1.906])
        assert n1.startswith("val_1DOF150_n12098_k0.53-1.91"), n1
        # DOE config from a template: kappa x template ap_ref (0.008)
        p = new_doe_config(train_cfg, "DOE_N", 10000.0, kappas=[0.5, 1.5])
        c = dr.load_config(p)
        assert c["sweep"]["$Ap_start$"] == [0.004, 0.012] and c["sweep"]["$spin_rate$"] == [10000.0, 10000.0]
        try:
            new_doe_config(train_cfg, "DOE_N", 10000.0, kappas=[0.5])
            raise AssertionError("config overwritten")
        except ValueError:
            pass
        # derive another n (same Ap: no SLD preset given), then duplicate / delete
        yaml_save(dict(own_yaml("train"), runs=[{"config": train_cfg}]), exp_path("train"))
        reload()
        derive("train", "train_n10000", spin=10000.0)
        ch = load("train_n10000")
        assert ch.runs[0].doe_name == "DOE_T_n10000" and ch.label["lim_sup_pct"] == 40 and ch.kind == "training"
        assert ch.runs[0].cfg["sweep"]["$spin_rate$"][0] == 10000.0
        duplicate("train_n10000", "train_copy")
        assert load("train_copy").name == "train_copy"
        delete("train_copy")
        assert "train_copy" not in list_experiments()
        try:
            delete("train")
            raise AssertionError("deleted an experiment others depend on")
        except ValueError as exc:
            assert "used by" in str(exc)
        create_experiment("fresh", "training", [{"config": "DOE_N"}], description="x",
                          label={"strategy": "amplitude"})
        assert load("fresh").runs[0].n_cases == 2
    finally:
        CONFIGS_DIR, dr.CONFIGS_DIR = old


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
    assert status(e)["label_build"][0] == "done" and next_step(e) is None
    assert any(b.startswith("input missing") for b in run_blockers(e, "label_build"))
    os.rename(lab + ".hide", lab)
    out = e.indicators["out"]
    assert run_stage(e.name, "indicators", yes=True, cmds=[[ok, out]]) == 0
    rec = read_record(e, "indicators")
    assert rec["status"] == "done" and rec["exit_code"] == 0 and "hello from the stage" in open(rec["log"]).read()
    with h5py.File(out, "r") as f:
        assert f.attrs["experiment"] == e.name and f.attrs["experiment_stage"] == "indicators"
    assert status(e)["indicators"][0] == "done" and existing_outputs(e, "indicators") == [out]
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
SYMBOL = {"done": "OK ", "stale": "OLD", "running": "RUN", "failed": "ERR", "pending": " . ", "blocked": " x "}


def status_text(exp: Exp, goal: str | None = None) -> str:
    sm, st = summary(exp), status(exp)
    head = (f"{exp.name}  [{exp.kind}]  "
            + (f"n={float(sm['spin']):.10g} rpm  " if sm["spin"] is not None else "")
            + f"{sm['cases']} cases" + (f"  kappa {sm['kappa'][0]:.2f}-{sm['kappa'][1]:.2f}" if sm["kappa"] else "")
            + f"  progress {sm['done']}/{sm['total']}")
    lines = [head]
    for key, (state, reason) in st.items():
        lines.append(f"  [{SYMBOL[state]}] {key:<18s} {state:<8s} {reason}" + ("   (optional)" if key in OPTIONAL else ""))
    goal = goal or DEFAULT_GOAL.get(exp.kind)
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
        yaml_save({"variants": {"v1": {"indicator": "MaxEnt_SPRT"}}}, VARIANTS_FILE)
        e = load("train")
        assert e.kind == "training" and e.runs[0].n_cases == 2 and e.data_h5.endswith(os.path.join("DOE_T", "doe_results.h5"))
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
        assert status(e)["label_build"][0] == "done" and next_step(e) is None
        write_record(e, "label_template", {"status": "done", "start": time.time(), "hash": "old"})
        st = status(e)
        assert st["label_template"][0] == "stale" and st["label_build"][0] == "stale" and "upstream" in st["label_build"][1]
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
        v = load("val")
        assert v.label["lim_sup_pct"] == 40 and v.label["out"].startswith(os.path.join(base, "DOE_V", "val"))
        assert v.reference == e.label["out"] and v.indicators["variants"] == ["v1"] and v.indicators["f_modal"] == 150.0
        st = status(v)
        assert st["simulate"][0] == "done" and st["extract"][0] == "done" and not stages(v)["extract"].runnable
        assert st["validate"][0] == "blocked" and next_step(v)[1] == "label_template", (st, next_step(v))
        errs, warns = check(v)
        assert any("discretisation" in x for x in errs) and any("kappa also in training" in w for w in warns)
        # the training dataset disappears -> the validation indicators say what is missing and where
        os.rename(e.label["out"], e.label["out"] + ".bak")
        st = status(v)
        assert st["indicators"][0] == "blocked" and "of experiment 'train'" in st["indicators"][1]
        os.rename(e.label["out"] + ".bak", e.label["out"])
        # a validation that edits the label parameters is an error
        yaml_save({"kind": "validation", "training": "train", "runs": [{"dir": os.path.join(base, "DOE_V")}],
                   "label": {"lim_sup_pct": 50}}, exp_path("val_bad"))
        assert any("differs from training" in x for x in check(load("val_bad"))[0])
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
        assert c.name == "child" and c.indicators["f_modal"] == 150.0 and c.indicators["variants"] == ["v2"]
        assert any("not in the library" in x for x in check(c)[0])
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
        assert st["extract"][0] == "done" and st["label_build"][0] == "done" and "earlier step missing" in st["label_build"][1], st
        assert "12098.28 rpm" in i.cfg["description"] and summary(i)["spin"] == 12098.28
        assert "progress" in status_text(i)
        try:
            import_dir("imp", imp)
            raise AssertionError("import over an existing experiment")
        except ValueError:
            pass
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
    im = sub.add_parser("import")
    im.add_argument("name")
    im.add_argument("dir")
    im.add_argument("--kind", default="training", choices=["training", "validation"])
    im.add_argument("--training")
    im.add_argument("--label-out")
    im.add_argument("--description")
    r = sub.add_parser("run", help="run one stage in this console (what the app opens)")
    r.add_argument("exp")
    r.add_argument("stage", choices=STAGES)
    r.add_argument("--yes", action="store_true", help="do not ask before replacing existing outputs")
    sub.add_parser("selftest")
    a = p.parse_args()
    if a.cmd == "selftest":
        return _selftest()
    if a.cmd == "run":
        sys.exit(run_stage(a.exp, a.stage, a.yes))
    if a.cmd == "status":
        names = [a.exp] if a.exp else list_experiments()
        print("\n\n".join(status_text(load(n), a.goal) for n in names) or "no experiments in " + EXP_DIR)
    elif a.cmd == "check":
        errs, warns = check(load(a.exp))
        print("\n".join([f"ERROR: {x}" for x in errs] + [f"warning: {x}" for x in warns]) or "OK")
        sys.exit(1 if errs else 0)
    elif a.cmd == "import":
        print("written:", import_dir(a.name, a.dir, a.kind, a.training, a.label_out, a.description))


if __name__ == "__main__":
    main()
