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
    return dict(cases=cases, first=first, kappa=kappa, deflex=deflex)


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
    if missing:   # output already there (e.g. imported) although an earlier step is missing: keep it usable
        return "done", f"outputs exist; earlier step missing: {missing}"
    ref = rec.get("start") if rec else min((_mtime(p) for p in st.outputs if os.path.exists(p)), default=0)
    if rec and rec.get("hash") and rec["hash"] != st.hash:
        return "stale", "configuration changed since the last run"
    newer = [p for p in st.inputs if p and _mtime(p) > (ref or 0) + 1]
    if newer:
        return "stale", f"input changed after the run: {os.path.basename(newer[0])}"
    up = [dkey for _, dkey, s in dep_states if s == "stale"]
    if up:
        return "stale", f"upstream '{up[0]}' is stale"
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
    for e, k in goal_chain(exp, goal):
        state, reason = status(e, memo)[k]
        if state != "done":
            return e, k, state, reason
    return None


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
        if os.path.isfile(os.path.join(doe_dir, "reference_labels.yaml")):
            label["labels_yaml"] = os.path.join(doe_dir, "reference_labels.yaml").replace("\\", "/")
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
    sub.add_parser("selftest")
    a = p.parse_args()
    if a.cmd == "selftest":
        return _selftest()
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
