#!/usr/bin/env python
# coding: utf-8
"""
DOE Runner for Nessy2m
======================
Automatiza la ejecucion de un DOE con el simulador Nessy2m.

TODA la configuracion vive en configs/*.yaml (un YAML por DOE; base.yaml lleva lo comun).
Este script no se edita para cambiar de DOE.

Uso:
    python doe_runner.py --list-configs
    python doe_runner.py --config tube_ap17 --case 1DOF_150Hz
    python doe_runner.py --config tube_ap17 --case 1DOF_150Hz --dry-run

El script:
  1. Genera val_var segun el modo del YAML (factorial / sweep / manual).
  2. Sobreescribe p/param.py del caso con los valores generados.
  3. Llama a n2m_sch.py (equivalente al comando n2m_sch).
  4. Restaura p/param.py original al terminar.
"""

import os
import sys
import itertools
import subprocess
import argparse
import logging
import math
import re
import shutil
import glob
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

import h5py

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIGURACION: todo vive en configs/*.yaml (un YAML por DOE; base.yaml = lo comun).
#   python doe_runner.py --list-configs        # ver los DOE disponibles
#   python doe_runner.py --config tube_ap17    # usar uno (o CONFIG_FILE = "tube_ap17" mas abajo)
# Las variables de abajo son el estado interno que rellena el YAML (apply_config):
# NO llevan valores de ejemplo y NO hay que editarlas.
# ---------------------------------------------------------------------------
DEFAULT_N2M_BAT = os.path.join(          # solo si el YAML no trae n2m_bat
    r"C:\Users\quiqu\OneDrive-ensam.eu\Desktop\Thesis\03-Code\01-Nessy2m"
    r"\VP2025.1.0\VP2025.1.0\nessy2m",
    "n2m.bat",
)

SCRIPT_DIR = None       # <- base_dir
DOE_NAME = None         # <- doe_name
NB_PROC = 1             # <- nb_proc
DOE_MODE = None         # <- mode (factorial | sweep | manual)
DOE_FACTORIAL = None    # <- factorial
DOE_SWEEP = None        # <- sweep
DOE_MANUAL_LST = None   # <- manual.variables
DOE_MANUAL_VAL = None   # <- manual.values
DOE_EXTRACT_SIGNALS = ["Axial_disp", "Axial_vel", "Axial_acc"]   # <- extract_signals
DOE_FORCE_SIGNAL = "res_R_p"                                      # <- force_signal

# kappa = Ap / AP_REF (adimensional).
# Ap_start == Ap_end (profundidad fija)  -> attr 'kappa'
# Ap_start != Ap_end (barrido en el caso) -> attrs 'kappa_start' / 'kappa_end'
AP_REF_MODE = "none"    # <- ap_ref.mode: none | manual | model | model_at_spin
AP_REF_MANUAL = None    # <- ap_ref.manual [m]
AP_REF_MODEL = None     # <- ap_ref.model: preset de DOE_plots/sld_model.py (necesita sld_tools)


def _sld_model():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DOE_plots"))
    import sld_model  # pyright: ignore[reportMissingImports]
    return sld_model


def resolve_ap_ref(spin: float | None = None) -> float | None:
    """AP_REF [m] segun AP_REF_MODE (None o "none": sin kappa). `spin` [rpm] solo lo usa model_at_spin."""
    mode = AP_REF_MODE or "none"
    if mode == "none":
        return None
    if mode == "manual":
        return AP_REF_MANUAL
    if mode == "model":
        return _sld_model().ap_crit(AP_REF_MODEL) * 1e-3   # mm -> m: minimo global del SLD, sin rpm
    if mode == "model_at_spin":
        if spin is None:
            raise ValueError("AP_REF_MODE='model_at_spin' necesita el spin_rate del caso")
        return _sld_model().ap_lim(AP_REF_MODEL, spin) * 1e-3   # mm -> m: limite del lobulo a esas rpm
    raise ValueError(f"AP_REF_MODE invalido: {AP_REF_MODE!r} (usa 'none', 'manual', 'model' o 'model_at_spin')")


# ==============================================================================
# ARCHIVO DE CONFIGURACION (opcional)
#   Precedencia: --config en la CLI  >  CONFIG_FILE  >  las variables de arriba.
#   El YAML debe declarar base_dir, doe_name y mode (+ la tabla de ese modo), directamente
#   o por 'extends'; ver doe_config_example.yaml. Lo opcional (nb_proc, ap_ref, senales)
#   tiene los defaults de arriba.
# ==============================================================================
CONFIG_FILE = None   # ruta o nombre corto de configs/ (o una lista de ellos); None -> sin YAML (salvo --config)
CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")   # un YAML por DOE

_CONFIG_KEYS = ("base_dir", "case", "n2m_bat", "doe_name", "nb_proc", "mode", "factorial",
                "sweep", "manual", "ap_ref", "extract_signals", "force_signal")


CONFIG_HELP = """
CONFIGURACION DE UN DOE (YAML)
==============================
Donde:   DOE_simulacion/configs/*.yaml   (un archivo por DOE; base.yaml = lo comun a todos)
Elegir:  --config <nombre|ruta> [<otro> ...]     o     CONFIG_FILE = "nombre" (o lista) en doe_runner.py
         --config gana a CONFIG_FILE. Sin ninguno de los dos el runner se detiene (no hay DOE por defecto).
         Un nombre corto se busca como ruta tal cual, como <nombre>.yaml y dentro de configs/.
         python doe_runner.py --list-configs   muestra los disponibles (DOE, modo, casos, spin, Ap).

CLAVES                                                                         obligatoria  default
  base_dir         carpeta que contiene el caso (resultados: base_dir/doe_name)      SI
  doe_name         nombre de la carpeta de salida (sin separadores de ruta)          SI
  mode             factorial | sweep | manual: la tabla con ese nombre debe existir  SI
  factorial/sweep/manual   tabla de variables (ver abajo)                            la del mode
  case             subcarpeta del caso Nessy2m                                       no           1DOF_150Hz
  n2m_bat          ruta del n2m.bat                                                  no           DEFAULT_N2M_BAT
  nb_proc          procesos en paralelo (entero >= 1)                                no           1
  ap_ref           referencia de kappa (ver abajo)                                   no           mode: none
  extract_signals  senales a extraer de sens_out.hdf5                                no           [Axial_disp, Axial_vel, Axial_acc]
  force_signal     senal de fuerza de out.hdf5                                       no           res_R_p
  extends          otro YAML del que hereda (ver abajo)                              no
  Cualquier otra clave se rechaza. base_dir y extends relativos: respecto al archivo que los declara. Usa / tambien en Windows.

TABLAS DE VARIABLES
  Las claves van entre $...$ y los valores en SI (Ap en metros, spin en rpm). Numeros con exponente: 2.0e-4.
  Habituales: $Ap_start$ y $Ap_end$ (iguales = profundidad fija; distintos = rampa dentro del caso),
              $spin_rate$, $f_tooth$, $dxl_size$, $nb_dt_rev$.
  sweep      un caso por posicion. Las listas deben tener el MISMO largo (si no, error); un escalar se repite.
  factorial  producto cartesiano de todas las listas (una variable con 1 valor queda fija).
  manual     variables: [$a$, $b$]  y  values: [[1, 2], [3, 4]]  (una fila por caso).

AP_REF  (kappa = Ap / AP_REF; Ap fijo -> attr kappa, rampa -> kappa_start y kappa_end)
  ap_ref: {mode: none}                                 sin kappa
  ap_ref: {mode: manual, manual: 8.605e-3}             Ap de referencia en metros, igual para todo el DOE
  ap_ref: {mode: model, model: 1DOF_150}               Ap critico MINIMO del SLD del preset (un valor, sin rpm)
  ap_ref: {mode: model_at_spin, model: 2DOF_150_250}   limite del SLD A LAS RPM de cada caso ($spin_rate$):
                                                       cambia con el spin; en un bolsillo entre lobulos el
                                                       limite es infinito (kappa 0); fuera de los lobulos
                                                       calculados el caso queda sin kappa (con aviso)
  Los presets (modos, k, zeta, theta, Kf) se declaran en DOE_plots/sld_model.py (MODELS). Se guarda ap_ref_m
  (por caso) y ap_ref_mode / ap_ref_model en el doe_results.h5.

HERENCIA (extends: base)
  El DOE hereda todo de base.yaml y solo declara lo suyo (doe_name, mode, la tabla...).
  Si una clave esta en los dos, gana la del DOE. Las secciones (sweep, ap_ref, extract_signals...) se
  reemplazan COMPLETAS, no se mezclan por dentro: si redefines ap_ref, escribelo entero.
  Cadenas permitidas (A extiende B, B extiende C): gana el mas cercano al DOE. Un ciclo es un error.

QUE GANA (de mayor a menor)
  CLI (--case, --n2m_bat, --doe_name, --workers)  >  YAML del DOE  >  base.yaml  >  defaults de la tabla.

ANTES DE SIMULAR
  - Se valida todo y los errores salen juntos (listas de distinto largo, variables sin $...$, modo sin su
    tabla, preset inexistente...).
  - Se muestra un resumen (DOE, carpeta, casos, variables, AP_REF, si la salida ya existe) y se pide [y/N].
    --yes lo omite y confirma tambien reemplazar un DOE existente (su carpeta se borra). --dry-run no pregunta.
  - Varios DOE (--config a b c): se validan todos antes de lanzar el primero, una sola confirmacion, cada uno
    en su proceso, y se detiene si uno falla. No se admiten dos con el mismo doe_name ni --doe_name.
  - Al terminar se copia el YAML usado a <DOE>/doe_config.yaml.

DOE NUEVO EN 3 PASOS
  1. Copia un YAML de configs/ (el mas parecido) con otro nombre.
  2. Cambia doe_name y la tabla de variables (y mode si cambia de tipo).
  3. python doe_runner.py --list-configs   (debe aparecer sin ERROR)   y luego   --config <nombre>
     Antes de gastar horas: python doe_planner.py <nombre>  muestra los casos sobre el SLD (zona estable o
     inestable de cada uno), valida el YAML y lanza el runner en una consola aparte (boton Lanzar).
     Despues, para extraer:  --config <nombre> --command extract
"""


def _num(x):
    """float(x) si es numero (o texto numerico: PyYAML lee 2e-4 como texto), si no None."""
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return None
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def find_config(name: str, rel_to: str | None = None) -> str:
    """Ruta del YAML: la ruta tal cual, relativa a rel_to, o un nombre corto de configs/ (con o sin .yaml)."""
    for base in (None, rel_to, CONFIGS_DIR):
        for cand in (name, name + ".yaml"):
            f = cand if base is None else os.path.join(base, cand)
            if os.path.isfile(f):
                return os.path.abspath(f)
    avail = sorted(os.path.splitext(f)[0] for f in os.listdir(CONFIGS_DIR)) if os.path.isdir(CONFIGS_DIR) else []
    raise ValueError(f"config '{name}' no encontrada (ruta, relativa o nombre en {CONFIGS_DIR}); disponibles: {avail}")


def _read_yaml(path: str, seen: tuple = ()) -> dict:
    """YAML crudo. Con 'extends: otro' parte de ese YAML y le pone encima lo propio (las secciones
    completas se reemplazan, no se mezclan por dentro)."""
    import yaml
    path = os.path.abspath(path)
    if path in seen:
        raise ValueError(f"{path}: 'extends' circular ({' -> '.join(seen + (path,))})")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: debe ser un mapa 'clave: valor'")
    if isinstance(cfg.get("base_dir"), str):   # relativa -> respecto al archivo que la declara
        cfg["base_dir"] = os.path.normpath(os.path.join(os.path.dirname(path), cfg["base_dir"]))
    parent = cfg.pop("extends", None)
    if parent is None:
        return cfg
    if not isinstance(parent, str):
        raise ValueError(f"{path}: extends debe ser el nombre o ruta de otro YAML")
    return {**_read_yaml(find_config(parent, os.path.dirname(path)), seen + (path,)), **cfg}


def load_config(path: str, require: bool = True) -> dict:
    """Lee y valida el YAML. Devuelve el dict normalizado; todos los problemas juntos en un ValueError.
    require=False admite YAML parciales (p. ej. base.yaml, que solo se usa via extends)."""
    path = os.path.abspath(path)
    cfg = _read_yaml(path)
    errs: list = []
    bad = sorted(set(cfg) - set(_CONFIG_KEYS), key=str)
    if bad:
        errs.append(f"claves desconocidas {bad}; validas: {list(_CONFIG_KEYS)}")

    def table(name):   # {"$var$": valor | [valores]} -> {"$var$": [floats]}; en sweep los escalares se repiten
        t, out, n_err = cfg[name], {}, len(errs)
        if not isinstance(t, dict) or not t:
            errs.append(f"{name}: debe ser un mapa '$variable$: valores' no vacio")
            return
        for k, v in t.items():
            vs = [_num(x) for x in (v if isinstance(v, list) else [v])]
            if not (isinstance(k, str) and len(k) > 2 and k.startswith("$") and k.endswith("$")):
                errs.append(f"{name}: la variable {k!r} debe ir entre $...$ (ej. $Ap_start$)")
            elif not vs or None in vs:
                errs.append(f"{name}.{k}: se esperan numeros, recibido {v!r}")
            else:
                out[k] = vs
        if name == "sweep" and len(errs) == n_err:
            lens = {k: len(v) for k, v in out.items() if isinstance(t[k], list)}
            if len(set(lens.values())) > 1:
                errs.append(f"sweep: las listas deben tener el mismo largo (si no, zip trunca), hay {lens}")
            else:
                n = next(iter(lens.values()), 1)
                out = {k: (v if isinstance(t[k], list) else v * n) for k, v in out.items()}
        cfg[name] = out

    if "mode" in cfg:
        if cfg["mode"] not in ("factorial", "sweep", "manual"):
            errs.append(f"mode: {cfg['mode']!r} invalido (usa factorial, sweep o manual)")
        elif cfg["mode"] not in cfg:
            errs.append(f"mode: {cfg['mode']} necesita la seccion '{cfg['mode']}' en el YAML")
    for name in ("factorial", "sweep"):
        if name in cfg:
            table(name)
    if "manual" in cfg:
        m = cfg["manual"]
        if not (isinstance(m, dict) and set(m) == {"variables", "values"}
                and isinstance(m["variables"], list) and isinstance(m["values"], list)):
            errs.append("manual: debe tener 'variables' (lista de $var$) y 'values' (lista de filas)")
        else:
            rows = [[_num(x) for x in row] if isinstance(row, list) else None for row in m["values"]]
            if not m["values"] or any(r is None or None in r or len(r) != len(m["variables"]) for r in rows):
                errs.append(f"manual.values: cada fila debe tener {len(m['variables'])} numeros (uno por variable)")
            else:
                cfg["manual"] = {"variables": m["variables"], "values": rows}
    if "nb_proc" in cfg and not (isinstance(cfg["nb_proc"], int) and not isinstance(cfg["nb_proc"], bool)
                                 and cfg["nb_proc"] >= 1):
        errs.append(f"nb_proc: entero >= 1, recibido {cfg['nb_proc']!r}")
    for k in ("doe_name", "case", "n2m_bat", "force_signal"):
        if k in cfg and not (isinstance(cfg[k], str) and cfg[k].strip()):
            errs.append(f"{k}: texto no vacio, recibido {cfg[k]!r}")
    if isinstance(cfg.get("doe_name"), str) and ("/" in cfg["doe_name"] or os.sep in cfg["doe_name"]):
        errs.append("doe_name: es un nombre de carpeta, sin separadores de ruta")
    if "extract_signals" in cfg and not (isinstance(cfg["extract_signals"], list) and cfg["extract_signals"]
                                         and all(isinstance(x, str) for x in cfg["extract_signals"])):
        errs.append("extract_signals: lista de nombres de senal, ej. [Axial_disp, Axial_vel]")
    if "base_dir" in cfg:
        if not isinstance(cfg["base_dir"], str):
            errs.append(f"base_dir: texto, recibido {cfg['base_dir']!r}")
        else:   # relativa -> respecto al YAML
            cfg["base_dir"] = os.path.normpath(os.path.join(os.path.dirname(path), cfg["base_dir"]))
            if not os.path.isdir(cfg["base_dir"]):
                errs.append(f"base_dir: no existe la carpeta {cfg['base_dir']}")
    if "ap_ref" in cfg:
        a = cfg["ap_ref"]
        if not (isinstance(a, dict) and set(a) <= {"mode", "manual", "model"}):
            errs.append("ap_ref: mapa con mode (none|manual|model|model_at_spin) y, segun el modo, manual o model")
        else:
            mode = a.get("mode") or "none"
            if mode not in ("none", "manual", "model", "model_at_spin"):
                errs.append(f"ap_ref.mode: {mode!r} invalido (none, manual, model o model_at_spin)")
            elif mode == "manual" and not (_num(a.get("manual")) and _num(a.get("manual")) > 0):
                errs.append(f"ap_ref.manual: Ap de referencia en metros (> 0), recibido {a.get('manual')!r}")
            elif mode in ("model", "model_at_spin"):
                try:
                    presets = list(_sld_model().MODELS)
                    if a.get("model") not in presets:
                        errs.append(f"ap_ref.model: {a.get('model')!r} no esta en sld_model.MODELS {presets}")
                except Exception as exc:
                    errs.append(f"ap_ref.model: no se pudo cargar sld_model ({exc})")
            cfg["ap_ref"] = dict(a, mode=mode)
    if require:
        miss = [k for k in ("base_dir", "doe_name", "mode") if k not in cfg]
        if miss:
            errs.append(f"faltan claves obligatorias {miss} (declaralas aqui o en el YAML de 'extends')")
    if errs:
        raise ValueError(f"{path}:\n  - " + "\n  - ".join(errs))
    return cfg


def apply_config(cfg: dict) -> None:
    """Pasa lo declarado en el YAML a las variables del modulo (las que leen build_doe_cases, extract, etc.)."""
    g = globals()
    for key, var in (("base_dir", "SCRIPT_DIR"), ("doe_name", "DOE_NAME"), ("nb_proc", "NB_PROC"),
                     ("mode", "DOE_MODE"), ("factorial", "DOE_FACTORIAL"), ("sweep", "DOE_SWEEP"),
                     ("extract_signals", "DOE_EXTRACT_SIGNALS"), ("force_signal", "DOE_FORCE_SIGNAL")):
        if key in cfg:
            g[var] = cfg[key]
    if "manual" in cfg:
        g["DOE_MANUAL_LST"], g["DOE_MANUAL_VAL"] = cfg["manual"]["variables"], cfg["manual"]["values"]
    if "ap_ref" in cfg:
        g["AP_REF_MODE"] = cfg["ap_ref"]["mode"]
        if "manual" in cfg["ap_ref"]:
            g["AP_REF_MANUAL"] = _num(cfg["ap_ref"]["manual"])
        if "model" in cfg["ap_ref"]:
            g["AP_REF_MODEL"] = cfg["ap_ref"]["model"]


_CONFIG_VARS = ("SCRIPT_DIR", "DOE_NAME", "NB_PROC", "DOE_MODE", "DOE_FACTORIAL", "DOE_SWEEP",
                "DOE_EXTRACT_SIGNALS", "DOE_FORCE_SIGNAL", "DOE_MANUAL_LST", "DOE_MANUAL_VAL",
                "AP_REF_MODE", "AP_REF_MANUAL", "AP_REF_MODEL")


def _with_config(cfg: dict, fn):
    """Corre fn() con cfg aplicado y deja las variables del modulo como estaban (para listar/resumir sin efectos)."""
    g = globals()
    saved = {k: g[k] for k in _CONFIG_VARS}
    try:
        apply_config(cfg)
        return fn()
    finally:
        g.update(saved)


def describe(doe_dir: str, case_dir: str) -> str:
    """Resumen de lo que se va a lanzar (con las variables actuales): para confirmar antes de simular."""
    lst, val = build_doe_cases(DOE_MODE)

    def show(k):
        sc, unit = (1e3, " mm") if k.startswith("$Ap") else (1, "")
        v = sorted({r[lst.index(k)] * sc for r in val})
        return f"{v[0]:g}{unit}" if len(v) == 1 else f"{v[0]:g}..{v[-1]:g}{unit} ({len(v)} valores)"

    ap = AP_REF_MODE or "none"
    if ap == "manual":
        ap += f" ({AP_REF_MANUAL * 1e3:g} mm)"
    elif ap == "model":
        ap += f" (modelo {AP_REF_MODEL}: minimo global)"
    elif ap == "model_at_spin":
        try:
            spins = sorted({r[lst.index("$spin_rate$")] for r in val}) if "$spin_rate$" in lst else []
            lims = [f"{_sld_model().ap_lim(AP_REF_MODEL, w):.3f} mm @ {w:g} rpm" for w in spins[:4]]
            ap += f" (modelo {AP_REF_MODEL}, por caso): " + (" | ".join(lims) + (" ..." if len(spins) > 4 else "") if lims else "sin spin_rate: no se calcula kappa")
        except Exception as exc:   # sld_tools ausente, rpm fuera de los lobulos...
            ap += f" (modelo {AP_REF_MODEL}): ERROR {exc}"
    exists = "YA EXISTE: se BORRARA al confirmar" if os.path.exists(doe_dir) else "carpeta nueva"
    return (f"  DOE       : {os.path.basename(doe_dir)}\n"
            f"  Carpeta   : {os.path.dirname(doe_dir)}  (caso {os.path.basename(case_dir)})\n"
            f"  Modo      : {DOE_MODE} - {len(val)} caso(s)\n"
            f"  Variables : " + " | ".join(f"{k.strip('$')}={show(k)}" for k in lst) + "\n"
            f"  AP_REF    : {ap}\n"
            f"  Salida    : {exists}")


def list_configs() -> None:
    """Tabla de los YAML de configs/: nombre, DOE, modo, casos, spin y Ap."""
    rows = []
    for f in sorted(glob.glob(os.path.join(CONFIGS_DIR, "*.yaml"))):
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            cfg = load_config(f, require=False)
            if "mode" not in cfg and "doe_name" not in cfg:
                rows.append((name, "(base: sin DOE propio)", "", "", "", ""))
                continue
            load_config(f)   # es un DOE: debe estar completo

            def info():
                lst, val = build_doe_cases(DOE_MODE)

                def rng(k, sc=1.0):
                    if k not in lst:
                        return "-"
                    v = sorted({r[lst.index(k)] * sc for r in val})
                    return f"{v[0]:g}" if len(v) == 1 else f"{v[0]:g}..{v[-1]:g}"

                return (DOE_NAME, DOE_MODE, str(len(val)), rng("$spin_rate$"), rng("$Ap_start$", 1e3) + " -> " + rng("$Ap_end$", 1e3))
            rows.append((name,) + _with_config(cfg, info))
        except (ValueError, OSError, ImportError) as exc:
            rows.append((name, "ERROR: " + str(exc).splitlines()[1].strip(" -") if "\n" in str(exc) else "ERROR: " + str(exc), "", "", "", ""))
    head = ("config", "DOE", "modo", "casos", "spin [rpm]", "Ap ini -> fin [mm]")
    w = [max(len(str(r[i])) for r in rows + [head]) for i in range(6)]
    for r_ in [head, tuple("-" * x for x in w)] + rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r_)).rstrip())
    print(f"\n  {len(rows)} archivo(s) en {CONFIGS_DIR}")


def _argv_without_config() -> list:
    out, skip = [], False
    for tok in sys.argv[1:]:
        if tok == "--config":
            skip = True
            continue
        if skip and not tok.startswith("-"):
            continue
        skip = False
        out.append(tok)
    return out


def _run_batch(paths: list, args) -> None:
    """Varios DOE en secuencia: valida TODOS primero, muestra el resumen de cada uno, confirma una vez y
    lanza este mismo script con cada config en su propio proceso (estado limpio; se detiene si uno falla)."""
    if args.doe_name:
        log.error("--doe_name no se puede combinar con varias configs (cada YAML trae el suyo)")
        sys.exit(1)
    cfgs, errs = {}, []
    for f in paths:
        try:
            cfgs[f] = load_config(f)
        except (ValueError, OSError, ImportError) as exc:
            errs.append(str(exc))
    if errs:
        log.error("Config invalida (no se lanzo ningun DOE):\n%s", "\n".join(errs))
        sys.exit(1)
    texts, by_name = [], {}
    for f, cfg in cfgs.items():
        def info():
            case = args.case or cfg.get("case") or "1DOF_150Hz"
            return DOE_NAME, describe(os.path.join(SCRIPT_DIR, DOE_NAME), os.path.join(SCRIPT_DIR, case))
        name, text = _with_config(cfg, info)
        by_name.setdefault((os.path.join(cfg.get("base_dir", SCRIPT_DIR)), name), []).append(os.path.basename(f))
        texts.append(f"[{os.path.basename(f)}]\n{text}")
    dup = {k[1]: v for k, v in by_name.items() if len(v) > 1}
    if dup:
        log.error("Varias configs escriben en el mismo DOE (cambia doe_name): %s", dup)
        sys.exit(1)
    print("\n\n".join(texts) + "\n")
    if not (args.yes or args.dry_run):
        if input(f"Lanzar los {len(paths)} DOE en secuencia? [y/N]: ").strip().lower() not in ("y", "yes"):
            log.info("Cancelado.")
            return
    rest = _argv_without_config()
    for k, f in enumerate(paths, 1):
        log.info("===== DOE %d/%d: %s =====", k, len(paths), os.path.basename(f))
        rc = subprocess.run([sys.executable, os.path.abspath(__file__), "--config", f, "--yes"] + rest).returncode
        if rc != 0:
            log.error("El DOE %d/%d (%s) termino con error %d: se detiene la secuencia.", k, len(paths), os.path.basename(f), rc)
            sys.exit(rc)
    log.info("Secuencia completa: %d DOE.", len(paths))


def _keep_config(cfg_path, doe_dir: str, dry_run: bool) -> None:
    """Copia del YAML usado dentro de la carpeta del DOE (tras simular: n2m_sch exige que no exista antes)."""
    if cfg_path and not dry_run and os.path.isdir(doe_dir):
        shutil.copy2(cfg_path, os.path.join(doe_dir, "doe_config.yaml"))
        log.info("Config copiada a %s", os.path.join(doe_dir, "doe_config.yaml"))


def build_doe_cases(mode: str) -> tuple[list, list]:
    """Devuelve (lst_var, val_var) segun DOE_MODE."""
    if mode == "factorial":
        lst_var = list(DOE_FACTORIAL.keys())
        val_var = [list(c) for c in itertools.product(*DOE_FACTORIAL.values())]
    elif mode == "sweep":
        lst_var = list(DOE_SWEEP.keys())
        val_var = [list(r) for r in zip(*DOE_SWEEP.values())]
    else:  # manual
        lst_var = DOE_MANUAL_LST
        val_var = DOE_MANUAL_VAL
    return lst_var, val_var


def write_post(post_path: str, cleanup_dirs: list) -> None:
    """Agrega al final de post.py un bloque que borra cleanup_dirs tras la sim."""
    block = (
        "\n"
        "# --- Limpieza POST-DOE (doe_runner.py) ---\n"
        "import shutil as _sh\n"
        "import os as _os\n"
        "_cdir = _os.path.abspath(_os.curdir)\n"
        + f"_cdirs = {cleanup_dirs!r}\n"
        + "for _d in _cdirs:\n"
        "    _p = _os.path.join(_cdir, _d)\n"
        "    if _os.path.isdir(_p):\n"
        "        _sh.rmtree(_p)\n"
        "        print(f'[CLEANUP] Borrado: {_p}')\n"
    )
    with open(post_path, "a", encoding="utf-8") as fh:
        fh.write(block)


def write_param(param_path: str, lst_var: list, val_var: list, doe_name: str, nb_proc: int) -> None:
    """Sobreescribe p/param.py con los valores generados."""
    lines = [
        "#!/usr/bin/env python\n",
        "# coding: utf-8\n",
        "# AUTO-GENERADO por doe_runner.py — no editar manualmente\n",
        "import os\n",
        "param = {\n",
        f"    'nb_proc'     : {nb_proc},\n",
        f"    'dir_ref2exe' : os.path.join('..', '{doe_name}'),\n",
        f"    'lst_var'     : {lst_var!r},\n",
        f"    'val_var'     : {val_var!r},\n",
        "}\n",
    ]
    with open(param_path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)


def write_param_abs(param_path: str, lst_var: list, val_var: list,
                    abs_doe_dir: str, nb_proc: int) -> None:
    """Como write_param pero con dir_ref2exe como ruta absoluta (modo --timed)."""
    lines = [
        "#!/usr/bin/env python\n",
        "# coding: utf-8\n",
        "# AUTO-GENERADO por doe_runner.py — no editar manualmente\n",
        "import os\n",
        "param = {\n",
        f"    'nb_proc'     : {nb_proc},\n",
        f"    'dir_ref2exe' : {abs_doe_dir!r},\n",
        f"    'lst_var'     : {lst_var!r},\n",
        f"    'val_var'     : {val_var!r},\n",
        "}\n",
    ]
    with open(param_path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)


def load_n2m_env(bat_path: str) -> dict:
    bat_path = os.path.abspath(bat_path)
    if not os.path.isfile(bat_path):
        raise FileNotFoundError(f"No se encontro n2m.bat en: {bat_path}")
    bat_dir = os.path.dirname(bat_path)
    log.info("Cargando entorno Nessy2m desde: %s", bat_path)
    env = dict(os.environ)
    env["cd"] = bat_dir
    env["CD"] = bat_dir
    set_re = re.compile(r"^\s*SET\s+([^=]+)=(.*)$", re.IGNORECASE)

    def expand(val, ctx):
        def _repl(m):
            name = m.group(1)
            for k, v in ctx.items():
                if k.upper() == name.upper():
                    return v
            return m.group(0)
        return re.sub(r"%([^%]+)%", _repl, val)

    with open(bat_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = set_re.match(line)
            if m:
                key = m.group(1).strip()
                val = expand(m.group(2).strip(), env)
                env[key] = val

    if "n2m" not in env:
        log.warning("Variable 'n2m' no encontrada tras parsear el .bat.")
    else:
        log.info("Variable n2m = %s", env["n2m"])
    return env


def get_python_exe(env: dict) -> str:
    for d in env.get("PATH", "").split(os.pathsep):
        candidate = os.path.join(d, "python.exe")
        if os.path.isfile(candidate):
            log.info("Usando Python de Nessy2m: %s", candidate)
            return candidate
    log.warning("Usando Python actual: %s", sys.executable)
    return sys.executable


# ==============================================================================
# FUNCIONES DE EXTRACCION
# ==============================================================================

def _hdf5_find_dataset(group: h5py.Group, name: str):
    """Busqueda recursiva de un Dataset por nombre dentro de un h5py.Group.

    Acepta dos estructuras:
      - Dataset directo:   .../name          (retorna ese dataset)
      - Grupo con data:    .../name/data      (retorna el dataset 'data' dentro)
    """
    for key in group:
        item = group[key]
        if key == name:
            if isinstance(item, h5py.Dataset):
                return item
            if isinstance(item, h5py.Group) and "data" in item:
                return item["data"]
        if isinstance(item, h5py.Group):
            result = _hdf5_find_dataset(item, name)
            if result is not None:
                return result
    return None


def _extract_series_from_dataset(dataset: h5py.Dataset):
    """Extrae tiempo y valores de un dataset 2D donde la primera columna es tiempo."""
    arr = dataset[()]
    if not isinstance(arr, np.ndarray) or arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Dataset incompatible para serie tiempo/valor: shape={getattr(arr, 'shape', None)}")

    time = arr[:, 0]
    values = arr[:, 1:]
    return time, values


def read_var_val(path: str) -> dict:
    """Lee var_val.py y retorna el dict var_val. Retorna {} si no existe."""
    if not os.path.isfile(path):
        log.warning("var_val.py no encontrado: %s", path)
        return {}
    ns: dict = {"__builtins__": {}}
    with open(path, "r", encoding="utf-8") as fh:
        exec(compile(fh.read(), path, "exec"), ns)  # noqa: S102
    return ns.get("var_val", {})


def extract_doe_results(doe_dir: str, case_name: str, signals: list, dry_run: bool = False) -> list:
    """Itera los casos del DOE, extrae señales de sens_out.hdf5 y out.hdf5, y guarda doe_results.h5."""
    pattern = os.path.join(doe_dir, "*", case_name)
    case_dirs = sorted(
        glob.glob(pattern),
        key=lambda p: int(os.path.basename(os.path.dirname(p)))
    )

    if not case_dirs:
        log.warning("No se encontraron casos con patron: %s", pattern)
        return []

    log.info("Casos encontrados: %d | DOE dir: %s", len(case_dirs), doe_dir)

    if dry_run:
        for cd in case_dirs:
            idx = os.path.basename(os.path.dirname(cd))
            log.info("[DRY-RUN] Caso %s → %s", idx, cd)
        return []

    per_case = AP_REF_MODE == "model_at_spin"   # AP_REF distinto en cada caso, segun su spin
    ap_ref = None if per_case else resolve_ap_ref()
    log.info("AP_REF (%s): %s", AP_REF_MODE or "none",
             f"segun el spin de cada caso (modelo {AP_REF_MODEL})" if per_case
             else f"{ap_ref * 1e3:.4f} mm" if ap_ref else "sin kappa")

    results = []
    out_path = os.path.join(doe_dir, "doe_results.h5")

    # Eliminar el archivo previo para evitar bloqueo de Windows (errno=0 / GetLastError=33)
    if os.path.isfile(out_path):
        try:
            os.remove(out_path)
        except OSError as e:
            log.error("No se puede sobreescribir %s — ciérralo en otros programas y vuelve a intentar. (%s)", out_path, e)
            return []

    with h5py.File(out_path, "w") as out_f:
        out_f.attrs["ap_ref_mode"] = AP_REF_MODE or "none"
        if ap_ref:
            out_f.attrs["ap_ref_m"] = ap_ref
        if AP_REF_MODE in ("model", "model_at_spin"):
            out_f.attrs["ap_ref_model"] = AP_REF_MODEL
        for case_path in case_dirs:
            idx = int(os.path.basename(os.path.dirname(case_path)))
            group_name = f"case_{idx:03d}"

            # -- var_val.py --
            var_val = read_var_val(os.path.join(case_path, "var_val.py"))
            grp = out_f.create_group(group_name)
            for k, v in var_val.items():
                try:
                    grp.attrs[k] = v
                except Exception:
                    grp.attrs[k] = str(v)

            # -- kappa = Ap / AP_REF (AP_REF fijo, o el limite del lobulo a las rpm de este caso) --
            ap_ref_c = ap_ref
            if per_case:
                spin = var_val.get("$spin_rate$")
                try:
                    ap_ref_c = resolve_ap_ref(float(spin)) if spin is not None else None
                except ValueError as exc:
                    ap_ref_c = None
                    log.warning("Caso %s: %s -> sin kappa", group_name, exc)
                if spin is None:
                    log.warning("Caso %s sin $spin_rate$: no se calcula kappa", group_name)
            if ap_ref_c:
                ap_start = var_val.get("$Ap_start$")
                ap_end   = var_val.get("$Ap_end$")
                if ap_start is not None and ap_end is not None:
                    grp.attrs["ap_ref_m"] = ap_ref_c   # el AP_REF realmente usado en este caso
                    if ap_start == ap_end:
                        grp.attrs["kappa"] = math.trunc(ap_start / ap_ref_c * 1000) / 1000
                    else:
                        grp.attrs["kappa_start"] = math.trunc(ap_start / ap_ref_c * 1000) / 1000
                        grp.attrs["kappa_end"]   = math.trunc(ap_end / ap_ref_c * 1000) / 1000

            # -- wall_time_s (guardado por --timed dentro de la carpeta del caso) --
            wt_file = os.path.join(os.path.dirname(case_path), "wall_time_s.txt")
            if os.path.isfile(wt_file):
                with open(wt_file, "r", encoding="utf-8") as _f:
                    grp.attrs["wall_time_s"] = float(_f.read().strip())

            # -- sens_out.hdf5 --
            hdf5_path = os.path.join(case_path, "sens_out.hdf5")
            case_result = {"idx": idx, "var_val": var_val}
            if os.path.isfile(hdf5_path):
                with h5py.File(hdf5_path, "r") as src_f:
                    for sig in signals:
                        ds = _hdf5_find_dataset(src_f, sig)
                        if ds is None:
                            log.warning("Senal '%s' no encontrada en %s", sig, hdf5_path)
                            continue
                        arr = ds[()]
                        t = arr[:, 0]
                        y = arr[:, 1]
                        sig_grp = grp.create_group(sig)
                        sig_grp.create_dataset("time",   data=t, compression="gzip")
                        sig_grp.create_dataset("values", data=y, compression="gzip")
                        case_result[sig] = (t, y)
            else:
                log.warning("sens_out.hdf5 no encontrado en: %s", case_path)

            # -- out.hdf5: fuerza res_R_P --
            out_hdf5_path = os.path.join(case_path, "out.hdf5")
            if os.path.isfile(out_hdf5_path):
                with h5py.File(out_hdf5_path, "r") as src_f:
                    ds = _hdf5_find_dataset(src_f, DOE_FORCE_SIGNAL)
                    if ds is None:
                        log.warning("Senal '%s' no encontrada en %s", DOE_FORCE_SIGNAL, out_hdf5_path)
                    else:
                        t_force, f_force = _extract_series_from_dataset(ds)
                        force_grp = grp.create_group(DOE_FORCE_SIGNAL)
                        force_grp.create_dataset("time", data=t_force, compression="gzip")
                        force_grp.create_dataset("values", data=f_force, compression="gzip")
                        case_result[DOE_FORCE_SIGNAL] = (t_force, f_force)
            else:
                log.warning("out.hdf5 no encontrado en: %s", case_path)

            results.append(case_result)
            log.info("Caso %03d extraido | var_val=%s", idx, var_val)

    log.info("doe_results.h5 guardado en: %s", out_path)
    return results


def merge_doe_results(base_doe_dir: str, patch_doe_dir: str, dry_run: bool = False) -> None:
    """Fusiona dos doe_results.h5 — añade los casos de patch al base sin duplicar.

    - Los grupos del patch se renumeran continuando desde el mayor índice del base.
    - Si un grupo con los mismos attrs ya existe en el base, se omite con warning.
    - El base no se toca si dry_run=True.
    """
    base_h5   = os.path.join(base_doe_dir,  "doe_results.h5")
    patch_h5  = os.path.join(patch_doe_dir, "doe_results.h5")

    if not os.path.isfile(base_h5):
        log.error("doe_results.h5 base no encontrado: %s", base_h5)
        return
    if not os.path.isfile(patch_h5):
        log.error("doe_results.h5 patch no encontrado: %s", patch_h5)
        return

    with h5py.File(base_h5, "r" if dry_run else "a") as base_f:
        # Indice máximo actual en el base
        existing_indices = []
        for k in base_f.keys():
            try:
                existing_indices.append(int(k.split("_")[-1]))
            except ValueError:
                pass
        next_idx = max(existing_indices) + 1 if existing_indices else 0

        # Attrs de grupos existentes para detectar duplicados
        # wall_time_s se ignora: dos corridas identicas difieren en tiempo
        def _attrs(g):
            return {k: v for k, v in dict(g.attrs).items() if k != "wall_time_s"}
        existing_attrs = [_attrs(base_f[k]) for k in base_f.keys()]

        with h5py.File(patch_h5, "r") as patch_f:
            added = 0
            skipped = 0
            for grp_name in sorted(patch_f.keys()):
                patch_grp  = patch_f[grp_name]
                patch_attrs = _attrs(patch_grp)

                # Comprobar si ya existe un caso con los mismos attrs
                if any(patch_attrs == ea for ea in existing_attrs):
                    log.warning("Omitido (ya existe): %s  attrs=%s", grp_name, patch_attrs)
                    skipped += 1
                    continue

                new_name = f"case_{next_idx:03d}"
                # Carpeta física del caso en el patch: patch_doe_dir/<patch_idx>/
                patch_idx    = int(grp_name.split("_")[-1])
                src_case_dir = os.path.join(patch_doe_dir, str(patch_idx))
                dst_case_dir = os.path.join(base_doe_dir,  str(next_idx))

                if dry_run:
                    log.info("[DRY-RUN] Añadiría: %s -> %s  attrs=%s",
                             grp_name, new_name, patch_attrs)
                    if os.path.isdir(src_case_dir):
                        log.info("[DRY-RUN] Copiaría carpeta: %s -> %s",
                                 src_case_dir, dst_case_dir)
                    else:
                        log.warning("[DRY-RUN] Carpeta física no encontrada: %s", src_case_dir)
                else:
                    patch_f.copy(patch_grp, base_f, name=new_name)
                    log.info("Añadido (HDF5): %s -> %s  attrs=%s",
                             grp_name, new_name, patch_attrs)
                    existing_attrs.append(patch_attrs)
                    if os.path.isdir(src_case_dir):
                        shutil.copytree(src_case_dir, dst_case_dir)
                        log.info("Copiada carpeta: %s -> %s", src_case_dir, dst_case_dir)
                    else:
                        log.warning("Carpeta física no encontrada (solo HDF5): %s", src_case_dir)

                next_idx += 1
                added += 1

    log.info("Merge completado: %d añadidos, %d omitidos (duplicados).", added, skipped)
    if not dry_run:
        log.info("base doe_results.h5 actualizado: %s", base_h5)


def run_n2m_command(command_script, case_dir, env, python_exe, extra_args=None, dry_run=False):
    cmd = [python_exe, command_script] + (extra_args or [])
    log.info("Directorio de trabajo : %s", case_dir)
    log.info("Comando               : %s", " ".join(cmd))
    if dry_run:
        log.info("[DRY-RUN] El comando no fue ejecutado.")
        return 0
    return subprocess.run(cmd, cwd=case_dir, env=env).returncode


def _run_one_case_timed(
    idx: int,
    lst_var: list,
    case_vals: list,
    doe_name: str,
    src_case_dir: str,
    command_script: str,
    env: dict,
    python_exe: str,
    script_dir: str,
    dry_run: bool,
) -> tuple:
    """Corre un único caso directamente en la carpeta final del DOE.
    n2m_sch exige que dir_ref2exe no exista → usamos un subdir staging que se renombra al final.
    Devuelve (idx, wall_time_s, rc).
    """
    final_doe_dir = os.path.join(script_dir, doe_name)
    os.makedirs(final_doe_dir, exist_ok=True)
    case_name    = os.path.basename(src_case_dir)         # ej. "1DOF_150Hz"

    # Directorio de trabajo del caso (copia del caso base) dentro del DOE final
    work_case_root = os.path.join(final_doe_dir, f"_work_{idx:03d}")
    work_case_dir  = os.path.join(work_case_root, case_name)
    # dir_ref2exe: n2m_sch exige que NO exista aún → staging dentro del DOE final
    staging_dir    = os.path.join(final_doe_dir, f"_staging_{idx:03d}")
    try:
        shutil.copytree(src_case_dir, work_case_dir)       # copia con nombre original
        param_path = os.path.join(work_case_dir, "p", "param.py")
        write_param_abs(param_path, lst_var, [case_vals], staging_dir, 1)
        log.info("Caso %03d INICIADO  | vars=%s", idx, case_vals)
        t0 = time.perf_counter()
        rc = 0
        if not dry_run:
            rc = subprocess.run(
                [python_exe, command_script],
                cwd=work_case_dir, env=env,
            ).returncode
        wall_time = time.perf_counter() - t0
        # Renombrar staging/0/ → final_doe_dir/idx/
        src_idx = os.path.join(staging_dir, "0")
        dst_idx = os.path.join(final_doe_dir, str(idx))
        if not dry_run and os.path.isdir(src_idx):
            if os.path.exists(dst_idx):
                shutil.rmtree(dst_idx)
            shutil.move(src_idx, dst_idx)
            # Guardar tiempo dentro de la carpeta del caso
            wt_file = os.path.join(dst_idx, "wall_time_s.txt")
            with open(wt_file, "w", encoding="utf-8") as _f:
                _f.write(str(wall_time))
    finally:
        shutil.rmtree(work_case_root, ignore_errors=True)
        shutil.rmtree(staging_dir,    ignore_errors=True)
    return idx, wall_time, rc


def run_doe_timed(
    lst_var: list,
    val_var: list,
    doe_name: str,
    src_case_dir: str,
    command_script: str,
    env: dict,
    python_exe: str,
    script_dir: str,
    nb_workers: int,
    dry_run: bool,
) -> None:
    """Ejecuta el DOE caso a caso con ThreadPoolExecutor (NB_PROC workers en paralelo).
    Mide el tiempo de cada caso; lo guarda en caso_dir/wall_time_s.txt (leído por extract).
    """
    n = len(val_var)
    log.info("DOE timed: %d casos, %d workers en paralelo", n, nb_workers)
    with ThreadPoolExecutor(max_workers=nb_workers) as executor:
        futures = {
            executor.submit(
                _run_one_case_timed,
                idx, lst_var, case_vals, doe_name,
                src_case_dir, command_script, env, python_exe,
                script_dir, dry_run,
            ): idx
            for idx, case_vals in enumerate(val_var)
        }
        for future in as_completed(futures):
            idx, wall_time, rc = future.result()
            status = "DRY-RUN" if dry_run else ("OK" if rc == 0 else f"ERROR rc={rc}")
            log.info("Caso %03d | %.1f s | %s | vars=%s",
                     idx, wall_time, status, val_var[idx])
    log.info("Total casos completados: %d", len(val_var))


def parse_args():
    epilog = """
DOE Runner — Automatización de simulaciones Nessy2m
====================================================
Genera val_var, sobreescribe param.py y ejecuta n2m_sch (o extrae resultados).
TODA la configuracion esta en configs/*.yaml; todos los comandos necesitan --config (o CONFIG_FILE):
de ahi salen la carpeta base y el nombre del DOE (--doe_name lo sustituye).

CONFIGURACION (resumen; referencia completa: --help-config)
------------------------------------------------------------
  Un YAML por DOE en configs/ (base.yaml = lo comun, se hereda con "extends: base").
  Obligatorio: base_dir, doe_name, mode (+ la tabla factorial / sweep / manual de ese modo).
  Opcional:    case, n2m_bat, nb_proc, ap_ref (none | manual | model | model_at_spin), extract_signals, force_signal.
  Se elige con --config <nombre> (o CONFIG_FILE en el script); --list-configs muestra los disponibles.
  Gana: CLI  >  YAML del DOE  >  base.yaml.   DOE nuevo = copiar un YAML y cambiar doe_name y la tabla.

COMANDOS  (--command)
---------------------
  n2m_sch    Ejecuta el DOE completo (simulación Nessy2m). [por defecto]
  extract    Lee sens_out.hdf5 de cada caso y guarda doe_results.h5.
             No requiere Nessy2m; corre con el entorno CAMP10.
  nessy2m    Lanza nessy2m directamente.
  pp_creat   Ejecuta pp_creat (post-proceso inicial).
  pp_init    Ejecuta pp_init.

EJEMPLOS
--------
  python doe_runner.py --list-configs
      Tabla de los YAML de configs/ (un YAML por DOE; base.yaml lleva lo comun, via 'extends: base').

  python doe_runner.py --config tube_ap17 --case 1DOF_150Hz
      Usa configs/tube_ap17.yaml (o una ruta). Muestra un resumen
      (DOE, casos, variables, AP_REF, si la salida ya existe) y pide confirmacion; --yes la omite.
      Equivale a CONFIG_FILE = "tube_ap17" en el script; --config tiene prioridad.
      El YAML debe declarar base_dir, doe_name y mode (+ su tabla); se valida antes de simular y se copia a <DOE>/doe_config.yaml.

  python doe_runner.py --config tube_ap17 tube_ap_sweep --case 1DOF_150Hz
      Varios DOE en secuencia: valida todos antes de lanzar el primero, confirma una vez y se detiene si uno falla.

  python doe_runner.py --config <doe> --case 1DOF_150Hz
      Ejecuta el DOE completo para el caso 1DOF_150Hz.

  python doe_runner.py --config <doe> --case 1DOF_150Hz --dry-run
      Simula sin ejecutar (muestra rutas y comandos).

  python doe_runner.py --config <doe> --case 1DOF_150Hz --command extract
      Extrae señales de todos los casos -> guarda doe_results.h5.

  python doe_runner.py --config <doe> --case 1DOF_150Hz --command extract --doe_name DOE_Influence_dt
      Extrae de la carpeta DOE_Influence_dt/ (sustituye el doe_name del YAML).

  python doe_runner.py --config <doe> --case 1DOF_150Hz --timed --auto-extract
      Corre el DOE completo (modo timed) y al terminar corre --command extract
      solo, sin tener que lanzarlo aparte despues.

  python doe_runner.py --config <doe> --command merge --doe_name DOE_base --merge_from DOE_patch
      Fusiona DOE_patch/doe_results.h5 en DOE_base/doe_results.h5.
      Los casos nuevos se renumeran continuando desde el último índice del base.
      Los casos con attrs idénticos se omiten automáticamente (no se duplican).

  python doe_runner.py --config <doe> --command merge --doe_name DOE_base --merge_from DOE_patch --merge_out DOE_merged
      Igual que el anterior, pero copia DOE_base -> DOE_merged primero y fusiona
      ahi. DOE_base y DOE_patch quedan intactos.

  python doe_runner.py --config <doe> --case 1DOF_150Hz --timed
      Corre cada caso en directorio aislado, NB_PROC casos en paralelo (runner-side).
      Guarda el tiempo por caso en cada carpeta de caso (wall_time_s.txt).
      Tras extract, doe_plotter muestra figura de tiempo de cómputo vs dt.

  python doe_runner.py --config <doe> --case 1DOF_150Hz --timed --dry-run
      Simula el modo timed sin ejecutar Nessy2m (verifica rutas y configuración).

FLUJO TÍPICO CON TIMING
-----------------------
  1. Poner nb_proc: 4 en el YAML (4 casos en paralelo)
  2. python doe_runner.py --config <doe> --case <caso> --timed          # simular + medir tiempos
  3. python doe_runner.py --config <doe> --case <caso> --command extract  # extraer (lee wall_time_s.txt)
  4. python doe_plotter.py  --doe_name <DOE>             # ver figura de tiempo

FLUJO PARA AÑADIR CASOS A UN DOE EXISTENTE
--------------------------------------------
  1. Un YAML nuevo (extends: base) con doe_name: DOE_base_patch  (nombre nuevo, para no chocar con n2m_sch)
  2. Declarar en ese YAML solo los N casos nuevos
  3. python doe_runner.py --config <doe> --case <caso>                              # simular patch
  4. python doe_runner.py --config <doe> --case <caso> --command extract            # extraer patch
  5. python doe_runner.py --config <doe> --command merge \
         --doe_name DOE_base --merge_from DOE_base_patch             # fusionar
  6. python doe_plotter.py  --doe_name DOE_base                      # visualizar todo

FLUJO TÍPICO
------------
  1. Crear o editar el YAML del DOE en configs/ (doe_name, mode, variables...)
  2. python doe_runner.py --config <doe> --case <caso>                        # simular
  3. python doe_runner.py --config <doe> --case <caso> --command extract      # extraer
  4. python doe_plotter.py  --doe_name <DOE>                   # visualizar
  5. python doe_selector.py --doe_name <DOE>                   # explorar
"""
    parser = argparse.ArgumentParser(
        description="DOE Runner — Nessy2m",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--case", default=None, help="Subdirectorio del caso (default: case del YAML o 1DOF_150Hz)")
    parser.add_argument("--config", nargs="+", default=None, metavar="YAML",
                        help="Config(s) del DOE: ruta o nombre corto de configs/ (varias = en secuencia); gana sobre CONFIG_FILE")
    parser.add_argument("--help-config", action="store_true",
                        help="Ayuda completa de la configuracion YAML (claves, defaults, ap_ref, herencia, precedencia) y sale")
    parser.add_argument("--list-configs", action="store_true", help="Lista los YAML de configs/ (DOE, modo, casos, spin, Ap) y sale")
    parser.add_argument("--yes", action="store_true", help="No pedir confirmacion (confirma tambien reemplazar un DOE existente)")
    parser.add_argument("--case_dir", default=None, help="Ruta completa al caso (opcional)")
    parser.add_argument("--n2m_bat", default=None, help="Ruta al n2m.bat (default: n2m_bat del YAML o DEFAULT_N2M_BAT)")
    parser.add_argument("--command", default="n2m_sch",
                        choices=["n2m_sch", "nessy2m", "pp_creat", "pp_init", "extract", "merge"])
    parser.add_argument("--doe_name", default=None,
                        help="DOE base (extract/merge). Sobreescribe DOE_NAME del config")
    parser.add_argument("--merge_from", default=None,
                        help="(merge) Nombre del DOE patch cuyos casos se añaden al base")
    parser.add_argument("--merge_out", default=None,
                        help="(merge) Si se da, copia --doe_name a esta carpeta nueva y fusiona ahi "
                             "en vez de modificar el DOE base original")
    parser.add_argument("--workers", type=int, default=None,
                        help="Numero de workers para --timed (por defecto usa NB_PROC)")
    parser.add_argument("--timed", action="store_true",
                        help="Corre cada caso individualmente, mide tiempo (wall_time_s.txt por caso -> attr HDF5 tras extract)")
    parser.add_argument("--auto-extract", action="store_true",
                        help="Corre --command extract automaticamente al terminar la corrida")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--args", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.help_config:
        print(CONFIG_HELP)
        return
    if args.list_configs:
        list_configs()
        return

    # 0. Config(s) YAML (--config > CONFIG_FILE > variables del script)
    names = args.config or ([CONFIG_FILE] if isinstance(CONFIG_FILE, str) else list(CONFIG_FILE or []))
    try:
        paths = [find_config(n) for n in names]
    except ValueError as exc:
        log.error("%s", exc)
        sys.exit(1)
    if not paths:
        avail = sorted(os.path.splitext(f)[0] for f in os.listdir(CONFIGS_DIR)) if os.path.isdir(CONFIGS_DIR) else []
        log.error("Falta la configuracion del DOE: usa --config <nombre> (o CONFIG_FILE en el script).\n"
                  "  Disponibles en %s: %s\n  Ver detalle: python doe_runner.py --list-configs\n  Ayuda de la configuracion: python doe_runner.py --help-config", CONFIGS_DIR, avail)
        sys.exit(1)
    if len(paths) > 1:
        _run_batch(paths, args)
        return
    cfg_path = paths[0] if paths else None
    cfg = {}
    if cfg_path:
        try:
            cfg = load_config(cfg_path)
        except (ValueError, OSError, ImportError) as exc:
            log.error("Config invalida:\n%s", exc)
            sys.exit(1)
        apply_config(cfg)
        log.info("Config: %s", os.path.abspath(cfg_path))
    args.case = args.case or cfg.get("case") or "1DOF_150Hz"
    args.n2m_bat = args.n2m_bat or cfg.get("n2m_bat") or DEFAULT_N2M_BAT

    # 1. Resolver directorio del caso
    case_dir = os.path.abspath(args.case_dir) if args.case_dir else os.path.join(SCRIPT_DIR, args.case)
    case_dir = os.path.normpath(case_dir)   # elimina trailing slash para que dirname() funcione bien
    if not os.path.isdir(case_dir):
        log.error("Directorio del caso no encontrado: %s", case_dir)
        sys.exit(1)
    log.info("Caso DOE: %s", case_dir)

    # --- Rama extract: no necesita param.py ni n2m.bat ---
    if args.command == "extract":
        doe_name_eff = args.doe_name if args.doe_name else DOE_NAME
        doe_dir = os.path.join(os.path.dirname(case_dir), doe_name_eff)
        log.info("Extrayendo DOE: %s", doe_dir)
        extract_doe_results(doe_dir, os.path.basename(case_dir),
                            DOE_EXTRACT_SIGNALS, dry_run=args.dry_run)
        return

    # --- Rama merge: fusiona dos doe_results.h5 ---
    if args.command == "merge":
        doe_name_eff = args.doe_name if args.doe_name else DOE_NAME
        if not args.merge_from:
            log.error("--merge_from es obligatorio con --command merge")
            sys.exit(1)
        base_dir  = os.path.join(os.path.dirname(case_dir), doe_name_eff)
        patch_dir = os.path.join(os.path.dirname(case_dir), args.merge_from)

        merge_target = base_dir
        if args.merge_out:
            out_dir = os.path.join(os.path.dirname(case_dir), args.merge_out)
            if args.dry_run:
                log.info("[DRY-RUN] Copiaria %s -> %s (el resto del preview se muestra sobre el original)",
                         base_dir, out_dir)
            else:
                if os.path.isdir(out_dir):
                    log.error("La carpeta destino ya existe: %s", out_dir)
                    sys.exit(1)
                shutil.copytree(base_dir, out_dir)
                log.info("Copiado %s -> %s", base_dir, out_dir)
                merge_target = out_dir

        log.info("Merge: %s  <--  %s", merge_target, patch_dir)
        merge_doe_results(merge_target, patch_dir, dry_run=args.dry_run)
        return

    # 2. Generar casos DOE
    lst_var, val_var = build_doe_cases(DOE_MODE)

    # Resumen + confirmacion (cubre tambien reemplazar un DOE que ya existe)
    doe_name_eff = args.doe_name if args.doe_name else DOE_NAME
    doe_dir = os.path.join(os.path.dirname(case_dir), doe_name_eff)
    print("\n" + describe(doe_dir, case_dir) + "\n")
    if args.dry_run:
        if os.path.exists(doe_dir):
            log.info("DOE existente detectado (dry-run): %s - se ignora el reemplazo.", doe_dir)
    else:
        if not args.yes and input("Lanzar este DOE? [y/N]: ").strip().lower() not in ("y", "yes"):
            log.info("Cancelado: no se lanzo nada.")
            return
        if os.path.exists(doe_dir):
            shutil.rmtree(doe_dir)
            log.info("DOE existente borrado: %s", doe_dir)

    # 3. Cargar entorno Nessy2m (necesario para ambos modos)
    env = load_n2m_env(args.n2m_bat)
    python_exe = get_python_exe(env)
    n2m_bin = env.get("n2m", "")
    command_script = os.path.join(n2m_bin, f"{args.command}.py")
    if not os.path.isfile(command_script):
        log.error("Script no encontrado: %s", command_script)
        sys.exit(1)

    # 4a. Modo --timed: paralelismo en runner, timing por caso
    if args.timed:
        nb_workers = args.workers if args.workers is not None else NB_PROC
        log.info("Workers configurados: %d", nb_workers)
        run_doe_timed(lst_var, val_var, doe_name_eff, case_dir,
                      command_script, env, python_exe,
                      SCRIPT_DIR, nb_workers, args.dry_run)
        _keep_config(cfg_path, doe_dir, args.dry_run)
        if args.auto_extract:
            extract_doe_results(doe_dir, os.path.basename(case_dir),
                                DOE_EXTRACT_SIGNALS, dry_run=args.dry_run)
        return

    # 4b. Modo normal: sobreescribir param.py + un solo n2m_sch
    param_path = os.path.join(case_dir, "p", "param.py")
    param_backup = param_path + ".bak"
    shutil.copy2(param_path, param_backup)
    log.info("Backup de param.py guardado en: %s", param_backup)

    try:
        write_param(param_path, lst_var, val_var, doe_name_eff, NB_PROC)
        log.info("param.py actualizado con %d casos.", len(val_var))

        # 5. Ejecutar n2m_sch
        rc = run_n2m_command(command_script, case_dir, env, python_exe,
                             extra_args=args.args, dry_run=args.dry_run)
    finally:
        # 6. Restaurar param.py original siempre (aunque haya error)
        shutil.copy2(param_backup, param_path)
        os.remove(param_backup)
        log.info("param.py original restaurado.")

    if rc != 0:
        log.error("Comando termino con error: %d", rc)
        sys.exit(rc)
    log.info("Completado exitosamente.")
    _keep_config(cfg_path, doe_dir, args.dry_run)

    if args.auto_extract:
        extract_doe_results(doe_dir, os.path.basename(case_dir),
                            DOE_EXTRACT_SIGNALS, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
