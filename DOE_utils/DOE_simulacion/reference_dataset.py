#!/usr/bin/env python
# coding: utf-8
"""reference_dataset.py — Dataset externo de señales de referencia (Fase 1 + 2A).

Hoy los indicadores (MaxEnt, RMS-CV, SST, Green-Area) calculan su umbral de
detección cortando un tramo de la MISMA señal que analizan. Este módulo arma
un dataset de referencia EXTERNO — señales ya etiquetadas (stable/unstable,
y gray = zona gris de la estrategia "amplitude") por intervalos de tiempo —
que en una fase futura los indicadores podrán consumir en lugar de recortar
su propia señal.

Es agnóstico del origen de los datos: `ReferenceDataset`/`ReferenceSignal` no
saben nada de DOE. `from_doe_h5` es el único adaptador que sabe leer
doe_results.h5 / doe_noise_results.h5 (layout de doe_runner.py).

Secuencia de uso:
    1. make_label_template(doe_results.h5, reference_labels.yaml, strategy=...)
       "manual" (vacío), "eta" (umbral teórico eta; alias "kappa") o "amplitude"
       (práctico/operacional: max|señal| vs % de una variable base del caso)
    2. (a mano) completar/corregir intervalos en reference_labels.yaml
    3. from_doe_h5(doe_results.h5, reference_labels.yaml, channels) -> ReferenceDataset
    4. dataset.to_hdf5("reference_dataset.h5")                      -> portable
    5. (en cualquier lado) ReferenceDataset.from_hdf5(...)
    6. combine_by_label(dataset) -> ReferenceDataset combinado (una señal
       continua por label+canal) -> save_combined("reference_combined.h5")
    7. (en cualquier lado) load_combined(...)

Fuera de alcance por ahora: etiquetado por ventanas de tiempo, features/GMM/GP por pedazo
(Fase 2, opción B), y cualquier cambio en doe_indicators.py o en los indicadores.
"""

from __future__ import annotations

import inspect
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # DOE_utils/
import eta_compat  # noqa: E402  (kappa -> eta: los lectores aceptan ambos nombres)

log = logging.getLogger(__name__)

VALID_LABELS = {"stable", "unstable", "gray"}  # gray: zona gris de la estrategia "amplitude"

# ==============================================================================
# CONFIG — editar acá los defaults del CLI; los flags de línea de comandos
# los pisan si se pasan (mismo patrón que _T_GT/_CUT_START en doe_indicators.py)
# ==============================================================================
DEFAULT_H5_PATH = (
    r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage"
    r"\Chatter-Criteria\CAMP10_Chatter_detection_Methodes\Convergency_Simulation"
    r"\4_DOE_Data_Training_Tube\DOE_Training_Tube_dxl_20e-5_RUN_10_0.5-2.0"
    r"\doe_results.h5"
)

DEFAULT_LABELS_PATH     = "label_amp.yaml"   # None -> "<carpeta de h5_path>/reference_labels.yaml"
DEFAULT_OUT_H5          = None   # None -> "<carpeta de h5_path>/reference_dataset.h5"
DEFAULT_CHANNELS        = None   # None -> autodetecta todos los canales de cada caso
DEFAULT_STRATEGY        = "amplitude" #manual, eta (alias kappa), amplitude
DEFAULT_ETA_THRESHOLD = 1.0
DEFAULT_KAPPA_THRESHOLD = DEFAULT_ETA_THRESHOLD   # alias (antes kappa)
DEFAULT_WARMUP          = 0.0
# --strategy amplitude: max|y| de DEFAULT_AMP_SIGNAL vs % de la base (attr del caso x escala)
DEFAULT_BASE_ATTR       = "$f_tooth$"   # avance por diente, en mm en var_val.py
DEFAULT_BASE_SCALE      = 1e-3          # lleva la base a las unidades de la señal (mm -> m)
DEFAULT_AMP_SIGNAL      = "Axial_disp"  # "Axial_disp_out_deflex" si la deflexión estática pesa
DEFAULT_LIM_INF_PCT     = 10.0          # amp < 10% base -> stable
DEFAULT_LIM_SUP_PCT     = 40.0          # amp > 40% base -> unstable; entre medio -> gray
DEFAULT_IN_H5           = None   # None -> "<carpeta de h5_path>/reference_dataset.h5" (entrada de "combine")
DEFAULT_OUT_COMBINED    = None   # None -> "<carpeta de h5_path>/reference_combined.h5" (salida de "combine")
# casos en rampa de Ap: la regla de amplitud ventana a ventana, con la ventana de los indicadores (PLAN_ramps.md)
DEFAULT_WINDOW_MODE     = "by_revolution"   # by_revolution (T = 60/n del caso) | by_modal (T = 1/f_modal)
DEFAULT_WINDOW_N        = 7.0               # largo de la ventana en T (la de decisión de las variantes, dec7)
DEFAULT_WINDOW_STEP     = 1.0               # paso entre ventanas en T
DEFAULT_T_START         = 0.05   # None -> sin corte al inicio. Recorte fijo de señal (ej. quitar entrada de herramienta)
DEFAULT_T_END           = 14.0   # None -> sin corte al final. Idem para la salida de herramienta


# ==============================================================================
# PIEZA 2 — Esquema canónico
# ==============================================================================

@dataclass
class ReferenceSignal:
    id: str                                    # ej. "case_007/Axial_vel"
    t: np.ndarray                               # señal completa, sin recortar
    y: np.ndarray
    fs: float
    intervals: List[Tuple[float, float, str]]   # [(t0, t1, label), ...]
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReferenceDataset:
    signals: List[ReferenceSignal]

    def segments(self, label: str) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Tramos recortados (id, t, y) de todas las señales que tengan `label`."""
        out: List[Tuple[str, np.ndarray, np.ndarray]] = []
        for sig in self.signals:
            for t0, t1, lab in sig.intervals:
                if lab != label:
                    continue
                mask = (sig.t >= t0) & (sig.t <= t1)
                out.append((sig.id, sig.t[mask], sig.y[mask]))
        return out

    def to_hdf5(self, path: str) -> None:
        """Guarda cada TRAMO etiquetado ya recortado, agrupado por label (stable/unstable/gray).

        Pierde a propósito los tramos sin etiquetar de cada señal (esa cruda
        sigue en el doe_results.h5 de origen) — este .h5 es el material de
        entrenamiento ya recortado, no un espejo lossless de la señal completa.
        """
        with h5py.File(path, "w") as f:
            top = {label: f.create_group(label) for label in sorted(VALID_LABELS)}
            counters: Dict[Tuple[str, str], int] = {}  # (signal_id, label) -> próximo índice
            for sig in self.signals:
                case_name, _, rest = sig.id.partition("/")
                piece_base = (rest or sig.id).replace("/", "__")
                for t0, t1, label in sig.intervals:
                    mask = (sig.t >= t0) & (sig.t <= t1)
                    idx = counters.get((sig.id, label), 0)
                    counters[(sig.id, label)] = idx + 1
                    piece_name = f"{piece_base}__{idx:03d}"

                    case_grp = top[label].require_group(case_name)
                    g = case_grp.create_group(piece_name)
                    g.create_dataset("t", data=sig.t[mask])
                    g.create_dataset("y", data=sig.y[mask])
                    g.attrs["signal_id"] = sig.id
                    g.attrs["t0"] = t0
                    g.attrs["t1"] = t1
                    g.attrs["fs"] = sig.fs
                    for k, v in sig.attrs.items():
                        try:
                            g.attrs[k] = v
                        except Exception:
                            g.attrs[k] = str(v)
                    g.attrs.update(_piece_depth(sig, t0, t1))

    @classmethod
    def from_hdf5(cls, path: str) -> "ReferenceDataset":
        """Reconstruye un ReferenceSignal por tramo guardado (cada uno con su único intervalo)."""
        signals = []
        with h5py.File(path, "r") as f:
            for label in sorted(VALID_LABELS):
                if label not in f:
                    continue
                for case_name in f[label].keys():
                    case_grp = f[label][case_name]
                    for piece_name in case_grp.keys():
                        g = case_grp[piece_name]
                        t = g["t"][()]
                        y = g["y"][()]
                        attrs = dict(g.attrs)
                        sig_id = attrs.pop("signal_id", f"{case_name}/{piece_name}")
                        t0 = attrs.pop("t0")
                        t1 = attrs.pop("t1")
                        fs = attrs.pop("fs", 1.0 / float(t[1] - t[0]))
                        signals.append(ReferenceSignal(
                            id=f"{sig_id}#{case_name}/{piece_name}", t=t, y=y, fs=fs,
                            intervals=[(float(t0), float(t1), label)], attrs=attrs,
                        ))
        return cls(signals=signals)


def _piece_depth(sig: "ReferenceSignal", t0: float, t1: float) -> dict:
    """Ap [mm] (y eta, si el caso lo tiene) al inicio y al fin de una pieza: en una rampa, Ap(t) lineal sobre
    la señal entera del caso (experiment.ap_of_t; signal_t0/t1 = su intervalo antes del recorte); constante:
    el Ap del caso en los dos extremos. {} si el caso no tiene $Ap_start$."""
    a = sig.attrs
    if "$Ap_start$" not in a:
        return {}
    span = (float(a.get("signal_t0", sig.t[0])), float(a.get("signal_t1", sig.t[-1])))
    ap0, ap1 = (_experiment().ap_of_t(a, t, span) for t in (t0, t1))
    out = {"Ap_start_mm": ap0 * 1e3, "Ap_end_mm": ap1 * 1e3}
    if is_ramp(a) and eta_compat.has(a, "eta_start") and eta_compat.has(a, "eta_end"):   # eta = Ap / limit at the case's n: linear too
        k0, k1 = (float(eta_compat.get(a, k)) for k in ("eta_start", "eta_end"))
        a0, a1 = (float(a[k]) for k in ("$Ap_start$", "$Ap_end$"))
        out.update({f"eta_{w}": k0 + (k1 - k0) * (ap - a0) / (a1 - a0) for w, ap in (("t0", ap0), ("t1", ap1))})
    return out


# ==============================================================================
# PIEZA 1 — Etiquetado
# ==============================================================================

def _label_manual(grp_name: str, attrs: dict, t_range: Tuple[float, float], grp) -> List[Tuple[float, float, str]]:
    """Estrategia por defecto: no etiqueta nada, el usuario completa a mano."""
    return []


def _label_by_eta(
    grp_name: str, attrs: dict, t_range: Tuple[float, float], grp,
    threshold: float = 1.0, warmup: float = 0.0,
) -> List[Tuple[float, float, str]]:
    """Etiqueta la señal entera (menos `warmup` al inicio) por umbral de eta."""
    if not eta_compat.has(attrs, "eta"):
        log.warning("Grupo '%s' sin attr 'kappa' (ni 'eta') — se deja sin etiquetar", grp_name)
        return []
    label = "stable" if eta_compat.get(attrs, "eta") < threshold else "unstable"
    return [(t_range[0] + warmup, t_range[1], label)]


def is_ramp(attrs: dict) -> bool:
    """Caso con rampa de Ap ($Ap_end$ != $Ap_start$; misma definición que experiment.is_ramp)."""
    try:
        return abs(float(attrs["$Ap_end$"]) - float(attrs["$Ap_start$"])) > 1e-9
    except (KeyError, TypeError, ValueError):
        return False


def _experiment():
    """DOE_utils/experiment.py: las definiciones comunes de las rampas (ap_of_t, resolve_physics)."""
    import sys
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:
        sys.path.insert(0, here)
    import experiment
    return experiment


def window_seconds(attrs: dict, window_mode: str, window_N: float, window_step: float,
                   f_modal: Optional[float] = None) -> Tuple[float, float]:
    """(largo, paso) [s] de la ventana de etiquetado en un caso, con la misma resolución que los indicadores:
    by_revolution -> N x T_rev (T_rev = 60 / $spin_rate$ del caso), by_modal -> N x T_modal (1 / f_modal)."""
    if window_mode not in ("by_revolution", "by_modal"):
        raise ValueError(f"window_mode {window_mode!r}: by_revolution o by_modal")
    spin = attrs.get("$spin_rate$")
    t_rev = 60.0 / float(spin) if spin else None
    t_modal = 1.0 / float(f_modal) if f_modal else None
    unit = "T_rev" if window_mode == "by_revolution" else "T_modal"
    rp = _experiment().resolve_physics
    return float(rp(f"{float(window_N)}*{unit}", t_rev, t_modal)), float(rp(f"{float(window_step)}*{unit}", t_rev, t_modal))


def windowed_labels(t: np.ndarray, y: np.ndarray, t0: float, t1: float, win: float, step: float,
                    lim_inf: float, lim_sup: float) -> List[Tuple[float, float, str]]:
    """Regla de amplitud ventana a ventana: ventanas de largo `win` que empiezan en t0, t0 + step, ... (enteras
    dentro de [t0, t1]); en cada una max|y| > lim_sup -> unstable, < lim_inf -> stable, entre medio -> gray.
    Cada instante lleva la etiqueta de la última ventana que empezó antes que él (los solapes del paso se
    resuelven así); ventanas seguidas iguales forman un intervalo y la última llega hasta t1. Sin suavizado: si
    las etiquetas alternan cerca del umbral, quedan tal cual. Señal más corta que una ventana: una sola ventana."""
    if win <= 0 or step <= 0:
        raise ValueError(f"ventana {win} s y paso {step} s deben ser > 0")
    starts = np.arange(t0, t1 - win + 1e-12, step) if t1 - t0 >= win else np.array([t0])
    ay = np.abs(y)
    i0 = np.searchsorted(t, starts, side="left")
    i1 = np.searchsorted(t, np.minimum(starts + win, t1), side="left")
    i1 = np.where(starts + win >= t1, np.searchsorted(t, t1, side="right"), i1)   # the last one includes t1
    labs = []
    for a, b in zip(i0, i1):
        amp = float(ay[a:b].max()) if b > a else 0.0
        labs.append("unstable" if amp > lim_sup else ("stable" if amp < lim_inf else "gray"))
    out: List[Tuple[float, float, str]] = []
    for i, (s, lab) in enumerate(zip(starts, labs)):
        end = float(starts[i + 1]) if i + 1 < len(starts) else float(t1)
        if out and out[-1][2] == lab:
            out[-1] = (out[-1][0], end, lab)
        else:
            out.append((float(s), end, lab))
    return out


def _label_by_amplitude(
    grp_name: str, attrs: dict, t_range: Tuple[float, float], grp,
    base_attr: str = DEFAULT_BASE_ATTR, base_scale: float = DEFAULT_BASE_SCALE,
    signal: str = DEFAULT_AMP_SIGNAL, lim_inf_pct: float = DEFAULT_LIM_INF_PCT,
    lim_sup_pct: float = DEFAULT_LIM_SUP_PCT, warmup: float = 0.0,
    window_mode: str = DEFAULT_WINDOW_MODE, window_N: float = DEFAULT_WINDOW_N,
    window_step: float = DEFAULT_WINDOW_STEP, f_modal: Optional[float] = None,
) -> List[Tuple[float, float, str]]:
    """Criterio práctico/operacional: etiqueta la señal entera (menos `warmup`)
    por su amplitud cruda max|y| frente a un % de una variable base del caso
    (ej. avance por diente): amp > lim_sup% -> unstable, amp < lim_inf% ->
    stable, entre medio -> gray.

    base = attrs[base_attr] * base_scale -- `base_scale` lleva la base a las
    unidades de `signal` ($f_tooth$ en mm, Axial_disp en m -> 1e-3). Solo
    tiene sentido si base y señal son la misma magnitud (avance <-> desplazamiento).
    max|y| incluye el offset de deflexión estática; si pesa frente a lim_inf,
    usar signal="Axial_disp_out_deflex".

    Caso en RAMPA de Ap ($Ap_end$ != $Ap_start$, PLAN_ramps.md): la misma regla, ventana a ventana
    (`windowed_labels`), con ventanas de `window_N` x T y paso `window_step` x T resueltos como en los
    indicadores (window_mode by_revolution: T = T_rev = 60/n del caso; by_modal: T = 1/f_modal). Varios
    intervalos por caso. Los casos constantes no cambian (una ventana = la señal entera).
    """
    if not 0 <= lim_inf_pct < lim_sup_pct:
        raise ValueError(f"se espera 0 <= lim_inf_pct < lim_sup_pct, dio {lim_inf_pct} / {lim_sup_pct}")
    base = float(attrs.get(base_attr, 0.0)) * base_scale
    if base <= 0:
        log.warning("Grupo '%s' sin attr '%s' válido — se deja sin etiquetar", grp_name, base_attr)
        return []
    ch_grp = _resolve_channel_group(grp, signal)
    if ch_grp is None:
        log.warning("Grupo '%s' sin señal '%s' — se deja sin etiquetar", grp_name, signal)
        return []

    t0, t1 = t_range[0] + warmup, t_range[1]
    t = ch_grp["time"][()]
    mask = (t >= t0) & (t <= t1)
    if not mask.any():
        log.warning("Grupo '%s': '%s' sin muestras en [%s, %s] — se deja sin etiquetar", grp_name, signal, t0, t1)
        return []
    if is_ramp(attrs):
        win, step = window_seconds(attrs, window_mode, window_N, window_step, f_modal)
        out = windowed_labels(t, ch_grp["values"][()], t0, t1, win, step, lim_inf_pct / 100.0 * base,
                              lim_sup_pct / 100.0 * base)
        log.info("%s (rampa): ventanas de %.4f s, paso %.4f s -> %s", grp_name, win, step,
                 " | ".join(f"{a:.3f}-{b:.3f} {lab}" for a, b, lab in out))
        return out
    amp = float(np.abs(ch_grp["values"][()][mask]).max())

    if amp > lim_sup_pct / 100.0 * base:
        label = "unstable"
    elif amp < lim_inf_pct / 100.0 * base:
        label = "stable"
    else:
        label = "gray"
    log.info("%s: max|%s| = %.3e = %.1f%% de base %.3e -> %s", grp_name, signal, amp, 100.0 * amp / base, base, label)
    return [(t0, t1, label)]


# Punto de extensión: sumar acá una estrategia nueva sin tocar make_label_template.
# Firma: (grp_name, attrs, t_range, grp, **kwargs) -> [(t0, t1, label), ...]
LABEL_STRATEGIES = {
    "manual":    _label_manual,
    "eta":       _label_by_eta,
    "kappa":     _label_by_eta,   # alias: eta se llamaba kappa
    "amplitude": _label_by_amplitude,
}


def _format_intervals(intervals: List[Tuple[float, float, str]]) -> str:
    if not intervals:
        return "[]"
    parts = ", ".join(f'[{t0}, {t1}, "{label}"]' for t0, t1, label in intervals)
    return f"[{parts}]"


def _masked_range(
    time_ds, t_start: Optional[float], t_end: Optional[float]
) -> Optional[Tuple[float, float]]:
    """Rango [t0, t1] real tras aplicar t_start/t_end, con el mismo criterio de
    máscara que usa `from_doe_h5` (t >= lo) & (t <= hi) -- así lo que la
    plantilla muestra/etiqueta con la estrategia "eta" siempre cae dentro
    de lo que `build` va a aceptar, sin asumir que t_start/t_end caen justo
    en un punto de la grilla de muestreo. None si no queda ninguna muestra.
    """
    if t_start is None and t_end is None:
        return float(time_ds[0]), float(time_ds[-1])
    t_full = time_ds[()]
    lo = t_full[0] if t_start is None else t_start
    hi = t_full[-1] if t_end is None else t_end
    mask = (t_full >= lo) & (t_full <= hi)
    if not mask.any():
        return None
    t_masked = t_full[mask]
    return float(t_masked[0]), float(t_masked[-1])


_OUT_DEFLEX_GROUP = "Out_Deflex"  # escrito por Etapa_1.py / static_deflection.py, no por este módulo


def _discover_channels(grp) -> List[str]:
    """Subgrupos de `grp` que son canales de señal (tienen dataset 'time' y 'values').

    Incluye también los que están un nivel más adentro, en `Out_Deflex` (ej.
    `Axial_disp_out_deflex`), si esa etapa ya corrió sobre este doe_results.h5.
    """
    channels = [
        key for key in grp.keys()
        if isinstance(grp[key], h5py.Group) and "time" in grp[key] and "values" in grp[key]
    ]
    out_grp = grp.get(_OUT_DEFLEX_GROUP)
    if isinstance(out_grp, h5py.Group):
        channels += [
            key for key in out_grp.keys()
            if isinstance(out_grp[key], h5py.Group) and "time" in out_grp[key] and "values" in out_grp[key]
        ]
    return channels


def _resolve_channel_group(grp, channel: str):
    """Grupo con 'time'/'values' para `channel` -- directo bajo `grp`, o (si no
    está ahí) dentro de `Out_Deflex`. None si no se encuentra en ningún lado."""
    if channel in grp and isinstance(grp[channel], h5py.Group) and "time" in grp[channel] and "values" in grp[channel]:
        return grp[channel]
    out_grp = grp.get(_OUT_DEFLEX_GROUP)
    if isinstance(out_grp, h5py.Group) and channel in out_grp:
        sub = out_grp[channel]
        if isinstance(sub, h5py.Group) and "time" in sub and "values" in sub:
            return sub
    return None


def make_label_template(
    h5_path: str, out_yaml: str, strategy: str = "manual",
    t_start: Optional[float] = None, t_end: Optional[float] = None,
    **strategy_kwargs,
) -> None:
    """Genera `out_yaml` con todos los grupos de `h5_path`.

    `strategy` decide el primer pase de etiquetado ("manual" = todo vacío,
    el default de siempre); el YAML resultante sigue siendo editable a mano
    después, sea cual sea la estrategia usada para generarlo.

    `t_start`/`t_end`: recorte fijo de la señal antes de calcular el rango
    (ej. para descartar entrada/salida de herramienta) — mismo recorte que
    aplica `from_doe_h5`, así la plantilla ya refleja el rango realmente
    disponible y no ofrece etiquetar algo que `build` después rechazaría.

    Se niega a sobrescribir un YAML ya existente, para no perder etiquetas
    hechas a mano.
    """
    if os.path.exists(out_yaml):
        raise FileExistsError(
            f"{out_yaml} ya existe — no se sobrescribe (podrías perder etiquetas hechas a mano)."
        )
    strategy = "eta" if strategy == "kappa" else strategy   # alias: el YAML escribe eta
    label_fn = LABEL_STRATEGIES[strategy]

    # Parámetros efectivos de la estrategia (defaults de la función + los pasados) ->
    # bloque "labeling" del YAML; build lo copia a cada tramo (el visor dibuja con
    # eso los límites de "amplitude").
    labeling = {"strategy": strategy}
    labeling.update({
        k: p.default for k, p in inspect.signature(label_fn).parameters.items()
        if p.default is not inspect.Parameter.empty
    })
    labeling.update(strategy_kwargs)
    labeling = {k: v for k, v in labeling.items() if v is not None}   # f_modal sin by_modal: no se escribe
    labeling_line = yaml.safe_dump(
        {"labeling": labeling}, default_flow_style=None, sort_keys=False, width=10**6,
    ).strip()

    lines = [f"source: {os.path.basename(h5_path)}", labeling_line, "cases:"]
    with h5py.File(h5_path, "r") as f:
        for grp_name in sorted(f.keys()):
            grp = f[grp_name]
            attrs = dict(grp.attrs)
            ramp = is_ramp(attrs)   # una rampa: su 'eta' (= el de inicio, si existe) no se muestra
            kappa_bits = {k: v for k, v in attrs.items() if str(k).startswith(("kappa", "eta")) and not (ramp and k in ("kappa", "eta"))}
            if ramp:
                kappa_bits = {"ramp Ap_mm": f"{float(attrs['$Ap_start$']) * 1e3:g}->{float(attrs['$Ap_end$']) * 1e3:g}",
                              **kappa_bits}

            case_channels = _discover_channels(grp)
            t_range = None
            if case_channels:
                first_ch_grp = _resolve_channel_group(grp, case_channels[0])
                t_range = _masked_range(first_ch_grp["time"], t_start, t_end)

            intervals = label_fn(grp_name, attrs, t_range, grp, **strategy_kwargs) if t_range is not None else []

            comment_bits = []
            if kappa_bits:
                comment_bits.append(", ".join(f"{k}={v}" for k, v in kappa_bits.items()))
            if t_range is not None:
                comment_bits.append(f"t=[{t_range[0]:.2f}, {t_range[1]:.2f}] s")
            comment = f"  # {'   '.join(comment_bits)}" if comment_bits else ""

            lines.append(f"  {grp_name}: {_format_intervals(intervals)}{comment}")

    with open(out_yaml, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _parse_labels_file(labels_path: str) -> Dict[str, List[Tuple[float, float, str]]]:
    """Lee y valida la sintaxis de `labels_path` (sin tocar el .h5)."""
    with open(labels_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cases = raw.get("cases") or {}

    parsed: Dict[str, List[Tuple[float, float, str]]] = {}
    for grp_name, raw_intervals in cases.items():
        intervals: List[Tuple[float, float, str]] = []
        for iv in raw_intervals or []:
            if len(iv) != 3:
                raise ValueError(
                    f"{labels_path}: intervalo inválido en '{grp_name}': {iv!r} (esperado [t0, t1, label])"
                )
            t0, t1, label = float(iv[0]), float(iv[1]), iv[2]
            if label not in VALID_LABELS:
                raise ValueError(
                    f"{labels_path}: label desconocido '{label}' en '{grp_name}' (válidos: {sorted(VALID_LABELS)})"
                )
            if not (t0 < t1):
                raise ValueError(f"{labels_path}: intervalo inválido en '{grp_name}': t0={t0} >= t1={t1}")
            intervals.append((t0, t1, label))

        intervals.sort(key=lambda iv: iv[0])
        for a, b in zip(intervals, intervals[1:]):
            if a[1] > b[0]:
                raise ValueError(f"{labels_path}: intervalos solapados en '{grp_name}': {a} y {b}")
        parsed[grp_name] = intervals

    return parsed


# ==============================================================================
# PIEZA 3 — Adaptador de origen (el único que conoce doe_runner)
# ==============================================================================

def from_doe_h5(
    h5_path: str, labels_path: str, channels: Optional[List[str]] = None,
    t_start: Optional[float] = None, t_end: Optional[float] = None,
) -> ReferenceDataset:
    """Construye un ReferenceDataset a partir de un doe_results.h5 + su YAML de etiquetas.

    `channels`: lista explícita de canales a usar, o None (default) para
    incluir TODOS los canales disponibles de cada caso (autodetectados) —
    qué canal usar queda para la Fase 3, acá se guardan todos.

    `t_start`/`t_end`: recorta la señal a ese rango ANTES de todo lo demás
    (ej. para descartar entrada/salida de herramienta) — None = sin recorte
    por ese lado. Los intervalos etiquetados tienen que caer dentro de lo
    que quede tras el recorte, si no, ValueError (mismo chequeo que si
    cayeran fuera del rango real de la señal).

    El bloque "labeling" del YAML (cómo se etiquetó, lo escribe make_label_template)
    se copia a los attrs de cada señal como `labeling_<clave>`; YAML sin ese
    bloque (viejos o hechos a mano) -> no se agrega nada.

    NO importa doe_indicators.py (acoplaría el dataset a los 4 indicadores) —
    la lectura de señal/attrs se replica acá, igual layout que `_load_case`.
    """
    cases = _parse_labels_file(labels_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        labeling = (yaml.safe_load(f) or {}).get("labeling") or {}
    signals: List[ReferenceSignal] = []

    with h5py.File(h5_path, "r") as f:
        for grp_name, intervals in cases.items():
            if not intervals:
                continue  # sin etiquetar -> se ignora
            if grp_name not in f:
                raise KeyError(f"Grupo '{grp_name}' de {labels_path} no existe en {h5_path}")

            grp = f[grp_name]
            base_attrs = dict(grp.attrs)
            case_channels = channels if channels is not None else _discover_channels(grp)

            for ch in case_channels:
                ch_grp = _resolve_channel_group(grp, ch)
                if ch_grp is None:
                    log.warning("Señal '%s' no está en grupo '%s' — omitida", ch, grp_name)
                    continue

                t = ch_grp["time"][()]
                y = ch_grp["values"][()]
                span = (float(t[0]), float(t[-1]))   # Ap(t) de una rampa va de Ap_start a Ap_end sobre esto
                if t_start is not None or t_end is not None:
                    lo = t[0] if t_start is None else t_start
                    hi = t[-1] if t_end is None else t_end
                    crop_mask = (t >= lo) & (t <= hi)
                    if not crop_mask.any():
                        raise ValueError(
                            f"t_start/t_end [{lo}, {hi}] no deja ninguna muestra en '{grp_name}/{ch}' "
                            f"(rango real [{t[0]}, {t[-1]}])"
                        )
                    t, y = t[crop_mask], y[crop_mask]
                t0_sig, t1_sig = float(t[0]), float(t[-1])
                for a, b, _ in intervals:
                    if a < t0_sig or b > t1_sig:
                        raise ValueError(
                            f"Intervalo [{a}, {b}] fuera del rango de la señal "
                            f"[{t0_sig}, {t1_sig}] en '{grp_name}/{ch}'"
                        )

                attrs = dict(base_attrs)
                attrs["source"] = "doe_h5"
                attrs["source_file"] = os.path.basename(h5_path)
                attrs["group"] = grp_name
                attrs["channel"] = ch
                attrs["signal_t0"], attrs["signal_t1"] = span
                attrs.update({f"labeling_{k}": v for k, v in labeling.items()})

                signals.append(ReferenceSignal(
                    id=f"{grp_name}/{ch}",
                    t=t, y=y,
                    fs=1.0 / float(t[1] - t[0]),
                    intervals=list(intervals),
                    attrs=attrs,
                ))

    return ReferenceDataset(signals=signals)


# ==============================================================================
# PIEZA 4 — Fase 2, opción A: combinar piezas del mismo (label, canal)
# ==============================================================================

def combine_by_label(dataset: ReferenceDataset) -> ReferenceDataset:
    """Concatena todas las piezas del mismo (label, canal) en una señal continua.

    Cada señal de `dataset.signals` se asume ya "una pieza de un solo label"
    (el resultado de ReferenceDataset.from_hdf5 tras el rediseño stable/unstable).
    fs se asume igual entre piezas del mismo grupo — si no lo es, ValueError
    explícito, NO se resamplea.
    """
    groups: Dict[Tuple[str, Optional[str]], List[ReferenceSignal]] = {}
    for sig in dataset.signals:
        labels = {lab for _, _, lab in sig.intervals}
        if len(labels) > 1:
            raise ValueError(f"Señal '{sig.id}' mezcla labels distintos: {sorted(labels)}")
        if not labels:
            continue  # sin intervalos -> nada que combinar
        label = next(iter(labels))
        channel = sig.attrs.get("channel")
        groups.setdefault((label, channel), []).append(sig)

    combined_signals: List[ReferenceSignal] = []
    for (label, channel), pieces in groups.items():
        fs0 = pieces[0].fs
        for p in pieces:
            if abs(p.fs - fs0) > 1e-9:
                raise ValueError(
                    f"fs distinto en el grupo (label={label!r}, channel={channel!r}): "
                    f"'{p.id}' tiene fs={p.fs}, esperado fs={fs0} (no se resamplea)"
                )

        y = np.concatenate([p.y for p in pieces])
        t = np.arange(len(y)) / fs0
        combined_signals.append(ReferenceSignal(
            id=f"{label}/{channel}",
            t=t, y=y, fs=fs0,
            intervals=[(0.0, float(t[-1]), label)],
            attrs={
                "label": label,
                "channel": channel,
                "n_pieces": len(pieces),
                "source_ids": [p.id for p in pieces],
                "piece_lengths": [len(p.y) for p in pieces],  # una entrada por source_ids, mismo orden
            },
        ))

    return ReferenceDataset(signals=combined_signals)


def save_combined(dataset: ReferenceDataset, path: str) -> None:
    """Persiste el resultado de `combine_by_label`: una señal continua por grupo."""
    with h5py.File(path, "w") as f:
        for sig in dataset.signals:
            grp = f.create_group(sig.id.replace("/", "__"))
            grp.create_dataset("t", data=sig.t)
            grp.create_dataset("y", data=sig.y)
            grp.attrs["id"] = sig.id
            grp.attrs["fs"] = sig.fs
            for k, v in sig.attrs.items():
                if k == "source_ids":
                    grp.create_dataset("source_ids", data=np.array(v, dtype=object), dtype=h5py.string_dtype())
                    continue
                if k == "piece_lengths":
                    grp.create_dataset("piece_lengths", data=np.array(v, dtype=np.int64))
                    continue
                try:
                    grp.attrs[k] = v
                except Exception:
                    grp.attrs[k] = str(v)


def load_combined(path: str) -> ReferenceDataset:
    """Recarga lo que escribió `save_combined` como un ReferenceDataset normal."""
    signals = []
    with h5py.File(path, "r") as f:
        for grp_name in f.keys():
            grp = f[grp_name]
            t = grp["t"][()]
            y = grp["y"][()]
            attrs = dict(grp.attrs)
            sig_id = attrs.pop("id", grp_name)
            fs = attrs.pop("fs", 1.0 / float(t[1] - t[0]))
            if "source_ids" in grp:
                attrs["source_ids"] = [
                    s.decode() if isinstance(s, bytes) else str(s) for s in grp["source_ids"][()]
                ]
            if "piece_lengths" in grp:
                attrs["piece_lengths"] = [int(n) for n in grp["piece_lengths"][()]]
            label = attrs.get("label")
            intervals = [(0.0, float(t[-1]), label)] if label is not None else []
            signals.append(ReferenceSignal(id=sig_id, t=t, y=y, fs=fs, intervals=intervals, attrs=attrs))
    return ReferenceDataset(signals=signals)


# ==============================================================================
# SELF-TEST
# ==============================================================================

def _self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        h5_path = os.path.join(tmp, "doe_results.h5")
        yaml_path = os.path.join(tmp, "reference_labels.yaml")

        # 1. .h5 sintético con layout DOE (3 grupos, canal Axial_vel, attrs kappa: un .h5 viejo)
        t = np.linspace(0.0, 10.0, 1001)
        with h5py.File(h5_path, "w") as f:
            for i, kappa in enumerate([0.8, 1.4, 1.1]):
                grp = f.create_group(f"case_{i:03d}")
                grp.attrs["kappa"] = kappa
                sub = grp.create_group("Axial_vel")
                sub.create_dataset("time", data=t)
                sub.create_dataset("values", data=np.sin(2 * np.pi * 5 * t) + i)

        # 2. make_label_template -> YAML con los 3 grupos vacíos; no sobrescribe
        make_label_template(h5_path, yaml_path)
        with open(yaml_path, encoding="utf-8") as f:
            content = f.read()
        assert "case_000" in content and "case_001" in content and "case_002" in content
        assert "kappa=0.8" in content
        try:
            make_label_template(h5_path, yaml_path)
            raise AssertionError("debía fallar por sobrescritura")
        except FileExistsError:
            pass

        # 2b. estrategia "kappa" (alias de eta): stable / unstable / sin eta (-> [] + warning, no error)
        kappa_h5 = os.path.join(tmp, "kappa_doe.h5")
        kappa_yaml = os.path.join(tmp, "kappa_labels.yaml")
        with h5py.File(kappa_h5, "w") as f:
            grp = f.create_group("case_low")
            grp.attrs["kappa"] = 0.5
            sub = grp.create_group("Axial_vel")
            sub.create_dataset("time", data=t)
            sub.create_dataset("values", data=t)

            grp = f.create_group("case_high")
            grp.attrs["kappa"] = 1.5
            sub = grp.create_group("Axial_vel")
            sub.create_dataset("time", data=t)
            sub.create_dataset("values", data=t)

            grp = f.create_group("case_no_kappa")
            sub = grp.create_group("Axial_vel")
            sub.create_dataset("time", data=t)
            sub.create_dataset("values", data=t)

            for name, e in (("case_eta_low", 0.5), ("case_eta_high", 1.5)):   # kappa -> eta: attr eta, no kappa
                grp = f.create_group(name)
                grp.attrs["eta"] = e
                sub = grp.create_group("Axial_vel")
                sub.create_dataset("time", data=t)
                sub.create_dataset("values", data=t)

        make_label_template(kappa_h5, kappa_yaml, strategy="kappa", threshold=1.0, warmup=0.5)   # alias of "eta"
        assert "strategy: eta" in open(kappa_yaml, encoding="utf-8").read()
        kappa_cases = _parse_labels_file(kappa_yaml)
        assert kappa_cases["case_low"] == [(0.5, 10.0, "stable")], kappa_cases["case_low"]
        assert kappa_cases["case_high"] == [(0.5, 10.0, "unstable")], kappa_cases["case_high"]
        assert kappa_cases["case_no_kappa"] == [], kappa_cases["case_no_kappa"]
        assert kappa_cases["case_eta_low"] == [(0.5, 10.0, "stable")] and kappa_cases["case_eta_high"] == [(0.5, 10.0, "unstable")], kappa_cases
        assert "eta=1.5" in open(kappa_yaml, encoding="utf-8").read()

        # 2c. estrategia "amplitude": max|y| vs % de $f_tooth$ (0.05 mm -> 5e-5 m):
        # 10% = 5e-6 m, 40% = 2e-5 m -> stable / gray / unstable / sin base (-> [])
        amp_h5 = os.path.join(tmp, "amp_doe.h5")
        amp_yaml = os.path.join(tmp, "amp_labels.yaml")
        with h5py.File(amp_h5, "w") as f:
            for name, amp in (("case_stable", 1e-6), ("case_gray", 1e-5), ("case_unstable", 3e-5), ("case_no_base", 3e-5)):
                grp = f.create_group(name)
                if name != "case_no_base":
                    grp.attrs["$f_tooth$"] = 0.05
                y = amp * np.sin(2 * np.pi * 5 * t)
                if name == "case_stable":
                    y[t < 0.3] = 1.0  # pico antes del warmup -> no cuenta
                sub = grp.create_group("Axial_disp")
                sub.create_dataset("time", data=t)
                sub.create_dataset("values", data=y)

        make_label_template(amp_h5, amp_yaml, strategy="amplitude", warmup=0.5)
        amp_cases = _parse_labels_file(amp_yaml)
        assert amp_cases["case_stable"] == [(0.5, 10.0, "stable")], amp_cases["case_stable"]
        assert amp_cases["case_gray"] == [(0.5, 10.0, "gray")], amp_cases["case_gray"]
        assert amp_cases["case_unstable"] == [(0.5, 10.0, "unstable")], amp_cases["case_unstable"]
        assert amp_cases["case_no_base"] == [], amp_cases["case_no_base"]

        try:
            make_label_template(amp_h5, os.path.join(tmp, "amp_bad.yaml"), strategy="amplitude",
                                lim_inf_pct=40.0, lim_sup_pct=10.0)
            raise AssertionError("debía fallar: lim_inf_pct >= lim_sup_pct")
        except ValueError:
            pass

        # "gray" pasa build -> to_hdf5 -> from_hdf5 -> combine como un label más
        amp_out = os.path.join(tmp, "amp_dataset.h5")
        from_doe_h5(amp_h5, amp_yaml).to_hdf5(amp_out)
        ds_amp = ReferenceDataset.from_hdf5(amp_out)
        assert [sid.split("#")[0] for sid, _, _ in ds_amp.segments("gray")] == ["case_gray/Axial_disp"]

        # bloque "labeling": el YAML guarda los parámetros efectivos (defaults + pasados)
        # y build los copia a cada tramo -> el visor puede dibujar ±lim_inf/±lim_sup
        with open(amp_yaml, encoding="utf-8") as f:
            amp_labeling = yaml.safe_load(f)["labeling"]
        assert amp_labeling == {
            "strategy": "amplitude", "base_attr": "$f_tooth$", "base_scale": 1e-3, "signal": "Axial_disp",
            "lim_inf_pct": 10.0, "lim_sup_pct": 40.0, "warmup": 0.5, "window_mode": "by_revolution",
            "window_N": 7.0, "window_step": 1.0,
        }, amp_labeling
        gray_piece = next(s for s in ds_amp.signals if s.intervals[0][2] == "gray")
        assert gray_piece.attrs["labeling_strategy"] == "amplitude"
        assert gray_piece.attrs["labeling_signal"] == "Axial_disp"
        assert gray_piece.attrs["labeling_lim_sup_pct"] == 40.0
        assert gray_piece.attrs[gray_piece.attrs["labeling_base_attr"]] == 0.05
        assert {s.id for s in combine_by_label(ds_amp).signals} == {
            "stable/Axial_disp", "gray/Axial_disp", "unstable/Axial_disp",
        }

        # 2d. rampas de Ap: la regla de amplitud ventana a ventana (ventana 7 vueltas, paso 1; n = 6000 rpm ->
        # T_rev = 0.01 s); lim_inf 5e-6 m, lim_sup 2e-5 m. Los casos constantes siguen igual (una ventana).
        ramp_h5, ramp_yaml = os.path.join(tmp, "ramp_doe.h5"), os.path.join(tmp, "ramp_labels.yaml")
        tr = np.arange(0.0, 10.0 + 1e-9, 1e-3)
        carrier = np.sin(2 * np.pi * 50 * tr)
        amps = {"up": 3e-5 * tr / 10, "down": 3e-5 * (1 - tr / 10), "flat": np.full(tr.shape, 1e-6),
                "alt": 2e-5 + 1e-6 * np.sin(2 * np.pi * 1.0 * tr), "const": 3e-5 * tr / 10}
        with h5py.File(ramp_h5, "w") as f:
            for name, amp in amps.items():
                grp = f.create_group(f"case_{name}")
                a0, a1 = (0.005, 0.015) if name != "down" else (0.015, 0.005)
                grp.attrs.update({"$f_tooth$": 0.05, "$spin_rate$": 6000.0, "$Ap_start$": a0,
                                  "$Ap_end$": a0 if name == "const" else a1, "kappa": 0.58,
                                  "kappa_start": a0 / 0.0086, "kappa_end": a1 / 0.0086})
                sub = grp.create_group("Axial_disp")
                sub.create_dataset("time", data=tr)
                sub.create_dataset("values", data=amp * carrier)
        make_label_template(ramp_h5, ramp_yaml, strategy="amplitude", t_start=None, t_end=None)
        rc = _parse_labels_file(ramp_yaml)
        labs = lambda c: [lab for _, _, lab in rc[c]]   # noqa: E731
        assert rc["case_const"] == [(0.0, 10.0, "unstable")], rc["case_const"]          # whole signal, as before
        assert labs("case_up") == ["stable", "gray", "unstable"], rc["case_up"]
        t_on = rc["case_up"][2][0]                     # start of the first unstable window: amp(s + 0.07) > 2e-5
        assert abs(t_on - 6.60) < 0.02 and rc["case_up"][-1][1] == 10.0, rc["case_up"]
        assert abs(rc["case_up"][1][0] - (5e-6 / 3e-6 - 0.07)) < 0.02                  # first gray window
        assert labs("case_down") == ["unstable", "gray", "stable"] and rc["case_down"][0][0] == 0.0, rc["case_down"]
        assert rc["case_flat"] == [(0.0, 10.0, "stable")], rc["case_flat"]               # a ramp that does not cross
        alt = rc["case_alt"]
        assert len(alt) > 10 and set(labs("case_alt")) == {"unstable", "gray"}, alt     # alternates: left as it is
        assert all(a[1] == b[0] and a[2] != b[2] for a, b in zip(alt, alt[1:]))         # touching, never merged
        with open(ramp_yaml, encoding="utf-8") as f:
            txt = f.read()
        assert "ramp Ap_mm=5->15" in txt and "kappa=0.58" not in txt.split("case_up")[1].split("\n")[0], txt
        # windows by_modal (f_modal 100 Hz -> T = 0.01 s, same windows) and an explicit window of 14 x T, step 2
        rm_yaml = os.path.join(tmp, "ramp_modal.yaml")
        make_label_template(ramp_h5, rm_yaml, strategy="amplitude", t_start=None, t_end=None,
                            window_mode="by_modal", f_modal=100.0)
        assert _parse_labels_file(rm_yaml)["case_up"] == rc["case_up"]
        assert window_seconds({"$spin_rate$": 6000.0}, "by_revolution", 14, 2) == (0.14, 0.02)
        try:
            window_seconds({"$spin_rate$": 6000.0}, "by_modal", 7, 1)
            raise AssertionError("by_modal without f_modal accepted")
        except ValueError:
            pass
        # build: every piece keeps Ap (and kappa) at its two ends, Ap(t) linear over the whole signal
        ramp_out = os.path.join(tmp, "ramp_dataset.h5")
        rb_yaml = os.path.join(tmp, "ramp_labels_build.yaml")
        make_label_template(ramp_h5, rb_yaml, strategy="amplitude", t_start=1.0, t_end=None)
        from_doe_h5(ramp_h5, rb_yaml, t_start=1.0, t_end=None).to_hdf5(ramp_out)
        with h5py.File(ramp_out, "r") as f:
            un = f["unstable/case_up/Axial_disp__000"].attrs
            assert abs(un["Ap_end_mm"] - 15.0) < 1e-9 and abs(un["Ap_start_mm"] - (5 + 10 * un["t0"] / 10)) < 1e-9
            assert abs(un["eta_t1"] - 0.015 / 0.0086) < 1e-9 and un["signal_t0"] == 0.0
            st0 = f["stable/case_up/Axial_disp__000"].attrs
            assert abs(st0["Ap_start_mm"] - 6.0) < 1e-9 and st0["t0"] == 1.0          # cropped at 1 s: Ap(1 s) = 6 mm
            dn = f["unstable/case_down/Axial_disp__000"].attrs
            assert abs(dn["Ap_start_mm"] - 14.0) < 1e-9 and dn["Ap_end_mm"] < dn["Ap_start_mm"]
            c = f["unstable/case_const/Axial_disp__000"].attrs
            assert c["Ap_start_mm"] == c["Ap_end_mm"] == 5.0 and "eta_t0" not in c

        # 3. completar el YAML programáticamente
        labels = {
            "source": "doe_results.h5",
            "cases": {
                "case_000": [[0.0, 10.0, "stable"]],
                "case_001": [[0.0, 5.0, "stable"], [5.0, 10.0, "unstable"]],
                "case_002": [],
            },
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(labels, f)

        # 4. from_doe_h5 -> 2 señales (case_002 se ignora), intervalos y attrs conservados
        ds = from_doe_h5(h5_path, yaml_path, channels=["Axial_vel"])
        assert len(ds.signals) == 2, f"esperaba 2 señales, dio {len(ds.signals)}"
        ids = {s.id for s in ds.signals}
        assert ids == {"case_000/Axial_vel", "case_001/Axial_vel"}
        sig1 = next(s for s in ds.signals if s.id == "case_001/Axial_vel")
        assert sig1.intervals == [(0.0, 5.0, "stable"), (5.0, 10.0, "unstable")]
        assert sig1.attrs["source"] == "doe_h5"
        assert sig1.attrs["group"] == "case_001"
        assert sig1.attrs["channel"] == "Axial_vel"
        assert abs(sig1.attrs["kappa"] - 1.4) < 1e-9
        assert not any(k.startswith("labeling_") for k in sig1.attrs)  # YAML a mano, sin bloque "labeling"

        # 5. segments("stable") -> 2 tramos; segments("unstable") -> 1 tramo, rango correcto
        stable = ds.segments("stable")
        unstable = ds.segments("unstable")
        assert len(stable) == 2
        assert len(unstable) == 1
        uid, ut, uy = unstable[0]
        assert uid == "case_001/Axial_vel"
        assert ut.min() >= 5.0 - 1e-9 and ut.max() <= 10.0 + 1e-9

        # 6. YAML inválido -> error explícito
        def _expect_value_error(cases_dict):
            bad_path = os.path.join(tmp, f"bad_{len(cases_dict)}_{list(cases_dict)[0]}.yaml")
            with open(bad_path, "w", encoding="utf-8") as f:
                yaml.safe_dump({"source": "x", "cases": cases_dict}, f)
            try:
                from_doe_h5(h5_path, bad_path, channels=["Axial_vel"])
                raise AssertionError(f"debía fallar: {cases_dict}")
            except ValueError:
                pass

        _expect_value_error({"case_000": [[0.0, 1.0, "weird_label"]]})       # label desconocido
        _expect_value_error({"case_000": [[0.0, 5.0, "stable"], [4.0, 8.0, "unstable"]]})  # solape
        _expect_value_error({"case_000": [[0.0, 999.0, "stable"]]})          # fuera de rango

        # 4b. from_doe_h5 sin channels (default None) -> autodetecta TODOS los canales del caso
        multi_h5 = os.path.join(tmp, "multi_ch.h5")
        multi_yaml = os.path.join(tmp, "multi_labels.yaml")
        with h5py.File(multi_h5, "w") as f:
            grp = f.create_group("case_a")
            for ch in ("Axial_vel", "Axial_disp"):
                sub = grp.create_group(ch)
                sub.create_dataset("time", data=t)
                sub.create_dataset("values", data=t)
            grp.create_dataset("res_R_p", data=np.array([1.0]))  # no es canal (no es grupo time/values)
        with open(multi_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump({"source": "x", "cases": {"case_a": [[0.0, 10.0, "stable"]]}}, f)

        ds_multi = from_doe_h5(multi_h5, multi_yaml)  # channels=None -> autodetecta
        assert {s.id for s in ds_multi.signals} == {"case_a/Axial_vel", "case_a/Axial_disp"}

        ds_restricted = from_doe_h5(multi_h5, multi_yaml, channels=["Axial_vel"])
        assert {s.id for s in ds_restricted.signals} == {"case_a/Axial_vel"}

        # 4d. canales dentro de Out_Deflex (Etapa_1.py / static_deflection.py) se autodetectan también
        deflex_h5 = os.path.join(tmp, "deflex_doe.h5")
        deflex_yaml = os.path.join(tmp, "deflex_labels.yaml")
        with h5py.File(deflex_h5, "w") as f:
            grp = f.create_group("case_a")
            sub = grp.create_group("Axial_disp")
            sub.create_dataset("time", data=t)
            sub.create_dataset("values", data=t)
            out_grp = grp.create_group("Out_Deflex")
            corrected = out_grp.create_group("Axial_disp_out_deflex")
            corrected.create_dataset("time", data=t)
            corrected.create_dataset("values", data=t - 0.5)  # ejemplo: deflexión restada
        with open(deflex_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump({"source": "x", "cases": {"case_a": [[0.0, 10.0, "stable"]]}}, f)

        ds_deflex = from_doe_h5(deflex_h5, deflex_yaml)  # channels=None -> autodetecta también Out_Deflex
        assert {s.id for s in ds_deflex.signals} == {"case_a/Axial_disp", "case_a/Axial_disp_out_deflex"}
        sig_corrected = next(s for s in ds_deflex.signals if s.id == "case_a/Axial_disp_out_deflex")
        assert np.allclose(sig_corrected.y, t - 0.5)

        # 4c. t_start/t_end recorta la señal ANTES de todo lo demás (ej. entrada/salida de herramienta)
        crop_h5 = os.path.join(tmp, "crop_doe.h5")
        with h5py.File(crop_h5, "w") as f:
            grp = f.create_group("case_000")
            grp.attrs["kappa"] = 0.5
            sub = grp.create_group("Axial_vel")
            sub.create_dataset("time", data=t)
            sub.create_dataset("values", data=t)

        crop_yaml_ok = os.path.join(tmp, "crop_labels_ok.yaml")
        with open(crop_yaml_ok, "w", encoding="utf-8") as f:
            yaml.safe_dump({"source": "x", "cases": {"case_000": [[2.0, 8.0, "stable"]]}}, f)
        ds_cropped = from_doe_h5(crop_h5, crop_yaml_ok, channels=["Axial_vel"], t_start=1.0, t_end=9.0)
        sig_c = ds_cropped.signals[0]
        assert abs(sig_c.t[0] - 1.0) < 1e-9 and abs(sig_c.t[-1] - 9.0) < 1e-9, (sig_c.t[0], sig_c.t[-1])
        assert sig_c.intervals == [(2.0, 8.0, "stable")]

        crop_yaml_bad = os.path.join(tmp, "crop_labels_bad.yaml")
        with open(crop_yaml_bad, "w", encoding="utf-8") as f:
            yaml.safe_dump({"source": "x", "cases": {"case_000": [[0.0, 10.0, "stable"]]}}, f)
        try:
            from_doe_h5(crop_h5, crop_yaml_bad, channels=["Axial_vel"], t_start=1.0, t_end=9.0)
            raise AssertionError("debía fallar: intervalo se sale del recorte t_start/t_end")
        except ValueError:
            pass

        # make_label_template con t_start/t_end: el rango mostrado/usado por "kappa" ya viene recortado,
        # y coincide exactamente con lo que from_doe_h5 va a aceptar (sin asumir grilla alineada)
        crop_template_yaml = os.path.join(tmp, "crop_template.yaml")
        make_label_template(crop_h5, crop_template_yaml, strategy="kappa", threshold=1.0, t_start=1.0, t_end=9.0)
        crop_cases = _parse_labels_file(crop_template_yaml)
        assert crop_cases["case_000"] == [(1.0, 9.0, "stable")], crop_cases["case_000"]
        # y ese mismo intervalo generado por la plantilla, build lo tiene que aceptar sin error
        from_doe_h5(crop_h5, crop_template_yaml, channels=["Axial_vel"], t_start=1.0, t_end=9.0)

        # 7. to_hdf5 -> from_hdf5: segments() da los mismos tramos (mismo id base, mismo t/y)
        # (from_hdf5 ya NO reconstruye la señal completa, solo los tramos etiquetados recortados)
        out_h5 = os.path.join(tmp, "reference_dataset.h5")
        ds.to_hdf5(out_h5)
        ds2 = ReferenceDataset.from_hdf5(out_h5)

        def _by_base_id(segs):
            out = {}
            for sid, st, sy in segs:
                out.setdefault(sid.split("#")[0], []).append((st, sy))
            return out

        for label in ("stable", "unstable"):
            orig = _by_base_id(ds.segments(label))
            reloaded = _by_base_id(ds2.segments(label))
            assert set(orig) == set(reloaded), (label, set(orig), set(reloaded))
            for base_id, pieces in orig.items():
                reloaded_pieces = reloaded[base_id]
                assert len(pieces) == len(reloaded_pieces)
                for (ot, oy), (rt, ry) in zip(
                    sorted(pieces, key=lambda p: p[0][0]),
                    sorted(reloaded_pieces, key=lambda p: p[0][0]),
                ):
                    assert np.array_equal(ot, rt)
                    assert np.array_equal(oy, ry)

        # 7b. dos intervalos del MISMO label en una señal -> índices __000/__001 no chocan
        same_label_h5 = os.path.join(tmp, "same_label.h5")
        same_label_yaml = os.path.join(tmp, "same_label_labels.yaml")
        with h5py.File(same_label_h5, "w") as f:
            grp = f.create_group("case_x")
            sub = grp.create_group("Axial_vel")
            sub.create_dataset("time", data=t)
            sub.create_dataset("values", data=t)
        with open(same_label_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {"source": "x", "cases": {"case_x": [[0.0, 3.0, "stable"], [3.0, 6.0, "unstable"], [6.0, 9.0, "stable"]]}},
                f,
            )
        ds_sl = from_doe_h5(same_label_h5, same_label_yaml, channels=["Axial_vel"])
        out_sl_h5 = os.path.join(tmp, "same_label_out.h5")
        ds_sl.to_hdf5(out_sl_h5)
        with h5py.File(out_sl_h5, "r") as f:
            assert sorted(f["stable"].keys()) == ["case_x"]
            assert sorted(f["unstable"].keys()) == ["case_x"]
            stable_pieces = sorted(f["stable"]["case_x"].keys())
            unstable_pieces = sorted(f["unstable"]["case_x"].keys())
        assert stable_pieces == ["Axial_vel__000", "Axial_vel__001"], stable_pieces
        assert unstable_pieces == ["Axial_vel__000"], unstable_pieces

        # 8. combine_by_label: concatena piezas del mismo (label, canal), mismo fs, orden preservado
        pieces_a = [
            ReferenceSignal(
                id="case_a/Axial_vel#case_a/Axial_vel__000",
                t=np.arange(5) / 10.0, y=np.array([1.0, 2.0, 3.0, 4.0, 5.0]), fs=10.0,
                intervals=[(0.0, 0.4, "stable")], attrs={"channel": "Axial_vel"},
            ),
            ReferenceSignal(
                id="case_b/Axial_vel#case_b/Axial_vel__000",
                t=np.arange(3) / 10.0, y=np.array([6.0, 7.0, 8.0]), fs=10.0,
                intervals=[(0.0, 0.2, "stable")], attrs={"channel": "Axial_vel"},
            ),
        ]
        combined_ds = combine_by_label(ReferenceDataset(signals=pieces_a))
        assert len(combined_ds.signals) == 1
        csig = combined_ds.signals[0]
        assert csig.id == "stable/Axial_vel"
        assert np.array_equal(csig.y, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))
        assert len(csig.t) == len(csig.y)
        assert abs(csig.t[1] - csig.t[0] - 1.0 / 10.0) < 1e-12
        assert csig.attrs["n_pieces"] == 2
        assert csig.attrs["source_ids"] == [
            "case_a/Axial_vel#case_a/Axial_vel__000", "case_b/Axial_vel#case_b/Axial_vel__000",
        ]
        assert csig.attrs["piece_lengths"] == [5, 3]
        assert sum(csig.attrs["piece_lengths"]) == len(csig.y)

        combined_out = os.path.join(tmp, "combined.h5")
        save_combined(combined_ds, combined_out)
        reloaded_combined = load_combined(combined_out)
        assert len(reloaded_combined.signals) == 1
        rcsig = reloaded_combined.signals[0]
        assert rcsig.id == "stable/Axial_vel"
        assert np.array_equal(rcsig.y, csig.y)
        assert np.array_equal(rcsig.t, csig.t)
        assert abs(rcsig.fs - csig.fs) < 1e-9
        assert rcsig.attrs["n_pieces"] == 2
        assert rcsig.attrs["source_ids"] == csig.attrs["source_ids"]
        assert rcsig.attrs["piece_lengths"] == csig.attrs["piece_lengths"]
        assert rcsig.intervals == [(0.0, float(csig.t[-1]), "stable")]

        # 9. fs distinto en el mismo grupo -> ValueError explícito, sin resamplear
        mismatched_fs = [
            ReferenceSignal(
                id="case_c/Axial_vel#p0", t=np.arange(4) / 10.0, y=np.arange(4.0), fs=10.0,
                intervals=[(0.0, 0.3, "unstable")], attrs={"channel": "Axial_vel"},
            ),
            ReferenceSignal(
                id="case_d/Axial_vel#p0", t=np.arange(4) / 20.0, y=np.arange(4.0), fs=20.0,
                intervals=[(0.0, 0.15, "unstable")], attrs={"channel": "Axial_vel"},
            ),
        ]
        try:
            combine_by_label(ReferenceDataset(signals=mismatched_fs))
            raise AssertionError("debía fallar por fs distinto")
        except ValueError:
            pass

        # 9b. una señal que mezcla labels distintos -> ValueError (guard, no debería pasar con el flujo actual)
        mixed_labels = [ReferenceSignal(
            id="case_e/Axial_vel#p0", t=np.arange(2) / 10.0, y=np.zeros(2), fs=10.0,
            intervals=[(0.0, 0.05, "stable"), (0.05, 0.1, "unstable")], attrs={"channel": "Axial_vel"},
        )]
        try:
            combine_by_label(ReferenceDataset(signals=mixed_labels))
            raise AssertionError("debía fallar por mezcla de labels")
        except ValueError:
            pass

    print("self-test OK")


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="reference_dataset.py",
        description=(
            "Dataset externo de señales de referencia (stable/unstable/gray) para los "
            "indicadores de chatter. Todo argumento posicional es opcional: si no "
            "se pasa, cae a la constante DEFAULT_* de la sección CONFIG arriba del "
            "script (editable ahí); si se pasa por línea de comandos, éste gana."
        ),
        epilog=(
            "Flujo típico:\n"
            "  reference_dataset.py template   doe_results.h5 reference_labels.yaml [--strategy ...]\n"
            "  (completar/corregir reference_labels.yaml a mano)\n"
            "  reference_dataset.py build      doe_results.h5 reference_labels.yaml reference_dataset.h5\n"
            "  reference_dataset.py combine    reference_dataset.h5 reference_combined.h5\n"
            "\n"
            "Criterio práctico por amplitud (avance 0.05 mm, <10% stable, >40% unstable, entre medio gray):\n"
            "  reference_dataset.py template   doe_results.h5 labels_amp.yaml --strategy amplitude \\\n"
            "                                  --base-attr '$f_tooth$' --base-scale 1e-3 --amp-signal Axial_disp \\\n"
            "                                  --lim-inf-pct 10 --lim-sup-pct 40\n"
            "  reference_dataset.py build      doe_results.h5 labels_amp.yaml reference_dataset_amp.h5\n"
            "\n"
            "Ojo: template nunca pisa un YAML existente, pero build y combine SÍ sobrescriben su .h5 de salida.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="{selftest,template,build,combine}")

    sub.add_parser("selftest", help="Corre el self-test interno (assert-based), no toca archivos reales")

    p_template = sub.add_parser(
        "template",
        help="Paso 1: genera el YAML de etiquetas (vacío o pre-llenado según --strategy) a partir de un doe_results.h5",
    )
    p_template.add_argument(
        "h5_path", nargs="?", default=DEFAULT_H5_PATH,
        help=f"doe_results.h5 de origen (default: DEFAULT_H5_PATH = {DEFAULT_H5_PATH!r})",
    )
    p_template.add_argument(
        "out_yaml", nargs="?", default=DEFAULT_LABELS_PATH,
        help="YAML de etiquetas a crear (default: DEFAULT_LABELS_PATH, o si es None, "
             "'<carpeta de h5_path>/reference_labels.yaml'); nunca sobrescribe uno existente",
    )
    p_template.add_argument(
        "--strategy", choices=sorted(LABEL_STRATEGIES), default=DEFAULT_STRATEGY,
        help=f"cómo pre-llenar el YAML por caso (default: {DEFAULT_STRATEGY!r}); "
             "'manual' deja todo vacío para completar a mano, 'eta' (alias 'kappa') etiqueta por umbral de eta, "
             "'amplitude' por max|señal| vs %% de una variable base del caso (stable/gray/unstable)",
    )
    p_template.add_argument(
        "--eta-threshold", "--kappa-threshold", dest="eta_threshold", type=float, default=DEFAULT_ETA_THRESHOLD,
        help=f"umbral de eta para --strategy eta (default: {DEFAULT_ETA_THRESHOLD}; --kappa-threshold sigue valiendo)",
    )
    p_template.add_argument(
        "--warmup", type=float, default=DEFAULT_WARMUP,
        help=f"segundos a excluir al inicio de la señal en --strategy eta/amplitude (default: {DEFAULT_WARMUP})",
    )
    p_template.add_argument(
        "--base-attr", default=DEFAULT_BASE_ATTR,
        help=f"--strategy amplitude: attr del caso usado como base del %% (default: {DEFAULT_BASE_ATTR!r}; "
             "entre comillas simples en la terminal por el '$')",
    )
    p_template.add_argument(
        "--base-scale", type=float, default=DEFAULT_BASE_SCALE,
        help=f"--strategy amplitude: factor que lleva la base a las unidades de la señal (default: {DEFAULT_BASE_SCALE}, mm -> m)",
    )
    p_template.add_argument(
        "--amp-signal", default=DEFAULT_AMP_SIGNAL,
        help=f"--strategy amplitude: canal cuya max|y| se compara, ej. Axial_disp, Axial_vel, Axial_acc, "
             f"Axial_disp_out_deflex (default: {DEFAULT_AMP_SIGNAL!r})",
    )
    p_template.add_argument(
        "--lim-inf-pct", type=float, default=DEFAULT_LIM_INF_PCT,
        help=f"--strategy amplitude: max|y| < este %% de la base -> stable (default: {DEFAULT_LIM_INF_PCT})",
    )
    p_template.add_argument(
        "--lim-sup-pct", type=float, default=DEFAULT_LIM_SUP_PCT,
        help=f"--strategy amplitude: max|y| > este %% de la base -> unstable; entre medio -> gray "
             f"(default: {DEFAULT_LIM_SUP_PCT})",
    )
    p_template.add_argument(
        "--window-mode", choices=["by_revolution", "by_modal"], default=DEFAULT_WINDOW_MODE,
        help=f"--strategy amplitude, casos en RAMPA de Ap: ventanas en vueltas (T = 60/n del caso) o en periodos "
             f"modales (T = 1/--f-modal) (default: {DEFAULT_WINDOW_MODE!r})",
    )
    p_template.add_argument(
        "--window-N", type=float, default=DEFAULT_WINDOW_N,
        help=f"rampas: largo de la ventana en T (default: {DEFAULT_WINDOW_N})",
    )
    p_template.add_argument(
        "--window-step", type=float, default=DEFAULT_WINDOW_STEP,
        help=f"rampas: paso entre ventanas en T (default: {DEFAULT_WINDOW_STEP})",
    )
    p_template.add_argument(
        "--f-modal", type=float, default=None, help="rampas con --window-mode by_modal: frecuencia modal [Hz]",
    )
    p_template.add_argument(
        "--t-start", type=float, default=DEFAULT_T_START,
        help=f"recorta la señal desde este tiempo [s] antes de calcular el rango, ej. para descartar "
             f"entrada de herramienta (default: DEFAULT_T_START = {DEFAULT_T_START!r}, None = sin recorte)",
    )
    p_template.add_argument(
        "--t-end", type=float, default=DEFAULT_T_END,
        help=f"recorta la señal hasta este tiempo [s], ej. para descartar salida de herramienta "
             f"(default: DEFAULT_T_END = {DEFAULT_T_END!r}, None = sin recorte)",
    )

    p_build = sub.add_parser(
        "build",
        help="Paso 2: lee el YAML ya etiquetado (a mano o pre-llenado por template --strategy) y "
             "arma+guarda el ReferenceDataset (tramos recortados, un grupo por label)",
    )
    p_build.add_argument(
        "h5_path", nargs="?", default=DEFAULT_H5_PATH,
        help=f"doe_results.h5 de origen (default: DEFAULT_H5_PATH = {DEFAULT_H5_PATH!r})",
    )
    p_build.add_argument(
        "labels_yaml", nargs="?", default=DEFAULT_LABELS_PATH,
        help="YAML de etiquetas ya completado (default: DEFAULT_LABELS_PATH, o si es "
             "None, '<carpeta de h5_path>/reference_labels.yaml')",
    )
    p_build.add_argument(
        "out_h5", nargs="?", default=DEFAULT_OUT_H5,
        help="reference_dataset.h5 a escribir -- SOBRESCRIBE si ya existe (default: DEFAULT_OUT_H5, "
             "o si es None, '<carpeta de h5_path>/reference_dataset.h5')",
    )
    p_build.add_argument(
        "--channels", nargs="+", default=DEFAULT_CHANNELS,
        help=f"canales a incluir, ej. Axial_vel Axial_disp (default: {DEFAULT_CHANNELS!r} "
             "-> autodetecta TODOS los canales del caso)",
    )
    p_build.add_argument(
        "--t-start", type=float, default=DEFAULT_T_START,
        help=f"recorta la señal desde este tiempo [s] antes de armar los tramos, ej. para descartar "
             f"entrada de herramienta (default: DEFAULT_T_START = {DEFAULT_T_START!r}, None = sin recorte)",
    )
    p_build.add_argument(
        "--t-end", type=float, default=DEFAULT_T_END,
        help=f"recorta la señal hasta este tiempo [s], ej. para descartar salida de herramienta "
             f"(default: DEFAULT_T_END = {DEFAULT_T_END!r}, None = sin recorte)",
    )

    p_combine = sub.add_parser(
        "combine",
        help="Paso 3 (Fase 2A): concatena todas las piezas del mismo (label, canal) en una señal continua",
    )
    p_combine.add_argument(
        "in_h5", nargs="?", default=DEFAULT_IN_H5,
        help="reference_dataset.h5 de entrada, salida de 'build' (default: DEFAULT_IN_H5, o si es "
             "None, '<carpeta de DEFAULT_H5_PATH>/reference_dataset.h5')",
    )
    p_combine.add_argument(
        "out_h5", nargs="?", default=DEFAULT_OUT_COMBINED,
        help="reference_combined.h5 a escribir (default: DEFAULT_OUT_COMBINED, o si es None, "
             "'<carpeta de DEFAULT_H5_PATH>/reference_combined.h5')",
    )

    args = parser.parse_args()

    if args.cmd == "selftest":
        _self_test()
        return

    if args.cmd == "combine":
        default_dir = os.path.dirname(os.path.abspath(DEFAULT_H5_PATH)) if DEFAULT_H5_PATH else None
        in_h5 = args.in_h5 or (default_dir and os.path.join(default_dir, "reference_dataset.h5"))
        out_h5 = args.out_h5 or (default_dir and os.path.join(default_dir, "reference_combined.h5"))
        if not in_h5 or not out_h5:
            parser.error(
                "faltan in_h5/out_h5 — pasalos como argumento o fijá DEFAULT_H5_PATH arriba del script"
            )
        combined = combine_by_label(ReferenceDataset.from_hdf5(in_h5))
        save_combined(combined, out_h5)
        print(f"{len(combined.signals)} señales combinadas -> {out_h5}")
        return

    if args.h5_path is None:
        parser.error(
            "falta h5_path — pasalo como argumento o fijá DEFAULT_H5_PATH arriba del script"
        )

    # out_yaml/labels_yaml/out_h5 no pasados por CLI -> misma carpeta que h5_path
    h5_dir = os.path.dirname(os.path.abspath(args.h5_path))

    if args.cmd == "template":
        out_yaml = args.out_yaml or os.path.join(h5_dir, "reference_labels.yaml")
        kwargs = {}
        if args.strategy in ("eta", "kappa"):
            kwargs = {"threshold": args.eta_threshold, "warmup": args.warmup}
        elif args.strategy == "amplitude":
            kwargs = {
                "base_attr": args.base_attr, "base_scale": args.base_scale, "signal": args.amp_signal,
                "lim_inf_pct": args.lim_inf_pct, "lim_sup_pct": args.lim_sup_pct, "warmup": args.warmup,
                "window_mode": args.window_mode, "window_N": args.window_N, "window_step": args.window_step,
                "f_modal": args.f_modal,
            }
        make_label_template(
            args.h5_path, out_yaml, strategy=args.strategy,
            t_start=args.t_start, t_end=args.t_end, **kwargs,
        )
        print(f"Plantilla escrita en {out_yaml}")
    elif args.cmd == "build":
        labels_yaml = args.labels_yaml or os.path.join(h5_dir, "reference_labels.yaml")
        out_h5 = args.out_h5 or os.path.join(h5_dir, "reference_dataset.h5")
        ds = from_doe_h5(
            args.h5_path, labels_yaml, channels=args.channels,
            t_start=args.t_start, t_end=args.t_end,
        )
        ds.to_hdf5(out_h5)
        print(f"{len(ds.signals)} señales -> {out_h5}")


if __name__ == "__main__":
    _main()
