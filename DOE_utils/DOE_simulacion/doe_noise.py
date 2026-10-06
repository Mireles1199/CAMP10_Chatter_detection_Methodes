r"""doe_noise.py — Aplica ruido gaussiano (varios niveles SNR) a casos de doe_results.h5.

Dos modos:
  * un caso de control (modo antiguo, sin CASES): doe_noise_results.h5 con 'control' + 'snr_XXX.XX';
    SNR relativo a la potencia de esa señal.
  * multi-caso (CASES = lista o "all"; docs/planes/PLAN_noise_validation.md §4.1): doe_noise_multi_results.h5 con
    un grupo por copia ruidosa 'snr_{SNR:06.2f}__{case}__r{K:02d}'. SNR ABSOLUTO: sigma = sqrt(P_ref / 10^(SNR/10)),
    P_ref = varianza de la señal del caso de referencia (SNR_REF_CASE; 'auto' = el caso inestable de menor kappa,
    de las etiquetas PROPIAS del experimento) -> el mismo sigma para todos los casos de un nivel. REALIZATIONS
    semillas independientes (np.random.default_rng([SEED, K, n_caso, n_señal])); dentro de una realización el
    mismo ruido unitario reescalado en todos los niveles. 'time' guardado una vez por caso (enlaces duros HDF5).

Uso:
    python doe_noise.py --doe_results .\DOE_xxx\doe_results.h5
    python doe_noise.py --doe_results X\doe_results.h5 --cases all --labels X\reference_dataset_amp.h5
    python doe_noise.py --doe_results X\doe_results.h5 --cases case_000 case_011 --snr_ref_case case_011 --realizations 3
    python doe_noise.py --selftest

Configuración (bloque CONFIG; la sección noise del experimento y el CLI la sobrescriben):
    CONTROL_CASE_IDX : índice del caso de control (modo antiguo)
    SNR_LIST         : niveles SNR en dB
    SNR_RANGE        : YA NO SE USA (se acepta y se ignora con un aviso)
    SEED             : semilla para reproducibilidad del ruido
    SIGNALS          : señales a procesar
    CASES            : None (modo antiguo) | "all" | ["case_000", ...]  (modo multi-caso)
    REALIZATIONS     : realizaciones por caso y nivel (modo multi-caso)
    SNR_REF_CASE     : "auto" | "case_NNN"  (modo multi-caso)
"""

import os
import sys
import argparse
import logging
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # DOE_utils/: noise_origins
import noise_origins  # noqa: E402
import eta_compat  # noqa: E402  (kappa -> eta: lee ambos nombres)

# ==============================================================================
# CONFIG — editar aquí
# ==============================================================================

CONTROL_CASE_IDX = 3                   # índice del caso de control en doe_results.h5 (modo antiguo)
SNR_LIST         = [80, 60, 40, 30, 20, 10]   # dB
SNR_RANGE        = None                 # YA NO SE USA: se acepta (YAML antiguos) y se ignora con un aviso
SEED             = 42                   # semilla para np.random (reproducibilidad)
SIGNALS          = ["Axial_disp", "Axial_vel"]
CASES            = None                 # None: modo antiguo (un caso de control) | "all" | ["case_000", ...]
REALIZATIONS     = 3                    # modo multi-caso: realizaciones de ruido por caso y nivel
SNR_REF_CASE     = "auto"               # modo multi-caso: "auto" (inestable de menor kappa) | "case_NNN"

# ==============================================================================
# Logging
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def build_snr_list(snr_fixed, snr_range=None):
    """SNR_LIST deduplicada y ordenada descendente. SNR_RANGE ya no se usa: se acepta y se ignora con un aviso."""
    if snr_range is not None:
        log.warning("SNR_RANGE / snr_range ya no se usa y se ignora: los niveles salen de SNR_LIST / snr_list")
    if not snr_fixed:
        raise ValueError("No hay niveles SNR definidos: edita SNR_LIST (o snr_list en la sección noise).")
    return sorted({float(v) for v in snr_fixed}, reverse=True)


def auto_ref_case(info: dict) -> str:
    """Caso de referencia del SNR absoluto: el etiquetado 'unstable' (constante) de menor kappa, de un
    {case: {label, kappa, ramp}} de experiment.label_info (las etiquetas PROPIAS del experimento)."""
    cand = [(eta_compat.get(d, "eta"), c) for c, d in info.items()
            if d.get("label") == "unstable" and not d.get("ramp") and np.isfinite(eta_compat.get(d, "eta"))]
    if not cand:
        raise ValueError("snr_ref_case 'auto': no hay casos constantes etiquetados 'unstable'; indica snr_ref_case")
    return min(cand)[1]


def write_multi_noise(doe_results: str, out_path: str, cases, snr_list: list, realizations: int, seed: int,
                      ref_case: str, signals: list, clean_indicators: str = None) -> list:
    """Modo multi-caso (PLAN_noise_validation.md §4.1). Devuelve los nombres de grupo escritos. Anota en attrs raíz el
    origen de las señales (source_signals, y clean_indicators si se da: §4.4)."""
    with h5py.File(doe_results, "r") as src:
        all_cases = sorted(k for k in src if k.startswith("case_"))
        cases = all_cases if cases == "all" else list(cases)
        missing = [c for c in cases + [ref_case] if c not in src]
        if missing:
            raise KeyError(f"casos no encontrados en {doe_results}: {missing}")
        p_ref = {s: float(np.var(src[ref_case][f"{s}/values"][()])) for s in signals if s in src[ref_case]}
        if not p_ref:
            raise ValueError(f"el caso de referencia {ref_case} no tiene ninguna de las señales {signals}")
        sigma = {(snr, s): float(np.sqrt(p / 10.0 ** (snr / 10.0))) for snr in snr_list for s, p in p_ref.items()}
        written = []
        with h5py.File(out_path, "w") as out:
            out.attrs.update(noise_layout="multi", snr_mode="absolute", snr_ref_case=ref_case,
                             snr_levels=np.array(snr_list, float), realizations=int(realizations), seed=int(seed),
                             **{f"snr_ref_power_{s}": p for s, p in p_ref.items()})
            out.attrs.create("cases", np.array(cases, dtype=object), dtype=h5py.string_dtype())
            for case in cases:
                g, n_case = src[case], int(case.split("_")[-1])   # n_case: same noise for a case whatever the list
                data = {s: (g[f"{s}/time"][()], g[f"{s}/values"][()]) for s in p_ref if s in g}
                first_time = {}   # señal -> dataset 'time' ya escrito (enlace duro en las demás copias)
                for k in range(realizations):
                    unit = {s: np.random.default_rng([seed, k, n_case, i]).standard_normal(y.shape)
                            for i, (s, (_, y)) in enumerate(data.items())}
                    for snr in snr_list:
                        name = f"snr_{snr:06.2f}__{case}__r{k:02d}"
                        og = out.create_group(name)
                        og.attrs.update(dict(g.attrs))
                        og.attrs.update(snr_db=float(snr), case_source=case, realization=int(k), seed=int(seed),
                                        **{f"sigma_{s}": sigma[(snr, s)] for s in data})
                        for s, (t, y) in data.items():
                            sg = og.create_group(s)
                            same = next((d for s2, d in first_time.items() if np.array_equal(data[s2][0], t)), None)
                            if same is None:
                                sg.create_dataset("time", data=t)
                                first_time[s] = sg["time"]
                            else:
                                sg["time"] = same   # enlace duro: el mismo dataset, sin copia
                            sg.create_dataset("values", data=y + sigma[(snr, s)] * unit[s])
                        written.append(name)
            log.info("%d copias ruidosas (%d casos x %d niveles x %d realizaciones), referencia %s -> %s",
                     len(written), len(cases), len(snr_list), realizations, ref_case, out_path)
    noise_origins.set_origins(out_path, source_signals=doe_results, clean_indicators=clean_indicators)
    return written


def add_gaussian_noise(y: np.ndarray, snr_db: float, z: np.ndarray) -> np.ndarray:
    """Añade ruido gaussiano blanco tal que SNR = snr_db dB (potencia AC, sin offset DC).

    z        = N(0,1) fijo por señal (mismo ruido escalado en todos los SNR)
    P_signal = var(y)
    sigma    = sqrt(P_signal / 10^(snr_db/10))
    y_noisy  = y + sigma * z
    """
    p_signal = np.var(y)
    if p_signal == 0.0:
        log.warning("Señal con potencia cero — el ruido no tendrá efecto práctico.")
        p_signal = 1e-30
    sigma = np.sqrt(p_signal / (10.0 ** (snr_db / 10.0)))
    return y + sigma * z


# ------------------------------------------------------------------------------
# Listado de casos
# ------------------------------------------------------------------------------

def list_cases(h5_path: str) -> None:
    """Imprime en consola una tabla con todos los casos del HDF5 y sus atributos."""
    with h5py.File(h5_path, "r") as f:
        groups = sorted(f.keys())
        if not groups:
            print("  (sin casos)")
            return

        # Recopilar todos los attrs de todos los grupos
        rows = []
        all_keys = []
        for grp_name in groups:
            attrs = dict(f[grp_name].attrs)
            rows.append((grp_name, attrs))
            for k in attrs:
                if k not in all_keys:
                    all_keys.append(k)

        # Columnas: idx | group | attrs...
        col_names = ["idx", "group"] + all_keys
        table = []
        for i, (grp_name, attrs) in enumerate(rows):
            row = [str(int(grp_name.split('_')[-1])), grp_name] + [str(attrs.get(k, "-")) for k in all_keys]
            table.append(row)

        # Calcular anchos de columna
        widths = [max(len(col_names[j]), max(len(r[j]) for r in table))
                  for j in range(len(col_names))]

        sep  = "  ".join("-" * w for w in widths)
        header = "  ".join(col_names[j].ljust(widths[j]) for j in range(len(col_names)))
        print()
        print(header)
        print(sep)
        for row in table:
            print("  ".join(row[j].ljust(widths[j]) for j in range(len(col_names))))
        print()
        print(f"  Total: {len(rows)} casos  |  Para usar como control: edita CONTROL_CASE_IDX en CONFIG")
        print()


# ------------------------------------------------------------------------------
# Carga del caso de control
# ------------------------------------------------------------------------------

def load_control_case(h5_path: str, case_idx: int) -> dict:
    """Lee el grupo case_{idx:03d} del HDF5 y retorna sus señales y atributos."""
    grp_name = f"case_{case_idx:03d}"
    with h5py.File(h5_path, "r") as f:
        if grp_name not in f:
            available = sorted(f.keys())
            raise KeyError(
                f"Grupo '{grp_name}' no encontrado en {h5_path}.\n"
                f"  Grupos disponibles: {available}"
            )
        grp = f[grp_name]
        attrs = dict(grp.attrs)
        signals = {}
        for sig in SIGNALS:
            if sig in grp:
                t = grp[f"{sig}/time"][()]
                y = grp[f"{sig}/values"][()]
                signals[sig] = (t, y)
            else:
                log.warning("Señal '%s' no encontrada en grupo '%s' — se omite.", sig, grp_name)
    return {"group": grp_name, "attrs": attrs, "signals": signals}


# ------------------------------------------------------------------------------
# Escritura HDF5 de salida
# ------------------------------------------------------------------------------

def write_noise_results(out_path: str, control: dict, snr_list: list, seed: int) -> None:
    """Escribe doe_noise_results.h5 con un grupo por nivel SNR + grupo 'control'."""
    rng = np.random.default_rng(seed)

    with h5py.File(out_path, "w") as out_f:
        # --- grupo control (señal original) ---
        ctrl_grp = out_f.create_group("control")
        for k, v in control["attrs"].items():
            ctrl_grp.attrs[k] = v
        ctrl_grp.attrs["snr_db"]       = "None"
        ctrl_grp.attrs["case_source"]  = control["group"]

        for sig, (t, y) in control["signals"].items():
            sg = ctrl_grp.create_group(sig)
            sg.create_dataset("time",   data=t, compression="gzip")
            sg.create_dataset("values", data=y, compression="gzip")

        log.info("Grupo 'control' escrito  (señal original, sin ruido)")

        # ruido unitario fijo por señal: los niveles SNR solo lo reescalan
        unit_noise = {sig: rng.standard_normal(y.shape)
                      for sig, (_, y) in control["signals"].items()}

        # --- un grupo por nivel SNR ---
        for snr_db in snr_list:
            grp_name = f"snr_{snr_db:06.2f}"
            snr_grp  = out_f.create_group(grp_name)

            for k, v in control["attrs"].items():
                snr_grp.attrs[k] = v
            snr_grp.attrs["snr_db"]      = float(snr_db)
            snr_grp.attrs["case_source"] = control["group"]
            snr_grp.attrs["seed"]        = seed

            for sig, (t, y) in control["signals"].items():
                y_noisy = add_gaussian_noise(y, snr_db, unit_noise[sig])
                sg = snr_grp.create_group(sig)
                sg.create_dataset("time",   data=t,       compression="gzip")
                sg.create_dataset("values", data=y_noisy, compression="gzip")

            log.info("Grupo '%s' escrito  (SNR = %.1f dB)", grp_name, snr_db)

    log.info("Archivo de salida: %s", out_path)
    log.info("Total grupos escritos: control + %d niveles SNR", len(snr_list))


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Aplica ruido gaussiano (multi-SNR) al caso de control de doe_results.h5.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Configuración (editar bloque CONFIG en el script):
  CONTROL_CASE_IDX  índice del caso de control (0 = case_000, 1 = case_001, …)
  SNR_LIST          lista fija de niveles SNR en dB  ej. [5, 10, 20, 40]
  SNR_RANGE         rango (min, max, step) en dB     ej. (5, 50, 5)
  SEED              semilla aleatoria para reproducibilidad (default 42)
  SIGNALS           señales a procesar               ej. ["Axial_disp", "Axial_vel"]

Estructura del HDF5 generado:
  doe_noise_results.h5
  ├── control/          señal original sin ruido
  ├── snr_040.00/       señal con SNR = 40 dB (poco ruido)
  ├── snr_020.00/
  ├── snr_010.00/
  └── snr_005.00/       señal con SNR =  5 dB (mucho ruido)

Ejemplos:
  python doe_noise.py --doe_results .\\DOE_xxx\\doe_results.h5
  python doe_noise.py --doe_results .\\DOE_xxx\\doe_results.h5 --out ruido.h5
        """,
    )
    p.add_argument(
        "--doe_results",
        metavar="PATH",
        help="Ruta al archivo doe_results.h5 generado por doe_runner --command extract",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Imprime la tabla de casos disponibles en el HDF5 y sale.",
    )
    p.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Ruta del HDF5 de salida (default: doe_noise_results.h5 junto a --doe_results)",
    )
    p.add_argument(
        "--experiment",
        default=None,
        help="experimento de DOE_utils/experiments: su sección noise sobrescribe CONTROL_CASE_IDX, SNR_LIST, "
             "SNR_RANGE (ignorado), SEED, SIGNALS, CASES, REALIZATIONS, SNR_REF_CASE (claves en minúscula); sus "
             "etiquetas (label out) dan el caso de referencia 'auto'",
    )
    p.add_argument("--cases", nargs="+", default=None, metavar="CASE",
                   help="modo multi-caso: 'all' o lista case_NNN (sobrescribe CASES)")
    p.add_argument("--snr_list", nargs="+", type=float, default=None, metavar="DB", help="niveles SNR en dB")
    p.add_argument("--realizations", type=int, default=None, help="realizaciones por caso y nivel (modo multi)")
    p.add_argument("--snr_ref_case", default=None, help="'auto' o case_NNN (modo multi)")
    p.add_argument("--labels", default=None, metavar="PATH",
                   help="reference_dataset*.h5 de ESTE experimento, para snr_ref_case 'auto' sin --experiment")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--selftest", action="store_true")
    return p.parse_args()


# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------

def _selftest():
    import tempfile
    d = tempfile.mkdtemp()
    src, out, out2 = (os.path.join(d, n) for n in ("doe_results.h5", "multi.h5", "multi2.h5"))
    t = np.linspace(0.0, 1.0, 2001)
    with h5py.File(src, "w") as f:
        for i, amp in enumerate((1e-8, 2e-8, 1e-5, 3e-5)):
            g = f.create_group(f"case_{i:03d}")
            g.attrs.update(kappa=0.5 + i * 0.4, **{"$spin_rate$": 12000.0})
            g["Axial_disp/time"], g["Axial_disp/values"] = t, amp * np.sin(2 * np.pi * 150 * t)
            g["Axial_vel/time"], g["Axial_vel/values"] = t, 2 * np.pi * 150 * amp * np.cos(2 * np.pi * 150 * t)
    # SNR_RANGE ya no se usa; la lista se ordena de limpio a ruidoso
    assert build_snr_list([20, 40, 40]) == [40.0, 20.0] and build_snr_list([10], [5, 200, 5]) == [10.0]
    try:
        build_snr_list(None)
        raise SystemExit("SNR_LIST vacía debería fallar")
    except ValueError:
        pass
    # referencia 'auto': el inestable constante de menor kappa
    info = {"case_000": dict(label="stable", kappa=0.5, ramp=False), "case_002": dict(label="unstable", kappa=1.3, ramp=False),
            "case_003": dict(label="unstable", kappa=1.7, ramp=False), "case_009": dict(label="unstable", kappa=1.1, ramp=True)}
    assert auto_ref_case(info) == "case_002"
    assert auto_ref_case({"case_000": dict(label="stable", eta=0.5, ramp=False), "case_002": dict(label="unstable", eta=1.3, ramp=False),
                          "case_003": dict(label="unstable", kappa=1.7, ramp=False)}) == "case_002"   # eta or kappa (label_info of either age)
    try:
        auto_ref_case({"case_000": dict(label="stable", kappa=0.5, ramp=False)})
        raise SystemExit("sin inestables debería fallar")
    except ValueError:
        pass
    names = write_multi_noise(src, out, ["case_000", "case_003"], [40.0, 20.0], 2, 7, "case_002", ["Axial_disp", "Axial_vel"])
    assert sorted(names) == sorted(f"snr_{s:06.2f}__{c}__r{k:02d}" for s in (40.0, 20.0) for c in ("case_000", "case_003") for k in (0, 1))
    with h5py.File(out, "r") as f, h5py.File(src, "r") as fs:
        assert f.attrs["noise_layout"] == "multi" and f.attrs["snr_mode"] == "absolute" and f.attrs["snr_ref_case"] == "case_002"
        assert list(f.attrs["snr_levels"]) == [40.0, 20.0] and f.attrs["realizations"] == 2
        assert list(f.attrs["cases"]) == ["case_000", "case_003"] and not any(k.startswith("case_") for k in f)   # se lee como ruido
        assert f.attrs["source_signals_abs"] == os.path.abspath(src) and "clean_indicators_abs" not in f.attrs   # §4.4
        assert noise_origins.origins(out)["source_signals"] == os.path.abspath(src)
        p_ref = float(np.var(fs["case_002/Axial_disp/values"][()]))
        assert abs(f.attrs["snr_ref_power_Axial_disp"] - p_ref) < 1e-30
        a, b = f["snr_040.00__case_000__r00"], f["snr_040.00__case_003__r00"]
        sig40 = np.sqrt(p_ref / 1e4)
        assert a.attrs["sigma_Axial_disp"] == b.attrs["sigma_Axial_disp"] and abs(a.attrs["sigma_Axial_disp"] - sig40) < 1e-20   # absoluto
        assert a.attrs["case_source"] == "case_000" and a.attrs["realization"] == 0 and a.attrs["snr_db"] == 40.0 and a.attrs["kappa"] == 0.5
        clean = fs["case_000/Axial_disp/values"][()]
        n40 = a["Axial_disp/values"][()] - clean
        n20 = f["snr_020.00__case_000__r00/Axial_disp/values"][()] - clean
        assert abs(n40.std() / sig40 - 1) < 0.1                                              # ruido de la sigma pedida
        assert np.allclose(n40 / sig40, n20 / f["snr_020.00__case_000__r00"].attrs["sigma_Axial_disp"])   # mismo ruido unitario
        assert not np.allclose(n40, f["snr_040.00__case_000__r01/Axial_disp/values"][()] - clean)          # otra realización
        # time: un dataset por caso, el resto son enlaces duros (también Axial_vel/time, igual a Axial_disp/time)
        assert a["Axial_disp/time"] == f["snr_020.00__case_000__r01/Axial_disp/time"] == a["Axial_vel/time"]
        assert np.array_equal(a["Axial_disp/time"][()], t)
    write_multi_noise(src, out2, ["case_000", "case_003"], [40.0, 20.0], 2, 7, "case_002", ["Axial_disp", "Axial_vel"])
    with h5py.File(out, "r") as f, h5py.File(out2, "r") as g:   # misma semilla -> mismo archivo
        assert all(np.array_equal(f[n]["Axial_vel/values"][()], g[n]["Axial_vel/values"][()]) for n in f)
    # un caso tiene el mismo ruido aunque cambie la lista de casos
    write_multi_noise(src, out2, ["case_003"], [40.0], 1, 7, "case_002", ["Axial_disp"])
    with h5py.File(out, "r") as f, h5py.File(out2, "r") as g:
        assert np.array_equal(f["snr_040.00__case_003__r00/Axial_disp/values"][()], g["snr_040.00__case_003__r00/Axial_disp/values"][()])
    try:
        write_multi_noise(src, out2, ["case_099"], [40.0], 1, 7, "case_002", ["Axial_disp"])
        raise SystemExit("caso inexistente debería fallar")
    except KeyError:
        pass
    # modo antiguo intacto: control + snr_*
    legacy = os.path.join(d, "legacy.h5")
    write_noise_results(legacy, load_control_case(src, 1), [40.0, 20.0], 42)
    with h5py.File(legacy, "r") as f:
        assert sorted(f) == ["control", "snr_020.00", "snr_040.00"] and "noise_layout" not in f.attrs
    print("doe_noise selftest OK")


def _experiment():
    """DOE_utils/experiment.py, only for --experiment."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:
        sys.path.insert(0, here)
    import experiment
    return experiment


def main():
    args = parse_args()
    if args.selftest:
        return _selftest()
    if not args.doe_results:
        sys.exit("--doe_results es obligatorio")
    labels = args.labels
    if args.experiment:   # la sección noise del experimento sobrescribe el CONFIG
        ov, exp = _experiment().section_overrides(args.experiment, "noise",
                                                  ("CONTROL_CASE_IDX", "SNR_LIST", "SNR_RANGE", "SEED", "SIGNALS",
                                                   "CASES", "REALIZATIONS", "SNR_REF_CASE"))
        globals().update(ov)
        labels = labels or exp.label["out"]   # etiquetas PROPIAS (no exp.reference: en una validación es el entrenamiento)
    # el CLI sobrescribe al experimento
    for key, val in (("CASES", args.cases), ("SNR_LIST", args.snr_list), ("REALIZATIONS", args.realizations),
                     ("SNR_REF_CASE", args.snr_ref_case), ("SEED", args.seed)):
        if val is not None:
            globals()[key] = val
    cases = CASES
    if isinstance(cases, (list, tuple)) and list(cases) == ["all"]:
        cases = "all"

    doe_results = os.path.normpath(args.doe_results)
    if not os.path.isfile(doe_results):
        log.error("Archivo no encontrado: %s", doe_results)
        sys.exit(1)

    if args.list:
        list_cases(doe_results)
        sys.exit(0)

    default_name = "doe_noise_results.h5" if cases is None else "doe_noise_multi_results.h5"
    out_path = os.path.normpath(args.out) if args.out else os.path.join(os.path.dirname(doe_results), default_name)

    # Construir lista SNR
    snr_list = build_snr_list(SNR_LIST, SNR_RANGE)
    log.info("Niveles SNR (dB): %s", snr_list)

    if cases is not None:   # modo multi-caso, SNR absoluto
        ref = SNR_REF_CASE
        if ref == "auto":
            if not (labels and os.path.isfile(labels)):
                sys.exit("snr_ref_case 'auto' necesita las etiquetas del experimento (--experiment o --labels), "
                         "o indica --snr_ref_case case_NNN")
            ref = auto_ref_case(_experiment().label_info(labels))
        log.info("Caso de referencia del SNR: %s", ref)
        write_multi_noise(doe_results, out_path, cases, snr_list, int(REALIZATIONS), int(SEED), ref, list(SIGNALS),
                          clean_indicators=exp.indicators.get("out") if args.experiment else None)
        return

    # Cargar caso de control
    log.info("Cargando caso de control: case_%03d  de  %s", CONTROL_CASE_IDX, doe_results)
    control = load_control_case(doe_results, CONTROL_CASE_IDX)
    log.info("Atributos del caso de control: %s", control["attrs"])
    log.info("Señales encontradas: %s", list(control["signals"].keys()))

    # Escribir resultados
    write_noise_results(out_path, control, snr_list, SEED)


if __name__ == "__main__":
    main()
