# %%
"""
Etapa4_bis.py
=============
Lectura, visualización y envolvente RMS de señales de simulación de maquinado
almacenadas en un archivo .h5 con la estructura:

    case_xxx/
        Axial_disp/
            times   [N]  [s]
            values  [N]  [m]
        Axial_vel/
            times   [N]  [s]
            values  [N]  [m/s]

Etapas implementadas:
    1. Lectura del archivo .h5
    2. Selección de casos
    3. Gráficas señales completas
    4. Extracción de intervalo temporal
    5. Gráficas señales recortadas
    6. Cálculo del RMS móvil
    7. Gráficas del RMS móvil
    8. Lectura y gráfica de res_R_p por caso
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

def _configurar_estilo_global() -> None:
    """Configura el estilo global de los gráficos."""
    # plt.style.use('dark_background')

    local_style = {
        # Tipografía general
        'font.family': 'serif',
        'font.size': 16,

        # Tamaños de títulos y etiquetas
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 16,

        # Estética de líneas
        'lines.linewidth': 1.2,
        'lines.markersize': 10,

        # Bordes y ejes
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,

        # Ticks
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2.5,
        'ytick.minor.size': 2.5,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,

        # Texto matemático
        'mathtext.fontset': 'stix',
        'axes.formatter.use_mathtext': True,

        # Leyenda
        'legend.frameon': False,
        'legend.loc': 'best',
        'legend.handlelength': 2.0,
        'legend.borderaxespad': 0.5,

        # Exportación
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'savefig.transparent': True,

        # Fondo
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        }

    plt.rcParams.update(local_style)


_configurar_estilo_global()  # aplica estilo al cargar el módulo
fig_scale = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS PRINCIPALES — modifica estos valores antes de correr el script
# ─────────────────────────────────────────────────────────────────────────────

h5_path = (
    "D:\\Thesis\\03-Code_Storage\\02-Altintlas_Nessy2m_Storage\\"
    "2DOF_Cone_DOE\\DOE_Influence_dexel_RPM_12000_ftooth_005_dt_200\\"
    "doe_results.h5"
)

# Casos a procesar con su propio intervalo temporal.
# Formato: { case_name: (t_start, t_end) }
# Usa None para procesar TODOS los casos con el intervalo por defecto.
selected_cases: dict[str, tuple[float, float]] | None = {

        "case_009": (10.0, 11.0),
        "case_008": (8.71, 10.38),
        "case_007": (8.61, 10.26),
        "case_006": (9.16, 10.63),
        "case_005": (9.0, 10.64),
        "case_004": (9.9, 10.94),
        "case_003": (10.18, 11.44),
        "case_002": (9.58, 10.99),
        "case_001": (11.19, 12.28),
        "case_000": (12.0, 12.94),


    # "case_009": (10.3, 10.8),
}

# Tiempo (absoluto, en s) donde se evalúa el proxy de perturbación P_h para
# la descomposición del decalage temporel.  Independiente del intervalo de fitting.
# Debe estar ANTES del inicio del crecimiento exponencial de cada caso.
# Formato:
#   None              → usar t_fit_start de cada caso (comportamiento anterior)
#   float             → mismo instante para todos los casos
#   dict[str, float]  → instante distinto por caso
t_proxy_Ph: dict[str, float] | float | None = {
    "case_009": 10.6,
    "case_008": 9.9,
    "case_007": 9.45,
    "case_006": 9.64,
    "case_005": 9.4,
    "case_004": 9.9,
    "case_003": 10.18,
    "case_002": 9.58,
    "case_001": 11.12,
    "case_000": 12.0,
}

# Intervalo por defecto — usado solo cuando selected_cases = None
t_start_default: float = 0.0
t_end_default: float   = 0.05

# Tamaño de la ventana RMS móvil [s]
window_size_time: float = 1/150 * 5

# Tiempo (absoluto, en s) donde se evalúa la perturbación de fuerza ANTES
# del régimen exponencial — proxy de la condición inicial pura del dexel.
# Puede ser:
#   None                        → usar t_fit_start de cada caso (desactivado)
#   float                       → mismo tiempo absoluto para todos los casos
#   dict {case_name: float}     → tiempo distinto por caso
# Debe estar antes del t_fit_start de cada caso.
t_eval_pre_F: float | dict | None = {
    "case_000":0,
    "case_001":0,
    "case_002":0,
}

# Si True, genera también una figura por caso mostrando señal + RMS superpuestos
plot_signal_with_rms: bool = False
plot_scale = 3.0

# Nivel de referencia  ln(A_ref)  por señal — escribe el valor de ln directamente
# Unidades: Axial_disp en m  |  Axial_vel en m/s
# Pon None en una clave para desactivar el cálculo para esa señal
# Referencia rápida:  ln(1e-3)≈-6.91  ln(1e-4)≈-9.21  ln(1e-5)≈-11.51  ln(1e-6)≈-13.82
ln_A_ref: dict[str, float | None] = {
    "Axial_disp": None,   # m
    "Axial_vel":  None,    # m/s
    "dF":         None,    # N        ← ajusta según tus datos
    "dF_norm":    None,    # N/m      ← ajusta según tus datos
}

# ─────────────────────────────────────────────────────────────────────────────
# Rampa de fuerza teórica para sustraer de res_R_p.
# Mismos valores que en Etapa_4.py.  Pon force_ramp_ref = None para desactivar.
# ─────────────────────────────────────────────────────────────────────────────
_n_rpm       = 12_094.28     # rpm
_Nz          = 1             # número de dientes
_f_tooth_mm  = 0.05          # mm/diente
_Kf_fisico   = 1.0e9         # N/m²
# Retardo regenerativo (1 diente)  y  pulsación dominante del chatter
_T_delay        = 60.0 / (_n_rpm * _Nz)          # [s]
_omega_chatter  = 2.0 * np.pi * 150.0            # [rad/s]  ≈ pulsación propre
_a0_m        = 5e-3          # profundidad inicial [m]
_a1_m        = 15e-3         # profundidad final   [m]
_L_cylindre  = 150e-3        # longitud del cilindro [m]
_vf          = _n_rpm * _f_tooth_mm / 1e3 / 60
_t_ramp      = _L_cylindre / _vf

# Desplazamiento temporal de la rampa: tiempo que tarda la simulación en
# alcanzar la deflexión inicial delta(a0) antes de que empiece la rampa.
# Equivale a t_ramp_corected = t_eval + t_reach en Etapa_4.py.


_t_ramp_vec  = np.linspace(0.0, _t_ramp, 20_000)
_ap_ramp_vec = np.where(
    _t_ramp_vec <= _t_ramp,
    _a0_m + (_a1_m - _a0_m) * (_t_ramp_vec / _t_ramp),
    _a1_m,
)
_force_ramp_vec = _Kf_fisico * (_ap_ramp_vec ) * _f_tooth_mm * 1e-3

# Deflexión estática teórica:  delta(t) = phi² * F(t) / omega_2
# phi = sin(theta) / sqrt(m),  omega_2 = k / m
_k2          = 2.13e8
_m2          = _k2 / (2.0 * np.pi * 150.0) ** 2
_theta_deg   = 135.0
_phi         = np.sin(np.deg2rad(_theta_deg)) / np.sqrt(_m2)
_omega_2     = _k2 / _m2
_deflection_ramp_vec = _phi**2 * _force_ramp_vec / _omega_2

# ddelta/dt = (phi² / omega_2) * (dF/da_p) * (da_p/dt)
# dF/da_p = Kf * f_tooth_mm * 1e-3,  da_p/dt = (a1-a0)/t_ramp durante la rampa, 0 despues
_dF_dap  = _Kf_fisico * _f_tooth_mm * 1e-3
_dap_dt_slope = (_a1_m - _a0_m) / _t_ramp
_ddelta_dt_vec = np.where(
    _t_ramp_vec <= _t_ramp,
    _phi**2 / _omega_2 * _dF_dap * _dap_dt_slope,
    0.0,
)

# Referencias exportadas: (t_ref, valores_ref)
_t_ramp_vec_corrected = _t_ramp_vec #+ 60/_n_rpm*2.5
force_ramp_ref: tuple[np.ndarray, np.ndarray] | None = (_t_ramp_vec_corrected, _force_ramp_vec)
deflection_ramp_ref: tuple[np.ndarray, np.ndarray] | None = (_t_ramp_vec_corrected, _deflection_ramp_vec)
ddelta_dt_ref: tuple[np.ndarray, np.ndarray] | None = (_t_ramp_vec_corrected, _ddelta_dt_vec)

print(f" Rampa Fuerza:       F_initial:{force_ramp_ref[1][0]:.3f} N  F_final:{force_ramp_ref[1][-1]:.3f} N  t_ramp:{force_ramp_ref[0][-1]:.4f} s")
print(f" Rampa Deflexión: d_initial:{deflection_ramp_ref[1][0]:.3e} m  d_final:{deflection_ramp_ref[1][-1]:.3e} m")
print(f" ddelta/dt:       {ddelta_dt_ref[1][0]:.3e} m/s  (0 fuera de la rampa)")

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────────────────────────────────────

# _t_reach = 60.0 / _n_rpm * 2.5  # tiempo de inicio de la rampa [s]
_t_reach = 0


def _compute_ap_at(t_abs: float) -> float:
    """
    Devuelve la profundidad de corte ap [m] en el instante t_abs [s]
    usando la rampa lineal definida en los parámetros del script.
    """
    t_rel = t_abs - _t_reach
    if t_rel <= 0.0:
        return float(_a0_m)
    if t_rel >= _t_ramp:
        return float(_a1_m)
    return float(_a0_m + (_a1_m - _a0_m) * t_rel / _t_ramp)


def read_case_attributes(
    h5_path: str | Path,
    cases: list[str],
) -> dict[str, dict]:
    """
    Lee los atributos de cada grupo 'case_xxx' del archivo .h5.
    Devuelve attrs[case_name] con todos los atributos disponibles.
    El atributo de interés principal es '$dxl_size$'.
    """
    attrs: dict = {}
    with open_h5(h5_path) as h5:
        for case_name in cases:
            if case_name not in h5:
                attrs[case_name] = {}
                continue
            attrs[case_name] = dict(h5[case_name].attrs)
    return attrs


def open_h5(h5_path: str | Path) -> h5py.File:
    """Abre el archivo .h5 y lo devuelve.  Aborta si no existe."""
    p = Path(h5_path)
    if not p.exists():
        sys.exit(f"[ERROR] Archivo no encontrado: {p}")
    return h5py.File(p, "r")


def list_cases(h5file: h5py.File) -> list[str]:
    """Devuelve todos los grupos cuyo nombre empieza por 'case_'."""
    return sorted(k for k in h5file.keys() if k.startswith("case_"))


def _read_signal(
    h5file: h5py.File,
    case_name: str,
    signal: str,          # "Axial_disp" o "Axial_vel"
) -> tuple[np.ndarray, np.ndarray]:
    """Lee times y values de un grupo señal dentro de un caso."""
    if case_name not in h5file:
        raise KeyError(f"Caso '{case_name}' no encontrado en el archivo.")
    case_grp = h5file[case_name]

    if signal not in case_grp:
        raise KeyError(f"Grupo '{signal}' no existe en '{case_name}'.")
    sig_grp = case_grp[signal]

    for ds in ("time", "values"):
        if ds not in sig_grp:
            raise KeyError(f"Dataset '{ds}' no existe en '{case_name}/{signal}'.")

    times  = sig_grp["time"][:]
    values = sig_grp["values"][:]

    if times.shape != values.shape:
        raise ValueError(
            f"'{case_name}/{signal}': times {times.shape} y values {values.shape} "
            "tienen formas distintas."
        )
    if times.size == 0:
        raise ValueError(f"'{case_name}/{signal}': señal vacía.")

    return times.astype(float), values.astype(float)


def _read_force_signal(
    h5file: h5py.File,
    case_name: str,
    signal: str = "res_R_p",
) -> tuple[np.ndarray, np.ndarray]:
    """Lee tiempo y valores de res_R_p dentro de un caso."""
    if case_name not in h5file:
        raise KeyError(f"Caso '{case_name}' no encontrado en el archivo.")
    case_grp = h5file[case_name]

    if signal not in case_grp:
        raise KeyError(f"Grupo '{signal}' no existe en '{case_name}'.")
    sig_grp = case_grp[signal]

    for ds in ("time", "values"):
        if ds not in sig_grp:
            raise KeyError(f"Dataset '{ds}' no existe en '{case_name}/{signal}'.")

    times = sig_grp["time"][:]
    values = sig_grp["values"][:]

    # res_R_p puede venir como matriz de varias columnas; nos quedamos con la primera.
    if values.ndim == 2:
        values = values[:, 0]
    elif values.ndim > 2:
        raise ValueError(
            f"'{case_name}/{signal}': values tiene ndim={values.ndim}, no soportado."
        )

    if times.shape != values.shape:
        raise ValueError(
            f"'{case_name}/{signal}': times {times.shape} y values {values.shape} "
            "tienen formas distintas."
        )
    if times.size == 0:
        raise ValueError(f"'{case_name}/{signal}': señal vacía.")

    return times.astype(float), values.astype(float)


def read_cases(
    h5_path: str | Path,
    selected_cases: dict[str, tuple[float, float]] | None = None,
    t_start_default: float = 0.0,
    t_end_default: float = 1.0,
) -> dict[str, dict]:
    """
    Lee los datos del archivo .h5.

    selected_cases puede ser:
      - None  - lee todos los casos con el intervalo (t_start_default, t_end_default)
      - dict  - { case_name: (t_start, t_end) }  con intervalo propio por caso

    Devuelve:
        data[case_name]["Axial_disp"]["times"]
        data[case_name]["Axial_disp"]["values"]
        data[case_name]["Axial_vel"]["times"]
        data[case_name]["Axial_vel"]["values"]
        data[case_name]["res_R_p"]["times"]
        data[case_name]["res_R_p"]["values"]
        data[case_name]["t_start"]
        data[case_name]["t_end"]
    """
    data: dict = {}

    with open_h5(h5_path) as h5:
        available = list_cases(h5)

        if not available:
            sys.exit("[ERROR] El archivo no contiene ningún grupo 'case_xxx'.")

        # Construir mapa { case_name: (t_start, t_end) }
        if selected_cases is None:
            cases_map = {c: (t_start_default, t_end_default) for c in available}
        else:
            missing = [c for c in selected_cases if c not in available]
            if missing:
                sys.exit(
                    f"[ERROR] Los siguientes casos no existen en el archivo: {missing}\n"
                    f"  Disponibles: {available}"
                )
            cases_map = selected_cases

        print(f"Casos encontrados en el archivo : {len(available)}")
        print(f"Casos a procesar                : {len(cases_map)}")

        for case_name, (ts, te) in cases_map.items():
            data[case_name] = {"t_start": ts, "t_end": te}
            for signal in ("Axial_disp", "Axial_vel"):
                try:
                    times, values = _read_signal(h5, case_name, signal)
                    data[case_name][signal] = {"times": times, "values": values}
                except (KeyError, ValueError) as exc:
                    print(f"  [AVISO] {exc} — caso omitido para '{signal}'.")
                    data[case_name][signal] = None

            try:
                f_time, f_values = _read_force_signal(h5, case_name, "res_R_p")
                data[case_name]["res_R_p"] = {"times": f_time, "values": f_values}
            except (KeyError, ValueError) as exc:
                print(f"  [AVISO] {exc} — caso omitido para 'res_R_p'.")
                data[case_name]["res_R_p"] = None

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Extracción de intervalo temporal
# ─────────────────────────────────────────────────────────────────────────────

def extract_time_window(
    times: np.ndarray,
    values: np.ndarray,
    t_start: float,
    t_end: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Devuelve el segmento de (times, values) dentro de [t_start, t_end].

    Lanza ValueError si el intervalo queda vacío.
    """
    if t_end <= t_start:
        raise ValueError(f"t_end ({t_end}) debe ser mayor que t_start ({t_start}).")

    mask = (times >= t_start) & (times <= t_end)
    if not np.any(mask):
        raise ValueError(
            f"No hay muestras en el intervalo [{t_start}, {t_end}] s. "
            f"Rango de la señal: [{times[0]:.4g}, {times[-1]:.4g}] s."
        )

    return times[mask], values[mask]


# ─────────────────────────────────────────────────────────────────────────────
# RMS móvil
# ─────────────────────────────────────────────────────────────────────────────

def moving_rms(
    times: np.ndarray,
    values: np.ndarray,
    window_size_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula la envolvente RMS con ventana móvil de tamaño window_size_time [s].

    Para que el RMS sea una envolvente suave y sin oscilaciones, la ventana debe
    cubrir AL MENOS un período completo de la señal (idealmente 2-5 períodos).

    Usa uniform_filter1d (scipy) sobre values**2, que refleja los bordes en lugar
    de rellenarlos con ceros, evitando artefactos en los extremos.

    Parámetros
    ----------
    times            : vector de tiempos [s]
    values           : señal de amplitudes
    window_size_time : tamaño de la ventana en segundos
                       REGLA: window_size_time >= 2 / f_dominante

    Devuelve
    --------
    times_rms  : mismo vector de tiempos que la entrada
    rms_values : envolvente RMS suave
    """
    if window_size_time <= 0:
        raise ValueError(f"window_size_time debe ser > 0, recibido: {window_size_time}")
    if times.size == 0:
        raise ValueError("La señal está vacía.")

    dt_mean = float(np.mean(np.diff(times))) if times.size > 1 else 1.0
    n_win   = max(1, int(round(window_size_time / dt_mean)))

    # Aviso si la ventana es pequeña respecto al contenido de la señal
    # (heurística: frecuencia dominante estimada por cruces por cero)
    zero_crossings = int(np.sum(np.diff(np.sign(values)) != 0))
    T_signal = times[-1] - times[0]
    if zero_crossings > 1 and T_signal > 0:
        f_est = zero_crossings / (2.0 * T_signal)   # [Hz] estimado
        T_est = 1.0 / f_est
        if window_size_time < T_est:
            print(
                f"  [AVISO] window_size_time={window_size_time:.4g}s es menor que"
                f" un período estimado ({T_est:.4g}s, f≈{f_est:.1f}Hz)."
                f" El RMS tendrá oscilaciones. Usa ventana >= {2*T_est:.4g}s."
            )

    # Promedio móvil sobre values² con reflexión en los bordes (sin artefactos)
    sq_smooth = uniform_filter1d(values ** 2, size=n_win, mode="reflect")
    rms_vals  = np.sqrt(np.maximum(sq_smooth, 0.0))

    return times, rms_vals


# ─────────────────────────────────────────────────────────────────────────────
# Funciones de graficado
# ─────────────────────────────────────────────────────────────────────────────

_SIGNAL_LABELS = {
    "Axial_disp":  ("Desplazamiento axial [m]",           "Axial_disp"),
    "Axial_vel":   ("Velocidad axial [m/s]",               "Axial_vel"),
    "dF":          ("Perturbación de fuerza ΔF [N]",         "dF"),
    "dF_norm":     ("Perturbación geométrica ΔF/ap [N/m]",  "dF_norm"),
}


def _plot_signals(
    data: dict,
    cases: list[str],
    signal: str,
    title: str,
    use_cut: bool = False,
) -> plt.Figure:
    """Figura genérica para señales completas o recortadas (intervalo por caso)."""
    ylabel, _ = _SIGNAL_LABELS[signal]
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.suptitle(title)

    for case_name in cases:
        sig = data[case_name].get(signal)
        if sig is None:
            continue
        t, v = sig["times"], sig["values"]
        if use_cut:
            ts = data[case_name]["t_start"]
            te = data[case_name]["t_end"]
            try:
                t, v = extract_time_window(t, v, ts, te)
            except ValueError as exc:
                print(f"  [AVISO] {case_name}/{signal}: {exc}")
                continue
            lbl = f"{case_name}  [{ts:.3g},{te:.3g}]s"
        else:
            lbl = case_name
        ax.plot(t, v, linewidth=0.8, label=lbl)

    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_full_signals(data: dict, cases: list[str]) -> tuple[plt.Figure, plt.Figure]:
    """Figuras 1 y 2: señales completas."""
    fig1 = _plot_signals(data, cases, "Axial_disp", "Axial_disp — señales completas")
    fig2 = _plot_signals(data, cases, "Axial_vel",  "Axial_vel  — señales completas")
    return fig1, fig2


def plot_cut_signals(
    data: dict,
    cases: list[str],
) -> tuple[plt.Figure, plt.Figure]:
    """Figuras 3 y 4: señales recortadas con el intervalo propio de cada caso."""
    fig3 = _plot_signals(
        data, cases, "Axial_disp",
        "Axial_disp — intervalos por caso",
        use_cut=True,
    )
    fig4 = _plot_signals(
        data, cases, "Axial_vel",
        "Axial_vel  — intervalos por caso",
        use_cut=True,
    )
    return fig3, fig4


def plot_rms(
    data: dict,
    cases: list[str],
    signal: str,
    window_size_time: float,
    case_attrs: dict | None = None,
) -> plt.Figure:

    _ylabels_math = {
        "Axial_disp": r"$\mathrm{RMS}(x_z)$  [m]",
        "Axial_vel":  r"$\mathrm{RMS}(\dot{x}_z)$  [m/s]",
        "dF":         r"$\mathrm{RMS}(\Delta F)$  [N]",
        "dF_norm":    r"$\mathrm{RMS}(\delta F_{i,\mathrm{n}})$  [N/m]",
    }
    _titles = {
        "Axial_disp": r"RMS envelope — Axial displacement $x_z$",
        "Axial_vel":  r"RMS envelope — Axial velocity $\dot{x}_z$",
        "dF":         r"RMS envelope — Force perturbation $\Delta F$",
        "dF_norm":    r"RMS envelope — $\delta F_{i,\mathrm{n}}$",
    }
    if True:  # estilo global aplicado por _configurar_estilo_global()
        from matplotlib.colors import LogNorm

        fig, ax = plt.subplots(figsize=(3.7 * fig_scale, 3.0 * fig_scale), constrained_layout=True)
        cmap = plt.colormaps["viridis"]

        # ── primera pasada: recoger h_vals y datos ──────────────────────────
        cases_plot: list[tuple[np.ndarray, np.ndarray, float]] = []
        h_vals: list[float] = []
        for idx, case_name in enumerate(cases):
            sig = data[case_name].get(signal)
            if sig is None:
                continue
            try:
                t_rms, rms_vals = moving_rms(sig["times"], sig["values"], window_size_time)
            except ValueError as exc:
                print(f"  [AVISO] RMS {case_name}/{signal}: {exc}")
                continue
            _attrs   = (case_attrs or {}).get(case_name, {})
            _dxl_raw = _attrs.get("$dxl_size$", _attrs.get("dxl_size", None))
            try:
                _dxl_val = float(_dxl_raw) if _dxl_raw is not None else float(idx + 1)
            except (TypeError, ValueError):
                _dxl_val = float(idx + 1)
            cases_plot.append((t_rms, rms_vals, _dxl_val))
            h_vals.append(_dxl_val)

        # ── norma log compartida entre curvas y barra ────────────────────────
        h_min = max(min(h_vals), 1e-15)
        h_max = max(h_vals)
        norm  = LogNorm(vmin=h_min, vmax=h_max)

        # ── segunda pasada: dibujar con color correcto ───────────────────────
        for t_rms, rms_vals, _dxl_val in cases_plot:
            ax.plot(t_rms, rms_vals, linewidth=1.2, color=cmap(norm(_dxl_val)))

        ax.set_yscale("log")
        ax.set_xlabel(r"Time$\ [\mathrm{s}]$")
        ax.set_ylabel(_ylabels_math.get(signal, r"$\mathrm{RMS}$"))
        ax.set_title(_titles.get(signal, f"RMS — {signal}"), pad=4)
        ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.5)

        # ── barra de color con ticks en notación científica ──────────────────
        if cases_plot:
            cbar = fig.colorbar(
                plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                ax=ax, pad=0.02,
            )
            # Ticks log-espaciados en los valores reales de h
            tick_vals = np.geomspace(h_min, h_max, min(len(h_vals), 5))
            # Exponente común del tick mayor → aparece solo en el label
            _exp   = int(np.floor(np.log10(h_max))) if h_max > 0 else 0
            _scale = 10.0 ** _exp
            cbar.set_ticks(tick_vals)
            cbar.set_ticklabels([f"{v / _scale:.2f}" for v in tick_vals])
            cbar.set_label(rf"$h_{{d,i}}\ (\times 10^{{{_exp}}}\ \mathrm{{m}})$")
    return fig


def plot_signal_and_rms(
    data: dict,
    cases: list[str],
    signal: str,
    window_size_time: float,
) -> list[plt.Figure]:
    """Una figura por caso con señal recortada + RMS — intervalo propio de cada caso."""
    ylabel, _ = _SIGNAL_LABELS[signal]
    figs = []

    for case_name in cases:
        sig = data[case_name].get(signal)
        if sig is None:
            continue
        ts = data[case_name]["t_start"]
        te = data[case_name]["t_end"]
        try:
            t_cut, v_cut = extract_time_window(sig["times"], sig["values"], ts, te)
            t_rms, rms_vals = moving_rms(t_cut, v_cut, window_size_time)
        except ValueError as exc:
            print(f"  [AVISO] {case_name}/{signal}: {exc}")
            continue

        fig, ax = plt.subplots(figsize=(11, 4))
        fig.suptitle(
            f"{case_name}  [{ts:.3g},{te:.3g}]s — {signal} "
            f"— señal + RMS (ventana {window_size_time*1e3:.2f} ms)"
        )
        ax.plot(t_cut, v_cut, color="steelblue", linewidth=0.6, alpha=0.7, label="señal")
        ax.plot(t_rms, rms_vals, color="tomato", linewidth=1.4, label="RMS móvil")
        ax.set_xlabel("Tiempo [s]")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        figs.append(fig)

    return figs


def plot_force_cases(data: dict, cases: list[str]) -> plt.Figure:
    """Grafica res_R_p por caso en una sola figura."""
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.suptitle("res_R_p — esfuerzo por caso")

    for case_name in cases:
        sig = data[case_name].get("res_R_p")
        if sig is None:
            continue
        t = sig["times"]
        v = sig["values"]
        if v.ndim > 1:
            v = v[:, 0]
        ax.plot(t, v, linewidth=0.9, label=case_name)

    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("res_R_p [N]")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def subtract_reference_force(
    t_signal: np.ndarray,
    v_signal: np.ndarray,
    t_ref: np.ndarray,
    v_ref: np.ndarray,
) -> np.ndarray:
    """
    Resta la fuerza de referencia v_ref(t_ref) de la señal v_signal(t_signal).

    Como los dos vectores pueden tener distinta longitud y distinto dt, usa
    np.interp para evaluar v_ref en los instantes de t_signal.
    Fuera del rango [t_ref[0], t_ref[-1]] extiende con los valores de borde.
    """
    v_ref_interp = np.interp(t_signal, t_ref, v_ref,
                             left=v_ref[0], right=v_ref[-1])
    return v_signal - v_ref_interp


def plot_force_corrected(
    data: dict,
    cases: list[str],
    t_ref: np.ndarray,
    v_ref: np.ndarray,
) -> plt.Figure:
    """Figura con dos subplots: señal original arriba, fuerza dinámica (corregida) abajo."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    axes[0].set_title("res_R_p — señal completa (original)")
    axes[1].set_title("res_R_p − F_rampa — fuerza dinámica")

    for case_name in cases:
        sig = data[case_name].get("res_R_p")
        if sig is None:
            continue
        t = sig["times"]
        v = sig["values"]
        axes[0].plot(t, v, linewidth=0.7, alpha=0.8, label=case_name)
        v_corr = subtract_reference_force(t, v, t_ref, v_ref)
        axes[1].plot(t, v_corr, linewidth=0.7, alpha=0.8, label=case_name)

    axes[0].plot(t_ref, v_ref, color="black", linewidth=1.5,
                 linestyle="--", label="F_rampa teórica")
    for ax in axes:
        ax.set_xlabel("Tiempo [s]")
        ax.set_ylabel("Fuerza [N]")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Sustracción de fuerza nominal (rampa teórica)")
    fig.tight_layout()
    return fig


def plot_disp_corrected(
    data: dict,
    cases: list[str],
    t_ref: np.ndarray,
    v_ref: np.ndarray,
) -> plt.Figure:
    """Figura con dos subplots: desplazamiento original arriba, componente dinámica (corregida) abajo."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    axes[0].set_title("Axial_disp — señal completa (original)")
    axes[1].set_title("Axial_disp − deflexión_estática — componente dinámica")

    for case_name in cases:
        sig = data[case_name].get("Axial_disp")
        if sig is None:
            continue
        t = sig["times"]
        v = sig["values"]
        axes[0].plot(t, v, linewidth=0.7, alpha=0.8, label=case_name)
        v_corr = subtract_reference_force(t, v, t_ref, v_ref)
        axes[1].plot(t, v_corr, linewidth=0.7, alpha=0.8, label=case_name)

    axes[0].plot(t_ref, v_ref, color="black", linewidth=1.5,
                 linestyle="--", label="deflexión teórica")
    for ax in axes:
        ax.set_xlabel("Tiempo [s]")
        ax.set_ylabel("Desplazamiento [m]")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Sustracción de deflexión estática (rampa teórica)")
    fig.tight_layout()
    return fig


def plot_vel_corrected(
    data: dict,
    cases: list[str],
    t_ref: np.ndarray,
    v_ref: np.ndarray,
) -> plt.Figure:
    """Figura con dos subplots: velocidad original arriba, componente dinámica (corregida) abajo."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    axes[0].set_title("Axial_vel — señal completa (original)")
    axes[1].set_title("Axial_vel − dδ/dt — componente dinámica")

    for case_name in cases:
        sig = data[case_name].get("Axial_vel")
        if sig is None:
            continue
        t = sig["times"]
        v = sig["values"]
        axes[0].plot(t, v, linewidth=0.7, alpha=0.8, label=case_name)
        v_corr = subtract_reference_force(t, v, t_ref, v_ref)
        axes[1].plot(t, v_corr, linewidth=0.7, alpha=0.8, label=case_name)

    axes[0].plot(t_ref, v_ref, color="black", linewidth=1.5,
                 linestyle="--", label="dδ/dt teórica")
    for ax in axes:
        ax.set_xlabel("Tiempo [s]")
        ax.set_ylabel("Velocidad [m/s]")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Sustracción de velocidad estática (dδ/dt rampa teórica)")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Ajuste exponencial  R(t) = y0 * exp(beta * t)  =>  ln(R) = c + beta*t
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExponentialFit:
    """Resultado del ajuste R(t) = y0 * exp(beta * t) sobre ln(RMS)."""
    case_name: str
    beta:      float        # tasa de crecimiento [1/s]
    c_ln:      float        # intercepto  c = ln(y0)
    y0:        float        # perturbación inicial equivalente
    m_log10:   float        # pendiente en log10 = beta / ln(10)
    a_log10:   float        # intercepto en log10 = log10(y0)
    r2:        float        # R² del ajuste lineal sobre ln(RMS)
    t_fit:     np.ndarray
    rms_fit:   np.ndarray


def fit_log_rms(
    times: np.ndarray,
    rms_vals: np.ndarray,
    case_name: str = "",
) -> "ExponentialFit | None":
    """
    Ajusta R(t) = y0 * exp(beta * t) por regresión lineal sobre ln(RMS).

    Devuelve None si no hay suficientes muestras válidas (< 3 con RMS > 0).
    """
    mask = rms_vals > 0
    t_v, r_v = times[mask], rms_vals[mask]
    if t_v.size < 3:
        print(f"  [AVISO] {case_name}: muy pocas muestras RMS>0 para ajustar.")
        return None
    ln_rms = np.log(r_v)
    coeffs = np.polyfit(t_v, ln_rms, 1)   # [beta, c_ln]
    beta, c_ln = float(coeffs[0]), float(coeffs[1])
    y0 = float(np.exp(c_ln))
    ln_pred = beta * t_v + c_ln
    ss_res  = float(np.sum((ln_rms - ln_pred) ** 2))
    ss_tot  = float(np.sum((ln_rms - np.mean(ln_rms)) ** 2))
    r2      = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else float("nan")
    return ExponentialFit(
        case_name=case_name, beta=beta, c_ln=c_ln, y0=y0,
        m_log10=beta / np.log(10), a_log10=c_ln / np.log(10),
        r2=r2, t_fit=times.copy(), rms_fit=y0 * np.exp(beta * times),
    )


def compute_delay(
    fit1: "ExponentialFit",
    fit2: "ExponentialFit",
    level_A: float | None = None,
    beta_rtol: float = 0.10,
) -> dict:
    """
    Calcula alpha y Delta_t a partir de dos ExponentialFit.

    Si |beta1-beta2|/beta_medio < beta_rtol  (misma dinámica):
        alpha   = exp(c1 - c2)
        Delta_t = (c1 - c2) / beta_medio  =  ln(alpha) / beta

    Si betas distintas, Delta_t depende del nivel A.
    """
    beta_mean = 0.5 * (fit1.beta + fit2.beta)
    alpha     = float(np.exp(fit1.c_ln - fit2.c_ln))
    ln_alpha  = float(np.log(alpha))
    rel_diff  = abs(fit1.beta - fit2.beta) / max(abs(beta_mean), 1e-12)
    same_beta = rel_diff < beta_rtol
    delta_t   = (fit1.c_ln - fit2.c_ln) / beta_mean if same_beta else None
    result: dict = {
        "case_1": fit1.case_name, "case_2": fit2.case_name,
        "beta_1": fit1.beta,      "beta_2": fit2.beta, "beta_mean": beta_mean,
        "c1_ln":  fit1.c_ln,      "c2_ln":  fit2.c_ln,
        "y0_1":   fit1.y0,        "y0_2":   fit2.y0,
        "alpha":  alpha, "ln_alpha": ln_alpha, "delta_t": delta_t,
        "same_beta": same_beta,   "r2_1": fit1.r2, "r2_2": fit2.r2,
    }
    if level_A is not None and level_A > 0:
        t1_A = (np.log(level_A) - fit1.c_ln) / fit1.beta
        t2_A = (np.log(level_A) - fit2.c_ln) / fit2.beta
        result.update({"level_A": level_A, "t1_at_A": float(t1_A),
                       "t2_at_A": float(t2_A), "delta_t_at_A": float(t2_A - t1_A)})
    return result


def print_delay_summary(res: dict) -> None:
    """Imprime en consola el resumen del análisis de retraso."""
    print("\n" + "="*60)
    print(f"  Análisis exponencial: {res['case_1']}  vs  {res['case_2']}")
    print("="*60)
    for key, lbl in [
        ("beta_1",    "beta_1      [1/s]"),
        ("beta_2",    "beta_2      [1/s]"),
        ("beta_mean", "beta_medio  [1/s]"),
        ("c1_ln",     "c1 = ln(y0_1)   "),
        ("c2_ln",     "c2 = ln(y0_2)   "),
        ("y0_1",      "y0_1            "),
        ("y0_2",      "y0_2            "),
        ("alpha",     "alpha = y01/y02 "),
        ("ln_alpha",  "ln(alpha)       "),
        ("r2_1",      "R²_1            "),
        ("r2_2",      "R²_2            "),
    ]:
        print(f"  {lbl} = {res[key]:.4g}")
    if res["same_beta"]:
        print(f"  => Misma dinámica")
        print(f"  => Delta_t = (c1-c2)/beta = {res['delta_t']:.4f}  s")
    else:
        print("  => Pendientes distintas: Delta_t depende del nivel.")
    if "delta_t_at_A" in res:
        print(f"  => Delta_t @ A={res['level_A']:.3e}: {res['delta_t_at_A']:.4f} s")
    print("="*60)


def _apply_fig_style(fig: plt.Figure) -> None:
    """Re-aplica el estilo de Etapa4_bis (rcParams actuales) sobre una figura
    ya creada — útil para figuras importadas desde time.py u otros módulos."""
    fs      = plt.rcParams["font.size"]
    fs_ax   = plt.rcParams["axes.labelsize"]
    fs_tick = plt.rcParams["xtick.labelsize"]
    fs_leg  = plt.rcParams["legend.fontsize"]
    fs_tit  = plt.rcParams["axes.titlesize"]
    lw      = plt.rcParams["axes.linewidth"]
    family  = plt.rcParams["font.family"]
    # font.family puede ser lista o string
    fname   = family[0] if isinstance(family, list) else family

    def _set_text(txt):
        txt.set_fontsize(fs)
        txt.set_fontfamily(fname)

    for ax in fig.axes:
        # Etiquetas y título
        ax.xaxis.label.set_fontsize(fs_ax);  ax.xaxis.label.set_fontfamily(fname)
        ax.yaxis.label.set_fontsize(fs_ax);  ax.yaxis.label.set_fontfamily(fname)
        ax.title.set_fontsize(fs_tit);       ax.title.set_fontfamily(fname)

        # Tick labels
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontsize(fs_tick)
            lbl.set_fontfamily(fname)

        # Parámetros de ticks
        ax.tick_params(
            axis="both", which="major",
            direction=plt.rcParams["xtick.direction"],
            width=plt.rcParams["xtick.major.width"],
            length=plt.rcParams["xtick.major.size"],
            labelsize=fs_tick,
        )

        # Bordes
        for spine in ax.spines.values():
            spine.set_linewidth(lw)

        # Fondo del axes
        ax.set_facecolor(plt.rcParams["axes.facecolor"])

        # offset_text del eje Y (notación científica) — time.py lo fija a 14
        off = ax.yaxis.get_offset_text()
        off.set_size(fs_tick)
        off.set_fontfamily(fname)

        # Leyenda (si existe)
        leg = ax.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                txt.set_fontsize(fs_leg)
                txt.set_fontfamily(fname)
            leg.get_frame().set_linewidth(0)
            if leg.get_title():
                leg.get_title().set_fontsize(fs_leg)
                leg.get_title().set_fontfamily(fname)

    fig.set_facecolor(plt.rcParams["figure.facecolor"])


def plot_stability_with_betas(
    case_results: dict,
    cases: list[str],
    signal: str,
    case_attrs: dict | None = None,
) -> plt.Figure:
    """
    Obtiene la figura de estabilidad desde time.py (toda la config queda allí)
    y superpone los puntos (t_at_Aref_fit, β_sim) de cada caso en un eje derecho.
    """
    import importlib.util as _ilu
    import sys as _sys

    _time_path = (
        r"c:\Users\quiqu\OneDrive-ensam.eu\Desktop\Thesis"
        r"\04-Articles\Manufacturing_21\Artiuclo_Manuf21_2026"
        r"\Comparation\time.py"
    )
    _spec = _ilu.spec_from_file_location("time_article", _time_path)
    _tm   = _ilu.module_from_spec(_spec)
    _sys.modules["time_article"] = _tm
    _spec.loader.exec_module(_tm)
    # time.py puede haber sobrescrito rcParams — los restauramos antes de crear figuras
    _configurar_estilo_global()

    # time.py genera la figura con toda su configuración interna
    fig, ax_stab = _tm.build_stability_figure(figsize=(3.7 * fig_scale, 3.0 * fig_scale))

    # ── Puntos β_sim vs t_at_Aref_fit — mismo eje Y que Re(λ_max) ────────
    from matplotlib.colors import LogNorm
    _cmap = plt.colormaps["viridis"]

    # Primera pasada: recoger h_vals válidos
    _pts_raw: list[tuple] = []
    for _idx, _cn in enumerate(cases):
        _r = case_results.get(_cn, {}).get(signal)
        if _r is None:
            continue
        _t_star = _r.get("t_at_Aref_fit", float("nan"))
        _beta   = _r.get("beta",          float("nan"))
        if not (np.isfinite(_t_star) and np.isfinite(_beta)):
            continue
        _attrs   = (case_attrs or {}).get(_cn, {})
        _dxl_raw = _attrs.get("$dxl_size$", _attrs.get("dxl_size", None))
        try:
            _dxl_val = float(_dxl_raw) if _dxl_raw is not None else float(_idx + 1)
        except (TypeError, ValueError):
            _dxl_val = float(_idx + 1)
        _pts_raw.append((_t_star, _beta, _dxl_val))

    if _pts_raw:
        _h_vals = [h for _, _, h in _pts_raw]
        _h_min  = max(min(_h_vals), 1e-15)
        _h_max  = max(_h_vals)
        _norm   = LogNorm(vmin=_h_min, vmax=_h_max)

        for _t, _b, _dxl_val in _pts_raw:
            ax_stab.scatter(_t, _b, color=_cmap(_norm(_dxl_val)),
                            s=150, zorder=5, edgecolors="black", linewidth=1.0)

        # Barra de color igual que plot_rms
        # _cbar = fig.colorbar(
        #     plt.cm.ScalarMappable(cmap=_cmap, norm=_norm),
        #     ax=ax_stab, pad=0.02,
        # )
        # _tick_vals = np.geomspace(_h_min, _h_max, min(len(_h_vals), 5))
        # _exp   = int(np.floor(np.log10(_h_max))) if _h_max > 0 else 0
        # _scale = 10.0 ** _exp
        # _cbar.set_ticks(_tick_vals)
        # _cbar.set_ticklabels([f"{v / _scale:.2f}" for v in _tick_vals])
        # _cbar.set_label(rf"$h_{{d,i}}\ (\times 10^{{{_exp}}}\ \mathrm{{m}})$")

    _apply_fig_style(fig)
    # Sobrescribir título y ylabel DESPUÉS del style para que usen las fuentes correctas
    ax_stab.set_title("")
    ax_stab.set_ylabel(r"$\beta =\ \mathrm{Re}\{\lambda_{\max}\}$  [s$^{-1}$]")
    fig.tight_layout()
    return fig


def plot_delta_t_bars(
    pair_results: dict,
    signal: str,
    case_attrs: dict | None = None,
    pair_mode: str = "consecutive",  # "consecutive" | "vs_ref" | "all"
    max_pairs: int = 10,              # submuestrea uniformemente si hay más pares
) -> plt.Figure:
    """
    Gráfico de barras agrupadas por par (i,j) con cinco series (todas barras simples):
      1. Δt_mes   — referencia medida       (delta_t_Aref)
      2. Δt_fit   — β_i=β_j, ajustamiento  (delta_t_fit)
      3. Δt_F     — β_i=β_j, proxy h       (delta_t_force_h_C)
      4. Δt_decomp_fit  — descomposición ajustamiento sumada (dt_decomp)
      5. Δt_decomp_F    — descomposición fuerza sumada       (dt_decomp3)

    pair_mode : "consecutive" — solo pares vecinos ordenados por dxl_size (recomendado)
                "vs_ref"      — todos vs el caso de menor dxl
                "all"         — todos los pares calculados
    """
    pairs_data = pair_results.get(signal, {})
    if not pairs_data:
        fig, ax = plt.subplots(figsize=(3.5 * fig_scale, 3.0 * fig_scale))
        ax.set_title(f"No hay pares para {signal}")
        return fig

    # ── Obtener dxl_size para ordenar casos ───────────────────────────────
    def _dxl_val(cn: str) -> float:
        _attrs = (case_attrs or {}).get(cn, {})
        _raw = _attrs.get("$dxl_size$", _attrs.get("dxl_size", None))
        try:
            return float(_raw)
        except (TypeError, ValueError):
            return float("inf")

    # Extraer todos los casos únicos que aparecen en los pares, ordenados por dxl
    _all_cases = sorted(
        {cn for pair in pairs_data for cn in pair},
        key=_dxl_val,
    )
    # Rango ordinal: 1 = dxl más pequeño
    _rank = {cn: k + 1 for k, cn in enumerate(_all_cases)}

    def _dxl_label(cn: str) -> str:
        return str(_rank.get(cn, cn.replace("case_", "")))

    # Seleccionar pares según mode
    if pair_mode == "consecutive":
        _selected_keys = [
            (_all_cases[k], _all_cases[k + 1])
            for k in range(len(_all_cases) - 1)
            if (_all_cases[k], _all_cases[k + 1]) in pairs_data
        ]
    elif pair_mode == "vs_ref":
        _ref = _all_cases[0]
        _selected_keys = [
            (_ref, cn) for cn in _all_cases[1:]
            if (_ref, cn) in pairs_data
        ]
    else:  # "all"
        _selected_keys = list(pairs_data.keys())

    if not _selected_keys:
        _selected_keys = list(pairs_data.keys())

    # ── Submuestra uniforme si hay más pares que max_pairs ───────────────
    if max_pairs > 0 and len(_selected_keys) > max_pairs:
        step = (len(_selected_keys) - 1) / (max_pairs - 1)
        _selected_keys = [_selected_keys[round(i * step)] for i in range(max_pairs)]

    # ── Acumular valores ───────────────────────────────────────────────────
    pair_labels = []
    v_mes, v_fit, v_force, v_decomp, v_decomp3 = [], [], [], [], []

    for key in _selected_keys:
        pr = pairs_data[key]
        cn_i, cn_j = key
        pair_labels.append(f"${_dxl_label(cn_i)} \\to {_dxl_label(cn_j)}$")
        v_mes.append(pr.get("delta_t_Aref",       float("nan")))
        v_fit.append(pr.get("delta_t_fit",         float("nan")))
        v_force.append(pr.get("delta_t_force_h_C", float("nan")))
        v_decomp.append(pr.get("dt_decomp",        float("nan")))
        v_decomp3.append(pr.get("dt_decomp3",      float("nan")))

    n_pairs = len(pair_labels)
    x = np.arange(n_pairs)

    # ── 5 barras por posición, centradas ──────────────────────────────────
    # Más hueco entre barras para distinguir mejor cada serie dentro del par.
    w  = 0.15
    gap = 0.02
    total = 5 * w + 4 * gap
    starts = [-total / 2 + k * (w + gap) for k in range(5)]

    c_mes    = "#4c78a8"  # azul steel
    c_fit    = "#f58518"  # naranja
    c_force  = "#54a24b"  # verde
    c_decomp = "#e45756"  # rojo
    c_dec3   = "#b279a2"  # violeta
    hatches = ["//", "\\\\", "..", "xx", "oo", "--", "++"]

    # Garantizar estilo global antes de crear la figura
    _configurar_estilo_global()
    fig, ax = plt.subplots(figsize=(3.5 * fig_scale * 2.0, 3.0 * fig_scale))

    for idx, xpos in enumerate(x):
         hatch = hatches[idx % len(hatches)]
         ax.bar(xpos + starts[0], v_mes[idx],    width=w, color=c_mes,
             label=r"$\Delta t^{\mathrm{mes}}$" if idx == 0 else None,
             edgecolor="black", linewidth=0.45, hatch=hatch)
         ax.bar(xpos + starts[1], v_fit[idx],    width=w, color=c_fit,
             label=r"$\Delta t^{\mathrm{fit}}\ (\beta_i{=}\beta_j)$" if idx == 0 else None,
             edgecolor="black", linewidth=0.45, hatch=hatch)
         ax.bar(xpos + starts[2], v_force[idx],  width=w, color=c_force,
             label=r"$\Delta t^{F}\ (\beta_i{=}\beta_j)$" if idx == 0 else None,
             edgecolor="black", linewidth=0.45, hatch=hatch)
         ax.bar(xpos + starts[3], v_decomp[idx], width=w, color=c_decomp,
             label=r"$\Delta t^{\mathrm{fit}}\  (\beta_i{\neq}\beta_j)$" if idx == 0 else None,
             edgecolor="black", linewidth=0.45, hatch=hatch)
         ax.bar(xpos + starts[4], v_decomp3[idx], width=w, color=c_dec3,
             label=r"$\Delta t^{F}\ (\beta_i{\neq}\beta_j)$" if idx == 0 else None,
             edgecolor="black", linewidth=0.45, hatch=hatch)

    ax.axhline(0.0, color="black", linewidth=0.7, linestyle="--", zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, rotation=0, ha="center")
    ax.set_ylabel(r"$\Delta t\ [\mathrm{s}]$")
    ax.set_xlabel(r"Paire (rang $i \to j$,  $h_{d,1}<\cdots<h_{d,N}$)")
    ax.set_title(r"Comparaison des estimateurs du décalage temporel par paire")
    _apply_fig_style(fig)
    ax.legend(ncol=2, loc="upper left",
              handlelength=1.4, columnspacing=0.8, labelspacing=0.3, fontsize=plt.rcParams.get("legend.fontsize", 10)*1)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    
    fig.tight_layout()
    return fig


def plot_rms_with_fits(
    data: dict,
    cases: list[str],
    signal: str,
    window_size_time: float,
    fits: dict | None = None,
    delay_result: dict | None = None,
    ln_A_ref_val: float | None = None,
    case_attrs: dict | None = None,
) -> plt.Figure:
    """Figura de ln(RMS) + rectas ajustadas — estilo artículo 2 columnas.

    Tamaño: una columna (~3.5 in). Tipografía serif / STIX. Leyenda ligera
    (una entrada por caso: nombre corto + β). Sin suptitle.
    """
    if True:  # estilo global aplicado por _configurar_estilo_global()
        from matplotlib.colors import LogNorm

        fig, ax = plt.subplots(figsize=(3.7 * fig_scale, 3.0 * fig_scale), constrained_layout=True)

        ylabel, _ = _SIGNAL_LABELS[signal]
        _ylabels_math = {
            "Axial_disp":  r"$\ln(\mathrm{RMS}(x_z))$  [m]",
            "Axial_vel":   r"$\ln(\mathrm{RMS}(\dot{x}_z))$  [m/s]",
            "dF":          r"$\ln(\mathrm{RMS}(\Delta F))$  [N]",
            "dF_norm":     r"$\ln(\mathrm{RMS}(\Delta F/a_p))$  [N/m]",
        }
        y_math_label = _ylabels_math.get(signal, r"$\ln(\mathrm{RMS})$")
        cmap = plt.colormaps["viridis"]
        if fits is None:
            fits = {}

        # ── primera pasada: recoger h_vals ───────────────────────────────
        case_info: list[tuple[str, float]] = []
        for idx, case_name in enumerate(cases):
            if data[case_name].get(signal) is None:
                continue
            _attrs   = (case_attrs or {}).get(case_name, {})
            _dxl_raw = _attrs.get("$dxl_size$", _attrs.get("dxl_size", None))
            try:
                _dxl_val = float(_dxl_raw) if _dxl_raw is not None else float(idx + 1)
            except (TypeError, ValueError):
                _dxl_val = float(idx + 1)
            case_info.append((case_name, _dxl_val))

        h_vals = [h for _, h in case_info]
        h_min  = max(min(h_vals), 1e-15) if h_vals else 1e-15
        h_max  = max(h_vals) if h_vals else 1.0
        norm   = LogNorm(vmin=h_min, vmax=h_max)

        # ── segunda pasada: dibujar ──────────────────────────────────────
        for case_name, _dxl_val in case_info:
            color = cmap(norm(_dxl_val))
            sig   = data[case_name].get(signal)
            ts = data[case_name]["t_start"]
            te = data[case_name]["t_end"]
            try:
                t_cut, v_cut = extract_time_window(sig["times"], sig["values"], ts, te)
                t_rms, rms_v = moving_rms(t_cut, v_cut, window_size_time)
            except ValueError as exc:
                print(f"  [AVISO] {case_name}/{signal}: {exc}")
                continue

            rms_mask  = rms_v > 0
            t_rms_log = t_rms[rms_mask]
            ln_rms    = np.log(rms_v[rms_mask])
            ax.plot(t_rms_log, ln_rms, color=color, linewidth=1.4, alpha=0.45)

            fit = fits.get(case_name)
            if fit is None:
                fit = fit_log_rms(t_rms, rms_v, case_name=case_name)
                fits[case_name] = fit
            if fit is not None:
                ax.plot(
                    fit.t_fit, fit.beta * fit.t_fit + fit.c_ln,
                    color=color, linewidth=2.5, linestyle="--",
                )

                if ln_A_ref_val is not None:
                    t_star = (ln_A_ref_val - fit.c_ln) / fit.beta
                    ax.axvline(t_star, color=color, linewidth=2.5,
                               linestyle=":", alpha=0.7)
                    ax.scatter([t_star], [ln_A_ref_val], color=color,
                               s=150, zorder=5, marker="o",
                               edgecolors="black", linewidths=1.0)

        # ── Línea horizontal ln_A_ref ────────────────────────────────────
        if ln_A_ref_val is not None:
            ax.axhline(ln_A_ref_val, color="k", linewidth=1.8,
                       linestyle="-.", alpha=0.8,
                       label=f"$\\ln A_{{\\rm ref}}={ln_A_ref_val:.1f}$")
            ax.legend(loc="upper left", handlelength=1.2)

        # ── Anotación Δt ─────────────────────────────────────────────────
        if delay_result is not None and delay_result.get("delta_t") is not None:
            txt = (
                f"$\\Delta t={delay_result['delta_t']:.3f}$ s\n"
                f"$\\alpha={delay_result['alpha']:.3f}$\n"
                f"$\\bar{{\\beta}}={delay_result['beta_mean']:.2f}$ s$^{{-1}}$"
            )
            ax.text(0.03, 0.97, txt, transform=ax.transAxes,
                    fontsize=plt.rcParams.get('legend.fontsize', 10) - 1,
                    va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="w",
                              edgecolor="0.7", alpha=0.85, linewidth=0.5))

        # ── Etiquetas, título, grid ──────────────────────────────────────
        ax.set_xlabel(r"Time $\ [\mathrm{s}]$")
        ax.set_ylabel(y_math_label)
        ax.set_title(
            r"Exponential growth — Axial displacement $x_z$"
            if signal == "Axial_disp"
            else r"Exponential growth — Axial velocity $\dot{x}_z$",
            pad=4,
        )
        ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

        # ── Barra de color (igual que plot_rms) ─────────────────────────
        if case_info:
            cbar = fig.colorbar(
                plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                ax=ax, pad=0.02,
            )
            tick_vals = np.geomspace(h_min, h_max, min(len(h_vals), 5))
            _exp   = int(np.floor(np.log10(h_max))) if h_max > 0 else 0
            _scale = 10.0 ** _exp
            cbar.set_ticks(tick_vals)
            cbar.set_ticklabels([f"{v / _scale:.2f}" for v in tick_vals])
            cbar.set_label(rf"$h_{{d,i}}\ (\times 10^{{{_exp}}}\ \mathrm{{m}})$")

    return fig


def plot_rms_fits_only(
    data: dict,
    cases: list[str],
    signal: str,
    window_size_time: float,
    fits: dict | None = None,
    case_attrs: dict | None = None,
) -> plt.Figure:
    """Como plot_rms_with_fits pero solo curva RMS + recta de ajuste.
    Sin marcadores A_ref, sin líneas verticales/horizontales.
    """
    from matplotlib.colors import LogNorm

    _ylabels_math = {
        "Axial_disp":  r"$\ln(\mathrm{RMS}(x_z))$  [m]",
        "Axial_vel":   r"$\ln(\mathrm{RMS}(\dot{x}_z))$  [m/s]",
        "dF":          r"$\ln(\mathrm{RMS}(\Delta F))$  [N]",
        "dF_norm":     r"$\ln(\mathrm{RMS}(\Delta F/a_p))$  [N/m]",
    }
    _titles = {
        "Axial_disp": r"Exponential growth — Axial displacement $x_z$",
        "Axial_vel":  r"Exponential growth — Axial velocity $\dot{x}_z$",
        "dF":         r"Exponential growth — $\Delta F$",
        "dF_norm":    r"Exponential growth — $\delta F_{i,\mathrm{n}}$",
    }

    fig, ax = plt.subplots(figsize=(3.7 * fig_scale, 3.0 * fig_scale), constrained_layout=True)
    cmap = plt.colormaps["viridis"]
    if fits is None:
        fits = {}

    # Primera pasada: recoger h_vals
    case_info: list[tuple[str, float]] = []
    for idx, case_name in enumerate(cases):
        if data[case_name].get(signal) is None:
            continue
        _attrs   = (case_attrs or {}).get(case_name, {})
        _dxl_raw = _attrs.get("$dxl_size$", _attrs.get("dxl_size", None))
        try:
            _dxl_val = float(_dxl_raw) if _dxl_raw is not None else float(idx + 1)
        except (TypeError, ValueError):
            _dxl_val = float(idx + 1)
        case_info.append((case_name, _dxl_val))

    h_vals = [h for _, h in case_info]
    h_min  = max(min(h_vals), 1e-15) if h_vals else 1e-15
    h_max  = max(h_vals) if h_vals else 1.0
    norm   = LogNorm(vmin=h_min, vmax=h_max)

    # Segunda pasada: dibujar
    for case_name, _dxl_val in case_info:
        color = cmap(norm(_dxl_val))
        sig   = data[case_name].get(signal)
        ts = data[case_name]["t_start"]
        te = data[case_name]["t_end"]
        try:
            t_cut, v_cut = extract_time_window(sig["times"], sig["values"], ts, te)
            t_rms, rms_v = moving_rms(t_cut, v_cut, window_size_time)
        except ValueError as exc:
            print(f"  [AVISO] {case_name}/{signal}: {exc}")
            continue

        mask = rms_v > 0
        ax.plot(t_rms[mask], np.log(rms_v[mask]), color=color, linewidth=1.4, alpha=0.45)

        fit = fits.get(case_name)
        if fit is None:
            fit = fit_log_rms(t_rms, rms_v, case_name=case_name)
            fits[case_name] = fit
        if fit is not None:
            ax.plot(fit.t_fit, fit.beta * fit.t_fit + fit.c_ln,
                    color=color, linewidth=2.5, linestyle="--")

    ax.set_xlabel(r"Time $\ [\mathrm{s}]$")
    ax.set_ylabel(_ylabels_math.get(signal, r"$\ln(\mathrm{RMS})$"))
    ax.set_title(_titles.get(signal, f"Exponential growth — {signal}"), pad=4)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

    if case_info:
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap=cmap, norm=norm),
            ax=ax, pad=0.02,
        )
        tick_vals = np.geomspace(h_min, h_max, min(len(h_vals), 5))
        _exp   = int(np.floor(np.log10(h_max))) if h_max > 0 else 0
        _scale = 10.0 ** _exp
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels([f"{v / _scale:.2f}" for v in tick_vals])
        cbar.set_label(rf"$h_{{d,i}}\ (\times 10^{{{_exp}}}\ \mathrm{{m}})$")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Leer datos
    data = read_cases(
        h5_path, selected_cases,
        t_start_default=t_start_default,
        t_end_default=t_end_default,
    )
    cases = list(data.keys())

    # 2. Figuras 1 y 2 — señales completas
    # fig1, fig2 = plot_full_signals(data, cases)

    # 3. Figuras 3 y 4 — señales recortadas (intervalo propio de cada caso)
    fig3, fig4 = plot_cut_signals(data, cases)

    # 4. Esfuerzo res_R_p por caso
    fig_force = plot_force_cases(data, cases)

    # 4b. Fuerza dinámica (res_R_p − rampa teórica)
    if force_ramp_ref is not None:
        t_ref, v_ref = force_ramp_ref
        # fig_force_corr = plot_force_corrected(data, cases, t_ref, v_ref)

    # 4c. Desplazamiento dinámico (Axial_disp − deflexión estática teórica)
    if deflection_ramp_ref is not None:
        t_ref_d, v_ref_d = deflection_ramp_ref
        # fig_disp_corr = plot_disp_corrected(data, cases, t_ref_d, v_ref_d)

    # 4d. Velocidad dinámica (Axial_vel − dδ/dt teórica)
    if ddelta_dt_ref is not None:
        t_ref_v, v_ref_v = ddelta_dt_ref
        # fig_vel_corr = plot_vel_corrected(data, cases, t_ref_v, v_ref_v)

    # 5. Señales corregidas para RMS y ajuste exponencial
    #    Se resta la rampa teórica a cada señal antes de calcular el RMS.
    import copy
    data_corr = copy.deepcopy(data)
    if deflection_ramp_ref is not None:
        t_rd, v_rd = deflection_ramp_ref
        for case_name in cases:
            sig = data_corr[case_name].get("Axial_disp")
            if sig is not None:
                sig["values"] = subtract_reference_force(
                    sig["times"], sig["values"], t_rd, v_rd
                )
                # pass
    if ddelta_dt_ref is not None:
        t_rv, v_rv = ddelta_dt_ref
        for case_name in cases:
            sig = data_corr[case_name].get("Axial_vel")
            if sig is not None:
                sig["values"] = subtract_reference_force(
                    sig["times"], sig["values"], t_rv, v_rv
                )

    # 5a-bis. dF = res_R_p - F_ramp  y  dF_norm = dF / ap(t)
    if force_ramp_ref is not None:
        t_rf, v_rf = force_ramp_ref
        _eps_ap = 1e-9   # evita división por cero
        for case_name in cases:
            sig_f = data_corr[case_name].get("res_R_p")
            if sig_f is None:
                continue
            t_sig = sig_f["times"]
            v_raw = sig_f["values"]
            if v_raw.ndim > 1:
                v_raw = v_raw[:, 0]
            dF = subtract_reference_force(t_sig, v_raw, t_rf, v_rf)
            # ap(t) vectorizado usando la fórmula de la rampa
            t_rel_arr = t_sig - _t_reach
            ap_arr = np.clip(
                _a0_m + (_a1_m - _a0_m) * t_rel_arr / _t_ramp,
                _a0_m, _a1_m,
            )
            dF_norm = dF / np.maximum(ap_arr, _eps_ap)
            data_corr[case_name]["dF"]      = {"times": t_sig, "values": dF}
            data_corr[case_name]["dF_norm"] = {"times": t_sig, "values": dF_norm}

    # 5b. Figuras RMS — señales corregidas
    case_attrs = read_case_attributes(h5_path, cases)
    fig_lnRMS_x      = plot_rms(data_corr, cases, "Axial_disp", window_size_time, case_attrs=case_attrs)
    fig_lnRMS_v      = plot_rms(data_corr, cases, "Axial_vel",  window_size_time, case_attrs=case_attrs)
    fig_lnRMS_dFnorm = plot_rms(data_corr, cases, "dF_norm",    window_size_time, case_attrs=case_attrs)



    # Precalcular RMS(dF) y RMS(dF_norm) en t_fit_start, en t_proxy_Ph y en t_eval_pre_F de cada caso
    _rms_at_tstart: dict = {cn: {"dF": float("nan"), "dF_norm": float("nan")} for cn in cases}
    # P_proxy: perturbación evaluada en t_proxy_Ph — instante elegido para el proxy de descomposición
    _rms_at_proxy:  dict = {cn: {"dF": float("nan"), "dF_norm": float("nan")} for cn in cases}
    _t_proxy_used:  dict = {}   # t_proxy efectivo por caso
    # P_pre: perturbación evaluada antes del chatter — proxy limpio del dexel
    _rms_at_pre:    dict = {cn: {"dF": float("nan"), "dF_norm": float("nan")} for cn in cases}
    _t_pre_used:    dict = {}   # t_pre efectivo por caso (para scatter)
    for cn in cases:
        ts = data[cn]["t_start"]
        # --- t_proxy_Ph ---
        if t_proxy_Ph is None:
            t_proxy = ts
        elif isinstance(t_proxy_Ph, dict):
            t_proxy = float(t_proxy_Ph.get(cn, ts))
        else:
            t_proxy = float(t_proxy_Ph)
        _t_proxy_used[cn] = t_proxy
        # --- t_eval_pre_F ---
        if t_eval_pre_F is None:
            t_pre = ts
        elif isinstance(t_eval_pre_F, dict):
            t_pre = float(t_eval_pre_F.get(cn, ts))
        else:
            t_pre = float(t_eval_pre_F)
        _t_pre_used[cn] = t_pre
        for _sig_key in ("dF", "dF_norm"):
            _s = data_corr[cn].get(_sig_key)
            if _s is None:
                continue
            try:
                _t_r, _rms_r = moving_rms(_s["times"], _s["values"], window_size_time)
                _rms_at_tstart[cn][_sig_key] = float(np.interp(ts,       _t_r, _rms_r))
                _rms_at_proxy[cn][_sig_key]  = float(np.interp(t_proxy,  _t_r, _rms_r))
                _rms_at_pre[cn][_sig_key]    = float(np.interp(t_pre,    _t_r, _rms_r))
            except (ValueError, Exception):
                pass

    # 5b-bis. Figuras de visualización  dF  y  dF_norm  — señal completa, sin recorte
    for _sig_vis, _ylabel_vis in (
        ("dF",      "ΔF  [N]"),
        ("dF_norm", "ΔF / ap(t)  [N/m]"),
    ):
        _fig_vis, _ax_vis = plt.subplots(figsize=(11, 4))
        _fig_vis.suptitle(f"{_sig_vis} — RMS móvil (señal completa)")
        _colors_vis = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for _idx_vis, _cn in enumerate(cases):
            _col = _colors_vis[_idx_vis % len(_colors_vis)]
            _s = data_corr[_cn].get(_sig_vis)
            if _s is None:
                continue
            try:
                _t_r, _rms_r = moving_rms(_s["times"], _s["values"], window_size_time)
                _ax_vis.plot(_t_r, _rms_r, linewidth=0.9, color=_col, label=_cn)
                _ts  = data[_cn]["t_start"]
                # Scatter en t_fit_start (circulo)
                _val = _rms_at_tstart[_cn][_sig_vis]
                if np.isfinite(_val) and _val > 0:
                    _ax_vis.scatter([_ts], [_val], color=_col, s=60, zorder=5,
                                    marker="o", edgecolors="white", linewidths=0.8)
                # Scatter en t_proxy_Ph (estrella)
                _t_proxy = _t_proxy_used.get(_cn, _ts)
                _val_proxy = _rms_at_proxy[_cn][_sig_vis]
                if np.isfinite(_val_proxy) and _val_proxy > 0 and _t_proxy != _ts:
                    _ax_vis.scatter([_t_proxy], [_val_proxy], color=_col, s=90, zorder=5,
                                    marker="*", edgecolors="white", linewidths=0.5)
                # Scatter en t_pre (diamante)
                _val_pre = _rms_at_pre[_cn][_sig_vis]
                _t_pre   = _t_pre_used.get(_cn, _ts)
                if np.isfinite(_val_pre) and _val_pre > 0 and _t_pre != _ts:
                    _ax_vis.scatter([_t_pre], [_val_pre], color=_col, s=70, zorder=5,
                                    marker="D", edgecolors="white", linewidths=0.8)
            except ValueError:
                pass
        _ax_vis.set_yscale("log")
        _ax_vis.set_xlabel("Tiempo [s]")
        _ax_vis.set_ylabel(f"RMS {_ylabel_vis}")
        _ax_vis.legend(fontsize=7, ncol=2)
        _ax_vis.grid(True, which="both", linestyle="--", alpha=0.4)

    # 5c. Leer atributos de los casos (dxl_size, etc.)
    case_attrs = read_case_attributes(h5_path, cases)

    # 6. Ajuste exponencial + Delta_t — señales corregidas
    #    case_results[case_name][signal] = dict con toda la info del fitting
    case_results: dict = {cn: {} for cn in cases}
    _plot_queue:   dict = {}   # {signal: dict(fits, delay_result, ln_A_ref_val)}

    for signal in ("Axial_disp", "Axial_vel"):
        fits_signal: dict = {}
        print(f"\n{'='*60}\n  AJUSTE EXPONENCIAL (corregido): {signal}\n{'='*60}")
        # ── Pasada 1: ajuste exponencial de cada caso ─────────────────────────
        _case_meta: dict = {}   # almacena ts, te, ap_start, ap_end, dxl_size por caso
        for case_name in cases:
            sig = data_corr[case_name].get(signal)
            if sig is None:
                continue
            ts = data[case_name]["t_start"]
            te = data[case_name]["t_end"]
            try:
                t_cut, v_cut = extract_time_window(sig["times"], sig["values"], ts, te)
                t_rms, rms_v = moving_rms(t_cut, v_cut, window_size_time)
                fit = fit_log_rms(t_rms, rms_v, case_name=case_name)
                fits_signal[case_name] = fit
                _case_meta[case_name] = {
                    "ts": ts, "te": te,
                    "ap_start": _compute_ap_at(ts),
                    "ap_end":   _compute_ap_at(te),
                    "dxl_size": case_attrs.get(case_name, {}).get("$dxl_size$", float("nan")),
                }
                if fit:
                    print(
                        f"  {case_name}: beta={fit.beta:.4f}/s  "
                        f"y0={fit.y0:.3e}  c={fit.c_ln:.4f}  R²={fit.r2:.4f}  "
                        f"ap=[{_case_meta[case_name]['ap_start']*1e3:.2f},"
                        f"{_case_meta[case_name]['ap_end']*1e3:.2f}]mm  "
                        f"dxl={_case_meta[case_name]['dxl_size']}"
                    )
            except ValueError as exc:
                print(f"  [AVISO] {case_name}/{signal}: {exc}")
                fits_signal[case_name] = None

        # ── Auto-cálculo de ln_A_ref si no está fijado manualmente ────────────
        # Centro geométrico (en escala log) entre el mínimo y el máximo de todos
        # los valores del ajuste evaluados en los extremos del intervalo de cada caso.
        _ln_ref = ln_A_ref.get(signal)
        if _ln_ref is None:
            _ln_all = []
            for case_name, fit in fits_signal.items():
                if fit is None or case_name not in _case_meta:
                    continue
                _ts_c = _case_meta[case_name]["ts"]
                _te_c = _case_meta[case_name]["te"]
                _ln_all.append(fit.c_ln + fit.beta * _ts_c)
                _ln_all.append(fit.c_ln + fit.beta * _te_c)
            if _ln_all:
                _ln_ref = (min(_ln_all) + max(_ln_all)) / 2.0
                print(
                    f"  [auto ln_A_ref] {signal}: "
                    f"min={min(_ln_all):.3f}  max={max(_ln_all):.3f}  "
                    f"centro={_ln_ref:.3f}"
                )

        # ── Pasada 2: t_at_Aref_fit y ap_at_Aref_fit con ln_ref ya conocido ──────────
        for case_name in cases:
            fit = fits_signal.get(case_name)
            meta = _case_meta.get(case_name)
            if meta is None:
                case_results[case_name][signal] = None
                continue
            t_at_Aref_fit  = float("nan")
            ap_at_Aref_fit = float("nan")
            if fit and _ln_ref is not None:
                t_at_Aref_fit  = (_ln_ref - fit.c_ln) / fit.beta
                ap_at_Aref_fit = _compute_ap_at(t_at_Aref_fit)

            # ── Validación: cruce real del RMS con A_ref ──────────────────────
            # Busca el primer instante donde ln(RMS) >= ln_A_ref usando la señal
            # real (no el fit).  Permite comparar con t_at_Aref_fit del fitting.
            t_cross_rms  = float("nan")
            ap_cross_rms = float("nan")
            dt_cross      = float("nan")
            if _ln_ref is not None:
                sig = data_corr[case_name].get(signal)
                if sig is not None:
                    try:
                        t_cut, v_cut = extract_time_window(
                            sig["times"], sig["values"], meta["ts"], meta["te"]
                        )
                        _, rms_v_val = moving_rms(t_cut, v_cut, window_size_time)
                        ln_rms_val   = np.where(rms_v_val > 0,
                                                np.log(rms_v_val), -np.inf)
                        idx_cross    = np.argmax(ln_rms_val >= _ln_ref)
                        if ln_rms_val[idx_cross] >= _ln_ref:
                            t_cross_rms = float(t_cut[idx_cross])
                            dt_cross    = t_cross_rms - t_at_Aref_fit

                            ap_cross_rms = _compute_ap_at(t_cross_rms)
                    except ValueError:
                        pass
            if not np.isnan(t_cross_rms):
                print(
                    f"  [validación A_ref] {case_name}/{signal}: "
                    f"t_fit={t_at_Aref_fit:.4f}s  t_rms={t_cross_rms:.4f}s  "
                    f"Δt={dt_cross:+.4f}s  "
                    f"({'OK' if abs(dt_cross) < 0.05 else 'DIVERGE'})"
                )
            else:
                print(
                    f"  [validación A_ref] {case_name}/{signal}: "
                    f"t_fit={t_at_Aref_fit:.4f}s  t_rms=N/A (RMS no alcanzó A_ref)"
                )

            case_results[case_name][signal] = {
                "t_fit_start":  meta["ts"],
                "t_fit_end":    meta["te"],
                "ap_start":     meta["ap_start"],
                "ap_end":       meta["ap_end"],
                "dxl_size":     meta["dxl_size"],
                "beta":         fit.beta  if fit else float("nan"),
                "c_ln":         fit.c_ln  if fit else float("nan"),
                "y0":           fit.y0    if fit else float("nan"),
                "r2":           fit.r2    if fit else float("nan"),
                "ln_A_ref":     _ln_ref,
                "t_at_Aref_fit":    t_at_Aref_fit,
                "ap_at_Aref_fit":   ap_at_Aref_fit,
                "t_cross_rms":  t_cross_rms,
                "dt_cross":     dt_cross,
                "ap_cross_rms": ap_cross_rms,
            }
        valid_fits = [f for f in fits_signal.values() if f is not None]
        delay_res  = None
        if len(valid_fits) == 2:
            delay_res = compute_delay(valid_fits[0], valid_fits[1])
            print_delay_summary(delay_res)
        elif len(valid_fits) > 2:
            print("  (más de 2 casos: usa compute_delay(fit_i, fit_j) para pares)")
        # ── Guardar para graficar al final ───────────────────────────────────
        _plot_queue[signal] = dict(
            fits=fits_signal, delay_result=delay_res, ln_A_ref_val=_ln_ref
        )


    # ── Análisis por pares: Δt(A_ref) y Δap(A_ref) ───────────────────────────
    # Para cada señal, calcula el retraso y la diferencia de ap entre todos los
    # pares de casos que tienen un ajuste válido y un t_at_Aref_fit finito.
    pair_results: dict = {}   # pair_results[signal][(cn_i, cn_j)] = dict
    for signal in ("Axial_disp", "Axial_vel"):
        pair_results[signal] = {}
        valid_cases = [
            cn for cn in cases
            if case_results[cn].get(signal) is not None
            and np.isfinite(case_results[cn][signal].get("t_at_Aref_fit", float("nan")))
        ]
        for i in range(len(valid_cases)):
            for j in range(i + 1, len(valid_cases)):
                cn_i, cn_j = valid_cases[i], valid_cases[j]
                ri = case_results[cn_i][signal]
                rj = case_results[cn_j][signal]
                # Convención: i=caso 1 (referencia), j=caso 2
                # Δt = t_j − t_i  (igual que la demostración: 1/β₂·ln(Aref/P₂) − 1/β₁·ln(Aref/P₁))
                # delta_t_Aref  = rj["t_at_Aref_fit"]  - ri["t_at_Aref_fit"]
                # delta_ap_Aref = rj["ap_at_Aref_fit"] - ri["ap_at_Aref_fit"]
                delta_t_Aref  = rj["t_cross_rms"]  - ri["t_cross_rms"]
                delta_ap_Aref = rj["ap_cross_rms"] - ri["ap_cross_rms"]

                beta_common   = 0.5 * (ri["beta"] + rj["beta"])
                # Valor del ajuste al inicio del intervalo de fitting de cada caso
                # R_start = exp(c + beta * t_start)  — evita extrapolar hasta t=0
                extrapolate_to_start = False
                if extrapolate_to_start:
                    ln_R_start_i = ri["c_ln"] + ri["beta"] * 0.0
                    ln_R_start_j = rj["c_ln"] + rj["beta"] * 0.0
                else:
                    ln_R_start_i = ri["c_ln"] + ri["beta"] * ri["t_fit_start"]
                    ln_R_start_j = rj["c_ln"] + rj["beta"] * rj["t_fit_start"]

                delta_t_fit  = (
                    (ln_R_start_j - ln_R_start_i) / beta_common
                    if beta_common != 0
                    else float("nan")
                )
                # P_F  = RMS(dF)      evaluado en t_fit_start de cada caso
                # P_h  = RMS(dF_norm) evaluado en t_fit_start de cada caso
                # P_F_pre / P_h_pre   evaluado en t_eval_pre_F (antes del chatter)
                P_F_i     = _rms_at_tstart[cn_i]["dF"]
                P_F_j     = _rms_at_tstart[cn_j]["dF"]
                P_h_i     = _rms_at_proxy[cn_i]["dF_norm"]
                P_h_j     = _rms_at_proxy[cn_j]["dF_norm"]
                P_F_pre_i = _rms_at_pre[cn_i]["dF"]
                P_F_pre_j = _rms_at_pre[cn_j]["dF"]
                P_h_pre_i = _rms_at_pre[cn_i]["dF_norm"]
                P_h_pre_j = _rms_at_pre[cn_j]["dF_norm"]
                # Δt_force = (1/β_com) * ln(P_j/P_i)  — convención j−i
                delta_t_force = (
                    (1.0 / beta_common) * np.log(P_F_i / P_F_j)
                    if beta_common != 0 and P_F_i > 0 and P_F_j > 0
                    else float("nan")
                )
                delta_t_force_h = (
                    (1.0 / beta_common) * np.log(P_h_i / P_h_j)
                    if beta_common != 0 and P_h_i > 0 and P_h_j > 0
                    else float("nan")
                )
                delta_t_force_pre = (
                    (1.0 / beta_common) * np.log(P_F_pre_i / P_F_pre_j)
                    if beta_common != 0 and P_F_pre_i > 0 and P_F_pre_j > 0
                    else float("nan")
                )
                delta_t_force_h_pre = (
                    (1.0 / beta_common) * np.log(P_h_pre_i / P_h_pre_j)
                    if beta_common != 0 and P_h_pre_i > 0 and P_h_pre_j > 0
                    else float("nan")
                )
                # ── Δt descompuesto: perturbación inicial + contribución de betas ──
                # Convención demostración (1=i, 2=j):
                #   Δt = 1/β_j·ln(Aref/P_j) − 1/β_i·ln(Aref/P_i)
                #   dt_P    = (1/β_j) · ln(P_i/P_j)  =  (c_i − c_j) / β_j
                #   dt_beta = (1/β_j − 1/β_i) · ln(Aref/P_i)  =  (1/β_j − 1/β_i)·(ln_Aref − c_i)
                #   dt_decomp = dt_P + dt_beta  =  t_j − t_i  = +delta_t_Aref  ✓
                _ln_ref_pair = ri.get("ln_A_ref")   # mismo para i y j (calculado por señal)
                _beta_i = ri["beta"]
                _beta_j = rj["beta"]
                _c_i    = ri["c_ln"]
                _c_j    = rj["c_ln"]
                _ts_i   = ri["t_fit_start"]
                _ts_j   = rj["t_fit_start"]
                if _beta_i != 0 and _beta_j != 0 and _ln_ref_pair is not None:
                    dt_P    = (_c_i - _c_j) / _beta_j
                    dt_beta = (_ln_ref_pair - _c_i) * (1.0 / _beta_j - 1.0 / _beta_i)
                    dt_decomp = dt_P + dt_beta
                else:
                    dt_P = dt_beta = dt_decomp = float("nan")

                # dt_force_P: proxy de dt_P = (1/β_j)·ln(P_i/P_j) usando fuerzas en t_start.
                # Aproximación: c_k ≈ ln(P_F_k) − β_k·t_start_k  →
                #   dt_P ≈ (1/β_j)·ln(P_F_i/P_F_j) − (β_i/β_j)·t_start_i + t_start_j
                dt_force_P = (
                    (1.0 / _beta_j) * np.log(P_F_i / P_F_j)
                    - (_beta_i / _beta_j) * _ts_i + _ts_j
                    if _beta_j != 0 and P_F_i > 0 and P_F_j > 0
                    else float("nan")
                )

                # dt_beta_P: proxy de dt_beta usando P_F_i en lugar de c_i.
                # Corrección: c_i ≈ ln(P_F_i) − β_i·t_start_i  →  ln(Aref/P_i) ≈ ln_ref − ln(P_F_i) + β_i·t_start_i
                dt_beta_P = (_ln_ref_pair - np.log(P_F_i) + _beta_i * _ts_i) * (1.0 / _beta_j - 1.0 / _beta_i)
                # dt_decomp2: descomposición híbrida  force_P + beta
                dt_decomp2 = (
                    dt_force_P + dt_beta_P
                    if np.isfinite(dt_force_P) and np.isfinite(dt_beta_P)
                    else float("nan")
                )

                # --- Variantes con dF_norm (P_h) — ap cancelado + corrección ln(Kt) + C_i ---
                # P_h = RMS(dF/ap)  ≈  Kt·C_i·RMS(x)  ⇒  ln(P_h) - β·ts = c + ln(Kt) + ln(C_i)
                # C_i = sqrt(1 + exp(-2β_i·T) - 2·exp(-β_i·T)·cos(ω·T))  [analítico]
                _ln_Kt = np.log(_Kf_fisico)  # ln(Kf)

                def _Ci(beta: float) -> float:
                    """Factor C_i analítico dado el taux de croissance β."""
                    e = np.exp(-beta * _T_delay)
                    return float(np.sqrt(max(1.0 + e**2 - 2.0 * e * np.cos(_omega_chatter * _T_delay), 0.0)))

                _Ci_val = _Ci(_beta_i)
                _Cj_val = _Ci(_beta_j)
                _ln_Ci  = np.log(_Ci_val) if _Ci_val > 0 else 0.0
                _ln_Cj  = np.log(_Cj_val) if _Cj_val > 0 else 0.0

                # Tiempos de evaluación del proxy P_h (independientes del intervalo de fit)
                _tp_i = float(_t_proxy_used.get(cn_i, _ts_i))
                _tp_j = float(_t_proxy_used.get(cn_j, _ts_j))

                # dt_force_h_P: incluye ratio C_j/C_i exacto (antes asumía C_i≈C_j)
                dt_force_h_P = (
                    (1.0 / _beta_j) * (np.log(P_h_i / P_h_j) + _ln_Cj - _ln_Ci)
                    - (_beta_i / _beta_j) * _tp_i + _tp_j
                    if _beta_j != 0 and P_h_i > 0 and P_h_j > 0
                    else float("nan")
                )
                # dt_beta_h_P: usa P_h_i + corrección ln(Kt) + ln(C_i) → exacto
                # c_i = ln(P_h_i) − β_i·tp_i − ln(Kt) − ln(C_i)
                # ⇒  ln_Aref − c_i = ln_Aref − ln(P_h_i) + β_i·tp_i + ln(Kt) + ln(C_i)
                dt_beta_h_P = (
                    (_ln_ref_pair - np.log(P_h_i) + _beta_i * _tp_i + _ln_Kt + _ln_Ci)
                    * (1.0 / _beta_j - 1.0 / _beta_i)
                    if P_h_i > 0
                    else float("nan")
                )
                # dt_decomp3: descomposición híbrida  force_h_P + beta_h
                dt_decomp3 = (
                    dt_force_h_P + dt_beta_h_P
                    if np.isfinite(dt_force_h_P) and np.isfinite(dt_beta_h_P)
                    else float("nan")
                )

                # delta_t_force_h_C: hipótesis β_i=β_j=β_com + corrección C_j/C_i
                # = (1/β_com) · (ln(P_h_i/P_h_j) + ln(C_j) − ln(C_i)) + tp_j − (1-β_i/β_j)·tp_i
                delta_t_force_h_C = (
                    (1.0 / beta_common) * (np.log(P_h_i / P_h_j) + _ln_Cj - _ln_Ci)
                    - (_beta_i / beta_common) * _tp_i + _tp_j
                    if beta_common != 0 and P_h_i > 0 and P_h_j > 0
                    else float("nan")
                )

                pair_results[signal][(cn_i, cn_j)] = {
                    "ln_A_ref":        _ln_ref_pair,
                    "t_i":             ri["t_at_Aref_fit"],
                    "t_j":             rj["t_at_Aref_fit"],
                    "delta_t_Aref":    delta_t_Aref,
                    "ap_i":            ri["ap_at_Aref_fit"],
                    "ap_j":            rj["ap_at_Aref_fit"],
                    "delta_ap_Aref":   delta_ap_Aref,
                    "beta_common":     beta_common,
                    "delta_t_fit":     delta_t_fit,
                    "dt_P":            dt_P,
                    "dt_beta":         dt_beta,
                    "dt_beta_P":       dt_beta_P,
                    "dt_decomp":       dt_decomp,
                    "dt_force_P":      dt_force_P,
                    "dt_decomp2":      dt_decomp2,
                    "dt_force_h_P":    dt_force_h_P,
                    "dt_beta_h_P":     dt_beta_h_P,
                    "dt_decomp3":      dt_decomp3,
                    "C_i":             _Ci_val,
                    "C_j":             _Cj_val,
                    "ln_Ci":           _ln_Ci,
                    "ln_Cj":           _ln_Cj,
                    "P_F_i":               P_F_i,
                    "P_F_j":               P_F_j,
                    "P_h_i":               P_h_i,
                    "P_h_j":               P_h_j,
                    "delta_t_force":       delta_t_force,
                    "delta_t_force_h":     delta_t_force_h,
                    "delta_t_force_h_C":   delta_t_force_h_C,
                    "P_F_pre_i":           P_F_pre_i,
                    "P_F_pre_j":           P_F_pre_j,
                    "P_h_pre_i":           P_h_pre_i,
                    "P_h_pre_j":           P_h_pre_j,
                    "delta_t_force_pre":   delta_t_force_pre,
                    "delta_t_force_h_pre": delta_t_force_h_pre,
                    "t_pre_i":             _t_pre_used.get(cn_i),
                    "t_pre_j":             _t_pre_used.get(cn_j),
                    "t_proxy_i":           _t_proxy_used.get(cn_i),
                    "t_proxy_j":           _t_proxy_used.get(cn_j),
                }

    print("\n" + "="*60)
    print("  RESUMEN case_results")
    print("="*60)
    for cn, signals in case_results.items():
        print(f"  {cn}:")
        for sig_name, res in signals.items():
            if res is None:
                print(f"    {sig_name}: [sin ajuste]")
            else:
                aref_str = ""
                if res.get("ln_A_ref") is not None and np.isfinite(res.get("t_at_Aref_fit", float("nan"))):
                    t_cross = res.get("t_cross_rms", float("nan"))
                    dt_cross = res.get("dt_cross", float("nan"))
                    cross_str = (
                        f"  t_rms={t_cross:.4f}s  Δ={dt_cross:+.4f}s"
                        if np.isfinite(t_cross) else "  t_rms=N/A"
                    )
                    aref_str = (
                        f"  |  ln(A_ref)={res['ln_A_ref']:.3f}  "
                        f"t_fit={res['t_at_Aref_fit']:.4f}s"
                        + cross_str +
                        f"  ap(A_ref)={res['ap_at_Aref_fit']*1e3:.3f}mm"
                    )
                print(
                    f"    {sig_name}: t=[{res['t_fit_start']:.3f},{res['t_fit_end']:.3f}]s  "
                    f"ap=[{res['ap_start']*1e3:.2f},{res['ap_end']*1e3:.2f}]mm  "
                    f"dxl={res['dxl_size']}  "
                    f"beta={res['beta']:.4f}  y0={res['y0']:.3e}  R²={res['r2']:.4f}"
                    + aref_str
                )

    # ── Validación: c_i ≈ ln(P_h_i) − β_i·ts_i − ln(Kt)  →  eps_i ≈ 0 ──────────────
    print(f"\n{'='*64}")
    print("  VALIDACION  eps_i = ln(P_h_i) - beta_i*ts_i - ln(Kt) - c_i")
    print(f"  ln(Kt) = ln({_Kf_fisico:.2e}) = {np.log(_Kf_fisico):.4f}")
    print(f"{'='*64}")
    for _sig_val in ("Axial_disp", "Axial_vel"):
        print(f"  [{_sig_val}]")
        for cn in cases:
            res = case_results[cn].get(_sig_val)
            if res is None:
                print(f"    {cn}: [sin ajuste]")
                continue
            _c_val  = res["c_ln"]
            _b_val  = res["beta"]
            _ts_val = res["t_fit_start"]
            _Ph_val = _rms_at_tstart[cn]["dF_norm"]
            if _Ph_val > 0:
                _aprox_val = np.log(_Ph_val) - _b_val * _ts_val - np.log(_Kf_fisico)
                _eps_val   = _aprox_val - _c_val
                print(f"    {cn}:  eps_i = {_eps_val:+.6f}   "
                      f"(c_i={_c_val:.4f}  aprox={_aprox_val:.4f})")
            else:
                print(f"    {cn}: P_h_i no disponible")

    if pair_results:
        # for signal in ("Axial_disp", "Axial_vel"):
        for signal in ("Axial_disp",):
            if not pair_results.get(signal):
                continue
            print(f"\n{'='*64}")
            print(f"  PARES  [{signal}]")
            print(f"{'='*64}")
            for (cn_i, cn_j), pr in pair_results[signal].items():
                print(f"\n  {cn_i}  ←→  {cn_j}   (β_com = {pr['beta_common']:.8f} /s)")
                print(f"  {'─'*56}")
                print(f"  {'Δap  [mm]':<28} {pr['delta_ap_Aref']*1e3:+.8f}"
                      f"   (ap_i={pr['ap_i']*1e3:.3f}  ap_j={pr['ap_j']*1e3:.8f})")
                print(f"  {'─'*56}")
                print(f"  {'Δt(A_ref)  [s]':<28} {pr['delta_t_Aref']:+.8f}")
                print(f"  {'Δt_fit     [s]':<28} {pr['delta_t_fit']:+.8f}")
                print(f"  {'Δt_force_h_C [s] (+C)':<28} {pr['delta_t_force_h_C']:+.8f}")

                print(f"  {'─'*56}")
                print(f"  {'Δt_P         [s]  (fit)':<28} {pr['dt_P']:+.8f}")
                print(f"  {'Δt_force_P   [s]  (proxy)':<28} {pr['dt_force_P']:+.8f}")
                print(f"  {'Δt_beta      [s]  (betas)':<28} {pr['dt_beta']:+.8f}")
                print(f"  {'Δt_beta_P    [s]  (beta_P)':<28} {pr['dt_beta_P']:+.8f}")
                print(f"  {'Δt_decomp    [s]  (P+beta)':<28} {pr['dt_decomp']:+.8f}")
                print(f"  {'Δt_decomp2   [s]  (fP+beta)':<28} {pr['dt_decomp2']:+.8f}")
                print(f"  {'─'*56}")
                print(f"  {'Δt_force_h_P [s]  (h-proxy)':<28} {pr['dt_force_h_P']:+.8f}")
                print(f"  {'Δt_beta_h_P  [s]  (h-beta)':<28} {pr['dt_beta_h_P']:+.8f}")
                print(f"  {'Δt_decomp3   [s]  (hP+hbeta)':<28} {pr['dt_decomp3']:+.8f}")
                _tpi = pr.get("t_proxy_i"); _tpj = pr.get("t_proxy_j")
                _proxy_lbl = (
                    f"t_proxy={_tpi:.3f}s" if _tpi == _tpj
                    else f"i={_tpi:.3f}s / j={_tpj:.3f}s"
                ) if _tpi is not None else "t_proxy=t_start"
                print(f"  {'  ↑ t_proxy':<28} {_proxy_lbl}"
                      f"   P_h:  {pr['P_h_i']:.3e}  →  {pr['P_h_j']:.3e}")
                print(f"  {'─'*56}")
                print(f"  {'C_i  (analítico)':<28} {pr['C_i']:.8f}   ln(C_i)={pr['ln_Ci']:+.6f}")
                print(f"  {'C_j  (analítico)':<28} {pr['C_j']:.8f}   ln(C_j)={pr['ln_Cj']:+.6f}")
                print(f"  {'─'*56}")
                print(f"  {'Δt_force   [s]':<28} {pr['delta_t_force']:+.8f}"
                      f"   P_F:  {pr['P_F_i']:.3e}  →  {pr['P_F_j']:.3e}")
                print(f"  {'Δt_force_h [s]':<28} {pr['delta_t_force_h']:+.8f}"
                      f"   P_h:  {pr['P_h_i']:.3e}  →  {pr['P_h_j']:.3e}")
                _ti = pr.get("t_pre_i"); _tj = pr.get("t_pre_j")
                _pre_lbl = (
                    f"t_pre={_ti:.3f}s" if _ti == _tj
                    else f"i={_ti:.3f}s / j={_tj:.3f}s"
                ) if _ti is not None else "t_pre=t_start"
                print(f"  {'Δt_force_pre [s]':<28} {pr['delta_t_force_pre']:+.8f}"
                      f"   P_F:  {pr['P_F_pre_i']:.3e}  →  {pr['P_F_pre_j']:.3e}")
                print(f"  {'Δt_force_h_pre [s]':<28} {pr['delta_t_force_h_pre']:+.8f}"
                      f"   P_h:  {pr['P_h_pre_i']:.3e}  →  {pr['P_h_pre_j']:.3e}")

    _img_dir = Path(__file__).parent / "Latex" / "Etapa4_1page" / "Images"
    _img_dir.mkdir(parents=True, exist_ok=True)
    
    # 7. (Opcional) señal + RMS por caso — corregidas
    if plot_signal_with_rms:
        figs_disp = plot_signal_and_rms(data_corr, cases, "Axial_disp", window_size_time)
        figs_vel  = plot_signal_and_rms(data_corr, cases, "Axial_vel",  window_size_time)

    # ── Figuras ln(RMS) + ajuste — todas las señales, al final ──────────────
    for _sig, _kw in _plot_queue.items():
        _fig_fits = plot_rms_with_fits(
            data_corr, cases, _sig, window_size_time,
            case_attrs=case_attrs, **_kw
        )
        _fig_fits.savefig(_img_dir / f"lnRMS_fits_{_sig}.jpg", dpi=300, bbox_inches="tight")
        _fig_fits.savefig(_img_dir / f"lnRMS_fits_{_sig}.pdf", dpi=300, bbox_inches="tight")

        _fig_fits_clean = plot_rms_fits_only(
            data_corr, cases, _sig, window_size_time,
            fits=_kw.get("fits"), case_attrs=case_attrs,
        )
        _fig_fits_clean.savefig(_img_dir / f"lnRMS_fits_clean_{_sig}.jpg", dpi=300, bbox_inches="tight")
        _fig_fits_clean.savefig(_img_dir / f"lnRMS_fits_clean_{_sig}.pdf", dpi=300, bbox_inches="tight")

    # ── Figura estabilidad tiempo-congelado + betas de simulación ────────────
    _fig_sb = plot_stability_with_betas(
        case_results, cases, "Axial_disp", case_attrs=case_attrs
    )
    _fig_sb.savefig(_img_dir / "stability_betas_Axial_disp.pdf", dpi=300, bbox_inches="tight")
    _fig_sb.savefig(_img_dir / "stability_betas_Axial_disp.jpg", dpi=300, bbox_inches="tight")

    # ── Figura barras Δt — descomposición por par ────────────────────────────
    _fig_bars = plot_delta_t_bars(pair_results, "Axial_disp", case_attrs=case_attrs)
    _fig_bars.savefig(_img_dir / "delta_t_bars_Axial_disp.pdf", dpi=300, bbox_inches="tight")
    _fig_bars.savefig(_img_dir / "delta_t_bars_Axial_disp.jpg", dpi=300, bbox_inches="tight")

    # 8. Figuras DOE: t(A_ref), ap(A_ref), beta  vs  dxl_size
    _signal_colors = {
        "Axial_disp": "steelblue",
        "Axial_vel":  "tomato",
        "dF":         "seagreen",
        "dF_norm":    "darkorange",
    }

    fig_t_dxl,  ax_t_dxl  = plt.subplots(figsize=(7, 4))
    fig_ap_dxl, ax_ap_dxl = plt.subplots(figsize=(7, 4))
    fig_b_dxl,  ax_b_dxl  = plt.subplots(figsize=(7, 4))
    fig_b_ap,   ax_b_ap   = plt.subplots(figsize=(7, 4))

    ax_t_dxl.set_title("t(A_ref)  vs  dxl_size")
    ax_t_dxl.set_xlabel("dxl_size  [m]")
    # ax_t_dxl.set_xscale("log")
    ax_t_dxl.set_ylabel("t(A_ref)  [s]")

    ax_ap_dxl.set_title("ap(A_ref)  vs  dxl_size")
    ax_ap_dxl.set_xlabel("dxl_size  [m]")
    # ax_ap_dxl.set_xscale("log")
    ax_ap_dxl.set_ylabel("ap(A_ref)  [mm]")

    ax_b_dxl.set_title("β  vs  dxl_size")
    ax_b_dxl.set_xlabel("dxl_size  [m]")
    ax_b_dxl.set_ylabel("β  [1/s]")
    # ax_b_dxl.set_xscale("log")

    ax_b_ap.set_title("β  vs  ap(A_ref)")
    ax_b_ap.set_xlabel("ap(A_ref)  [mm]")
    ax_b_ap.set_ylabel("β  [1/s]")

    fig_b_t,    ax_b_t    = plt.subplots(figsize=(7, 4))
    ax_b_t.set_title("β  vs  t(A_ref)")
    ax_b_t.set_xlabel("t(A_ref)  [s]")
    ax_b_t.set_ylabel("β  [1/s]")

    for signal in ("Axial_disp", "Axial_vel"):
        color = _signal_colors[signal]
        dxl_list, t_list, ap_list, b_list, ap_b_list, b_b_list, t_b_list, b_bt_list = [], [], [], [], [], [], [], []
        for cn in cases:
            res = case_results[cn].get(signal)
            if res is None:
                continue
            dxl   = res["dxl_size"]
            beta  = res["beta"]
            t_ar  = res["t_at_Aref_fit"]
            ap_ar = res["ap_at_Aref_fit"]
            if np.isfinite(dxl) and np.isfinite(t_ar):
                dxl_list.append(dxl);  t_list.append(t_ar)
            if np.isfinite(dxl) and np.isfinite(ap_ar):
                ap_list.append((dxl, ap_ar * 1e3))
            if np.isfinite(dxl) and np.isfinite(beta):
                b_list.append((dxl, beta))
            if np.isfinite(ap_ar) and np.isfinite(beta):
                ap_b_list.append(ap_ar * 1e3);  b_b_list.append(beta)
            if np.isfinite(t_ar) and np.isfinite(beta):
                t_b_list.append(t_ar);  b_bt_list.append(beta)

        if dxl_list:
            order = np.argsort(dxl_list)
            ax_t_dxl.plot(
                np.array(dxl_list)[order], np.array(t_list)[order],
                color=color, marker="o", linewidth=1.5, markersize=5, label=signal,
            )
        if ap_list:
            ap_list.sort(key=lambda x: x[0])
            ax_ap_dxl.plot(
                [p[0] for p in ap_list], [p[1] for p in ap_list],
                color=color, marker="o", linewidth=1.5, markersize=5, label=signal,
            )
        if b_list:
            b_list.sort(key=lambda x: x[0])
            ax_b_dxl.plot(
                [p[0] for p in b_list], [p[1] for p in b_list],
                color=color, marker="o", linewidth=1.5, markersize=5, label=signal,
            )
        if ap_b_list:
            order = np.argsort(ap_b_list)
            ax_b_ap.plot(
                np.array(ap_b_list)[order], np.array(b_b_list)[order],
                color=color, marker="o", linewidth=1.5, markersize=5, label=signal,
            )
        if t_b_list:
            order = np.argsort(t_b_list)
            ax_b_t.plot(
                np.array(t_b_list)[order], np.array(b_bt_list)[order],
                color=color, marker="o", linewidth=1.5, markersize=5, label=signal,
            )

    # Notación científica en el eje x donde aparece dxl_size
    for ax in (ax_t_dxl, ax_ap_dxl, ax_b_dxl):
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    for ax in (ax_t_dxl, ax_ap_dxl, ax_b_dxl, ax_b_ap, ax_b_t):
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    for fig in (fig_t_dxl, fig_ap_dxl, fig_b_dxl, fig_b_ap, fig_b_t):
        fig.tight_layout()

  
    fig_lnRMS_x.savefig(_img_dir / "lnRMS_Axial_disp.jpg", dpi=300, bbox_inches="tight")
    fig_lnRMS_v.savefig(_img_dir / "lnRMS_Axial_vel.jpg", dpi=300, bbox_inches="tight")
    fig_lnRMS_dFnorm.savefig(_img_dir / "lnRMS_dF_norm.jpg", dpi=300, bbox_inches="tight")

    fig_lnRMS_x.savefig(_img_dir / "lnRMS_Axial_disp.pdf", dpi=300, bbox_inches="tight")
    fig_lnRMS_v.savefig(_img_dir / "lnRMS_Axial_vel.pdf", dpi=300, bbox_inches="tight")
    fig_lnRMS_dFnorm.savefig(_img_dir / "lnRMS_dF_norm.pdf", dpi=300, bbox_inches="tight")

    plt.show()
