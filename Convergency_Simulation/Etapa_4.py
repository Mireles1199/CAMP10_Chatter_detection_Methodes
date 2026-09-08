#%%
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol, Tuple

from scipy.optimize import root
import h5py

# import plotly.tools as tls


# ─────────────────────────────────────────────────────────────
# Data classes — containers with useful derived properties
# ─────────────────────────────────────────────────────────────

@dataclass
class ModalParams:
    """Modal parameters of the 1-DOF system."""
    m: float   # modal mass [kg]
    c: float   # modal damping [N·s/m]
    k: float   # stiffness [N/m]
    omega_2: float


    @property
    def omega_n(self) -> float:
        """Natural frequency [rad/s]."""
        return float(np.sqrt(self.k / self.m))

    @property
    def zeta(self) -> float:
        """Damping ratio (dimensionless)."""
        return self.c / (2.0 * self.m * self.omega_n)
    
    @property
    def omega_d(self) -> float:
        """Damped natural frequency [rad/s]."""
        return self.omega_n * np.sqrt(1.0 - self.zeta**2)


    @classmethod
    def from_modal_freq(cls, f_hz: float, zeta: float, k: float) -> "ModalParams":
        """Builds ModalParams from frequency [Hz], zeta, and stiffness."""
        omega_n = 2.0 * np.pi * f_hz
        m = k / omega_n**2
        c = 2.0 * zeta * m * omega_n
        return cls(m=m, c=c, k=k, omega_2=omega_n**2)


@dataclass
class MachiningParams:
    """Machining operation parameters."""
    Kf: float      # cutting force coefficient [N/m²]
    T: float       # regenerative delay [s]
    a0: float      # initial depth of cut [m]
    a1: float      # final depth of cut [m]
    t_ramp: float  # ramp duration [s]
    dt: float      # frozen-time time step [s]

    @classmethod
    def from_rpm(
        cls,
        n_rpm: float,
        Nz: int,
        Kf: float,
        a0: float,
        a1: float,
        t_ramp: float,
        dt_factor: float = 5.0,
    ) -> "MachiningParams":
        """Computes T from n_rpm and Nz; dt = T / dt_factor."""
        T = 60.0 / (n_rpm * Nz)
        return cls(Kf=Kf, T=T, a0=a0, a1=a1, t_ramp=t_ramp, dt=T / dt_factor)


@dataclass
class StaticDeflection:
    """Static deflection calculator using the double projection used in Etapa 1."""
    case_ap: float
    modal: ModalParams
    f_tooth_mm: float = 0.05
    k_cut: float = 1.0e3
    alpha_deg: float = 90.0
    theta_deg: float = 135.0

    @staticmethod
    def compute_case_force(
        case_ap: float,
        f_tooth_mm: float,
        k_cut: float,
    ) -> float:
        """Compute cutting force from the case depth of cut."""
        return k_cut* (case_ap) * f_tooth_mm*1e-3

    def ap_of_t(self, t: np.ndarray, a0: float, a1: float, t_ramp: float) -> np.ndarray:
        """Depth of cut ramp a_p(t) from a0 to a1 over t_ramp.

        a_p(t) = a0 + (a1 - a0) * t / t_ramp, for 0 <= t <= t_ramp.
        """
        t_arr = np.asarray(t, dtype=float)
        if t_ramp <= 0:
            raise ValueError("t_ramp must be > 0")
        return np.where(t_arr <= t_ramp, a0 + (a1 - a0) * (t_arr / t_ramp), a1)

    def force_of_t(self, t: np.ndarray, a0: float, a1: float, t_ramp: float) -> np.ndarray:
        """Cutting force time series for the ramp.

        F(t) = k_cut * a_p(t) * f_tooth_mm, with a_p(t) in mm.
        """
        ap_t = self.ap_of_t(t, a0, a1, t_ramp)
        return self.compute_case_force(ap_t, self.f_tooth_mm, self.k_cut)

    def deflection_of_t(self, t: np.ndarray, a0: float, a1: float, t_ramp: float) -> np.ndarray:
        """Static deflection time series for the ramp.

        delta(t) = F(t) * cos^2(alpha - theta) / stiffness.
        """
        force_t = self.force_of_t(t, a0, a1, t_ramp)
        modal_force = force_t * np.cos(np.deg2rad(self.alpha_deg - self.theta_deg))
        modal_deflection = modal_force / (self.modal.k * self.modal.m**0.5)
        return np.cos(np.deg2rad(self.alpha_deg - self.theta_deg))**2 * modal_deflection

    @staticmethod
    def compute_deflection_from_force(
        force: float,
        modal: ModalParams,
        alpha_deg: float = 90.0,
        theta_deg: float = 135.0,
    ) -> float:
        """Return the static deflection in meters.

        """
        phi = 1/np.sqrt(modal.m)*np.sin(np.deg2rad(theta_deg))
        phi_2 = phi**2
        deflection = phi_2 * force / modal.omega_2
        return  deflection

    @staticmethod
    def compute_deflection(
        case_ap: float,
        modal: ModalParams,
        f_tooth_mm: float = 0.05,
        k_cut: float = 1.0e3,
        alpha_deg: float = 90.0,
        theta_deg: float = 135.0,
    ) -> float:
        """Return the static deflection from a case depth of cut.

        F = k_cut * case_ap[mm] * f_tooth_mm,
        delta = F * cos^2(alpha - theta) / stiffness.
        """
        force_case = StaticDeflection.compute_case_force(case_ap, f_tooth_mm, k_cut)
        return StaticDeflection.compute_deflection_from_force(
            force_case,
            modal,
            alpha_deg=alpha_deg,
            theta_deg=theta_deg,
        )

    @property
    def t_delta_est(self) -> float:
        t_delta_est = np.arccos(-self.modal.zeta) / self.modal.omega_d
        return t_delta_est


    @property
    def force(self) -> float:
        """Cutting force associated with this case."""
        return self.compute_case_force(self.case_ap, self.f_tooth_mm, self.k_cut)

    @property
    def static_deflection(self) -> float:
        """Static deflection associated with this case."""
        return self.compute_deflection_from_force(
            self.force,
            self.modal,
            alpha_deg=self.alpha_deg,
            theta_deg=self.theta_deg,
        )

    def compute(self) -> float:
        """Compute the static deflection using the stored case parameters."""
        return self.static_deflection


@dataclass
class StaticDeflectionRamp:
    """Static deflection time-series calculator for a linear ramp a0 -> a1."""
    a0: float
    a1: float
    t_ramp: float
    modal: ModalParams
    f_tooth_mm: float = 0.05
    k_cut: float = 1.0e3
    alpha_deg: float = 90.0
    theta_deg: float = 135.0

    def ap_of_t(self, t: np.ndarray) -> np.ndarray:
        """Depth of cut ramp a_p(t) from a0 to a1 over t_ramp."""
        t_arr = np.asarray(t, dtype=float)
        if self.t_ramp <= 0:
            raise ValueError("t_ramp must be > 0")
        # Linear ramp: a_p(t) grows from a0 to a1 and then stays at a1.
        return np.where(t_arr <= self.t_ramp, self.a0 + (self.a1 - self.a0) * (t_arr / self.t_ramp), self.a1)

    def dap_dt(self, t: np.ndarray) -> np.ndarray:
        """Time derivative of the depth of cut ramp.

        da_p/dt = (a1 - a0) / t_ramp for 0 <= t <= t_ramp, and 0 after the ramp.
        """
        t_arr = np.asarray(t, dtype=float)
        slope = (self.a1 - self.a0) / self.t_ramp
        return np.where(t_arr <= self.t_ramp, slope, 0.0)

    @staticmethod
    def case_force_from_ap(case_ap: np.ndarray | float, f_tooth_mm: float, k_cut: float) -> np.ndarray:
        """Cutting force from a scalar or array depth of cut."""
        return k_cut * (np.asarray(case_ap, dtype=float)) * f_tooth_mm * 1.e-3

    def dforce_dap(self) -> float:
        """Derivative of cutting force with respect to depth of cut.

        dF/da_p = k_cut * f_tooth_mm * 1e-3 because F(a_p) = k_cut * (a_p [mm]) * f_tooth_mm.
        """
        return self.k_cut * self.f_tooth_mm * 1e-3

    def ddelta_dF(self) -> float:
        """Derivative of static deflection with respect to force.

        ddelta/dF = phi^2 / omega_2, where phi = sin(theta) / sqrt(m) and omega_2 = k / m.
        """
        return self.phi**2 / self.modal.omega_2

    def force_of_t(self, t: np.ndarray) -> np.ndarray:
        """Cutting force time series for the ramp.

        F(t) = k_cut * a_p(t) * f_tooth_mm, with a_p(t) in mm.
        """
        return self.case_force_from_ap(self.ap_of_t(t), self.f_tooth_mm, self.k_cut)

    def dforce_dt(self, t: np.ndarray) -> np.ndarray:
        """Time derivative of the cutting force.

        dF/dt = (dF/da_p) * (da_p/dt).
        """
        return self.dforce_dap() * self.dap_dt(t)

    def static_deflection_of_t(self, t: np.ndarray) -> np.ndarray:
        """Static deflection time series for the ramp.

        delta(t) = F(t) * sin^2(theta) / (k * sqrt(m)).
        """
        force_t = self.force_of_t(t)
        phi = self.phi
        deflection = phi**2 * force_t / self.modal.omega_2
        return deflection

    def ddelta_dt(self, t: np.ndarray) -> np.ndarray:
        """Time derivative of the static deflection.

        ddelta/dt = (ddelta/dF) * (dF/dt).
        """
        return self.ddelta_dF() * self.dforce_dt(t)

    def dforce_da_p(self) -> float:
        """Derivative of cutting force with respect to depth of cut.

        Same slope as dF/da_p: constant in this linear cutting model.
        """
        return self.dforce_dap()

    def ddelta_da_p(self) -> float:
        """Derivative of static deflection with respect to depth of cut.

        ddelta/da_p = (ddelta/dF) * (dF/da_p).
        """
        return self.ddelta_dF() * self.dforce_da_p()
    
    @property
    def phi(self) -> float:
        """Modal participation factor for the given theta."""
        return 1 / np.sqrt(self.modal.m) * np.sin(np.deg2rad(self.theta_deg))

    @property
    def ap_t(self) -> Callable[[np.ndarray], np.ndarray]:
        return self.ap_of_t

    @property
    def dap_dt_t(self) -> Callable[[np.ndarray], np.ndarray]:
        return self.dap_dt

    @property
    def force_t(self) -> Callable[[np.ndarray], np.ndarray]:
        return self.force_of_t

    @property
    def dforce_dt_t(self) -> Callable[[np.ndarray], np.ndarray]:
        return self.dforce_dt

    @property
    def dforce_da_p_value(self) -> float:
        return self.dforce_da_p()

    @property
    def static_deflection_t(self) -> Callable[[np.ndarray], np.ndarray]:
        return self.static_deflection_of_t

    @property
    def ddelta_dt_t(self) -> Callable[[np.ndarray], np.ndarray]:
        return self.ddelta_dt

    @property
    def ddelta_da_p_value(self) -> float:
        return self.ddelta_da_p()

    @property
    def ddelta_dF_value(self) -> float:
        return self.ddelta_dF()


# ─────────────────────────────────────────────────────────────
# DOECase — loader for doe_results.h5
#
# Expected structure inside the .h5:
#   <case_name>/
#       axial_disp/
#           time    [N]  [s]
#           values  [N]  [m]
#       axial_vel/
#           time    [N]  [s]
#           values  [N]  [m/s]
# ─────────────────────────────────────────────────────────────

@dataclass
class DOECase:
    """Loader for a single case inside a doe_results.h5 file.

    Parameters
    ----------
    h5_path : Path or str
        Path to the doe_results.h5 file.
    case_name : str
        Name of the group inside the file, e.g. 'case_001'.
    """
    h5_path: Path
    case_name: str

    def __post_init__(self) -> None:
        self.h5_path = Path(self.h5_path)

    def _read(self, signal: str, key: str) -> np.ndarray:
        """Read time or values array for axial_disp or axial_vel."""
        with h5py.File(self.h5_path, "r") as h5:
            return h5[f"{self.case_name}/{signal}/{key}"][:]

    @staticmethod
    def first_time_to_target(
        time: np.ndarray,
        values: np.ndarray,
        target: float,
        tol: float = 0.0,
    ) -> tuple[float | None, int | None]:
        """Return the first time where the curve reaches the target.

        The function looks for the first sample that enters the band
        [target - tol, target + tol]. If the input curve never reaches that
        band, it returns (None, None).

        Parameters
        ----------
        time : np.ndarray
            Time vector.
        values : np.ndarray
            Curve samples aligned with time.
        target : float
            Desired displacement value.
        tol : float, default 0.0
            Symmetric tolerance band around target.
        """
        time_arr = np.asarray(time, dtype=float)
        values_arr = np.asarray(values, dtype=float)

        if time_arr.shape != values_arr.shape:
            raise ValueError("time and values must have the same shape")
        if tol < 0:
            raise ValueError("tol must be >= 0")

        mask = np.abs(values_arr - target) <= tol
        if not np.any(mask):
            return None, None

        index = int(np.flatnonzero(mask)[0])
        return float(time_arr[index]), index

    def time_to_disp_target(self, target: float, tol: float = 0.0) -> tuple[float | None, int | None]:
        """Return the first time when displacement reaches a target value."""
        return self.first_time_to_target(self.disp_time, self.disp_values, target, tol=tol)

    @property
    def disp_time(self) -> np.ndarray:
        """Time vector for axial_disp [s]."""
        return self._read("Axial_disp", "time")


    @property
    def disp_values(self) -> np.ndarray:
        """Displacement values for axial_disp [m]."""
        return self._read("Axial_disp", "values")

    @property
    def vel_time(self) -> np.ndarray:
        """Time vector for axial_vel [s]."""
        return self._read("Axial_vel", "time")

    @property
    def vel_values(self) -> np.ndarray:
        """Velocity values for axial_vel [m/s]."""
        return self._read("Axial_vel", "values")
    
    @property
    def force_time(self) -> np.ndarray:
        return self._read("res_R_p", "time")
    
    @property
    def force_values(self) -> np.ndarray:
        return self._read("res_R_p", "values")

    def plot_disp(self, ax: Axes | None = None) -> tuple[Figure, Axes]:
        """Plot axial_disp in its own figure or on a provided axis."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            fig.suptitle(f"{self.h5_path.parent.name} — {self.case_name} — Axial Disp")
        else:
            fig = ax.get_figure()

        ax.plot(self.disp_time, self.disp_values, label=self.case_name)
        ax.set_ylabel("Axial disp [m]")
        ax.set_xlabel("Time [s]")
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)
        return fig, ax

    def plot_vel(self, ax: Axes | None = None) -> tuple[Figure, Axes]:
        """Plot axial_vel in its own figure or on a provided axis."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            fig.suptitle(f"{self.h5_path.parent.name} — {self.case_name} — Axial Vel")
        else:
            fig = ax.get_figure()

        ax.plot(self.vel_time, self.vel_values, label=self.case_name)
        ax.set_ylabel("Axial vel [m/s]")
        ax.set_xlabel("Time [s]")
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)
        return fig, ax
    
    def plot_force(self, ax: Axes | None = None) -> tuple[Figure, Axes]:
        """Plot res_R_p in its own figure or on a provided axis."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            fig.suptitle(f"{self.h5_path.parent.name} — {self.case_name} — Cutting Force")
        else:
            fig = ax.get_figure()

        ax.plot(self.force_time, self.force_values[:, 0], label=self.case_name)
        ax.set_ylabel("Cutting Force [N]")
        ax.set_xlabel("Time [s]")
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)
        return fig, ax

    def plot(self) -> tuple[tuple[Figure, Axes], tuple[Figure, Axes], tuple[Figure, Axes]]:
        """Plot axial_disp, axial_vel, and cutting force in three separate figures."""
        return self.plot_disp(), self.plot_vel(), self.plot_force()


def plot_cases(cases: list[DOECase], title_disp: str = "Axial Disp comparison", title_vel: str = "Axial Vel comparison", title_force: str = "Cutting Force comparison") -> tuple[tuple[Figure, Axes], tuple[Figure, Axes], tuple[Figure, Axes]]:
    """Overlay multiple DOECase objects into three separate figures."""
    fig_disp, ax_disp = plt.subplots(1, 1, figsize=(10, 4))
    fig_vel, ax_vel = plt.subplots(1, 1, figsize=(10, 4))
    fig_force, ax_force = plt.subplots(1, 1, figsize=(10, 4))
    fig_disp.suptitle(title_disp)
    fig_vel.suptitle(title_vel)
    fig_force.suptitle(title_force)
    for case in cases:
        case.plot_disp(ax=ax_disp)
        case.plot_vel(ax=ax_vel)
        case.plot_force(ax=ax_force)
    plt.tight_layout()
    return (fig_disp, ax_disp), (fig_vel, ax_vel), (fig_force, ax_force)


#%%
# -------------------------------------------------------------------
# USAGE EXAMPLE — OOP API
# -------------------------------------------------------------------

case = DOECase(
    h5_path=(
        "D:\\Thesis\\03-Code_Storage\\02-Altintlas_Nessy2m_Storage\\"
        "2DOF_Cone_DOE\\DOE_Influence_dexel_RPM_12000_ftooth_005_dt_200\\"
        "doe_results.h5"
        # "D:\\Thesis\\03-Code_Storage\\02-Altintlas_Nessy2m_Storage\\"
        # "Chatter-Criteria\\CAMP10_Chatter_detection_Methodes\\Convergency_Simulation\\"
        # "1_Detection_Limite_Lobes\\DOE_Detection_Limite_Lobes_dxl_1.25e-5_RUN_10\\doe_results.h5"
    ),
    case_name="case_000",
)

f2       = 150.0
xsi2     = 0.01
k2       = 2.13e8
theta_2  = 135.0 * np.pi / 180.0
phi2_z   = np.sin(theta_2)

Kf_fisico = 1.0e9       # N/m^2 (generic example)
Kf_modal  = (phi2_z**2) * Kf_fisico

Nz      = 1
n_rpm   = 12_094.28    # rpm
# n_rpm   = 12_000.0    # rpm
f_tooth = 0.05         # mm/tooth
vf      = n_rpm * f_tooth / 1e3 / 60   # m/s
L_cylindre = 150.e-3
t_ramp  = L_cylindre / vf

modal     = ModalParams.from_modal_freq(f_hz=f2, zeta=xsi2, k=k2)
machining_cone = MachiningParams.from_rpm(
    n_rpm=n_rpm, Nz=Nz,
    Kf=Kf_modal,
    a0=5.0e-3, a1=15.0e-3,
    t_ramp=t_ramp,
    dt_factor=5.0,
)

print(f"Modal Frecuncy:             {modal.omega_n/(2*np.pi):.4f} Hz")
print(f"Modal Damping:              {modal.zeta:.4f}")
print(f"Modal Mass:                 {modal.m:.4f} kg")
print(f"Modal Stiffness:            {modal.k:.4e} N/m")
print(f"Modal Damping Coefficient:  {modal.c:.4f} N·s/m")
print(f"Modal Zeta:                 {modal.zeta:.4f}")
print(f"Modal Damped Frequency:     {modal.omega_d/(2*np.pi):.4f} Hz")

# 4.3026e-3
case_ap = 4.3026e-3
ap_limit_deflection = StaticDeflection(
    case_ap=case_ap,
    modal=modal,
    f_tooth_mm=f_tooth,
    k_cut=Kf_fisico,
    alpha_deg=90.0,
    theta_deg=135.0,
)

ap_limit_deflection.compute()

print(f"Stabilisation Time: {ap_limit_deflection.t_delta_est:.6f} s")
t_quilibrio = 4/(modal.zeta * modal.omega_n)
print(f"Time to reach 98% of final deflection: {t_quilibrio:.6f} s")

ramp_deflection = StaticDeflectionRamp(
    a0=machining_cone.a0,
    a1=machining_cone.a1,
    t_ramp=machining_cone.t_ramp,
    modal=modal,
    f_tooth_mm=f_tooth,
    k_cut=Kf_fisico,
    alpha_deg=90.0,
    theta_deg=135.0,
)

dt_signal = case.disp_time[1] - case.disp_time[0]
N_echanillons_Sim = len(case.disp_time)
N_echanillons_ramp = int(t_ramp / dt_signal)

t_eval = np.linspace(0.0, t_ramp, N_echanillons_ramp)
ap_ramp = ramp_deflection.ap_t(t_eval)
force_ramp = ramp_deflection.force_t(t_eval)
deflection_ramp = ramp_deflection.static_deflection_t(t_eval)






ddelta_dt = ramp_deflection.ddelta_dt_t(t_eval)
dap_dt = ramp_deflection.dap_dt_t(t_eval)
dF_dt = ramp_deflection.dforce_dt_t(t_eval)
ddelta_dF = ramp_deflection.ddelta_dF_value
ddelta_da_p = ramp_deflection.ddelta_da_p_value
dF_da_p = ramp_deflection.dforce_da_p_value


print("\ndeflexion estatica teorica:")
print(f'  V_f        = {vf:.6f} m/s')
print(f"  F_ref      = {ap_limit_deflection.force:.6f} N")
print(f"  delta_ref  = {ap_limit_deflection.static_deflection:.6e} m")
print("\nrampa temporal:")
print(f"  ap(t)      = [{ap_ramp[0]:.6e}, ..., {ap_ramp[-1]:.6e}] m")
print(f"  F(t)       = [{force_ramp[0]:.6f}, ..., {force_ramp[-1]:.6f}] N")
print(f"  delta(t)   = [{deflection_ramp[0]:.6e}, ..., {deflection_ramp[-1]:.6e}] m")
print(f"  t_ramp     = {t_ramp:.6f} s")
print(f"  dt         = {machining_cone.dt:.6e} s")
print(f"  N_dt       = {int(t_ramp / machining_cone.dt)} steps")
print(f"  dF/da_p    = {dF_da_p:.6e} N/m")
print(f"  ddelta/da_p = {ddelta_da_p:.6e} m/m")
print(f"  ddelta/dF  = {ddelta_dF:.6e} m/N")
print(f"  ddelta/dt  = [{ddelta_dt[0]:.6e}, ..., {ddelta_dt[-1]:.6e}] m/s")

print(f"  dap/dt     = [{dap_dt[0]:.6e}, ..., {dap_dt[-1]:.6e}] m/s")
print(f"  dF/dt      = [{dF_dt[0]:.6e}, ..., {dF_dt[-1]:.6e}] N/s")




target_disp = deflection_ramp[0]        # valor objetivo = deflexion final de la rampa
tol_disp    = 0.5 * abs(target_disp)      # tolerancia del  1 %

t_reach, idx_reach = case.time_to_disp_target(target_disp, tol=tol_disp)
print(f"\ntiempo para alcanzar la deflexion objetivo:")
print(f"  target      = {target_disp:.6e} m")
print(f"  tolerancia  = ±{tol_disp:.6e} m  (1 %)")
if t_reach is not None:
    print(f"  t_reach     = {t_reach:.6f} s  (muestra {idx_reach})")
else:
    print("  la curva nunca entra en la banda objetivo")

print(f"\nTotal Time of Simulation: {case.disp_time[-1]:.6f} s")
print(f'Total Echantillons: {N_echanillons_Sim}')
print(f"Total time ramp: {t_ramp:.6f} s")
print(f"Total Time simulation  - Total time ramp: {case.disp_time[-1] - t_ramp:.6f} s")
print(f"Total Time Simulation  + Time to reach target: {case.disp_time[-1] + t_reach:.6f} s")


t_ramp_corected = t_eval + 60/n_rpm*2.5
fig_disp, ax_disp = case.plot_disp()
ax_disp.plot([t_ramp_corected[0], t_ramp_corected[-1]], [deflection_ramp[0], deflection_ramp[-1]], color="red", linestyle="--", label="delta(t) ramp")
ax_disp.axhline(deflection_ramp[0], color="green", linestyle="--", label="delta_initial")
ax_disp.axhline(target_disp, color="green", linestyle="--", label="target displacement")
ax_disp.axvline(t_reach, color="orange", linestyle="--", label="target reached")



ax_disp.legend(fontsize=14)

fig_vel, ax_vel = case.plot_vel()
ax_vel.axhline(ddelta_dt[0], color="red", linestyle="--", label="ddelta/dt")

fig_force, ax_force = case.plot_force()

ax_force.plot([t_ramp_corected[0], t_ramp_corected[-1]], [force_ramp[0], force_ramp[-1]], color="red", linestyle="--", label="F(t) ramp")



plt.show()

