"""Augmented Phase Trajectory — 3D geometric descriptors over Green Integral windows.

Extends ``run_fixed_window`` (Green Integral indicator) by enriching each
window it already computes with the following geometric descriptors of the
augmented trajectory  r(t) = [x(t), v(t), ap(t)]:

  Projected areas
  ---------------
  A_xv     — phase-plane area (x, v)  ← taken directly from result_fw.areas
  A_xap    — projected area in (x, ap) plane
  A_vap    — projected area in (v, ap) plane
  A_vec_*  — components and norm of the 3D vector area (cross-product method)

  Differential geometry
  ---------------------
  arc_length — length of r(t) in augmented phase space per window
  kappa      — mean ± std of pointwise curvature κ
  tau        — mean ± std of pointwise torsion τ

  Statistical occupation
  ----------------------
  H_xv / H_xap / H_vap — normalised Shannon entropy in each 2D projection
  H_3d                  — normalised Shannon entropy in 3D space

Conceptual notes
----------------
* A_xv is the classical phase-plane area — same formula as the Green indicator.
* A_xap and A_vap quantify how much the trajectory mixes with the ap axis.
  For constant-ap signals (all real HDF5 cases here) they are exactly zero.
* A_vec_norm summarises the total spatial extent of the augmented orbit.
* κ > 0 always; τ = 0 for planar curves (constant ap → trajectory stays 2D).
* H' ≈ 1 means the trajectory spreads uniformly in phase space (diffuse, chatter).
  H' ≈ 0 means the trajectory is concentrated (narrow, stable orbit).
* These are geometric DESCRIPTORS of a non-autonomous trajectory — they do NOT
  replace the Lyapunov-based σ̂ / Ĝ provided by the Green indicator.

Usage
-----
    cd indicators/green_integral
    python examples/augmented_trajectory_exploration.py

Toggle
------
    USE_REAL_DATA = False  →  synthetic example (ramp ap, growing oscillation)
    USE_REAL_DATA = True   →  real HDF5 (same case registry as
                               Green_Integral_Detection_NEW.py)
"""

from __future__ import annotations

import pathlib
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple

import colorsys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (activates 3D projection)

# ── Path setup — allows running directly without pip install ───────────────
_HERE = pathlib.Path(__file__).resolve().parent
_SRC  = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from green_integral.logging_setup import configure_logging, LOGGING_LEVELS

configure_logging(level=LOGGING_LEVELS["info"])

from green_integral import (
    FixedWindowResult,
    HDF5Reader,
    SignalData,
    plots_fixed_window,
    run_fixed_window,
)

# ══════════════════════════════════════════════════════════════════════════════
# COLOR PALETTE — canonical CAMP10 colors (skill: indicator-plot-style)
# ══════════════════════════════════════════════════════════════════════════════
r, g, b = colorsys.hls_to_rgb(346/360, 0.45, 0.99);  color_red    = (r, g, b)
r, g, b = colorsys.hls_to_rgb(36/360,  0.45, 0.99);  color_orange = (r, g, b)
r, g, b = colorsys.hls_to_rgb(279/360, 0.36, 0.99);  color_purple = (r, g, b)
r, g, b = colorsys.hls_to_rgb(98/360,  0.36, 0.99);  color_verde  = (r, g, b)
r, g, b = colorsys.hls_to_rgb(206.957/360, 0.40941, 0.55603);  color_azul = (r, g, b)

# ── Figure helpers ─────────────────────────────────────────────────────────────
def fig_size(scale: float = 1.0, ncols: int = 1, base_width: float = 3.4):
    """Return (width, height) in inches — IEEE/Elsevier journal format."""
    width = base_width * ncols * scale
    return (width, width * 0.40)


def configurar_estilo_global() -> None:
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 9,
        'axes.titlesize': 25,   'axes.labelsize': 25,
        'xtick.labelsize': 23,  'ytick.labelsize': 23, 'legend.fontsize': 23,
        'lines.linewidth': 1.25, 'lines.markersize': 6,
        'axes.linewidth': 0.8,  'grid.linewidth': 0.5,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.major.size': 4,  'ytick.major.size': 4,
        'xtick.minor.size': 2.5, 'ytick.minor.size': 2.5,
        'xtick.minor.width': 0.6, 'ytick.minor.width': 0.6,
        'mathtext.fontset': 'stix', 'axes.formatter.use_mathtext': True,
        'legend.frameon': False, 'legend.loc': 'best',
        'legend.handlelength': 2.0, 'legend.borderaxespad': 0.5,
        'figure.dpi': 100, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'savefig.transparent': True,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    })


configurar_estilo_global()


def _draw_vlines(ax, vlines, default_color="black", default_ls="--"):
    """Draw vertical event lines with rotated text labels.

    Each entry in ``vlines`` may be:
      - float                    → dashed line, no label
      - (float, label)           → dashed line + vertical label (default_color)
      - (float, label, color)    → dashed line + vertical label (custom color)
    """
    if vlines is None:
        return
    for entry in vlines:
        if isinstance(entry, (list, tuple)):
            vx    = float(entry[0])
            label = str(entry[1]) if len(entry) > 1 else None
            color = entry[2]      if len(entry) > 2 else default_color
        else:
            vx, label, color = float(entry), None, default_color
        ax.axvline(x=vx, color=color, linestyle=default_ls, lw=1.2)
        if label:
            ax.text(
                vx, 0.97, f"  {label}",
                rotation=90, va="top", ha="right", fontsize=16,
                color=color, transform=ax.get_xaxis_transform(),
            )


# ══════════════════════════════════════════════════════════════════════════════
# DEPTH-OF-CUT PROFILES  (same interface as time.py: callable t → ap array)
# ══════════════════════════════════════════════════════════════════════════════

class DepthProfile(Protocol):
    """Protocol: any callable t → ap(t)."""
    def __call__(self, t: np.ndarray) -> np.ndarray: ...


class LinearRampProfile:
    """Linear ramp from *a0* [m] to *a1* [m] over *t_ramp* seconds."""
    def __init__(self, a0: float, a1: float, t_ramp: float) -> None:
        self.a0, self.a1, self.t_ramp = a0, a1, t_ramp

    def __call__(self, t: np.ndarray) -> np.ndarray:
        r = (self.a1 - self.a0) / self.t_ramp
        return np.where(t <= self.t_ramp, self.a0 + r * t, self.a1)


class ConstantProfile:
    """Constant depth of cut throughout the operation."""
    def __init__(self, ap: float) -> None:
        self.ap = ap

    def __call__(self, t: np.ndarray) -> np.ndarray:
        return np.full_like(t, self.ap, dtype=float)


class StepProfile:
    """Step profile: *a0* until *t1*, then *a1* afterward."""
    def __init__(self, a0: float, a1: float, t1: float) -> None:
        self.a0, self.a1, self.t1 = a0, a1, t1

    def __call__(self, t: np.ndarray) -> np.ndarray:
        return np.where(t <= self.t1, self.a0, self.a1)


# ══════════════════════════════════════════════════════════════════════════════
# MODAL / MACHINING PARAMETER CLASSES  (same as time.py)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModalParams:
    """Modal parameters of the 1-DOF system."""
    m: float   # modal mass [kg]
    c: float   # modal damping [N·s/m]
    k: float   # stiffness [N/m]

    @property
    def omega_n(self) -> float:
        return float(np.sqrt(self.k / self.m))

    @property
    def zeta(self) -> float:
        return self.c / (2.0 * self.m * self.omega_n)

    @classmethod
    def from_modal_freq(cls, f_hz: float, zeta: float, k: float) -> "ModalParams":
        omega_n = 2.0 * np.pi * f_hz
        m = k / omega_n ** 2
        c = 2.0 * zeta * m * omega_n
        return cls(m=m, c=c, k=k)


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


# ══════════════════════════════════════════════════════════════════════════════
# MACHINING PARAMETERS — define ap profiles here (same pattern as time.py)
# ══════════════════════════════════════════════════════════════════════════════

f2          = 150.0
xsi2        = 0.01
k2          = 2.13e8
theta_2     = 135.0 * np.pi / 180.0
phi2_z      = np.sin(theta_2)

Kf_fisico   = 1.0e9          # N/m²
Kf_modal    = (phi2_z**2) * Kf_fisico

Nz          = 1
# n_rpm     = 12_093.995_36  # rpm
n_rpm       = 12_099.275244  # rpm
f_tooth     = 0.05           # mm/tooth
vf          = n_rpm * f_tooth / 1e3 / 60   # m/s
L_cylindre  = 150.0e-3       # m
t_ramp      = L_cylindre / vf

_modal      = ModalParams.from_modal_freq(f_hz=f2, zeta=xsi2, k=k2)
_machining  = MachiningParams.from_rpm(
    n_rpm=n_rpm, Nz=Nz,
    Kf=Kf_modal,
    a0=5.0e-3, a1=15.0e-3,
    t_ramp=t_ramp,
    dt_factor=5.0,
)

# ══════════════════════════════════════════════════════════════════════════════
# TOGGLES
# ══════════════════════════════════════════════════════════════════════════════
USE_REAL_DATA: bool = True   # False → synthetic   True → HDF5

# ══════════════════════════════════════════════════════════════════════════════
# CASE SELECTOR — change only this line to switch between signals
# ══════════════════════════════════════════════════════════════════════════════
_ACTIVE_CASE = "cono"   # "cono" | "stable_5mm" | "chatter_15mm"

_BASE = r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage"

_CASES: Dict = {
    # ── Original 2DOF cone (chatter onset at 5.366 s) ──────────────────────
    "cono": {
        "hdf5":               rf"{_BASE}\2DOF_Cono\1DOF_150Hz\out.hdf5",
        "name":               "cono",
        "t_range":            (0.05, 16.0),
        "t_gt":               5.36577,
        "f_modal":            200.0,
        "num_T":              16,
        # ── ap profile — set by the user, not read from HDF5 ──────────────
        # Use LinearRampProfile, ConstantProfile, StepProfile, or any
        # callable f(t) -> np.ndarray.  Units: metres (SI).
        "ap":                 LinearRampProfile(a0=_machining.a0, a1=_machining.a1, t_ramp=_machining.t_ramp),
        "use_area_threshold": True,
        "training_intervals": [
            (0.05, 5.36577, "stable_1"),
            (3.3,  4.46,    "stable_2"),
            (4.46, 5.36577, "stable_1"),
        ],
    },
    # ── Stable case — ap = 5 mm ─────────────────────────────────────────────
    "stable_5mm": {
        "hdf5":               (rf"{_BASE}\Chatter-Criteria\CAMP8-Ventanna_Glisante"
                               r"\Nessy2m_Case_Test_Explicit\1DOF_150Hz_5mm\1DOF_150Hz\out.hdf5"),
        "name":               "5mm_stable",
        "t_range":            (0.05, 16.0),
        "t_gt":               None,
        "f_modal":            150.0,
        "num_T":              1,
        "ap":                 ConstantProfile(ap=5.0e-3),   # float (constant) or callable f(t)->array
        "use_area_threshold": False,
    },
    # ── Chatter case — ap = 15 mm ───────────────────────────────────────────
    "chatter_15mm": {
        "hdf5":               (rf"{_BASE}\Chatter-Criteria\CAMP8-Ventanna_Glisante"
                               r"\Nessy2m_Case_Test_Explicit\1DOF_150Hz_15mm\1DOF_150Hz\out.hdf5"),
        "name":               "15mm_chatter",
        "t_range":            (0.01, 0.88),
        "t_gt":               0.05,
        "f_modal":            150.0,
        "num_T":              5,
        "ap":                 ConstantProfile(ap=15.0e-3),  # float (constant) or callable f(t)->array
        "use_area_threshold": False,
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Mathematical helpers
# (only functions that the Green indicator does NOT already provide)
# ══════════════════════════════════════════════════════════════════════════════

def _shoelace(u: np.ndarray, w: np.ndarray) -> float:
    """Signed shoelace area of a closed 2D polygon.

    Closes the polygon by connecting the last vertex back to the first.
    Returns nan if fewer than 3 finite points are available.

    This is the **same formula** used internally by the Green Integral indicator
    to compute A_xv. Here it is applied to the (x, ap) and (v, ap) projections.

    Notes
    -----
    For constant ``w`` the result is exactly 0 (algebraically):
        Σ(u_n w_{n+1} - u_{n+1} w_n)  with w_n = c  →  c · Σ(u_n - u_{n+1}) = 0
    This means A_xap = A_vap = 0 for constant-ap (fixed-depth) simulations.
    """
    u = np.asarray(u, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    ok = np.isfinite(u) & np.isfinite(w)
    u, w = u[ok], w[ok]
    if len(u) < 3:
        return np.nan
    u_c = np.append(u, u[0])
    w_c = np.append(w, w[0])
    return float(0.5 * np.sum(u_c[:-1] * w_c[1:] - u_c[1:] * w_c[:-1]))


def vector_area_3d(
    x: np.ndarray,
    v: np.ndarray,
    ap: np.ndarray,
) -> Tuple[np.ndarray, float, float, float, float]:
    """Vector area of the augmented trajectory r(t) = [x(t), v(t), ap(t)].

    Computes the closed-curve formula:

        A_vec = 1/2 · Σ_n  r_n × r_{n+1}

    The cross-product components map to the projected planes as follows:

        A_vec[0]  →  (v, ap)  plane  (A_vap)
        A_vec[1]  →  (ap, x)  plane  (A_apx)
        A_vec[2]  →  (x, v)   plane  ← equals _shoelace(x, v) algebraically

    Returns
    -------
    A_vec : ndarray, shape (3,)
    norm_A : float — Euclidean norm of A_vec
    A_vap, A_apx, A_xv : float — signed components
    """
    x  = np.asarray(x,  dtype=float).ravel()
    v  = np.asarray(v,  dtype=float).ravel()
    ap = np.asarray(ap, dtype=float).ravel()
    ok = np.isfinite(x) & np.isfinite(v) & np.isfinite(ap)
    x, v, ap = x[ok], v[ok], ap[ok]
    if len(x) < 3:
        nan3 = np.full(3, np.nan)
        return nan3, np.nan, np.nan, np.nan, np.nan
    r   = np.column_stack([x, v, ap])       # (N, 3)
    r_c = np.vstack([r, r[0]])              # closed: append first point
    A_vec  = 0.5 * np.cross(r_c[:-1], r_c[1:]).sum(axis=0)
    norm_A = float(np.linalg.norm(A_vec))
    return A_vec, norm_A, float(A_vec[0]), float(A_vec[1]), float(A_vec[2])


def arc_length_3d(
    x: np.ndarray,
    v: np.ndarray,
    ap: np.ndarray,
) -> float:
    """Total arc length of r(t) = [x, v, ap] in augmented phase space.

        L = Σ_n  ||r_{n+1} − r_n||

    Returns nan if fewer than 2 finite points are available.

    Notes
    -----
    For constant ap the arc length reduces to the 2D phase-plane arc length.
    A growing oscillation increases arc length even if the orbit shape is fixed,
    because the orbit becomes larger.  arc_length is therefore complementary to
    the area: it captures the *perimeter* of the orbit rather than its *area*.
    """
    x  = np.asarray(x,  dtype=float).ravel()
    v  = np.asarray(v,  dtype=float).ravel()
    ap = np.asarray(ap, dtype=float).ravel()
    ok = np.isfinite(x) & np.isfinite(v) & np.isfinite(ap)
    x, v, ap = x[ok], v[ok], ap[ok]
    if len(x) < 2:
        return np.nan
    dr = np.diff(np.column_stack([x, v, ap]), axis=0)
    return float(np.sum(np.linalg.norm(dr, axis=1)))


def curvature_torsion_3d(
    x: np.ndarray,
    v: np.ndarray,
    ap: np.ndarray,
    dt: float,
) -> Tuple[float, float, float, float]:
    """Pointwise curvature κ and torsion τ of r(t) = [x, v, ap].

    Uses central-difference via numpy.gradient:

        κ_n = ||r'_n × r''_n|| / ||r'_n||³
        τ_n = (r'_n × r''_n) · r'''_n / ||r'_n × r''_n||²

    Singularities (||r'|| ≈ 0 or ||r' × r''|| ≈ 0) are replaced by nan.

    Parameters
    ----------
    dt : float
        Sampling interval of the points in this window [s].

    Returns
    -------
    kappa_mean, kappa_std, tau_mean, tau_std : float

    Notes
    -----
    For a **planar curve** (constant ap) τ = 0 exactly because
    r' × r'' lies entirely along the ap-axis and r''' has no ap component.
    This provides a built-in sanity check for constant-ap real data.
    """
    x  = np.asarray(x,  dtype=float).ravel()
    v  = np.asarray(v,  dtype=float).ravel()
    ap = np.asarray(ap, dtype=float).ravel()
    ok = np.isfinite(x) & np.isfinite(v) & np.isfinite(ap)
    x, v, ap = x[ok], v[ok], ap[ok]
    if len(x) < 5:
        return np.nan, np.nan, np.nan, np.nan

    r  = np.column_stack([x, v, ap])            # (N, 3)
    r1 = np.gradient(r,  dt, axis=0)            # r'
    r2 = np.gradient(r1, dt, axis=0)            # r''
    r3 = np.gradient(r2, dt, axis=0)            # r'''

    cross12  = np.cross(r1, r2)                 # (N, 3)
    norm_r1  = np.linalg.norm(r1,      axis=1)  # (N,)
    norm_c12 = np.linalg.norm(cross12, axis=1)  # (N,)

    _eps = 1e-30
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(norm_r1 ** 3 > _eps,
                         norm_c12 / norm_r1 ** 3,
                         np.nan)
        tau   = np.where(norm_c12 ** 2 > _eps,
                         np.einsum("ij,ij->i", cross12, r3) / norm_c12 ** 2,
                         np.nan)

    kf = kappa[np.isfinite(kappa)]
    tf = tau[np.isfinite(tau)]
    km = float(np.mean(kf))  if len(kf) > 0 else np.nan
    ks = float(np.std(kf))   if len(kf) > 0 else np.nan
    tm = float(np.mean(tf))  if len(tf) > 0 else np.nan
    ts = float(np.std(tf))   if len(tf) > 0 else np.nan
    return km, ks, tm, ts


def statistical_occupation(
    x: np.ndarray,
    v: np.ndarray,
    ap: np.ndarray,
    n_bins: int = 20,
) -> Dict[str, float]:
    """Normalised Shannon entropy of the phase-space occupation.

    For each 2D projection (and the full 3D space) builds a histogram and
    computes the normalised entropy:

        H' = H / H_max   where  H_max = ln(n_bins²) for 2D, ln(n_bins³) for 3D

    H' ∈ [0, 1].

    H' ≈ 1  →  trajectory spreads uniformly  (diffuse / chatter).
    H' ≈ 0  →  trajectory concentrates       (narrow / stable orbit).

    Returns
    -------
    dict with keys ``H_xv``, ``H_xap``, ``H_vap``, ``H_3d``.
    """
    def _h2d(u: np.ndarray, w: np.ndarray) -> float:
        ok = np.isfinite(u) & np.isfinite(w)
        u, w = u[ok], w[ok]
        if len(u) < 4:
            return np.nan
        counts, _, _ = np.histogram2d(u, w, bins=n_bins)
        c = counts.ravel()
        total = c.sum()
        if total == 0:
            return np.nan
        p = c[c > 0] / total
        return float(-np.sum(p * np.log(p)) / np.log(n_bins ** 2))

    x  = np.asarray(x,  dtype=float).ravel()
    v  = np.asarray(v,  dtype=float).ravel()
    ap = np.asarray(ap, dtype=float).ravel()

    H_xv  = _h2d(x, v)
    H_xap = _h2d(x, ap)
    H_vap = _h2d(v, ap)

    # 3D entropy — use fewer bins to avoid sparsity with per-window data
    n3  = max(5, n_bins // 4)
    ok3 = np.isfinite(x) & np.isfinite(v) & np.isfinite(ap)
    x3, v3, ap3 = x[ok3], v[ok3], ap[ok3]
    if len(x3) >= 8:
        c3d, _ = np.histogramdd(np.column_stack([x3, v3, ap3]), bins=n3)
        c3     = c3d.ravel()
        p3     = c3[c3 > 0] / c3.sum()
        H_3d   = float(-np.sum(p3 * np.log(p3)) / np.log(n3 ** 3))
    else:
        H_3d = np.nan

    return {"H_xv": H_xv, "H_xap": H_xap, "H_vap": H_vap, "H_3d": H_3d}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Central enrichment function
# ══════════════════════════════════════════════════════════════════════════════

def enrich_windows(
    sig: SignalData,
    ap: np.ndarray,
    result_fw: FixedWindowResult,
    window_size_s: float,
    n_bins_occ: int = 20,
) -> pd.DataFrame:
    """Enrich each Green-indicator window with 3D geometric descriptors.

    Uses ``result_fw.t_wins`` and ``result_fw.areas`` **directly** — A_xv is
    never recomputed.  For each window the function adds:

    * A_xap, A_vap        — projected areas in (x, ap) and (v, ap) planes
    * A_vec_x/y/z, norm   — 3D vector area components and Euclidean norm
    * arc_length           — length of r(t)=[x,v,ap] through the window
    * kappa_mean/std       — mean ± std of pointwise curvature κ
    * tau_mean/std         — mean ± std of pointwise torsion τ
    * H_xv/H_xap/H_vap   — normalised 2D Shannon entropy per projection
    * H_3d                — normalised 3D Shannon entropy

    Parameters
    ----------
    sig : SignalData
        Signal container (.t, .displacement, .velocity).
    ap : np.ndarray
        Depth-of-cut array aligned with sig.t [mm].
    result_fw : FixedWindowResult
        Output of run_fixed_window — provides t_wins and areas.
    window_size_s : float
        Duration of each window [s]  (= num_T / f_modal).
    n_bins_occ : int
        Histogram bins per axis for statistical occupation.

    Returns
    -------
    pd.DataFrame
        One row per window with columns:
        t_center, A_xv, A_xap, A_vap, A_vec_x, A_vec_y, A_vec_z,
        A_vec_norm, arc_length, kappa_mean, kappa_std, tau_mean, tau_std,
        H_xv, H_xap, H_vap, H_3d, ap_center, delta_ap, n_points.
    """
    t  = sig.t
    x  = sig.displacement
    v  = sig.velocity
    ap = np.asarray(ap, dtype=float).ravel()

    if len(t) < 2 or len(result_fw.t_wins) == 0:
        return pd.DataFrame()

    dt   = float(t[1] - t[0])
    rows: List[dict] = []

    for i, t_start in enumerate(result_fw.t_wins):
        mask  = (t >= t_start) & (t < t_start + window_size_s)
        n_pts = int(mask.sum())
        if n_pts < 5:           # need ≥ 5 pts for curvature (np.gradient needs 3×)
            continue

        x_w  = x[mask]
        v_w  = v[mask]
        ap_w = ap[mask]
        t_w  = t[mask]

        t_center  = float(np.mean(t_w))
        ap_center = float(np.mean(ap_w))
        delta_ap  = float(ap_w[-1] - ap_w[0])

        # ── Areas ─────────────────────────────────────────────────────────
        # A_xv comes directly from the Green indicator — do NOT recompute
        A_xv  = float(result_fw.areas[i]) if i < len(result_fw.areas) else np.nan
        A_xap = abs(_shoelace(x_w, ap_w))
        A_vap = abs(_shoelace(v_w, ap_w))

        A_vec, norm_A, _Avap, _Aapx, _Axv = vector_area_3d(x_w, v_w, ap_w)

        # ── Differential geometry ─────────────────────────────────────────
        L             = arc_length_3d(x_w, v_w, ap_w)
        km, ks, tm, ts = curvature_torsion_3d(x_w, v_w, ap_w, dt)

        # ── Statistical occupation ────────────────────────────────────────
        occ = statistical_occupation(x_w, v_w, ap_w, n_bins=n_bins_occ)

        rows.append({
            "t_center":   t_center,
            "A_xv":       A_xv,
            "A_xap":      A_xap,
            "A_vap":      A_vap,
            "A_vec_x":    float(A_vec[0]) if np.isfinite(A_vec[0]) else np.nan,
            "A_vec_y":    float(A_vec[1]) if np.isfinite(A_vec[1]) else np.nan,
            "A_vec_z":    float(A_vec[2]) if np.isfinite(A_vec[2]) else np.nan,
            "A_vec_norm": norm_A,
            "arc_length": L,
            "kappa_mean": km,
            "kappa_std":  ks,
            "tau_mean":   tm,
            "tau_std":    ts,
            "H_xv":       occ["H_xv"],
            "H_xap":      occ["H_xap"],
            "H_vap":      occ["H_vap"],
            "H_3d":       occ["H_3d"],
            "ap_center":  ap_center,
            "delta_ap":   delta_ap,
            "n_points":   n_pts,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Plot functions
# ══════════════════════════════════════════════════════════════════════════════

def _add_tgt_vline(ax: "plt.Axes", t_gt: Optional[float], t_d: Optional[float] = None) -> None:
    """Add canonical vertical event lines for t_gt (black) and t_d (color_orange)."""
    vlines = []
    if t_gt is not None:
        vlines.append((t_gt, rf"$t_{{gt}}={t_gt:.3f}$ s", "black"))
    if t_d is not None:
        vlines.append((t_d, rf"$t_d={t_d:.3f}$ s", color_orange))
    _draw_vlines(ax, vlines)


def plot_3d_trajectory(
    t: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    ap: np.ndarray,
    title: str = "Augmented Phase Trajectory  r(t) = [x, v, ap]",
    t_gt: Optional[float] = None,
) -> None:
    """3D line (or 2D fallback) of the augmented trajectory, coloured by time.

    When ap is constant (fixed depth-of-cut simulations) the 3D plot degenerates
    to a flat plane and the ap axis carries no information.  In that case the
    function automatically falls back to a 2D phase-plane plot (x vs v) and
    annotates the fixed ap value in the title.

    The colour scale (viridis) runs from the start (dark) to the end (light),
    making stable vs. chatter regions visually distinguishable.
    """
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    stride = max(1, len(t) // 20_000)   # for 2-D fallback (LineCollection is fast)
    t_, x_, v_, ap_ = t[::stride], x[::stride], v[::stride], ap[::stride]
    if len(t_) < 2:
        return

    ap_std  = float(np.std(ap_))
    ap_mean = float(np.mean(ap_))
    ap_is_constant = ap_std < 1e-6 * max(abs(ap_mean), 1.0)

    # 3-D rendering is slow (software-only) → cap at 2 000 points so rotation stays smooth
    _MAX_3D = 2_000
    stride3d = max(1, len(t_) // _MAX_3D)
    t3, x3, v3, ap3 = t_[::stride3d], x_[::stride3d], v_[::stride3d], ap_[::stride3d]

    auto_vlines = []
    if t_gt is not None:
        auto_vlines.append((t_gt, rf"$t_{{gt}}={t_gt:.3f}$ s", "black"))

    if ap_is_constant:
        # ── 2-D fallback: phase plane (x, v) ─────────────────────────────
        pts  = np.stack([x_, v_], axis=1).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        from matplotlib.collections import LineCollection
        fig, ax = plt.subplots(figsize=fig_size(scale=3.0), constrained_layout=True)
        norm = plt.Normalize(t_[0], t_[-1])
        lc   = LineCollection(segs, cmap="viridis", norm=norm,
                              linewidth=0.8, alpha=0.85)
        lc.set_array(t_[:-1])
        ax.add_collection(lc)
        ax.set_xlim(x_.min(), x_.max())
        ax.set_ylim(v_.min(), v_.max())
        ax.set_xlabel(r"$x$  [m]")
        ax.set_ylabel(r"$v$  [m/s]")
        ax.set_title(
            rf"{title}"
            "\n"
            rf"2D phase plane  —  $ap = {ap_mean:.4g}$ mm (constant)"
        )
        cbar = fig.colorbar(lc, ax=ax, label=r"$t$  [s]")
        _draw_vlines(ax, auto_vlines)
        plt.show()
        plt.close(fig)
    else:
        # ── 3-D plot: genuine augmented trajectory ────────────────────────
        pts  = np.stack([x3, v3, ap3], axis=1).reshape(-1, 1, 3)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        fig  = plt.figure(figsize=fig_size(scale=3.0))
        ax   = fig.add_subplot(111, projection="3d")
        norm = plt.Normalize(t3[0], t3[-1])
        lc   = Line3DCollection(segs, cmap="viridis", norm=norm,
                                linewidth=0.7, alpha=0.85)
        lc.set_array(t3[:-1])
        ax.add_collection(lc)
        ax.set_xlim(x3.min(), x3.max())
        ax.set_ylim(v3.min(), v3.max())
        ax.set_zlim(ap3.min(), ap3.max())
        ax.set_xlabel(r"$x$  [m]")
        ax.set_ylabel(r"$v$  [m/s]")
        ax.set_zlabel(r"$ap$  [mm]")
        ax.set_title(title + f"\n({len(t3)} pts displayed  —  stride {stride3d}×{stride})")
        fig.colorbar(lc, ax=ax, pad=0.1, shrink=0.55, label=r"$t$  [s]")
        plt.show()
        plt.close(fig)


def plot_area_features(
    df: pd.DataFrame,
    t_gt: Optional[float] = None,
    title_prefix: str = "",
) -> None:
    """Time series of projected and 3D vector areas.

    Four separate figures:
      1. A_xv   — phase-plane area from the Green indicator (reference)
      2. A_xap  — projected area in (x, ap) plane
      3. A_vap  — projected area in (v, ap) plane
      4. A_vec_norm — magnitude of the 3D vector area
    """
    t   = df["t_center"].to_numpy()
    pfx = f"{title_prefix} — " if title_prefix else ""

    auto_vlines = []
    if t_gt is not None:
        auto_vlines.append((t_gt, rf"$t_{{gt}}={t_gt:.3f}$ s", "black"))

    specs = [
        ("A_xv",       r"$A_{xv}$  [m·m/s]",  "Phase-plane area  $(x,\,v)$  \u2190 Green indicator"),
        ("A_xap",      r"$A_{xap}$  [m·mm]",   "Projected area  $(x,\,ap)$"),
        ("A_vap",      r"$A_{vap}$  [m·mm/s]", "Projected area  $(v,\,ap)$"),
        ("A_vec_norm", r"$\|A_{vec}\|$",        r"3D vector area norm  $\|A_{vec}\|$"),
    ]
    for col, ylabel, subtitle in specs:
        if col not in df.columns:
            continue
        y = df[col].to_numpy()
        fig, ax = plt.subplots(figsize=fig_size(scale=3.5))
        # Shade stable / chatter regions
        if t_gt is not None:
            ax.fill_between(t, 0, y, where=(t < t_gt),  alpha=0.07, color=color_azul)
            ax.fill_between(t, 0, y, where=(t >= t_gt), alpha=0.07, color=color_orange)
        ax.plot(t, y, color=color_azul, linewidth=1.25)
        ax.set_xlabel(r"$t$  [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{pfx}{subtitle}")
        ax.grid(True, linestyle="--", alpha=0.4)
        _draw_vlines(ax, auto_vlines)
        plt.tight_layout()
        plt.show()
        plt.close(fig)


def plot_geometry_features(
    df: pd.DataFrame,
    t_gt: Optional[float] = None,
    title_prefix: str = "",
) -> None:
    """Time series of arc length, curvature (with ±std band), and torsion.

    Three separate figures:
      1. arc_length — total path length of r(t) per window
      2. kappa_mean ± kappa_std — curvature with uncertainty band
      3. tau_mean   ± tau_std   — torsion with uncertainty band

    Interpretation hints
    --------------------
    * arc_length grows when the orbit becomes larger (chatter → larger amplitude).
    * kappa drops for large-radius orbits (less curved) and rises for highly
      bent trajectories.
    * tau ≈ 0 for planar (constant-ap) trajectories; non-zero tau indicates
      the trajectory genuinely leaves the 2D phase plane.
    """
    t   = df["t_center"].to_numpy()
    pfx = f"{title_prefix} — " if title_prefix else ""

    auto_vlines = []
    if t_gt is not None:
        auto_vlines.append((t_gt, rf"$t_{{gt}}={t_gt:.3f}$ s", "black"))

    # 1. Arc length
    if "arc_length" in df.columns:
        fig, ax = plt.subplots(figsize=fig_size(scale=3.5))
        ax.plot(t, df["arc_length"].to_numpy(), color=color_azul, linewidth=1.25)
        ax.set_xlabel(r"$t$  [s]")
        ax.set_ylabel(r"$L$  [mixed units]")
        ax.set_title(rf"{pfx}Arc length of $r(t)=[x,\,v,\,ap]$ per window")
        ax.grid(True, linestyle="--", alpha=0.4)
        _draw_vlines(ax, auto_vlines)
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    # 2. Curvature
    if "kappa_mean" in df.columns and "kappa_std" in df.columns:
        km = df["kappa_mean"].to_numpy()
        ks = df["kappa_std"].to_numpy()
        fig, ax = plt.subplots(figsize=fig_size(scale=3.5))
        ax.plot(t, km, color=color_purple, linewidth=1.25, label=r"$\kappa$ mean")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax.fill_between(t, km - ks, km + ks, alpha=0.12,
                            color=color_purple, label=r"$\pm$ std")
        ax.set_xlabel(r"$t$  [s]")
        ax.set_ylabel(r"$\kappa$  [mixed units]")
        ax.set_title(rf"{pfx}Curvature $\kappa$  (mean $\pm$ std per window)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        _draw_vlines(ax, auto_vlines)
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    # 3. Torsion
    if "tau_mean" in df.columns and "tau_std" in df.columns:
        tm = df["tau_mean"].to_numpy()
        ts = df["tau_std"].to_numpy()
        fig, ax = plt.subplots(figsize=fig_size(scale=3.5))
        ax.plot(t, tm, color=color_purple, linewidth=1.25, label=r"$\tau$ mean")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax.fill_between(t, tm - ts, tm + ts, alpha=0.12,
                            color=color_purple, label=r"$\pm$ std")
        ax.set_xlabel(r"$t$  [s]")
        ax.set_ylabel(r"$\tau$  [mixed units]")
        ax.set_title(rf"{pfx}Torsion $\tau$  (mean $\pm$ std per window) — 0 for planar curves")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        _draw_vlines(ax, auto_vlines)
        plt.tight_layout()
        plt.show()
        plt.close(fig)


def plot_occupation_features(
    df: pd.DataFrame,
    t_gt: Optional[float] = None,
    title_prefix: str = "",
) -> None:
    """Time series of normalised Shannon entropy (phase-space occupation).

    Four separate figures (H_xv, H_xap, H_vap, H_3d), y-axis in [0, 1].

    H' ≈ 1 → trajectory fills phase space uniformly  (chatter / diffuse).
    H' ≈ 0 → trajectory is confined to a small region (stable / concentrated).
    """
    t   = df["t_center"].to_numpy()
    pfx = f"{title_prefix} — " if title_prefix else ""

    auto_vlines = []
    if t_gt is not None:
        auto_vlines.append((t_gt, rf"$t_{{gt}}={t_gt:.3f}$ s", "black"))

    specs = [
        ("H_xv",  r"$H'_{xv}$",  r"Occupation entropy — $(x,\,v)$ plane"),
        ("H_xap", r"$H'_{xap}$", r"Occupation entropy — $(x,\,ap)$ plane"),
        ("H_vap", r"$H'_{vap}$", r"Occupation entropy — $(v,\,ap)$ plane"),
        ("H_3d",  r"$H'_{3d}$",  r"Occupation entropy — 3D space $[x,\,v,\,ap]$"),
    ]
    for col, ylabel, subtitle in specs:
        if col not in df.columns:
            continue
        y = df[col].to_numpy()
        fig, ax = plt.subplots(figsize=fig_size(scale=3.5))
        # Shade stable / chatter background
        if t_gt is not None:
            ax.fill_between(t, 0, 1, where=(t < t_gt),  alpha=0.06,
                            color=color_azul)
            ax.fill_between(t, 0, 1, where=(t >= t_gt), alpha=0.06,
                            color=color_orange)
        ax.plot(t, y, color=color_azul, linewidth=1.25)
        ax.set_xlabel(r"$t$  [s]")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.08)
        ax.set_title(rf"{pfx}{subtitle}  (0 = concentrated, 1 = uniform)")
        ax.grid(True, linestyle="--", alpha=0.4)
        _draw_vlines(ax, auto_vlines)
        plt.tight_layout()
        plt.show()
        plt.close(fig)


def plot_phase_snapshots(
    sig: SignalData,
    ap: np.ndarray,
    result_fw: FixedWindowResult,
    window_size_s: float,
    n_snapshots: int = 4,
    t_gt: Optional[float] = None,
) -> None:
    """Phase-plane orbit (x, v) for n evenly-spaced windows.

    Each figure shows one orbit coloured by sample index (early → late within
    the window).  The title reports the window centre time, the A_xv value
    taken directly from the Green indicator, and whether the window falls
    in the stable or chatter region (when t_gt is provided).

    This gives an intuitive feel for what the area metric captures at
    different stages of the machining process — small circle = stable,
    large spreading orbit = chatter.
    """
    t = sig.t
    x = sig.displacement
    v = sig.velocity

    n_wins = len(result_fw.t_wins)
    if n_wins == 0:
        return

    indices = np.unique(
        np.linspace(0, n_wins - 1, min(n_snapshots, n_wins), dtype=int)
    )

    for idx in indices:
        t_start  = result_fw.t_wins[idx]
        mask     = (t >= t_start) & (t < t_start + window_size_s)
        x_w, v_w = x[mask], v[mask]
        t_w      = t[mask]
        if len(x_w) < 3:
            continue

        t_center   = float(np.mean(t_w))
        area_val   = float(result_fw.areas[idx]) if idx < len(result_fw.areas) else np.nan
        is_chatter = (t_gt is not None) and (t_center >= t_gt)
        state      = "CHATTER" if is_chatter else "stable"
        c_orbit    = color_orange if is_chatter else color_azul

        _fs = fig_size(scale=2.0)
        fig, ax = plt.subplots(figsize=(_fs[0], _fs[0]))   # force square
        sc = ax.scatter(x_w, v_w, c=np.arange(len(x_w)),
                        cmap="plasma", s=5, zorder=5, alpha=0.85)
        ax.plot(x_w, v_w, color=c_orbit, linewidth=0.7, alpha=0.45)
        ax.set_xlabel(r"$x$  [m]")
        ax.set_ylabel(r"$v$  [m/s]")
        ax.set_title(
            rf"Phase orbit  [{state}]   $t_c = {t_center:.3f}$ s"
            "\n"
            rf"$A_{{xv}}$ (Green) $= {area_val:.3e}$"
        )
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.colorbar(sc, ax=ax, label="sample index")
        plt.tight_layout()
        plt.show()
        plt.close(fig)


def plot_occupation_heatmaps(
    sig: SignalData,
    ap: np.ndarray,
    result_fw: FixedWindowResult,
    window_size_s: float,
    t_stable: float,
    t_chatter: float,
    n_bins: int = 20,
) -> None:
    """2D histogram of (x, v) at a stable and a chatter window.

    Visualises what H_xv ≈ low (stable, concentrated orbit) vs. H_xv ≈ high
    (chatter, diffuse orbit) looks like in phase space.

    Parameters
    ----------
    t_stable, t_chatter : float
        Target times for the 'stable' and 'chatter' snapshot windows.
        The nearest window start in result_fw.t_wins is used.
    """
    t = sig.t
    x = sig.displacement
    v = sig.velocity

    def _nearest_window(t_target: float):
        idx   = int(np.argmin(np.abs(result_fw.t_wins - t_target)))
        t_s   = result_fw.t_wins[idx]
        mask  = (t >= t_s) & (t < t_s + window_size_s)
        return x[mask], v[mask], float(t_s)

    for label, t_target in [("stable", t_stable), ("chatter", t_chatter)]:
        x_w, v_w, t_s = _nearest_window(t_target)
        if len(x_w) < 4:
            continue
        H2d, xe, ve = np.histogram2d(x_w, v_w, bins=n_bins)
        cmap_name = "Blues" if label == "stable" else "Oranges"
        _fs = fig_size(scale=2.0)
        fig, ax = plt.subplots(figsize=(_fs[0], _fs[0]))   # square
        im = ax.imshow(
            H2d.T, origin="lower",
            extent=[xe[0], xe[-1], ve[0], ve[-1]],
            aspect="auto", cmap=cmap_name,
        )
        plt.colorbar(im, ax=ax, label="counts")
        ax.set_xlabel(r"$x$  [m]")
        ax.set_ylabel(r"$v$  [m/s]")
        ax.set_title(rf"Phase-space occupation  [{label}]   $t \approx {t_s:.3f}$ s")
        plt.tight_layout()
        plt.show()
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Data acquisition
# ══════════════════════════════════════════════════════════════════════════════

def create_synthetic_example(
    fs: float = 40_000.0,
    duration: float = 2.0,
    f: float = 150.0,
    ap0: float = 5.0,
    ap1: float = 15.0,
    noise_level: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic non-autonomous machining oscillation.

    Mimics a stable-to-chatter transition:

    * ap(t) — linear ramp from ap0 to ap1 (non-autonomous forcing).
    * A(t)  — constant until t_transition, then exponential growth.
    * x(t) = A(t) · cos(2π f t) + noise.
    * v(t) = dx/dt via np.gradient (consistent with discrete x).

    Parameters
    ----------
    fs : float
        Sampling frequency [Hz].
    duration : float
        Signal duration [s].
    f : float
        Modal frequency [Hz].
    ap0, ap1 : float
        Initial and final depth-of-cut [mm].
    noise_level : float
        Std-dev of additive Gaussian noise on x (relative to unit amplitude).
        Set to 0 for a clean signal.

    Returns
    -------
    t, x, v, ap : np.ndarray
    """
    rng          = np.random.default_rng(seed=42)
    dt           = 1.0 / fs
    t            = np.arange(0, duration, dt)
    t_transition = 0.5 * duration
    tau          = 0.3                      # growth time constant [s]

    A_t = np.where(
        t < t_transition,
        1.0,
        1.0 + 3.0 * (1.0 - np.exp(-(t - t_transition) / tau)),
    )
    x = A_t * np.cos(2.0 * np.pi * f * t)
    if noise_level > 0:
        x += rng.normal(0.0, noise_level, size=len(t))
    v  = np.gradient(x, dt)
    ap = np.linspace(ap0, ap1, len(t))
    return t, x, v, ap


def _cut_signal(
    t: np.ndarray,
    x: np.ndarray,
    t_range: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    mask = (t >= t_range[0]) & (t <= t_range[1])
    return t[mask], x[mask]


def load_real_data(case: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load displacement, velocity and ap from HDF5.

    ``ap`` is always user-defined in ``_CASES`` via the ``"ap"`` key:
      - **float**    → constant depth-of-cut for the whole signal.
      - **callable** → ``f(t) -> np.ndarray``, evaluated on the masked time
        vector; use this for variable-ap cases (e.g. cone workpiece).

    Returns
    -------
    t, x, v, ap : np.ndarray
    """
    cfg    = _CASES[case]
    reader = HDF5Reader(cfg["hdf5"])
    raw    = reader.get_element("tool_dyn/data")
    t_full = raw[:, 0]
    x_full = raw[:, 1]
    v_full = reader.get_element("tool_dyn_o/data")[:, 1]

    t0, t1 = cfg["t_range"]
    mask   = (t_full >= t0) & (t_full <= t1)
    t      = t_full[mask]
    x      = x_full[mask]
    v      = v_full[mask]

    ap_spec = cfg["ap"]
    ap = ap_spec(t) if callable(ap_spec) else np.full_like(t, float(ap_spec))
    return t, x, v, ap


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── 1. Signal acquisition ─────────────────────────────────────────────
    if USE_REAL_DATA:
        print(f"[main] Loading real data — case '{_ACTIVE_CASE}' …")
        cfg      = _CASES[_ACTIVE_CASE]
        t, x, v, ap = load_real_data(_ACTIVE_CASE)
        f_modal  = cfg["f_modal"]
        num_T    = cfg["num_T"]
        t_gt     = cfg.get("t_gt")
        train_iv = cfg.get("training_intervals", [])
        use_thr  = cfg.get("use_area_threshold", False)
        sig_name = cfg["name"]
        # Reference times for heatmap snapshots
        if t_gt is not None:
            t_stable  = max(t[0] + 0.1, t_gt - 1.0)
            t_chatter = min(t[-1] - 0.1, t_gt + 1.0)
        else:
            t_stable  = t[0] + (t[-1] - t[0]) * 0.2
            t_chatter = t[0] + (t[-1] - t[0]) * 0.8
    else:
        print("[main] Generating synthetic example …")
        f_modal  = 150.0
        num_T    = 5
        t_gt     = None
        train_iv = []
        use_thr  = False
        sig_name = "synthetic"
        t, x, v, ap = create_synthetic_example(f=f_modal)
        t_stable  = t[0] + 0.2 * (t[-1] - t[0])   # first 20% — stable
        t_chatter = t[0] + 0.8 * (t[-1] - t[0])   # last 20%  — chatter

    # ── 2. Green Integral fixed-window indicator ───────────────────────────
    sig           = SignalData(t=t, displacement=x, velocity=v, name=sig_name)
    window_size_s = float(num_T) / f_modal

    config_fixed = {
        "func": "FixedWindow",
        "params": {
            "f_modal":            f_modal,
            "num_T":              num_T,
            "dt":                 1.0 / f_modal,
            "data_filtrated":     True,
            "lambda_ewma":        None,
            "accumulate":         False,
            "G_memory":           None,
            "sigma_method":       "ratio",
            "sigma_local_n":      5,
            "area_noise_eps":     1e-17,
            "use_area_threshold": use_thr,
            "training_intervals": train_iv,
            "z_sigma":            3.0,
            "debug_level":        1,
        },
    }

    print("[main] Running Green Integral indicator …")
    result_fw  = run_fixed_window(sig, config_fixed)
    n_valid    = int(np.sum(np.isfinite(result_fw.sigma)))
    sigma_mean = float(np.nanmean(result_fw.sigma))
    print(f"  Windows computed : {len(result_fw.areas)}")
    print(f"  Valid σ̂ points   : {n_valid}")
    print(f"  Mean σ̂           : {sigma_mean:.4f} 1/s")
    if result_fw.t_d is not None and t_gt is not None:
        print(f"  t_d (area thr)   : {result_fw.t_d:.4f} s  (t_gt = {t_gt:.5f} s)")
    elif result_fw.t_d is not None:
        print(f"  t_d (area thr)   : {result_fw.t_d:.4f} s")
    else:
        print("  t_d (area thr)   : not detected")

    # ── 3. Enrich windows with 3D geometric descriptors ───────────────────
    print("[main] Computing 3D geometric descriptors …")
    df = enrich_windows(sig, ap, result_fw, window_size_s)
    print(f"  Enriched windows : {len(df)}")

    print("\n── First 10 rows of feature DataFrame ──")
    pd.set_option("display.float_format", "{:.3e}".format)
    print(df.head(10).to_string(index=False))
    pd.reset_option("display.float_format")
    print()

    # Sanity check: A_xv (Green) vs. |A_vec_z| (cross-product).
    # The algebraic identity A_vec[2] = _shoelace(x, v) holds exactly when both
    # are computed on the SAME signal.  The Green indicator applies a bandpass
    # filter internally (data_filtrated=True), so result_fw.areas is computed on
    # the filtered signal while enrich_windows uses the raw sig.displacement.
    # A small residual (~few %) is therefore expected and is NOT a bug.
    if len(df) > 0 and "A_vec_z" in df.columns:
        diff     = (df["A_xv"] - df["A_vec_z"].abs()).abs()
        rel_diff = (diff / df["A_xv"].replace(0, np.nan)).abs()
        print(f"  Sanity |A_xv − |A_vec_z||  max={diff.max():.2e}  mean={diff.mean():.2e}")
        print(f"  Relative diff              max={rel_diff.max():.1%}  mean={rel_diff.mean():.1%}")
        print("  (Non-zero due to internal bandpass filtering in run_fixed_window)")
        print()

    # ── 4. Plots ───────────────────────────────────────────────────────────
    print("[main] Plotting …")

    # Standard Green indicator output (σ̂, Ĝ, threshold)
    plots_fixed_window(
        signal=sig,
        result=result_fw,
        t_gt=t_gt,
        training_intervals=train_iv,
    )

    # 3D augmented trajectory (auto-falls back to 2D when ap is constant)
    stride = max(1, len(t) // 20_000)
    plot_3d_trajectory(
        t[::stride], x[::stride], v[::stride], ap[::stride],
        title=f"Augmented Phase Trajectory — {sig_name}",
        t_gt=t_gt,
    )

    # Time-series: projected and vector areas
    plot_area_features(df, t_gt=t_gt, title_prefix=sig_name)

    # Time-series: arc length, curvature, torsion
    plot_geometry_features(df, t_gt=t_gt, title_prefix=sig_name)

    # Time-series: Shannon entropy (phase-space occupation)
    plot_occupation_features(df, t_gt=t_gt, title_prefix=sig_name)

    # Phase-plane orbit snapshots at 4 evenly-distributed windows
    plot_phase_snapshots(
        sig, ap, result_fw, window_size_s, n_snapshots=4, t_gt=t_gt
    )

    # Occupation heatmaps: stable vs. chatter
    plot_occupation_heatmaps(
        sig, ap, result_fw, window_size_s,
        t_stable=t_stable, t_chatter=t_chatter,
    )

    print("[main] Done.")


if __name__ == "__main__":
    main()

