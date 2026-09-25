#!/usr/bin/env python3

"""Minimal script to read the HDF5 signals used by the MaxEnt examples.

This script only loads the HDF5 file, resolves the signal paths, and prints a
small summary so you can reuse the loaded arrays in later scripts.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# -- path setup -----------------------------------------------------------
# Prefer the local (worktree) src/ over whatever MaxEnt_SPRT is installed
# editable-mode against, which may point at a different checkout/worktree.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from MaxEnt_SPRT import HDF5Reader, load_signal


# -----------------------------------------------------------------------------
# Input data -- see COMMON_TEMPLATE.md for the SIGNAL_SOURCE convention.
# -----------------------------------------------------------------------------
SIGNAL_SOURCE = {
    "hdf5_path": (
        r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage"
        r"\2DOF_Cone_DOE\DOE_Influence_dexel_RPM_12000_ftooth_005_dt_200"
        r"\0\1DOF_150Hz\sens_out.hdf5"
    ),
    "case_name": None,  # None (layout crudo) | "3" / "case_003" (layout DOE)
    "disp_name": "Axial_disp",
    "vel_name": "Axial_vel",
    "force_name": "force_N",
}


def cut_signal(t: np.ndarray, x: np.ndarray, start_time: float, end_time: float) -> tuple[np.ndarray, np.ndarray]:
    mask = (t >= start_time) & (t <= end_time)
    return t[mask], x[mask]


def load_signals(source: dict) -> dict[str, np.ndarray]:
    reader = HDF5Reader(source["hdf5_path"])
    case_name = source.get("case_name")

    t, disp = load_signal(reader, source["disp_name"], case_name)
    _, vel = load_signal(reader, source["vel_name"], case_name)
    try:
        _, force_n = load_signal(reader, source["force_name"], case_name)
    except KeyError:
        force_n = np.zeros_like(t)

    return {
        "t": t,
        "disp": disp,
        "vel": vel,
        "force_N": force_n,
    }


def main() -> None:
    signals = load_signals(SIGNAL_SOURCE)

    t = signals["t"]
    disp = signals["disp"]
    vel = signals["vel"]
    force_n = signals["force_N"]

    cut_start = 0.0
    cut_end = 16.0
    t_cut, vel_cut = cut_signal(t, vel, cut_start, cut_end)
    _, disp_cut = cut_signal(t, disp, cut_start, cut_end)
    _, force_cut = cut_signal(t, force_n, cut_start, cut_end)

    fs = 1.0 / (t[1] - t[0]) if t.size > 1 else float("nan")

    print(f"HDF5: {SIGNAL_SOURCE['hdf5_path']}")
    print(f"Samples: {t.size}")
    print(f"fs: {fs:.3f} Hz")
    print(f"Cut range: {cut_start} to {cut_end} s")
    print(f"Cut samples: {t_cut.size}")
    print(f"disp range: {np.min(disp_cut):.6g} .. {np.max(disp_cut):.6g}")
    print(f"vel range: {np.min(vel_cut):.6g} .. {np.max(vel_cut):.6g}")
    print(f"force range: {np.min(force_cut):.6g} .. {np.max(force_cut):.6g}")


if __name__ == "__main__":
    main()