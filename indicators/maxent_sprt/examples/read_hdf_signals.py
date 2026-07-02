#!/usr/bin/env python3

"""Minimal script to read the HDF5 signals used by the MaxEnt examples.

This script only loads the HDF5 file, resolves the signal paths, and prints a
small summary so you can reuse the loaded arrays in later scripts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from MaxEnt_SPRT import HDF5Reader


# -----------------------------------------------------------------------------
# Input data
# -----------------------------------------------------------------------------
data_dir = Path(
    r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage"
    r"\2DOF_Cone_DOE\DOE_Influence_dexel_RPM_12000_ftooth_005_dt_200"
    r"\0\1DOF_150Hz\sens_out.hdf5"
)

# Set this to a case folder name like "3" if you need case-prefixed paths.
CASE_NAME = None


def cut_signal(t: np.ndarray, x: np.ndarray, start_time: float, end_time: float) -> tuple[np.ndarray, np.ndarray]:
    mask = (t >= start_time) & (t <= end_time)
    return t[mask], x[mask]


def resolve_paths(case_name: str | None) -> tuple[str, str, str]:
    case_prefix = f"{case_name}/" if case_name else ""
    disp_path_hdf5 = f"{case_prefix}Axial_disp/values" if case_name else "Axial_disp/data"
    vel_path_hdf5 = f"{case_prefix}Axial_vel/values" if case_name else "Axial_vel/data"
    time_path_hdf5 = f"{case_prefix}Axial_disp/time" if case_name else "Axial_disp/data"
    return disp_path_hdf5, vel_path_hdf5, time_path_hdf5


def load_signals(hdf5_path: Path, case_name: str | None = None) -> dict[str, np.ndarray]:
    data = HDF5Reader(str(hdf5_path))
    disp_path_hdf5, vel_path_hdf5, time_path_hdf5 = resolve_paths(case_name)

    if case_name is not None:
        t = np.asarray(data.get_element(time_path_hdf5), dtype=float)
        x = np.asarray(data.get_element(disp_path_hdf5), dtype=float)
        v = np.asarray(data.get_element(vel_path_hdf5), dtype=float)
        if x.ndim == 2:
            x = x[:, 1]
        if v.ndim == 2:
            v = v[:, 1]
    else:
        tool_dyn = np.asarray(data.get_element(disp_path_hdf5), dtype=float)
        t = tool_dyn[:, 0]
        x = tool_dyn[:, 1]
        v = np.asarray(data.get_element(vel_path_hdf5), dtype=float)[:, 1]

    try:
        force_n = np.asarray(data.get_element("force_N/data"), dtype=float)[:, 1]
    except KeyError:
        force_n = np.zeros_like(t)

    return {
        "t": t,
        "disp": x,
        "vel": v,
        "force_N": force_n,
    }


def main() -> None:
    signals = load_signals(data_dir, CASE_NAME)

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

    print(f"HDF5: {data_dir}")
    print(f"Samples: {t.size}")
    print(f"fs: {fs:.3f} Hz")
    print(f"Cut range: {cut_start} to {cut_end} s")
    print(f"Cut samples: {t_cut.size}")
    print(f"disp range: {np.min(disp_cut):.6g} .. {np.max(disp_cut):.6g}")
    print(f"vel range: {np.min(vel_cut):.6g} .. {np.max(vel_cut):.6g}")
    print(f"force range: {np.min(force_cut):.6g} .. {np.max(force_cut):.6g}")


if __name__ == "__main__":
    main()