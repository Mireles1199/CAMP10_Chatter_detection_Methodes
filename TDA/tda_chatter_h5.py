#!/usr/bin/env python3

"""Build TDA outputs per cycle window and store them in HDF5 files.

This script reuses the topology helpers from Topology.py and saves, for each
window, the persistence diagram, lifetime diagram, persistence image, and
basic metadata into one .h5 file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from matplotlib import pyplot as plt
from persim import plot_diagrams

from MaxEnt_SPRT import HDF5Reader
from Topology import time_series_to_diagram, diagram_to_image


def configurar_estilo_global() -> None:
    local_style = {
        'font.family': 'serif',
        'font.size': 9,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 1.25,
        'lines.markersize': 6,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
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
        'mathtext.fontset': 'stix',
        'axes.formatter.use_mathtext': True,
        'legend.frameon': False,
        'legend.loc': 'best',
        'legend.handlelength': 2.0,
        'legend.borderaxespad': 0.5,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'savefig.transparent': True,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    }
    plt.rcParams.update(local_style)


def fig_size(scale: float = 1.0, ncols: int = 1, base_width: float = 3.4) -> tuple[float, float]:
    width = base_width * ncols * scale
    height = width * 0.70
    return (width, height)


configurar_estilo_global()


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
        disp = np.asarray(data.get_element(disp_path_hdf5), dtype=float)
        vel = np.asarray(data.get_element(vel_path_hdf5), dtype=float)
        if disp.ndim == 2:
            disp = disp[:, 1]
        if vel.ndim == 2:
            vel = vel[:, 1]
    else:
        tool_dyn = np.asarray(data.get_element(disp_path_hdf5), dtype=float)
        t = tool_dyn[:, 0]
        disp = tool_dyn[:, 1]
        vel = np.asarray(data.get_element(vel_path_hdf5), dtype=float)[:, 1]

    return {"t": t, "disp": disp, "vel": vel}


def cut_signal(t: np.ndarray, x: np.ndarray, start_time: float, end_time: float) -> tuple[np.ndarray, np.ndarray]:
    mask = (t >= start_time) & (t <= end_time)
    return t[mask], x[mask]


def window_by_cycles(
    t: np.ndarray,
    x: np.ndarray,
    frequency_hz: float,
    n_cycles: int,
    step_cycles: int,
) -> list[dict[str, np.ndarray | float]]:
    if t.size == 0 or x.size == 0 or t.size != x.size:
        return []
    if frequency_hz <= 0 or n_cycles <= 0 or step_cycles <= 0:
        return []

    cycle_period = 1.0 / frequency_hz
    window_size_s = n_cycles * cycle_period
    step_size_s = step_cycles * cycle_period

    windows: list[dict[str, np.ndarray | float]] = []
    current_start = float(t[0])
    end_time = float(t[-1])

    while current_start + window_size_s <= end_time:
        current_end = current_start + window_size_s
        mask = (t >= current_start) & (t < current_end)
        t_win = t[mask]
        x_win = x[mask]
        if t_win.size > 0:
            windows.append(
                {
                    "t_start": current_start,
                    "t_end": current_end,
                    "t": t_win,
                    "x": x_win,
                }
            )
        current_start += step_size_s

    return windows


def split_diagram_by_persistence(dgm: np.ndarray, min_persistence: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    if dgm.size == 0 or dgm.ndim != 2 or dgm.shape[1] < 2:
        empty = np.empty((0, 2), dtype=float)
        return empty, empty

    persistence = dgm[:, 1] - dgm[:, 0]
    valid_mask = persistence > min_persistence
    diagonal_mask = persistence <= 0.0
    return dgm[valid_mask], dgm[diagonal_mask]


def lifetime_from_diagram(dgm: np.ndarray) -> np.ndarray:
    if dgm.size == 0:
        return np.empty((0, 2), dtype=float)
    return np.column_stack((dgm[:, 0], dgm[:, 1] - dgm[:, 0]))


def save_window_png(
    output_path: Path,
    window_info: dict[str, np.ndarray | float],
    dgm: np.ndarray,
    img: np.ndarray,
    diagonal_dgm: np.ndarray | None = None,
) -> None:
    fig, axs = plt.subplot_mosaic(
        [["Time Series", "Persistence Diagram", "Lifetime Diagram", "Persistence Image"]],
        figsize=(16, 4),
    )
    for title, ax in axs.items():
        ax.set_title(title)

    t_vals = np.asarray(window_info["t"], dtype=float)
    x_vals = np.asarray(window_info["x"], dtype=float)
    axs["Time Series"].plot(t_vals, x_vals, lw=1.0)

    if dgm.size > 0:
        plot_diagrams(dgm, ax=axs["Persistence Diagram"], show=False)
        plot_diagrams(dgm, ax=axs["Lifetime Diagram"], lifetime=True, show=False)
        if diagonal_dgm is not None and diagonal_dgm.size > 0:
            axs["Persistence Diagram"].scatter(
                diagonal_dgm[:, 0],
                diagonal_dgm[:, 1],
                c="red",
                s=18,
                marker="o",
                edgecolors="none",
                zorder=5,
            )
            axs["Lifetime Diagram"].scatter(
                diagonal_dgm[:, 0],
                diagonal_dgm[:, 1] - diagonal_dgm[:, 0],
                c="red",
                s=18,
                marker="o",
                edgecolors="none",
                zorder=5,
            )
    else:
        axs["Persistence Diagram"].set_axis_off()
        axs["Lifetime Diagram"].set_axis_off()

    im = axs["Persistence Image"].imshow(
        img,
        cmap="viridis",
        origin="lower",
        interpolation="nearest",
    )
    axs["Persistence Image"].axis("off")
    fig.colorbar(im, ax=axs["Persistence Image"], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_window_h5(
    output_path: Path,
    window_info: dict[str, np.ndarray | float],
    dgm: np.ndarray,
    img: np.ndarray,
    frequency_hz: float,
    n_cycles: int,
    step_cycles: int,
) -> None:
    lifetime = lifetime_from_diagram(dgm)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5f:
        h5f.create_dataset("time", data=np.asarray(window_info["t"], dtype=float))
        h5f.create_dataset("signal", data=np.asarray(window_info["x"], dtype=float))
        h5f.create_dataset("persistence_diagram", data=np.asarray(dgm, dtype=float))
        h5f.create_dataset("lifetime_diagram", data=np.asarray(lifetime, dtype=float))
        h5f.create_dataset("persistence_image", data=np.asarray(img, dtype=float))

        h5f.attrs["t_start"] = float(window_info["t_start"])
        h5f.attrs["t_end"] = float(window_info["t_end"])
        h5f.attrs["frequency_hz"] = frequency_hz
        h5f.attrs["n_cycles"] = n_cycles
        h5f.attrs["step_cycles"] = step_cycles


def process_signal(
    signal_name: str,
    t: np.ndarray,
    x: np.ndarray,
    data_dir: Path,
    plots_dir: Path,
    cut_start: float,
    cut_end: float,
    frequency_hz: float,
    n_cycles: int,
    step_cycles: int,
    min_persistence: float,
    window_index: int | None = None,
    save_all_windows: bool = True,
) -> None:
    t_cut, x_cut = cut_signal(t, x, cut_start, cut_end)
    windows = window_by_cycles(
        t_cut,
        x_cut,
        frequency_hz=frequency_hz,
        n_cycles=n_cycles,
        step_cycles=step_cycles,
    )

    sig_data_dir = data_dir / signal_name
    sig_plots_dir = plots_dir / signal_name
    sig_data_dir.mkdir(parents=True, exist_ok=True)
    sig_plots_dir.mkdir(parents=True, exist_ok=True)

    if window_index is not None:
        if window_index < 1 or window_index > len(windows):
            print(f"{signal_name}: window {window_index} is out of range (1..{len(windows)})")
            return
        windows_to_process = [windows[window_index - 1]]
        start_index = window_index
    else:
        windows_to_process = windows
        start_index = 1

    # --- Primera pasada: calcular todos los diagramas e imágenes ---
    computed: list[tuple[int, dict, np.ndarray, np.ndarray, np.ndarray]] = []
    for offset, window_info in enumerate(windows_to_process):
        index = start_index + offset
        dgm = time_series_to_diagram(np.asarray(window_info["x"], dtype=float))
        dgm, diagonal_dgm = split_diagram_by_persistence(dgm, min_persistence=min_persistence)
        img = diagram_to_image(dgm)
        computed.append((index, window_info, dgm, diagonal_dgm, img))

    # --- Segunda pasada: guardar ---
    for index, window_info, dgm, diagonal_dgm, img in computed:
        h5_path = sig_data_dir / f"window_{index:04d}.h5"
        png_path = sig_plots_dir / f"window_{index:04d}.png"

        if save_all_windows or window_index is not None:
            save_window_h5(h5_path, window_info, dgm, img, frequency_hz, n_cycles, step_cycles)
        save_window_png(
            png_path, window_info, dgm, img,
            diagonal_dgm=diagonal_dgm,
        )

    print(f"{signal_name}: {len(windows_to_process)} windows → data={sig_data_dir} | plots={sig_plots_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TDA outputs per cycle window and store them in HDF5 files.")
    parser.add_argument("--window-index", type=int, default=None, help="Process only one window index (1-based).")
    parser.add_argument("--preview-only", action="store_true", help="Render only a preview PNG for the selected window, without saving HDF5 files.")
    parser.add_argument("--min-persistence", type=float, default=0.0, help="Discard persistence points with death-birth <= this value.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = Path(
        r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage"
        r"\2DOF_Cone_DOE\DOE_Influence_dexel_RPM_12000_ftooth_005_dt_200"
        r"\3\1DOF_150Hz\sens_out.hdf5"
    )

    # data_dir = Path(
    #     r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage"
    #     r"\2DOF_Cone_DOE\DOE_Influence_dexel_RPM_12000_ftooth_005_dt_200_AP_9mm"
    #     r"\3\1DOF_150Hz\sens_out.hdf5"
    # )
    
    base_output_dir = Path(__file__).resolve().parent / "tda_chatter"

    input_case_name = None
    case_name = "DOE_Influence_dexel_RPM_12000_ftooth_005_dt_200"
    # case_name   = "DOE_Influence_dexel_RPM_12000_ftooth_005_dt_200_AP_9mm"
    cut_start = 0.0
    cut_end = 16
    window_frequency_hz = 150.0
    window_n_cycles = 7
    window_step_cycles = 1

    # Configuración local dentro del script.
    # - None: procesa todas las ventanas.
    # - 1, 2, 3, ...: procesa solo esa ventana.
    window_index = None

    # True: procesa solo la ventana 1 (o la indicada en window_index) y guarda solo PNG.
    # False: procesa todas las ventanas y guarda HDF5 + PNG.
    preview_only = False

    # Umbral de persistencia mínima. Puntos con (death - birth) <= min_persistence
    # se eliminan del diagrama de persistencia (y se muestran en rojo si death <= birth).
    # 0.0 = sin filtro; valores típicos: 0.001, 0.01, 0.05
    min_persistence = 0.0

    # Overrides desde CLI (tienen prioridad SOLO si el valor del script es 0.0)
    if args.window_index is not None:
        window_index = args.window_index
    if args.preview_only:
        preview_only = True
    if min_persistence == 0.0 and args.min_persistence != 0.0:
        min_persistence = args.min_persistence

    # Si preview_only y no se eligió ventana, forzar la ventana 1
    if preview_only and window_index is None:
        window_index = 1

    data_output_dir = base_output_dir / case_name / "data"
    plots_output_dir = base_output_dir / case_name / "plots"

    print("TDA HDF5 windowing run")
    print(f"Input file: {data_dir}")
    print(f"Case name: {case_name}")
    print(f"Data dir: {data_output_dir}")
    print(f"Plots dir: {plots_output_dir}")
    print(f"Cut range: {cut_start} .. {cut_end} s")
    print(f"Window frequency: {window_frequency_hz} Hz")
    print(f"Window size: {window_n_cycles} cycles")
    print(f"Window step: {window_step_cycles} cycles")

    signals = load_signals(data_dir, input_case_name)
    for sig_name, sig_arr in [("disp", signals["disp"]), ("vel", signals["vel"])]:
        process_signal(
            sig_name,
            signals["t"],
            sig_arr,
            data_output_dir,
            plots_output_dir,
            cut_start,
            cut_end,
            window_frequency_hz,
            window_n_cycles,
            window_step_cycles,
            min_persistence,
            window_index=window_index,
            save_all_windows=not preview_only,
        )


if __name__ == "__main__":
    main()