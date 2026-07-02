#!/usr/bin/env python3

"""Create compact summary figures from per-window TDA HDF5 outputs.

Reads the .h5 files produced by tda_chatter_h5.py from:
  tda_chatter/data/<case_name>/<signal>/window_XXXX.h5

Writes block summary PNGs to:
  tda_chatter/summary/<case_name>/<signal>/block_XXXX.png
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from matplotlib import pyplot as plt
from persim import plot_diagrams

from Topology import diagram_to_image

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent / "tda_chatter"
# ──────────────────────────────────────────────────────────────────────────────


def list_window_files(signal_dir: Path) -> list[Path]:
    return sorted(signal_dir.glob("window_*.h5"))


def load_window_file(window_path: Path) -> dict[str, np.ndarray | float]:
    with h5py.File(window_path, "r") as h5f:
        time = np.asarray(h5f["time"], dtype=float)
        signal = np.asarray(h5f["signal"], dtype=float)
        diagram = np.asarray(h5f["persistence_diagram"], dtype=float)
        lifetime = np.asarray(h5f["lifetime_diagram"], dtype=float)
        return {
            "time": time,
            "signal": signal,
            "diagram": diagram,
            "lifetime": lifetime,
            "t_start": float(h5f.attrs.get("t_start", time[0] if time.size else 0.0)),
            "t_end": float(h5f.attrs.get("t_end", time[-1] if time.size else 0.0)),
        }


def chunk_items(items: list[Path], block_size: int) -> list[list[Path]]:
    return [items[i : i + block_size] for i in range(0, len(items), block_size)]


def _plot_single_row(fig: plt.Figure, axs: dict, wd: dict, is_first_row: bool) -> None:
    """Render one window row using the same layout as tda_chatter_h5.py's save_window_png."""
    img = diagram_to_image(wd["diagram"])
    dgm = wd["diagram"]

    # Separate diagonal points (persistence <= 0)
    if dgm.size > 0 and dgm.ndim == 2 and dgm.shape[1] >= 2:
        persistence = dgm[:, 1] - dgm[:, 0]
        valid_dgm = dgm[persistence > 0.0]
        diag_dgm = dgm[persistence <= 0.0]
    else:
        valid_dgm = np.empty((0, 2), dtype=float)
        diag_dgm = np.empty((0, 2), dtype=float)

    if is_first_row:
        axs["Time Series"].set_title("Time Series")
        axs["Persistence Diagram"].set_title("Persistence Diagram")
        axs["Lifetime Diagram"].set_title("Lifetime Diagram")
        axs["Persistence Image"].set_title("Persistence Image")

    # Time series
    axs["Time Series"].plot(wd["time"], wd["signal"], lw=1.0)
    axs["Time Series"].set_ylabel(f"t={wd['t_start']:.2f}s", fontsize=8, rotation=0, ha="right", va="center")

    # Persistence & lifetime diagrams
    if valid_dgm.size > 0:
        plot_diagrams(valid_dgm, ax=axs["Persistence Diagram"], show=False)
        plot_diagrams(valid_dgm, ax=axs["Lifetime Diagram"], lifetime=True, show=False)
    else:
        axs["Persistence Diagram"].set_axis_off()
        axs["Lifetime Diagram"].set_axis_off()

    # Diagonal points in red
    if diag_dgm.size > 0:
        axs["Persistence Diagram"].scatter(diag_dgm[:, 0], diag_dgm[:, 1], c="red", s=18, marker="o", edgecolors="none", zorder=5)
        axs["Lifetime Diagram"].scatter(diag_dgm[:, 0], diag_dgm[:, 1] - diag_dgm[:, 0], c="red", s=18, marker="o", edgecolors="none", zorder=5)

    # Persistence image
    im = axs["Persistence Image"].imshow(img, cmap="viridis", origin="lower", interpolation="nearest")
    axs["Persistence Image"].axis("off")
    fig.colorbar(im, ax=axs["Persistence Image"], fraction=0.046, pad=0.04)


def plot_block(
    signal_name: str,
    block_index: int,
    window_files: list[Path],
    output_path: Path,
    block_size: int = 10,
) -> None:
    n_windows = len(window_files)
    ROW_HEIGHT = 4  # same height as individual window figures (figsize=(16,4))

    fig, all_axs = plt.subplots(
        n_windows,
        4,
        figsize=(16, ROW_HEIGHT * n_windows),
        squeeze=False,
    )
    # Build per-row axs dicts with same keys as subplot_mosaic
    keys = ["Time Series", "Persistence Diagram", "Lifetime Diagram", "Persistence Image"]

    for row, wf in enumerate(window_files):
        wd = load_window_file(wf)
        row_axs = {keys[col]: all_axs[row, col] for col in range(4)}
        _plot_single_row(fig, row_axs, wd, is_first_row=(row == 0))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summarize_signal(signal_dir: Path, summary_case_dir: Path, block_size: int = 10) -> None:
    window_files = list_window_files(signal_dir)
    if not window_files:
        print(f"  {signal_dir.name}: no .h5 files found")
        return

    blocks = chunk_items(window_files, block_size)
    out_dir = summary_case_dir / signal_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    for block_index, block_files in enumerate(blocks):
        start_label = block_index * block_size + 1
        end_label = start_label + len(block_files) - 1
        out_path = out_dir / f"block_{start_label:04d}_{end_label:04d}.png"
        plot_block(signal_dir.name, block_index, block_files, out_path, block_size=block_size)

    print(f"  {signal_dir.name}: {len(blocks)} block(s) → {out_dir}")


def main() -> None:
    # ── Configuración ─────────────────────────────────────────────────────────
    # Debe coincidir con el case_name usado en tda_chatter_h5.py
    case_name = "DOE_Influence_dexel_RPM_12000_ftooth_005_dt_200"
    # case_name   = "DOE_Influence_dexel_RPM_12000_ftooth_005_dt_200_AP_9mm"

    # Número de ventanas por figura resumen
    block_size = 10
    # ──────────────────────────────────────────────────────────────────────────

    data_case_dir = BASE_DIR / case_name / "data"
    summary_case_dir = BASE_DIR / case_name / "summary"

    if not data_case_dir.exists():
        print(f"Data directory not found: {data_case_dir}")
        return

    signal_dirs = sorted(p for p in data_case_dir.iterdir() if p.is_dir())
    if not signal_dirs:
        print(f"No signal folders found in {data_case_dir}")
        return

    print(f"Case: {case_name}")
    for signal_dir in signal_dirs:
        summarize_signal(signal_dir, summary_case_dir, block_size)


if __name__ == "__main__":
    main()
