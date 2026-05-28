"""Green Integral Method — example using real HDF5 data.

Usage
-----
    cd indicators/green_integral
    pip install -e .
    python examples/Green_Integral_Detection_NEW.py

Toggle
------
    Set USE_FIXED_WINDOW = True  to run the no-clustering fixed-window variant.
    Set USE_FIXED_WINDOW = False to run the original clustering-based indicator.
"""

from typing import Tuple
import os
import sys
import pathlib
import numpy as np


# Allow running directly without installing (adds src/ to path)
_here = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

# ── Logging ────────────────────────────────────────────────────────────────
from green_integral.logging_setup import configure_logging, LOGGING_LEVELS

configure_logging(level=LOGGING_LEVELS["info"])

# ── Public API ─────────────────────────────────────────────────────────────
from green_integral import (
    HDF5Reader,
    SignalData,
    run_green_integral,
    plots_green_integral,
    run_fixed_window,
    plots_fixed_window,
    plots_signal_diagnostics,
    INFO_PLUS_LEVEL,
    INDICATOR_CONFIG,
)

# ── Toggle ─────────────────────────────────────────────────────────────────
USE_FIXED_WINDOW: bool = True   # True → fixed-window (no clustering)
                                  # False → original clustering indicator

# -- helpers ------------------------------------------------------------------
def _cut_signal(t, x, time_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    start_time, end_time = time_range
    mask = (t >= start_time) & (t <= end_time)
    return t[mask], x[mask]

# ── Load signal from HDF5 ──────────────────────────────────────────────────
# -- datos --------------------------------------------------------------------
dir_cono     = r"D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\2DOF_Cono\1DOF_150Hz"
work_space_5mm   = 'D:/Thesis/03-Code_Storage/02-Altintlas_Nessy2m_Storage/Chatter-Criteria/CAMP8-Ventanna_Glisante/Nessy2m_Case_Test_Explicit/1DOF_150Hz_5mm/1DOF_150Hz'

dir_path_use = dir_cono

data_dir = os.path.abspath(os.path.join(dir_path_use, "out.hdf5"))
data     = HDF5Reader(data_dir)

tool_dyn     = data.get_element("tool_dyn/data")
t            = tool_dyn[:, 0]
tool_dyn     = tool_dyn[:, 1]
tool_dyn_vel = data.get_element("tool_dyn_o/data")[:, 1]
force_N      = data.get_element("res_R_p/data")[:, 1]

v  = tool_dyn_vel
fs = 1.0 / (t[1] - t[0])

_CUT_START = 0.05

t_cut, v_cut  = _cut_signal(t, v,        (_CUT_START, 16))
_,     x_cut  = _cut_signal(t, tool_dyn, (_CUT_START, 16))
_,     f_cut  = _cut_signal(t, force_N,  (_CUT_START, 16))
# ── Build SignalData ────────────────────────────────────────────────────────
sig = SignalData(
    t=t_cut,
    displacement=x_cut,
    velocity=v_cut,
    name="cono",
)

# Ground-truth chatter onset (used for training_intervals)
_T_GT = 5.36577  # [s]  — onset anotado del cono

# ── Indicator configuration ─────────────────────────────────────────────────
# Original clustering-based indicator — kept unchanged
config = {
    "func": "Default",
    "params": {
        "f_modal": 200.0,    # Hz — adjust to the actual modal frequency
        "num_T": 4,
        "dt": 0.005,
        "data_filtrated": True,
        "hilbert": False,
        "while_loop_extend": False,
        "cycles_cluster_points": 35,
        "thein_sen": False,
        # --- mu ± 3sigma threshold ---
        "use_area_threshold": True,
        "training_intervals": [
            (0.05,   _T_GT, "stable"),
            (_T_GT,  10.0,  "chatter"),
        ],
        "z_sigma": 3.0,
        # --- debug ---
        "debug_level": 1,
        "debug_window_range": (10, 15),
        "save_figures_windows": False,
        "work_space": None,
    },
}

# Fixed-window indicator (no clustering)
config_fixed = {
    "func": "FixedWindow",
    "params": {
        "f_modal":        200.0,   # Hz
        "num_T":          16,       # window = 1 × T_modal
        "dt":             1./200.,    # None → non-overlapping; float [s] → custom step
        "data_filtrated": True,
        "lambda_ewma":    None,     # None to disable EWMA
        "accumulate":     False,    # False/None to disable Ĝ accumulation (from t=0)
        "G_memory":       None,    # float [s] → sliding Ĝ over last G_memory seconds
        "sigma_method":   "ratio", # "ratio" or "frozen_time"
        "sigma_local_n":  5,
        "area_noise_eps": 1e-13,        # --- mu ± 3sigma threshold ---
        "use_area_threshold": True,
        "training_intervals": [
                (_CUT_START,3.3,    "stable_1"),  # chatter-free training region
                (3.3, 4.46, "stable_2"), # stable training region
                (4.46, _T_GT, "stable_1"), #
        ],
        "z_sigma": 3.0,        "debug_level":    1,
    },
}



# ── Run indicator ───────────────────────────────────────────────────────────
if not USE_FIXED_WINDOW:
    result = run_green_integral(sig, config)

    print(f"\nMediana delta_n : {result.Mediana_delta_n:.4f}")
    print(
        f"Interpretation  : {'UNSTABLE (chatter)' if result.Mediana_delta_n < 0 else 'STABLE'}"
    )
    print(f"Windows analysed: {len(result.data_window)}")
    if result.t_d is not None:
        print(f"t_d (area thr)  : {result.t_d:.4f} s  (t_gt = {_T_GT:.5f} s)")
    else:
        print("t_d (area thr)  : not detected (or threshold disabled)")

    plots_green_integral(signal=sig, result=result)

else:
    result_fw = run_fixed_window(sig, config_fixed)

    n_valid = int(np.sum(np.isfinite(result_fw.sigma)))
    sigma_mean = float(np.nanmean(result_fw.sigma))
    print(f"\nWindows computed: {len(result_fw.areas)}")
    print(f"Valid σ̂ points  : {n_valid}")
    print(f"Mean σ̂          : {sigma_mean:.4f} 1/s")
    if result_fw.G_hat.size > 0:
        G_final = float(result_fw.G_hat[-1])
        print(f"Ĝ final         : {G_final:.4f}")
        print(
            f"Interpretation  : {'UNSTABLE (chatter)' if G_final > 0 else 'STABLE'}"
        )
    else:
        print(
            f"Interpretation  : {'UNSTABLE (chatter)' if sigma_mean > 0 else 'STABLE'}"
        )
    if result_fw.t_d is not None:
        print(f"t_d (area thr)  : {result_fw.t_d:.4f} s  (t_gt = {_T_GT:.5f} s)")
    else:
        print("t_d (area thr)  : not detected (or threshold disabled)")

    plots_fixed_window(
        signal=sig,
        result=result_fw,
        t_gt=_T_GT,
        training_intervals=config_fixed["params"]["training_intervals"],
    )
    # plots_signal_diagnostics(
    #     signal=sig,
    #     result=result_fw,
    #     stable_range=(0.5, 5.0),   # zona estable del cono
    #     zoom_range=(6.6,8),     # zoom 200 ms para ver la señal
    #     eq_smooth_s=0.050,         # 50 ms → moving avg para x_eq
    # )

