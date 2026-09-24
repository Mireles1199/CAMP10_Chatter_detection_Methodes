"""Smoke test for the three param_mode branches of run_maxent_sprt / the
physical-parameter resolver, covering the bugs fixed in this session:
NameError on param_mode="native", f_cycle derived from param_mode (not from
which key is present), T_rev optional in by_modal, and integer enforcement
in segmentation="opr" vs. fractional acceptance in "raw"."""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from MaxEnt_SPRT import SignalData, run_maxent_sprt
from MaxEnt_SPRT.lib.runner import _resolve_physical_params_maxent

rng = np.random.default_rng(0)
fs = 2000.0
t = np.arange(0, 4.0, 1.0 / fs)
x = rng.normal(0.0, 1.0, size=t.size)
sig = SignalData(t_analysis=t, signal_analysis=x, path="synthetic", fs=fs)

_COMMON = {
    "t_stable_total": 2.0, "alpha": 0.05, "beta": 0.05, "reset_on_H0": True,
    "cut_start_time": 0.0, "cut_end_time": 4.0, "t_theorical": 2.0,
}

# 1) native mode: must not raise NameError, t_d must be an ndarray.
res_native = run_maxent_sprt(sig, {
    "func": "Default",
    "params": {"rpm": 6000.0, "N_seg": 5, **_COMMON},
})
assert isinstance(res_native.t_d, np.ndarray), "native: t_d must be ndarray"

# 2) by_modal without T_rev: must run, and f_cycle must come from T_modal.
T_modal = 0.02  # 50 Hz
res_modal = run_maxent_sprt(sig, {
    "func": "Default",
    "param_mode": "by_modal",
    "params_physical": {
        "T_modal": T_modal, "N_modal_window": 5, "step_modal": 1, **_COMMON,
    },
})
assert isinstance(res_modal.t_d, np.ndarray), "by_modal: t_d must be ndarray"
assert abs(res_modal.meta["f_cycle"] - 1.0 / T_modal) < 1e-9, (
    f"by_modal: f_cycle should be 1/T_modal={1/T_modal}, got {res_modal.meta['f_cycle']}"
)

# 3) opr + fractional window -> ValueError (never silently truncated).
try:
    _resolve_physical_params_maxent(
        "by_revolution",
        {"T_rev": 0.01, "N_rev_window": 2.5, "step_rev": 1},
        fs,
    )
    raise AssertionError("opr with fractional N_rev_window should raise ValueError")
except ValueError:
    pass

# 4) raw + fractional window -> accepted.
native_raw, trace_raw = _resolve_physical_params_maxent(
    "by_revolution",
    {"T_rev": 0.01, "N_rev_window": 2.5, "step_rev": 1, "segmentation": "raw"},
    fs,
)
assert "N_samples_per_seg" in native_raw, "raw mode should resolve N_samples_per_seg"

# 5) missing step -> ValueError.
try:
    _resolve_physical_params_maxent(
        "by_revolution", {"T_rev": 0.01, "N_rev_window": 5}, fs,
    )
    raise AssertionError("missing step_rev should raise ValueError")
except ValueError:
    pass

print("All assertions passed")
