"""Ad-hoc before/after verification for the sigma-scaling fix in
runner_lyapunov.py::_estimate_sigma (ratio method) -- not committed.

Reproduces the active example config's windowing shape (overlapping windows,
hop < window duration, same ratio as by_revolution N_rev_window=4/step_rev=1)
with a synthetic signal, and checks:

1. `areas` and `t_d` are BYTE-IDENTICAL between the OLD formula
   (sigma = dlogA / (2*T_window)) and the NEW one
   (sigma = dlogA / (2*dt_real)) -- they must be, since t_d only depends on
   `areas` via the mu+z*sigma threshold, never on `sigma` itself.
2. `sigma` differs, and by the expected factor (T_window / hop).
"""
import sys
import pathlib
import numpy as np

_here = pathlib.Path(__file__).resolve().parent / "src"
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from green_integral.logging_setup import configure_logging, LOGGING_LEVELS
configure_logging(level=LOGGING_LEVELS["warning"])

from green_integral import StdSignalData, run_green_std
from green_integral.lib import runner_lyapunov as rl

fs = 5000.0
f_modal = 100.0          # T_modal = 0.01 s
T_MODAL = 1.0 / f_modal
NUM_T = 4                # T_window = 4 * T_modal = 0.04 s -- same as N_rev_window=4
DT = T_MODAL             # hop = 1 * T_modal            -- same as step_rev=1.0
HOP_TO_WINDOW_RATIO = DT / (NUM_T * T_MODAL)  # 0.25, matches step_rev/N_rev_window


def _make_signal(t, sigma_of_t):
    dt = 1.0 / fs
    A = np.exp(np.cumsum(sigma_of_t(t)) * dt)
    rng = np.random.default_rng(0)
    return A * np.sin(2.0 * np.pi * f_modal * t) + 1e-4 * rng.standard_normal(len(t))


t_an = np.arange(0.0, 1.0, 1.0 / fs)
T_ONSET = 0.4
x_an = _make_signal(t_an, lambda t: np.where(t < T_ONSET, 0.0, 3.0))
sig = StdSignalData(t_analysis=t_an, signal_analysis=x_an, path="analysis", fs=fs)

# Same as the active example case: beta=False, norm="none", use_area_threshold=True.
CFG = dict(
    f_modal=f_modal, num_T=NUM_T, dt=DT,
    data_filtrated=True, use_area_threshold=True, z_sigma=3.0,
    sigma_method="ratio",
    training_intervals=[(0.0, T_ONSET, "stable")],
    use_zero_crossing_cycles=True,
    use_beta_from_cycles=False,
    cycle_area_norm="none",
)


def _run():
    result = run_green_std(sig, {"func": "Lyapunov", "param_mode": "native", "params": dict(CFG)})
    raw = result.meta["raw_result"]
    return raw


# ---- run with the CURRENT (fixed) _estimate_sigma ---------------------------
raw_new = _run()

# ---- run with the OLD formula, monkeypatched in for this call only ----------
_orig_estimate_sigma = rl._estimate_sigma


def _old_estimate_sigma(areas, t_wins, T_window, eps, method, local_n):
    A = np.where(areas > eps, areas, np.nan)
    sigma = np.full(len(A), np.nan)
    if method.strip().lower() == "ratio":
        log_A = np.log(A)
        sigma[1:] = (log_A[1:] - log_A[:-1]) / (2.0 * T_window)
    else:
        return _orig_estimate_sigma(areas, t_wins, T_window, eps, method, local_n)
    return sigma


rl._estimate_sigma = _old_estimate_sigma
try:
    raw_old = _run()
finally:
    rl._estimate_sigma = _orig_estimate_sigma

# ---- 1. areas / t_d must be byte-identical -----------------------------------
assert np.array_equal(raw_new.areas, raw_old.areas, equal_nan=True), "areas changed!"
t_d_new = np.asarray(raw_new.t_d) if raw_new.t_d is not None else np.array([])
t_d_old = np.asarray(raw_old.t_d) if raw_old.t_d is not None else np.array([])
assert np.array_equal(t_d_new, t_d_old), (t_d_new, t_d_old, "t_d changed!")
assert t_d_new.size > 0, "test setup should produce a real detection"
print(f"OK: areas identical ({raw_new.areas.size} windows), "
      f"t_d identical ({t_d_new[0]:.5f} s)")

# ---- 2. sigma differs, by ~ T_window / hop = 1/0.25 = 4x ---------------------
valid = np.isfinite(raw_new.sigma) & np.isfinite(raw_old.sigma) & (raw_old.sigma != 0)
ratio = raw_new.sigma[valid] / raw_old.sigma[valid]
expected_ratio = 1.0 / HOP_TO_WINDOW_RATIO  # T_window / dt_real = 4.0
print(f"sigma ratio (new/old): median={np.median(ratio):.4f}, "
      f"expected={expected_ratio:.4f} (T_window/hop)")
assert np.allclose(ratio, expected_ratio, rtol=1e-6), (
    f"expected sigma to scale by {expected_ratio}x, got median ratio {np.median(ratio)}"
)
print(f"OK: sigma scales by the expected factor ({expected_ratio:.2f}x, "
      f"T_window/hop with N_rev_window=4/step_rev=1 -- same shape as "
      f"lyapunov_by_revolution's active config)")
