"""Self-check for the `reference_signal` external-training extension point.

Phase 3 (DOE reference-dataset project): `INDICATOR_CONFIG["reference_signal"]`
lets the mu +- z*sigma area threshold train on an external "stable" signal
instead of an internal `training_intervals` cut. Asserts:

1. Baseline (no reference_signal, no training_intervals): area-threshold
   branch is skipped, exactly like before this feature existed.
2. Internal training_intervals still works (unchanged) and reports
   meta["training_source"] == "internal".
3. reference_signal drives detection and reports
   meta["training_source"] == "external_reference", for both the Default
   and the Lyapunov variant.
4. Passing both at once does not crash; reference_signal wins.
5. The training-population figures (histogram + verification curve, see
   plots.plot_training_distribution) always plot the population that
   actually trained the threshold — not a training_intervals slice of
   whatever signal is being plotted, which would silently show the wrong
   data (or nothing) for a reference_signal-trained result. Titles don't
   name the source (internal vs. external); the linkage is verified via
   global_data["training_areas"]/["training_t_wins"] plus an internal
   assert in plot_training_distribution() that recomputing mu/sigma from
   that exact population reproduces area_mu_3sigma.
"""

from __future__ import annotations
import sys
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from green_integral.logging_setup import configure_logging, LOGGING_LEVELS
configure_logging(level=LOGGING_LEVELS["warning"])

from green_integral import StdSignalData, run_green_std, plots_green_integral, plots_lyapunov

# ── Synthetic signals ───────────────────────────────────────────────────────
fs = 5000.0
f_modal = 150.0

def _make_signal(t: np.ndarray, sigma_of_t) -> np.ndarray:
    dt = 1.0 / fs
    A = np.exp(np.cumsum(sigma_of_t(t)) * dt)
    rng = np.random.default_rng(0)
    return A * np.sin(2.0 * np.pi * f_modal * t) + 1e-4 * rng.standard_normal(len(t))

# Reference: constant amplitude (sigma=0) for 1s — external "stable" signal.
T_ONSET = 1.5  # chatter onset in the analyzed signal
t_ref = np.arange(0.0, 1.0, 1.0 / fs)
x_ref = _make_signal(t_ref, lambda t: np.zeros_like(t))
ref_std = StdSignalData(t_analysis=t_ref, signal_analysis=x_ref, path="ref", fs=fs)

# Analysis: stable for [0, T_ONSET), growing (chatter, sigma=+1.5) after.
t_an = np.arange(0.0, 3.0, 1.0 / fs)
x_an = _make_signal(t_an, lambda t: np.where(t < T_ONSET, 0.0, 1.5))
an_std = StdSignalData(t_analysis=t_an, signal_analysis=x_an, path="analysis", fs=fs)

BASE_PARAMS = dict(f_modal=f_modal, num_T=4, dt=0.01, data_filtrated=True,
                    while_loop_extend=True, use_area_threshold=True, z_sigma=3.0)

def _run(func: str, params: dict, reference_signal=None):
    cfg = {"func": func, "param_mode": "native", "params": dict(params)}
    if reference_signal is not None:
        cfg["reference_signal"] = reference_signal
    return run_green_std(an_std, cfg)

# 1. Baseline: no training source at all -> branch skipped, cero impacto.
res = _run("Default", BASE_PARAMS)
assert res.meta["training_source"] == "internal"
assert "area_mu_3sigma" not in res.meta["raw_result"].global_data
assert res.t_d.size == 0

# 2. Internal training_intervals (existing behavior) still detects growth.
params_internal = {**BASE_PARAMS, "training_intervals": [(0.0, T_ONSET, "stable")]}
res = _run("Default", params_internal)
assert res.meta["training_source"] == "internal"
assert "area_mu_3sigma" in res.meta["raw_result"].global_data
assert res.t_d.size > 0 and abs(res.t_d[0] - T_ONSET) < 0.1

# 3a. External reference_signal (Default variant).
res = _run("Default", BASE_PARAMS, reference_signal=ref_std)
assert res.meta["training_source"] == "external_reference"
assert "area_mu_3sigma" in res.meta["raw_result"].global_data
assert res.t_d.size > 0 and abs(res.t_d[0] - T_ONSET) < 0.1

# 3b. External reference_signal (Lyapunov variant).
lyap_params = dict(f_modal=f_modal, num_T=6, dt=1.0 / f_modal, data_filtrated=True,
                    use_area_threshold=True, z_sigma=3.0, sigma_method="ratio")
res = _run("Lyapunov", lyap_params, reference_signal=ref_std)
assert res.meta["training_source"] == "external_reference"
assert res.meta["raw_result"].global_data["training_source"] == "external_reference"

# 4. Both provided at once: reference_signal wins, no crash.
params_both = {**BASE_PARAMS, "training_intervals": [(0.0, T_ONSET, "stable")]}
res = _run("Default", params_both, reference_signal=ref_std)
assert res.meta["training_source"] == "external_reference"


def _figure_titles() -> list:
    return [f.axes[0].get_title() for f in map(plt.figure, plt.get_fignums()) if f.axes]


def _assert_training_plots(res, titles: list) -> None:
    # Titles intentionally do NOT mention training_source (plots are meant
    # to read the same either way) — the real verification is that the
    # runner actually populated the population that trained the threshold,
    # which plot_training_distribution() itself re-asserts (mu/sigma
    # recomputed from training_areas must match area_mu_3sigma) every time
    # it runs, on top of this.
    gd = res.meta["raw_result"].global_data
    assert np.asarray(gd.get("training_areas", [])).size > 0, "training_areas not populated"
    assert np.asarray(gd.get("training_t_wins", [])).size > 0, "training_t_wins not populated"
    assert any(t.startswith("Training Area Distribution") for t in titles), titles
    assert any(t.startswith("Training Curve") for t in titles), titles


# 5a. Default variant plots: histogram + curve, population actually trained.
res_internal = _run("Default", params_internal)
plt.close("all")
plots_green_integral(signal=res_internal.meta["signal"], result=res_internal.meta["raw_result"], show=False)
_assert_training_plots(res_internal, _figure_titles())

res_ref = _run("Default", BASE_PARAMS, reference_signal=ref_std)
plt.close("all")
plots_green_integral(signal=res_ref.meta["signal"], result=res_ref.meta["raw_result"], show=False)
_assert_training_plots(res_ref, _figure_titles())

# 5b. Lyapunov variant plots: same check.
lyap_internal = _run("Lyapunov", {**lyap_params, "training_intervals": [(0.0, T_ONSET, "stable")]})
plt.close("all")
plots_lyapunov(signal=lyap_internal.meta["signal"], result=lyap_internal.meta["raw_result"],
               t_gt=T_ONSET, training_intervals=[(0.0, T_ONSET, "stable")], show=False)
_assert_training_plots(lyap_internal, _figure_titles())

lyap_ref = _run("Lyapunov", lyap_params, reference_signal=ref_std)
plt.close("all")
plots_lyapunov(signal=lyap_ref.meta["signal"], result=lyap_ref.meta["raw_result"],
               t_gt=T_ONSET, training_intervals=None, show=False)
_assert_training_plots(lyap_ref, _figure_titles())
plt.close("all")

print("OK — all reference_signal self-checks passed.")
