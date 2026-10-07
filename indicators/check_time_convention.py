"""Common check of the time convention of the 4 indicators (see COMMON_TEMPLATE.md, "Convención de tiempo").

Rule: the time `t` of an output is the time of the LAST sample of the data that output used.

Method: the same signal (150 Hz oscillation + noise) is run with and without a one-sample pulse at a known time.
The first output whose change is significant (>= FRAC of the largest change) is the earliest window containing the
pulse: its end is at the pulse or at most one hop (1 revolution) after it, so  0 <= t_first - t_pulse < hop  (a
small tolerance covers filters that spread the pulse a few samples). A label later than the end of its data (as SST
had: centre + one window instead of centre + half a window, +9.9 ms) falls outside that interval.

"Significant" and not "any difference": SST differentiates its window in the frequency domain
(trans_mio/windows.get_window: ifft(fft(window) * 1j * xi)); the truncated Gaussian makes that derivative ring over the
whole n_fft (203 ms), so every SST output depends faintly (1e-6..1e-3 relative) on samples up to ~100 ms away.

    <python of the indicators' environment> indicators/check_time_convention.py

Uses the src/ of THIS checkout (not the editable installs), like the examples/*_NEW.py.
"""
import contextlib
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for pkg in ("maxent_sprt", "rms_cv", "ssq_chatter", "green_integral"):
    sys.path.insert(0, os.path.join(HERE, pkg, "src"))

import MaxEnt_SPRT as _mx   # noqa: E402
import rms_cv as _rms       # noqa: E402
import ssq_chatter as _ssq  # noqa: E402
import green_integral as _gi  # noqa: E402

RPM, FS, F_MODAL = 12098.28, 40327.6, 150.0
T_REV = 60.0 / RPM
HOP = T_REV                       # step_rev = 1 in every configuration below
TOL = 1.5e-3                      # s; savgol / zero-crossing filters spread a pulse a few samples
FRAC = 0.05                       # an output "responds" to the pulse when its change is >= FRAC * the largest change
T_END, T_PULSE_NOMINAL = 6.0, 5.0
DEC = int(np.ceil(FS / (RPM / 60.0)))   # MaxEnt OPR decimation (every DEC-th sample)


def make_signal(pulse: bool):
    """t, x (displacement), v (velocity) and the pulse time; chatter-like amplitude between 2 s and 4 s."""
    n = int(T_END * FS)
    t = np.arange(1, n + 1) / FS
    rng = np.random.default_rng(0)
    amp = np.where((t > 2.0) & (t < 4.0), 3.0, 1.0)
    w = 2 * np.pi * F_MODAL
    x = amp * np.sin(w * t) + 0.05 * rng.standard_normal(n)
    v = amp * w * np.cos(w * t)
    k = int(round(T_PULSE_NOMINAL * FS / DEC)) * DEC    # an OPR-sampled index (MaxEnt only sees those)
    if pulse:
        x[k] += 50.0
        v[k] += 50.0 * w
    return t, x, v, t[k]


def first_change(t_out, i_with, i_without):
    """Time of the first output whose change caused by the pulse is significant (>= FRAC of the largest)."""
    n = min(len(i_with), len(i_without))
    change = np.nan_to_num(np.abs(np.asarray(i_with, float)[:n] - np.asarray(i_without, float)[:n]))
    assert change.max() > 0, "the pulse changed no output"
    return float(np.asarray(t_out, float)[np.argmax(change >= FRAC * change.max())])


def run_maxent(t, x, v):
    sig = _mx.SignalData(t_analysis=t, signal_analysis=x, path="synthetic", fs=FS, meta={})
    cfg = {"id": "MaxEnt_SPRT", "func": "Default", "param_mode": "by_revolution", "params_physical": {
        "T_rev": T_REV, "N_rev_window": 4, "step_rev": 1, "segmentation": "opr",
        "alpha": 0.00135, "beta": 0.00135, "reset_on_H0": True, "t_stable_total": None,
        "training_intervals": [(0.1, 1.9, "stable"), (2.1, 3.9, "chatter")]}}
    r = _mx.run_maxent_sprt(sig, cfg)
    return r.t, r.I_t


def run_rms(t, x, v):
    sig = _rms.SignalData(t_analysis=t, signal_analysis=x, path="synthetic", fs=FS, meta={})
    cfg = {"id": "RMS_CV", "func": "Default", "param_mode": "by_revolution", "params_physical": {
        "T_rev": T_REV, "N_rev_window": 4, "step_rev": 1, "n_max_mode": "frames", "n_max_rev": 4,
        "cv_threshold": None, "rms_threshold": None, "n_min_cv": 2, "warmup_ignore_alerts": False,
        "use_unbiased_std": True, "eps": 1e-12, "detrend": False, "pad_mode": "none",
        "stable_time": None, "z": 3.0, "alpha": 0.05, "fallback_mad": True}}
    r = _rms.run_rms_cv(sig, cfg)
    return r.t, r.I_t


def run_sst(t, x, v):
    sig = _ssq.SignalData(t_analysis=t, signal_analysis=x, path="synthetic", fs=FS, meta={})
    cfg = {"id": "SST_SVD", "func": "Default", "param_mode": "by_revolution", "params_physical": {
        "T_rev": T_REV, "N_rev_window": 4, "step_rev": 1, "Ai_length_mode": "frames", "Ai_length_rev": 4,
        "n_fft_power": 3, "mode": "causal_inclusive", "sigma": 6.0, "frac_stable": 0.36,
        "training_intervals": None, "alpha": 0.05, "z": 3.0, "fallback_mad": False}}
    r = _ssq.run_sst_svd(sig, cfg)
    return r.t, r.I_t


def run_green(t, x, v):
    sig = _gi.StdSignalData(t_analysis=t, signal_analysis=x, path="synthetic", fs=FS, meta={"velocity": v, "name": "s"})
    cfg = {"id": "Green_Integral", "func": "Lyapunov", "param_mode": "by_revolution", "params_physical": {
        "T_rev": T_REV, "N_rev_window": 4, "step_rev": 1.0, "use_area_threshold": False, "data_filtrated": True,
        "lambda_ewma": None, "accumulate": False, "G_memory": None, "sigma_method": "ratio", "sigma_local_n": 10,
        "area_noise_eps": 1e-30, "use_zero_crossing_cycles": True, "use_beta_from_cycles": False,
        "zc_detrend": True, "v_cycle_mode": "zero", "cycle_area_norm": "none", "debug_level": 0}}
    r = _gi.run_green_std(sig, cfg)
    raw = r.meta["raw_result"]          # the areas themselves: sigma_ewma would also depend on the previous window
    return raw.t_wins, raw.areas


RUNNERS = {"MaxEnt_SPRT": run_maxent, "RMS_CV": run_rms, "SST_SVD": run_sst, "Green_Integral": run_green}


def measure(names=None):
    """{indicator: t_first - t_pulse} in seconds."""
    out = {}
    base = make_signal(False)
    pulsed = make_signal(True)
    t_pulse = pulsed[3]
    for name in names or RUNNERS:
        with contextlib.redirect_stdout(io.StringIO()):
            t0, i0 = RUNNERS[name](*base[:3])
            t1, i1 = RUNNERS[name](*pulsed[:3])
        out[name] = first_change(t1, i1, i0) - t_pulse
    return out


def main(names=None):
    offsets = measure(names)
    bad = []
    print(f"hop = {HOP * 1e3:.2f} ms; allowed 0 <= t_first - t_pulse < hop (tolerance {TOL * 1e3:.1f} ms)")
    for name, d in offsets.items():
        ok = -TOL <= d < HOP + TOL
        print(f"  {name:15s} t_first - t_pulse = {d * 1e3:7.2f} ms   {'OK' if ok else 'OUT: label later/earlier than its data'}")
        if not ok:
            bad.append(name)
    assert not bad, f"time convention not met by {bad}"
    print("OK")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
