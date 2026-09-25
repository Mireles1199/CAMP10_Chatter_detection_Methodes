"""Ad-hoc self-check for the 'reference_signal' extension point (not committed).

Verifies: (1) default behavior (no reference_signal) is unchanged -> training_source
"internal"; (2) reference_signal, when provided, drives training -> training_source
"external_reference", using ONLY the reference's own d1 population (never touching
training_intervals' internal-mask logic); (3) the unified meta["training_t"]/
["training_d1"] population is correct for both modes and the plotting code (C3
histogram + new C5 training-signal panel) runs without error against it.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from ssq_chatter import SignalData, run_sst_svd
from ssq_chatter.viz.sst_svd_plots import plots_sst_svd

FS = 2000.0


def _make_signal(duration_s: float, amp: float, seed: int) -> SignalData:
    rng = np.random.default_rng(seed)
    n = int(duration_s * FS)
    t = np.arange(n) / FS
    x = amp * np.sin(2 * np.pi * 120.0 * t) + 0.01 * rng.standard_normal(n)
    return SignalData(t_analysis=t, signal_analysis=x, path="synthetic", fs=FS)


NATIVE_PARAMS = dict(
    n_fft_power=0, win_length_ms=50.0, hop_ms=25.0, Ai_length=4,
    mode="causal_inclusive", sigma=5.0, frac_stable=0.3,
    alpha=0.05, z=3.0, fallback_mad=True,
)

signal = _make_signal(1.0, amp=0.5, seed=0)
reference = _make_signal(0.6, amp=0.05, seed=1)  # quiet/stable reference, different content

# 1) default: no reference_signal -> internal, unchanged behavior
cfg_internal = {"func": "Default", "param_mode": "native", "params": dict(NATIVE_PARAMS)}
res_internal = run_sst_svd(signal, cfg_internal)
assert res_internal.meta["training_source"] == "internal", res_internal.meta["training_source"]

# 2) reference_signal provided -> external_reference, training stats come from it
cfg_ref = {
    "func": "Default", "param_mode": "native", "params": dict(NATIVE_PARAMS),
    "reference_signal": reference,
}
res_ref = run_sst_svd(signal, cfg_ref)
assert res_ref.meta["training_source"] == "external_reference", res_ref.meta["training_source"]

# training stats must differ from the internal-mask run (reference signal has a
# much smaller amplitude, so mu/sigma computed on it should not match the internal one)
assert res_ref.meta["training_mu"] != res_internal.meta["training_mu"]

# result shape/contract must stay intact regardless of training source
assert res_ref.I_t.shape == res_internal.I_t.shape
assert isinstance(res_ref.t_d, np.ndarray) and isinstance(res_ref.t_d_no_FAR, np.ndarray)

# 3) reference_signal takes priority over training_intervals when both given
cfg_both = dict(cfg_ref)
cfg_both["params"] = dict(NATIVE_PARAMS, training_intervals=[(0.0, 0.1, "stable")])
res_both = run_sst_svd(signal, cfg_both)
assert res_both.meta["training_source"] == "external_reference"

# 4) unified meta["training_t"]/["training_d1"] population -- the single source of
# truth the plotting code (C3/C5) now reads instead of re-guessing from t_gt/intervals.
for res in (res_internal, res_ref):
    tt, td1 = res.meta["training_t"], res.meta["training_d1"]
    assert tt is not None and td1 is not None
    assert tt.shape == td1.shape and tt.size > 1

def _matches_training_mu(res) -> bool:
    # detect() reports mean (sigma method) or median (MAD fallback) -- either is
    # a valid "training_mu"; just confirm it came from training_d1 itself.
    d1_pop, mu = res.meta["training_d1"], res.meta["training_mu"]
    return np.isclose(np.mean(d1_pop), mu) or np.isclose(np.median(d1_pop), mu)

# internal: training_d1 must be an actual subset of the analyzed signal's own d1
assert _matches_training_mu(res_internal), res_internal.meta

# external: training_d1 must be the reference's OWN population, not the real d1's
# (sizes differ: analyzed signal is 1.0s, reference is 0.6s -> fewer SVD frames)
assert res_ref.meta["training_d1"].size != res_ref.I_t.size
assert _matches_training_mu(res_ref), res_ref.meta

# 5) plotting must not crash and must produce the fixed C3 histogram + new C5 panel
# for both training sources (Agg backend -> no GUI, plt.show(block=True) is a no-op)
for res, cfg, ref_sig, label in (
    (res_internal, cfg_internal, None,      "internal"),
    (res_ref,      cfg_ref,      reference, "external_reference"),
):
    plt.close("all")
    plots_sst_svd(
        signal=signal, result=res, t_gt=0.5,
        reference_signal=ref_sig,
        training_intervals=cfg.get("params", {}).get("training_intervals"),
    )
    fig_labels = [plt.figure(n).get_label() for n in plt.get_fignums()]
    assert any(lbl.startswith("C3 ") for lbl in fig_labels), (label, fig_labels)
    assert any(lbl.startswith("C5 ") for lbl in fig_labels), (label, fig_labels)
    plt.close("all")

# 6) zoom_y must be honored EXACTLY even when threshold lines/vlines sit far
# outside it (the reported bug: Y-axis silently ballooned to include lim_sup/
# lim_inf because zoom_y was a dead parameter in most panels).
lim_sup_internal = res_internal.meta["lim_sup"]
_zoom_y = (lim_sup_internal * 5, lim_sup_internal * 10)  # well above the data AND the threshold line
plt.close("all")
plots_sst_svd(
    signal=signal, result=res_internal, t_gt=0.5,
    zoom_y=_zoom_y,
)
fig_f3 = next(plt.figure(n) for n in plt.get_fignums()
              if plt.figure(n).get_label().startswith("F3 "))
got_ylim = fig_f3.axes[0].get_ylim()
assert np.allclose(got_ylim, _zoom_y), (got_ylim, _zoom_y)
plt.close("all")

# 7) training_intervals that silently falls back (< 2 matching SVD frames, e.g.
# a range outside the signal) must NOT be treated as if it had trained the
# detector: training_mode must read "frac_stable", not "training_intervals",
# and the per-label segment panels (C2b/C3b/F3b) must not render at all.
cfg_bad_intervals = {
    "func": "Default", "param_mode": "native",
    "params": dict(NATIVE_PARAMS, training_intervals=[(50.0, 51.0, "stable")]),  # way past 1.0s signal
}
res_bad = run_sst_svd(signal, cfg_bad_intervals)
assert res_bad.meta["training_source"] == "internal", res_bad.meta["training_source"]
assert res_bad.meta["training_mode"] == "frac_stable", res_bad.meta["training_mode"]
plt.close("all")
plots_sst_svd(
    signal=signal, result=res_bad, t_gt=0.5,
    training_intervals=cfg_bad_intervals["params"]["training_intervals"],
)
fig_labels = [plt.figure(n).get_label() for n in plt.get_fignums()]
assert not any(lbl.startswith(("C2b", "C3b", "F3b")) for lbl in fig_labels), fig_labels
plt.close("all")

# 8) at most ONE "first detection" vline (t_d) ever appears alongside t_gt --
# build a signal that's loud (many over-threshold points) after t_gt to force
# a real multi-detection case through the actual pipeline, not a fabricated one.
_n1, _n2 = int(0.5 * FS), int(0.5 * FS)
_t1 = np.arange(_n1) / FS
_t2 = _n1 / FS + np.arange(_n2) / FS
_rng = np.random.default_rng(7)
_x1 = 0.3 * np.sin(2 * np.pi * 120.0 * _t1) + 0.01 * _rng.standard_normal(_n1)
_x2 = 8.0 * np.sin(2 * np.pi * 120.0 * _t2) + 0.01 * _rng.standard_normal(_n2)  # loud -> many detections
loud_signal = SignalData(
    t_analysis=np.concatenate([_t1, _t2]),
    signal_analysis=np.concatenate([_x1, _x2]),
    path="synthetic", fs=FS,
)
res_loud = run_sst_svd(loud_signal, {"func": "Default", "param_mode": "native", "params": dict(NATIVE_PARAMS)})
assert res_loud.t_d.size > 3, "test signal didn't actually produce multiple detections"
plt.close("all")
plots_sst_svd(signal=loud_signal, result=res_loud, t_gt=0.5)
fig_f3_loud = next(plt.figure(n) for n in plt.get_fignums()
                    if plt.figure(n).get_label().startswith("F3 "))
_vline_labels = [t for t in fig_f3_loud.axes[0].texts if t.get_rotation() == 90]
assert len(_vline_labels) <= 2, [t.get_text() for t in _vline_labels]
plt.close("all")

# 9) an annotated vline/hline (e.g. t_gt) falling OUTSIDE the current zoom must
# NOT change the plotted-area size (tight_layout()/bbox reacting to unclipped
# Text rendered off-canvas -- fixed with clip_on=True on every line-attached
# ax.text() call). Same zoom_x, only t_gt moves in vs. out of view; axes bbox
# must be byte-identical either way.
def _f3_axes_bbox(t_gt_value, zoom_x):
    plt.close("all")
    plots_sst_svd(signal=signal, result=res_internal, t_gt=t_gt_value, zoom_x=zoom_x)
    fig = next(plt.figure(n) for n in plt.get_fignums()
               if plt.figure(n).get_label().startswith("F3 "))
    bbox = fig.axes[0].get_position().bounds
    plt.close("all")
    return bbox

_ZOOM_X = (0.0, 0.3)
bbox_visible = _f3_axes_bbox(0.15, _ZOOM_X)   # t_gt inside the zoom -> label visible
bbox_offscreen = _f3_axes_bbox(0.9, _ZOOM_X)  # t_gt outside the zoom -> label would be off-canvas
assert bbox_visible == bbox_offscreen, (bbox_visible, bbox_offscreen)

# 10) F1-F2c (STFT/SST spectrograms + slices + waterfalls) are on-demand only:
# absent by default, present when show_spectrograms=True is passed explicitly.
plt.close("all")
plots_sst_svd(signal=signal, result=res_internal, t_gt=0.5)
fig_labels_default = [plt.figure(n).get_label() for n in plt.get_fignums()]
assert not any(lbl.startswith(("F1", "F2")) for lbl in fig_labels_default), fig_labels_default
plt.close("all")
plots_sst_svd(signal=signal, result=res_internal, t_gt=0.5, show_spectrograms=True)
fig_labels_on_demand = [plt.figure(n).get_label() for n in plt.get_fignums()]
assert any(lbl.startswith("F1 ") for lbl in fig_labels_on_demand), fig_labels_on_demand
assert any(lbl.startswith("F2 ") for lbl in fig_labels_on_demand), fig_labels_on_demand
plt.close("all")

# 11) C1/C2 and C4's top panel must NOT split by ground truth region anymore
# (single color, per user correction) -- exactly ONE data curve (many points),
# ignoring threshold/vline artists (2-point Line2D from axhline/axvline).
def _data_curve_colors(ax):
    return {ln.get_color() for ln in ax.get_lines() if len(ln.get_xdata()) > 10}

plt.close("all")
plots_sst_svd(signal=signal, result=res_internal, t_gt=0.5)
fig_c1 = next(plt.figure(n) for n in plt.get_fignums() if plt.figure(n).get_label().startswith("C1 "))
fig_c4 = next(plt.figure(n) for n in plt.get_fignums() if plt.figure(n).get_label().startswith("C4 "))
_c1_colors = _data_curve_colors(fig_c1.axes[0])
_c4_top_colors = _data_curve_colors(fig_c4.axes[0])
assert len(_c1_colors) == 1, _c1_colors
assert len(_c4_top_colors) == 1, _c4_top_colors
plt.close("all")

# 12) C3 must show mu/sigma as text inside the plot (MaxEnt convention), not
# only via axvline labels -- look for a bbox-boxed Text artist.
plt.close("all")
plots_sst_svd(signal=signal, result=res_internal, t_gt=0.5)
fig_c3 = next(plt.figure(n) for n in plt.get_fignums() if plt.figure(n).get_label().startswith("C3 "))
_boxed_texts = [t for t in fig_c3.axes[0].texts if t.get_bbox_patch() is not None]
assert any("mu" in t.get_text().lower() or "\\mu" in t.get_text() for t in _boxed_texts), \
    [t.get_text() for t in fig_c3.axes[0].texts]
plt.close("all")

# 13) seam-safety: reference_signal accepts a LIST of pieces, each windowed
# and SVD-analyzed in ISOLATION (own pipe.run call) -- only the per-piece d1/t
# RESULTS are pooled afterwards, never the raw signal. Two pieces of very
# different amplitude (quiet vs. loud) prove no frame spans the seam: pooling
# them must give byte-identical per-piece d1 to analyzing each piece alone.
piece_quiet = _make_signal(0.6, amp=1.0, seed=10)
piece_loud = _make_signal(0.6, amp=10.0, seed=11)

cfg_pool = {
    "func": "Default", "param_mode": "native", "params": dict(NATIVE_PARAMS),
    "reference_signal": [piece_quiet, piece_loud],
}
res_pool = run_sst_svd(signal, cfg_pool)

res_quiet_only = run_sst_svd(signal, {
    "func": "Default", "param_mode": "native", "params": dict(NATIVE_PARAMS),
    "reference_signal": [piece_quiet],
})
res_loud_only = run_sst_svd(signal, {
    "func": "Default", "param_mode": "native", "params": dict(NATIVE_PARAMS),
    "reference_signal": [piece_loud],
})

frames = res_pool.meta["reference_frames_per_piece"]
assert res_pool.meta["reference_n_pieces"] == 2
assert len(frames) == 2
n_quiet = res_quiet_only.meta["training_d1"].size
n_loud = res_loud_only.meta["training_d1"].size
assert [f["frames"] for f in frames] == [n_quiet, n_loud], frames

pooled_d1 = res_pool.meta["training_d1"]
assert pooled_d1.size == n_quiet + n_loud
# the pooled d1's own per-piece slices must equal each piece analyzed alone --
# if a frame had mixed the quiet tail with the loud head (or vice versa), these
# would differ.
assert np.array_equal(pooled_d1[:n_quiet], res_quiet_only.meta["training_d1"])
assert np.array_equal(pooled_d1[n_quiet:], res_loud_only.meta["training_d1"])

print("OK: reference_signal extension point + fixed C3/C5/zoom_y/training_mode/vlines/layout/on-demand/colors behave as specified.")
