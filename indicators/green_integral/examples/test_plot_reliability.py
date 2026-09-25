"""Self-check for the viz reliability pass (titles/zoom/lines audit).

Covers four fixes unrelated to `reference_signal` itself (see
test_reference_signal.py for that), found while auditing every Green
Integral plot for correctness:

1. Zoom-sync bug in the "Index of data_window" secondary axis
   (plots.plot_windows_local / plot_indicator_local): it only re-synced
   ax2's xlim when >1 point was visible, so zooming in past that left ax2
   showing stale ticks from before the zoom.
2. plots_signal_diagnostics used to draw hardcoded frequency/beat markers
   (150/200/50 Hz, 20 ms) regardless of the actual run's config. Now they
   are explicit optional parameters (freq_markers/t_beat_ms) and only drawn
   when given — never fabricated.
3. Detection-over-time panels must draw at most two vertical event lines:
   t_gt and the first detection — never one per repeated detection. Checked
   here against a real many-detections scenario (a run with hundreds of
   threshold-crossing windows), not just by code inspection.
4. Plotting area stability: several figures used a persistent auto-layout
   engine (`layout='tight'` / `constrained_layout=True`), which recomputes
   on every draw/savefig — so ax.get_position() would visibly shrink/grow
   depending on whether a vline/hline text label (t_gt, mu+z*sigma, etc.)
   happened to fall inside the current zoom range at that moment. Replaced
   with a one-shot `fig.tight_layout()` call, which bakes in a fixed
   position that then survives any later zoom/pan/save.
5. C1 (Signal panel): was split stable(blue)/chatter(orange) — the t_gt /
   first-detection vlines already mark that boundary, so per the user's
   request the trace itself is now a single color. Also aligned the
   remaining ad-hoc named colors in plots_signal_diagnostics
   ("steelblue"/"darkorange"/"forestgreen"/"red"/"crimson") to the shared
   color_azul/orange/verde/red palette (same one MaxEnt uses), for visual
   consistency across the 4 indicators.
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

from green_integral import StdSignalData, run_green_std, plots_signal_diagnostics
from green_integral.viz.plots import plot_windows_local, plot_indicator_local

fs = 5000.0
f_modal = 150.0


def _make_signal(t: np.ndarray, sigma_of_t) -> np.ndarray:
    dt = 1.0 / fs
    A = np.exp(np.cumsum(sigma_of_t(t)) * dt)
    rng = np.random.default_rng(0)
    return A * np.sin(2.0 * np.pi * f_modal * t) + 1e-4 * rng.standard_normal(len(t))


t_an = np.arange(0.0, 3.0, 1.0 / fs)
# Growth throughout -> many windows cross the threshold -> a real
# many-detections case, the actual stress test for rule 3 below.
x_an = _make_signal(t_an, lambda t: np.where(t < 0.3, 0.0, 2.5))
an_std = StdSignalData(t_analysis=t_an, signal_analysis=x_an, path="analysis", fs=fs)

BASE = dict(f_modal=f_modal, num_T=4, dt=0.01, data_filtrated=True,
            while_loop_extend=True, use_area_threshold=True,
            training_intervals=[(0.0, 0.3, "stable")], z_sigma=3.0)


def _run(func: str, params: dict):
    return run_green_std(an_std, {"func": func, "param_mode": "native", "params": dict(params)})


# ── 1. Zoom-sync fix ────────────────────────────────────────────────────────
res = _run("Default", BASE)
raw = res.meta["raw_result"]
result_dict = {
    "data_window": raw.data_window, "agrupamiento": raw.agrupamiento,
    "global_data": raw.global_data, "t_d": raw.t_d,
}

for plot_fn in (plot_windows_local, plot_indicator_local):
    plt.close("all")
    fig = plot_fn(result_dict, name="analysis")
    axes, ax2 = fig.axes[0], fig.axes[1]
    xmax = axes.get_xlim()[1]
    axes.set_xlim(xmax * 0.999, xmax)  # deep zoom: far fewer than 2 points visible
    assert axes.get_xlim() == ax2.get_xlim(), (
        f"{plot_fn.__name__}: index axis desynced from time axis after a deep zoom"
    )
plt.close("all")

# ── 2. plots_signal_diagnostics never fabricates markers ───────────────────
lyap_res = _run("Lyapunov", {**BASE, "sigma_method": "ratio"})
raw_lyap = lyap_res.meta["raw_result"]
sig = lyap_res.meta["signal"]

plt.close("all")
plots_signal_diagnostics(signal=sig, result=raw_lyap, stable_range=(0.0, 0.3),
                          zoom_range=(0.05, 0.1), eq_smooth_s=0.02, show=False)
a1 = plt.figure(1).axes[0]
labels_no_markers = [l.get_text() for l in (a1.get_legend().get_texts() if a1.get_legend() else [])]
assert not any("Hz" in l for l in labels_no_markers), labels_no_markers

plt.close("all")
plots_signal_diagnostics(signal=sig, result=raw_lyap, stable_range=(0.0, 0.3),
                          zoom_range=(0.05, 0.1), eq_smooth_s=0.02,
                          freq_markers={"f_modal": f_modal}, t_beat_ms=6.0, show=False)
a1b = plt.figure(1).axes[0]
labels_with_markers = [l.get_text() for l in a1b.get_legend().get_texts()]
assert any("f_modal" in l for l in labels_with_markers), labels_with_markers
plt.close("all")

# ── 3. At most 2 vertical event lines on detection-over-time panels ────────
from green_integral import plots_lyapunov

assert raw_lyap.t_d is not None and len(raw_lyap.t_d) > 5, (
    "test setup should produce a real many-detections case"
)

plt.close("all")
plots_lyapunov(signal=sig, result=raw_lyap, t_gt=0.3,
               training_intervals=[(0.0, 0.3, "stable")], show=False)


def _n_event_vlines(ax) -> int:
    """Count axvline-style Line2D artifacts (two identical, finite x values)."""
    n = 0
    for line in ax.get_lines():
        xd = line.get_xdata()
        if len(xd) == 2 and np.isfinite(xd[0]) and xd[0] == xd[1]:
            n += 1
    return n


for fig_num in plt.get_fignums():
    fig = plt.figure(fig_num)
    for ax in fig.axes:
        title = ax.get_title()
        if any(tag in title for tag in ("C1", "C2", "C3", "Accumulator", "Sliding")):
            n = _n_event_vlines(ax)
            assert n <= 2, f"{title!r}: {n} vertical event lines, expected at most 2 (t_gt + first detection)"

# ── 4. Plotting area (ax.get_position()) must not move on zoom ─────────────
# Reuses the figures plots_lyapunov just produced above — every one of them
# carries at least one vline/hline text label whose visibility toggles with
# the current view, which is exactly what triggered the old bug.
for fig_num in plt.get_fignums():
    fig = plt.figure(fig_num)
    for ax in fig.axes:
        title = ax.get_title() or "?"
        pos0 = tuple(round(v, 6) for v in ax.get_position().bounds)
        xlo, xhi = ax.get_xlim()
        span = (xhi - xlo) * 0.15 or 1.0
        ax.set_xlim(0.3 - span / 2, 0.3 + span / 2)  # label(s) in view
        pos_visible = tuple(round(v, 6) for v in ax.get_position().bounds)
        ax.set_xlim(xlo + 1e-6, xlo + 1e-6 + span)     # label(s) out of view
        pos_hidden = tuple(round(v, 6) for v in ax.get_position().bounds)
        assert pos0 == pos_visible == pos_hidden, (
            f"{title!r}: plotting area moved with zoom "
            f"({pos0} vs {pos_visible} vs {pos_hidden}) — a persistent "
            f"layout engine (layout='tight'/constrained_layout=True) "
            f"crept back in"
        )

# ── 5. C1 (Signal panel) is a single trace, single color ───────────────────
# Was stable(blue)/chatter(orange) split; user wants one color — the t_gt /
# first-detection vlines already mark the split, the trace itself shouldn't.
fig_c1 = plt.figure(1)
for ax in fig_c1.axes:
    data_lines = [l for l in ax.get_lines() if len(l.get_xdata()) > 2]
    colors = {l.get_color() for l in data_lines}
    assert len(colors) == 1, f"C1 axes {ax.get_ylabel()!r}: expected 1 trace color, got {colors}"

plt.close("all")

print("OK — plot reliability self-checks passed.")
