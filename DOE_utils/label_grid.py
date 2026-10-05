#!/usr/bin/env python
# coding: utf-8
"""label_grid.py — the labelled dataset (reference_dataset*.h5 of Label build) as a grid, one panel per case.

Each panel shows the signal of a case coloured by its label (green stable, grey gray, red unstable) and, if the
dataset was labelled by amplitude, the lines of the criterion (+-lim_inf, +-lim_sup of max|y|; they can be switched
off: when the signal is far below the limits they flatten it, and the autoscale then fits the signal). Cases are
sorted by kappa and paged. Read-only: the .h5 is never written.

Usage (entorno_CAMP10 Python):
    python label_grid.py --h5 reference_dataset_amp.h5 [--channel Axial_disp]
    python label_grid.py --h5 FILE --png OUT.png [--page N]     # no window: saves one page
    python label_grid.py --selftest
"""
import argparse
import os
import sys
import tempfile

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import experiment as ex  # noqa: E402  (is_ramp / _attr: how the app reads Ap and kappa of a case)

LABELS = ("stable", "gray", "unstable")
COLOR = {"stable": "#2e7d32", "gray": "#8d8d8d", "unstable": "#c62828"}
DEFAULT_CHANNEL = "Axial_disp"
BINS = 1000   # min/max pairs per panel: keeps the peaks (what the amplitude limits are about) at any length


def _limits(attrs, channel):
    """(inf, sup) in signal units of the amplitude criterion if the piece was labelled by amplitude ON THIS channel
    (the limits are in the units of the labelled signal), else None. Same rule as the viewer."""
    if attrs.get("labeling_strategy") != "amplitude" or attrs.get("labeling_signal") != channel:
        return None
    try:
        base = float(attrs[str(attrs["labeling_base_attr"])]) * float(attrs["labeling_base_scale"])
        return base * float(attrs["labeling_lim_inf_pct"]) / 100.0, base * float(attrs["labeling_lim_sup_pct"]) / 100.0
    except (KeyError, TypeError, ValueError):
        return None


def index(h5_path: str) -> dict:
    """{'channels': [...], 'default_channel': str, 'cases': [...]} (attrs only, no signals), cases sorted by kappa.
    A case: {case, kappa (sort key, None if unknown), ktxt, ramp, pieces: [(label, piece_name, channel, limits)]}."""
    cases, channels, amp_ch = {}, set(), None
    with h5py.File(h5_path, "r") as f:
        for label in LABELS:
            if label not in f:
                continue
            for cn, cg in f[label].items():
                for pn, pg in cg.items():
                    a = dict(pg.attrs)
                    ch = str(a.get("channel") or pn.rsplit("__", 1)[0])
                    channels.add(ch)
                    if a.get("labeling_strategy") == "amplitude" and amp_ch is None:
                        amp_ch = a.get("labeling_signal")
                    c = cases.setdefault(cn, {"case": cn, "pieces": [], "attrs": a})
                    c["pieces"].append((label, pn, ch, _limits(a, ch)))
    out = []
    for c in cases.values():
        a = c.pop("attrs")
        ramp = ex.is_ramp(a)
        ks = [x for x in (ex._attr(a, "kappa_t0" if ramp else "kappa"),) if x is not None]
        if ramp:   # ponytail: ramps only shown, not studied (parked in PLAN_ramps); kappa of the start of the piece
            ktxt = "ramp " + ex.ap_text(a, ".1f")
        else:
            ktxt = (f"κ={ks[0]:.3g}" if ks else ex.ap_text(a, ".1f"))
        c.update(kappa=ks[0] if ks else None, ktxt=ktxt, ramp=ramp)
        out.append(c)
    out.sort(key=lambda c: (c["kappa"] is None, c["kappa"] if c["kappa"] is not None else 0.0, c["case"]))
    ch = sorted(channels)
    default = amp_ch if amp_ch in ch else DEFAULT_CHANNEL if DEFAULT_CHANNEL in ch else (ch[0] if ch else "")
    return {"channels": ch, "default_channel": default, "cases": out}


def _minmax(t, y, bins=BINS):
    """Envelope of a long signal as min/max pairs per bin (peaks kept)."""
    n = len(y)
    k = n // bins
    if k < 2:
        return t, y
    m = bins * k
    yy = y[:m].reshape(bins, k)
    return np.repeat(t[:m:k], 2), np.stack([yy.min(1), yy.max(1)], 1).ravel()


def counts(idx: dict, channel: str) -> dict:
    """{label: number of cases that have a piece of this channel with that label}."""
    return {lab: sum(any(p[0] == lab and p[2] == channel for p in c["pieces"]) for c in idx["cases"]) for lab in LABELS}


def select(idx: dict, channel: str, label: str = "all") -> list:
    """Cases (sorted by kappa) with a piece of `channel` and, if label != 'all', of that label."""
    return [c for c in idx["cases"] if any(p[2] == channel and label in ("all", p[0]) for p in c["pieces"])]


def draw(fig, h5_path: str, idx: dict, channel: str, label: str = "all", limits: bool = True, page: int = 0,
         rows: int = 3, cols: int = 3) -> tuple:
    """Draw one page of the grid on `fig`. Returns (page used, number of pages, number of cases)."""
    cs = select(idx, channel, label)
    per = max(1, rows * cols)
    pages = max(1, -(-len(cs) // per))
    page = max(0, min(page, pages - 1))
    fig.clear()
    axes = fig.subplots(rows, cols, squeeze=False)
    with h5py.File(h5_path, "r") as f:
        for ax, c in zip(axes.ravel(), cs[page * per:(page + 1) * per]):
            labs = set()
            lim = None
            for lab, pn, ch, lm in c["pieces"]:
                if ch != channel or label not in ("all", lab):
                    continue
                g = f[lab][c["case"]][pn]
                t, y = _minmax(g["t"][()], g["y"][()])
                ax.plot(t, y, color=COLOR[lab], lw=0.7)
                labs.add(lab)
                lim = lim or lm
            if limits and lim:
                for v, ls in ((lim[0], ":"), (lim[1], "-.")):
                    for s in (v, -v):
                        ax.axhline(s, color="k", lw=0.7, ls=ls, alpha=0.7)
            ax.set_title(f"{c['case']}  {c['ktxt']}", fontsize=9,
                         color=COLOR[next(iter(labs))] if len(labs) == 1 else "k")
            ax.tick_params(labelsize=7)
        for ax in axes.ravel()[len(cs[page * per:(page + 1) * per]):]:
            ax.axis("off")
    fig.tight_layout()
    return page, pages, len(cs)


class GridWindow:
    def __init__(self, h5_path: str, channel: str | None = None, root=None):
        import tkinter as tk
        from tkinter import ttk
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure
        self.h5, self.idx, self.page = h5_path, index(h5_path), 0
        self.root = root or tk.Tk()
        self.root.title(f"Labelled data — {os.path.basename(h5_path)}")
        self.root.geometry("1200x800")
        v = lambda x: tk.StringVar(value=x)   # noqa: E731
        self.channel = v(channel if channel in self.idx["channels"] else self.idx["default_channel"])
        self.label, self.rows, self.cols = v("all"), v("3"), v("3")
        self.limits = tk.BooleanVar(value=True)
        self.info = v("")
        bar = ttk.Frame(self.root, padding=(6, 4))
        bar.pack(side=tk.TOP, fill=tk.X)

        def combo(text, var, values, w):
            ttk.Label(bar, text=text).pack(side=tk.LEFT, padx=(8, 2))
            cb = ttk.Combobox(bar, textvariable=var, values=values, state="readonly", width=w)
            cb.pack(side=tk.LEFT)
            cb.bind("<<ComboboxSelected>>", lambda _e: self.replot(0))
        combo("channel", self.channel, self.idx["channels"], 14)
        combo("label", self.label, ["all", *LABELS], 9)
        ttk.Checkbutton(bar, text="criterion lines (±lim_inf, ±lim_sup)", variable=self.limits,
                        command=lambda: self.replot()).pack(side=tk.LEFT, padx=10)
        combo("rows", self.rows, ["1", "2", "3", "4", "5"], 3)
        combo("cols", self.cols, ["1", "2", "3", "4", "5", "6"], 3)
        ttk.Button(bar, text="◀", width=3, command=lambda: self.replot(self.page - 1)).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(bar, text="▶", width=3, command=lambda: self.replot(self.page + 1)).pack(side=tk.LEFT)
        ttk.Label(bar, textvariable=self.info).pack(side=tk.LEFT, padx=10)
        self.fig = Figure(figsize=(12, 7.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        NavigationToolbar2Tk(self.canvas, self.root).update()   # zoom / pan / save: zoom is per panel
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.replot(0)

    def replot(self, page=None):
        self.page, pages, n = draw(self.fig, self.h5, self.idx, self.channel.get(), self.label.get(),
                                   self.limits.get(), self.page if page is None else page,
                                   int(self.rows.get()), int(self.cols.get()))
        c = counts(self.idx, self.channel.get())
        self.info.set(f"page {self.page + 1}/{pages}  ·  {n} cases (sorted by κ)  ·  "
                      + "  ".join(f"{k} {c[k]}" for k in LABELS if c[k]))
        self.canvas.draw_idle()


def _selftest():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "reference_dataset_amp.h5")
        t = np.linspace(0, 1, 5000)
        with h5py.File(p, "w") as f:
            for i, (lab, k, amp) in enumerate([("stable", 0.5, 1e-6), ("stable", 0.9, 2e-6), ("unstable", 1.2, 1e-4),
                                               ("gray", 1.05, 3e-5)]):
                g = f.require_group(f"{lab}/case_{i:03d}/Axial_disp__000")
                g.create_dataset("t", data=t)
                g.create_dataset("y", data=amp * np.sin(2e3 * t))
                g.attrs.update({"channel": "Axial_disp", "kappa": k, "$Ap_start$": 0.005, "$Ap_end$": 0.005,
                                "labeling_strategy": "amplitude", "labeling_signal": "Axial_disp",
                                "labeling_base_attr": "$f_tooth$", "$f_tooth$": 0.05, "labeling_base_scale": 1e-3,
                                "labeling_lim_inf_pct": 10.0, "labeling_lim_sup_pct": 40.0})
        idx = index(p)
        assert [c["case"] for c in idx["cases"]] == ["case_000", "case_001", "case_003", "case_002"]   # by kappa
        assert idx["default_channel"] == "Axial_disp" and counts(idx, "Axial_disp") == {"stable": 2, "gray": 1, "unstable": 1}
        assert [c["case"] for c in select(idx, "Axial_disp", "unstable")] == ["case_002"]
        lm = idx["cases"][0]["pieces"][0][3]
        assert abs(lm[0] - 5e-6) < 1e-12 and abs(lm[1] - 2e-5) < 1e-12, lm
        assert _limits({"labeling_strategy": "amplitude", "labeling_signal": "Axial_vel"}, "Axial_disp") is None
        from matplotlib.figure import Figure
        fig = Figure()
        assert draw(fig, p, idx, "Axial_disp", rows=1, cols=2) == (0, 2, 4)
        assert len(fig.axes[0].lines) == 1 + 4 and len(fig.axes[1].lines) == 1 + 4   # signal + 4 criterion lines
        draw(fig, p, idx, "Axial_disp", limits=False, rows=1, cols=2)
        assert len(fig.axes[0].lines) == 1                                         # lines off: signal only
        assert draw(fig, p, idx, "Axial_disp", page=9, rows=1, cols=2)[0] == 1     # page clamped
        x, y = _minmax(np.arange(100000.0), np.sin(np.arange(100000.0)))
        assert len(x) == len(y) == 2 * BINS
    print("label_grid selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--h5")
    ap.add_argument("--channel")
    ap.add_argument("--png", help="save one page to this file and exit (no window)")
    ap.add_argument("--page", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.h5:
        ap.error("--h5 is required")
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        idx = index(a.h5)
        fig = Figure(figsize=(12, 7.5))
        draw(fig, a.h5, idx, a.channel or idx["default_channel"], page=a.page)
        fig.savefig(a.png, dpi=90)
        return
    GridWindow(a.h5, a.channel).root.mainloop()


if __name__ == "__main__":
    main()
