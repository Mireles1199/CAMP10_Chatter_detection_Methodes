#!/usr/bin/env python
# coding: utf-8
"""check_plot_style.py — checks of the unified figure style (PLAN_plot_style.md, T6). Read only on the data.

    python check_plot_style.py --render DIR                 draws the baseline figures of real files -> DIR/*.png + DIR/digests.json
    python check_plot_style.py --compare DIR_A DIR_B        the DATA drawn by every figure is identical (styles may differ)
    python check_plot_style.py --sync                       the 5 package copies of plot_style.py are byte-identical to the canonical
    python check_plot_style.py --lint                       no local rcParams / hand-set font sizes in the migrated files
    python check_plot_style.py --selftest

The digest hashes what a figure DRAWS (line xy, bars, collections offsets/paths, images), never its look, so a style
change must leave it untouched: that is the "no numeric result changes" proof of every phase.
"""
import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CANON = os.path.join(HERE, "plot_style.py")
PACKAGE_COPIES = [os.path.join(REPO, "indicators", *p, "viz", "plot_style.py") for p in (
    ("maxent_sprt", "src", "MaxEnt_SPRT"), ("rms_cv", "src", "rms_cv"), ("ssq_chatter", "src", "ssq_chatter"),
    ("green_integral", "src", "green_integral"), ("emd_hht", "src", "C_emd_hht"))]
# files already migrated to the unified style (grows phase by phase); the lint is strict on them
MIGRATED = [os.path.join(HERE, n) for n in ("plot_style.py", "validation_figures.py", "sld_model.py", "fig_lang.py")]
FORBIDDEN = [(r"rcParams\.update\(", "local rcParams"), (r"def (_?configurar_estilo_global|configure_global_style)", "old style function"),
             (r"(?<![\w.])(fontsize|labelsize|titlesize)\s*=\s*\d", "hand-set font size (use a relative name)")]
BASELINE_DEFAULT = dict(
    clean=os.path.join(REPO, "validacion_figs", "n12000", "doe_validation_results.h5"),
    noise=os.path.join(REPO, "validacion_figs", "noise_all22_r1", "doe_noise_validation_results.h5"), snr=40.0)


def _rel(p):
    return os.path.relpath(p, REPO) if os.path.splitdrive(p)[0].lower() == os.path.splitdrive(REPO)[0].lower() else p


# ============================================================================== digest of the drawn data
def _h(h, a):
    a = np.asarray(a)
    h.update(np.nan_to_num(np.ascontiguousarray(a, dtype=float), nan=-9.99e99, posinf=9.99e99, neginf=-9.99e99).tobytes()
             if a.dtype.kind in "fiub" else repr(a.tolist()).encode())


def digest(fig) -> str:
    """SHA-1 of the data drawn in `fig`, axis by axis, in drawing order (look and text excluded)."""
    h = hashlib.sha1()
    for ax in fig.axes:
        h.update(b"AX")
        for ln in ax.lines:
            _h(h, ln.get_xydata())
        for p in ax.patches:
            if hasattr(p, "get_width") and hasattr(p, "get_height") and hasattr(p, "get_x"):
                _h(h, [p.get_x(), p.get_y(), p.get_width(), p.get_height()])
            else:
                _h(h, p.get_path().vertices)
        for c in ax.collections:
            _h(h, c.get_offsets())
            if hasattr(c, "get_segments"):
                for s in c.get_segments():
                    _h(h, s)
            else:
                for pa in c.get_paths():
                    _h(h, pa.vertices)
        for im in ax.images:
            _h(h, im.get_array().filled(np.nan) if hasattr(im.get_array(), "filled") else im.get_array())
    return h.hexdigest()


# ============================================================================== baseline render
def render(out_dir, clean, noise, snr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, HERE)
    import validation_figures as vf
    os.makedirs(out_dir, exist_ok=True)
    res = {}

    def one(tag, fn, *a, **kw):
        fig = fn(*a, **kw)
        res[tag] = digest(fig)
        fig.savefig(os.path.join(out_dir, tag + ".png"), dpi=100)
        plt.close(fig)
    for name, fn in vf.FIGURES.items():
        if os.path.exists(clean):
            one(f"clean_{name}", fn, clean)
        if os.path.exists(noise):
            one(f"noise{int(snr):03d}_{name}", fn, noise, snr=snr)
    for name, fn in vf.NOISE_FIGURES.items():
        if os.path.exists(noise):
            one(f"noisesum_{name}", fn, noise)
    with open(os.path.join(out_dir, "digests.json"), "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print(f"{len(res)} figures -> {out_dir}")
    return res


def compare(a, b):
    A, B = (json.load(open(os.path.join(d, "digests.json"))) for d in (a, b))
    bad = sorted(k for k in set(A) | set(B) if A.get(k) != B.get(k))
    print(f"{len(A)} vs {len(B)} figures; data digests differ in {len(bad)}: {bad}")
    return not bad


# ============================================================================== sync and lint
def sync_check(canon=CANON, copies=PACKAGE_COPIES):
    ref = open(canon, "rb").read()
    bad = [p for p in copies if not os.path.exists(p) or open(p, "rb").read() != ref]
    for p in bad:
        print("DIFFERS" if os.path.exists(p) else "MISSING", _rel(p))
    print(f"sync: {len(copies) - len(bad)}/{len(copies)} copies identical to the canonical")
    return not bad


def lint(files=MIGRATED):
    n = 0
    for p in files:
        for i, line in enumerate(open(p, encoding="utf8").read().splitlines(), 1):
            code = line.split("#", 1)[0]
            if os.path.basename(p) == "plot_style.py" and "rcParams.update(" in code:
                continue   # the canonical may document/offer apply()
            for rx, why in FORBIDDEN:
                if re.search(rx, code):
                    n += 1
                    print(f"{_rel(p)}:{i}: {why}: {line.strip()[:90]}")
    print(f"lint: {n} findings in {len(files)} files")
    return n == 0


def _selftest():
    import tempfile
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.linspace(0, 1, 50)

    def fig(rc, y):
        with plt.rc_context(rc):
            f, ax = plt.subplots()
            ax.plot(x, y, "o-")
            ax.bar([0, 1], [1.0, 2.0])
            ax.scatter(x, y)
            ax.imshow(np.outer(x, x))
            return f
    d1 = digest(fig({}, x ** 2))
    assert d1 == digest(fig({"font.size": 30, "lines.markersize": 2, "figure.figsize": (9, 9)}, x ** 2)), "style changed the digest"
    assert d1 != digest(fig({}, x ** 3)), "digest blind to the data"
    d = tempfile.mkdtemp()
    a, b = (os.path.join(d, n) for n in "ab")
    open(a, "wb").write(b"x")
    open(b, "wb").write(b"x")
    assert sync_check(a, [b]) and not sync_check(a, [b, os.path.join(d, "missing")])
    open(b, "wb").write(b"y")
    assert not sync_check(a, [b])
    ok, bad = os.path.join(d, "ok.py"), os.path.join(d, "bad.py")
    open(ok, "w").write("ax.legend(fontsize='small')  # fontsize=8 in a comment\nx = size = 3\n")
    open(bad, "w").write("plt.rcParams.update({})\nax.text(0, 0, 's', fontsize=8)\ndef configurar_estilo_global(): pass\n")
    assert lint([ok]) and not lint([bad])
    print("check_plot_style selftest OK")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--render", metavar="DIR")
    p.add_argument("--compare", nargs=2, metavar=("DIR_A", "DIR_B"))
    p.add_argument("--sync", action="store_true")
    p.add_argument("--lint", action="store_true")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--clean", default=BASELINE_DEFAULT["clean"])
    p.add_argument("--noise", default=BASELINE_DEFAULT["noise"])
    p.add_argument("--snr", type=float, default=BASELINE_DEFAULT["snr"])
    a = p.parse_args()
    ok = True
    if a.selftest:
        return _selftest()
    if a.render:
        render(a.render, a.clean, a.noise, a.snr)
    if a.compare:
        ok &= compare(*a.compare)
    if a.sync:
        ok &= sync_check()
    if a.lint:
        ok &= lint()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
