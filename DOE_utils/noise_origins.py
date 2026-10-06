"""noise_origins.py — where the data of a result file come from (docs/planes/PLAN_noise_validation.md §4.4).

Every result file of the noise validation says, in root attrs, where its data come from. For each origin K two attrs:
    K_rel  path relative to the folder of this .h5 (absent when it cannot be relativized: another drive)
    K_abs  absolute path (backup)
K in KEYS. A reader resolves K_rel first, then K_abs; nothing is copied or linked between files.

Light on purpose (h5py + numpy + os): viewers import it without matplotlib or the indicator packages.

    origins(h5)                         {key: resolved path | None}
    read_signal(origin, case, name)     (t, y) of a case (clean) or a noisy copy, from doe_results.h5 / the noise file
    read_indicator(origin, case, run)   {"t", "I_t", "t_d", "attrs"} from an indicator results file
    signal_of(h5, group, name[, subtree])  (t, y) from ANY file of the noise validation: the signal of a noisy copy comes
                                        from noise_results, that of a clean case from source_signals; the case of a
                                        validation level subtree (§4.5) is the copy in its attr 'copy'
    set_origins(h5, **paths) / inherit_origins(dst, src)    writers (used by doe_noise, doe_indicators, validate_*)
A missing origin raises OriginMissing (.key, .tried).
"""
import os
import re

import h5py
import numpy as np

KEYS = ("source_signals", "clean_indicators", "noise_results", "noise_indicators", "clean_validation")
_COPY = re.compile(r"^snr_.+__r\d+$")   # name of a noisy copy: snr_040.00__case_011__r00


class OriginMissing(KeyError):
    def __init__(self, key, tried=()):
        self.key, self.tried = key, list(tried)
        super().__init__(key)

    def __str__(self):
        return f"origin '{self.key}' not found" + (f" (tried: {', '.join(self.tried)})" if self.tried else " (not recorded in the file)")


def set_origins(h5_path: str, **paths) -> None:
    """Write K_abs and, when possible, K_rel (relative to the folder of h5_path) for every non-empty path."""
    folder = os.path.dirname(os.path.abspath(h5_path))
    with h5py.File(h5_path, "a") as f:
        for k, p in paths.items():
            if k not in KEYS:
                raise ValueError(f"unknown origin {k!r}; one of {KEYS}")
            if not p:
                continue
            a = os.path.abspath(p)
            f.attrs[k + "_abs"] = a
            try:
                f.attrs[k + "_rel"] = os.path.relpath(a, folder).replace("\\", "/")
            except ValueError:   # another drive
                f.attrs.pop(k + "_rel", None)


def _candidates(h5_path: str) -> dict:
    folder = os.path.dirname(os.path.abspath(h5_path))
    out = {}
    with h5py.File(h5_path, "r") as f:
        for k in KEYS:
            c = []
            if k + "_rel" in f.attrs:
                c.append(os.path.normpath(os.path.join(folder, str(f.attrs[k + "_rel"]))))
            if k + "_abs" in f.attrs:
                c.append(os.path.normpath(str(f.attrs[k + "_abs"])))
            out[k] = c
    return out


def origins(h5_path: str) -> dict:
    """{key: path of the origin | None}: relative first, then absolute; None when not recorded or not found."""
    return {k: next((p for p in c if os.path.isfile(p)), None) for k, c in _candidates(h5_path).items()}


def _origin(h5_path: str, key: str) -> str:
    p = origins(h5_path)[key]
    if p is None:
        raise OriginMissing(key, _candidates(h5_path)[key])
    return p


def inherit_origins(dst_h5: str, src_h5: str) -> None:
    """Copy the origins of src_h5 into dst_h5 (relative paths recomputed for the folder of dst_h5; an origin that does
    not resolve keeps its recorded absolute path)."""
    found = {}
    for k, c in _candidates(src_h5).items():
        p = next((x for x in c if os.path.isfile(x)), None) or (c[-1] if c else None)
        if p:
            found[k] = p
    set_origins(dst_h5, **found)


def read_signal(origin_path: str, case: str, name: str = "Axial_disp"):
    """(t, y) of signal `name` of a case or noisy copy in doe_results.h5 / doe_noise_multi_results.h5."""
    with h5py.File(origin_path, "r") as f:
        g = f[case][name]
        return g["time"][()], g["values"][()]


def read_indicator(origin_path: str, case: str, variant: str) -> dict:
    """{"t", "I_t", "t_d", "attrs"} of one indicator run of a case / copy in an indicator results file."""
    with h5py.File(origin_path, "r") as f:
        g = f[case][variant]
        return dict(t=g["t"][()], I_t=g["I_t"][()], t_d=g["t_d"][()] if "t_d" in g else np.array([]), attrs=dict(g.attrs))


def signal_of(h5_path: str, group: str, name: str = "Axial_disp", subtree: str = None):
    """(t, y) of `group` from any file of the noise validation. The file itself first (a clean validation or a noise file
    keeps its signals), then the origin: noisy copy -> noise_results, clean case -> source_signals. `subtree`: group of
    a validation level (§4.5) whose case `group` is a case_NNN with the noisy copy in attr 'copy'."""
    with h5py.File(h5_path, "r") as f:
        base = f[subtree] if subtree else f
        if group in base and name in base[group]:
            g = base[group][name]
            return g["time"][()], g["values"][()]
        if subtree and group in base and "copy" in base[group].attrs:
            group = str(base[group].attrs["copy"])
    key = "noise_results" if _COPY.match(group) else "source_signals"
    return read_signal(_origin(h5_path, key), group, name)


def _selftest():
    import tempfile
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "a"))
    os.makedirs(os.path.join(d, "b"))
    t = np.arange(5.0)
    src, noi, ind = os.path.join(d, "a", "doe_results.h5"), os.path.join(d, "a", "noise.h5"), os.path.join(d, "b", "ind.h5")
    with h5py.File(src, "w") as f:
        f["case_000/Axial_disp/time"], f["case_000/Axial_disp/values"] = t, t * 2
    with h5py.File(noi, "w") as f:
        f["snr_040.00__case_000__r00/Axial_disp/time"], f["snr_040.00__case_000__r00/Axial_disp/values"] = t, t * 2 + 0.5
    with h5py.File(ind, "w") as f:
        f["snr_040.00__case_000__r00/run_a/t"], f["snr_040.00__case_000__r00/run_a/I_t"] = t, t + 1
        f.create_group("snr_040.00/case_000").attrs["copy"] = "snr_040.00__case_000__r00"   # a level subtree (§4.5)
    set_origins(noi, source_signals=src)
    set_origins(ind, source_signals=src, noise_results=noi, clean_indicators=os.path.join(d, "nope.h5"))
    with h5py.File(ind, "r") as f:
        assert f.attrs["noise_results_rel"] == "../a/noise.h5" and f.attrs["source_signals_abs"] == os.path.abspath(src)
    o = origins(ind)
    assert o["noise_results"] == noi and o["source_signals"] == src and o["clean_indicators"] is None and o["clean_validation"] is None
    # relative first: the folder moved, the absolute path (backup) is stale -> the relative one still resolves
    moved = d + "_moved"
    os.rename(d, moved)
    ind2 = os.path.join(moved, "b", "ind.h5")
    assert origins(ind2)["noise_results"] == os.path.join(moved, "a", "noise.h5")
    # signals: a clean case from source_signals, a noisy copy from noise_results, never from the file without them
    t0, y0 = signal_of(ind2, "case_000")
    t1, y1 = signal_of(ind2, "snr_040.00__case_000__r00")
    assert np.allclose(y0, t * 2) and np.allclose(y1, t * 2 + 0.5)
    t2, y2 = signal_of(ind2, "case_000", subtree="snr_040.00")   # its case_000 is the copy in attr 'copy'
    assert np.allclose(y2, t * 2 + 0.5)
    r = read_indicator(ind2, "snr_040.00__case_000__r00", "run_a")
    assert np.allclose(r["I_t"], t + 1) and r["t_d"].size == 0
    # an origin that is not there says which and where it looked
    try:
        _origin(ind2, "clean_indicators")
        raise SystemExit("clean_indicators should be missing")
    except OriginMissing as e:
        assert e.key == "clean_indicators" and e.tried and "nope.h5" in str(e)
    try:
        _origin(os.path.join(moved, "a", "noise.h5"), "clean_validation")   # nothing recorded
        raise SystemExit("should be missing")
    except OriginMissing as e:
        assert e.tried == []
    # inherit: the noise indicator file passes its origins on, relative paths recomputed
    out = os.path.join(moved, "c.h5")
    with h5py.File(out, "w"):
        pass
    inherit_origins(out, ind2)
    assert origins(out)["noise_results"] == os.path.join(moved, "a", "noise.h5")
    with h5py.File(out, "r") as f:
        assert f.attrs["noise_results_rel"] == "a/noise.h5"
    print("noise_origins selftest OK")


if __name__ == "__main__":
    _selftest()
