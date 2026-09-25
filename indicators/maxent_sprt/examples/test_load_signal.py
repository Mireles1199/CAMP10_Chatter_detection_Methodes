"""
test_load_signal.py
====================
Self-check for ``load_signal`` (see ``indicators/COMMON_TEMPLATE.md``): reads
(t, y) uniformly from either the raw simulator HDF5 layout
(``"<signal>/data"`` as an (N, 2) array) or the DOE-repackaged layout
(``"<case>/<signal>/time"`` + ``"<case>/<signal>/values"`` as separate
datasets).

Synthetic ``.h5`` fixture built on the fly, no real dataset needed.
Run directly: asserts only.
"""
from __future__ import annotations

import os
import sys
import tempfile

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from MaxEnt_SPRT import HDF5Reader, load_signal


def _make_fixture(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, 100)
    y_raw = np.sin(t)
    y_doe = np.cos(t)
    with h5py.File(path, "w") as f:
        # raw simulator layout: "<signal>/data" as (N, 2)
        f.create_dataset("Axial_vel/data", data=np.column_stack([t, y_raw]))
        # DOE-repackaged layout: "<case>/<signal>/time" + ".../values"
        f.create_dataset("case_003/Axial_vel/time", data=t)
        f.create_dataset("case_003/Axial_vel/values", data=y_doe)
    return t, y_raw, y_doe


def main() -> None:
    fd, path = tempfile.mkstemp(suffix=".h5")
    os.close(fd)
    try:
        t_expected, y_raw_expected, y_doe_expected = _make_fixture(path)
        reader = HDF5Reader(path)

        # ── raw layout (case_name=None) ────────────────────────────────────
        t, y = load_signal(reader, "Axial_vel", case_name=None)
        assert np.allclose(t, t_expected)
        assert np.allclose(y, y_raw_expected)

        # ── DOE layout (case_name given) ───────────────────────────────────
        t2, y2 = load_signal(reader, "Axial_vel", case_name="case_003")
        assert np.allclose(t2, t_expected)
        assert np.allclose(y2, y_doe_expected)

        print("test_load_signal: OK")
    finally:
        os.remove(path)


if __name__ == "__main__":
    main()
