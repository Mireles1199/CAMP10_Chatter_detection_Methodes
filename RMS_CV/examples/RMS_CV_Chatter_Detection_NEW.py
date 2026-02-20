
from typing import  Tuple
import os
import sys
import h5py
import numpy as np

from pathlib import Path

from rms_cv import SignalData
from rms_cv import HDF5Reader
from rms_cv import run_rms_cv
from rms_cv import plots_rms_cv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _cut_signal( t,x , time_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cuts the signal to the specified time range.

    Parameters
    ----------
    t : np.ndarray
        Time array of the signal.
    x : np.ndarray
        Signal values corresponding to time points.
    time_range : Tuple[float, float]
        A tuple containing (start_time, end_time) defining the time window
        to extract from the signal.
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - t_cut : np.ndarray
            Time array values within the specified range.
        - x_cut : np.ndarray
            Signal values within the specified time range.
    Examples
    --------
    >>> t = np.array([0, 1, 2, 3, 4, 5])
    >>> x = np.array([1, 2, 3, 4, 5, 6])
    >>> t_cut, x_cut = _cut_signal(t, x, (1.5, 4.5))
    >>> t_cut
    array([2., 3., 4.])
    >>> x_cut
    array([3, 4, 5])
    """

    start_time, end_time = time_range
    mask = (t >= start_time) & (t <= end_time)
    return t[mask], x[mask]

dir_cono =  r'D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\2DOF_Cono\1DOF_150Hz'
dir_path_use = dir_cono

data_dir = os.path.abspath(os.path.join(dir_path_use, 'out.hdf5' ))
data = HDF5Reader(data_dir)

tool_dyn = data.get_element('tool_dyn/data',)
t = tool_dyn[:,0]
tool_dyn = tool_dyn[:,1]
tool_dyn_vel = data.get_element('tool_dyn_o/data',)[:,1]
force_N = data.get_element('res_R_p/data',)[:,1] #Newtons

t = t
v = tool_dyn_vel
fs = 1.0 / (t[1]-t[0])
curt_range: Tuple[float, float] = (0.05, 15)

t_cut, v_cut = _cut_signal( t, v , curt_range )
_ , x_cut = _cut_signal( t, tool_dyn , curt_range )
_ , force_cut = _cut_signal( t, force_N , curt_range )

INDICATOR_CONFIG = {
    "id": "RMS_CV",
    "func": "Default",
    "params": {
        "n_max": 20,
        "samples_per_window": 400,
        "overlap_pct": 0.0,
        "detrend": False,
        "pad_mode": "none",
        "use_unbiased_std": True,
        "eps": 1e-12,
        "cv_threshold": 1.05,
        "rms_threshold": 0.9,
        "n_min_cv": 2,
        "warmup_ignore_alerts": False,
        "start_time": 0.05,
    },
}

sig = SignalData(
    t_cut=t_cut,
    v_cut=v_cut,
    x_cut = x_cut,
    force_cut = force_cut,
    t_original=t,
    x_original=tool_dyn,
    v_original=v,
    t_analysis=t_cut,
    signal_analysis=v_cut,
    force_original=force_N,
    path=data_dir,
    fs=fs,
    meta={"AP": "5mm-15mm",
            "RPM": 12_000,}
)

resultat_rms = run_rms_cv(sig, INDICATOR_CONFIG)

zoom_x = (4,11)
zoom_x = None
zoom_y = None
vlines = [5.365770208787228, 7.947208594272872]

plots_rms_cv(signal=sig, result=resultat_rms, show_signal=True,
             zoom_x=zoom_x, zoom_y=zoom_y, vlines=vlines, hlines=None,)
