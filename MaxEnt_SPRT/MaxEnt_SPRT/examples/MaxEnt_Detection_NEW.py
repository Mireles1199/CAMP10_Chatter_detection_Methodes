import logging
from typing import  Tuple
import os
import sys
import h5py
import numpy as np

from pathlib import Path

from MaxEnt_SPRT import SignalData
from MaxEnt_SPRT import HDF5Reader
from MaxEnt_SPRT import run_maxent_sprt
from MaxEnt_SPRT import plots_maxent_sprt

logger = logging.getLogger(__name__)

INFO_PLUS_LEVEL = 15
logging.addLevelName(INFO_PLUS_LEVEL, "INFO_PLUS")

def info_plus(self, message, *args, **kwargs):
    if self.isEnabledFor(INFO_PLUS_LEVEL):
        self._log(INFO_PLUS_LEVEL, message, args, **kwargs)

logging.Logger.info_plus = info_plus

logging.basicConfig(
    level=logging.INFO,   # DEBUG para ver todo
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def _cut_signal( t,x , time_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a portion of a signal within a specified time range.
    Parameters
    ----------
    t : np.ndarray
        Time array containing timestamp values.
    x : np.ndarray
        Signal array containing corresponding signal values.
    time_range : Tuple[float, float]
        A tuple containing (start_time, end_time) defining the time window
        for extraction. Both boundaries are inclusive.
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - t_cut : np.ndarray
            Time values within the specified range.
        - x_cut : np.ndarray
            Signal values corresponding to the time range.
    Examples
    --------
    >>> t = np.array([0, 1, 2, 3, 4, 5])
    >>> x = np.array([10, 20, 30, 40, 50, 60])
    >>> t_cut, x_cut = _cut_signal(t, x, (1, 4))
    >>> t_cut
    array([1, 2, 3, 4])
    >>> x_cut
    array([20, 30, 40, 50])
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
    "id": "MaxEnt_SPRT",                  # internal identifier 
    "func": "Default",                    # indicator wrapper
    "params": {                           # default parameters for this benchmark
        "rpm": 9_000.0,
        # "ratio_sampling": 50.0, :
        "N_seg": 5,
        "t_stable_total": 5.365770208787228,
        "alpha": 0.05,
        "beta": 0.05,
        "reset_on_H0": True,
        "cut_start_time":  1.00608,
        "cut_end_time":10.30330 ,
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

resultat_maxent_sprt = run_maxent_sprt(sig, INDICATOR_CONFIG)
time_detection = resultat_maxent_sprt.t_d[0]
print (f"Time Detection {time_detection}")

zoom_x = None
zoom_y = None

vlines = [5.3]  # tiempos de eventos importantes
plots_maxent_sprt(signal=sig, result=resultat_maxent_sprt, show_signal=True,
            zoom_x=zoom_x, zoom_y=zoom_y, vlines=None, hlines=None,)