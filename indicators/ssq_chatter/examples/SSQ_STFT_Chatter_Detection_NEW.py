import logging
from typing import  Tuple
import os
import sys
import h5py
import numpy as np

from pathlib import Path

from ssq_chatter import SignalData, HDF5Reader
from ssq_chatter import run_sst_svd
from ssq_chatter import plots_sst_svd

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
    Cuts the signal to the specified time range.
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

INDICATOR_CONFIG ={
        "id": "SST_SVD",
        "func": "Default",
        "params": {
            "n_fft_power": 3,
            "win_length_ms": 40.0,
            "hop_ms": 15.0,
            "Ai_length": 3,
            "mode": "causal_inclusive",
            "sigma": 6.0,
            "frac_stable": 0.36052,
            "alpha": 0.05,
            "z": 3.0,
            "fallback_mad": False,
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
import ssq_chatter
print(ssq_chatter.__file__)

results_SST_SVD = run_sst_svd(sig, INDICATOR_CONFIG) 
zoom_x = None
zoom_y = None
vlines = [5.3]

plots_sst_svd(signal=sig, result=results_SST_SVD, show_signal=True,
            zoom_x=zoom_x, zoom_y=zoom_y, vlines=vlines, hlines=None,)