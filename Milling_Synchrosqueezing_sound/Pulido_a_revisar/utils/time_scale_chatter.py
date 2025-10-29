#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from typing import Optional, Tuple, Literal
import logging

def _process_fs_and_t(fs: Optional[float], t: Optional[np.ndarray], N: int
                      ) -> Tuple[float, float, Optional[np.ndarray]]:
    if fs is not None and t is not None:
        logging.warning("`t` sobrescribe `fs` (ambos fueron pasados)." )
    if t is not None:
        if len(t) != N:
            raise ValueError(f"`t` debe tener la misma longitud que x ({len(t)} != {N}).")
        fs = float(1.0 / (t[1] - t[0]))
    else:
        if fs is None:
            fs = 1.0
        elif fs <= 0:
            raise ValueError("`fs` debe ser > 0.")
    dt = 1.0 / fs
    return dt, fs, t

def infer_scaletype(arr: np.ndarray) -> Literal["linear", "log", "unknown"]:
    arr = np.asarray(arr)
    if arr.ndim != 1 or len(arr) < 3:
        return "unknown"
    dif = np.diff(arr)
    if np.allclose(dif, dif[0], atol=1e-6, rtol=1e-6):
        return "linear"
    ratios = dif[1:] / dif[:-1]
    if np.allclose(ratios, ratios[0], atol=1e-3, rtol=1e-3):
        return "log"
    return "unknown"
