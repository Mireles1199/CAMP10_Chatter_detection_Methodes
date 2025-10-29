#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ssq_stft_chatter (standalone-friendly, no ssqueeze_fast)
-------------------------------------------------------
Pipeline principal que llama a STFT + fase + ssqueezing (sin ssqueeze_fast).
"""

import os, sys
from typing import Any, Tuple, Optional, Literal
import numpy as np

# --- importaciones robustas ---
try:
    # Modo paquete
    from .utils.time_scale_chatter import _process_fs_and_t, infer_scaletype
    from .utils.common_min_chatter import WARN, EPS32, EPS64
    from .utils.stft_core_chatter import stft
    from .utils.algos_min_chatter import phase_stft_cpu, phase_stft_gpu
    from .ssqueezing_chatter import ssqueeze_2, _check_ssqueezing_args
    from .utils import backend_chatter as S
except ImportError:
    # Modo script suelto
    CURDIR = os.path.dirname(os.path.abspath(__file__))
    if CURDIR not in sys.path:
        sys.path.insert(0, CURDIR)
    from utils.time_scale_chatter import _process_fs_and_t, infer_scaletype
    from utils.common_min_chatter import WARN, EPS32, EPS64
    from utils.stft_core_chatter import stft
    from utils.algos_min_chatter import phase_stft_cpu
    from ssqueezing_chatter import ssqueeze_2, _check_ssqueezing_args
    from utils import backend_chatter as S

def _make_Sfs(Sx, fs: float):
    dtype = 'float32' if 'complex64' in str(Sx.dtype) else 'float64'
    n_rows = len(Sx) if Sx.ndim == 2 else Sx.shape[1]
    return np.linspace(0, fs / 2, n_rows, endpoint=True)

def _get_gamma_default(Sx) -> float:
    return 10 * (EPS32 if getattr(Sx, "dtype", np.complex64) in (np.complex64, np.float32) else EPS64)

def phase_stft(Sx, dSx, Sfs, gamma=None, parallel=None):

    return phase_stft_cpu(Sx, dSx, Sfs, gamma)

def ssq_stft(x,
             window=None,
             n_fft: Optional[int] = None,
             win_len: Optional[int] = None,
             hop_len: int = 1,
             fs: Optional[float] = None,
             t: Optional[np.ndarray] = None,
             padtype: str = 'reflect',
             modulated: bool = True,
             dtype: str = 'float32',
             gamma: Optional[float] = None,
             ssq_freqs: Optional[np.ndarray] = None,
             squeezing: str = 'sum',
             flipud: bool = False,
             get_w: bool = False,
             get_dWx: bool = False,
             preserve_transform: bool = None,
             astensor: bool = True):
    _, fs, _ = _process_fs_and_t(fs, t, x.shape[-1])
    _check_ssqueezing_args(squeezing)
    
    
    Sx, dSx = stft(x, window=window, n_fft=n_fft, win_len=win_len, hop_len=hop_len,
                   fs=fs, t=t, padtype=padtype, modulated=modulated, dtype=dtype, derivative=True)
    
    

    
    Sx = Sx [::-1, :]
    dSx = dSx [::-1, :]
    if preserve_transform is None:
        preserve_transform = not S.is_tensor(Sx)
    if preserve_transform:
        _Sx = (Sx.copy() if not S.is_tensor(Sx) else
               Sx.detach().clone())
    else:
        _Sx = Sx
    
    Sfs = _make_Sfs(Sx, fs)
    
    if gamma is None:
        gamma = _get_gamma_default(Sx)
        
    
        # compute `w` if `get_w` and free `dWx` from memory if `not get_dWx`
    if get_w:
        w = phase_stft(_Sx, dSx, Sfs, gamma)
        _dSx = None  # don't use in `ssqueeze`
        if not get_dWx:
            dSx = None
    else:
        w = None
        _dSx = dSx   
        
    # synchrosqueeze
    if ssq_freqs is None:
        ssq_freqs = Sfs
    Tx, ssq_freqs = ssqueeze_2(_Sx, w, squeezing=squeezing, ssq_freqs=ssq_freqs,
                             Sfs=Sfs, flipud=flipud, gamma=gamma, dWx=_dSx,
                             maprange='maximal', transform='stft')
    
    # return
    if not astensor and S.is_tensor(Tx):
        # Tx, Sx, ssq_freqs, Sfs, w, dSx = [
        #     g.cpu().numpy() if S.is_tensor(g) else g
        #     for g in (Tx, Sx, ssq_freqs, Sfs, w, dSx)]
        print("ssq_stft: conversion from tensor to numpy not implemented yet.")

    if get_w and get_dWx:
        return Tx, Sx, ssq_freqs, Sfs, w, dSx
    elif get_w:
        return Tx, Sx, ssq_freqs, Sfs, w
    elif get_dWx:
        return Tx, Sx, ssq_freqs, Sfs, dSx
    else:
        return Tx, Sx, ssq_freqs, Sfs  
        
        
