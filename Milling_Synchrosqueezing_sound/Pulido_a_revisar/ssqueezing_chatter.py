#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ssqueezing_chatter (standalone-friendly, no ssqueeze_fast)
---------------------------------------------------------
Implementa `ssqueeze` sin depender de `ssqueeze_fast`.
Usa reasignación discreta (nearest-bin) en eje lineal.
"""

import os, sys
import numpy as np

try:
    # Modo paquete
    from .utils.time_scale_chatter import _process_fs_and_t, infer_scaletype
    from .utils.types_chatter import FunctionType
    from .utils import backend_chatter as S
    from .utils.algos_min_chatter import ssqueeze_fast, indexed_sum_onfly
    from .utils.common_min_chatter import WARN, EPS32, EPS64, NOTE, assert_is_one_of, p2up
    from .utils.stft_core_chatter import stft, logscale_transition_idx, center_frequency

except ImportError:
    # Modo script suelto
    CURDIR = os.path.dirname(os.path.abspath(__file__))
    if CURDIR not in sys.path:
        sys.path.insert(0, CURDIR)
    from utils.time_scale_chatter import _process_fs_and_t, infer_scaletype
    from utils.types_chatter import FunctionType
    from utils import backend_chatter as S
    from utils.algos_min_chatter import ssqueeze_fast, indexed_sum_onfly
    from utils.common_min_chatter import WARN, EPS32, EPS64, NOTE, assert_is_one_of, p2up
    from utils.stft_core_chatter import stft, logscale_transition_idx, center_frequency
    
def _check_ssqueezing_args(squeezing, maprange=None, wavelet=None, difftype=None,
                           difforder=None, get_w=None, transform='cwt'):
    if transform not in ('cwt', 'stft'):
        raise ValueError("`transform` must be one of: cwt, stft "
                         "(got %s)" % squeezing)

    if not isinstance(squeezing, (str, FunctionType)):
        raise TypeError("`squeezing` must be string or function "
                        "(got %s)" % type(squeezing))
    elif isinstance(squeezing, str):
        assert_is_one_of(squeezing, 'squeezing', ('sum', 'lebesgue', 'abs'))

    # maprange
    if maprange is not None:
        if isinstance(maprange, (tuple, list)):
            if not all(isinstance(m, (float, int)) for m in maprange):
                raise ValueError("all elements of `maprange` must be "
                                 "float or int")
        elif isinstance(maprange, str):
            assert_is_one_of(maprange, 'maprange', ('maximal', 'peak', 'energy'))
        else:
            raise TypeError("`maprange` must be str, tuple, or list "
                            "(got %s)" % type(maprange))

        if isinstance(maprange, str) and maprange != 'maximal':
            if transform != 'cwt':
                NOTE("string `maprange` currently only functional with "
                     "`transform='cwt'`")
            elif wavelet is None:
                raise ValueError(f"maprange='{maprange}' requires `wavelet`")

    # difftype
    if difftype is not None:
        if difftype not in ('trig', 'phase', 'numeric'):
            raise ValueError("`difftype` must be one of: direct, phase, numeric"
                             " (got %s)" % difftype)
        elif difftype != 'trig':
            # from .configs import USE_GPU
            # if USE_GPU():
            #     raise ValueError("GPU computation only supports "
            #                      "`difftype = 'trig'`")
            # elif not get_w:
            #     raise ValueError("`difftype != 'trig'` requires `get_w = True`")
            print("Note: only `difftype = 'trig'` is implemented in this version.")

    # difforder
    if difforder is not None:
        if difftype != 'numeric':
            WARN("`difforder` is ignored if `difftype != 'numeric'")
        elif difforder not in (1, 2, 4):
            raise ValueError("`difforder` must be one of: 1, 2, 4 "
                             "(got %s)" % difforder)
    elif difftype == 'numeric':
        difforder = 4

    return difforder



def ssqueeze(Wx_or_Sx, w=None, ssq_freqs=None, Sfs=None, gamma=0.0, dWx=None,
             squeezing='sum', transform='stft'):
    """Reasignación discreta tipo synchrosqueezing.
    - Si `w` es None: devuelve Wx_or_Sx (o |.| si squeezing='energy').
    - Si `w` existe: acumula sobre el bin más cercano de `ssq_freqs`.
    """
    if ssq_freqs is None:
        if Sfs is None:
            raise ValueError("ssq_freqs o Sfs deben proporcionarse.")
        ssq_freqs = Sfs
    ssq_freqs = np.asarray(ssq_freqs)

    K, N = Wx_or_Sx.shape
    Tx = np.zeros((len(ssq_freqs), N), dtype=Wx_or_Sx.dtype)

    if w is None:
        return np.abs(Wx_or_Sx) if squeezing == 'energy' else Wx_or_Sx.copy()

    # Umbral gamma se aplica en la estimación de fase previa; aquí solo mapear.
    for n in range(N):
        wk = w[:, n]
        mask = np.isfinite(wk)
        if not np.any(mask):
            continue
        wk_valid = wk[mask]
        k_idx = np.nonzero(mask)[0]

        j = np.searchsorted(ssq_freqs, wk_valid, side='left')
        j = np.clip(j, 0, len(ssq_freqs)-1)

        vals = np.abs(Wx_or_Sx[k_idx, n]) if squeezing == 'energy' else Wx_or_Sx[k_idx, n]
        for jj, val in zip(j, vals):
            Tx[jj, n] += val

    return Tx

#### `ssqueeze` utils ########################################################
def _ssq_freqrange(maprange, dt, N, wavelet, scales, was_padded):
    if isinstance(maprange, tuple):
        fm, fM = maprange
    elif maprange == 'maximal':
        dT = dt * N
        # normalized frequencies to map discrete-domain to physical:
        #     f[[cycles/samples]] -> f[[cycles/second]]
        # minimum measurable (fundamental) frequency of data
        fm = 1 / dT
        # maximum measurable (Nyquist) frequency of data
        fM = 1 / (2 * dt)
    elif maprange in ('peak', 'energy'):
        kw = dict(wavelet=wavelet, N=N, maprange=maprange, dt=dt,
                  was_padded=was_padded)
        fm = _get_center_frequency(**kw, scale=scales[-1])
        fM = _get_center_frequency(**kw, scale=scales[0])
    return fm, fM

def _compute_associated_frequencies(scales, N, wavelet, ssq_scaletype, maprange,
                                    was_padded=True, dt=1, transform='cwt'):
    fm, fM = _ssq_freqrange(maprange, dt, N, wavelet, scales, was_padded)

    na = len(scales)
    # frequency divisions `w_l` to reassign to in Synchrosqueezing
    if ssq_scaletype == 'log':
        # [fm, ..., fM]
        ssq_freqs = fm * np.power(fM / fm, np.arange(na)/(na - 1))

    elif ssq_scaletype == 'log-piecewise':
        idx = logscale_transition_idx(scales)
        if idx is None:
            ssq_freqs = fm * np.power(fM / fm, np.arange(na)/(na - 1))
        else:
            f0, f2 = fm, fM
            # note that it's possible for f1 == f0 per discretization limitations,
            # in which case `sqf1` will contain the same value repeated
            f1 = _get_center_frequency(wavelet, N, maprange, dt, scales[idx],
                                       was_padded)

            # here we don't know what the pre-downsampled `len(scales)` was,
            # so we take a longer route by piecewising respective center freqs
            t1 = np.arange(0,  na - idx - 1)/(na - 1)
            t2 = np.arange(na - idx - 1, na)/(na - 1)
            # simulates effect of "endpoint" since we'd need to know `f2`
            # with `endpoint=False`
            t1 = np.hstack([t1, t2[0]])

            sqf1 = _exp_fm(t1, f0, f1)[:-1]
            sqf2 = _exp_fm(t2, f1, f2)
            ssq_freqs = np.hstack([sqf1, sqf2])

            ssq_idx = logscale_transition_idx(ssq_freqs)
            if ssq_idx is None:
                raise Exception("couldn't find logscale transition index of "
                                "generated `ssq_freqs`; something went wrong")
            assert (na - ssq_idx) == idx, "{} != {}".format(na - ssq_idx, idx)

    else:
        if transform == 'cwt':
            ssq_freqs = np.linspace(fm, fM, na)
        elif transform == 'stft':
            ssq_freqs = np.linspace(0, .5, na) / dt
    return ssq_freqs

def _exp_fm(t, fmin, fmax):
    tmin, tmax = t.min(), t.max()
    a = (fmin**tmax / fmax**tmin) ** (1/(tmax - tmin))
    b = fmax**(1/tmax) * (1/a)**(1/tmax)
    return a*b**t


def _get_center_frequency(wavelet, N, maprange, dt, scale, was_padded):
    if was_padded:
        N = p2up(N)[0]
    kw = dict(wavelet=wavelet, N=N, scale=scale, kind=maprange)
    if maprange == 'energy':
        kw['force_int'] = True

    wc = center_frequency(**kw)
    fc = wc / (2*np.pi) / dt
    return fc




def _ssqueeze(Tx, w, Wx, dWx, nv, ssq_freqs, scales, transform, ssq_scaletype,
                cwt_scaletype, flipud, gamma, Sfs):
    if transform == 'cwt':
        # Eq 14 [2]; Eq 2.3 [1]
        if cwt_scaletype.startswith('log'):
            # ln(2)/nv == diff(ln(scales))[0] == ln(2**(1/nv))
            const = np.log(2) / nv

        elif cwt_scaletype == 'linear':
            # omit /dw since it's cancelled by *dw in inversion anyway
            const = ((scales[1] - scales[0]) / scales).squeeze()
    elif transform == 'stft':
        const = (ssq_freqs[1] - ssq_freqs[0])  # 'alpha' from [3]

    ssq_logscale = ssq_scaletype.startswith('log')
    # do squeezing by finding which frequency bin each phase transform point
    # w[a, b] lands in (i.e. to which f in ssq_freqs each w[a, b] is closest)
    # equivalent to argmin(abs(w[a, b] - ssq_freqs)) for every a, b
    # Tx[k[i, j], j] += Wx[i, j] * norm -- (see below method's docstring)
    if w is None:
        ssqueeze_fast(Wx, dWx, ssq_freqs, const, ssq_logscale, flipud,
                        gamma, out=Tx, Sfs=Sfs)
    else:
        indexed_sum_onfly(Wx, w, ssq_freqs, const, ssq_logscale, flipud,
                            out=Tx)


def ssqueeze_2(Wx, w=None, ssq_freqs=None, scales=None, Sfs=None, fs=None, t=None,
             squeezing='sum', maprange='maximal', wavelet=None, gamma=None,
             was_padded=True, flipud=False, dWx=None, transform='cwt'):
    
    def _process_args(Wx, w, fs, t, transform, squeezing, scales, maprange,
                      wavelet, dWx):
        if w is None and (dWx is None or gamma is None):
            raise ValueError("if `w` is None, `dWx` and `gamma` must not be.")
        elif w is not None and w.min() < 0:
            raise ValueError("found negatives in `w`")

        _check_ssqueezing_args(squeezing, maprange, transform=transform,
                               wavelet=wavelet)

        if scales is None and transform == 'cwt':
            raise ValueError("`scales` can't be None if `transform == 'cwt'`")

        N = Wx.shape[-1]
        dt, *_ = _process_fs_and_t(fs, t, N)
        return N, dt
    
    N, dt = _process_args(Wx, w, fs, t, transform, squeezing, scales,
                          maprange, wavelet, dWx)
    
    if transform == 'cwt':
        # scales, cwt_scaletype, _, nv = process_scales(scales, N, get_params=True)
        print("ssqueeze_2: CWT not implemented yet.")
    else:
        cwt_scaletype, nv = None, None
        
        
        # handle `ssq_freqs` & `ssq_scaletype`
    if not (isinstance(ssq_freqs, np.ndarray) or S.is_tensor(ssq_freqs)):
        if isinstance(ssq_freqs, str):
            ssq_scaletype = ssq_freqs
        else:
            # default to same scheme used by `scales`
            ssq_scaletype = cwt_scaletype

        if ((maprange == 'maximal' or isinstance(maprange, tuple)) and
                ssq_scaletype == 'log-piecewise'):
            raise ValueError("can't have `ssq_scaletype = log-piecewise` or "
                             "tuple with `maprange = 'maximal'` "
                             "(got %s)" % str(maprange))
        ssq_freqs = _compute_associated_frequencies(
            scales, N, wavelet, ssq_scaletype, maprange, was_padded, dt,
            transform)
    elif transform == 'stft':
        # removes warning per issue with `infer_scaletype`
        # future TODO: shouldn't need this
        ssq_scaletype = 'linear'
    else:
        ssq_scaletype, _ = infer_scaletype(ssq_freqs)
        
        
        # transform `Wx` if needed
    if isinstance(squeezing, FunctionType):
        Wx = squeezing(Wx)
    elif squeezing == 'lebesgue':  # from reference [3]
        Wx = S.ones(Wx.shape, dtype=Wx.dtype) / len(Wx)
    # elif squeezing == 'abs':
    #     Wx = Q.abs(Wx)
    else:
        print("ssqueeze_2: only custom squeezing functions implemented.")
        
        
        # synchrosqueeze
    Tx = S.zeros(Wx.shape, dtype=Wx.dtype)
    args = (nv, ssq_freqs, scales, transform, ssq_scaletype,
            cwt_scaletype, flipud, gamma, Sfs)
    
    if Wx.ndim == 2:
        _ssqueeze(Tx, w, Wx, dWx, *args)
    elif Wx.ndim == 3:
        w, dWx = [(g if g is not None else [None]*len(Tx))
                  for g in (w, dWx)]
        for _Tx, _w, _Wx, _dWx in zip(Tx, w, Wx, dWx):
            _ssqueeze(_Tx, _w, _Wx, _dWx, *args)

    # `scales` go high -> low
    if (transform == 'cwt' and not flipud) or flipud:
        if not isinstance(ssq_freqs, np.ndarray):
            import torch
            ssq_freqs = torch.flip(ssq_freqs, (0,))
        else:
            ssq_freqs = ssq_freqs[::-1]

    return Tx, ssq_freqs


        
    
        
        
    
        
    
    
