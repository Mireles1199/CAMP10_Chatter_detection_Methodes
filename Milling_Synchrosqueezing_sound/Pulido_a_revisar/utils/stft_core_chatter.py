#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal as sig
from scipy import integrate
from numba import jit
from scipy.fft import fft, ifft, rfft, ifftshift
from typing import Optional, Any, Tuple
import logging
import os, sys



try:
    from .backend_chatter import asnumpy, cp, torch
    import backend_chatter as S
    from .algos_min_chatter import find_maximum
    from .common_min_chatter import WARN, EPS32, EPS64, NOTE, assert_is_one_of


except ImportError:
    # Modo script suelto
    CURDIR = os.path.dirname(os.path.abspath(__file__))
    if CURDIR not in sys.path:
        sys.path.insert(0, CURDIR)
        
        from backend_chatter import asnumpy, cp, torch
        import backend_chatter as S
        from algos_min_chatter import find_maximum
        from common_min_chatter import WARN, EPS32, EPS64, NOTE, assert_is_one_of
        

def padsignal(x: np.ndarray, padtype: str = 'reflect', padlength: Optional[int] = None
              ) -> np.ndarray:
    N = x.shape[-1]
    if padlength is None:
        padlength = N
    diff = padlength - N
    if diff < 0:
        logging.warning("`padlength` < N; se devuelve x sin padding.")
        return x
    n1 = diff // 2
    n2 = diff - n1
    return np.pad(x, (n1, n2), mode=padtype)

def buffer(x: np.ndarray, seg_len: int, n_overlap: int, modulated: bool = False
           ) -> np.ndarray:
    hop = seg_len - n_overlap
    n_segs = (len(x) - seg_len) // hop + 1
    out = np.zeros((seg_len, n_segs), dtype=x.dtype)
    for i in range(n_segs):
        start = hop * i
        end = start + seg_len
        out[:, i] = x[start:end]
    return out

def get_window(window: Optional[Any], win_len: int, n_fft: Optional[int] = None,
               derivative: bool = False, dtype: str = 'float32'
               ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if n_fft is None:
        n_fft = win_len
    if isinstance(window, (str, tuple)):
        w = sig.get_window(window, win_len, fftbins=True)
    elif window is None:
        w = sig.windows.hann(win_len, sym=False)
    else:
        w = window
    if len(w) < n_fft:
        pad_left = (n_fft - win_len) // 2
        pad_right = n_fft - win_len - pad_left
        w = np.pad(w, (pad_left, pad_right))
    if not derivative:
        return w.astype(dtype), None
    wf = fft(w)
    Nw = len(w)
    xi = 2 * np.pi * np.fft.fftfreq(Nw, 1 / Nw)
    dw = np.real(ifft(wf * 1j * xi)).astype(dtype)
    return w.astype(dtype), dw

def _check_NOLA(window: np.ndarray, hop_len: int, dtype: str = 'float32') -> None:
    try:
        if not sig.check_NOLA(window, len(window), len(window) - hop_len):
            logging.warning("La ventana no cumple NOLA.")
    except Exception:
        pass

def stft(x: np.ndarray,
         window: Optional[Any] = None,
         n_fft: Optional[int] = None,
         win_len: Optional[int] = None,
         hop_len: int = 1,
         fs: Optional[float] = None,
         t: Optional[np.ndarray] = None,
         padtype: str = 'reflect',
         modulated: bool = True,
         derivative: bool = False,
         dtype: str = 'float32'
         ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    N = x.shape[-1]
    if n_fft is None:
        n_fft = min(N // max(hop_len, 1), 512)
    win_len = win_len or n_fft
    w, dw = get_window(window, win_len, n_fft, derivative=True, dtype=dtype)
    _check_NOLA(w, hop_len, dtype=dtype)
    xp = padsignal(x, padtype, padlength=N)
    hop = hop_len
    S_frames = buffer(xp, n_fft, n_fft - hop)
    dS_frames = buffer(xp, n_fft, n_fft - hop)
    w_eff = ifftshift(w)
    dw_eff = ifftshift(dw) * (fs or 1)
    S_frames *= w_eff[:, None]
    dS_frames *= dw_eff[:, None]
    Sx = rfft(S_frames, axis=0)
    dSx = rfft(dS_frames, axis=0) if derivative else None
    return Sx, dSx


def logscale_transition_idx(scales):
    """Returns `idx` that splits `scales` as `[scales[:idx], scales[idx:]]`.
    """
    scales = asnumpy(scales)
    scales_diff2 = np.abs(np.diff(np.log(scales), 2, axis=0))
    idx = np.argmax(scales_diff2) + 2
    diff2_max = scales_diff2.max()
    # every other value must be zero, assert it is so
    scales_diff2[idx - 2] = 0

    th = 1e-14 if scales.dtype == np.float64 else 1e-6

    if not np.any(diff2_max > 100*np.abs(scales_diff2).mean()):
        # everything's zero, i.e. no transition detected
        return None
    elif not np.all(np.abs(scales_diff2) < th):
        # other nonzero diffs found, more than one transition point
        return None
    else:
        return idx
    
    
    #### Wavelet properties ######################################################
def aifftshift(xh):
    """Inversion also different; moves left N//2+1 bins to right."""
    if len(xh) % 2 == 0:
        return _aifftshift_even(xh, np.zeros(len(xh), dtype=xh.dtype))
    return ifftshift(xh)


@jit(nopython=True, cache=True)
def _aifftshift_even(xh, xhs):
    N = len(xh)
    for i in range(N // 2 + 1):
        xhs[i + N//2 - 1] = xh[i]
    for i in range(N // 2 + 1, N):
        xhs[i - N//2 - 1] = xh[i]
    return xhs

@jit(nopython=True, cache=True)
def _xifn(scale, N, dtype=np.float64):
    """N=128: [0, 1, 2, ..., 64, -63, -62, ..., -1] * (2*pi / N) * scale
       N=129: [0, 1, 2, ..., 64, -64, -63, ..., -1] * (2*pi / N) * scale
    """
    xi = np.zeros(N, dtype=dtype)
    h = scale * (2 * np.pi) / N
    for i in range(N // 2 + 1):
        xi[i] = i * h
    for i in range(N // 2 + 1, N):
        xi[i] = (i - N) * h
    return xi


def center_frequency(wavelet, scale=None, N=1024, kind='energy', force_int=None,
                     viz=False):
    """Center frequency (radian) of `wavelet`, either 'energy', 'peak',
    or 'peak-ct'.

    Detailed overviews:
        (1) https://dsp.stackexchange.com/a/76371/50076
        (2) https://dsp.stackexchange.com/q/72042/50076

    **Note**: implementations of `center_frequency`, `time_resolution`, and
    `freq_resolution` are discretized approximations of underlying
    continuous-time parameters. This is a flawed approach (see (1)).
      - Caution is advised for scales near minimum and maximim (obtained via
        `cwt_scalebounds(..., preset='maximal')`), where inaccuracies may be
        significant.
      - For intermediate scales and sufficiently large N (>=1024), the methods
        are reliable. May improve in the future

    # Arguments
        wavelet: wavelets.Wavelet

        scale: float / None
            Scale at which to compute `wc`; ignored if `kind='peak-ct'`.

        N: int
            Length of wavelet.

        kind: str['energy', 'peak', 'peak-ct']
            - 'energy': weighted mean of wavelet energy, or energy expectation;
              Eq 4.52 of [1]:
                wc_1     = int w |wavelet(w)|^2 dw  0..inf
                wc_scale = int (scale*w) |wavelet(scale*w)|^2 dw 0..inf
                         = wc_1 / scale
            - 'peak': value of `w` at which `wavelet` at `scale` peaks
              (is maximum) in discrete time, i.e. constrained 0 to pi.
            - 'peak-ct': value of `w` at which `wavelet` peaks (without `scale`,
              i.e. `scale=1`), i.e. peak location of the continuous-time function.
              Can be used to find `scale` at which `wavelet` is most well-behaved,
              e.g. at eighth of sampling frequency (centered between 0 and fs/4).
            - 'energy' == 'peak' for wavelets exactly even-symmetric about mode
              (peak location)

        force_int: bool / None
            Relevant only if `kind='energy'`, then defaulting to True. Set to
            False to compute via formula - i.e. first integrate at a
            "well-behaved" scale, then rescale. For intermediate scales, this
            won't yield much difference. For extremes, it matches the
            continuous-time results closer - but this isn't recommended, as it
            overlooks limitations imposed by discretization (trimmed/undersampled
            freq-domain bell).

        viz: bool (default False)
            Whether to visualize obtained center frequency.

    **Misc**

    For very high scales, 'energy' w/ `force_int=True` will match 'peak'; for
    very low scales, 'energy' will always be less than 'peak'.

    To convert to Hz:
        wc [(cycles*radians)/samples] / (2pi [radians]) * fs [samples/second]
        = fc [cycles/second]

    See tests/props_test.py for further info.

    # References
        1. Wavelet Tour of Signal Processing, 3rd ed. S. Mallat.
        https://www.di.ens.fr/~mallat/papiers/WaveletTourChap1-2-3.pdf
    """
    # def _viz(wc, params):
    #     w, psih, apsih2 = params
    #     _w = w[N//2-1:]; _psih = psih[N//2-1:]; _apsih2 = apsih2[N//2-1:]

    #     wc = wc if (kind != 'peak-ct') else pi/4
    #     vline = (wc, dict(color='tab:red', linestyle='--'))
    #     plot(_w, _psih, show=1, vlines=vline,
    #          title="psih(w)+ (frequency-domain wavelet, pos half)")
    #     plot(_w, _w * _apsih2, show=1,
    #          title="w^2 |psih(w)+|^2 (used to compute wc)")
    #     print("wc={}".format(wc))

    def _params(wavelet, scale, N):
        w = S.asarray(aifftshift(_xifn(1, N)))
        psih = asnumpy(wavelet(S.asarray(scale) * w))
        apsih2 = np.abs(psih)**2
        w = asnumpy(w)
        return w, psih, apsih2

    def _energy_wc(wavelet, scale, N, force_int):
        use_formula = not force_int
        if use_formula:
            scale_orig = scale
            wc_ct = _peak_ct_wc(wavelet, N)[0]
            scale = (4/np.pi) * wc_ct

        w, psih, apsih2 = _params(wavelet, scale, N)
        wc = (integrate.trapezoid(apsih2 * w) /
              integrate.trapezoid(apsih2))

        if use_formula:
            wc *= (scale / scale_orig)
        return float(wc), (w, psih, apsih2)

    def _peak_wc(wavelet, scale, N):
        w, psih, apsih2 = _params(wavelet, scale, N)
        wc = w[np.argmax(apsih2)]
        return float(wc), (w, psih, apsih2)

    def _peak_ct_wc(wavelet, N):
        wc, _ = find_maximum(wavelet.fn)
        # need `scale` such that `wavelet` peaks at `scale * xi.max()/4`
        # thus: `wc = scale * (pi/2)` --> `scale = (4/pi)*wc`
        scale = S.asarray((4/np.pi) * wc)
        w, psih, apsih2 = _params(wavelet, scale, N)
        return float(wc), (w, psih, apsih2)

    if force_int and 'peak' in kind:
        NOTE("`force_int` ignored with 'peak' in `kind`")
    assert_is_one_of(kind, 'kind', ('energy', 'peak', 'peak-ct'))

    if kind == 'peak-ct' and scale is not None:
        NOTE("`scale` ignored with `peak = 'peak-ct'`")

    if scale is None and kind != 'peak-ct':
        # see _peak_ct_wc
        wc, _ = find_maximum(wavelet.fn)
        scale = (4/np.pi) * wc

    # wavelet = Wavelet._init_if_not_isinstance(wavelet)
    # if kind == 'energy':
    #     force_int = force_int or True
    #     wc, params = _energy_wc(wavelet, scale, N, force_int)
    # elif kind == 'peak':
    #     wc, params = _peak_wc(wavelet, scale, N)
    # elif kind == 'peak-ct':
    #     wc, params = _peak_ct_wc(wavelet, N)

    return wc

