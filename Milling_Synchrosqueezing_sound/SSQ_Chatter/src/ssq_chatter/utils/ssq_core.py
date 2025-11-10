# Comentario: Núcleo SSQ-STFT — algoritmo principal y kernels asociados
from __future__ import annotations
from typing import Any, Tuple
import numpy as np

from numba import jit, prange

# Dependencias internas
from .fft_utils import rfft, ifftshift
from .windows import get_window, _check_NOLA
from .buffer_utils import buffer, padsignal
from .scales import (
    infer_scaletype, _get_params_find_closest_log, _ensure_nonzero_nonnegative
)
from .config_defaults import gdefaults, EPS32, EPS64, WARN
from .misc import _process_params_dtype, Q, assert_is_one_of, IS_PARALLEL


from types import FunctionType                   
from .scales import logscale_transition_idx    

__all__ = [
    "ssq_stft_T", "stft", "phase_stft", "phase_stft_cpu",
    "ssqueeze", "ssqueeze_fast", "indexed_sum_onfly",
    "_process_ssq_params", "_make_Sfs",
    # kernels y helpers:
    "_ssq_stft", "_ssq_stft_par",
    "_indexed_sum_log", "_indexed_sum_log_par",
    "_indexed_sum_log_piecewise", "_indexed_sum_log_piecewise_par",
    "_indexed_sum_lin", "_indexed_sum_lin_par",
    "_ssq_cwt_log_piecewise", "_ssq_cwt_log_piecewise_par",
    "_ssq_cwt_log", "_ssq_cwt_log_par",
    "_ssq_cwt_lin", "_ssq_cwt_lin_par",
    "_phase_stft", "_phase_stft_par",
    "_process_fs_and_t",
    "_check_ssqueezing_args",
    "phase_stft_cpu",
    "_cpu_fns",
]




def ssq_stft_T(x, window=None, n_fft=None, win_len=None, hop_len=1, fs=None, t=None,
             modulated=True, ssq_freqs=None, padtype='reflect', squeezing='sum',
             gamma=None, preserve_transform=None, dtype=None, astensor=True,
             flipud=False, get_w=False, get_dWx=False):
    """Synchrosqueezed Short-Time Fourier Transform.
    Implements the algorithm described in Sec. III of [1].

    MATLAB docs: https://www.mathworks.com/help/signal/ref/fsst.html

    # Arguments:
        x: np.ndarray
            Input vector(s), 1D or 2D. See `help(cwt)`.

        window, n_fft, win_len, hop_len, fs, t, padtype, modulated
            See `help(stft)`.

        ssq_freqs, squeezing
            See `help(ssqueezing.ssqueeze)`.
            `ssq_freqs`, if array, must be linearly distributed.

        gamma: float / None
            See `help(ssqueezepy.ssq_cwt)`.

        preserve_transform: bool (default True)
            Whether to return `Sx` as directly output from `stft` (it might be
            altered by `ssqueeze` or `phase_transform`). Uses more memory
            per storing extra copy of `Sx`.

        dtype: str['float32', 'float64'] / None
            See `help(stft)`.

        astensor: bool (default True)
            If `'SSQ_GPU' == '1'`, whether to return arrays as on-GPU tensors
            or move them back to CPU & convert to Numpy arrays.

        flipud: bool (default False)
            See `help(ssqueeze)`.

        get_w, get_dWx
            See `help(ssq_cwt)`.
            (Named `_dWx` instead of `_dSx` for consistency.)

    # Returns:
        Tx: np.ndarray
            Synchrosqueezed STFT of `x`, of same shape as `Sx`.
        Sx: np.ndarray
            STFT of `x`. See `help(stft)`.
        ssq_freqs: np.ndarray
            Frequencies associated with rows of `Tx`.
        Sfs: np.ndarray
            Frequencies associated with rows of `Sx` (by default == `ssq_freqs`).
        w: np.ndarray (if `get_w=True`)
            Phase transform of STFT of `x`. See `help(phase_stft)`.
        dSx: np.ndarray (if `get_dWx=True`)
            Time-derivative of STFT of `x`. See `help(stft)`.

    # References:
        1. Synchrosqueezing-based Recovery of Instantaneous Frequency from
        Nonuniform Samples. G. Thakur and H.-T. Wu.
        https://arxiv.org/abs/1006.2533

        2. Synchrosqueezing Toolbox, (C) 2014--present. E. Brevdo, G. Thakur.
        https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/
        synsq_stft_fw.m
    """
    if x.ndim == 2 and get_w:
        raise NotImplementedError("`get_w=True` unsupported with batched input.")
    _, fs, _ = _process_fs_and_t(fs, t, x.shape[-1])
    _check_ssqueezing_args(squeezing)
    # assert ssq_freqs, if array, is linear
    if (isinstance(ssq_freqs, np.ndarray) and
            infer_scaletype(ssq_freqs)[0] != 'linear'):
        raise ValueError("`ssq_freqs` must be linearly distributed "
                         "for `ssq_stft`")

    Sx, dSx = stft(x, window, n_fft=n_fft, win_len=win_len, hop_len=hop_len,
                   fs=fs, padtype=padtype, modulated=modulated, derivative=True,
                   dtype=dtype)

    # preserve original `Sx` or not
    if preserve_transform is None:
        is_tensor = False
        # preserve_transform = not S.is_tensor(Sx)
        preserve_transform = not is_tensor  # always preserve for now
    if preserve_transform:
        # _Sx = (Sx.copy() if not S.is_tensor(Sx) else
        #        Sx.detach().clone())
        _Sx = (Sx.copy() if not is_tensor else
        Sx.detach().clone())
    else:
        _Sx = Sx

    # make `Sfs`
    Sfs = _make_Sfs(Sx, fs)
    # gamma
    if gamma is None:
        # gamma = 10 * (EPS64 if S.is_dtype(Sx, 'complex128') else EPS32)
        gamma = 10 * EPS32

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

    Tx, ssq_freqs = ssqueeze(_Sx, w, squeezing=squeezing, ssq_freqs=ssq_freqs,
                             Sfs=Sfs, flipud=flipud, gamma=gamma, dWx=_dSx,
                             maprange='maximal', transform='stft')
    # return
    # if not astensor and S.is_tensor(Tx):
    #     Tx, Sx, ssq_freqs, Sfs, w, dSx = [
    #         g.cpu().numpy() if S.is_tensor(g) else g
    #         for g in (Tx, Sx, ssq_freqs, Sfs, w, dSx)]

    if get_w and get_dWx:
        return Tx, Sx, ssq_freqs, Sfs, w, dSx
    elif get_w:
        return Tx, Sx, ssq_freqs, Sfs, w
    elif get_dWx:
        return Tx, Sx, ssq_freqs, Sfs, dSx
    else:
        return Tx, Sx, ssq_freqs, Sfs
    
def stft(x, window=None, n_fft=None, win_len=None, hop_len=1, fs=None, t=None,
         padtype='reflect', modulated=True, derivative=False, dtype=None):
    """Short-Time Fourier Transform.

    `modulated=True` computes "modified" variant from [1] which is advantageous
    to reconstruction & synchrosqueezing (see "Modulation" below).

    # Arguments:
        x: np.ndarray
            Input vector(s), 1D or 2D. See `help(cwt)`.

        window: str / np.ndarray / None
            STFT windowing kernel. If string, will fetch per
            `scipy.signal.get_window(window, win_len, fftbins=True)`.
            Defaults to `scipy.signal.windows.dpss(win_len, win_len//8)`;
            the DPSS window provides the best time-frequency resolution.

            Always padded to `n_fft`, so for accurate filter characteristics
            (side lobe decay, etc), best to pass in pre-designed `window`
            with `win_len == n_fft`.

        n_fft: int >= 0 / None
            FFT length, or `(STFT column length) // 2 + 1`.
            If `win_len < n_fft`, will pad `window`. Every STFT column is
            `fft(window * x_slice)`.
            Defaults to `len(x)//hop_len`, up to 512.

        win_len: int >= 0 / None
            Length of `window` to use. Used to generate a window if `window`
            is string, and ignored if it's np.ndarray.
            Defaults to `n_fft//8` or `len(window)` (if `window` is np.ndarray).

        hop_len: int > 0
            STFT stride, or number of samples to skip/hop over between subsequent
            windowings. Relates to 'overlap' as `overlap = n_fft - hop_len`.
            Must be 1 for invertible synchrosqueezed STFT.

        fs: float / None
            Sampling frequency of `x`. Defaults to 1, which makes ssq frequencies
            range from 0 to 0.5*fs, i.e. as fraction of reference sampling rate
            up to Nyquist limit. Used to compute `dSx` and `ssq_freqs`.

        t: np.ndarray / None
            Vector of times at which samples are taken (eg np.linspace(0, 1, n)).
            Must be uniformly-spaced.
            Defaults to `np.linspace(0, len(x)/fs, len(x), endpoint=False)`.
            Overrides `fs` if not None.

        padtype: str
            Pad scheme to apply on input. See `help(utils.padsignal)`.

        modulated: bool (default True)
            Whether to use "modified" variant as in [1], which centers DFT
            cisoids at the window for each shift `u`. `False` will not invert
            once synchrosqueezed.
            Recommended `True`. See "Modulation" and [2] below.

        derivative: bool (default False)
            Whether to compute and return `dSx`. Uses `fs`.

        dtype: str['float32', 'float64'] / None
            Compute precision; use 'float32` for speed & memory at expense of
            accuracy (negligible for most purposes).
            If None, uses value from `configs.ini`.

            To be safe with `'float32'`, time-localized `window`, and large
            `hop_len`, use

                from ssqueezepy._stft import _check_NOLA
                _check_NOLA(window, hop_len, 'float32', imprecision_strict=True)

    **Modulation**
        `True` will center DFT cisoids at the window for each shift `u`:
            Sm[u, k] = sum_{0}^{N-1} f[n] * g[n - u] * exp(-j*2pi*k*(n - u)/N)
        as opposed to usual STFT:
            S[u, k]  = sum_{0}^{N-1} f[n] * g[n - u] * exp(-j*2pi*k*n/N)

        Most implementations (including `scipy`, `librosa`) compute *neither*,
        but rather center the window for each slice, thus shifting DFT bases
        relative to n=0 (t=0). These create spectra that, viewed as signals, are
        of high frequency, making inversion and synchrosqueezing very unstable.
        Details & visuals: https://dsp.stackexchange.com/a/72590/50076

        Better explanation in ref [2].

    # Returns:
        Sx: [(n_fft//2 + 1) x n_hops] np.ndarray
            STFT of `x`. Positive frequencies only (+dc), via `rfft`.
            (n_hops = (len(x) - 1)//hop_len + 1)
            (rows=scales, cols=timeshifts)

        dWx: [(n_fft//2 + 1) x n_hops] np.ndarray
            Returned only if `derivative=True`.
            Time-derivative of the STFT of `x`, computed via STFT done with
            time-differentiated `window`, as in [1]. This differs from CWT's,
            where its (and Sx's) DFTs are taken along columns rather than rows.
            d/dt(window) obtained via freq-domain differentiation (help(cwt)).

    # References:
        1. Synchrosqueezing-based Recovery of Instantaneous Frequency from
        Nonuniform Samples. G. Thakur and H.-T. Wu.
        https://arxiv.org/abs/1006.2533

        2. Equivalence between "windowed Fourier transform" and STFT as
        convolutions/filtering. John Muradeli.
        https://dsp.stackexchange.com/a/86938/50076

        3. STFT: why overlapping the window? John Muradeli.
        https://dsp.stackexchange.com/a/88124/50076

        4. Synchrosqueezing Toolbox, (C) 2014--present. E. Brevdo, G. Thakur.
        https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/
        stft_fw.m 
    """
    def _stft(xp, window, diff_window, n_fft, hop_len, fs, modulated, derivative):
        Sx = buffer(xp, n_fft, n_fft - hop_len, modulated)
        if derivative:
            dSx = buffer(xp, n_fft, n_fft - hop_len, modulated)

        if modulated:
            window = ifftshift(window, astensor=True)
            if derivative:
                diff_window = ifftshift(diff_window, astensor=True) * fs

        reshape = (-1, 1) if xp.ndim == 1 else (1, -1, 1)
        Sx *= window.reshape(*reshape)
        if derivative:
            dSx *= (diff_window.reshape(*reshape))

        # keep only positive frequencies (Hermitian symmetry assuming real `x`)
        axis = 0 if xp.ndim == 1 else 1
        Sx = rfft(Sx, axis=axis, astensor=True)
        if derivative:
            dSx = rfft(dSx, axis=axis, astensor=True)
        return (Sx, dSx) if derivative else (Sx, None)

    # process args
    assert x.ndim in (1, 2)
    N = x.shape[-1]
    _, fs, _ = _process_fs_and_t(fs, t, N)
    n_fft = n_fft or min(N//hop_len, 512)

    # process `window`, make `diff_window`, check NOLA, enforce `dtype`
    if win_len is None:
        win_len = (len(window) if isinstance(window, np.ndarray) else
                   n_fft)
    dtype = gdefaults('_stft.stft', dtype=dtype)
    window, diff_window = get_window(window, win_len, n_fft, derivative=True,
                                     dtype=dtype)
    _check_NOLA(window, hop_len, dtype)
    x = _process_params_dtype(x, dtype=dtype, auto_gpu=False)

    # pad `x` to length `padlength`
    padlength = N + n_fft - 1
    xp = padsignal(x, padtype, padlength=padlength)

    # arrays -> tensors if using GPU
    # if USE_GPU():
    #     xp, window, diff_window = [torch.as_tensor(g, device='cuda') for g in
    #                                (xp, window, diff_window)]
    # take STFT
    Sx, dSx = _stft(xp, window, diff_window, n_fft, hop_len, fs, modulated,
                    derivative)

    # ensure indexing works as expected downstream (cupy)
    # Sx  = Sx.contiguous()  if is_tensor(Sx)  else Sx
    # dSx = dSx.contiguous() if is_tensor(dSx) else dSx

    return (Sx, dSx) if derivative else Sx

def _process_fs_and_t(fs, t, N):
    """Ensures `t` is uniformly-spaced and of same length as `x` (==N)
    and returns `fs` and `dt` based on it, or from defaults if `t` is None.
    """
    if fs is not None and t is not None:
        WARN("`t` will override `fs` (both were passed)")
    if t is not None:
        if len(t) != N:
            # not explicitly used anywhere but ensures wrong `t` wasn't supplied
            raise Exception("`t` must be of same length as `x` "
                            "(%s != %s)" % (len(t), N))
        elif not np.mean(np.abs(np.diff(t, 2, axis=0))) < 1e-7:  # float32 thr.
            raise Exception("Time vector `t` must be uniformly sampled.")
        fs = 1 / (t[1] - t[0])
    else:
        if fs is None:
            fs = 1
        elif fs <= 0:
            raise ValueError("`fs` must be > 0")
    dt = 1 / fs
    return dt, fs, t

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
        print("`maprange` checking currently disabled")
        # if isinstance(maprange, (tuple, list)):
        #     if not all(isinstance(m, (float, int)) for m in maprange):
        #         raise ValueError("all elements of `maprange` must be "
        #                          "float or int")
        # elif isinstance(maprange, str):
        #     assert_is_one_of(maprange, 'maprange', ('maximal', 'peak', 'energy'))
        # else:
        #     raise TypeError("`maprange` must be str, tuple, or list "
        #                     "(got %s)" % type(maprange))

        # if isinstance(maprange, str) and maprange != 'maximal':
        #     if transform != 'cwt':
        #         NOTE("string `maprange` currently only functional with "
        #              "`transform='cwt'`")
        #     elif wavelet is None:
        #         raise ValueError(f"maprange='{maprange}' requires `wavelet`")

    # difftype
    if difftype is not None:
        print("`difftype` checking currently disabled")
        # if difftype not in ('trig', 'phase', 'numeric'):
        #     raise ValueError("`difftype` must be one of: direct, phase, numeric"
        #                      " (got %s)" % difftype)
        # elif difftype != 'trig':
        #     from .configs import USE_GPU
        #     if USE_GPU():
        #         raise ValueError("GPU computation only supports "
        #                          "`difftype = 'trig'`")
        #     elif not get_w:
        #         raise ValueError("`difftype != 'trig'` requires `get_w = True`")

    # difforder
    if difforder is not None:
        print("`difforder` checking currently disabled")
        # if difftype != 'numeric':
        #     WARN("`difforder` is ignored if `difftype != 'numeric'")
        # elif difforder not in (1, 2, 4):
        #     raise ValueError("`difforder` must be one of: 1, 2, 4 "
        #                      "(got %s)" % difforder)
    elif difftype == 'numeric':
        print("Defaulting `difforder` to 4")
        difforder = 4

    return difforder



def phase_stft(Sx, dSx, Sfs, gamma=None, parallel=None):
    """Phase transform of STFT:
        w[u, k] = Im( k - d/dt(Sx[u, k]) / Sx[u, k] / (j*2pi) )

    Defined in Sec. 3 of [1]. Additionally explained in:
        https://dsp.stackexchange.com/a/72589/50076

    # Arguments:
        Sx: np.ndarray
            STFT of `x`, where `x` is 1D.

        dSx: np.ndarray
            Time-derivative of STFT of `x`

        Sfs: np.ndarray
            Associated physical frequencies, according to `dt` used in `stft`.
            Spans 0 to fs/2, linearly.

        gamma: float / None
            See `help(ssqueezepy.ssq_cwt)`.

    # Returns:
        w: np.ndarray
            Phase transform for each element of `Sx`. w.shape == Sx.shape.

    # References:
        1. Synchrosqueezing-based Recovery of Instantaneous Frequency from
        Nonuniform Samples. G. Thakur and H.-T. Wu.
        https://arxiv.org/abs/1006.2533

        2. The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications.
        G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu.
        https://arxiv.org/abs/1105.0010

        3. Synchrosqueezing Toolbox, (C) 2014--present. E. Brevdo, G. Thakur.
        https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/
        phase_stft.m
    """
    # S.warn_if_tensor_and_par(Sx, parallel)
    if gamma is None:
        # gamma = 10 * (EPS64 if S.is_dtype(Sx, 'complex128') else EPS32)
        gamma = EPS32

    # if S.is_tensor(Sx):
    #     return phase_stft_gpu(Sx, dSx, Sfs, gamma)
    return phase_stft_cpu(Sx, dSx, Sfs, gamma, parallel)


def ssqueeze(Wx, w=None, ssq_freqs=None, scales=None, Sfs=None, fs=None, t=None,
             squeezing='sum', maprange='maximal', wavelet=None, gamma=None,
             was_padded=True, flipud=False, dWx=None, transform='cwt'):
    """Synchrosqueezes the CWT or STFT of `x`.

    # Arguments:
        Wx or Sx: np.ndarray
            CWT or STFT of `x`. CWT is assumed L1-normed, and STFT with
            `modulated=True`. If 3D, will treat elements along dim0 as independent
            inputs, synchrosqueezing one-by-one (but memory-efficiently).

        w: np.ndarray / None
            Phase transform of `Wx` or `Sx`. Must be >=0.
            If None, `gamma` & `dWx` must be supplied (and `Sfs` for SSQ_STFT).

        ssq_freqs: str['log', 'log-piecewise', 'linear'] / np.ndarray / None
            Frequencies to synchrosqueeze CWT scales onto. Scale-frequency
            mapping is only approximate and wavelet-dependent.
            If None, will infer from and set to same distribution as `scales`.
            See `help(cwt)` on `'log-piecewise'`.

        scales: str['log', 'log-piecewise', 'linear', ...] / np.ndarray
            See `help(cwt)`.

        Sfs: np.ndarray
            Needed if `transform='stft'` and `dWx=None`. See `help(ssq_stft)`.

        fs: float / None
            Sampling frequency of `x`. Defaults to 1, which makes ssq
            frequencies range from 1/dT to 0.5*fs, i.e. as fraction of reference
            sampling rate up to Nyquist limit; dT = total duration (N/fs).
            Overridden by `t`, if provided.
            Relevant on `t` and `dT`: https://dsp.stackexchange.com/a/71580/50076

        t: np.ndarray / None
            Vector of times at which samples are taken (eg np.linspace(0, 1, n)).
            Must be uniformly-spaced.
            Defaults to `np.linspace(0, len(x)/fs, len(x), endpoint=False)`.
            Overrides `fs` if not None.

        squeezing: str['sum', 'lebesgue'] / function
            - 'sum': summing `Wx` according to `w`. Standard synchrosqueezing.
            Invertible.
            - 'lebesgue': as in [3], summing `Wx=ones()/len(Wx)`. Effectively,
            raw `Wx` phase is synchrosqueezed, independent of `Wx` values. Not
            recommended with CWT or STFT with `modulated=True`. Not invertible.
            For `modulated=False`, provides a more stable and accurate
            representation.
            - 'abs': summing `abs(Wx)` according to `w`. Not invertible
            (but theoretically possible to get close with least-squares estimate,
             so much "more invertible" than 'lebesgue'). Alt to 'lebesgue',
            providing same benefits while losing much less information.

            Custom function can be used to transform `Wx` arbitrarily for
            summation, e.g. `Wx**2` via `lambda x: x**2`. Output shape
            must match `Wx.shape`.

        maprange: str['maximal', 'peak', 'energy'] / tuple(float, float)
            See `help(ssq_cwt)`. Only `'maximal'` supported with STFT.

        wavelet: wavelets.Wavelet
            Only used if maprange != 'maximal' to compute center frequencies.
            See `help(cwt)`.

        gamma: float
            See `help(ssq_cwt)`.

        was_padded: bool (default `rpadded`)
            Whether `x` was padded to next power of 2 in `cwt`, in which case
            `maprange` is computed differently.
              - Used only with `transform=='cwt'`.
              - Ignored if `maprange` is tuple.

        flipud: bool (default False)
            Whether to fill `Tx` equivalently to `flipud(Tx)` (faster & less
            memory than calling `Tx = np.flipud(Tx)` afterwards).

        dWx: np.ndarray,
            Used internally by `ssq_cwt` / `ssq_stft`; must pass when `w` is None.

        transform: str['cwt', 'stft']
            Whether `Wx` is from CWT or STFT (`Sx`).

    # Returns:
        Tx: np.ndarray [nf x n]
            Synchrosqueezed CWT of `x`. (rows=~frequencies, cols=timeshifts)
            (nf = len(ssq_freqs); n = len(x))
            `nf = na` by default, where `na = len(scales)`.
        ssq_freqs: np.ndarray [nf]
            Frequencies associated with rows of `Tx`.

    # References:
        1. Synchrosqueezed Wavelet Transforms: a Tool for Empirical Mode
        Decomposition. I. Daubechies, J. Lu, H.T. Wu.
        https://arxiv.org/pdf/0912.2437.pdf

        2. The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications.
        G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu.
        https://arxiv.org/abs/1105.0010

        3. Synchrosqueezing-based Recovery of Instantaneous Frequency from
        Nonuniform Samples. G. Thakur and H.-T. Wu.
        https://arxiv.org/abs/1006.2533

        4. Synchrosqueezing Toolbox, (C) 2014--present. E. Brevdo, G. Thakur.
        https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/
        synsq_squeeze.m
    """
    def _ssqueeze(Tx, w, Wx, dWx, nv, ssq_freqs, scales, transform, ssq_scaletype,
                  cwt_scaletype, flipud, gamma, Sfs):
        if transform == 'cwt':
            print("CWT synchrosqueezing not implemented in `_ssqueeze`")
            # # Eq 14 [2]; Eq 2.3 [1]
            # if cwt_scaletype.startswith('log'):
            #     # ln(2)/nv == diff(ln(scales))[0] == ln(2**(1/nv))
            #     const = np.log(2) / nv

            # elif cwt_scaletype == 'linear':
            #     # omit /dw since it's cancelled by *dw in inversion anyway
            #     const = ((scales[1] - scales[0]) / scales).squeeze()
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
        print("CWT scales processing not implemented in `ssqueeze`")
    else:
        cwt_scaletype, nv = None, None

    # handle `ssq_freqs` & `ssq_scaletype`
    if not (isinstance(ssq_freqs, np.ndarray)):
        # if isinstance(ssq_freqs, str):
        #     ssq_scaletype = ssq_freqs
        # else:
        #     # default to same scheme used by `scales`
        #     ssq_scaletype = cwt_scaletype

        # if ((maprange == 'maximal' or isinstance(maprange, tuple)) and
        #         ssq_scaletype == 'log-piecewise'):
        #     raise ValueError("can't have `ssq_scaletype = log-piecewise` or "
        #                      "tuple with `maprange = 'maximal'` "
        #                      "(got %s)" % str(maprange))
        # ssq_freqs = _compute_associated_frequencies(
        #     scales, N, wavelet, ssq_scaletype, maprange, was_padded, dt,
        #     transform)
        print("Computation of `ssq_freqs` not implemented in `ssqueeze`")
    elif transform == 'stft':
        # removes warning per issue with `infer_scaletype`
        # future TODO: shouldn't need this
        ssq_scaletype = 'linear'
    else:
        ssq_scaletype, _ = infer_scaletype(ssq_freqs)

    # transform `Wx` if needed
    if isinstance(squeezing, FunctionType):
        print(f"ssqueeze: using custom squeezing function {squeezing}.")
        Wx = squeezing(Wx)
    elif squeezing == 'lebesgue':  # from reference [3]
        # Wx = S.ones(Wx.shape, dtype=Wx.dtype) / len(Wx)
        print("ssqueeze: 'lebesgue' squeezing not implemented.")
    elif squeezing == 'abs':
        Wx = Q.abs(Wx)

    # synchrosqueeze
    # Tx = S.zeros(Wx.shape, dtype=Wx.dtype)
    Tx = np.zeros(Wx.shape, dtype=Wx.dtype)
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
            # import torch
            # ssq_freqs = torch.flip(ssq_freqs, (0,))
            print("ssq_freqs flipping not implemented.")
        else:
            ssq_freqs = ssq_freqs[::-1]

    return Tx, ssq_freqs

def ssqueeze_fast(Wx, dWx, ssq_freqs, const, logscale=False, flipud=False,
                  gamma=None, out=None, Sfs=None, parallel=None):
    """`indexed_sum`, `find_closest`, and `phase_transform` within same loop,
    sparing two arrays and intermediate elementwise conditionals; see
    `help(algos.find_closest)` on how `k` is computed.
    """
    def fn_name(transform, ssq_scaletype):
        return ('ssq_stft' if transform == 'stft' else
                f'ssq_cwt_{ssq_scaletype}')

    outs = _process_ssq_params(Wx, dWx, ssq_freqs, const, logscale, flipud, out,
                               gamma, parallel, complex_out=True, Sfs=Sfs)
    transform = 'cwt' if Sfs is None else 'stft'
    # if S.is_tensor(Wx):
    #     out, params, args, kernel_kw, ssq_scaletype = outs
    #     kernel = _kernel_codes[fn_name(transform, ssq_scaletype)]
    #     _run_on_gpu(kernel, *args, **kernel_kw)
    #     out = torch.view_as_complex(out)
    # else:
    Wx, dWx, out, params, ssq_scaletype = outs
    fn = _cpu_fns[fn_name(transform, ssq_scaletype)]
    args = ([Wx, dWx, out] if transform == 'cwt' else
            [Wx, dWx, params.pop('Sfs'), out])
    fn(*args, **params)
    return out


def indexed_sum_onfly(Wx, w, ssq_freqs, const=1, logscale=False, flipud=False,
                      out=None, parallel=None):
    """`indexed_sum` and `find_closest` within same loop, sparing an array;
    see `help(algos.find_closest)` on how `k` is computed.
    """
    outs = _process_ssq_params(Wx, w, ssq_freqs, const, logscale, flipud, out,
                               gamma=None, parallel=parallel, complex_out=True)
    # if S.is_tensor(Wx): # GPU version
    #     out, params, args, kernel_kw, ssq_scaletype = outs
    #     kernel = _kernel_codes[f'indexed_sum_{ssq_scaletype}']
    #     _run_on_gpu(kernel, *args, **kernel_kw)
    #     out = torch.view_as_complex(out)
    # else:
    Wx, w, out, params, ssq_scaletype = outs
    fn = _cpu_fns[f'indexed_sum_{ssq_scaletype}']
    fn(Wx, w, out, **params)
    return out


def _process_ssq_params(Wx, w_or_dWx, ssq_freqs, const, logscale, flipud, out,
                        gamma, parallel, complex_out=True, Sfs=None):
    # S.warn_if_tensor_and_par(Wx, parallel)
    # gpu = S.is_tensor(Wx)
    gpu = False
    # parallel = (parallel or IS_PARALLEL()) and not gpu

    # process `Wx`, `w_or_dWx`, `out`
    if out is None:
        out_shape = (*Wx.shape, 2) if (gpu and complex_out) else Wx.shape
        if gpu:
            # out_dtype = (torch.float32 if Wx.dtype == torch.complex64 else
            #              torch.float64)
            # out = torch.zeros(out_shape, dtype=out_dtype, device=Wx.device)
            print("_process_ssq_params with gpu=True not implemented")
        else:
            out = np.zeros(out_shape, dtype=Wx.dtype)
    elif complex_out and gpu:
        # out = torch.view_as_real(out)
        print("_process_ssq_params with gpu=True not implemented")
    if gpu:
        # Wx = torch.view_as_real(Wx)
        # if 'complex' in str(w_or_dWx.dtype):
        #     w_or_dWx = torch.view_as_real(w_or_dWx)
        print("_process_ssq_params with gpu=True not implemented")

    # process `const`
    # len_const = (const.numel() if isinstance(const, torch.Tensor) else
    #              (const.size if isinstance(const, np.ndarray) else 1))
    len_const = (len(const) if isinstance(const, np.ndarray) else 1)
    if len_const != len(Wx):
        if gpu:
            # const_arr = torch.full((len(Wx),), fill_value=const,
            #                          device=Wx.device, dtype=Wx.dtype)
            print("_process_ssq_params with gpu=True not implemented")
        else:
            
                                   
            const_arr = np.full(len(Wx), const, dtype=Wx.dtype)
    elif gpu and isinstance(const, np.ndarray):
        # const_arr = torch.as_tensor(const, dtype=Wx.dtype, device=Wx.device)
        print("_process_ssq_params with gpu=True not implemented")
    else:
        const_arr = const
    const_arr = const_arr.squeeze()

    # process other constants
    if logscale:
        _, params = _get_params_find_closest_log(ssq_freqs)
    else:
        dv = float(ssq_freqs[1] - ssq_freqs[0])
        dv = _ensure_nonzero_nonnegative('dv', dv)
        params = dict(vmin=float(ssq_freqs[0]), dv=dv)

    if gpu:
        # # process kernel params
        # (blockspergrid, threadsperblock, kernel_kw, str_dtype
        #  ) = _get_kernel_params(Wx, dim=1)
        # M = kernel_kw['M']
        # kernel_kw.update(dict(f='f' if kernel_kw['dtype'] == 'float' else '',
        #                       extra=f"k = {M} - 1 - k;" if flipud else ""))

        # # collect tensors & constants
        # if 'idx1' in params:
        #     params['idx1'] = int(params['idx1'])
        # kernel_args = [Wx.data_ptr(), w_or_dWx.data_ptr(), out.data_ptr(),
        #                const_arr.data_ptr(), *list(params.values())]
        # if gamma is not None:
        #     kernel_args.insert(4, cp.asarray(gamma, dtype=str_dtype))
        # if Sfs is not None:
        #     kernel_args.insert(2, Sfs.data_ptr())

        # ssq_scaletype = (('log_piecewise' if 'idx1' in params else 'log')
        #                  if logscale else 'lin')
        print("_process_ssq_params with gpu=True not implemented")
    else:
        # cpu function params
        params.update(dict(const=const_arr, flipud=flipud, omax=len(out) - 1))
        if gamma is not None:
            params['gamma'] = gamma
        if Sfs is not None:
            params['Sfs'] = Sfs
        ssq_scaletype = (('log_piecewise' if 'idx1' in params else 'log')
                         if logscale else 'lin')
        ssq_scaletype += '_par' if parallel else ''

    if gpu:
        # args = (blockspergrid, threadsperblock, *kernel_args)
        # return (out, params, args, kernel_kw, ssq_scaletype)
        print("_process_ssq_params with gpu=True not implemented")
    return (Wx, w_or_dWx, out, params, ssq_scaletype)

def _get_params_find_closest_log(v):
    idx = logscale_transition_idx(v)
    vlmin = float(np.log2(v[0]))

    if idx is None:
        dvl = float(np.log2(v[1]) - np.log2(v[0]))
        dvl = _ensure_nonzero_nonnegative('dvl', dvl)
        params = dict(vlmin=vlmin, dvl=dvl)
    else:
        vlmin0, vlmin1 = vlmin, float(np.log2(v[idx - 1]))
        dvl0 = float(np.log2(v[1])   - np.log2(v[0]))
        dvl1 = float(np.log2(v[idx]) - np.log2(v[idx - 1]))
        # see comment above `f1` in `ssqueezing._compute_associated_frequencies`
        dvl0 = _ensure_nonzero_nonnegative('dvl0', dvl0, silent=True)
        dvl1 = _ensure_nonzero_nonnegative('dvl1', dvl1)
        idx1 = np.asarray(idx - 1, dtype=np.int32)
        params = dict(vlmin0=vlmin0, vlmin1=vlmin1, dvl0=dvl0, dvl1=dvl1,
                      idx1=idx1)
    return idx, params


def _ensure_nonzero_nonnegative(name, x, silent=False):
    if x < EPS64:
        if not silent:
            WARN("computed `%s` (%.2e) is below EPS64; will set to " % (name, x)
                 + "EPS64. Advised to check `ssq_freqs`.")
        x = EPS64
    return x

def _make_Sfs(Sx, fs):
    dtype = 'float32' if 'complex64' in str(Sx.dtype) else 'float64'
    n_rows = len(Sx) if Sx.ndim == 2 else Sx.shape[1]
    # if S.is_tensor(Sx):
    #     Sfs = torch.linspace(0, .5*fs, n_rows, device=Sx.device,
    #                          dtype=getattr(torch, dtype))
    # else:
    # not tensor suportted for now
    Sfs = np.linspace(0, .5*fs, n_rows, dtype=dtype)
    return Sfs


@jit(nopython=True, cache=True)
def _ssq_stft(Wx, dWx, Sfs, out, const, gamma, vmin, dv, omax, flipud=False):
    for i in range(Wx.shape[0]):
        for j in range(Wx.shape[1]):
            if abs(Wx[i, j]) > gamma:
                A, B = dWx[i, j].real, dWx[i, j].imag
                C, D = Wx[i, j].real,  Wx[i, j].imag
                w_ij = abs(
                    Sfs[i] - (B*C - A*D) / ((C**2 + D**2) * 2*np.pi))

                k = int(min(round(max((w_ij - vmin) / dv, 0)), omax))
                if flipud:
                    k = omax - k
                out[k, j] += Wx[i, j] * const[i]
                
                


@jit(nopython=True, cache=True, parallel=True)
def _ssq_stft_par(Wx, dWx, Sfs, out, const, gamma, vmin, dv, omax, flipud=False):
    for j in prange(Wx.shape[1]):
        for i in range(Wx.shape[0]):
            if abs(Wx[i, j]) > gamma:
                A, B = dWx[i, j].real, dWx[i, j].imag
                C, D = Wx[i, j].real,  Wx[i, j].imag
                w_ij = abs(
                    Sfs[i] - (B*C - A*D) / ((C**2 + D**2) * 2*np.pi))

                k = int(min(round(max((w_ij - vmin) / dv, 0)), omax))
                if flipud:
                    k = omax - k
                out[k, j] += Wx[i, j] * const[i]
                

@jit(nopython=True, cache=True)
def _indexed_sum_log(Wx, w, out, const, vlmin, dvl, omax, flipud=False):
    for i in range(Wx.shape[0]):
        for j in range(Wx.shape[1]):
            if np.isinf(w[i, j]):
                continue
            k = int(min(round(max((np.log2(w[i, j]) - vlmin) / dvl, 0)), omax))
            if flipud:
                k = omax - k
            out[k, j] += Wx[i, j] * const[i]

@jit(nopython=True, cache=True, parallel=True)
def _indexed_sum_log_par(Wx, w, out, const, vlmin, dvl, omax, flipud=False):
    for j in prange(Wx.shape[1]):
        for i in range(Wx.shape[0]):
            if np.isinf(w[i, j]):
                continue
            k = int(min(round(max((np.log2(w[i, j]) - vlmin) / dvl, 0)), omax))
            if flipud:
                k = omax - k
            out[k, j] += Wx[i, j] * const[i]


@jit(nopython=True, cache=True)
def _indexed_sum_log_piecewise(Wx, w, out, const, vlmin0, vlmin1, dvl0, dvl1,
                               idx1, omax, flipud=False):
    for i in range(Wx.shape[0]):
        for j in range(Wx.shape[1]):
            if np.isinf(w[i, j]):
                continue
            wl = np.log2(w[i, j])
            if wl > vlmin1:
                k = int(min(round((wl - vlmin1) / dvl1) + idx1, omax))
            else:
                k = int(round(max((wl - vlmin0) / dvl0, 0)))
            if flipud:
                k = omax - k
            out[k, j] += Wx[i, j] * const[i]

@jit(nopython=True, cache=True, parallel=True)
def _indexed_sum_log_piecewise_par(Wx, w, out, const, vlmin0, vlmin1, dvl0, dvl1,
                                   idx1, omax, flipud=False):
    # it's also possible to construct the if-else logic in terms of mappables
    # of `vlmin`, `dvl`, and `idx`, which generalizes to any number of transitions
    for j in prange(Wx.shape[1]):
        for i in range(Wx.shape[0]):
            if np.isinf(w[i, j]):
                continue
            wl = np.log2(w[i, j])
            if wl > vlmin1:
                k = int(min(round((wl - vlmin1) / dvl1) + idx1, omax))
            else:
                k = int(round(max((wl - vlmin0) / dvl0, 0)))
            if flipud:
                k = omax - k
            out[k, j] += Wx[i, j] * const[i]


@jit(nopython=True, cache=True)
def _indexed_sum_lin(Wx, w, out, const, vmin, dv, omax, flipud=False):
    for i in range(Wx.shape[0]):
        for j in range(Wx.shape[1]):
            if np.isinf(w[i, j]):
                continue
            k = int(min(round(max((w[i, j] - vmin) / dv, 0)), omax))
            if flipud:
                k = omax - k
            out[k, j] += Wx[i, j] * const[i]

@jit(nopython=True, cache=True, parallel=True)
def _indexed_sum_lin_par(Wx, w, out, const, vmin, dv, omax, flipud=False):
    for j in prange(Wx.shape[1]):
        for i in range(Wx.shape[0]):
            if np.isinf(w[i, j]):
                continue
            k = int(min(round(max((w[i, j] - vmin) / dv, 0)), omax))
            if flipud:
                k = omax - k
            out[k, j] += Wx[i, j] * const[i]



@jit(nopython=True, cache=True)
def _ssq_cwt_log_piecewise(Wx, dWx, out, const, gamma, vlmin0, vlmin1,
                           dvl0, dvl1, idx1, omax, flipud=False):
    for i in range(Wx.shape[0]):
        for j in range(Wx.shape[1]):
            if abs(Wx[i, j]) > gamma:
                A, B = dWx[i, j].real, dWx[i, j].imag
                C, D = Wx[i, j].real,  Wx[i, j].imag
                w_ij = abs((B*C - A*D) / ((C**2 + D**2) * 2*np.pi))

                wl = np.log2(w_ij)
                if wl > vlmin1:
                    k = int(min(round((wl - vlmin1) / dvl1) + idx1, omax))
                else:
                    k = int(max(round((wl - vlmin0) / dvl0), 0))
                if flipud:
                    k = omax - k
                out[k, j] += Wx[i, j] * const[i]

@jit(nopython=True, cache=True, parallel=True)
def _ssq_cwt_log_piecewise_par(Wx, dWx, out, const, gamma, vlmin0, vlmin1,
                               dvl0, dvl1, idx1, omax, flipud=False):
    for j in prange(Wx.shape[1]):
        for i in range(Wx.shape[0]):
            if abs(Wx[i, j]) > gamma:
                A, B = dWx[i, j].real, dWx[i, j].imag
                C, D = Wx[i, j].real,  Wx[i, j].imag
                w_ij = abs((B*C - A*D) / ((C**2 + D**2) * 2*np.pi))

                wl = np.log2(w_ij)
                if wl > vlmin1:
                    k = int(min(round((wl - vlmin1) / dvl1) + idx1, omax))
                else:
                    k = int(max(round((wl - vlmin0) / dvl0), 0))
                if flipud:
                    k = omax - k
                out[k, j] += Wx[i, j] * const[i]


@jit(nopython=True, cache=True)
def _ssq_cwt_log(Wx, dWx, out, const, gamma, vlmin, dvl, omax, flipud=False):
    for i in range(Wx.shape[0]):
        for j in range(Wx.shape[1]):
            if abs(Wx[i, j]) > gamma:
                A, B = dWx[i, j].real, dWx[i, j].imag
                C, D = Wx[i, j].real,  Wx[i, j].imag
                w_ij = abs((B*C - A*D) / ((C**2 + D**2) * 2*np.pi))

                k = int(min(round(max((np.log2(w_ij) - vlmin) / dvl, 0)), omax))
                if flipud:
                    k = omax - k
                out[k, j] += Wx[i, j] * const[i]

@jit(nopython=True, cache=True, parallel=True)
def _ssq_cwt_log_par(Wx, dWx, out, const, gamma, vlmin, dvl, omax, flipud=False):
    for j in prange(Wx.shape[1]):
        for i in range(Wx.shape[0]):
            if abs(Wx[i, j]) > gamma:
                A, B = dWx[i, j].real, dWx[i, j].imag
                C, D = Wx[i, j].real,  Wx[i, j].imag
                w_ij = abs((B*C - A*D) / ((C**2 + D**2) * 2*np.pi))

                k = int(min(round(max((np.log2(w_ij) - vlmin) / dvl, 0)), omax))
                if flipud:
                    k = omax - k
                out[k, j] += Wx[i, j] * const[i]


@jit(nopython=True, cache=True)
def _ssq_cwt_lin(Wx, dWx, out, const, gamma, vmin, dv, omax, flipud=False):
    for i in range(Wx.shape[0]):
        for j in range(Wx.shape[1]):
            if abs(Wx[i, j]) > gamma:
                A, B = dWx[i, j].real, dWx[i, j].imag
                C, D = Wx[i, j].real,  Wx[i, j].imag
                w_ij = abs((B*C - A*D) / ((C**2 + D**2) * 2*np.pi))

                k = int(min(round(max((w_ij - vmin) / dv, 0)), omax))
                if flipud:
                    k = omax - k
                out[k, j] += Wx[i, j] * const[i]

@jit(nopython=True, cache=True, parallel=True)
def _ssq_cwt_lin_par(Wx, dWx, out, const, gamma, vmin, dv, omax, flipud=False):
    for j in prange(Wx.shape[1]):
        for i in range(Wx.shape[0]):
            if abs(Wx[i, j]) > gamma:
                A, B = dWx[i, j].real, dWx[i, j].imag
                C, D = Wx[i, j].real,  Wx[i, j].imag
                w_ij = abs((B*C - A*D) / ((C**2 + D**2) * 2*np.pi))

                k = int(min(round(max((w_ij - vmin) / dv, 0)), omax))
                if flipud:
                    k = omax - k
                out[k, j] += Wx[i, j] * const[i]
                
                

@jit(nopython=True, cache=True)
def _phase_stft(Wx, dWx, Sfs, out, gamma):
    print("Inside _phase_stft")
    for i in range(Wx.shape[0]):
        for j in range(Wx.shape[1]):
            if abs(Wx[i, j]) < gamma:
                out[i, j] = np.inf
            else:
                A, B = dWx[i, j].real, dWx[i, j].imag
                C, D = Wx[i, j].real,  Wx[i, j].imag
                out[i, j] = abs(
                    Sfs[i] - (B*C - A*D) / ((C**2 + D**2) * 2*np.pi))

@jit(nopython=True, cache=True, parallel=True)
def _phase_stft_par(Wx, dWx, Sfs, out, gamma):
    print("Inside _phase_stft_par")
    for i in prange(Wx.shape[0]):
        for j in prange(Wx.shape[1]):
            if abs(Wx[i, j]) < gamma:
                out[i, j] = np.inf
            else:
                A, B = dWx[i, j].real, dWx[i, j].imag
                C, D = Wx[i, j].real,  Wx[i, j].imag
                out[i, j] = abs(
                    Sfs[i] - (B*C - A*D) / ((C**2 + D**2) * 2*np.pi))
                
def phase_stft_cpu(Wx, dWx, Sfs, gamma, parallel=None):
    dtype = 'float32' if Wx.dtype == np.complex64 else 'float64'
    out = np.zeros(Wx.shape, dtype=dtype)
    gamma = np.asarray(gamma, dtype=dtype)

    parallel = parallel or IS_PARALLEL()
    fn = _phase_stft_par if parallel else _phase_stft
    fn(Wx, dWx, Sfs, out, gamma)
    return out

#### CPU funcs & GPU kernel codes ############################################
_cpu_fns = {
    'ssq_cwt_log_piecewise':     _ssq_cwt_log_piecewise,
    'ssq_cwt_log_piecewise_par': _ssq_cwt_log_piecewise_par,
    'ssq_cwt_log':               _ssq_cwt_log,
    'ssq_cwt_log_par':           _ssq_cwt_log_par,
    'ssq_cwt_lin':               _ssq_cwt_lin,
    'ssq_cwt_lin_par':           _ssq_cwt_lin_par,

    'ssq_stft':     _ssq_stft,
    'ssq_stft_par': _ssq_stft_par,

    'indexed_sum_log_piecewise':     _indexed_sum_log_piecewise,
    'indexed_sum_log_piecewise_par': _indexed_sum_log_piecewise_par,
    'indexed_sum_log':               _indexed_sum_log,
    'indexed_sum_log_par':           _indexed_sum_log_par,
    'indexed_sum_lin':               _indexed_sum_lin,
    'indexed_sum_lin_par':           _indexed_sum_lin_par,
}