import numpy as np
from types import FunctionType
import os
import inspect
import logging
import scipy.signal as sig


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


def assert_is_one_of(x, name, supported, e=ValueError):
    if x not in supported:
        raise e("`{}` must be one of: {} (got {})".format(
            name, ', '.join(supported), x))

    
    

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


def infer_scaletype(scales):
    """Infer whether `scales` is linearly or exponentially distributed (if latter,
    also infers `nv`). Used internally on `scales` and `ssq_freqs`.

    Returns one of: 'linear', 'log', 'log-piecewise'
    """
    scales = asnumpy(scales).reshape(-1, 1)
    if not isinstance(scales, np.ndarray):
        raise TypeError("`scales` must be a numpy array (got %s)" % type(scales))
    elif scales.dtype not in (np.float32, np.float64):
        raise TypeError("`scales.dtype` must be np.float32 or np.float64 "
                        "(got %s)" % scales.dtype)

    th_log = 4e-15 if scales.dtype == np.float64 else 8e-7
    th_lin = th_log * 1e3  # less accurate for some reason

    if np.mean(np.abs(np.diff(np.log(scales), 2, axis=0))) < th_log:
        scaletype = 'log'
        # ceil to avoid faulty float-int roundoffs
        nv = int(np.round(1 / np.diff(np.log2(scales), axis=0)[0].squeeze()))

    elif np.mean(np.abs(np.diff(scales, 2, axis=0))) < th_lin:
        scaletype = 'linear'
        nv = None

    elif logscale_transition_idx(scales) is None:
        raise ValueError("could not infer `scaletype` from `scales`; "
                         "`scales` array must be linear or exponential. "
                         "(got diff(scales)=%s..." % np.diff(scales, axis=0)[:4])

    else:
        scaletype = 'log-piecewise'
        nv = nv_from_scales(scales)

    return scaletype, nv

def asnumpy(x):
    print("`asnumpy` called")
    # if is_tensor(x):
    #     return x.cpu().numpy()
    return x

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
    
    
def nv_from_scales(scales):
    """Infers `nv` from `scales` assuming `2**` scales; returns array
    of length `len(scales)` if `scaletype = 'log-piecewise'`.
    """
    scales = asnumpy(scales)
    logdiffs = 1 / np.diff(np.log2(scales), axis=0)
    nv = np.vstack([logdiffs[:1], logdiffs])

    idx = logscale_transition_idx(scales)
    if idx is not None:
        nv_transition_idx = np.argmax(np.abs(np.diff(nv, axis=0))) + 1
        assert nv_transition_idx == idx, "%s != %s" % (nv_transition_idx, idx)
    return nv


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


# ----------------- File confing.ini -----------------------
path = os.path.join(os.path.dirname(__file__), 'configs.ini')
WARN = lambda msg: logging.warning("WARNING: %s" % msg)
EPS32 = np.finfo(np.float32).eps  # machine epsilon
EPS64 = np.finfo(np.float64).eps
NOTE = lambda msg: logging.warning("NOTE: %s" % msg)  # else it's mostly ignored



def gdefaults(module_and_obj=None, get_all=False, as_dict=None,
              default_order=False, **kw):
    """Fetches default arguments from `ssqueezepy/configs.ini` and fills them
    in `kw` where they're None (or always if `get_all=True`). See code comments.
    """
    if as_dict is None:
        as_dict = bool(get_all)

    if module_and_obj is None:
        stack = inspect.stack(0)  # `(0)` faster than `()`
        obj = stack[1][3]
        module = stack[1][1].split(os.path.sep)[-1].rstrip('.py')
    else:
        # may have e.g. `utils.common.obj`
        mos = module_and_obj.split('.')
        module, obj = '.'.join(mos[:-1]), mos[-1]

    # fetch latest
    GDEFAULTS = _get_gdefaults()

    # if `module` & `obj` are found in `GDEFAULTS`, proceed to write values
    # from `GDEFAULTS` onto `kw` if `kw`'s are `None`
    # if `get_all=True`, load values from `GDEFAULTS` even if they're not in
    # `kw`, but don't overwrite those that are in `kw`.
    # if `default_order=True`, will return `kw` with keys sorted as in
    # `configs.ini`, for e.g. plotting purposes
    if module not in GDEFAULTS:
        WARN(f"module {module} not found in GDEFAULTS (see configs.ini)")
    elif obj not in GDEFAULTS[module]:
        WARN(f"object {obj} not found in GDEFAULTS['{module}'] "
             "(see configs.ini)")
    else:
        DEFAULTS = GDEFAULTS[module][obj]
        for key, value in kw.items():
            if value is None:
                kw[key] = DEFAULTS.get(key, value)

        if get_all:
            for key, value in DEFAULTS.items():
                if key not in kw:
                    kw[key] = value
        if default_order:
            # first make a dict with correct order
            # then overwrite its values with `kw`'s, without changing order
            # if `kw` has keys that `ordered_kw` doesn't, they're inserted at end
            ordered_kw = {}
            for key, value in DEFAULTS.items():
                if key in kw:  # `get_all` already accounted for
                    ordered_kw[key] = value
            ordered_kw.update(**kw)
            kw = ordered_kw

    if as_dict:
        return kw
    return (kw.values() if len(kw) != 1 else
            list(kw.values())[0])
    
    
def _get_gdefaults():
    """Global defaults fetched from configs.ini."""
    def float_if_number(s):
        """If float works, so should int."""
        if isinstance(s, (bool, type(None))):
            return s
        try:
            return float(s)
        except ValueError:
            return s

    def process_special(s):
        return {
            'None':  None,
            'True':  True,
            'False': False,
        }.get(s, s)

    def process_value(value):
        value = value.strip('"').strip("'")
        return float_if_number(process_special(value))

    with open(path, 'r') as f:
        txt = f.read().split('\n')
        txt = txt[:txt.index('#### END')]
        txt = [line.strip(' ') for line in txt if line != '']

    GDEFAULTS = {}
    module, obj = '', ''
    for line in txt:
        if line.startswith('## '):
            module = line[3:]
            GDEFAULTS[module] = {}
        elif line.startswith('# '):
            obj = line[2:]
            GDEFAULTS[module][obj] = {}
        else:
            key, value = [s.strip(' ') for s in line.split('=')]
            GDEFAULTS[module][obj][key] = process_value(value)
    return GDEFAULTS

def IS_PARALLEL():
    """Returns False if 'SSQ_PARALLEL' environment flag was set to '0', or
    if `parallel` in `configs.ini` is set to `0`; former overrides latter.
    """
    not_par_env = (os.environ.get('SSQ_PARALLEL', '1') == '0')
    if not_par_env:
        return False

    not_par_config = (gdefaults('configs.IS_PARALLEL', parallel=None) == 0)
    if not_par_config:
        return False

    return True

def USE_GPU():
    # Returns True if 'SSQ_GPU' environment flag was set to '1'.
    # if os.environ.get('SSQ_GPU', '0') == '1':
    #     if torch is None or cupy is None:
    #         raise ValueError("'SSQ_GPU' requires PyTorch and CuPy installed.")
    #     return True
    return False

# ----------------------------------------------------------------------------

def get_window(window, win_len, n_fft=None, derivative=False, dtype=None):
    """See `window` in `help(stft)`. Will return window of length `n_fft`,
    regardless of `win_len` (will pad if needed).
    """
    if n_fft is None:
        pl, pr = 0, 0
    else:
        if win_len > n_fft:
            raise ValueError("Can't have `win_len > n_fft` ({} > {})".format(
                win_len, n_fft))
        pl = (n_fft - win_len) // 2
        pr = (n_fft - win_len - pl)

    if window is not None:
        if isinstance(window, str):
            # fftbins=True -> 'periodic' window -> narrower main side-lobe and
            # closer to zero-phase in left=right padded case
            # for windows edging at 0
            window = sig.get_window(window, win_len, fftbins=True)

        elif isinstance(window, np.ndarray):
            if len(window) != win_len:
                WARN("len(window) != win_len (%s != %s)" % (len(window), win_len))

        else:
            raise ValueError("`window` must be string or np.ndarray "
                             "(got %s)" % window)
    else:
        # sym=False <-> fftbins=True (see above)
        window = sig.windows.dpss(win_len, max(4, win_len//8), sym=False)

    if len(window) < (win_len + pl + pr):
        window = np.pad(window, [pl, pr])

    if derivative:
        wf = fft(window)
        Nw = len(window)
        xi = _xifn(1, Nw)
        if Nw % 2 == 0:
            xi[Nw // 2] = 0
        # frequency-domain differentiation; see `dWx` return docs in `help(cwt)`
        diff_window = ifft(wf * 1j * xi).real

    # cast `dtype`, zero denormals (extremely small numbers that slow down CPU)
    window = _process_params_dtype(window, dtype=dtype, auto_gpu=False)
    zero_denormals(window)

    if derivative:
        diff_window = _process_params_dtype(diff_window, dtype=dtype,
                                            auto_gpu=False)
        zero_denormals(diff_window)
    return (window, diff_window) if derivative else window



# ----------------------------- File fft_utils.py -----------------------------
# -*- coding: utf-8 -*-
import numpy as np
import multiprocessing
from scipy.fft import fftshift as sfftshift, ifftshift as sifftshift
from scipy.fft import fft as sfft, rfft as srfft, ifft as sifft, irfft as sirfft
from pathlib import Path


try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(600)
except ImportError:
    pyfftw = None

UTILS_DIR = Path(__file__).parent

__all__ = [
    'fft',
    'rfft',
    'ifft',
    'irfft',
    'fftshift',
    'ifftshift',
    'FFT',
    'FFT_GLOBAL',
]

#############################################################################


class FFT():
    """Global class for ssqueezepy FFT methods.

    Will use GPU via PyTorch if environment flag `'SSQ_GPU'` is set to `'1'`.
    Will use `scipy.fft` or `pyfftw` depending on `patience` argument (and
    whether `pyfftw` is installed).
    Both will use `threads` CPUs to accelerate computing.

    In a nutshell, if you plan on re-running FFT on input of same shape and dtype,
    prefer `patience=1`, which introduces a lengthy first-time overhead but may
    compute significantly faster afterwards.

    # Arguments (`fft`, `rfft`, `ifft`, `irfft`):
        x: np.ndarray
            1D or 2D.

        axis: int
            FFT axis. One of `0, 1, -1`.

        patience: int / tuple[int, int]
            If int:
                0: will use `scipy.fft`
                1: `pyfftw` with flag `'FFTW_PATIENT'`
                2: `pyfftw` with flag `'FFTW_EXHAUSTIVE'`
            Else, if tuple, second element specifies `planning_timelimit`
            passed to `pyfftw.FFTW` (so tuple requires `patience[0] != 0`).

            Set `planning_timelimit = None` to allow planning to finish,
            but beware; `patience = 1` can take hours for large inputs, and `2`
            even longer.

        astensor: bool (default False)
            If computing on GPU, whether to return as `torch.Tensor` (if False,
            will move to CPU and convert to `numpy.ndarray`).

        n: int / None
            Only for `irfft`; length of original input. If None, will default to
            `2*(x.shape[axis] - 1)`.

    __________________________________________________________________________
    # Arguments (`__init__`):
        planning_timelimit: int
            Default.

        wisdom_dir: str
            Where to save wisdom to or load from. Empty string means
            `ssqueezepy/utils/`.

        threads: int
            Number of CPU threads to use. -1 = maximum.

        patience: int
            Default `patience`.

        cache_fft_objects: bool (default False)
            If True, `pyfftw` objects generated throughout session are stored in
            `FFT._input_history`, and retrieved if all of below match:
                `(x.shape, x.dtype, real, patience, n)`
            where `patience` includes `planning_timelimit` as a tuple.
            Default False since loading from wisdom is very fast anyway.

        verbose: bool (default True)
            Controls whether a message is printed upon `patience >= 1`.
    __________________________________________________________________________
    **Wisdom**

    `pyfftw` uses "wisdom", basically storing and reusing generated FFT plans
    if input attributes match:
        (`x.shape`, `x.dtype`, `axis`, `flags`, `planning_timelimit`)
    `flags` and `planning_timelimit` are set via `patience`.

    With each `pyfftw` use, `save_wisdom()` is called, writing to `wisdom32` and
    `wisdom64` bytes files in `ssqueezepy/utils`. Each time ssqueezepy runs in a
    new session, `load_wisdom()` is called to load these values, so wisdom is
    only expansive.
    """
    def __init__(self, planning_timelimit=120, wisdom_dir=UTILS_DIR, threads=None,
                 patience=0, cache_fft_objects=False, verbose=1):
        self.planning_timelimit = planning_timelimit
        self.wisdom_dir = wisdom_dir
        self._user_threads = threads
        self._patience = patience  # default patience
        self._process_patience(patience)  # error if !=0 and pyfftw not installed
        self.cache_fft_objects = cache_fft_objects
        self.verbose = verbose

        if pyfftw is not None:
            pyfftw.config.NUM_THREADS = self.threads

            self._wisdom32_path = str(Path(self.wisdom_dir, 'wisdom32'))
            self._wisdom64_path = str(Path(self.wisdom_dir, 'wisdom64'))
            self._wisdom32, self._wisdom64 = b'', b''
            self._input_history = {}
            self.load_wisdom()

    @property
    def threads(self):
        """Set dynamically if `threads` wasn't passed in __init__."""
        if self._user_threads is None:

            return (multiprocessing.cpu_count() if IS_PARALLEL() else 1)
            return 1
        return self._user_threads

    @property
    def patience(self):
        """Setter will also set `planning_timelimit` if setting to tuple."""
        return self._patience

    @patience.setter
    def patience(self, value):
        self._validate_patience(value)
        if isinstance(value, tuple):
            self._patience, self.planning_timelimit = value
        else:
            self._patience = value

    #### Main methods #########################################################
    def fft(self, x, axis=-1, patience=None, astensor=False):
        """See `help(ssqueezepy.utils.FFT)`."""
        out = self._maybe_gpu('fft', x, dim=axis, astensor=astensor)
        if out is not None:
            return out

        patience = self._process_patience(patience)
        if patience == 0:
            return sfft(x, axis=axis, workers=self.threads)

        fft_object = self._get_save_fill(x, axis, patience, real=False)
        return fft_object()

    def rfft(self, x, axis=-1, patience=None, astensor=False):
        """See `help(ssqueezepy.utils.FFT)`."""
        out = self._maybe_gpu('rfft', x, dim=axis, astensor=astensor)
        if out is not None:
            return out

        patience = self._process_patience(patience)
        if patience == 0:
            return srfft(x, axis=axis, workers=self.threads)

        fft_object = self._get_save_fill(x, axis, patience, real=True)
        return fft_object()

    def ifft(self, x, axis=-1, patience=None, astensor=False):
        """See `help(ssqueezepy.utils.FFT)`."""
        out = self._maybe_gpu('ifft', x, dim=axis, astensor=astensor)
        if out is not None:
            return out

        patience = self._process_patience(patience)
        if patience == 0:
            return sifft(x, axis=axis, workers=self.threads)

        fft_object = self._get_save_fill(x, axis, patience, real=False,
                                         inverse=True)
        return fft_object()

    def irfft(self, x, axis=-1, patience=None, astensor=False, n=None):
        """See `help(ssqueezepy.utils.FFT)`."""
        out = self._maybe_gpu('irfft', x, dim=axis, astensor=astensor, n=n)
        if out is not None:
            return out

        patience = self._process_patience(patience)
        if patience == 0:
            return sirfft(x, axis=axis, workers=self.threads, n=n)

        fft_object = self._get_save_fill(x, axis, patience, real=True,
                                         inverse=True, n=n)
        return fft_object()

    def fftshift(self, x, axes=-1, astensor=False):
        out = self._maybe_gpu('fftshift', x, dim=axes, astensor=astensor)
        if out is not None:
            return out
        return sfftshift(x, axes=axes)

    def ifftshift(self, x, axes=-1, astensor=False):
        out = self._maybe_gpu('ifftshift', x, dim=axes, astensor=astensor)
        if out is not None:
            return out
        return sifftshift(x, axes=axes)

    def _maybe_gpu(self, name, x, astensor=False, **kw):
        # if S.is_tensor(x):
        #     fn = {'fft': tfft, 'ifft': tifft,
        #           'rfft': trfft, 'irfft': tirfft,
        #           'fftshift': tfftshift, 'ifftshift': tifftshift}[name]
        #     out = fn(S.asarray(x), **kw)
        #     return out if astensor else out.cpu().numpy()
        return None

    #### FFT makers ###########################################################
    def _get_save_fill(self, x, axis, patience, real, inverse=False, n=None):
        fft_object = self.get_fft_object(x, axis, patience, real, inverse, n)
        self.save_wisdom()
        fft_object.input_array[:] = x
        return fft_object

    def get_fft_object(self, x, axis, patience=1, real=False, inverse=False,
                       n=None):
        combo = (x.shape, x.dtype, axis, real, n)
        if self.cache_fft_objects and combo in self._input_history:
            fft_object = self._input_history[combo]
        else:
            fft_object = self._get_fft_object(x, axis, patience, real, inverse, n)
            if self.cache_fft_objects:
                self._input_history[combo] = fft_object
        return fft_object

    def _get_fft_object(self, x, axis, patience, real, inverse, n):
        (shapes, dtypes, flags, planning_timelimit, direction
         ) = self._process_input(x, axis, patience, real, inverse, n)
        shape_in, shape_out = shapes
        dtype_in, dtype_out = dtypes

        a = pyfftw.empty_aligned(shape_in,  dtype=dtype_in)
        b = pyfftw.empty_aligned(shape_out, dtype=dtype_out)
        fft_object = pyfftw.FFTW(a, b, axes=(axis,), flags=flags,
                                 planning_timelimit=planning_timelimit,
                                 direction=direction, threads=self.threads)
        return fft_object

    def _process_input(self, x, axis, patience, real, inverse, n):
        self._validate_input(x, axis, real, patience, inverse)

        # patience, planning time, forward / inverse
        if isinstance(patience, tuple):
            patience, planning_timelimit = patience
        else:
            planning_timelimit = self.planning_timelimit
        flags = ['FFTW_PATIENT'] if patience == 1 else ['FFTW_EXHAUSTIVE']
        direction = 'FFTW_BACKWARD' if inverse else 'FFTW_FORWARD'

        # shapes
        shape_in = x.shape
        shape_out = self._get_output_shape(x, axis, real, inverse, n)

        # dtypes
        double = x.dtype in (np.float64, np.complex128)
        cdtype = 'complex128' if double else 'complex64'
        rdtype = 'float64'    if double else 'float32'
        dtype_in  = rdtype if (real and not inverse) else cdtype
        dtype_out = rdtype if (real and inverse)     else cdtype

        # notify user of procedure
        if self.verbose:
            if planning_timelimit is None:
                adjective = "very long" if patience == 2 else "long"
                print("Planning optimal FFT algorithm; this may "
                      "take %s..." % adjective)
            else:
                print("Planning optimal FFT algorithm; this will take up to "
                      "%s secs" % planning_timelimit)

        return ((shape_in, shape_out), (dtype_in, dtype_out), flags,
                planning_timelimit, direction)

    def _get_output_shape(self, x, axis, real=False, inverse=False, n=None):
        if not inverse:
            n_fft = x.shape[axis]
            fft_out_len = (n_fft//2 + 1) if real else n_fft
        else:
            if real:
                n_fft = n if (n is not None) else 2*(x.shape[axis] - 1)
            else:
                n_fft = x.shape[axis]
            fft_out_len = n_fft

        if x.ndim != 1:
            shape = list(x.shape)
            shape[axis] = fft_out_len
            shape = tuple(shape)
        else:
            shape = (fft_out_len,)
        return shape

    #### Misc #################################################################
    def load_wisdom(self):
        for name in ('wisdom32', 'wisdom64'):
            path = getattr(self, f"_{name}_path")
            if Path(path).is_file():
                with open(path, 'rb') as f:
                    setattr(self, f"_{name}", f.read())
        pyfftw.import_wisdom((self._wisdom64, self._wisdom32, b''))

    def save_wisdom(self):
        """Will overwrite."""
        self._wisdom64, self._wisdom32, _ = pyfftw.export_wisdom()
        for name in ('wisdom32', 'wisdom64'):
            path = getattr(self, f"_{name}_path")
            with open(path, 'wb') as f:
                f.write(getattr(self, f"_{name}"))

    def _validate_input(self, x, axis, real, patience, inverse):
        """Assert is single/double precision and is 1D/2D."""
        supported = ('float32', 'float64', 'complex64', 'complex128')
        dtype = str(x.dtype)
        if dtype not in supported:
            raise TypeError("unsupported `x.dtype`: %s " % dtype
                            + "(must be one of: %s)" % ', '.join(supported))
        if (real and not inverse) and dtype.startswith('complex'):
            raise TypeError("`x` cannot be complex for `rfft`")

        if axis not in (0, 1, -1):
            raise ValueError("unsupported `axis`: %s " % axis
                             + "; must be 0, 1, or -1")

        self._validate_patience(patience)

    def _validate_patience(self, patience):
        if not isinstance(patience, (int, tuple)):
            raise TypeError("`patience` must be int or tuple "
                            "(got %s)" % type(patience))
        elif isinstance(patience, int):
            # from .common import assert_is_one_of
            assert_is_one_of(patience, 'patience', (0, 1, 2))

    def _process_patience(self, patience):
        patience = patience if (patience is not None) else self.patience
        if pyfftw is None and patience != 0:
            raise ValueError("`patience != 0` requires `pyfftw` installed.")
        return patience


FFT_GLOBAL = FFT()

fft   = FFT_GLOBAL.fft
rfft  = FFT_GLOBAL.rfft
ifft  = FFT_GLOBAL.ifft
irfft = FFT_GLOBAL.irfft
fftshift  = FFT_GLOBAL.fftshift
ifftshift = FFT_GLOBAL.ifftshift
# ----------------------------------------------------------------------------

from numba import jit
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



def _process_params_dtype(*params, dtype, auto_gpu=True):
    if dtype is None:
        # dtype = S.asarray(params[0]).dtype
        print("`_process_params_dtype` called with dtype=None")
    if auto_gpu:
        # dtype = Wavelet._process_dtype(dtype, as_str=True)
        # params = [S.astype(S.asarray(p), dtype) for p in params]
        raise NotImplementedError("_process_params_dtype with auto_gpu=True")
    else:
        dtype = _process_dtype(dtype, as_str=True)
        params = [np.asarray(p).astype(dtype) for p in params]
    return params if len(params) > 1 else params[0]

DTYPES = {'float32', 'float64'}
def _process_dtype(dtype, as_str=None):
    """Ensures `dtype` is supported, and converts per `as_str` (if True,
    numpy/torch -> str, else vice versa; if None, returns as-is).
    """
    if isinstance(dtype, str):
        assert_is_one_of(dtype, 'dtype', DTYPES)
        if not as_str:
            return getattr(Q, dtype)
    # elif not isinstance(dtype, (type, np.dtype, torch.dtype)):
    #     raise TypeError("`dtype` must be string or type (np./torch.dtype) "
    #                     "(got %s)" % dtype)
    return dtype if not as_str else str(dtype).split('.')[-1]

class _Q():
    """Class for accessing `numpy` or `torch` attributes according to `USE_GPU()`.
    """
    def __getattr__(self, name):
        # if USE_GPU():
        #     return getattr(torch, name)
        # always use numpy for now
        return getattr(np, name)


##############################################################################
Q = _Q()


# ------------- file algos_utils.py -----------------------------
from numba import jit, prange
def zero_denormals(x, parallel=None):
    """Denormals are very small non-zero numbers that can significantly slow CPU
    execution (e.g. FFT). See https://github.com/scipy/scipy/issues/13764
    """
    # take a little bigger than smallest, seems to improve FFT speed
    parallel = parallel if parallel is not None else IS_PARALLEL()
    tiny = 1000 * np.finfo(x.dtype).tiny
    fn = _zero_denormals_par if parallel else _zero_denormals
    fn(x.ravel(), tiny)


@jit(nopython=True, cache=True)
def _zero_denormals(x, tiny):
    for i in range(x.size):
        if x[i] < tiny and x[i] > -tiny:
            x[i] = 0

@jit(nopython=True, cache=True, parallel=True)
def _zero_denormals_par(x, tiny):
    for i in prange(x.size):
        if x[i] < tiny and x[i] > -tiny:
            x[i] = 0
            
def phase_stft_cpu(Wx, dWx, Sfs, gamma, parallel=None):
    dtype = 'float32' if Wx.dtype == np.complex64 else 'float64'
    out = np.zeros(Wx.shape, dtype=dtype)
    gamma = np.asarray(gamma, dtype=dtype)

    parallel = parallel or IS_PARALLEL()
    fn = _phase_stft_par if parallel else _phase_stft
    fn(Wx, dWx, Sfs, out, gamma)
    return out

@jit(nopython=True, cache=True)
def _phase_stft(Wx, dWx, Sfs, out, gamma):
    print("Using _phase_stft")
    for i in range(Wx.shape[0]):
        for j in range(Wx.shape[1]):
            if abs(Wx[i, j]) < gamma:
                out[i, j] = np.inf
            else:
                A, B = dWx[i, j].real, dWx[i, j].imag
                C, D = Wx[i, j].real,  Wx[i, j].imag
                out[i, j] = abs(
                    Sfs[i] - (B*C - A*D) / ((C**2 + D**2) * 6.283185307179586))

@jit(nopython=True, cache=True, parallel=True)
def _phase_stft_par(Wx, dWx, Sfs, out, gamma):
    print("Using _phase_stft_par")
    for i in prange(Wx.shape[0]):
        for j in prange(Wx.shape[1]):
            if abs(Wx[i, j]) < gamma:
                out[i, j] = np.inf
            else:
                A, B = dWx[i, j].real, dWx[i, j].imag
                C, D = Wx[i, j].real,  Wx[i, j].imag
                out[i, j] = abs(
                    Sfs[i] - (B*C - A*D) / ((C**2 + D**2) * 6.283185307179586))
                
                
def find_maximum(fn, step_size=1e-3, steps_per_search=1e4, step_start=0,
                 step_limit=1000, min_value=-1):
    """Finds max of any function with a single maximum, and input value
    at which the maximum occurs. Inputs and outputs must be 1D.

    Must be strictly non-decreasing from step_start up to maximum of interest.
    Takes absolute value of fn's outputs.
    """
    steps_per_search = int(steps_per_search)
    largest_max = min_value
    increment = int(steps_per_search * step_size)

    input_values = np.linspace(step_start, increment)
    output_values = -1 * np.ones(steps_per_search)

    search_idx = 0
    while True:
        start = step_start + increment * search_idx
        end   = start + increment
        input_values = np.linspace(start, end, steps_per_search, endpoint=False)

        output_values[:] = np.abs(asnumpy(fn(input_values)))

        output_max = output_values.max()
        if output_max > largest_max:
            largest_max = output_max
            input_value = input_values[np.argmax(output_values)]
        elif output_max < largest_max:
            break
        search_idx += 1

        if input_values.max() > step_limit:
            raise ValueError(("could not find function maximum with given "
                              "(step_size, steps_per_search, step_start, "
                              "step_limit, min_value)=({}, {}, {}, {}, {})"
                              ).format(step_size, steps_per_search, step_start,
                                       step_limit, min_value))
    return input_value, largest_max


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


@jit(nopython=True, cache=True)
def _ssq_cwt_log_piecewise(Wx, dWx, out, const, gamma, vlmin0, vlmin1,
                           dvl0, dvl1, idx1, omax, flipud=False):
    for i in range(Wx.shape[0]):
        for j in range(Wx.shape[1]):
            if abs(Wx[i, j]) > gamma:
                A, B = dWx[i, j].real, dWx[i, j].imag
                C, D = Wx[i, j].real,  Wx[i, j].imag
                w_ij = abs((B*C - A*D) / ((C**2 + D**2) * 6.283185307179586))

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
                w_ij = abs((B*C - A*D) / ((C**2 + D**2) * 6.283185307179586))

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
                w_ij = abs((B*C - A*D) / ((C**2 + D**2) * 6.283185307179586))

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
                w_ij = abs((B*C - A*D) / ((C**2 + D**2) * 6.283185307179586))

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
                w_ij = abs((B*C - A*D) / ((C**2 + D**2) * 6.283185307179586))

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
                w_ij = abs((B*C - A*D) / ((C**2 + D**2) * 6.283185307179586))

                k = int(min(round(max((w_ij - vmin) / dv, 0)), omax))
                if flipud:
                    k = omax - k
                out[k, j] += Wx[i, j] * const[i]


@jit(nopython=True, cache=True)
def _ssq_stft(Wx, dWx, Sfs, out, const, gamma, vmin, dv, omax, flipud=False):
    for i in range(Wx.shape[0]):
        for j in range(Wx.shape[1]):
            if abs(Wx[i, j]) > gamma:
                A, B = dWx[i, j].real, dWx[i, j].imag
                C, D = Wx[i, j].real,  Wx[i, j].imag
                w_ij = abs(
                    Sfs[i] - (B*C - A*D) / ((C**2 + D**2) * 6.283185307179586))

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
                    Sfs[i] - (B*C - A*D) / ((C**2 + D**2) * 6.283185307179586))

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


            
# ---------------------------------------------------------------------

def _check_NOLA(window, hop_len, dtype=None, imprecision_strict=False):
    """https://gauss256.github.io/blog/cola.html"""
    # basic NOLA
    if hop_len > len(window):
        WARN("`hop_len > len(window)`; STFT not invertible")
    elif not sig.check_NOLA(window, len(window), len(window) - hop_len):
        WARN("`window` fails Non-zero Overlap Add (NOLA) criterion; "
             "STFT not invertible")

    # handle `dtype`; note this is just a guess, what matters is `Sx.dtype`
    if dtype is None:
        dtype = str(window.dtype)

    # check for right boundary effect: as ssqueezepy's number of output frames
    # is critically sampled (not more than needed), it creates an issue with
    # float32 and time-localized windows, which struggle to invert the last frame
    tol = 0.15 if imprecision_strict else 1e-3
    if dtype == 'float32' and not sig.check_NOLA(
            window, len(window), len(window) - hop_len, tol=tol):
        # 1e-3 can still have imprecision detectable by eye, but only upon few
        # samples, so avoid paranoia. Use 1e-2 to be safe, and 0.15 for ~exact
        WARN("Imprecision expected at right-most hop of signal, in inversion. "
             "Lower `hop_len`, choose wider `window`, or use `dtype='float64'`.")



def padsignal(x, padtype='reflect', padlength=None, get_params=False):
    """Pads signal and returns trim indices to recover original.

    # Arguments:
        x: np.ndarray / torch.Tensor
            Input vector, 1D or 2D. 2D has time in dim1, e.g. `(n_inputs, time)`.

        padtype: str
            Pad scheme to apply on input. One of:
                ('reflect', 'symmetric', 'replicate', 'wrap', 'zero').
            'zero' is most naive, while 'reflect' (default) partly mitigates
            boundary effects. See [1] & [2].

            Torch doesn't support all padding schemes, but `cwt` will still
            pad it via NumPy.

        padlength: int / None
            Number of samples to pad input to (i.e. len(x_padded) == padlength).
            Even: left = right, Odd: left = right + 1.
            Defaults to next highest power of 2 w.r.t. `len(x)`.

    # Returns:
        xp: np.ndarray
            Padded signal.
        n_up: int
            Next power of 2, or `padlength` if provided.
        n1: int
            Left  pad length.
        n2: int
            Right pad length.

    # References:
        1. Signal extension modes. PyWavelets contributors
        https://pywavelets.readthedocs.io/en/latest/ref/
        signal-extension-modes.html

        2. Wavelet Bases and Lifting Wavelets. H. Xiong.
        http://min.sjtu.edu.cn/files/wavelet/
        6-lifting%20wavelet%20and%20filterbank.pdf
    """
    def _process_args(x, padtype):
        is_numpy = bool(isinstance(x, np.ndarray))
        supported = (('zero', 'reflect', 'symmetric', 'replicate', 'wrap')
                     if is_numpy else
                     ('zero', 'reflect'))
        assert_is_one_of(padtype, 'padtype', supported)

        if not hasattr(x, 'ndim'):
            raise TypeError("`x` must be a numpy array or torch Tensor "
                            "(got %s)" % type(x))
        elif x.ndim not in (1, 2):
            raise ValueError("`x` must be 1D or 2D (got x.ndim == %s)" % x.ndim)
        return is_numpy

    is_numpy = _process_args(x, padtype)
    N = x.shape[-1]

    if padlength is None:
        # pad up to the nearest power of 2
        n_up, n1, n2 = p2up(N)
    else:
        n_up = padlength
        if abs(padlength - N) % 2 == 0:
            n1 = n2 = (n_up - N) // 2
        else:
            n2 = (n_up - N) // 2
            n1 = n2 + 1
    n_up, n1, n2 = int(n_up), int(n1), int(n2)

    # set functional spec
    if x.ndim == 1:
        pad_width = (n1, n2)
    elif x.ndim == 2:
        pad_width = ([(0, 0), (n1, n2)] if is_numpy else
                     (n1, n2))

    # comments use (n=4, n1=4, n2=3) as example, but this combination can't occur
    if is_numpy:
        if padtype == 'zero':
            # [1,2,3,4] -> [0,0,0,0, 1,2,3,4, 0,0,0]
            xp = np.pad(x, pad_width)
        elif padtype == 'reflect':
            # [1,2,3,4] -> [3,4,3,2, 1,2,3,4, 3,2,1]
            xp = np.pad(x, pad_width, mode='reflect')
        elif padtype == 'replicate':
            # [1,2,3,4] -> [1,1,1,1, 1,2,3,4, 4,4,4]
            xp = np.pad(x, pad_width, mode='edge')
        elif padtype == 'wrap':
            # [1,2,3,4] -> [1,2,3,4, 1,2,3,4, 1,2,3]
            xp = np.pad(x, pad_width, mode='wrap')
        elif padtype == 'symmetric':
            # [1,2,3,4] -> [4,3,2,1, 1,2,3,4, 4,3,2]
            if x.ndim == 1:
                xp = np.hstack([x[::-1][-n1:], x, x[::-1][:n2]])
            elif x.ndim == 2:
                xp = np.hstack([x[:, ::-1][:, -n1:], x, x[:, ::-1][:, :n2]])
    else:
        # import torch
        # mode = 'constant' if padtype == 'zero' else 'reflect'
        # if x.ndim == 1:
        #     xp = torch.nn.functional.pad(x[None], pad_width, mode)[0]
        # else:
        #     xp = torch.nn.functional.pad(x, pad_width, mode)
        print("padsignal with torch.Tensor not implemented")

    return (xp, n_up, n1, n2) if get_params else xp

def p2up(n):
    """Calculates next power of 2, and left/right padding to center
    the original `n` locations.

    # Arguments:
        n: int
            Length of original (unpadded) signal.

    # Returns:
        n_up: int
            Next power of 2.
        n1: int
            Left  pad length.
        n2: int
            Right pad length.
    """
    up = int(2**(1 + np.round(np.log2(n))))
    n2 = int((up - n) // 2)
    n1 = int(up - n - n2)
    return up, n1, n2


def buffer(x, seg_len, n_overlap, modulated=False, parallel=None):
    """Build 2D array where each column is a successive slice of `x` of length
    `seg_len` and overlapping by `n_overlap` (or equivalently incrementing
    starting index of each slice by `hop_len = seg_len - n_overlap`).

    Mimics MATLAB's `buffer`, with less functionality.

    Supports batched input with samples along dim 0, i.e. `(n_inputs, input_len)`.
    See `help(stft)` on `modulated`.

    Ex:
        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        xb = buffer(x, seg_len=5, n_overlap=3)
        xb == [[0, 1, 2, 3, 4],
               [2, 3, 4, 5, 6],
               [4, 5, 6, 7, 8]].T
    """
    # S.warn_if_tensor_and_par(x, parallel)
    assert x.ndim in (1, 2)

    hop_len = seg_len - n_overlap
    n_segs = (x.shape[-1] - seg_len) // hop_len + 1
    s20 = int(np.ceil(seg_len / 2))
    s21 = s20 - 1 if (seg_len % 2 == 1) else s20

    args = (seg_len, n_segs, hop_len, s20, s21, modulated)
    # if S.is_tensor(x):
        # if x.ndim == 1:
        #     out = _buffer_gpu(x, seg_len, n_segs, hop_len, s20, s21, modulated)

        # elif x.ndim == 2:
        #     out = x.new_zeros((len(x), seg_len, n_segs))
        #     for _x, _out in zip(x, out):
        #         _buffer_gpu(_x,  *args, out=_out)
    # else:
    parallel = parallel or IS_PARALLEL()
    fn = _buffer_par if parallel else _buffer

    if x.ndim == 1:
        out = np.zeros((seg_len, n_segs), dtype=x.dtype, order='F')
        fn(x, out, *args)

    elif x.ndim == 2:
        out = np.zeros((len(x), seg_len, n_segs), dtype=x.dtype, order='F')
        for _x, _out in zip(x, out):
            fn(_x, _out, *args)
    return out

@jit(nopython=True, cache=True)
def _buffer(x, out, seg_len, n_segs, hop_len, s20, s21, modulated=False):
    for i in range(n_segs):
        if not modulated:
            start = hop_len * i
            end   = start + seg_len
            out[:, i] = x[start:end]
        else:
            start0 = hop_len * i
            end0   = start0 + s21
            start1 = end0
            end1   = start1 + s20
            out[:s20, i] = x[start1:end1]
            out[s20:, i] = x[start0:end0]



@jit(nopython=True, cache=True, parallel=True)
def _buffer_par(x, out, seg_len, n_segs, hop_len, s20, s21, modulated=False):
    for i in prange(n_segs):
        if not modulated:
            start = hop_len * i
            end   = start + seg_len
            out[:, i] = x[start:end]
        else:
            start0 = hop_len * i
            end0   = start0 + s21
            start1 = end0
            end1   = start1 + s20
            out[:s20, i] = x[start1:end1]
            out[s20:, i] = x[start0:end0]
            
            
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







# ---------- file visual.py -----------------
from matplotlib import pyplot as plt
def plot(x, y=None, title=None, show=0, ax_equal=False, complex=0, abs=0,
         c_annot=False, w=None, h=None, dx1=False, xlims=None, ylims=None,
         vert=False, vlines=None, hlines=None, xlabel=None, ylabel=None,
         xticks=None, yticks=None, ax=None, fig=None, ticks=True, squeeze=True,
         auto_xlims=True, **kw):
    """
    norm: color norm, tuple of (vmin, vmax)
    abs: take abs(data) before plotting
    complex: plot `x.real` & `x.imag`; `2` to also plot `abs(x)`
    ticks: False to not plot x & y ticks
    w, h: rescale width & height
    kw: passed to `plt.imshow()`

    others
    """
    ax  = ax  or plt.gca()
    fig = fig or plt.gcf()

    if auto_xlims is None:
        auto_xlims = bool((x is not None and len(x) != 0) or
                          (y is not None and len(y) != 0))

    if x is None and y is None:
        raise Exception("`x` and `y` cannot both be None")
    elif x is None:
        y = y if isinstance(y, list) or not squeeze else y.squeeze()
        x = np.arange(len(y))
    elif y is None:
        x = x if isinstance(x, list) or not squeeze else x.squeeze()
        y = x
        x = np.arange(len(x))
    x = x if isinstance(x, list) or not squeeze else x.squeeze()
    y = y if isinstance(y, list) or not squeeze else y.squeeze()

    if vert:
        x, y = y, x
    if complex:
        ax.plot(x, y.real, color='tab:blue', **kw)
        ax.plot(x, y.imag, color='tab:orange', **kw)
        if complex == 2:
            ax.plot(x, np.abs(y), color='k', linestyle='--', **kw)
        if c_annot:
            _kw = dict(fontsize=15, xycoords='axes fraction', weight='bold')
            ax.annotate("real", xy=(.93, .95), color='tab:blue', **_kw)
            ax.annotate("imag", xy=(.93, .90), color='tab:orange', **_kw)
    else:
        if abs:
            y = np.abs(y)
        ax.plot(x, y, **kw)
    if dx1:
        ax.set_xticks(np.arange(len(x)))

    # styling
    if vlines:
        _vhlines(vlines, kind='v', ax=ax)
    if hlines:
        _vhlines(hlines, kind='h', ax=ax)
    if abs and ylims is None:
        ylims = (0, None)

    ticks = ticks if isinstance(ticks, (list, tuple)) else (ticks, ticks)
    if not ticks[0]:
        ax.set_xticks([])
    if not ticks[1]:
        ax.set_yticks([])
    if xticks is not None or yticks is not None:
        _ticks(xticks, yticks, ax)
    if xticks is not None or yticks is not None:
        _ticks(xticks, yticks, ax)

    _maybe_title(title, ax=ax)
    _scale_plot(fig, ax, show=show, ax_equal=ax_equal, w=w, h=h,
                xlims=xlims, ylims=ylims, dx1=(len(x) if dx1 else 0),
                xlabel=xlabel, ylabel=ylabel, auto_xlims=auto_xlims)
    
    
def _vhlines(lines, kind='v', ax=None):
    lfn = getattr(plt if ax is None else ax, f'ax{kind}line')

    if not isinstance(lines, (list, tuple)):
        lines, lkw = [lines], {}
    elif isinstance(lines, (list, np.ndarray)):
        lkw = {}
    elif isinstance(lines, tuple):
        lines, lkw = lines
        lines = lines if isinstance(lines, (list, np.ndarray)) else [lines]
    else:
        raise ValueError("`lines` must be list or (list, dict) "
                         "(got %s)" % lines)

    for line in lines:
        lfn(line, **lkw)
        
def _ticks(xticks, yticks, ax):
    def fmt(ticks):
        if all(isinstance(h, str) for h in ticks):
            return "%s"
        return ("%.d" if all(float(h).is_integer() for h in ticks) else
                "%.2f")

    if yticks is not None:
        if not hasattr(yticks, '__len__') and not yticks:
            ax.set_yticks([])
        else:
            idxs = np.linspace(0, len(yticks) - 1, 8).astype('int32')
            yt = [fmt(yticks) % h for h in np.asarray(yticks)[idxs]]
            ax.set_yticks(idxs)
            ax.set_yticklabels(yt)
    if xticks is not None:
        if not hasattr(xticks, '__len__') and not xticks:
            ax.set_xticks([])
        else:
            idxs = np.linspace(0, len(xticks) - 1, 8).astype('int32')
            xt = [fmt(xticks) % h for h in np.asarray(xticks)[idxs]]
            ax.set_xticks(idxs)
            ax.set_xticklabels(xt)
            

def _maybe_title(title, ax=None):
    if title is None:
        return

    title, kw = (title if isinstance(title, tuple) else
                 (title, {}))
    defaults = gdefaults('visuals._maybe_title', get_all=True, as_dict=True)
    for name in defaults:
        kw[name] = kw.get(name, defaults[name])

    if ax:
        ax.set_title(str(title), **kw)
    else:
        plt.title(str(title), **kw)
        
        

def _scale_plot(fig, ax, show=False, ax_equal=False, w=None, h=None,
                xlims=None, ylims=None, dx1=False, xlabel=None, ylabel=None,
                auto_xlims=True):
    if xlims:
        ax.set_xlim(*xlims)
    elif auto_xlims:
        xmin, xmax = ax.get_xlim()
        rng = xmax - xmin
        ax.set_xlim(xmin + .018 * rng, xmax - .018 * rng)

    if ax_equal:
        yabsmax = max(np.abs([*ax.get_ylim()]))
        mx = max(yabsmax, max(np.abs([xmin, xmax])))
        ax.set_xlim(-mx, mx)
        ax.set_ylim(-mx, mx)
        fig.set_size_inches(8*(w or 1), 8*(h or 1))
    if xlims:
        ax.set_xlim(*xlims)
    if ylims:
        ax.set_ylim(*ylims)
    if dx1:
        plt.xticks(np.arange(dx1))
    if w or h:
        fig.set_size_inches(14*(w or 1), 8*(h or 1))
    if xlabel is not None:
        plt.xlabel(xlabel, weight='bold', fontsize=15)
    if ylabel is not None:
        plt.ylabel(ylabel, weight='bold', fontsize=15)
    if show:
        plt.show()
        
# ------------------------------------------












