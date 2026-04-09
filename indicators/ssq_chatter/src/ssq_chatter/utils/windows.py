# Comentario: ventanas y padding de señal
from __future__ import annotations
from typing import Any
import numpy as np
import scipy.signal as sig

from .fft_utils import fft, ifft, ifftshift, _xifn
from .misc import _process_params_dtype, zero_denormals
from .config_defaults import WARN

__all__ = [
    "get_window", "_check_NOLA",
]


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
