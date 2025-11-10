# Comentario: framing/buffer y padding de señal
from __future__ import annotations
from typing import Tuple, Any
import numpy as np

from numba import jit, prange
from .misc import assert_is_one_of, IS_PARALLEL


__all__ = [
    "padsignal", "p2up",
    "buffer", "_buffer", "_buffer_par",
]

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
