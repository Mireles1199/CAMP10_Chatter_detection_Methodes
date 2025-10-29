#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from typing import Optional
import logging
from numba import jit, prange
import os, sys

try:
    from .backend_chatter import asnumpy, cp, torch
    from .stft_core_chatter import logscale_transition_idx
    from .common_min_chatter import WARN, EPS32, EPS64
    import backend_chatter as S


except ImportError:
    # Modo script suelto
    CURDIR = os.path.dirname(os.path.abspath(__file__))
    if CURDIR not in sys.path:
        sys.path.insert(0, CURDIR)
        
        from backend_chatter import asnumpy, cp, torch
        import backend_chatter as S
        from stft_core_chatter import logscale_transition_idx
        from common_min_chatter import WARN, EPS32, EPS64
        
def _ensure_nonzero_nonnegative(name, x, silent=False):
    if x < EPS64:
        if not silent:
            WARN("computed `%s` (%.2e) is below EPS64; will set to " % (name, x)
                 + "EPS64. Advised to check `ssq_freqs`.")
        x = EPS64
    return x
     
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
        


def phase_stft_cpu(Sx: np.ndarray, dSx: np.ndarray, Sfs: np.ndarray, gamma: float
                   ) -> np.ndarray:
    dtype = 'float32' if Sx.dtype == np.complex64 else 'float64'
    out = np.zeros(Sx.shape, dtype=dtype)
    gamma = np.asarray(gamma, dtype=dtype)

    _phase_stft(Sx, dSx, Sfs, out, gamma)
    return out

@jit(nopython=True, cache=True)
def _phase_stft(Wx, dWx, Sfs, out, gamma):
    for i in range(Wx.shape[0]):
        for j in range(Wx.shape[1]):
            if abs(Wx[i, j]) < gamma:
                out[i, j] = np.inf
            else:
                A, B = dWx[i, j].real, dWx[i, j].imag
                C, D = Wx[i, j].real,  Wx[i, j].imag
                out[i, j] = abs(
                    Sfs[i] - (B*C - A*D) / ((C**2 + D**2) * 2*np.pi))
                
                
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
    if S.is_tensor(Wx):
        # out, params, args, kernel_kw, ssq_scaletype = outs
        # kernel = _kernel_codes[fn_name(transform, ssq_scaletype)]
        # _run_on_gpu(kernel, *args, **kernel_kw)
        # out = torch.view_as_complex(out)
        print("ssqueeze_fast: GPU not implemented yet.")
    else:
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
    if S.is_tensor(Wx):
        # out, params, args, kernel_kw, ssq_scaletype = outs
        # kernel = _kernel_codes[f'indexed_sum_{ssq_scaletype}']
        # _run_on_gpu(kernel, *args, **kernel_kw)
        # out = torch.view_as_complex(out)
        print("indexed_sum_onfly: GPU not implemented yet.")
    else:
        Wx, w, out, params, ssq_scaletype = outs
        fn = _cpu_fns[f'indexed_sum_{ssq_scaletype}']
        fn(Wx, w, out, **params)
    return out

def _process_ssq_params(Wx, w_or_dWx, ssq_freqs, const, logscale, flipud, out,
                        gamma, parallel, complex_out=True, Sfs=None):
    S.warn_if_tensor_and_par(Wx, parallel)
    gpu = S.is_tensor(Wx)
    # parallel = (parallel or IS_PARALLEL()) and not gpu

    # process `Wx`, `w_or_dWx`, `out`
    if out is None:
        out_shape = (*Wx.shape, 2) if (gpu and complex_out) else Wx.shape
        if gpu:
            out_dtype = (torch.float32 if Wx.dtype == torch.complex64 else
                         torch.float64)
            out = torch.zeros(out_shape, dtype=out_dtype, device=Wx.device)
        else:
            out = np.zeros(out_shape, dtype=Wx.dtype)
    elif complex_out and gpu:
        out = torch.view_as_real(out)
    if gpu:
        Wx = torch.view_as_real(Wx)
        if 'complex' in str(w_or_dWx.dtype):
            w_or_dWx = torch.view_as_real(w_or_dWx)

    # process `const`
    len_const = (const.numel() if isinstance(const, torch.Tensor) else
                 (const.size if isinstance(const, np.ndarray) else 1))
    if len_const != len(Wx):
        if gpu:
            const_arr = torch.full((len(Wx),), fill_value=const,
                                   device=Wx.device, dtype=Wx.dtype)
        else:
            const_arr = np.full(len(Wx), const, dtype=Wx.dtype)
    elif gpu and isinstance(const, np.ndarray):
        const_arr = torch.as_tensor(const, dtype=Wx.dtype, device=Wx.device)
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
        # process kernel params
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
        print("ssqueeze_fast: GPU not implemented yet.")
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
        print("ssqueeze_fast: GPU not implemented yet.")
    return (Wx, w_or_dWx, out, params, ssq_scaletype)


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
