# Comentario: utilidades generales (tipos, paralelismo, numeritos)
from __future__ import annotations
from typing import Any, Union
import numpy as np
from numba import jit, prange
import os

from .config_defaults import gdefaults  # si IS_PARALLEL depende de defaults

__all__ = [
    "asnumpy", "float_if_number", "zero_denormals",
    "_zero_denormals", "_zero_denormals_par",
    "_process_params_dtype", "_process_dtype", "Q",
    "IS_PARALLEL", "USE_GPU", "find_maximum",
    "assert_is_one_of",
]


def asnumpy(x):
    print("`asnumpy` called")
    # if is_tensor(x):
    #     return x.cpu().numpy()
    return x

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

Q = _Q()

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


def assert_is_one_of(x, name, supported, e=ValueError):
    if x not in supported:
        raise e("`{}` must be one of: {} (got {})".format(
            name, ', '.join(supported), x))

