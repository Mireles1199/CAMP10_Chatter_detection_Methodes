# Comentario: escalas y mapeos índice↔frecuencia
from __future__ import annotations
import numpy as np

from .misc import asnumpy
from .config_defaults import EPS64, WARN

__all__ = [
    "infer_scaletype", "logscale_transition_idx", "nv_from_scales",
    "_get_params_find_closest_log", "_ensure_nonzero_nonnegative",
]


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
