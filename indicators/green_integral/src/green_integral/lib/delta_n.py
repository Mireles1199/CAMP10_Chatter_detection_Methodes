"""Delta-n indicator computation.

``delta_n`` measures the rate of per-cycle amplitude growth:

* **Ratio mode** (default): ``delta_n = median(-LOG_CTC * (log A_{n+1} - log A_n))``
* **Theil–Sen mode**: fits ``log(A)`` vs cycle-index via Theil–Sen regression and
  returns ``delta_n = -100 * slope``.

``LOG_CTC = 100`` is the scale factor used throughout the project.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

try:
    from sklearn.linear_model import TheilSenRegressor
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

# Scale factor for the consecutive-area growth indicator
LOG_CTC: int = 100


def compute_delta_n(
    A: np.ndarray,
    t_n: np.ndarray,
    theil_sen: bool = False,
) -> Dict[str, Any]:
    """Compute the ``delta_n`` indicator for one analysis window.

    Parameters
    ----------
    A         : per-cycle area values (positive floats).
    t_n       : time stamps corresponding to each area value.
    theil_sen : when ``True`` use Theil–Sen regression instead of
                consecutive log-ratios.

    Returns
    -------
    dict with keys:

    * ``delta_n``   – scalar indicator value (``np.nan`` on failure).
    * ``t_n``       – (possibly filtered) time array.
    * ``slope``     – Theil–Sen slope (*None* in ratio mode).
    * ``intercept`` – Theil–Sen intercept (*None* in ratio mode).
    * ``r_med``     – ``exp(slope)`` (*None* in ratio mode).
    * ``r_n``       – per-step log-ratio array (*None* in Theil–Sen mode).
    * ``t_rn``      – midpoint times between consecutive cycles
                      (*None* in Theil–Sen mode).
    """
    result: Dict[str, Any] = {
        "delta_n": np.nan,
        "t_n": t_n,
        "slope": None,
        "intercept": None,
        "r_med": None,
        "r_n": None,
        "t_rn": None,
    }

    if theil_sen:
        if not _SKLEARN_OK:
            raise ImportError(
                "scikit-learn is required for Theil–Sen mode. "
                "Install it with: pip install scikit-learn"
            )
        mask = A > 0
        A_f = A[mask]
        t_f = t_n[mask]
        result["t_n"] = t_f
        if len(A_f) >= 3:
            n = np.arange(len(A_f), dtype=float)
            model = TheilSenRegressor(random_state=0)
            model.fit(n.reshape(-1, 1), np.log(A_f))
            slope = float(model.coef_[0])
            intercept = float(model.intercept_)
            result.update(
                delta_n=-100.0 * slope,
                slope=slope,
                intercept=intercept,
                r_med=float(np.exp(slope)),
            )
    else:
        if len(A) < 2:
            return result
        t_rn = (t_n[1:] + t_n[:-1]) / 2.0
        r_n = np.log(A[1:]) - np.log(A[:-1])
        result.update(
            delta_n=float(np.median(-LOG_CTC * r_n)),
            r_n=r_n,
            t_rn=t_rn,
        )

    return result
