from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import TheilSenRegressor
import math


@dataclass
class RALEConfig:
    name: str
    spin_rate: float
    n_teeth: int
    T_modal: float
    d_t: float
    window_overlap: float = 0.0
    lambda_ewma: float = 0.2
    area_noise_eps: float = 1e-30
    area_noise_n_cycles: int = 0
    sigma_method: str = "frozen_time"
    sigma_local_n_windows: int = 5
    sigma_fit_method: str = "ols"


class SimpleDebug:
    def __init__(self, level: int = 0, window_range: Tuple[int, int] = (0, 0)):
        self.level = level
        self.window_range = window_range

    def log(self, msg: str, level: int = 1):
        if self.level >= level:
            print(msg)

    def log_window_progress(self, k: int, total: int = 0):
        if self.level >= 1:
            pass

    def is_window_in_debug_range(self, k: int) -> bool:
        lo, hi = self.window_range
        return lo <= k <= hi


# Helper functions ported to match Areas_Indicator_V1.py behavior

def _compute_delta_signal(t: np.ndarray, q: np.ndarray, q_o: np.ndarray, T_regen: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    interp_q = interp1d(t, q, bounds_error=False, fill_value=np.nan, kind='linear')
    interp_qo = interp1d(t, q_o, bounds_error=False, fill_value=np.nan, kind='linear')
    dq = q - interp_q(t - T_regen)
    dqo = q_o - interp_qo(t - T_regen)
    valid = ~(np.isnan(dq) | np.isnan(dqo))
    return dq[valid], dqo[valid], t[valid]


def _build_rale_windows(n_valid: int, N_T_regen: int, window_overlap: float) -> List[Dict]:
    step = max(1, int(round((1.0 - window_overlap) * N_T_regen)))
    windows: List[Dict] = []
    i = 0
    num = 0
    while i + N_T_regen <= n_valid:
        windows.append({"num_window": num, "i_start": i, "i_end": i + N_T_regen})
        i += step
        num += 1
    return windows


def _compute_cycle_area(dx: np.ndarray, dxdot: np.ndarray) -> float:
    if len(dx) < 3:
        return np.nan
    x, y = dx, dxdot
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _estimate_sigma(
    areas: np.ndarray,
    t_wins: np.ndarray,
    T_regen: float,
    eps: float,
    local_n_windows: int = 5,
    sigma_method: str = "frozen_time",
    fit_method: str = "ols",
) -> np.ndarray:
    A = np.asarray(areas, dtype=float)
    t = np.asarray(t_wins, dtype=float)
    sigma = np.full_like(A, np.nan)

    if len(A) == 0:
        return sigma

    method = str(sigma_method).strip().lower()
    fit_kind = str(fit_method).strip().lower()

    if method == "ratio":
        log_A = np.where(A > eps, np.log(np.where(A > eps, A, 1.0)), np.nan)
        sigma[1:] = (log_A[1:] - log_A[:-1]) / (2.0 * T_regen)
        return sigma

    n_local = max(3, int(local_n_windows))
    if n_local % 2 == 0:
        n_local += 1
    half = n_local // 2

    for k in range(len(A)):
        i0 = max(0, k - half)
        i1 = min(len(A), k + half + 1)

        A_loc = A[i0:i1]
        t_loc = t[i0:i1]
        valid = np.isfinite(A_loc) & np.isfinite(t_loc) & (A_loc > eps)
        if np.count_nonzero(valid) < 2:
            continue

        y_fit = np.log(A_loc[valid])
        x_fit = t_loc[valid]

        if fit_kind == "theilsen" and len(x_fit) >= 3:
            model = TheilSenRegressor(random_state=0)
            model.fit(x_fit.reshape(-1, 1), y_fit)
            slope = float(model.coef_[0])
        else:
            slope, _ = np.polyfit(x_fit, y_fit, 1)
            slope = float(slope)

        sigma[k] = 0.5 * slope

    return sigma


def _apply_ewma(sigma: np.ndarray, lam: float) -> np.ndarray:
    out = np.full_like(sigma, np.nan)
    s_prev = np.nan
    for i, s in enumerate(sigma):
        if np.isnan(s):
            out[i] = s_prev
        elif np.isnan(s_prev):
            out[i] = s
        else:
            out[i] = (1.0 - lam) * s_prev + lam * s
        s_prev = out[i]
    return out


def _integrate_G(sigma_ewma: np.ndarray, t_sigma: np.ndarray) -> np.ndarray:
    sigma_arr = np.asarray(sigma_ewma, dtype=float)
    t_arr = np.asarray(t_sigma, dtype=float)
    G_hat = np.zeros_like(sigma_arr, dtype=float)

    if len(sigma_arr) == 0:
        return G_hat

    for i in range(1, len(sigma_arr)):
        s_prev = 0.0 if np.isnan(sigma_arr[i - 1]) else sigma_arr[i - 1]
        s_curr = 0.0 if np.isnan(sigma_arr[i]) else sigma_arr[i]
        dt = max(0.0, float(t_arr[i] - t_arr[i - 1]))
        G_hat[i] = G_hat[i - 1] + 0.5 * (s_prev + s_curr) * dt

    return G_hat


def run_rale(t: np.ndarray, q: np.ndarray, q_o: np.ndarray, cfg: RALEConfig, dbg: Optional[SimpleDebug] = None) -> Dict[str, Any]:
    if dbg is None:
        dbg = SimpleDebug(level=0)

    T_regen = 60.0 / (cfg.spin_rate * cfg.n_teeth)
    d_t = cfg.d_t
    N_T_regen = max(2, int(round(T_regen / d_t)))
    window_overlap = float(cfg.window_overlap)
    lam = float(cfg.lambda_ewma)
    eps = float(cfg.area_noise_eps)
    noise_cycles = int(cfg.area_noise_n_cycles)
    sigma_method = str(cfg.sigma_method)
    sigma_local_n = int(cfg.sigma_local_n_windows)
    sigma_fit = str(cfg.sigma_fit_method)

    dbg.log(f"[RALE] T_regen={T_regen} N_T_regen={N_T_regen} overlap={window_overlap}")

    dq, dqo, t_valid = _compute_delta_signal(t, q, q_o, T_regen)

    windows = _build_rale_windows(len(t_valid), N_T_regen, window_overlap)

    areas = np.empty(len(windows), dtype=float)
    t_wins = np.empty(len(windows), dtype=float)

    for k, w in enumerate(windows):
        i0, i1 = w["i_start"], w["i_end"]
        areas[k] = _compute_cycle_area(dq[i0:i1], dqo[i0:i1])
        t_wins[k] = float(t_valid[i0])
        if dbg.is_window_in_debug_range(k):
            dbg.log(f"window {k}: A={areas[k]}")

    # auto estimate noise floor
    if 0 < noise_cycles < len(areas):
        seed = areas[:noise_cycles]
        seed = seed[seed > 0]
        if len(seed) > 0:
            eps = max(eps, float(np.median(seed)) * 0.01)

    sigma = _estimate_sigma(areas, t_wins, T_regen, eps, local_n_windows=sigma_local_n, sigma_method=sigma_method, fit_method=sigma_fit)
    t_sigma = t_wins.copy()

    sigma_ewma = _apply_ewma(sigma, lam)

    G_hat = _integrate_G(sigma_ewma, t_sigma)

    return {
        "method": "RALE_Method",
        "Name": cfg.name,
        "T_regen": T_regen,
        "N_T_regen": N_T_regen,
        "t_wins": t_wins,
        "areas": areas,
        "t_sigma": t_sigma,
        "sigma": sigma,
        "sigma_ewma": sigma_ewma,
        "G_hat": G_hat,
        "data_window": [],
        "global_data": {
            "t": t,
            "q_signal": q,
            "q_o_signal": q_o,
            "t_valid": t_valid,
            "dq": dq,
            "dqo": dqo,
            "type_signal": "RALE",
            "type_method": "RALE_Method",
        },
    }
