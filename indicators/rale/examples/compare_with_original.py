"""Example: run RALE from the new `rale` package and compare with original Areas_Indicator_V1 implementation.

Usage:
    python compare_with_original.py <work_space_path>

The script tries to import the original `Areas_Indicator_V1.py` from CAMP8. If that import fails,
only the new packaged implementation will be run.
"""
import sys
import os
import runpy
import numpy as np
import h5py

# Path to original code (adjust if needed)
ORIG_DIR = r"d:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\Chatter-Criteria\CAMP8-Ventanna_Glisante"
# Optional helper modules used by original (plotter)
PLOTTER_DIR = os.path.join(os.path.dirname(ORIG_DIR), 'Modulos_python', 'plotter')

try:
    sys.path.insert(0, ORIG_DIR)
    if os.path.isdir(PLOTTER_DIR):
        sys.path.insert(0, PLOTTER_DIR)
    import Areas_Indicator_V1 as AIV1
    HAVE_ORIG = True
except Exception as e:
    print("Could not import original Areas_Indicator_V1.py:", e)
    HAVE_ORIG = False

from rale import run_rale, RALEConfig


def load_signal_from_out_hdf5(work_space: str, ruta: str, data_range=(0.0, -1)):
    out_hdf5 = os.path.join(work_space, 'out.hdf5')
    if not os.path.exists(out_hdf5):
        raise FileNotFoundError(f"out.hdf5 not found in {work_space}")
    f = h5py.File(out_hdf5, 'r')
    dset = f[ruta + '/data']
    data = np.empty(dset.shape, dtype=np.float64)
    dset.read_direct(data)
    f.close()
    data = np.asarray(data)

    # apply time filter (data_range similar to Config.TimeFilter)
    start, end = data_range
    mask = data[:, 0] >= start
    if end is not None and end != -1:
        mask &= (data[:, 0] <= end)
    return data[mask]


def run_compare(work_space: str):
    # parameters used in Areas_Indicator_V1 example
    dt = None
    # load model file to obtain spin_rate and modal freq
    ruta_mdl = os.path.join(work_space, 'in', 'mdl_data.py')
    model_vars = {}
    if os.path.exists(ruta_mdl):
        model_vars = runpy.run_path(ruta_mdl)

    spin_rate = float(model_vars.get('spin_rate', 6000.0))
    f_modal = float(model_vars.get('f2', model_vars.get('f_modal', 150.0)))

    # load signals
    tool_dyn = load_signal_from_out_hdf5(work_space, 'n2m_data/out_n2m/tool_dyn', data_range=(0.05, 16))
    tool_dyn_o = load_signal_from_out_hdf5(work_space, 'n2m_data/out_n2m/tool_dyn_o', data_range=(0.05, 16))

    t = tool_dyn[:, 0]
    q = tool_dyn[:, 1]
    qo = tool_dyn_o[:, 1]

    d_t = float(t[1] - t[0])

    cfg = RALEConfig(
        name='cono_RALE',
        spin_rate=spin_rate,
        n_teeth=1,
        T_modal=1.0 / f_modal,
        d_t=d_t,
        window_overlap=0.25,
        lambda_ewma=0.3,
        area_noise_eps=1e-18,
        area_noise_n_cycles=0,
        sigma_method='frozen_time',
        sigma_local_n_windows=10,
        sigma_fit_method='ols',
    )

    print('Running packaged RALE...')
    res_pkg = run_rale(t, q, qo, cfg)
    print('Packaged RALE done. G_final =', float(res_pkg['G_hat'][-1]) if len(res_pkg['G_hat'])>0 else np.nan)

    if HAVE_ORIG:
        print('Running original RALE (unbound methods) for comparison...')
        # use the original class methods as functions (unbound)
        T_regen = 60.0 / (cfg.spin_rate * cfg.n_teeth)
        dq, dqo, t_valid = AIV1.RALE_Method._compute_delta_signal(None, t, q, qo, T_regen)
        N_T_regen = max(2, int(round(T_regen / cfg.d_t)))
        windows = AIV1.RALE_Method._build_rale_windows(None, len(t_valid), N_T_regen, cfg.window_overlap)

        areas = []
        t_wins = []
        for w in windows:
            i0, i1 = w['i_start'], w['i_end']
            areas.append(AIV1.RALE_Method._compute_cycle_area(None, dq[i0:i1], dqo[i0:i1]))
            t_wins.append(float(t_valid[i0]))
        areas = np.array(areas)
        t_wins = np.array(t_wins)
        sigma = AIV1.RALE_Method._estimate_sigma(None, areas, t_wins, T_regen, cfg.area_noise_eps, local_n_windows=cfg.sigma_local_n_windows, sigma_method=cfg.sigma_method, fit_method=cfg.sigma_fit_method)
        sigma_ewma = AIV1.RALE_Method._apply_ewma(None, sigma, cfg.lambda_ewma)
        G_hat = AIV1.RALE_Method._integrate_G(None, sigma_ewma, t_wins)
        print('Original (unbound) RALE done. G_final =', float(G_hat[-1]) if len(G_hat)>0 else np.nan)

        # compare arrays
        def compare_arrays(a, b, name):
            a = np.asarray(a)
            b = np.asarray(b)
            n = min(len(a), len(b))
            dif = np.nan
            if n > 0:
                dif = np.nanmax(np.abs(a[:n] - b[:n]))
            print(f"Compare {name}: length(pkg)={len(a)} length(orig)={len(b)} max_abs_diff(first {n})={dif}")

        compare_arrays(res_pkg['areas'], areas, 'areas')
        compare_arrays(res_pkg['sigma'], sigma, 'sigma')
        compare_arrays(res_pkg['sigma_ewma'], sigma_ewma, 'sigma_ewma')
        compare_arrays(res_pkg['G_hat'], G_hat, 'G_hat')

    else:
        print('Original implementation not available for comparison (import failed).')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        work_space = sys.argv[1]
    else:
        work_space = input('work_space path: ').strip()
    run_compare(work_space)
