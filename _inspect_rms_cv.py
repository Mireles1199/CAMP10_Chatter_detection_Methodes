import numpy as np, sys, logging
sys.path.insert(0, r'd:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\Chatter-Criteria\CAMP10_Chatter_detection_Methodes\indicators\rms_cv\src')
from rms_cv import HDF5Reader, SignalData, run_rms_cv
logging.disable(logging.CRITICAL)

d   = HDF5Reader(r'D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\2DOF_Cono\1DOF_150Hz\out.hdf5')
t   = d.get_element('tool_dyn/data')[:,0]
v   = d.get_element('tool_dyn_o/data')[:,1]
fs  = 1.0/(t[1]-t[0])
mask = (t>=0.05)&(t<=15)
t_c, v_c = t[mask], v[mask]

sig = SignalData(t_analysis=t_c, signal_analysis=v_c, fs=fs,
                path=r'D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\2DOF_Cono\1DOF_150Hz\out.hdf5')
cfg = {
    'id':'RMS_CV','func':'Default','param_mode':'by_revolution',
    'params_physical':{
        'T_rev':60/12000,'N_rev_window':20,'step_rev':1,
        'n_max_mode':'frames','n_max_rev':35,
        'cv_threshold':None,'rms_threshold':None,'n_min_cv':2,
        'warmup_ignore_alerts':False,'use_unbiased_std':True,'eps':1e-12,
        'detrend':False,'pad_mode':'none',
        'stable_time':(0.0,5.34),'frac_stable':0.30,'z':3.0,'alpha':0.05,'fallback_mad':True,
    }
}
r   = run_rms_cv(sig, cfg)
m   = r.meta
t_cv  = np.asarray(m['cv_time'])
mu    = np.asarray(m['mu'])
sigma = np.asarray(m['sigma'])
cv    = np.asarray(m['cv_values'])

T_GT = 5.3658
i_gt = int(np.searchsorted(t_cv, T_GT))

thr  = m.get('cv_threshold_used', float('nan'))
print(f"n_frames={len(t_cv)}  t[0]={t_cv[0]:.3f}  t[-1]={t_cv[-1]:.3f}")
print(f"cv_threshold_used = {thr:.6g}")
print()
hdr = f"{'t':>8} {'mu':>12} {'sigma':>12} {'CV':>10}  note"
sep = "-"*58
print(hdr); print(sep)

def row(i, note=""):
    print(f"{t_cv[i]:8.3f} {mu[i]:12.3e} {sigma[i]:12.3e} {cv[i]:10.4f}  {note}")

# --- estable (muestra cada 100 frames) ---
print("--- STABLE ---")
for i in range(0, i_gt, max(1, i_gt//8)):
    row(i, "stable")
print("...")
# --- onset ---
print("--- ONSET t_gt ---")
for i in range(max(0, i_gt-3), min(len(t_cv), i_gt+5)):
    tag = "<-- t_gt" if abs(t_cv[i]-T_GT) < 0.015 else ""
    row(i, tag)
print("...")
# --- chatter desarrollado ---
print("--- CHATTER (ultimos 30 frames) ---")
for i in range(max(0,len(t_cv)-30), len(t_cv), max(1,30//8)):
    row(i, "chatter")

print()
print("RESUMEN ESTADISTICO")
sep2 = "-"*40
print(sep2)
print(f"  region stable   CV:  mean={cv[:i_gt].mean():.4f}  std={cv[:i_gt].std():.4f}  max={cv[:i_gt].max():.4f}")
print(f"  region chatter  CV:  mean={cv[i_gt:].mean():.4f}  std={cv[i_gt:].std():.4f}  max={cv[i_gt:].max():.4f}")
print(f"  mu   ratio  chatter/stable = {mu[i_gt:].mean()/mu[:i_gt].mean():.3f}")
print(f"  sig  ratio  chatter/stable = {sigma[i_gt:].mean()/sigma[:i_gt].mean():.3f}")
print(f"  cv   ratio  chatter/stable = {cv[i_gt:].mean()/cv[:i_gt].mean():.3f}")
