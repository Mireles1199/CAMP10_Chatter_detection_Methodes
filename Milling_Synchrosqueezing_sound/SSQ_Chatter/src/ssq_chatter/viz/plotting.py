


import numpy as np
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt


def prep_binary_spectro_for_pcolormesh(
    S: np.ndarray,
    fs: float,
    t_vec: np.ndarray | None = None,     # opcional: tiempos (centros). Si None, usa np.arange(n_frames)/fps
    f_vec: np.ndarray | None = None,     # opcional: frecuencias (centros). Si None, usa np.linspace(0, fs/2, n_bins)
    method: str = "mad",                 # "mad" (robusto) o "quantile"
    k: float = 7.0,                      # para "mad": umbral = med + k*MAD
    q: float = 0.995,                    # para "quantile": umbral = quantil q
    smooth_kernel: int = 0,              # >0 para suavizado mediana 2D (ventana impar)
):
    """
    Prepara datos y parámetros para graficar un espectrograma binario (2 colores) con pcolormesh.
    Devuelve: dict con x, y, C, cmap, norm, cbar_label, shading.
    - Azul (#0b2a67): por debajo del umbral (no relevante)
    - Amarillo (#ffd800): por encima del umbral (relevante)
    """

    # --- 1) Magnitud en dB (robusto para gran rango dinámico)
    eps = 1e-12
    S_mag = np.abs(S).astype(float)
    S_db  = 20.0 * np.log10(S_mag + eps)  # shape: (n_freq, n_time)

    # --- 2) (Opcional) Suavizado mediana 2D para quitar puntitos
    if smooth_kernel and smooth_kernel >= 3 and smooth_kernel % 2 == 1:
        from scipy.ndimage import median_filter
        S_db = median_filter(S_db, size=(smooth_kernel, smooth_kernel))

    # --- 3) Umbral robusto
    if method.lower() == "mad":
        med = np.median(S_db)
        mad = np.median(np.abs(S_db - med)) + 1e-12
        thr = med + k * mad
    elif method.lower() == "quantile":
        thr = np.quantile(S_db, q)
    else:
        raise ValueError("method debe ser 'mad' o 'quantile'.")

    # --- 4) Colormap binario y normalización por límites
    cmap = mcolors.ListedColormap(["#0b2a67", "#ffd800"])  # azul, amarillo
    bounds = [S_db.min(), thr, S_db.max()]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # --- 5) Ejes (centros). pcolormesh con shading='auto' acepta 1D.
    n_freq, n_time = S_db.shape
    if t_vec is None:
        # Si no tienes tiempos, asume frames uniformes: 0..n_time-1
        t_vec = np.arange(n_time, dtype=float)
    if f_vec is None:
        # Si no tienes eje de freq de tu transformada, usa 0..fs/2 lineal (ajústalo a tu caso)
        f_vec = np.linspace(0.0, fs/2.0, n_freq, dtype=float)

    # --- 6) Empaquetar todo para pcolormesh
    pack = {
        "x": t_vec,             # eje tiempo (1D, longitud = n_time)
        "y": f_vec,             # eje frecuencia (1D, longitud = n_freq)
        "C": S_db,              # matriz en dB (n_freq x n_time)
        "cmap": cmap,           # 2 colores
        "norm": norm,           # mapea a dos niveles usando el umbral
        "shading": "auto",      # hace match de centros/edges
        "cbar_label": "Magnitud (dB, binario)",
        "threshold_db": thr,    # útil por si quieres anotarlo
        "method": method,
    }
    return pack
