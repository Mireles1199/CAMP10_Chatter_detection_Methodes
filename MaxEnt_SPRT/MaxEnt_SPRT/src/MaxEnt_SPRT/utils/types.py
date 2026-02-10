from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


# =========================
# 1) Datos de entrada comunes
# =========================
@dataclass
class SignalData:
    """
    Contenedor estándar para las señales de entrada a los indicadores.

    Attributes
    ----------
    t : np.ndarray
        Vector de tiempo (1D) en segundos.
    v : np.ndarray
        Señal principal de análisis (por ejemplo, velocidad de punta de herramienta).
    x : np.ndarray
        Señal secundaria de análisis (por ejemplo, desplazamiento de herramienta).
    force : np.ndarray
        Señal de fuerza (por ejemplo, fuerza de corte).
    t_original : np.ndarray
        Vector de tiempo original (antes de cualquier recorte o procesamiento).
    v_original : np.ndarray
        Señal original (antes de cualquier recorte o procesamiento).
    x_original : np.ndarray
        Señal secundaria original (antes de cualquier recorte o procesamiento).
    t_analysis : np.ndarray
        Vector de tiempo específico para el análisis del indicador (puede ser igual a t_cut o diferente).
    signal_analysis : np.ndarray
        Señal específica para el análisis del indicador (puede ser v_cut, x_cut, o alguna combinación/procesamiento).

    force_original : np.ndarray
        Señal de fuerza original (antes de cualquier recorte o procesamiento).
    fs : float
        Frecuencia de muestreo en Hz.
    path : str
        Ruta del archivo de donde se cargaron los datos.
    meta : Dict[str, Any]
        Metadatos adicionales (ap(t), rpm, id de simulación, etc.).
    """
    t_cut: np.ndarray
    v_cut: np.ndarray
    x_cut: np.ndarray
    force_cut: np.ndarray
    t_original: np.ndarray
    x_original: np.ndarray
    v_original: np.ndarray
    t_analysis: np.ndarray
    signal_analysis: np.ndarray
    force_original: np.ndarray
    path: str
    fs: float
    meta: Dict[str, Any] = field(default_factory=dict)
    
    


# =========================
# 2) Resultado estándar de un indicador
# =========================
@dataclass
class IndicatorResult:
    """
    Resultado estándar de un indicador de chatter.

    Attributes
    ----------
    name : str
        Nombre del indicador (ej. 'EMD_IMF1', 'MaxEnt_SPRT', 'CV_RMS').
    t : np.ndarray
        Eje temporal asociado al índice I(t). Puede ser tiempo o ciclos.
    I_t : np.ndarray
        Valor del índice.
    t_d : Optional[float]
        Tiempo de detección.
    meta : Dict[str, Any]
        Metadatos adicionales específicos del indicador.
    """
    name: str
    t: np.ndarray
    I_t: np.ndarray
    t_d: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


# =========================
# 3) Metadatos del escenario (opcional)
# =========================
@dataclass
class ScenarioMetadata:
    """
    Información del escenario de simulación / experimento.

    Attributes
    ----------
    scenario_id : str
    ap_ramp : Optional tuple
    rpm : Optional float
    snr_db : Optional float
    extra : Dict[str, Any]
    """
    scenario_id: str
    ap_ramp: Optional[tuple[float, float]] = None
    rpm: Optional[float] = None
    snr_db: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)
