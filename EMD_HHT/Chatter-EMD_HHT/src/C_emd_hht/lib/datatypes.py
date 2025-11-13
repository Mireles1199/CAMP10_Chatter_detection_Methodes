from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ChatterResult:
    """Contenedor de resultados intermedios del pipeline HHT/EMD."""
    imfs: Optional[np.ndarray ]
    k_selected: int
    selected_imf: np.ndarray 
    A: np.ndarray 
    f_inst: np.ndarray 
    counts: np.ndarray 
    t_counts_samples: np.ndarray 
    HHS: Optional[np.ndarray ]
    fgrid: Optional[np.ndarray ]
    meta: Dict[str, Any]

