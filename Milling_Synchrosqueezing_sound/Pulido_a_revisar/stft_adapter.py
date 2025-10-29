#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptador de STFT para ssq_stft
--------------------------------
Envuelve tu `stft` de stft_pipeline.py para que siempre devuelva (Sx, dSx),
tal como espera `_ssq_stft.ssq_stft` del proyecto original.

Notas
-----
- No modifica tu lógica interna de padding: si en tu STFT usas padlength = N,
  se respeta tal cual.
- Soporta tanto tu versión que retorna (Sx, dSx, meta) como (Sx, dSx) a secas.
"""

from typing import Any, Tuple

# Importa tu implementación
from stft_pipeline import stft as _stft_core


def stft(x, **kwargs) -> Tuple[Any, Any]:
    """
    Enforce derivative=True y devolver exactamente (Sx, dSx).

    Parámetros
    ----------
    x : array-like
        Señal 1D.
    **kwargs : dict
        Parámetros que tu `stft` ya entiende (window, n_fft, win_len, hop_len,
        fs, t, padtype, modulated, dtype, ...).

    Devuelve
    --------
    (Sx, dSx)
    """
    # Forzar que haya derivada (ssq_stft la necesita)
    kwargs["derivative"] = True

    out = _stft_core(x, **kwargs)

    # Compatibilidad con (Sx, dSx, meta) y (Sx, dSx)
    if isinstance(out, (tuple, list)):
        if len(out) >= 2:
            Sx, dSx = out[0], out[1]
            return Sx, dSx
        else:
            raise RuntimeError("La STFT no devolvió suficientes valores.")
    else:
        raise RuntimeError("La STFT no devolvió una tupla/lista esperada.")
