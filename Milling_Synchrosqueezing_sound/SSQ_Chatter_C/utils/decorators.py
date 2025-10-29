from __future__ import annotations
# Comentario: decoradores utilitarios (tiempo de ejecución y validaciones)
import time
import functools
from typing import Callable, Any
import numpy as np
import inspect

def timeit(func: Callable[..., Any]) -> Callable[..., Any]:
    # Comentario: reporta el tiempo de ejecución en segundos
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        dt = time.perf_counter() - t0
        print(f"[timeit] {func.__name__}: {dt:.6f}s")
        return result
    return wrapper

def ensure_1d_array(func: Callable[..., Any]) -> Callable[..., Any]:
    # Comentario: asegura que el argumento de señal sea vector 1D (sirve para funciones y métodos)
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())  # Comentario: p. ej. ['self','x','return_TF']
    candidate_names = ("x", "signal")

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Comentario: enlaza args/kwargs a nombres reales del func (soporta métodos con self)
        bound = sig.bind_partial(*args, **kwargs)

        # Comentario: intenta primero por nombre lógico
        target_name = None
        for name in candidate_names:
            if name in bound.arguments:
                target_name = name
                break

        # Comentario: si no está por nombre, intenta hallar el primer parámetro no 'self'/'cls'
        if target_name is None:
            for name in param_names:
                if name in ("self", "cls"):
                    continue
                if name in bound.arguments:
                    target_name = name
                    break

        if target_name is None:
            raise TypeError("ensure_1d_array: no se encontró argumento de señal.")

        # Comentario: valida y normaliza a vector 1D float
        arr = np.asarray(bound.arguments[target_name], dtype=float)
        if arr.ndim != 1:
            raise ValueError("signal must be a 1D array")
        bound.arguments[target_name] = arr

        return func(*bound.args, **bound.kwargs)

    return wrapper