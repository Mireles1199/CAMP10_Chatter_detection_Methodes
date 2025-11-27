
from __future__ import annotations
from typing import Callable, TypeVar, ParamSpec
from functools import wraps

P = ParamSpec("P")
R = TypeVar("R")

def validate_alpha_beta(func: Callable[P, R]) -> Callable[P, R]:
    """
    Valida que alpha y beta (si se pasan como kwargs) estén en (0, 1).

    Este decorador añade validación sin modificar la lógica interna del
    test estadístico (ejemplo de preocupación transversal).
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        alpha = kwargs.get("alpha", None)
        beta = kwargs.get("beta", None)

        if alpha is not None and beta is not None:
            alpha_f = float(alpha)
            beta_f = float(beta)
            if not (0.0 < alpha_f < 1.0 and 0.0 < beta_f < 1.0):
                raise ValueError("alpha y beta deben estar en el intervalo abierto (0, 1).")

        return func(*args, **kwargs)

    return wrapper