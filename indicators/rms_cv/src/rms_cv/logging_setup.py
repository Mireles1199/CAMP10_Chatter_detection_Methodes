"""Logging helper for rms_cv.

Levels (ascending severity, fewer messages as you go up):
    DEBUG     (10) -- todo: detalle interno de la libreria
    INFO_PLUS (15) -- progreso del pipeline  [por defecto]
    INFO      (20) -- resumen de configuracion del indicador
    WARNING   (30) -- solo resultado critico (primera deteccion)

Uso tipico
----------
    from rms_cv.logging_setup import configure_logging, LOGGING_LEVELS
    configure_logging(level=LOGGING_LEVELS["info_plus"])  # o "debug", "info", "warning"
"""

from __future__ import annotations

import logging
from . import INFO_PLUS_LEVEL

# Mapa nombre -> nivel numerico para callers externos
LOGGING_LEVELS: dict = {
    "debug":     logging.DEBUG,      # 10
    "info_plus": INFO_PLUS_LEVEL,    # 15 -- logs internos de la libreria
    "info":      logging.INFO,       # 20
    "warning":   logging.WARNING,    # 30
}


def configure_logging(
    level: int | None = None,
    fmt: str = "%(levelname)-8s | %(message)s",
) -> None:
    """Configura el nivel de logging para el ejemplo o script externo.

    El nivel se aplica al root logger y se propaga automaticamente a todos
    los loggers del paquete (rms_cv.*) via la jerarquia de Python.

    Parameters
    ----------
    level:
        Nivel numerico. Si es None, usa INFO_PLUS_LEVEL (15).
        Usa LOGGING_LEVELS["info_plus"] etc. para no recordar numeros.
    fmt:
        Formato de los mensajes. Solo se usa si no habia handlers previos.
    """
    lvl = INFO_PLUS_LEVEL if level is None else level

    if not logging.root.handlers:
        logging.basicConfig(level=lvl, format=fmt)
    else:
        logging.root.setLevel(lvl)
        for h in logging.root.handlers:
            h.setLevel(lvl)

    # NOTSET -> hereda del root; permite ver cualquier logger.info_plus() del paquete
    logging.getLogger("rms_cv").setLevel(logging.NOTSET)


__all__ = ["configure_logging", "LOGGING_LEVELS"]

def _section(title: str, width: int = 54) -> str:
    bar = "=" * width
    return f"\n{bar}\n  {title}\n{bar}"
