"""Logging helper for green_integral.

Levels (ascending severity):
    DEBUG     (10) -- internal library detail
    INFO_PLUS (15) -- pipeline progress  [default]
    INFO      (20) -- indicator configuration summary
    WARNING   (30) -- critical results only

Typical usage
-------------
    from green_integral.logging_setup import configure_logging, LOGGING_LEVELS
    configure_logging(level=LOGGING_LEVELS["info_plus"])
"""

from __future__ import annotations

import logging
from . import INFO_PLUS_LEVEL

LOGGING_LEVELS: dict = {
    "debug":     logging.DEBUG,      # 10
    "info_plus": INFO_PLUS_LEVEL,    # 15
    "info":      logging.INFO,       # 20
    "warning":   logging.WARNING,    # 30
}


def configure_logging(
    level: int | None = None,
    fmt: str = "%(levelname)-8s | %(message)s",
) -> None:
    """Configure logging level for the green_integral package.

    Parameters
    ----------
    level : int, optional
        Numeric logging level.  If ``None``, defaults to ``INFO_PLUS_LEVEL`` (15).
        Use :data:`LOGGING_LEVELS` keys to avoid remembering numbers.
    fmt   : str
        Log-record format string.  Applied only when no handlers exist yet.
    """
    lvl = INFO_PLUS_LEVEL if level is None else level

    if not logging.root.handlers:
        logging.basicConfig(level=lvl, format=fmt)
    else:
        logging.root.setLevel(lvl)
        for handler in logging.root.handlers:
            handler.setLevel(lvl)

def _section(title: str, width: int = 54) -> str:
    bar = "=" * width
    return f"\n{bar}\n  {title}\n{bar}"

