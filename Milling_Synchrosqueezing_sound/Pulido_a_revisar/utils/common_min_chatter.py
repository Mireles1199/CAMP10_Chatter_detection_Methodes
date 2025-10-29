#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import logging

EPS32 = np.finfo(np.float32).eps
EPS64 = np.finfo(np.float64).eps

logging.basicConfig(format='')
WARN = lambda msg: logging.warning("WARNING: %s" % msg)
NOTE = lambda msg: logging.warning("NOTE: %s" % msg)  # else it's mostly ignored


def assert_is_one_of(x, name, supported, e=ValueError):
    if x not in supported:
        raise e("`{}` must be one of: {} (got {})".format(
            name, ', '.join(supported), x))
        
def p2up(n):
    """Calculates next power of 2, and left/right padding to center
    the original `n` locations.

    # Arguments:
        n: int
            Length of original (unpadded) signal.

    # Returns:
        n_up: int
            Next power of 2.
        n1: int
            Left  pad length.
        n2: int
            Right pad length.
    """
    up = int(2**(1 + np.round(np.log2(n))))
    n2 = int((up - n) // 2)
    n1 = int(up - n - n2)
    return up, n1, n2

