# Importaciones originales reagrupadas (sin cambios de nombre)
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Tuple, List, Sequence, Callable, TypeVar, ParamSpec
from abc import ABC, abstractmethod
from functools import wraps
import numpy as np
import math
import matplotlib.pyplot as plt
from C_emd_hht import signal_chatter_example, sinus_6_C_SNR
