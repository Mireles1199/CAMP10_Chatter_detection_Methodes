INFO_PLUS_LEVEL = 15

def _register_info_plus_level() -> None:
    import logging as _lg
    # expose the numeric constant on the logging module for convenience
    if not hasattr(_lg, "INFO_PLUS"):
        _lg.INFO_PLUS = INFO_PLUS_LEVEL

    # ensure a readable name is associated with the numeric level
    current_name = _lg.getLevelName(INFO_PLUS_LEVEL)
    if current_name != "INFO_PLUS":
        _lg.addLevelName(INFO_PLUS_LEVEL, "INFO_PLUS")

    # Provide `verbose` for backward compatibility if nothing else set it
    if not hasattr(_lg.Logger, "verbose"):
        def _verbose(self, msg, *args, **kwargs):
            if self.isEnabledFor(INFO_PLUS_LEVEL):
                self._log(INFO_PLUS_LEVEL, msg, args, **kwargs)
        _lg.Logger.verbose = _verbose  # type: ignore[attr-defined]

    # Provide `info_plus` as the canonical API pointing to verbose
    if not hasattr(_lg.Logger, "info_plus"):
        _lg.Logger.info_plus = _lg.Logger.verbose  # type: ignore[attr-defined]

    # Provide module-level convenience function `logging.info_plus(...)`
    if not hasattr(_lg, "info_plus"):
        def _module_info_plus(msg, *args, **kwargs):
            _lg.log(INFO_PLUS_LEVEL, msg, *args, **kwargs)
        _lg.info_plus = _module_info_plus


_register_info_plus_level()
del _register_info_plus_level

from .models.prob import GaussianPDF
from .models.maxent import MaxEntModels, fit_maxent_gaussians
from .lib.entropy import EntropyEstimator, GaussianMaxEntEstimator, EmpiricalHistogramEntropyEstimator, entropy_from_segments
from .lib.llr import LLRModel, GaussianIndicatorLLR
from .lib.sprt import SPRTConfig, SPRTResult, SequentialProbabilityRatioTest
from .lib.detector import MaxEntSPRTConfig, MaxEntSPRTDetector
from .utils.types import SignalData, IndicatorResult
from .utils.hdf5_utils import HDF5Reader
from .lib.runner import run_maxent_sprt
from .viz.maxent_sprt_plots import plots_maxent_sprt
from .logging_setup import LOGGING_LEVELS


__all__ = [
    "GaussianPDF",
    "MaxEntModels",
    "fit_maxent_gaussians",
    "EntropyEstimator",
    "GaussianMaxEntEstimator",
    "EmpiricalHistogramEntropyEstimator",
    "entropy_from_segments",
    "LLRModel",
    "GaussianIndicatorLLR",
    "SPRTConfig",
    "SPRTResult",
    "SequentialProbabilityRatioTest",
    "MaxEntSPRTConfig",
    "MaxEntSPRTDetector",
    "SignalData",
    "IndicatorResult",
    "HDF5Reader",
    "run_maxent_sprt",
    "plots_maxent_sprt",
    "INFO_PLUS_LEVEL",
    "LOGGING_LEVELS",
]

