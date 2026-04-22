INFO_PLUS_LEVEL: int = 15


def _register_info_plus_level() -> None:
    import logging as _lg
    if not hasattr(_lg, "INFO_PLUS"):
        _lg.INFO_PLUS = INFO_PLUS_LEVEL  # type: ignore[attr-defined]
    if _lg.getLevelName(INFO_PLUS_LEVEL) != "INFO_PLUS":
        _lg.addLevelName(INFO_PLUS_LEVEL, "INFO_PLUS")
    if not hasattr(_lg.Logger, "verbose"):
        def _verbose(self, msg, *args, **kw):
            if self.isEnabledFor(INFO_PLUS_LEVEL):
                self._log(INFO_PLUS_LEVEL, msg, args, **kw)
        _lg.Logger.verbose = _verbose  # type: ignore[attr-defined]
    if not hasattr(_lg.Logger, "info_plus"):
        _lg.Logger.info_plus = _lg.Logger.verbose  # type: ignore[attr-defined]
    if not hasattr(_lg, "info_plus"):
        def _module_info_plus(msg, *args, **kwargs):
            _lg.log(INFO_PLUS_LEVEL, msg, *args, **kwargs)
        _lg.info_plus = _module_info_plus  # type: ignore[attr-defined]


_register_info_plus_level()
del _register_info_plus_level

# Comentario: API pública del paquete refactorizado
from .lib.pipeline_chatter import ChatterPipeline, PipelineConfig
from .lib.tf_transformers import SSQ_STFT, STFT
from .lib.detection_strategies import ThreeSigmaWithLilliefors
from .lib.runner import run_sst_svd
from .utils.tf_windows import WindowExtractor
from .utils.decorators import timeit
from .utils.types import SignalData, IndicatorResult
from .utils.hdf5_utils import HDF5Reader
from .viz.plotting import prep_binary_spectro_for_pcolormesh
from .viz.sst_svd_plots import plots_sst_svd
from .logging_setup import LOGGING_LEVELS



# Compatibilidad
from .compat.core import sqq_chatter
from .compat.detection import detectar_chatter_3sigma
from .compat.generators import five_senos, signal_1

__all__ = [
    "ChatterPipeline",
    "PipelineConfig",
    "SSQ_STFT",
    "STFT",
    "ThreeSigmaWithLilliefors",
    "WindowExtractor",
    "timeit",
    "five_senos",
    "signal_1",
    "prep_binary_spectro_for_pcolormesh",
    "SignalData",
    "IndicatorResult",
    "HDF5Reader",
    "run_sst_svd",
    "plots_sst_svd",
    "INFO_PLUS_LEVEL",
    "LOGGING_LEVELS",
]
