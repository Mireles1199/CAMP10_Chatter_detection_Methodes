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
    SignalData,
    IndicatorResult,
    HDF5Reader,
    run_sst_svd,
    plots_sst_svd,
]
