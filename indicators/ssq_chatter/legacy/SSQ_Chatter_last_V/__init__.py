# Comentario: API pública del paquete refactorizado
from .lib.pipeline_chatter import ChatterPipeline, PipelineConfig
from .lib.tf_transformers import SSQ_STFT, STFT
from .lib.detection_strategies import ThreeSigmaWithLilliefors
from .utils.tf_windows import WindowExtractor
from .utils.decorators import timeit



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
]
