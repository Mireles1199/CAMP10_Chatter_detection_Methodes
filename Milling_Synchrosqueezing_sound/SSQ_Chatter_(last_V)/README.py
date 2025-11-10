# Comentario: guía de uso rápida (en español)
# Nota: nombres de variables y strings en inglés, comentarios en español

"""
Instalación requerida:
    pip install numpy scipy ssqueezepy statsmodels

Ejemplo mínimo:

from ssq_chatter.lib.pipeline_chatter import ChatterPipeline, PipelineConfig
from ssq_chatter.lib.tf_transformers import SSQSTFTTransformer, STFTTransformer
from ssq_chatter.lib.detection_strategies import ThreeSigmaWithLilliefors

cfg = PipelineConfig(fs=48000, win_length_ms=40.0, hop_ms=10.0, n_fft=1024, Ai_length=4)
tf = SSQSTFTTransformer(win_length=int(cfg.win_length_ms*1e-3*cfg.fs),
                        hop_length=int(cfg.hop_ms*1e-3*cfg.fs),
                        n_fft=cfg.n_fft,
                        sigma=6.0)
det = ThreeSigmaWithLilliefors(frac_stable=0.25, alpha=0.05, z=3.0)

pipe = ChatterPipeline(transformer=tf, detector=det, config=cfg)
Tsx, Sx, fs, t, A_i, t_i, D, d1, res = pipe.run(x)

print(res["mask"])

"""
