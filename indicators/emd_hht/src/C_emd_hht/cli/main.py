from __future__ import annotations
import argparse

from ..utils.signal_chatter import make_chatter_like_signal
from ..lib.core import detect_chatter_from_force
from ..viz.plotting import plot_imf_seleccionado

def main() -> None:
    parser = argparse.ArgumentParser(description="ssq-chatter: demo y utilidades rápidas.")
    sub = parser.add_subparsers(dest="cmd")

    demo = sub.add_parser("demo", help="Correr una demo sintética de detección.")
    demo.add_argument("--fs", type=float, default=2000.0, help="Frecuencia de muestreo (Hz).")
    demo.add_argument("--dur", type=float, default=6.0, help="Duración (s).")
    demo.add_argument("--plot", action="store_true", help="Mostrar gráficos.")
    args = parser.parse_args()

    if args.cmd == "demo":
        sig, meta = make_chatter_like_signal(fs=args.fs, T=args.dur, signal_chatter=True)
        res = detect_chatter_from_force(sig, fs=meta["fs"], hhs_enable=True)
        print("IMF seleccionado:", res.k_selected)
        print("Counts:", res.counts.shape)
        if args.plot:
            plot_imf_seleccionado(res.selected_imf, fs=meta["fs"], A=res.A, f_inst=res.f_inst, show=True)
    else:
        parser.print_help()
