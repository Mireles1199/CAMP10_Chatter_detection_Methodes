"""
Optimizacion/run_all_plots.py
==============================
Master runner — executes all 10 plot scripts sequentially.

Each script saves its PNG files to  sweep_output/plots/  automatically.
Total figures generated: 24.

Usage
-----
    python run_all_plots.py [--pkl PATH]

    # Default pkl path: sweep_output/sweep_result.pkl  (relative to this file)
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import traceback

# ── Resolve paths ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── Script registry ───────────────────────────────────────────────────────────
#  (module_name, expected_figures, source_method)
PLOT_SCRIPTS = [
    ("plot_01_df_raw",           9,  "sweep.df"),
    ("plot_02_pareto",           1,  "pareto()"),
    ("plot_03_sensitivity",      2,  "sensitivity()"),
    ("plot_04_gap_curve",        1,  "gap_curve()"),
    ("plot_05_best_table",       2,  "best_table()"),
    ("plot_06_convergence",      2,  "convergence_vs_k()"),
    ("plot_07_param_sensitivity",3,  "param_sensitivity()"),
    ("plot_08_importance",       1,  "importance_ranking()"),
    ("plot_09_tradeoff",         1,  "tradeoff_table()"),
    ("plot_10_feasible",         2,  "feasible_space_quality()"),
    ("plot_11_score_hist",       2,  "sweep.df  [score distributions]"),
]

TOTAL_FIGS = sum(n for _, n, _ in PLOT_SCRIPTS)


def _bar(label: str, n: int, source: str) -> str:
    return f"  [{n:>2} fig{'s' if n > 1 else ' '}]  {label:<35}  ← {source}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run all sweep plot scripts and generate 24 figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--pkl",
        default=os.path.join(_HERE, "sweep_output", "sweep_result.pkl"),
        help="Path to SweepResult pickle file (default: sweep_output/sweep_result.pkl)",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.pkl):
        print(f"[ERROR] Pickle file not found: {args.pkl}")
        print("        Run study_phase2.py first to generate it.")
        sys.exit(1)

    print("=" * 70)
    print(f"  run_all_plots.py — generating {TOTAL_FIGS} figures")
    print(f"  pkl : {args.pkl}")
    print("=" * 70)
    print()

    # Inject --pkl into sys.argv so each module's argparse picks it up
    # We monkey-patch sys.argv temporarily per module call.
    successes = 0
    failures  = []
    t_start   = time.perf_counter()

    for module_name, n_figs, source in PLOT_SCRIPTS:
        print(_bar(module_name, n_figs, source))
        t0 = time.perf_counter()
        try:
            old_argv   = sys.argv
            sys.argv   = [module_name + ".py", "--pkl", args.pkl]
            mod        = importlib.import_module(module_name)
            importlib.reload(mod)          # force re-run if already imported
            mod.main()
            sys.argv   = old_argv
            elapsed    = time.perf_counter() - t0
            print(f"        OK  ({elapsed:.1f}s)")
            successes += 1
        except Exception:
            sys.argv = old_argv
            elapsed  = time.perf_counter() - t0
            print(f"        FAILED ({elapsed:.1f}s)")
            traceback.print_exc()
            failures.append(module_name)
        print()

    elapsed_total = time.perf_counter() - t_start
    print("=" * 70)
    print(f"  Done in {elapsed_total:.1f}s — "
          f"{successes}/{len(PLOT_SCRIPTS)} scripts OK")
    if failures:
        print(f"  Failed: {', '.join(failures)}")
    plots_dir = os.path.join(_HERE, "sweep_output", "plots")
    if os.path.isdir(plots_dir):
        pngs = [f for f in os.listdir(plots_dir) if f.endswith(".png")]
        print(f"  {len(pngs)} PNG files in {plots_dir}")
    print("=" * 70)

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
