#!/usr/bin/env python
# coding: utf-8
"""launcher.py — One window to open every DOE_utils tool, each in its own independent process.

  - Windowed tools: opened with no console; if one dies at start-up its error is shown here.
  - Command-line tools: opened in a new console that stays open. With the args box empty they run with --help
    (so a click never starts a long simulation by accident); type the real arguments to run them.
Closing the launcher (or any tool) does not close the others.

To add a tool: one line in TOOLS (kind, script relative to this folder, description).

Usage (with the entorno_CAMP10 Python):
    python launcher.py
    python launcher.py --selftest
"""
import os
import shlex
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_PYTHON = "D:/Thesis/03-Code_Storage/02-Altintlas_Nessy2m_Storage/Env/entorno_CAMP10/Scripts/python.exe"
# (kind, script, description). kind: "gui" = has its own window | "cli" = command line.
TOOLS = [
    ("gui", "DOE_simulacion/doe_planner.py", "Plan a DOE on the SLD and launch the runner"),
    ("gui", "DOE_simulacion/doe_val_planner.py", "Generate the validation-DOE YAML from the training dataset"),
    ("gui", "DOE_plots/doe_selector.py", "Select and plot cases of a doe_results.h5"),
    ("gui", "DOE_plots/doe_unified_selector.py", "Unified viewer for the 5 .h5 formats"),
    ("cli", "DOE_simulacion/doe_runner.py", "Run / extract / merge a DOE (--config NAME ...)"),
    ("cli", "DOE_simulacion/doe_merge_auto.py", "Merge DOE families automatically"),
    ("cli", "DOE_simulacion/static_deflection.py", "Add static deflection to doe_results.h5"),
    ("cli", "DOE_simulacion/doe_noise.py", "Build doe_noise_results.h5"),
    ("cli", "DOE_simulacion/reference_dataset.py", "Labels template / build / combine reference dataset"),
    ("cli", "DOE_analisis/doe_indicators.py", "Run the indicators over a DOE"),
    ("cli", "DOE_analisis/validate_indicators.py", "Score the indicators against the validation labels -> doe_validation_results.h5"),
    ("cli", "DOE_analisis/doe_model_snr.py", "Model SNR analysis"),
    ("cli", "DOE_plots/doe_plotter.py", "Plot DOE results"),
    ("cli", "DOE_plots/doe_indicator_plotter.py", "Plot indicator results"),
    ("cli", "DOE_plots/doe_noise_plotter.py", "Plot noise / noise-indicator results"),
    ("cli", "DOE_plots/doe_model_snr_plotter.py", "Plot model-SNR results"),
]


def python_exe() -> str:
    return ENV_PYTHON if os.path.isfile(ENV_PYTHON) else sys.executable


def build_cmd(kind: str, script: str, args: str) -> list:
    """Command list for one tool. cli: cmd /k keeps the console open; empty args -> --help."""
    path = os.path.join(HERE, script)
    if kind == "cli":
        return ["cmd", "/k", python_exe(), path] + (shlex.split(args, posix=False) if args.strip() else ["--help"])
    return [python_exe(), path] + shlex.split(args, posix=False)


def launch(kind: str, script: str, args: str):
    """Start the tool. Returns (Popen, stderr_log_path or None)."""
    cwd = os.path.dirname(os.path.join(HERE, script))
    cmd = build_cmd(kind, script, args)
    if kind == "cli":
        return subprocess.Popen(cmd, cwd=cwd, creationflags=subprocess.CREATE_NEW_CONSOLE), None
    log = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False, prefix="launcher_")
    p = subprocess.Popen(cmd, cwd=cwd, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
    return p, log.name


def run_gui():
    import tkinter as tk
    from tkinter import messagebox, ttk

    root = tk.Tk()
    root.title("DOE_utils launcher")
    frame = ttk.Frame(root, padding=8)
    frame.pack(fill=tk.BOTH, expand=True)

    def watch(p, log, script, tries=0):
        """After start-up, if the tool already died with an error, show its stderr."""
        rc = p.poll()
        if rc is None:
            if tries < 6:   # ~3 s: a tool still alive by then started fine
                root.after(500, watch, p, log, script, tries + 1)
            return
        if rc != 0:
            with open(log, encoding="utf-8", errors="replace") as f:
                err = f.read().strip().splitlines()[-12:]
            messagebox.showerror(os.path.basename(script), "\n".join(err) or f"exit code {rc}")

    def go(kind, script, args):
        try:
            p, log = launch(kind, script, args.get())
        except OSError as exc:
            messagebox.showerror(script, str(exc))
            return
        if log:
            root.after(500, watch, p, log, script)

    r = 0
    for title, kind in (("Windowed tools", "gui"), ("Command-line tools  (console; empty args = --help)", "cli")):
        ttk.Label(frame, text=title, font=("Segoe UI", 10, "bold")).grid(row=r, column=0, columnspan=3,
                                                                        sticky="w", pady=(8 if r else 0, 2))
        r += 1
        for k, script, desc in (t for t in TOOLS if t[0] == kind):
            args = tk.StringVar()
            ttk.Button(frame, text=os.path.basename(script), width=30,
                       command=lambda k=k, s=script, a=args: go(k, s, a)).grid(row=r, column=0, sticky="w")
            ttk.Entry(frame, textvariable=args, width=34).grid(row=r, column=1, padx=6)
            ttk.Label(frame, text=desc, foreground="#555").grid(row=r, column=2, sticky="w")
            r += 1
    root.mainloop()


def _selftest():
    for kind, script, _ in TOOLS:
        assert kind in ("gui", "cli") and os.path.isfile(os.path.join(HERE, script)), script
    c = build_cmd("cli", "DOE_simulacion/doe_runner.py", "")
    assert c[:2] == ["cmd", "/k"] and c[-1] == "--help"
    assert build_cmd("cli", "DOE_simulacion/doe_runner.py", "--config tube --dry-run")[-3:] == ["--config", "tube", "--dry-run"]
    g = build_cmd("gui", "DOE_plots/doe_selector.py", "--doe_name X")
    assert g[0] != "cmd" and g[-2:] == ["--doe_name", "X"]
    print("selftest OK")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else run_gui()
