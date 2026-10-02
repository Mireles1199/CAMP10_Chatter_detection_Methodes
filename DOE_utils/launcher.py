#!/usr/bin/env python
# coding: utf-8
"""launcher.py — DOE experiments app: configure and follow every experiment, stage by stage.

Tabs
  Experiments  list of experiments (experiments/*.yaml), the stage diagram (colour = status), the goal and its
               next step, the panel of the selected stage (run it in a console, copy the command, log, output,
               edit its configuration) and the buttons to create experiments.
  Compare      metrics of two validation experiments side by side.
  Tools        every DOE_utils tool on its own (what launcher.py was before), each in its own process.

Stages run in their own console through `experiment.py run` (the wrapper that records start / end / exit code and
the log); closing the app never stops them. The logic lives in experiment.py; this file is only the window.
See PLAN_app_experimentos.md.

Usage (entorno_CAMP10 Python):
    python launcher.py
    python launcher.py --selftest            # no window
    python launcher.py --screenshot OUT.png [EXP [STAGE]]   # opens, draws, saves a capture, closes
"""
import json
import os
import shlex
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import experiment as ex  # noqa: E402

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
    ("cli", "DOE_analisis/validate_indicators.py", "Score the indicators against the validation labels"),
    ("cli", "DOE_analisis/doe_model_snr.py", "Model SNR analysis"),
    ("cli", "DOE_plots/doe_plotter.py", "Plot DOE results"),
    ("cli", "DOE_plots/doe_indicator_plotter.py", "Plot indicator results"),
    ("cli", "DOE_plots/doe_noise_plotter.py", "Plot noise / noise-indicator results"),
    ("cli", "DOE_plots/doe_model_snr_plotter.py", "Plot model-SNR results"),
    ("cli", "experiment.py", "Experiments from the console (status / check / run / import / selftest)"),
]

# ------------------------------------------------------------------------------ look
STATE_FILL = {"done": "#c8e6c9", "stale": "#ffe0b2", "running": "#bbdefb", "failed": "#ffcdd2",
              "pending": "#eeeeee", "blocked": "#fafafa"}
STATE_TEXT = {"done": "done", "stale": "stale", "running": "running", "failed": "failed", "pending": "pending",
              "blocked": "blocked"}
# diagram grid (column, row); merge only when the experiment has several runs
POS = {"simulate": (0, 0), "extract": (1, 0), "merge": (2, 0), "label_template": (3, 0), "label_build": (4, 0),
       "indicators": (4, 1), "validate": (5, 1), "model_snr": (0, 2), "static_deflection": (1, 2), "noise": (2, 2),
       "noise_indicators": (3, 2)}
BOX_W, BOX_H, STEP_X, STEP_Y, MARGIN = 132, 64, 160, 100, 18
LOG_TAIL = 15


TUTORIAL_FILE = os.path.join(HERE, "TUTORIAL.md")


def read_tutorial() -> str:
    try:
        with open(TUTORIAL_FILE, encoding="utf-8") as f:
            return f.read()
    except OSError as exc:
        return f"# Tutorial{chr(10)}{chr(10)}The file {TUTORIAL_FILE} could not be read: {exc}{chr(10)}"


def render_markdown(widget, text: str) -> None:
    """Minimal markdown into a Tk Text (read-only afterwards): # / ## headings, **bold**, `code`, ``` blocks,
    pipe tables (monospace, separator row dropped) and - / 1. lists. Enough for TUTORIAL.md; not a full parser."""
    import re
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    in_code = False

    def inline(line, base=()):
        pos = 0
        for m in re.finditer(r"\*\*(.+?)\*\*|`([^`]+)`", line):
            widget.insert("end", line[pos:m.start()], base)
            if m.group(1) is not None:
                widget.insert("end", m.group(1), base + ("bold",))
            else:
                widget.insert("end", m.group(2), base + ("code",))
            pos = m.end()
        widget.insert("end", line[pos:], base)

    for raw in text.splitlines():
        if raw.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            widget.insert("end", raw + chr(10), ("code",))
        elif raw.startswith("# "):
            widget.insert("end", raw[2:] + chr(10), ("h1",))
        elif raw.startswith("## "):
            widget.insert("end", raw[3:] + chr(10), ("h2",))
        elif raw.lstrip().startswith("|"):
            if re.fullmatch(r"[|\s:-]+", raw.strip()):
                continue
            cells = [c.strip().replace("**", "").replace("`", "") for c in raw.strip().strip("|").split("|")]
            widget.insert("end", "  " + "  |  ".join(cells) + chr(10), ("table",))
        elif re.match(r"\s*([-*]|\d+\.)\s", raw):
            inline(raw.strip() + chr(10), ("bullet",))
        elif raw.strip():
            inline(raw + chr(10))
        else:
            widget.insert("end", chr(10))
    widget.configure(state="disabled")


HELP = """How the app works

An EXPERIMENT (left list) is one DOE taken from simulation to results: its runs (configs/*.yaml or an
imported folder), how its cases are labelled, which indicator variants run and, for a validation, the
training it is checked against.

The DIAGRAM shows its stages. Colour = status:
  green done · orange stale (its configuration or an input changed after it ran) · blue running ·
  red failed · grey pending (ready) · dashed blocked (an earlier stage is missing).
Hover a box to see why it is in that state; click it to open its panel below.

GOAL (top): what you want to obtain. The app marks the stages it needs (dark outline) and the NEXT STEP
(blue outline). "Run next step" starts it.

STAGE PANEL: what the stage does, what its output contains, what to check, its last run (time, duration,
exit code) and, if it cannot run, why and what to do.
Buttons: Run in console (a console opens; closing the app does not stop it) · Copy command · Open log ·
Open output in viewer · Edit config · Open folder · Go to blocking experiment.

CREATE: New DOE (from a template config: n, kappa, name) · Import folder (already simulated) ·
New validation (opens the validation planner) · Derive (another n or other variants) · Duplicate · Delete
(only the experiment file; data are never deleted).

COMPARE tab: metrics of two validations side by side. TOOLS tab: every script on its own.
TUTORIAL tab: step-by-step guide (the same text as DOE_utils/TUTORIAL.md).
Console: python experiment.py status | check EXP | run EXP STAGE | resolve | import.
"""


# ============================================================================== process helpers
def python_exe() -> str:
    return ENV_PYTHON if os.path.isfile(ENV_PYTHON) else sys.executable


def build_cmd(kind: str, script: str, args: str) -> list:
    """Command list for one tool. cli: cmd /k keeps the console open; empty args -> --help."""
    path = os.path.join(HERE, script)
    if kind == "cli":
        return ["cmd", "/k", python_exe(), path] + (shlex.split(args, posix=False) if args.strip() else ["--help"])
    return [python_exe(), path] + shlex.split(args, posix=False)


def launch(kind: str, script: str, args: str):
    """Start a tool. Returns (Popen, stderr_log_path or None)."""
    cwd = os.path.dirname(os.path.join(HERE, script))
    cmd = build_cmd(kind, script, args)
    if kind == "cli":
        return subprocess.Popen(cmd, cwd=cwd, creationflags=subprocess.CREATE_NEW_CONSOLE), None
    log = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False, prefix="launcher_")
    p = subprocess.Popen(cmd, cwd=cwd, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
    return p, log.name


_PY = None


def stage_python():
    """(python, warning): the Python that runs the stages, chosen by doe_planner.pick_python() (cached)."""
    global _PY
    if _PY is None:
        try:
            if ex.SIM not in sys.path:
                sys.path.insert(0, ex.SIM)
            import doe_planner
            _PY = doe_planner.pick_python()
        except Exception as exc:   # planner not importable: fall back to entorno_CAMP10
            _PY = (python_exe(), f"pick_python() unavailable ({exc}); using {python_exe()}")
    return _PY


def open_console(argv: list, cwd: str = HERE):
    return subprocess.Popen(["cmd", "/k", *argv], cwd=cwd, creationflags=subprocess.CREATE_NEW_CONSOLE)


def quote(argv: list) -> str:
    return subprocess.list2cmdline([str(a) for a in argv])


def parse_kappas(text: str) -> list:
    """'0.5, 0.6, 1.1' or 'start:stop:step' (stop included) -> floats."""
    text = text.strip()
    if ":" in text:
        a, b, s = (float(x) for x in text.split(":"))
        n = int(round((b - a) / s)) + 1
        return [round(a + i * s, 10) for i in range(n)]
    return [float(x) for x in text.replace(";", ",").replace(" ", ",").split(",") if x]


def diagram_layout(keys: list, has_training: bool, step_x: float = STEP_X) -> dict:
    """{key: (x, y)} top-left corner of each box; '@training' is the external training-dataset box."""
    shift = 0 if "merge" in keys else 1
    out = {}
    for k in keys:
        c, r = POS[k]
        if c >= 3 and r <= 1:
            c -= shift
        out[k] = (MARGIN + c * step_x, MARGIN + r * STEP_Y)
    if has_training and "indicators" in out:
        x, y = out["indicators"]
        out["@training"] = (x - step_x, y)
    return out


def edge_points(a: tuple, b: tuple, box_w: float, slot: float = 0.0) -> list:
    """Orthogonal route from box a to box b (top-left corners): straight when they share a row or a column,
    otherwise down from a, along the free corridor just above b's row, then down into b. slot shifts the
    entry point so several arrows into the same box do not overlap."""
    (x1, y1), (x2, y2) = a, b
    if y1 == y2:
        return [x1 + box_w, y1 + BOX_H / 2, x2, y2 + BOX_H / 2] if x2 > x1 else [x1, y1 + BOX_H / 2, x2 + box_w, y2 + BOX_H / 2]
    if x1 == x2:
        return [x1 + box_w / 2 + slot, y1 + BOX_H, x2 + box_w / 2 + slot, y2]
    corridor = y2 - (STEP_Y - BOX_H) / 2
    sx, tx = x1 + box_w / 2, x2 + box_w / 2 + slot
    return [sx, y1 + BOX_H, sx, corridor, tx, corridor, tx, y2]


# ============================================================================== the app
class App:
    REFRESH_MS = 3000

    def __init__(self, root, auto_refresh=True):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk, self.root = tk, ttk, root
        root.title("DOE experiments")
        root.minsize(1180, 720)
        self.sel_exp, self.sel_stage = None, None
        self.goal = tk.StringVar()
        self._goal_for = None
        self._last_draw = None
        nb = ttk.Notebook(root)
        nb.pack(fill=tk.BOTH, expand=True)
        self.tab_exp, self.tab_cmp, self.tab_tools = ttk.Frame(nb), ttk.Frame(nb), ttk.Frame(nb)
        self.tab_tut = ttk.Frame(nb)
        nb.add(self.tab_exp, text="Experiments")
        nb.add(self.tab_cmp, text="Compare")
        nb.add(self.tab_tools, text="Tools")
        nb.add(self.tab_tut, text="Tutorial")
        self.notebook = nb
        self._build_experiments()
        self._build_compare()
        self._build_tools()
        self._build_tutorial()
        self.refresh(force=True)
        if auto_refresh:
            root.after(self.REFRESH_MS, self._tick)

    # ------------------------------------------------------------------ layout
    def _build_experiments(self):
        tk, ttk = self.tk, self.ttk
        pw = tk.PanedWindow(self.tab_exp, orient=tk.HORIZONTAL, sashwidth=5)
        pw.pack(fill=tk.BOTH, expand=True)
        left, right = ttk.Frame(pw, padding=6), ttk.Frame(pw, padding=6)
        pw.add(left, minsize=300, width=420)
        pw.add(right, minsize=760)

        ttk.Label(left, text="Experiments", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.tree = ttk.Treeview(left, columns=("kind", "n", "cases", "prog"), show="tree headings", height=14,
                                 selectmode="browse")
        for c, h, w in (("#0", "name", 200), ("kind", "kind", 64), ("n", "n [rpm]", 64), ("cases", "cases", 44),
                        ("prog", "done", 44)):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, minwidth=30, stretch=c == "#0")
        self.tree.pack(fill=tk.BOTH, expand=True, pady=4)
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._on_select_exp())
        for tag, col in (("failed", "#c62828"), ("running", "#1565c0"), ("reached", "#2e7d32")):
            self.tree.tag_configure(tag, foreground=col)
        self.desc = tk.StringVar()
        ttk.Label(left, text="green = goal reached · blue = something running · red = a stage failed",
                  foreground="#666", font=("Segoe UI", 8)).pack(anchor="w")
        bf = ttk.Frame(left)
        bf.pack(fill=tk.X, pady=(6, 0))
        for i, (txt, fn) in enumerate((("New DOE…", self.new_doe), ("Import folder…", self.import_folder),
                                       ("New validation…", self.new_validation), ("Derive…", self.derive),
                                       ("Duplicate…", self.duplicate), ("Delete", self.delete),
                                       ("Edit YAML", self.edit_yaml), ("Refresh", lambda: self.refresh(True)))):
            ttk.Button(bf, text=txt, command=fn).grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)
        bf.columnconfigure(0, weight=1)
        bf.columnconfigure(1, weight=1)

        card = ttk.Frame(right)
        card.pack(fill=tk.X, pady=(0, 4))
        self.card_title, self.card_line, self.card_link = tk.StringVar(), tk.StringVar(), tk.StringVar()
        ttk.Label(card, textvariable=self.card_title, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(card, textvariable=self.card_line, foreground="#333").pack(anchor="w")
        ttk.Label(card, textvariable=self.desc, foreground="#555", wraplength=900, justify="left").pack(anchor="w")
        ttk.Label(card, textvariable=self.card_link, foreground="#1565c0").pack(anchor="w")
        top = ttk.Frame(right)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Goal:").pack(side=tk.LEFT)
        cb = ttk.Combobox(top, textvariable=self.goal, values=list(ex.GOALS), state="readonly", width=24)
        cb.pack(side=tk.LEFT, padx=4)
        cb.bind("<<ComboboxSelected>>", lambda _e: self.refresh(True))
        self.next_lbl = tk.StringVar()
        ttk.Label(top, textvariable=self.next_lbl, font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=10)
        self.btn_next = ttk.Button(top, text="Show", command=self.go_next)
        self.btn_next.pack(side=tk.LEFT)
        self.btn_run_next = ttk.Button(top, text="▶ Run next step", command=self.run_next)
        self.btn_run_next.pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Tutorial", command=self.show_tutorial).pack(side=tk.RIGHT)
        ttk.Button(top, text="?  Help", command=self.show_help).pack(side=tk.RIGHT, padx=4)

        self.canvas = tk.Canvas(right, height=3 * STEP_Y + 2 * MARGIN - (STEP_Y - BOX_H), bg="white",
                                highlightthickness=1, highlightbackground="#ccc")
        self.canvas.pack(fill=tk.X, pady=6)
        self.canvas.bind("<Configure>", lambda _e: self.redraw(True) if self.sel_exp else None)
        legend = ttk.Frame(right)
        legend.pack(fill=tk.X)
        for st, fill in STATE_FILL.items():
            tk.Label(legend, text=f" {st} ", bg=fill, relief="solid", bd=1, font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=2)
        ttk.Label(legend, text="   dark outline = needed for the goal · blue = next step · purple = selected",
                  foreground="#555").pack(side=tk.LEFT)
        self.hover = tk.StringVar(value="Hover a stage to see why it is in that state; click it for details.")
        ttk.Label(right, textvariable=self.hover, foreground="#37474f", font=("Segoe UI", 9, "italic")).pack(
            anchor="w", pady=(2, 0))

        panel = ttk.LabelFrame(right, text="Stage", padding=6)
        panel.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.panel = panel
        box = ttk.Frame(panel)
        box.pack(fill=tk.BOTH, expand=True)
        self.info = tk.Text(box, height=14, wrap="char", font=("Consolas", 9), state="disabled")
        sb = ttk.Scrollbar(box, orient=tk.VERTICAL, command=self.info.yview)
        self.info.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        for tag, col in (("ok", "#2e7d32"), ("bad", "#c62828"), ("warn", "#b26a00"), ("hint", "#546e7a"),
                         ("run", "#1565c0")):
            self.info.tag_configure(tag, foreground=col)
        self.info.tag_configure("head", font=("Consolas", 9, "bold"), spacing1=6)
        self.info.tag_configure("title", font=("Segoe UI", 11, "bold"))
        self.info.tag_configure("what", font=("Segoe UI", 9))
        bar = ttk.Frame(panel)
        bar.pack(fill=tk.X, pady=(6, 0))
        self.stage_btns = {}
        for key, txt, fn in (("run", "▶ Run in console", self.run_stage), ("copy", "Copy command", self.copy_cmd),
                             ("log", "Log", self.open_log), ("view", "Viewer", self.open_output),
                             ("edit", "Edit config", self.edit_stage), ("labels", "Labels YAML", self.open_labels),
                             ("folder", "Folder", self.open_folder),
                             ("goto", "Go to blocker", self.goto_blocker)):
            b = ttk.Button(bar, text=txt, command=fn)
            b.pack(side=tk.LEFT, padx=2)
            self.stage_btns[key] = b

    def _build_compare(self):
        tk, ttk = self.tk, self.ttk
        f = ttk.Frame(self.tab_cmp, padding=8)
        f.pack(fill=tk.BOTH, expand=True)
        row = ttk.Frame(f)
        row.pack(fill=tk.X)
        self.cmp_a, self.cmp_b = tk.StringVar(), tk.StringVar()
        ttk.Label(row, text="A:").pack(side=tk.LEFT)
        self.cmp_cb_a = ttk.Combobox(row, textvariable=self.cmp_a, state="readonly", width=34)
        self.cmp_cb_a.pack(side=tk.LEFT, padx=4)
        ttk.Label(row, text="B:").pack(side=tk.LEFT)
        self.cmp_cb_b = ttk.Combobox(row, textvariable=self.cmp_b, state="readonly", width=34)
        self.cmp_cb_b.pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Compare", command=self.compare).pack(side=tk.LEFT, padx=6)
        self.cmp_note = tk.StringVar(value="Validation experiments with doe_validation_results.h5")
        ttk.Label(f, textvariable=self.cmp_note, foreground="#555").pack(anchor="w", pady=4)
        cols = ("variant",) + tuple(f"{s}:{m}" for m in ex.METRIC_COLUMNS for s in ("A", "B"))
        self.cmp_tree = ttk.Treeview(f, columns=cols, show="headings")
        for c in cols:
            self.cmp_tree.heading(c, text=c.replace("balanced_accuracy", "bal.acc").replace("mean_", "")
                                  .replace("median_delay_onset_s", "delay_onset").replace("_stable", ""))
            self.cmp_tree.column(c, width=230 if c == "variant" else 62, anchor="w" if c == "variant" else "center")
        xs = ttk.Scrollbar(f, orient=tk.HORIZONTAL, command=self.cmp_tree.xview)
        self.cmp_tree.configure(xscrollcommand=xs.set)
        self.cmp_tree.pack(fill=tk.BOTH, expand=True)
        xs.pack(fill=tk.X)

    def _build_tutorial(self):
        tk, ttk = self.tk, self.ttk
        f = ttk.Frame(self.tab_tut, padding=8)
        f.pack(fill=tk.BOTH, expand=True)
        bar = ttk.Frame(f)
        bar.pack(fill=tk.X)
        ttk.Label(bar, text=f"Tutorial ({os.path.basename(TUTORIAL_FILE)})", foreground="#555").pack(side=tk.LEFT)
        ttk.Button(bar, text="Open file", command=lambda: os.startfile(TUTORIAL_FILE)).pack(side=tk.RIGHT)
        box = ttk.Frame(f)
        box.pack(fill=tk.BOTH, expand=True, pady=4)
        t = tk.Text(box, wrap="word", font=("Segoe UI", 10), padx=14, pady=8, state="disabled", spacing3=3)
        sb = ttk.Scrollbar(box, orient=tk.VERTICAL, command=t.yview)
        t.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        t.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        for tag, cfg in (("h1", dict(font=("Segoe UI", 16, "bold"), spacing1=6, spacing3=8)),
                         ("h2", dict(font=("Segoe UI", 13, "bold"), foreground="#1565c0", spacing1=14, spacing3=4)),
                         ("bold", dict(font=("Segoe UI", 10, "bold"))),
                         ("code", dict(font=("Consolas", 9), foreground="#37474f", background="#f1f3f4")),
                         ("table", dict(font=("Consolas", 9), foreground="#263238")),
                         ("bullet", dict(lmargin1=24, lmargin2=40))):
            t.tag_configure(tag, **cfg)
        render_markdown(t, read_tutorial())
        self.tut_text = t

    def _build_tools(self):
        tk, ttk = self.tk, self.ttk
        from tkinter import messagebox
        frame = ttk.Frame(self.tab_tools, padding=8)
        frame.pack(fill=tk.BOTH, expand=True)

        def watch(p, log, script, tries=0):
            rc = p.poll()
            if rc is None:
                if tries < 6:
                    self.root.after(500, watch, p, log, script, tries + 1)
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
                self.root.after(500, watch, p, log, script)

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
        self._go_tool = go

    # ------------------------------------------------------------------ refresh
    def _tick(self):
        try:
            self.refresh()
        finally:
            self.root.after(self.REFRESH_MS, self._tick)

    def refresh(self, force=False):
        names = ex.list_experiments()
        rows = {}
        for n in names:
            try:
                e = ex.load(n)
                sm = ex.summary(e)
                rows[n] = (e.kind, f"{float(sm['spin']):.6g}" if sm["spin"] is not None else "?", sm["cases"],
                           f"{sm['done']}/{sm['total']}")
            except Exception as exc:   # a broken YAML is listed, its error shown when selected
                rows[n] = ("error", "", "", "")
                rows[n + "\0err"] = str(exc)
        have = set(self.tree.get_children())
        for n in names:
            vals = rows[n]
            tags = ()
            if vals[0] != "error":
                e = ex.load(n)
                states = {s for s, _ in ex.status(e).values()}
                tags = (("failed",) if "failed" in states else ("running",) if "running" in states
                        else ("reached",) if ex.next_step(e) is None else ())
            if n in have:
                self.tree.item(n, values=vals, tags=tags)
            else:
                self.tree.insert("", "end", iid=n, text=n, values=vals, tags=tags)
        for gone in have - set(names):
            self.tree.delete(gone)
        self._row_errors = {k[:-4]: v for k, v in rows.items() if k.endswith("\0err")}
        vals = sorted(n for n in names if rows[n][0] == "validation")
        self.cmp_cb_a["values"] = self.cmp_cb_b["values"] = vals
        if self.sel_exp not in names:
            self.sel_exp = None
            if names:
                self.tree.selection_set(names[0])
                return   # the selection event redraws
        self.redraw(force)

    def _on_select_exp(self):
        sel = self.tree.selection()
        new = sel[0] if sel else None
        if new != self.sel_exp:   # select() already set the experiment and its stage: keep that stage
            self.sel_stage = None
        self.sel_exp = new
        self.redraw(True)

    def exp(self):
        return ex.load(self.sel_exp) if self.sel_exp else None

    def redraw(self, force=False):
        if not self.sel_exp:
            self.canvas.delete("all")
            self._set_info([("No experiment. Use New DOE…, Import folder… or New validation…", None)])
            return
        if self.sel_exp in getattr(self, "_row_errors", {}):
            self.canvas.delete("all")
            self._set_info([(f"{self.sel_exp}: cannot load\n{self._row_errors[self.sel_exp]}", "bad")])
            return
        e = self.exp()
        self._fill_card(e)
        if self._goal_for != e.name:   # default goal by kind when an experiment is selected
            self.goal.set(ex.DEFAULT_GOAL.get(e.kind, "Training dataset"))
            self._goal_for = e.name
        st = ex.status(e)
        nxt = ex.next_step(e, self.goal.get())
        chain = {(x.name, k) for x, k in ex.goal_chain(e, self.goal.get())}
        snapshot = (e.name, json.dumps(st, sort_keys=True), self.goal.get(), self.sel_stage, str(nxt and nxt[:2]))
        if snapshot == self._last_draw and not force:
            return
        self._last_draw = snapshot
        if nxt is None:
            self.next_lbl.set("Goal reached ✓")
        else:
            where = "" if nxt[0] is e else f" of '{nxt[0].name}'"
            self.next_lbl.set(f"Next step: {ex.TITLES[nxt[1]]}{where}  ({nxt[2]}: {nxt[3]})")
        self._next = nxt
        self._enable(self.btn_run_next, bool(nxt and nxt[2] in ("pending", "stale", "failed")
                                             and not ex.run_blockers(nxt[0], nxt[1])))
        self._draw_diagram(e, st, chain, nxt)
        if self.sel_stage not in st:
            self.sel_stage = nxt[1] if nxt and nxt[0] is e else next(iter(st), None)
            self._draw_diagram(e, st, chain, nxt)
        self._show_stage(e, st)

    # ------------------------------------------------------------------ diagram
    def _draw_diagram(self, e, st, chain, nxt):
        c = self.canvas
        c.delete("all")
        S = ex.stages(e)
        keys = list(S)
        if not keys:   # e.g. a validation whose config the planner has not saved yet
            msg = "No stages yet:\n" + "\n".join(e.errors or ["the experiment has no runs"])
            c.create_text(MARGIN, MARGIN, text=msg, anchor="nw", fill="#c62828", font=("Segoe UI", 9),
                          width=max(c.winfo_width(), 600) - 2 * MARGIN)
            return
        ncols = 1 + max(diagram_layout(keys, e.training is not None, 1.0)[k][0] - MARGIN for k in keys)
        avail = max(c.winfo_width(), 600)
        step_x = max(110.0, min(STEP_X, (avail - 2 * MARGIN) / max(ncols, 1)))   # fit the panel width
        box_w = step_x - 26
        pos = diagram_layout(keys, e.training is not None, step_x)
        edges = [(dk if de is e else "@training", k, None if de is e else (4, 3))
                 for k, s in S.items() for de, dk in s.deps if (dk in pos if de is e else "@training" in pos)]
        incoming = {}
        for a, b, _ in edges:
            incoming.setdefault(b, []).append(a)
        for a, b, dash in edges:
            n = incoming[b]
            slot = (n.index(a) - (len(n) - 1) / 2) * 12 if len(n) > 1 and pos[a][1] != pos[b][1] else 0.0
            c.create_line(*edge_points(pos[a], pos[b], box_w, slot), arrow="last", fill="#90a4ae", width=1.5,
                          dash=dash)
        if "@training" in pos:
            x, y = pos["@training"]
            tst = ex.status(e.training).get("label_build", ("blocked", ""))[0]
            c.create_rectangle(x, y, x + box_w, y + BOX_H, fill=STATE_FILL[tst], outline="#78909c", dash=(4, 3),
                               tags=("ext",))
            c.create_text(x + box_w / 2, y + BOX_H / 2, text=f"training dataset\n({tst})", font=("Segoe UI", 8),
                          justify="center", tags=("ext",))
            c.tag_bind("ext", "<Button-1>", lambda _e: self.select(e.training.name, "label_build"))
        for k in keys:
            x, y = pos[k]
            state, _ = st[k]
            outline, width, dash = "#9e9e9e", 1, None
            if (e.name, k) in chain:
                outline, width = "#37474f", 2
            if nxt and nxt[0] is e and nxt[1] == k:
                outline, width = "#1565c0", 3
            if k == self.sel_stage:
                outline, width = "#6a1b9a", 3
            if state == "blocked":
                dash = (3, 2)
            tag = f"stage:{k}"
            c.create_rectangle(x, y, x + box_w, y + BOX_H, fill=STATE_FILL[state], outline=outline, width=width,
                               dash=dash, tags=(tag,))
            c.create_text(x + box_w / 2, y + 15, text=ex.TITLES[k], font=("Segoe UI", 9, "bold"), tags=(tag,))
            sub = STATE_TEXT[state] + ("  · optional" if k in ex.OPTIONAL else "")
            c.create_text(x + box_w / 2, y + 33, text=sub, font=("Segoe UI", 8), fill="#555", tags=(tag,))
            badge = ex.stage_badge(e, k)
            if badge:
                c.create_text(x + box_w / 2, y + 50, text=badge, font=("Segoe UI", 8, "bold"),
                              fill="#1565c0" if state == "running" else "#37474f", tags=(tag,))
            c.tag_bind(tag, "<Button-1>", lambda _e, k=k: self.select_stage(k))
            c.tag_bind(tag, "<Enter>", lambda _e, k=k, s=state, r=st[k][1]: self.hover.set(
                f"{ex.TITLES[k]} — {s}: {r}"))
            c.tag_bind(tag, "<Leave>", lambda _e: self.hover.set(""))

    def select_stage(self, key):
        self.sel_stage = key
        self.redraw(True)

    def select(self, exp_name, stage=None):
        self.tree.selection_set(exp_name)
        self.tree.see(exp_name)
        self.sel_exp = exp_name
        self.sel_stage = stage
        self.redraw(True)

    def go_next(self):
        nxt = getattr(self, "_next", None)
        if nxt:
            self.select(nxt[0].name, nxt[1])

    # ------------------------------------------------------------------ stage panel
    def _set_info(self, parts):
        t = self.info
        t.configure(state="normal")
        t.delete("1.0", "end")
        for text, tag in parts:
            t.insert("end", text, tag) if tag else t.insert("end", text)
        t.configure(state="disabled")

    def _fill_card(self, e):
        sm = ex.summary(e)
        self.card_title.set(e.name)
        bits = [e.kind]
        if sm["spin"] is not None:
            bits.append(f"n = {float(sm['spin']):.6g} rpm")
        bits.append(f"{sm['cases']} cases")
        if sm["kappa"]:
            bits.append(f"kappa {sm['kappa'][0]:g}-{sm['kappa'][1]:g}")
        bits.append(f"main path {sm['done']}/{sm['total']} done")
        self.card_line.set("  ·  ".join(bits))
        self.desc.set(e.cfg.get("description", ""))
        if e.training is not None:
            self.card_link.set(f"trained on: {e.training.name}")
        else:
            users = [n for n in ex.dependents(e.name)]
            self.card_link.set(f"reference for: {', '.join(users)}" if users else "")

    @staticmethod
    def _hint(reason: str) -> str:
        """What to do about a reason why a stage cannot run."""
        if "of experiment" in reason:
            return "→ 'Go to blocking experiment' and finish that stage there"
        if reason.startswith(("needs", "waits for")):
            return "→ run that earlier stage first (click its box)" if "needs" in reason else "→ wait until it finishes"
        if "imported run" in reason:
            return "→ data imported from a folder: use 'New DOE…' to simulate again"
        if reason.startswith("configuration error"):
            return "→ fix it with 'Edit config' or 'Edit YAML' (left)"
        if reason.startswith("input missing"):
            return "→ run the earlier stage that produces it (its box comes before this one)"
        if "is running and writes" in reason:
            return "→ wait for that stage to finish (two writers would damage the file)"
        return ""

    def _show_stage(self, e, st):
        k = self.sel_stage
        if not st:
            self._set_info([("This experiment has no stages yet.\n", "head")] +
                           [(f"  ERROR {x}\n", "bad") for x in ex.check(e)[0]])
            return
        if k is None or k not in st:
            self._set_info([("Click a stage of the diagram.", None)])
            return
        S = ex.stages(e)
        s = S[k]
        state, reason = st[k]
        self.panel.configure(text=f"Stage: {ex.TITLES[k]}")
        tag = {"done": "ok", "failed": "bad", "stale": "warn", "blocked": "warn", "running": "run"}.get(state)
        what, check = ex.STAGE_INFO.get(k, ("", ""))
        parts = [(f"{ex.TITLES[k]}   ", "title"), (state.upper(), tag), (f"  {reason}\n", None),
                 (what + "\n", "what")]
        if k in ex.OPTIONAL:
            parts.append((f"Optional stage: set it in the '{k}' section of the experiment YAML.\n", "hint"))
        # what the output holds
        summ = ex.stage_summary(e, k)
        if summ:
            parts.append(("\nResult\n", "head"))
            parts += [(f"  {t}\n", tg) for t, tg in summ]
        if check:
            parts.append((f"  Check: {check}\n", "hint"))
        # progress / timing
        rec = ex.read_record(e, k)
        tm = ex.run_timing(e, k)
        if state == "running":
            pr = ex.stage_progress(e, k)
            parts.append(("\nProgress\n", "head"))
            if pr and pr[1]:
                n = int(30 * pr[0] / pr[1])
                parts.append((f"  [{'#' * n}{'-' * (30 - n)}] {pr[0]}/{pr[1]}  ({100 * pr[0] / pr[1]:.0f} %)\n", "run"))
            parts.append((f"  running for {ex._fmt_dur(tm.get('elapsed'))}"
                          + (f", about {ex._fmt_dur(tm['eta'])} left" if tm.get("eta") else "") + "\n", "run"))
        if rec:
            parts.append(("\nLast run\n", "head"))
            line = f"  {rec.get('status')}  {ex._fmt_time(rec.get('end') or rec.get('start'))}"
            if tm.get("duration") is not None:
                line += f"  ·  took {ex._fmt_dur(tm['duration'])}"
            if rec.get("exit_code") not in (None, 0):
                line += f"  ·  exit code {rec.get('exit_code')}"
            parts.append((line + "\n", "bad" if rec.get("status") == "failed" else None))
            log = rec.get("log")
            if log and os.path.isfile(log) and state in ("failed", "running"):
                with open(log, encoding="utf-8", errors="replace") as f:
                    tail = [x for x in f.read().splitlines() if x.strip()][-LOG_TAIL:]
                parts.append(("  last lines of the log:\n" + "\n".join("    " + x for x in tail) + "\n", None))
        # why it cannot run + what to do
        blockers = ex.run_blockers(e, k) if state != "running" else ["already running"]
        if blockers and state != "running":
            parts.append(("\nCannot run now\n", "head"))
            for b in blockers:
                parts.append((f"  - {b}\n", "bad"))
                h = self._hint(b)
                if h:
                    parts.append((f"    {h}\n", "hint"))
        # files, short: names + their folder
        parts.append(("\nFiles\n", "head"))
        for title, paths in (("in ", s.inputs), ("out", s.outputs)):
            for pth in paths:
                if not pth:
                    continue
                ok = os.path.exists(pth)
                mark, mtag = ("✓", "ok") if ok else (("✗", "bad") if title == "in " else ("·", None))
                parts += [(f"  {title} {mark} ", mtag),
                          (f"{os.path.basename(pth)}   ", None), (f"{os.path.dirname(pth)}\n", "hint")]
        if s.cmds:
            parts.append(("\nCommand  ", "head"))
            parts.append((f"experiment.py run {e.name} {k}   (full command: 'Copy command')\n", "hint"))
        errs, warns = ex.check(e)
        if errs or warns:
            parts.append(("\nExperiment checks\n", "head"))
            parts += [(f"  ERROR {x}\n", "bad") for x in errs] + [(f"  warning {x}\n", "warn") for x in warns]
        self._set_info(parts)
        self._blocking = next(((de, dk) for de, dk in s.deps if de is not e
                               and ex.status(de)[dk][0] != "done"), None)
        btn = self.stage_btns
        self._enable(btn["run"], not blockers and bool(s.cmds))
        self._enable(btn["copy"], bool(s.cmds))
        self._enable(btn["log"], bool(rec and rec.get("log") and os.path.isfile(rec["log"])))
        self._enable(btn["view"], state != "running" and any(p.endswith(".h5") and os.path.isfile(p) for p in s.outputs))
        self._enable(btn["goto"], self._blocking is not None)
        self._enable(btn["labels"], k in ("label_template", "label_build") and os.path.isfile(e.label["labels_yaml"]))
        self._enable(btn["folder"], any(os.path.exists(os.path.dirname(p)) for p in s.outputs))

    @staticmethod
    def _enable(b, on):
        b.state(["!disabled"] if on else ["disabled"])

    # ------------------------------------------------------------------ stage actions
    def run_stage(self):
        from tkinter import messagebox
        e, k = self.exp(), self.sel_stage
        blockers = ex.run_blockers(e, k)
        if blockers:
            messagebox.showerror("Cannot run", "\n".join(blockers))
            return
        old = ex.existing_outputs(e, k)
        if old:
            lst = "\n".join(f"  {os.path.basename(p)}  ({ex._fmt_time(ex._mtime(p))})" for p in old)
            if not messagebox.askyesno("Replace outputs?", f"This run replaces:\n{lst}\n\nContinue?"):
                return
        py, warn = stage_python()
        if warn:
            messagebox.showwarning("Python", warn)
        open_console(ex.run_command(e.name, k, py, yes=True))
        self.root.after(1500, lambda: self.refresh(True))

    def run_next(self):
        nxt = getattr(self, "_next", None)
        if nxt:
            self.select(nxt[0].name, nxt[1])
            self.run_stage()

    def open_labels(self):
        os.startfile(self.exp().label["labels_yaml"])

    def show_tutorial(self):
        self.notebook.select(self.tab_tut)

    def show_help(self):
        tk = self.tk
        w = tk.Toplevel(self.root)
        w.title("How the app works")
        t = tk.Text(w, width=104, height=30, wrap="word", font=("Segoe UI", 10))
        t.insert("1.0", HELP)
        t.configure(state="disabled")
        t.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def copy_cmd(self):
        e, k = self.exp(), self.sel_stage
        self.root.clipboard_clear()
        self.root.clipboard_append(quote(ex.run_command(e.name, k, stage_python()[0], yes=False)))

    def open_log(self):
        rec = ex.read_record(self.exp(), self.sel_stage)
        if rec and rec.get("log"):
            os.startfile(rec["log"])

    def open_output(self):
        s = ex.stages(self.exp())[self.sel_stage]
        h5 = next(p for p in s.outputs if p.endswith(".h5") and os.path.isfile(p))
        launch("gui", "DOE_plots/doe_unified_selector.py", f'--h5 "{h5}"')

    def open_folder(self):
        s = ex.stages(self.exp())[self.sel_stage]
        d = next(os.path.dirname(p) for p in s.outputs if os.path.exists(os.path.dirname(p)))
        os.startfile(d)

    def goto_blocker(self):
        if self._blocking:
            self.select(self._blocking[0].name, self._blocking[1])

    def edit_stage(self):
        e, k = self.exp(), self.sel_stage
        if k in ("simulate", "extract"):
            cfgs = [r.config for r in e.runs if r.config]
            if not cfgs:
                self._msg("Imported run", "This run was imported from a folder: it has no config to edit.")
                return
            launch("gui", "DOE_simulacion/doe_planner.py", quote(cfgs))
        elif k in ("label_template", "label_build"):
            LabelForm(self, e)
        elif k in ("indicators", "noise_indicators"):
            IndicatorsForm(self, e)
        elif k == "validate":
            ValidateForm(self, e)
        elif k == "merge":
            MergeForm(self, e)
        else:
            self._msg("Optional stage", f"Edit the '{k}' section of\n{e.path}\n(keys = the lower-case CONFIG names "
                                        "of its script). The file opens now.")
            os.startfile(e.path)

    def _msg(self, title, text, kind="info"):
        from tkinter import messagebox
        getattr(messagebox, {"info": "showinfo", "error": "showerror", "warn": "showwarning"}[kind])(title, text)

    # ------------------------------------------------------------------ experiments buttons
    def edit_yaml(self):
        if self.sel_exp:
            os.startfile(ex.exp_path(self.sel_exp))

    def new_doe(self):
        NewDoeDialog(self)

    def import_folder(self):
        ImportDialog(self)

    def new_validation(self):
        NewValidationDialog(self)

    def derive(self):
        if self.sel_exp:
            DeriveDialog(self, self.exp())

    def duplicate(self):
        from tkinter import simpledialog
        if not self.sel_exp:
            return
        name = simpledialog.askstring("Duplicate", "Name of the copy:", initialvalue=self.sel_exp + "_copy",
                                      parent=self.root)
        if name:
            self._guard(lambda: ex.duplicate(self.sel_exp, name.strip()), select=name.strip())

    def delete(self):
        from tkinter import messagebox
        if not self.sel_exp:
            return
        if messagebox.askyesno("Delete experiment",
                               f"Delete '{self.sel_exp}'?\n\nOnly its YAML and run records are removed. "
                               "Data (.h5, DOE folders, configs) are never touched."):
            self._guard(lambda: ex.delete(self.sel_exp))

    def _guard(self, fn, select=None):
        """Run an action; show its error instead of crashing; refresh and select the result."""
        try:
            fn()
        except Exception as exc:
            self._msg("Error", str(exc), "error")
            return False
        ex.reload()
        self.refresh(True)
        if select and select in ex.list_experiments():
            self.select(select)
        return True

    # ------------------------------------------------------------------ compare
    def compare(self):
        a, b = self.cmp_a.get(), self.cmp_b.get()
        if not a or not b:
            return
        ma, mb = ex.validation_metrics(ex.load(a)), ex.validation_metrics(ex.load(b))
        self.cmp_tree.delete(*self.cmp_tree.get_children())
        if not ma and not mb:
            self.cmp_note.set("Neither experiment has validation results yet.")
            return
        fmt = lambda m, k: "" if m is None or m.get(k) is None else (f"{m[k]:.3f}" if isinstance(m[k], float) else str(m[k]))
        for v in sorted(set(ma) | set(mb)):
            row = [v] + [fmt(src.get(v), m) for m in ex.METRIC_COLUMNS for src in (ma, mb)]
            self.cmp_tree.insert("", "end", values=row)
        self.cmp_note.set(f"A = {a}   B = {b}   (empty = that variant was not run in that experiment)")


# ============================================================================== dialogs
class _Dialog:
    """Small modal form: rows of (label, widget); Save calls self.save()."""

    def __init__(self, app, title):
        tk, ttk = app.tk, app.ttk
        self.app, self.tk, self.ttk = app, tk, ttk
        self.win = tk.Toplevel(app.root)
        self.win.title(title)
        self.win.transient(app.root)
        self.body = ttk.Frame(self.win, padding=10)
        self.body.pack(fill=tk.BOTH, expand=True)
        self.row = 0

    def field(self, label, var=None, width=40, values=None, state="normal", note=""):
        tk, ttk = self.tk, self.ttk
        var = var if var is not None else tk.StringVar()
        ttk.Label(self.body, text=label).grid(row=self.row, column=0, sticky="w", pady=2)
        if values is not None:
            w = ttk.Combobox(self.body, textvariable=var, values=values, width=width - 2,
                             state="readonly" if state == "normal" else state)
        else:
            w = ttk.Entry(self.body, textvariable=var, width=width, state=state)
        w.grid(row=self.row, column=1, sticky="w", pady=2)
        if note:
            ttk.Label(self.body, text=note, foreground="#666").grid(row=self.row, column=2, sticky="w", padx=6)
        self.row += 1
        return var

    def buttons(self, ok_text="Save"):
        ttk = self.ttk
        bf = ttk.Frame(self.win, padding=(10, 0, 10, 10))
        bf.pack(fill="x")
        ttk.Button(bf, text=ok_text, command=self._ok).pack(side="right")
        ttk.Button(bf, text="Cancel", command=self.win.destroy).pack(side="right", padx=6)

    def _ok(self):
        try:
            done = self.save()
        except Exception as exc:
            self.app._msg("Error", str(exc), "error")
            return
        if done is not False:
            self.win.destroy()
            ex.reload()
            self.app.refresh(True)


def _num(text, kind=float, allow_none=True):
    text = str(text).strip()
    if text in ("", "None", "none") and allow_none:
        return None
    return kind(text)


class LabelForm(_Dialog):
    FIELDS = ("strategy", "amp_signal", "base_attr", "base_scale", "lim_inf_pct", "lim_sup_pct", "warmup",
              "kappa_threshold", "t_start", "t_end")

    def __init__(self, app, e):
        super().__init__(app, f"Labelling — {e.name}")
        self.e = e
        locked = e.training is not None
        if locked:
            self.ttk.Label(self.body, foreground="#b26a00",
                           text=f"Inherited from the training '{e.training.name}': validation labels must use the "
                                "same parameters (read-only).").grid(row=0, column=0, columnspan=3, sticky="w")
            self.row = 1
        self.vars = {}
        for k in self.FIELDS:
            v = self.tk.StringVar(value="" if e.label.get(k) is None else str(e.label.get(k)))
            vals = ["amplitude", "kappa", "manual"] if k == "strategy" else (
                ["Axial_disp", "Axial_vel", "Axial_acc", "Axial_disp_out_deflex"] if k == "amp_signal" else None)
            self.vars[k] = self.field(k, v, values=vals, state="disabled" if locked else "normal")
        self.field("labels YAML", self.tk.StringVar(value=e.label["labels_yaml"]), width=70, state="readonly")
        self.field("dataset out", self.tk.StringVar(value=e.label["out"]), width=70, state="readonly")
        bf = self.ttk.Frame(self.body)
        bf.grid(row=self.row, column=0, columnspan=3, sticky="w", pady=4)
        self.ttk.Button(bf, text="Open labels YAML (review)",
                        command=lambda: os.path.isfile(e.label["labels_yaml"]) and os.startfile(e.label["labels_yaml"])
                        ).pack(side="left")
        if not locked:
            self.buttons()
        else:
            self.buttons("Close")

    def save(self):
        if self.e.training is not None:
            return True
        d = {}
        for k, v in self.vars.items():
            t = v.get().strip()
            if t == "":
                continue
            d[k] = t if k in ("strategy", "amp_signal", "base_attr") else float(t)
        old = ex.own_yaml(self.e.name).get("label") or {}
        for k in ("out", "labels_yaml"):   # explicit paths (imported experiments) are kept
            if k in old:
                d[k] = old[k]
        ex.save_section(self.e.name, "label", d)
        return True


class MergeForm(_Dialog):
    def __init__(self, app, e):
        super().__init__(app, f"Merge — {e.name}")
        self.e = e
        self.into = self.field("merged folder name", self.tk.StringVar(value=(e.cfg.get("merge") or {}).get("into", "")),
                               note="new folder next to the runs (doe_runner --merge_out copies the first run)")
        self.buttons()

    def save(self):
        if not self.into.get().strip():
            raise ValueError("give the name of the merged folder")
        ex.save_section(self.e.name, "merge", {"into": self.into.get().strip()})


class ValidateForm(_Dialog):
    def __init__(self, app, e):
        super().__init__(app, f"Validate — {e.name}")
        self.e = e
        self.ch = self.field("channel of the labels", self.tk.StringVar(value=e.section("validate").get("channel", "Axial_disp")),
                             values=["Axial_disp", "Axial_vel", "Axial_acc"])
        self.buttons()

    def save(self):
        sec = ex.own_yaml(self.e.name).get("validate") or {}
        sec["channel"] = self.ch.get()
        ex.save_section(self.e.name, "validate", sec)


class IndicatorsForm(_Dialog):
    def __init__(self, app, e):
        tk, ttk = app.tk, app.ttk
        super().__init__(app, f"Indicators — {e.name}")
        self.e = e
        lib = ex.variants_library().get("variants") or {}
        own = ex.own_yaml(e.name).get("indicators") or {}
        self.inherit = tk.BooleanVar(value=e.training is not None and own.get("variants", "inherit") == "inherit")
        if e.training is not None:
            ttk.Checkbutton(self.body, text=f"Same variants as the training '{e.training.name}'", variable=self.inherit,
                            command=self._toggle).grid(row=self.row, column=0, columnspan=3, sticky="w")
            self.row += 1
        box = ttk.LabelFrame(self.body, text="Variants (library: experiments/indicator_variants.yaml)", padding=4)
        box.grid(row=self.row, column=0, columnspan=3, sticky="nsew", pady=4)
        self.row += 1
        self.checks, self.focus = {}, tk.StringVar(value=next(iter(lib), ""))
        for i, (name, v) in enumerate(lib.items()):
            var = tk.BooleanVar(value=name in e.indicators["variants"])
            pp = v["params_physical"]
            win = pp.get("N_rev_window", pp.get("N_modal_window"))
            txt = f"{name}   [{v['indicator']}/{v.get('func', 'Default')}, {v['signal']}, {v['mode']}, window {win}]"
            cb = ttk.Checkbutton(box, text=txt, variable=var)
            cb.grid(row=i, column=0, sticky="w")
            ttk.Radiobutton(box, variable=self.focus, value=name).grid(row=i, column=1, padx=6)
            self.checks[name] = (var, cb)
        ttk.Label(box, text="(the round button picks the variant for Duplicate / Edit / Show)",
                  foreground="#666").grid(row=len(lib), column=0, sticky="w")
        self.f_modal = self.field("f_modal [Hz]", tk.StringVar(value=str(e.indicators.get("f_modal") or "")),
                                  note="only by_modal variants use it")
        cases = e.indicators.get("cases", "all")
        self.cases = self.field("cases", tk.StringVar(value=cases if isinstance(cases, str) else " ".join(cases)),
                                note="'all' or case_000 case_003 …")
        self.workers = self.field("workers", tk.StringVar(value=str(e.indicators.get("workers") or 6)))
        bf = ttk.Frame(self.body)
        bf.grid(row=self.row, column=0, columnspan=3, sticky="w", pady=4)
        ttk.Button(bf, text="Duplicate variant…", command=lambda: VariantEditor(app, self, self.focus.get(), True)).pack(side="left")
        ttk.Button(bf, text="Edit variant…", command=lambda: VariantEditor(app, self, self.focus.get(), False)).pack(side="left", padx=4)
        ttk.Button(bf, text="Show resolved config for a case…", command=self._resolved).pack(side="left")
        self._toggle()
        self.buttons()

    def _toggle(self):
        for var, cb in self.checks.values():
            cb.state(["disabled"] if self.inherit.get() else ["!disabled"])

    def _resolved(self):
        info = ex.h5_info(self.e.data_h5)
        if not info:
            self.app._msg("No data", "The experiment data (doe_results.h5) does not exist yet.", "warn")
            return
        ResolvedView(self.app, self.e, self.focus.get(), info["cases"])

    def reload_library(self):
        self.win.destroy()
        IndicatorsForm(self.app, ex.load(self.e.name))

    def save(self):
        d = ex.own_yaml(self.e.name).get("indicators") or {}
        d["variants"] = "inherit" if self.inherit.get() else [n for n, (v, _) in self.checks.items() if v.get()]
        d["f_modal"] = _num(self.f_modal.get())
        c = self.cases.get().strip()
        d["cases"] = "all" if c in ("", "all") else c.replace(",", " ").split()
        d["workers"] = int(self.workers.get())
        ex.save_section(self.e.name, "indicators", {k: v for k, v in d.items() if v is not None})


class VariantEditor(_Dialog):
    def __init__(self, app, form, name, new):
        import yaml
        super().__init__(app, ("Duplicate" if new else "Edit") + f" variant {name}")
        self.form, self.new, self.orig = form, new, name
        lib = ex.variants_library().get("variants") or {}
        if name not in lib:
            self.win.destroy()
            app._msg("Variant", "Pick a variant with the round button first.", "warn")
            return
        if not new:
            used = ex.variant_used(name)
            if used:
                self.win.destroy()
                app._msg("Variant in use", f"'{name}' already has results in {used}.\nDuplicate it to change it.", "warn")
                return
        spec = lib[name]
        self.name = self.field("name", self.tk.StringVar(value=ex.propose_variant_name(spec) if new else name),
                               state="normal" if new else "readonly")
        self.ttk.Label(self.body, text="YAML of the variant (T_rev / T_modal / 10*T_rev are resolved per case):"
                       ).grid(row=self.row, column=0, columnspan=3, sticky="w")
        self.row += 1
        self.text = self.tk.Text(self.body, width=80, height=26, font=("Consolas", 9))
        self.text.grid(row=self.row, column=0, columnspan=3)
        self.text.insert("1.0", yaml.safe_dump(spec, sort_keys=False))
        self.buttons()

    def save(self):
        import yaml
        spec = yaml.safe_load(self.text.get("1.0", "end"))
        ex.save_variant(self.name.get().strip(), spec, self.new)
        self.form.win.after(10, self.form.reload_library)


class ResolvedView(_Dialog):
    def __init__(self, app, e, variant, cases):
        super().__init__(app, f"Resolved config — {variant}")
        self.e, self.variant = e, variant
        self.case = self.field("case", self.tk.StringVar(value=cases[0]), values=cases)
        self.case.trace_add("write", lambda *_: self._show())
        self.text = self.tk.Text(self.body, width=90, height=30, font=("Consolas", 9))
        self.text.grid(row=self.row, column=0, columnspan=3)
        self._show()
        self.buttons("Close")

    def _show(self):
        try:
            d = ex.resolve(self.e, self.variant, self.case.get())
            txt = json.dumps(d, indent=1, default=str)
        except Exception as exc:
            txt = f"error: {exc}"
        self.text.delete("1.0", "end")
        self.text.insert("1.0", txt)

    def save(self):
        return True


class NewDoeDialog(_Dialog):
    """New DOE to simulate: config from a template (only n, kappa/Ap and the name change) + its experiment."""

    def __init__(self, app):
        tk = app.tk
        super().__init__(app, "New DOE")
        trainings = [n for n in ex.list_experiments() if self._kind(n) == "training"]
        self.kind = self.field("kind", tk.StringVar(value="training"), values=["training", "validation"])
        self.training = self.field("training (validation only)", tk.StringVar(value=trainings[0] if trainings else ""),
                                   values=trainings)
        cfgs = ex.list_configs()
        self.template = self.field("template config", tk.StringVar(value=cfgs[0] if cfgs else ""), values=cfgs,
                                   note="base_dir, case, discretisation and ap_ref come from it")
        self.spin = self.field("n [rpm]", tk.StringVar(), note="empty = the template's")
        self.kappa = self.field("kappa", tk.StringVar(value="0.5:2.0:0.1"), note="list '0.5, 0.9' or start:stop:step")
        self.apref = self.field("ap_ref from", tk.StringVar(value="template"),
                                values=["template"] + self._presets(), note="or the SLD limit at this n")
        self.doe = self.field("doe_name", tk.StringVar())
        self.name = self.field("experiment name", tk.StringVar())
        self.descr = self.field("description", tk.StringVar(), width=60)
        self.ttk.Button(self.body, text="Propose names", command=self._propose).grid(row=self.row, column=1, sticky="w")
        self.row += 1
        self._propose()
        self.buttons("Create")

    @staticmethod
    def _kind(n):
        try:
            return ex.load(n).kind
        except Exception:
            return ""

    @staticmethod
    def _presets():
        try:
            if ex.PLOTS not in sys.path:
                sys.path.insert(0, ex.PLOTS)
            import sld_model
            return list(sld_model.MODELS)
        except Exception:
            return []

    def _tpl(self):
        dr = ex._doe_runner()
        return dr.load_config(dr.find_config(self.template.get()))

    def _propose(self):
        try:
            cfg = self._tpl()
            spin = _num(self.spin.get()) or (cfg.get("sweep") or cfg.get("factorial") or {}).get("$spin_rate$", [None])[0]
            ks = parse_kappas(self.kappa.get())
            name = ex.propose_name(self.kind.get(), cfg.get("case"), spin, ks)
            self.name.set(name)
            self.doe.set("DOE_" + name)
            self.descr.set(f"{self.kind.get()}, {cfg.get('case')}, n = {float(spin):g} rpm, {len(ks)} kappa "
                           f"{min(ks):g}-{max(ks):g}, template {self.template.get()}")
        except Exception as exc:
            self.descr.set(f"(cannot propose: {exc})")

    def save(self):
        cfg = self._tpl()
        spin = _num(self.spin.get()) or (cfg.get("sweep") or cfg.get("factorial") or {}).get("$spin_rate$", [None])[0]
        if spin is None:
            raise ValueError("give n (the template has no $spin_rate$)")
        ks = parse_kappas(self.kappa.get())
        ap_ref = None
        if self.apref.get() != "template":
            import sld_model
            ap_ref = sld_model.ap_lim(self.apref.get(), float(spin)) / 1e3
        kind, name = self.kind.get(), self.name.get().strip()
        if kind == "validation" and not self.training.get():
            raise ValueError("a validation needs its training experiment")
        if os.path.exists(ex.exp_path(name)):
            raise ValueError(f"experiment '{name}' already exists")
        ex.new_doe_config(self.template.get(), self.doe.get().strip(), float(spin), kappas=ks, ap_ref=ap_ref,
                          header=f" for experiment {name}")
        sections = {} if kind == "validation" else {"label": dict(ex.LABEL_DEFAULTS),
                                                    "indicators": dict(ex.INDICATOR_DEFAULTS)}
        if kind == "validation":
            sections = {"indicators": {"variants": "inherit"}, "validate": {"channel": "Axial_disp"}}
        ex.create_experiment(name, kind, [{"config": self.doe.get().strip()}],
                             training=self.training.get() if kind == "validation" else None,
                             description=self.descr.get(), **sections)
        self.app.root.after(50, lambda: self.app.select(name))


class ImportDialog(_Dialog):
    def __init__(self, app):
        from tkinter import filedialog
        tk = app.tk
        d = filedialog.askdirectory(title="DOE folder (with doe_results.h5)", parent=app.root)
        super().__init__(app, "Import folder")
        if not d:
            self.win.destroy()
            return
        self.dir = d
        info = ex.h5_info(os.path.join(d, "doe_results.h5"))
        trainings = [n for n in ex.list_experiments() if NewDoeDialog._kind(n) == "training"]
        self.field("folder", tk.StringVar(value=d), width=70, state="readonly")
        self.kind = self.field("kind", tk.StringVar(value="training"), values=["training", "validation"])
        self.training = self.field("training (validation only)", tk.StringVar(value=trainings[0] if trainings else ""),
                                   values=trainings)
        spin = info["first"].get("$spin_rate$") if info else None
        self.name = self.field("experiment name", tk.StringVar(
            value=ex.propose_name("training", ex.detect_case(d), spin, info["kappa"] if info else None)))
        self.buttons("Import")

    def save(self):
        if not os.path.isfile(os.path.join(self.dir, "doe_results.h5")):
            raise ValueError("the folder has no doe_results.h5")
        kind = self.kind.get()
        ex.import_dir(self.name.get().strip(), self.dir, kind, self.training.get() if kind == "validation" else None)
        name = self.name.get().strip()
        self.app.root.after(50, lambda: self.app.select(name))


class NewValidationDialog(_Dialog):
    """Validation linked to a training: doe_val_planner opens with the training dataset and the DOE name; the
    experiment is created now and becomes ready once the planner has saved configs/<doe_name>.yaml."""

    def __init__(self, app):
        tk = app.tk
        super().__init__(app, "New validation")
        trainings = [n for n in ex.list_experiments() if NewDoeDialog._kind(n) == "training"]
        if not trainings:
            self.win.destroy()
            app._msg("New validation", "Create or import a training experiment first.", "warn")
            return
        self.training = self.field("training", tk.StringVar(value=trainings[0]), values=trainings)
        self.name = self.field("experiment name", tk.StringVar())
        self.doe = self.field("doe_name (planner)", tk.StringVar())
        self.training.trace_add("write", lambda *_: self._propose())
        self._propose()
        self.ttk.Label(self.body, foreground="#555", text="The validation planner opens with this doe_name: generate the "
                       "cases and save the YAML in configs/ with that name.").grid(row=self.row, column=0, columnspan=3,
                                                                                   sticky="w")
        self.buttons("Create and open planner")

    def _propose(self):
        t = ex.load(self.training.get())
        sm = ex.summary(t)
        name = ex.propose_name("validation", t.runs[0].case if t.runs else "doe", sm["spin"], None)
        self.name.set(name)
        self.doe.set("DOE_" + name)

    def save(self):
        t = ex.load(self.training.get())
        name, doe = self.name.get().strip(), self.doe.get().strip()
        ex.create_experiment(name, "validation", [{"config": doe}], training=t.name,
                             description=f"validation of the indicators against {t.name}",
                             indicators={"variants": "inherit"}, validate={"channel": "Axial_disp"})
        launch("gui", "DOE_simulacion/doe_val_planner.py", quote([t.label["out"], "--name", doe]))
        self.app.root.after(50, lambda: self.app.select(name))


class DeriveDialog(_Dialog):
    def __init__(self, app, parent):
        tk = app.tk
        super().__init__(app, f"Derive from {parent.name}")
        self.parent = parent
        sm = ex.summary(parent)
        self.spin = self.field("new n [rpm]", tk.StringVar(), note="empty = same n (e.g. only other variants)")
        self.apref = self.field("Ap at the new n", tk.StringVar(value="same Ap"),
                                values=["same Ap"] + NewDoeDialog._presets(),
                                note="or the same kappa with the SLD limit of a model at the new n")
        lib = list((ex.variants_library().get("variants") or {}))
        self.variants = self.field("variants", tk.StringVar(value="same"), width=60,
                                   note="'same' or names separated by spaces")
        self.lib_hint = ", ".join(lib)
        self.name = self.field("experiment name", tk.StringVar(value=parent.name + "_derived"))
        self.spin.trace_add("write", lambda *_: self._propose(sm))
        self.buttons("Derive")

    def _propose(self, sm):
        try:
            spin = float(self.spin.get())
            case = self.parent.runs[0].case if self.parent.runs else "doe"
            self.name.set(ex.propose_name(self.parent.kind, case, spin, None))
        except ValueError:
            pass

    def save(self):
        spin = _num(self.spin.get())
        v = self.variants.get().strip()
        variants = None if v in ("", "same") else v.replace(",", " ").split()
        preset = None if self.apref.get() == "same Ap" else self.apref.get()
        name = self.name.get().strip()
        ex.derive(self.parent.name, name, spin=spin, variants=variants, ap_ref_preset=preset)
        self.app.root.after(50, lambda: self.app.select(name))


# ============================================================================== entry points
def run_gui():
    import tkinter as tk
    root = tk.Tk()
    try:
        root.state("zoomed")   # maximised: the diagram and the stage panel need the room
    except tk.TclError:
        root.geometry("1500x900")
    App(root)
    root.mainloop()


def screenshot(out: str, exp_name: str | None = None, stage: str | None = None):
    """Open the app, select an experiment/stage, save a capture of the window, close."""
    import tkinter as tk
    from PIL import ImageGrab
    root = tk.Tk()
    root.geometry("1300x800+10+10")
    root.attributes("-topmost", True)   # in front of everything while it is captured
    app = App(root, auto_refresh=False)
    if exp_name == "@tutorial":   # capture of the Tutorial tab
        app.show_tutorial()
        if stage:
            app.tut_text.see(app.tut_text.search(stage, "1.0") or "1.0")
    elif exp_name:
        app.select(exp_name, stage)
    root.update()
    root.lift()
    root.focus_force()
    try:
        import ctypes
        f = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100.0   # e.g. 1.5 at 150 % display scaling
    except Exception:
        f = 1.0

    def grab():
        root.update()
        x, y, w, h = root.winfo_rootx(), root.winfo_rooty(), root.winfo_width(), root.winfo_height()
        ImageGrab.grab(bbox=tuple(int(v * f) for v in (x, y, x + w, y + h))).save(out)
        root.destroy()

    root.after(1200, grab)
    root.mainloop()
    print("saved", out)


def _selftest():
    for kind, script, _ in TOOLS:
        assert kind in ("gui", "cli") and os.path.isfile(os.path.join(HERE, script)), script
    c = build_cmd("cli", "DOE_simulacion/doe_runner.py", "")
    assert c[:2] == ["cmd", "/k"] and c[-1] == "--help"
    assert build_cmd("cli", "DOE_simulacion/doe_runner.py", "--config tube --dry-run")[-3:] == ["--config", "tube", "--dry-run"]
    assert parse_kappas("0.5:0.8:0.1") == [0.5, 0.6, 0.7, 0.8] and parse_kappas("0.5, 1.2;1.4") == [0.5, 1.2, 1.4]
    assert edge_points((0, 0), (100, 0), 80) == [80, BOX_H / 2, 100, BOX_H / 2]
    pts = edge_points((0, 0), (300, STEP_Y), 80)
    assert len(pts) == 8 and pts[3] == pts[5] == STEP_Y - (STEP_Y - BOX_H) / 2   # through the corridor
    lay = diagram_layout(["simulate", "extract", "label_template", "label_build", "indicators", "validate"], True)
    assert lay["label_template"][0] == MARGIN + 2 * STEP_X and lay["@training"][0] == lay["indicators"][0] - STEP_X
    assert diagram_layout(["simulate", "extract", "merge", "label_template"], False)["label_template"][0] == MARGIN + 3 * STEP_X
    md = "# T" + chr(10) + "## S" + chr(10) + "a **b** `c`" + chr(10) + "| x | y |" + chr(10) + "|---|---|" + chr(10) \
        + "| 1 | 2 |" + chr(10) + "- item" + chr(10)
    assert os.path.isfile(TUTORIAL_FILE) and "## 3." in read_tutorial()
    # the window builds and draws every real experiment without errors (no mainloop)
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    try:
        app = App(root, auto_refresh=False)
        render_markdown(app.tut_text, md)
        txt = app.tut_text.get("1.0", "end")
        assert "T" in txt and "b c" in txt and "---" not in txt and "x  |  y" in txt and "item" in txt, txt
        render_markdown(app.tut_text, read_tutorial())
        assert "Recorrido completo" in app.tut_text.get("1.0", "end")
        for n in ex.list_experiments():
            app.select(n)
            st = ex.status(ex.load(n))
            for k in st:
                app.select_stage(k)
                assert app.canvas.find_withtag(f"stage:{k}"), (n, k)
                assert app.info.get("1.0", "end").strip(), (n, k)
    finally:
        root.destroy()
    print("launcher selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    elif "--screenshot" in sys.argv:
        i = sys.argv.index("--screenshot")
        rest = sys.argv[i + 1:]
        screenshot(rest[0], *(rest[1:3]))
    else:
        run_gui()
