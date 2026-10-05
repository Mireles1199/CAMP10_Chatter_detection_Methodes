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
              "pending": "#eeeeee", "blocked": "#fafafa", "skipped": "#e8f5e9"}
STATE_TEXT = {"done": "done", "stale": "stale", "running": "running", "failed": "failed", "pending": "pending",
              "blocked": "blocked", "skipped": "not needed"}
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


TUT_IMG_DIR = os.path.join(HERE, "tutorial_img")
TUT_WIDTH = 980   # px of the reading column; tables and images are fitted to it


def configure_tutorial_tags(t, size: int = 10) -> None:
    t.configure(font=("Segoe UI", size), spacing3=4, background="#ffffff")
    for tag, cfg in (("h1", dict(font=("Segoe UI", size + 8, "bold"), spacing1=8, spacing3=10)),
                     ("h2", dict(font=("Segoe UI", size + 4, "bold"), foreground="#1565c0", spacing1=18, spacing3=6)),
                     ("h3", dict(font=("Segoe UI", size + 1, "bold"), spacing1=8)),
                     ("bold", dict(font=("Segoe UI", size, "bold"))),
                     ("ital", dict(font=("Segoe UI", size, "italic"))),
                     ("code", dict(font=("Consolas", size), foreground="#37474f", background="#eceff1")),
                     ("codeblock", dict(font=("Consolas", size - 1), foreground="#263238", background="#eceff1",
                                        lmargin1=20, lmargin2=20, spacing1=1, spacing3=1)),
                     ("bullet", dict(lmargin1=22, lmargin2=40, spacing3=3)),
                     ("num", dict(lmargin1=22, lmargin2=42, spacing3=3)),
                     ("note", dict(background="#fff8e1", lmargin1=18, lmargin2=18, rmargin=18, spacing1=6,
                                   spacing3=6, foreground="#5d4037")),
                     ("cap", dict(font=("Segoe UI", size - 1, "italic"), foreground="#607d8b", justify="center",
                                  spacing3=10))):
        t.tag_configure(tag, **cfg)


def _md_table(widget, rows: list, size: int, width: int) -> None:
    """A real grid (one Label per cell) embedded in the Text: header row, striped rows, wrapped cells."""
    import tkinter as tk
    fr = tk.Frame(widget, bg="#b0bec5")
    ncol = max(len(r) for r in rows)
    weight = [max(8, min(40, max(len(r[c]) if c < len(r) else 0 for r in rows))) for c in range(ncol)]
    total = sum(weight)
    for ri, r in enumerate(rows):
        for ci in range(ncol):
            wrap = int((width - 60) * weight[ci] / total)
            bg = "#e3f2fd" if ri == 0 else ("#ffffff" if ri % 2 else "#f5f7f8")
            tk.Label(fr, text=r[ci] if ci < len(r) else "", justify="left", anchor="nw", wraplength=wrap - 14,
                     bg=bg, font=("Segoe UI", size, "bold" if ri == 0 else "normal"), padx=7, pady=4
                     ).grid(row=ri, column=ci, sticky="nsew", padx=(0, 1), pady=(0, 1))
    widget.window_create("end", window=fr, padx=8, pady=8)
    widget.insert("end", chr(10))


def render_markdown(widget, text: str, size: int = 10, width: int = TUT_WIDTH, base_dir: str = TUT_IMG_DIR) -> list:
    """Markdown subset into a Tk Text, read-only afterwards. Supports # / ## / ### headings, **bold**, `code`,
    ``` blocks, pipe tables (drawn as grids), - and 1. lists, '> ' notes and ![caption](file) images (scaled to
    the column). Returns [(title, mark)] for each '## ' section so the app can build an index."""
    import math
    import re
    import tkinter as tk
    configure_tutorial_tags(widget, size)
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    widget._images = []
    sections, in_code, table = [], False, []

    def inline(line, base=()):
        pos = 0
        for m in re.finditer(r"\*\*(.+?)\*\*|`([^`]+)`|(?<![*\w])\*([^*\s][^*]*?)\*(?![*\w])", line):
            widget.insert("end", line[pos:m.start()], base)
            kind = "bold" if m.group(1) else ("code" if m.group(2) else "ital")
            widget.insert("end", m.group(1) or m.group(2) or m.group(3), base + (kind,))
            pos = m.end()
        widget.insert("end", line[pos:], base)

    def flush_table():
        nonlocal table
        if table:
            _md_table(widget, table, size, width)
            table = []

    for raw in text.splitlines():
        if raw.strip().startswith("```"):
            flush_table()
            in_code = not in_code
            continue
        if in_code:
            widget.insert("end", raw + chr(10), ("codeblock",))
            continue
        if raw.lstrip().startswith("|"):
            if not re.fullmatch(r"[|\s:-]+", raw.strip()):
                table.append([c.strip().replace("**", "").replace("`", "") for c in raw.strip().strip("|").split("|")])
            continue
        flush_table()
        m_img = re.fullmatch(r"!\[(.*?)\]\((.+?)\)", raw.strip())
        if m_img:
            path = os.path.join(base_dir, m_img.group(2))
            try:
                img = tk.PhotoImage(file=path)
                z = 2
                img = img.zoom(z).subsample(max(1, math.ceil(z * img.width() / (width - 20))))
                widget._images.append(img)
                widget.insert("end", chr(10))
                widget.image_create("end", image=img, padx=10, pady=4)
                widget.insert("end", chr(10) + m_img.group(1) + chr(10), ("cap",))
            except tk.TclError:
                widget.insert("end", f"[image missing: {m_img.group(2)}]" + chr(10), ("note",))
        elif raw.startswith("# "):
            widget.insert("end", raw[2:] + chr(10), ("h1",))
        elif raw.startswith("## "):
            n = len(sections)
            widget.mark_set(f"sec{n}", "end-1c")
            widget.mark_gravity(f"sec{n}", "left")
            sections.append((raw[3:], f"sec{n}"))
            widget.insert("end", raw[3:] + chr(10), ("h2",))
        elif raw.startswith("### "):
            widget.insert("end", raw[4:] + chr(10), ("h3",))
        elif raw.startswith("> "):
            inline("  " + raw[2:] + chr(10), ("note",))
        elif re.match(r"\s*[-*]\s", raw):
            inline("\u2022  " + re.sub(r"^\s*[-*]\s+", "", raw) + chr(10), ("bullet",))
        elif re.match(r"\s*\d+\.\s", raw):
            inline(raw.strip() + chr(10), ("num",))
        elif raw.strip():
            inline(raw + chr(10))
        else:
            widget.insert("end", chr(10))
    flush_table()
    widget.configure(state="disabled")
    return sections


HELP = """How the app works

An EXPERIMENT (left list) is one file (experiments/<name>.yaml) with everything: its simulation written in
full (or an already simulated folder), the stages it uses and their settings. Training and validation are just
two flows: an experiment can stop after the simulation, after the labelled dataset, after the indicators, or
point to a REFERENCE experiment (whose labelled dataset trains the indicators) and validate against it.

The DIAGRAM shows its stages. Colour = status:
  green done · light green not needed (a later stage already has its result) · orange stale (its configuration
  or an input changed after it ran) · blue running · red failed · grey pending (ready) · dashed blocked.
Hover a box to see why it is in that state; click it to open its panel below.

GOAL (top): what you want to obtain (it starts at the furthest stage turned on). Dark outline = stages it needs;
blue outline = NEXT STEP. "Select that stage" shows it, "Run next step" starts it. "Dry-run" lists, without
running anything, the cases of every run, the folders, the commands and every problem.

STAGE PANEL: what the stage does, what its output contains, what to check, its inputs and outputs (what each file
is), the channel it uses, its last run and, if it cannot run, why and what to do.
Buttons: Run in console · Copy command · Log (everything the stage printed, kept after the console closes) ·
Viewer · Edit config (the settings of THIS stage; Simulate/Extract = the simulation) · Labels YAML · Folder ·
Go to blocker · Mark up to date (a stale stage whose configuration change does not alter its result).
"close the console when it ends": the console closes by itself (the log stays).

LEFT: New experiment (from scratch; 'load values from' only pre-fills) · Import folder (already simulated) ·
Standardize an .h5 (made outside the app: adds the attributes it needs and creates its experiment) · Copy ·
Experiment settings (description, stages, reference, output folder) · Delete (the file and its run records;
data are never deleted) · Edit YAML (advanced: everything the forms do is in that file).

RAMPS OF Ap: a case whose depth changes along the cut ('depths end' in the simulation form, or 'add: ramps' on the
SLD picker). Its kappa is kappa_start -> kappa_end; its ground truth is the amplitude rule window by window (the
window of the indicators, Edit config of the labelling) and t_onset is where it turns unstable. Validate scores
the ramps that cross apart from the global metrics (ramp_* columns), with the same rule as every unstable case:
any alarm on an unstable constant case = TP; on a ramp that crosses, an alarm before t_onset = false alarm, after = TP.

COMPARE tab: metrics of two validations side by side. TOOLS tab: every script on its own.
TUTORIAL tab: step-by-step guide (the same text as DOE_utils/TUTORIAL.md).
Console: python experiment.py status | check EXP | dryrun EXP | accept EXP | run EXP STAGE | import.
"""


# ============================================================================== process helpers
def python_exe() -> str:
    return ENV_PYTHON if os.path.isfile(ENV_PYTHON) else sys.executable


def split_args(args) -> list:
    """Typed arguments -> list. A list passes as is; text is split like a Windows command line and the quotes
    around a token are removed (shlex posix=False keeps them, and then a quoted path is not found)."""
    if isinstance(args, (list, tuple)):
        return [str(a) for a in args]
    toks = shlex.split(args, posix=False) if args.strip() else []
    return [t[1:-1] if len(t) > 1 and t[0] == t[-1] and t[0] in "\"'" else t for t in toks]


def build_cmd(kind: str, script: str, args) -> list:
    """Command list for one tool (args: list, or text typed in the Tools tab). cli: cmd /k keeps the console
    open; no args -> --help."""
    path = os.path.join(HERE, script)
    a = split_args(args)
    if kind == "cli":
        return ["cmd", "/k", python_exe(), path] + (a or ["--help"])
    return [python_exe(), path] + a


def launch(kind: str, script: str, args):
    """Start a tool (args: list of arguments, or text). Returns (Popen, stderr_log_path or None)."""
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


def open_console(argv: list, cwd: str = HERE, close: bool = False):
    """New console running argv; close=True closes it when the command ends (its log stays in .runs/)."""
    return subprocess.Popen(["cmd", "/c" if close else "/k", *argv], cwd=cwd,
                            creationflags=subprocess.CREATE_NEW_CONSOLE)


def _settings_path() -> str:
    return os.path.join(ex.RUNS_DIR, "app_settings.json")


def load_settings() -> dict:
    """Small per-machine preferences of the window (e.g. close the console when a stage ends)."""
    try:
        with open(_settings_path(), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_settings(**kw) -> None:
    d = dict(load_settings(), **kw)
    os.makedirs(os.path.dirname(_settings_path()), exist_ok=True)
    with open(_settings_path(), "w", encoding="utf-8") as f:
        json.dump(d, f)


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
        for c, h, w in (("#0", "name", 200), ("kind", "flow", 70), ("n", "n [rpm]", 64), ("cases", "cases", 58),
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
        self.left_btns = {}
        for i, (txt, fn) in enumerate((("New experiment…", self.new_experiment), ("Import folder…", self.import_folder),
                                       ("Standardize an .h5…", self.standardize), ("Copy…", self.copy_exp),
                                       ("Experiment settings…", self.settings), ("Delete", self.delete),
                                       ("Edit YAML (advanced)", self.edit_yaml), ("Refresh", lambda: self.refresh(True)))):
            b = ttk.Button(bf, text=txt, command=fn)
            b.grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)
            self.left_btns[txt] = b
        ttk.Label(left, foreground="#666", font=("Segoe UI", 8), wraplength=380, justify="left",
                  text="An experiment = one file with everything: its simulation, the stages it uses and their "
                       "settings. 'Experiment settings' = description, stages, reference, output folder; 'Edit "
                       "config' (stage panel) = the settings of one stage.").pack(anchor="w", pady=(4, 0))
        bf.columnconfigure(0, weight=1)
        bf.columnconfigure(1, weight=1)

        card = ttk.Frame(right)
        card.pack(fill=tk.X, pady=(0, 4))
        self.card_title, self.card_line, self.card_link = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.card_src = tk.StringVar()
        ttk.Label(card, textvariable=self.card_title, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(card, textvariable=self.card_line, foreground="#333").pack(anchor="w")
        ttk.Label(card, textvariable=self.desc, foreground="#555", wraplength=900, justify="left").pack(anchor="w")
        ttk.Label(card, textvariable=self.card_src, foreground="#37474f", wraplength=1100, justify="left",
                  font=("Segoe UI", 8)).pack(anchor="w")
        ttk.Label(card, textvariable=self.card_link, foreground="#1565c0").pack(anchor="w")
        top = ttk.Frame(right)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Goal:").pack(side=tk.LEFT)
        cb = ttk.Combobox(top, textvariable=self.goal, values=list(ex.GOALS), state="readonly", width=22)
        cb.pack(side=tk.LEFT, padx=4)
        cb.bind("<<ComboboxSelected>>", lambda _e: self.refresh(True))
        self.goal_cb = cb
        self.next_lbl = tk.StringVar()
        ttk.Label(top, textvariable=self.next_lbl, font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=10)
        self.btn_next = ttk.Button(top, text="Select that stage", command=self.go_next)
        self.btn_next.pack(side=tk.LEFT)
        self.btn_run_next = ttk.Button(top, text="▶ Run next step", command=self.run_next)
        self.btn_run_next.pack(side=tk.LEFT, padx=4)
        self.btn_chain = ttk.Button(top, text="▶▶ Run to goal", command=self.run_to_goal)
        self.btn_chain.pack(side=tk.LEFT)
        self.review = tk.BooleanVar(value=load_settings().get("review_labels", True))
        ttk.Checkbutton(top, text="stop after Label template (review)", variable=self.review,
                        command=lambda: save_settings(review_labels=self.review.get())).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Tutorial", command=self.show_tutorial).pack(side=tk.RIGHT)
        ttk.Button(top, text="?  Help", command=self.show_help).pack(side=tk.RIGHT, padx=4)
        ttk.Button(top, text="Dry-run (check all)", command=self.dry_run).pack(side=tk.RIGHT, padx=4)

        self.canvas = tk.Canvas(right, height=3 * STEP_Y + 2 * MARGIN - (STEP_Y - BOX_H), bg="white",
                                highlightthickness=1, highlightbackground="#ccc")
        self.canvas.pack(fill=tk.X, pady=6)
        self.canvas.bind("<Configure>", lambda _e: self.redraw(True) if self.sel_exp else None)
        legend = ttk.Frame(right)
        legend.pack(fill=tk.X)
        for st, fill in STATE_FILL.items():
            tk.Label(legend, text=f" {STATE_TEXT[st]} ", bg=fill, relief="solid", bd=1, font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=2)
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
                             ("grid", "Label grid", self.open_grid),
                             ("folder", "Folder", self.open_folder),
                             ("goto", "Go to blocker", self.goto_blocker),
                             ("accept", "Mark up to date", self.mark_uptodate)):
            b = ttk.Button(bar, text=txt, command=fn)
            b.pack(side=tk.LEFT, padx=2)
            self.stage_btns[key] = b
        low = ttk.Frame(panel)
        low.pack(fill=tk.X)
        self.close_console = tk.BooleanVar(value=load_settings().get("close_console", True))
        ttk.Checkbutton(low, text="close the console when the stage ends OK (stays open on error; log: 'Log')",
                        variable=self.close_console,
                        command=lambda: save_settings(close_console=self.close_console.get())).pack(side=tk.RIGHT)
        self.status_msg = tk.StringVar()
        ttk.Label(low, textvariable=self.status_msg, foreground="#1565c0").pack(side=tk.LEFT)

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
        ttk.Button(row, text="Plot", command=self.plot_compare).pack(side=tk.LEFT)
        self.cmp_note = tk.StringVar(value="Validation experiments with doe_validation_results.h5   (ramp* columns: provisional criterion)")
        ttk.Label(f, textvariable=self.cmp_note, foreground="#555").pack(anchor="w", pady=4)
        cols = ("variant",) + tuple(f"{s}:{m}" for m in ex.METRIC_COLUMNS for s in ("A", "B"))
        self.cmp_tree = ttk.Treeview(f, columns=cols, show="headings")
        for c in cols:
            self.cmp_tree.heading(c, text=c.replace("balanced_accuracy", "bal.acc").replace("mean_", "")
                                  .replace("median_delay_onset_s", "delay_onset").replace("_stable", "")
                                  .replace("ramp_median_delay_s", "ramp delay").replace("_rate", "")
                                  .replace("early_alarm", "early").replace("gray_as_stable_", "gray=st ")
                                  .replace("gray_as_unstable_", "gray=unst ")
                                  .replace("detection", "det.").replace("ramp_", "ramp* "))
            self.cmp_tree.column(c, width=230 if c == "variant" else 62, anchor="w" if c == "variant" else "center")
        xs = ttk.Scrollbar(f, orient=tk.HORIZONTAL, command=self.cmp_tree.xview)
        self.cmp_tree.configure(xscrollcommand=xs.set)
        self.cmp_tree.pack(fill=tk.BOTH, expand=True)
        xs.pack(fill=tk.X)

    def _build_tutorial(self):
        tk, ttk = self.tk, self.ttk
        f = ttk.Frame(self.tab_tut, padding=6)
        f.pack(fill=tk.BOTH, expand=True)
        self.tut_size = 10
        bar = ttk.Frame(f)
        bar.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(bar, text="Tutorial", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        ttk.Label(bar, text=f"  {os.path.basename(TUTORIAL_FILE)}", foreground="#777").pack(side=tk.LEFT)
        ttk.Button(bar, text="Open file", command=lambda: os.startfile(TUTORIAL_FILE)).pack(side=tk.RIGHT)
        ttk.Button(bar, text="Reload", command=self._render_tutorial).pack(side=tk.RIGHT, padx=4)
        ttk.Button(bar, text="A+", width=4, command=lambda: self._tut_zoom(1)).pack(side=tk.RIGHT)
        ttk.Button(bar, text="A\u2212", width=4, command=lambda: self._tut_zoom(-1)).pack(side=tk.RIGHT, padx=2)
        body = ttk.Frame(f)
        body.pack(fill=tk.BOTH, expand=True)
        side = ttk.Frame(body, width=270)
        side.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        side.pack_propagate(False)
        ttk.Label(side, text="Contents", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.tut_toc = tk.Listbox(side, activestyle="none", exportselection=False, font=("Segoe UI", 9),
                                  borderwidth=0, highlightthickness=1, highlightbackground="#ccc",
                                  selectbackground="#e3f2fd", selectforeground="#0d47a1")
        self.tut_toc.pack(fill=tk.BOTH, expand=True, pady=4)
        self.tut_toc.bind("<<ListboxSelect>>", self._tut_goto)
        box = ttk.Frame(body)
        box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        t = tk.Text(box, wrap="word", padx=24, pady=10, state="disabled", borderwidth=0, cursor="arrow")
        sb = ttk.Scrollbar(box, orient=tk.VERTICAL, command=t.yview)
        t.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        t.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        t.bind("<Configure>", lambda e: t.configure(padx=max(20, (e.width - TUT_WIDTH) // 2)))   # centred column
        self.tut_text, self._tut_marks = t, []
        self._render_tutorial()

    def _render_tutorial(self):
        sections = render_markdown(self.tut_text, read_tutorial(), self.tut_size)
        self._tut_marks = [m for _, m in sections]
        self.tut_toc.delete(0, "end")
        for title, _ in sections:
            self.tut_toc.insert("end", title)

    def _tut_zoom(self, d):
        self.tut_size = max(8, min(16, self.tut_size + d))
        top = self.tut_text.yview()[0]
        self._render_tutorial()
        self.tut_text.yview_moveto(top)

    def _tut_images_ok(self):
        return list(getattr(self.tut_text, "_images", []))

    def _tut_goto(self, _e=None):
        sel = self.tut_toc.curselection()
        if sel:
            self.tut_text.see("end")   # scroll so that the section lands at the top, not at the bottom
            self.tut_text.yview(self._tut_marks[sel[0]])

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
                flow = {"Simulation only": "simulation", "Labelled dataset (training)": "dataset",
                        "Indicators": "indicators", "Validation against a reference": "validation"}.get(e.flow, "custom")
                rows[n] = (flow, f"{float(sm['spin']):.6g}" if sm["spin"] is not None else "?",
                           f"{sm['cases']}" + (f" ({sm['ramps']}r)" if sm.get("ramps") else ""),
                           f"{sm['done']}/{sm['total']}")
                rows[n + "\0val"] = "validate" in ex.stages(e)
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
        vals = sorted(n for n in names if rows.get(n + "\0val"))
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
            self._set_info([("No experiment. Use New experiment…, Import folder… or Standardize an .h5…", None)])
            return
        if self.sel_exp in getattr(self, "_row_errors", {}):
            self.canvas.delete("all")
            self._set_info([(f"{self.sel_exp}: cannot load\n{self._row_errors[self.sel_exp]}", "bad")])
            return
        e = self.exp()
        self._fill_card(e)
        gl = ex.goals(e)
        self.goal_cb["values"] = gl
        if self._goal_for != e.name or self.goal.get() not in gl:   # default goal = furthest stage turned on
            self.goal.set(ex.default_goal(e))
            self._goal_for = e.name
        st = ex.status(e)
        nxt = ex.next_step(e, self.goal.get())
        chain = {(x.name, k) for x, k in ex.goal_chain(e, self.goal.get())} if self.goal.get() in ex.GOALS else set()
        # progress and the log of the selected stage are part of the snapshot: the panel follows a running stage
        rec = ex.read_record(e, self.sel_stage) if self.sel_stage else None
        live = (ex.stage_progress(e, self.sel_stage) if self.sel_stage else None,
                ex._mtime(rec["log"]) if rec and rec.get("log") else 0)
        snapshot = (e.name, json.dumps(st, sort_keys=True), self.goal.get(), self.sel_stage, str(nxt and nxt[:2]),
                    str(live))
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
        self._enable(self.btn_chain, bool(nxt and nxt[2] in ("pending", "stale", "failed")
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
        ncols = 1 + max(diagram_layout(keys, e.ref is not None, 1.0)[k][0] - MARGIN for k in keys)
        avail = max(c.winfo_width(), 600)
        step_x = max(110.0, min(STEP_X, (avail - 2 * MARGIN) / max(ncols, 1)))   # fit the panel width
        box_w = step_x - 26
        pos = diagram_layout(keys, e.ref is not None, step_x)
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
            tst = ex.status(e.ref).get("label_build", ("blocked", ""))[0]
            c.create_rectangle(x, y, x + box_w, y + BOX_H, fill=STATE_FILL[tst], outline="#78909c", dash=(4, 3),
                               tags=("ext",))
            c.create_text(x + box_w / 2, y + BOX_H / 2, text=f"reference dataset\n{e.ref.name[:22]}\n({tst})",
                          font=("Segoe UI", 7), justify="center", tags=("ext",))
            c.tag_bind("ext", "<Button-1>", lambda _e: self.select(e.ref.name, "label_build"))
            c.tag_bind("ext", "<Enter>", lambda _e: self.hover.set(
                f"Labelled dataset of '{e.ref.name}': the indicators learn from it and this experiment's labels "
                "use its parameters. Click to open it."))
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
        bits = [f"flow: {e.flow}"]
        if sm["spin"] is not None:
            bits.append(f"n = {float(sm['spin']):.6g} rpm")
        bits.append(f"{sm['cases']} case" + ("s" if sm["cases"] != 1 else "")
                    + (f" (ramps {sm['ramps']} of {sm['cases']})" if sm.get("ramps") else ""))
        if sm["kappa"]:
            bits.append(f"kappa {sm['kappa'][0]:g}" + (f"-{sm['kappa'][1]:g}" if sm["kappa"][1] != sm["kappa"][0] else ""))
        bits.append(f"{sm['done']}/{sm['total']} stages done")
        self.card_line.set("  ·  ".join(bits))
        self.desc.set(e.cfg.get("description", ""))
        # where everything comes from and goes to (which config, which data, which outputs)
        src = [f"run {i + 1}: {r.source}  →  data {r.doe_dir}   ·  model: {r.model or '? (Standardize an .h5…)'}"
               for i, r in enumerate(e.runs)]
        src.append(f"outputs of this experiment: {e.out_dir}" + ("" if e.cfg.get("out_dir") else "  (default)"))
        src.append(f"file: {e.path}")
        self.card_src.set("\n".join(src))
        users = ex.dependents(e.name)
        link = [f"reference: {e.ref.name} (its labelled dataset trains the indicators)"] if e.ref is not None else []
        if users:
            link.append(f"reference of: {', '.join(users)}")
        self.card_link.set("   ·   ".join(link))

    @staticmethod
    def _hint(reason: str) -> str:
        """What to do about a reason why a stage cannot run."""
        if "of experiment" in reason:
            return "→ 'Go to blocker' and finish that stage there"
        if "'label_build'" in reason and reason.startswith("needs"):
            return ("→ the indicators learn their thresholds from a labelled dataset: run Label template + Label "
                    "build first (or set a reference experiment in 'Experiment settings…')")
        if reason.startswith(("needs", "waits for")):
            return "→ run that earlier stage first (click its box)" if "needs" in reason else "→ wait until it finishes"
        if "imported run" in reason:
            return "→ data imported from a folder: use 'New experiment…' (or Copy + Edit config) to simulate again"
        if reason.startswith("configuration error"):
            return "→ fix it with 'Edit config' of the stage it names, or 'Experiment settings…' (left)"
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
        tag = {"done": "ok", "skipped": "ok", "failed": "bad", "stale": "warn", "blocked": "warn",
               "running": "run"}.get(state)
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
        # inputs / outputs: what each file is, whether it exists, where it is
        parts.append(("\nInputs and outputs\n", "head"))
        for title, paths, roles in (("in ", s.inputs, s.roles[0]), ("out", s.outputs, s.roles[1])):
            for i, pth in enumerate(paths):
                if not pth:
                    continue
                ok = os.path.exists(pth)
                mark, mtag = ("✓", "ok") if ok else (("✗", "bad") if title == "in " else ("·", None))
                role = roles[i] if i < len(roles) and roles[i] else ""
                parts += [(f"  {title} {mark} ", mtag), (f"{role}: " if role else "", "what"),
                          (f"{os.path.basename(pth)}   ", None), (f"{os.path.dirname(pth)}\n", "hint")]
        parts += [(f"  {t}\n", "hint") for t in self._channels(e, k)]
        if k == "simulate" and any(not r.imported for r in e.runs):
            o = e.section("simulate")
            parts.append((f"  run options: timed {'on' if o.get('timed') else 'off'} (cases run one by one in "
                          f"parallel, time per case saved) · auto-extract {'on' if o.get('auto_extract') else 'off'} "
                          f"(Extract runs right after) — change them in Edit config\n", "hint"))
        if s.cmds:
            parts.append(("\nCommand  ", "head"))
            parts.append((f"experiment.py run {e.name} {k}   (full command: 'Copy command')\n", "hint"))
        errs, warns = ex.check(e)
        if errs or warns:
            parts.append(("\nExperiment checks\n", "head"))
            parts += [(f"  ERROR {x}\n", "bad") for x in errs] + [(f"  warning {x}\n", "warn") for x in warns]
        self._set_info(parts)
        self._blocking = next(((de, dk) for de, dk in s.deps if de is not e
                               and ex.status(de).get(dk, ("missing",))[0] != "done"), None)
        btn = self.stage_btns
        self._enable(btn["run"], not blockers and bool(s.cmds))
        self._enable(btn["copy"], bool(s.cmds))
        self._enable(btn["log"], bool(rec and rec.get("log") and os.path.isfile(rec["log"])))
        self._enable(btn["view"], state != "running" and any(p.endswith(".h5") and os.path.isfile(p) for p in s.outputs))
        self._enable(btn["goto"], self._blocking is not None)
        self._enable(btn["labels"], k in ("label_template", "label_build") and os.path.isfile(e.label["labels_yaml"]))
        self._enable(btn["grid"], k == "label_build" and os.path.isfile(e.label["out"]))
        self._enable(btn["folder"], any(os.path.exists(os.path.dirname(p)) for p in s.outputs))
        self._enable(btn["accept"], state == "stale" and ("configuration changed" in reason
                                                           or "input changed after the run" in reason))

    @staticmethod
    def _channels(e, k) -> list:
        """Which signal channel the stage uses, and what that channel is."""
        d = lambda c: f"{c} = {ex.CHANNELS.get(c, 'signal of doe_results.h5')}"   # noqa: E731
        if k in ("label_template", "label_build"):
            out = [f"labelling channel: {d(e.label.get('amp_signal', 'Axial_disp'))}"]
            if k == "label_build":
                out.append("channels cut into the dataset: " + (", ".join(e.label["channels"]) if e.label.get("channels")
                                                                else "all signals of the case"))
            return out
        if k in ("indicators", "noise_indicators"):
            sig = {}
            for v, spec in e.indicators["specs"].items():
                sig.setdefault(spec["signal"], []).append(v)
            return [f"analysed channel: {d(c)}  ({', '.join(vs)})" for c, vs in sig.items()]
        if k == "validate":
            return [f"channel of the ground-truth labels: {d(e.section('validate').get('channel', 'Axial_disp'))}"]
        return []

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
        open_console(ex.run_command(e.name, k, py, yes=True) + (["--pause-on-error"] if self.close_console.get() else []),
                     close=self.close_console.get())
        self.status_msg.set(f"{ex.TITLES[k]} started in a new console"
                            + (" (it closes by itself at the end; the log stays: 'Log')" if self.close_console.get() else ""))
        self.root.after(1500, lambda: self.refresh(True))

    def run_to_goal(self):
        """One console runs everything the goal needs, one stage after the other: first the stages of the
        reference experiment (e.g. the training's simulation and labelled dataset), then this experiment's.
        It stops on an error and, if wanted, after each Label template (to review the labels)."""
        from tkinter import messagebox
        e, goal = self.exp(), self.goal.get()
        todo = ex.chain_stages(e, goal)
        if not todo:
            return
        names = [ex.TITLES[k] + ("" if n == e.name else f" ({n})") for n, k in todo]
        review = self.review.get()
        stop = ("\n\nIt STOPS after each Label template so you can review the labels (untick 'stop after Label "
                "template' to run straight through)." if review else "\n\nIt does not stop for the label review "
                "(tick 'stop after Label template' to stop).") if any(k == "label_template" for _, k in todo) else ""
        if not messagebox.askyesno("Run to goal", "Run in one console, one after the other:\n  "
                                   + "\n  ".join(f"{i}. {s}" for i, s in enumerate(names, 1)) + f"\n\nGoal: {goal}."
                                   + stop + "\nExisting outputs of these stages are replaced. Continue?"):
            return
        py, warn = stage_python()
        if warn:
            messagebox.showwarning("Python", warn)
        open_console(ex.chain_command(e.name, goal, py) + ([] if review else ["--no-review"])
                     + (["--pause-on-error"] if self.close_console.get() else []), close=self.close_console.get())
        self.status_msg.set(f"running to '{goal}' in a new console: " + " → ".join(names))
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
        t = tk.Text(w, width=104, height=34, wrap="word", font=("Segoe UI", 10))
        t.insert("1.0", HELP)
        t.configure(state="disabled")
        t.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def dry_run(self):
        if self.sel_exp:
            TextView(self, f"Dry-run — {self.sel_exp}", ex.dry_run(self.exp()))

    def copy_cmd(self):
        e, k = self.exp(), self.sel_stage
        self.root.clipboard_clear()
        self.root.clipboard_append(quote(ex.run_command(e.name, k, stage_python()[0], yes=False)))
        self.status_msg.set("command copied to the clipboard")

    def open_log(self):
        rec = ex.read_record(self.exp(), self.sel_stage)
        if rec and rec.get("log"):
            LogViewer(self, rec["log"], f"{self.sel_exp} — {ex.TITLES[self.sel_stage]}")

    def open_output(self):
        s = ex.stages(self.exp())[self.sel_stage]
        h5 = next(p for p in s.outputs if p.endswith(".h5") and os.path.isfile(p))
        self.view_h5(h5)

    def view_h5(self, h5: str):
        """Open the viewer on an .h5; says that it is opening (loading takes a few seconds) and shows its error
        if it fails to start."""
        try:
            p, log = launch("gui", "DOE_plots/doe_unified_selector.py", ["--h5", h5])
        except OSError as exc:
            self._msg("Viewer", str(exc), "error")
            return
        self.status_msg.set(f"Opening the viewer on {os.path.basename(h5)}… (loading the file takes a few seconds)")
        self._watch(p, log, "doe_unified_selector.py")

    def open_grid(self):
        """The labelled dataset of the selected experiment as a grid of cases (label_grid.py, read-only)."""
        h5 = self.exp().label["out"]
        try:
            p, log = launch("gui", "label_grid.py", ["--h5", h5])
        except OSError as exc:
            self._msg("Label grid", str(exc), "error")
            return
        self.status_msg.set(f"Opening the label grid on {os.path.basename(h5)}…")
        self._watch(p, log, "label_grid.py")

    def _watch(self, p, log, script, tries=0):
        """Error of a tool started with launch(): shown if it exits with an error within ~15 s."""
        if p is None:
            return
        rc = p.poll()
        if rc is None:
            if tries < 30:
                self.root.after(500, self._watch, p, log, script, tries + 1)
            else:
                self.status_msg.set("")
            return
        self.status_msg.set("")
        if rc != 0 and log:
            with open(log, encoding="utf-8", errors="replace") as f:
                err = f.read().strip().splitlines()[-12:]
            self._msg(os.path.basename(script), "\n".join(err) or f"exit code {rc}", "error")

    def open_folder(self):
        s = ex.stages(self.exp())[self.sel_stage]
        d = next(os.path.dirname(p) for p in s.outputs if os.path.exists(os.path.dirname(p)))
        os.startfile(d)

    def goto_blocker(self):
        if self._blocking:
            self.select(self._blocking[0].name, self._blocking[1])

    def mark_uptodate(self):
        from tkinter import messagebox
        e, k = self.exp(), self.sel_stage
        if messagebox.askyesno(
                "Mark up to date",
                f"'{ex.TITLES[k]}' is stale: {ex.status(e)[k][1]}.\n\n"
                "Mark it up to date only if that change does not alter its result (e.g. the same folder written "
                "another way, the number of parallel processes, attributes added to an input). Otherwise run it "
                "again.\n\nMark it up to date?"):
            ex.accept(e, [k])
            self.refresh(True)

    def edit_stage(self):
        e, k = self.exp(), self.sel_stage
        if k in ("simulate", "extract"):
            SimulationForm(self, e)
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

    def new_experiment(self):
        NewExperimentDialog(self)

    def import_folder(self):
        ImportDialog(self)

    def standardize(self):
        StandardizeDialog(self)

    def settings(self):
        if self.sel_exp:
            SettingsDialog(self, self.exp())

    def copy_exp(self):
        if self.sel_exp:
            CopyDialog(self, self.exp())

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
        self.cmp_note.set(f"A = {a}   B = {b}   (empty = that variant was not run in that experiment; ramp columns: "
                          "the ramps whose truth crosses, scored apart from the global metrics)")

    def plot_compare(self):
        """A/B figure of the two validations (validation_figures.fig_compare) in the export window
        (figures_window.py): language, size, dpi, format; saved in <folder of A's results>/figs_validation/."""
        a, b = self.cmp_a.get(), self.cmp_b.get()
        if not a or not b:
            return
        pa, pb = (ex.load(n).out("validate", "out", "doe_validation_results.h5") for n in (a, b))
        missing = [p for p in (pa, pb) if not os.path.isfile(p)]
        if missing:
            self._msg("Compare", "No validation results yet:\n" + "\n".join(missing), "warn")
            return
        import matplotlib
        matplotlib.use("TkAgg")
        if ex.PLOTS not in sys.path:
            sys.path.insert(0, ex.PLOTS)
        import validation_figures as vf
        from figures_window import FiguresWindow, Item

        def style(language, scale):
            vf.LANGUAGE, vf.FIGSCALE = language, scale
        self._cmp_win = FiguresWindow(
            self.root, f"Validation compare — A = {a}   B = {b}",
            [Item(f"compare_{a}_vs_{b}", lambda: vf.fig_compare(pa, pb), native=True)], style=style,
            out_dir=os.path.join(os.path.dirname(pa), "figs_validation"), language=vf.LANGUAGE, scale=vf.FIGSCALE)
        self.cmp_note.set(f"A/B figure: Save in its window ({os.path.join(os.path.dirname(pa), 'figs_validation')})")


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

    def field(self, label, var=None, width=40, values=None, state="normal", note="", editable=False):
        """One row: label, entry (or combobox with `values`; editable=True lets the user type too), note."""
        tk, ttk = self.tk, self.ttk
        var = var if var is not None else tk.StringVar()
        self.__dict__.setdefault("_keep", []).append(var)   # Tk forgets a variable nobody references (empty field)
        ttk.Label(self.body, text=label).grid(row=self.row, column=0, sticky="w", pady=2)
        if values is not None:
            w = ttk.Combobox(self.body, textvariable=var, values=values, width=width - 2,
                             state=("normal" if editable else "readonly") if state == "normal" else state)
        else:
            w = ttk.Entry(self.body, textvariable=var, width=width, state=state)
        w.grid(row=self.row, column=1, sticky="w", pady=2)
        self.last = w
        if note:
            ttk.Label(self.body, text=note, foreground="#666").grid(row=self.row, column=2, sticky="w", padx=6)
        self.row += 1
        return var

    def browse(self, label, var, kind="dir", note=""):
        """Entry + 'Browse…' (folder or file)."""
        from tkinter import filedialog
        self.field(label, var, width=70)
        r = self.row - 1

        def pick():
            p = (filedialog.askdirectory(parent=self.win, initialdir=var.get() or None) if kind == "dir" else
                 filedialog.askopenfilename(parent=self.win, initialdir=os.path.dirname(var.get()) or None))
            if p:
                var.set(p)
        f = self.ttk.Frame(self.body)
        f.grid(row=r, column=2, sticky="w", padx=6)
        self.ttk.Button(f, text="Browse…", command=pick).pack(side="left")
        if note:
            self.ttk.Label(f, text=note, foreground="#666").pack(side="left", padx=6)
        return var

    def note(self, text, color="#555"):
        self.ttk.Label(self.body, text=text, foreground=color, wraplength=900, justify="left").grid(
            row=self.row, column=0, columnspan=3, sticky="w", pady=(6, 2))
        self.row += 1

    def section(self, text):
        self.ttk.Label(self.body, text=text, font=("Segoe UI", 9, "bold")).grid(
            row=self.row, column=0, columnspan=3, sticky="w", pady=(10, 2))
        self.row += 1

    def buttons(self, ok_text="Save"):
        ttk = self.ttk
        bf = ttk.Frame(self.win, padding=(10, 0, 10, 10))
        bf.pack(fill="x")
        ttk.Button(bf, text=ok_text, command=self._ok).pack(side="right")
        ttk.Button(bf, text="Cancel", command=self.win.destroy).pack(side="right", padx=6)
        self.bar = bf

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


def _fmt_list(v) -> str:
    v = v if isinstance(v, (list, tuple)) else [v]
    return ", ".join(f"{float(x):.10g}" for x in v)


class TextView:
    """Read-only window with coloured lines: [(text, tag)] (the dry-run of an experiment)."""

    def __init__(self, app, title, parts):
        tk, ttk = app.tk, app.ttk
        w = tk.Toplevel(app.root)
        w.title(title)
        t = tk.Text(w, width=130, height=40, wrap="none", font=("Consolas", 9))
        sb = ttk.Scrollbar(w, orient=tk.VERTICAL, command=t.yview)
        t.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        t.pack(fill=tk.BOTH, expand=True)
        for tag, col in (("ok", "#2e7d32"), ("bad", "#c62828"), ("warn", "#b26a00"), ("hint", "#546e7a")):
            t.tag_configure(tag, foreground=col)
        t.tag_configure("head", font=("Consolas", 9, "bold"), spacing1=6)
        for text, tag in parts:
            t.insert("end", text + "\n", tag) if tag else t.insert("end", text + "\n")
        t.configure(state="disabled")
        self.win, self.text = w, t


class LogViewer:
    """The log of a stage = everything its console printed, kept in .runs/<experiment>/<stage>.log after the
    console closes. Coloured, searchable, follows the file while the stage runs."""
    RULES = (("bad", r"ERROR|Traceback|FAILED|Error:|exit code [1-9]|MAL:"), ("warn", r"WARNING|WARN|Advertencia|warning"),
             ("run", r"\[\d+/\d+\] completado|\[DRY-RUN\]"), ("head", r"^-- case|^\$ |^\[experiment\]"))

    def __init__(self, app, path, title):
        tk, ttk = app.tk, app.ttk
        self.app, self.path = app, path
        w = tk.Toplevel(app.root)
        w.title(f"Log — {title}")
        bar = ttk.Frame(w, padding=4)
        bar.pack(fill=tk.X)
        ttk.Label(bar, text=os.path.basename(path), font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        ttk.Label(bar, text="  everything the stage printed; it stays after the console closes", foreground="#666"
                  ).pack(side=tk.LEFT)
        ttk.Button(bar, text="Open in editor", command=lambda: os.startfile(path)).pack(side=tk.RIGHT)
        self.follow = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="follow the end", variable=self.follow).pack(side=tk.RIGHT, padx=6)
        self.q = tk.StringVar()
        ttk.Button(bar, text="Find next", command=self.find).pack(side=tk.RIGHT)
        e = ttk.Entry(bar, textvariable=self.q, width=24)
        e.pack(side=tk.RIGHT, padx=4)
        e.bind("<Return>", lambda _e: self.find())
        ttk.Label(bar, text="search:").pack(side=tk.RIGHT)
        box = ttk.Frame(w)
        box.pack(fill=tk.BOTH, expand=True)
        t = tk.Text(box, width=140, height=42, wrap="none", font=("Consolas", 9))
        sb = ttk.Scrollbar(box, orient=tk.VERTICAL, command=t.yview)
        t.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        t.pack(fill=tk.BOTH, expand=True)
        for tag, col in (("bad", "#c62828"), ("warn", "#b26a00"), ("run", "#1565c0"), ("head", "#263238")):
            t.tag_configure(tag, foreground=col)
        t.tag_configure("head", font=("Consolas", 9, "bold"))
        t.tag_configure("hit", background="#fff59d")
        self.win, self.text, self._mt, self._pos = w, t, None, "1.0"
        self.load()
        w.after(2000, self._tick)

    def load(self):
        import re
        mt = ex._mtime(self.path)
        if mt == self._mt:
            return
        self._mt = mt
        try:
            with open(self.path, encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
        except OSError as exc:
            lines = [f"cannot read {self.path}: {exc}"]
        t = self.text
        t.configure(state="normal")
        t.delete("1.0", "end")
        for line in lines:
            tag = next((tg for tg, rx in self.RULES if re.search(rx, line)), None)
            t.insert("end", line + "\n", tag) if tag else t.insert("end", line + "\n")
        t.configure(state="disabled")
        if self.follow.get():
            t.see("end")

    def _tick(self):
        if self.win.winfo_exists():
            self.load()
            self.win.after(2000, self._tick)

    def find(self):
        q = self.q.get()
        if not q:
            return
        t = self.text
        t.tag_remove("hit", "1.0", "end")
        pos = t.search(q, self._pos, nocase=True, stopindex="end") or t.search(q, "1.0", nocase=True, stopindex="end")
        if pos:
            end = f"{pos}+{len(q)}c"
            t.tag_add("hit", pos, end)
            t.see(pos)
            self.follow.set(False)
            self._pos = end


class LabelForm(_Dialog):
    FIELDS = ("strategy", "amp_signal", "base_attr", "base_scale", "lim_inf_pct", "lim_sup_pct", "warmup",
              "kappa_threshold", "t_start", "t_end", "window_mode", "window_N", "window_step", "f_modal")
    TEXT = ("strategy", "amp_signal", "base_attr", "window_mode")

    def __init__(self, app, e):
        super().__init__(app, f"Labelling — {e.name}")
        self.e = e
        locked = e.ref is not None
        if locked:
            self.note(f"Inherited from the reference '{e.ref.name}': the ground truth must be labelled exactly like "
                      "the dataset the indicators learn from (read-only here; edit it in the reference).", "#b26a00")
        self.vars = {}
        for k in self.FIELDS:
            v = self.tk.StringVar(value="" if e.label.get(k) is None else str(e.label.get(k)))
            vals = ["amplitude", "kappa", "manual"] if k == "strategy" else (
                ["Axial_disp", "Axial_vel", "Axial_acc", "Axial_disp_out_deflex"] if k == "amp_signal" else (
                    ["by_revolution", "by_modal"] if k == "window_mode" else None))
            note = {"amp_signal": "channel the labels are computed from",
                    "base_attr": "attribute the limits are a % of (feed per tooth)",
                    "lim_inf_pct": "max|signal| below this % of the base → stable",
                    "lim_sup_pct": "above this % → unstable (in between: gray)",
                    "kappa_threshold": "strategy kappa: stable below it",
                    "t_start": "[s] start of the cut pieces (empty = auto)",
                    "window_mode": "RAMP cases only: the same rule window by window (T = 60/n of the case, or 1/f_modal)",
                    "window_N": "window length in T (as an indicator's window)" + (
                        f"; now from {e.label['window_from']}" if e.label.get("window_from") else ""),
                    "window_step": "step between windows in T",
                    "f_modal": "[Hz] only for window_mode by_modal"}.get(k, "")
            self.vars[k] = self.field(k, v, values=vals, state="disabled" if locked else "normal", note=note)
        self.note("Channels: " + "   ".join(f"{c} = {d}" for c, d in ex.CHANNELS.items() if c.startswith("Axial")))
        self.field("labels YAML", self.tk.StringVar(value=e.label["labels_yaml"]), width=90, state="readonly")
        self.field("dataset out", self.tk.StringVar(value=e.label["out"]), width=90, state="readonly")
        bf = self.ttk.Frame(self.body)
        bf.grid(row=self.row, column=0, columnspan=3, sticky="w", pady=4)
        self.ttk.Button(bf, text="Open labels YAML (review)",
                        command=lambda: os.path.isfile(e.label["labels_yaml"]) and os.startfile(e.label["labels_yaml"])
                        ).pack(side="left")
        self.buttons("Close" if locked else "Save")

    def save(self):
        if self.e.ref is not None:
            return True
        d = {}
        for k, v in self.vars.items():
            t = v.get().strip()
            if t == "":
                continue
            d[k] = t if k in self.TEXT else float(t)
        old = ex.own_yaml(self.e.name).get("label") or {}
        for k in ("out", "labels_yaml", "channels"):   # explicit paths (imported experiments) are kept
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
                             values=["Axial_disp", "Axial_vel", "Axial_acc"],
                             note="the channel of the ground-truth dataset whose labels score the detections")
        self.note("Ground truth = this experiment's labelled dataset (" + os.path.basename(e.label["out"]) + "), labelled "
                  f"with the '{e.label.get('strategy')}' strategy on {e.label.get('amp_signal', 'Axial_disp')}. "
                  "The indicators were trained on " + os.path.basename(e.reference) +
                  (f" (reference '{e.ref.name}')." if e.ref else " (this experiment's own: no reference)."))
        self.buttons()

    def save(self):
        sec = ex.own_yaml(self.e.name).get("validate") or {}
        sec["channel"] = self.ch.get()
        sec.pop("early_tol_s", None)   # the rule has no tolerance any more
        ex.save_section(self.e.name, "validate", sec)


def _var_keys(spec: dict) -> tuple:
    """(time base, window, step, aux) keys of a variant spec for its mode (rev / modal)."""
    u = "rev" if spec["mode"] == "by_revolution" else "modal"
    aux = {"RMS_CV": f"n_max_{u}", "SST_SVD": f"Ai_length_{u}"}.get(spec["indicator"])
    return ("T_rev" if u == "rev" else "T_modal"), f"N_{u}_window", f"step_{u}", aux


class IndicatorsForm(_Dialog):
    """The indicator variants of the experiment as a table (each row = one group in the results file). New rows
    start from a preset of experiments/indicator_variants.yaml; the experiment keeps its own copy."""
    COLS = (("name", 290), ("indicator", 150), ("signal", 80), ("mode", 95), ("window", 55), ("step", 45),
            ("aux", 45))

    def __init__(self, app, e):
        import copy
        tk, ttk = app.tk, app.ttk
        super().__init__(app, f"Indicators — {e.name}")
        self.e = e
        own = ex.own_yaml(e.name).get("indicators") or {}
        self.specs = copy.deepcopy(e.indicators["specs"])
        self.inherit = tk.BooleanVar(value=e.ref is not None and own.get("variants", "inherit") == "inherit")
        if e.ref is not None:
            ttk.Checkbutton(self.body, text=f"Same variants as the reference '{e.ref.name}'", variable=self.inherit,
                            command=self._toggle).grid(row=self.row, column=0, columnspan=3, sticky="w")
            self.row += 1
        box = ttk.Frame(self.body)
        box.grid(row=self.row, column=0, columnspan=3, sticky="nsew", pady=4)
        self.row += 1
        self.tree = ttk.Treeview(box, columns=[c for c, _ in self.COLS], show="headings", height=9, selectmode="browse")
        for c, w in self.COLS:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<Double-1>", lambda _e: self.edit())
        bar = ttk.Frame(self.body)
        bar.grid(row=self.row, column=0, columnspan=3, sticky="w")
        self.row += 1
        lib = ex.presets()
        self.preset = tk.StringVar(value=next(iter(lib), ""))
        ttk.Label(bar, text="add from preset:").pack(side="left")
        ttk.Combobox(bar, textvariable=self.preset, values=list(lib), state="readonly", width=40).pack(side="left", padx=4)
        self.btns = [ttk.Button(bar, text="Add", command=self.add), ttk.Button(bar, text="Edit row…", command=self.edit),
                     ttk.Button(bar, text="Remove", command=self.remove)]
        for b in self.btns:
            b.pack(side="left", padx=2)
        ttk.Button(bar, text="Show resolved config for a case…", command=self._resolved).pack(side="left", padx=8)
        self.note("window / step: in revolutions (by_revolution, T_rev = 60/n of each case) or modal periods "
                  "(by_modal, T_modal = 1/f_modal). aux = RMS-CV n_max / SST-SVD Ai_length. Double-click a row to edit "
                  "it (other parameters under 'Advanced').")
        self.f_modal = self.field("f_modal [Hz]", tk.StringVar(value=str(e.indicators.get("f_modal") or "")),
                                  note="only by_modal variants use it")
        cases = e.indicators.get("cases", "all")
        self.cases = self.field("cases", tk.StringVar(value=cases if isinstance(cases, str) else " ".join(cases)),
                                note="'all' or case_000 case_003 …")
        self.workers = self.field("workers", tk.StringVar(value=str(e.indicators.get("workers") or 6)),
                                  note="processes in parallel (each reads the reference: ~0.7 GB RAM)")
        self._fill()
        self._toggle()
        self.buttons()

    def _fill(self):
        self.tree.delete(*self.tree.get_children())
        for n, s in self.specs.items():
            tb, win, step, aux = _var_keys(s)
            pp = s["params_physical"]
            self.tree.insert("", "end", iid=n, values=(n, f"{s['indicator']}/{s.get('func', 'Default')}", s["signal"],
                                                       s["mode"], pp.get(win, ""), pp.get(step, ""),
                                                       pp.get(aux, "") if aux else ""))

    def _toggle(self):
        on = not self.inherit.get()
        for b in self.btns:
            b.state(["!disabled"] if on else ["disabled"])

    def selected(self):
        sel = self.tree.selection()
        return sel[0] if sel else None

    def add(self):
        import copy
        spec = copy.deepcopy(ex.presets()[self.preset.get()])
        name = ex.propose_variant_name(spec, self.specs)
        self.specs[name] = spec
        self._fill()
        self.tree.selection_set(name)

    def remove(self):
        n = self.selected()
        if n:
            del self.specs[n]
            self._fill()

    def edit(self):
        n = self.selected()
        if n and not self.inherit.get():
            RowEditor(self.app, self, n)

    def _resolved(self):
        info = ex.h5_info(self.e.data_h5)
        n = self.selected() or next(iter(self.e.indicators["specs"]), None)
        if not info:
            self.app._msg("No data", "The experiment data (doe_results.h5) does not exist yet.", "warn")
        elif n not in self.e.indicators["specs"]:
            self.app._msg("Not saved", "Save the table first: the resolved config is built from the saved experiment.", "warn")
        else:
            ResolvedView(self.app, self.e, n, info["cases"])

    def save(self):
        d = ex.own_yaml(self.e.name).get("indicators") or {}
        d["variants"] = "inherit" if self.inherit.get() else self.specs
        d["f_modal"] = _num(self.f_modal.get())
        c = self.cases.get().strip()
        d["cases"] = "all" if c in ("", "all") else c.replace(",", " ").split()
        d["workers"] = int(self.workers.get())
        ex.save_section(self.e.name, "indicators", {k: v for k, v in d.items() if v is not None})


class RowEditor(_Dialog):
    """One row of the indicator table: the usual fields, the rest of the parameters as YAML ('Advanced')."""

    def __init__(self, app, form, name):
        import yaml
        tk = app.tk
        super().__init__(app, f"Variant {name}")
        self.form, self.orig = form, name
        spec = form.specs[name]
        self.spec = spec
        tb, win, step, aux = _var_keys(spec)
        pp = spec["params_physical"]
        self.field("indicator", tk.StringVar(value=f"{spec['indicator']} / {spec.get('func', 'Default')}"),
                   state="readonly")
        self.signal = self.field("signal", tk.StringVar(value=spec["signal"]), values=["Axial_disp", "Axial_vel", "Axial_acc"],
                                 note="channel analysed (" + ex.CHANNELS.get(spec["signal"], "") + ")")
        self.mode = self.field("mode", tk.StringVar(value=spec["mode"]), values=["by_revolution", "by_modal"])
        self.win_n = self.field("window", tk.StringVar(value=str(pp.get(win, ""))), note="revolutions or modal periods")
        self.step = self.field("step", tk.StringVar(value=str(pp.get(step, ""))))
        self.aux = self.field("aux (n_max / Ai_length)", tk.StringVar(value=str(pp.get(aux, "")) if aux else ""),
                              state="normal" if aux else "disabled")
        self.name = self.field("name", tk.StringVar(value=name), width=50, note="group in the results file")
        self.ttk.Button(self.body, text="Propose name", command=lambda: self.name.set(
            ex.propose_variant_name(self._spec(), [n for n in form.specs if n != name]))).grid(row=self.row, column=1, sticky="w")
        self.row += 1
        self.section("Advanced parameters (YAML; 'T_rev', 'T_modal', '10*T_rev' are resolved per case)")
        rest = {k: v for k, v in pp.items() if k not in (tb, win, step, aux)}
        self.text = tk.Text(self.body, width=80, height=16, font=("Consolas", 9))
        self.text.grid(row=self.row, column=0, columnspan=3)
        self.text.insert("1.0", yaml.safe_dump(rest, sort_keys=False))
        self.row += 1
        self.buttons("Apply")

    def _spec(self) -> dict:
        import yaml
        s = {k: v for k, v in self.spec.items() if k != "params_physical"}
        s.update(signal=self.signal.get(), mode=self.mode.get())
        tb, win, step, aux = _var_keys(s)
        pp = {tb: tb, win: _num(self.win_n.get(), float), step: _num(self.step.get(), float)}
        if aux:
            pp[aux] = _num(self.aux.get(), float)
        rest = yaml.safe_load(self.text.get("1.0", "end")) or {}
        if not isinstance(rest, dict):
            raise ValueError("advanced parameters: a 'key: value' map")
        for k in ("T_rev", "T_modal", "N_rev_window", "N_modal_window", "step_rev", "step_modal", "n_max_rev",
                  "n_max_modal", "Ai_length_rev", "Ai_length_modal"):
            rest.pop(k, None)
        for k in (win, step) + ((aux,) if aux else ()):   # integers stay integers in the YAML
            if pp.get(k) is not None and float(pp[k]).is_integer():
                pp[k] = int(pp[k])
        s["params_physical"] = {**pp, **rest}
        return s

    def save(self):
        spec, name = self._spec(), self.name.get().strip()
        if name != self.orig and name in self.form.specs:
            raise ValueError(f"'{name}' is already a row of the table")
        specs = {}
        for n, s in self.form.specs.items():   # keep the order of the rows
            specs[name if n == self.orig else n] = spec if n == self.orig else s
        self.form.specs.clear()
        self.form.specs.update(specs)
        self.form._fill()
        self.win.destroy()
        return False   # nothing saved to disk yet: the table is saved by its own Save


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


# ------------------------------------------------------------------------------ simulation
KNOWN_VARS = (("$f_tooth$", "f_tooth", "feed per tooth (as in the case)"), ("$dxl_size$", "dxl_size [m]",
              "dexel size"), ("$nb_dt_rev$", "nb_dt_rev", "time steps per revolution"))


def _base_defaults() -> dict:
    """Values a new simulation starts with: configs/base.yaml + the discretisation of the current DOEs."""
    out = {"base_dir": "", "case": "1DOF_150Hz", "doe_name": "", "nb_proc": 1, "n2m_bat": "", "depths": [],
           "spins": [], "ap_ref": {"mode": "none"}, "extract_signals": ["Axial_disp", "Axial_vel", "Axial_acc"],
           "force_signal": "res_R_p", "variables": {"$f_tooth$": [0.05], "$dxl_size$": [0.0002], "$nb_dt_rev$": [200]}}
    # read as written: the validated load fails when base.yaml's base_dir folder no longer exists, and then every
    # default (n2m.bat included) silently disappeared
    b = ex.base_yaml()
    out.update({k: b[k] for k in ("base_dir", "case", "nb_proc", "n2m_bat", "extract_signals", "force_signal")
                if k in b})
    out["base_dir"] = str(out["base_dir"] or "").replace("\\", "/")
    out["n2m_bat"] = str(out["n2m_bat"] or "").replace("\\", "/")
    return out


class SimulationFrame:
    """The fields of one explicit simulation (what doe_runner needs, nothing inherited) inside a dialog, with a
    preview of the cases and of the problems (the dry-run of the simulation) before saving."""

    def __init__(self, dlg, values: dict, run_opts: dict | None = None):
        tk, ttk = dlg.tk, dlg.ttk
        self.dlg = dlg
        v = {**_base_defaults(), **(values or {})}
        dlg.section("Simulation (written in full inside the experiment: what you see is what runs)")
        self.base_dir = dlg.browse("base_dir (folder of the case)", tk.StringVar(value=v["base_dir"]))
        self.case = dlg.field("case", tk.StringVar(value=v["case"]), values=[], editable=True,
                              note="Nessy2m case folder inside base_dir (the model that is simulated)")
        self.case_cb = dlg.last
        self.base_dir.trace_add("write", lambda *_: self._cases())
        self._cases()
        self.doe_name = dlg.field("doe_name (data folder)", tk.StringVar(value=v["doe_name"]),
                                  note="results go to base_dir/doe_name (no prefix needed)")
        self.spins = dlg.field("n [rpm]", tk.StringVar(value=_fmt_list(v["spins"]) if v["spins"] else ""),
                               note="one value, a list '9000, 12000' or start:stop:step")
        self.unit = tk.StringVar(value="Ap [mm]")
        self.depths = dlg.field("depths", tk.StringVar(value=_fmt_list(v["depths"]) if v["depths"] else ""))
        f = ttk.Frame(dlg.body)
        f.grid(row=dlg.row - 1, column=2, sticky="w", padx=6)
        for u in ("Ap [mm]", "kappa"):
            ttk.Radiobutton(f, text=u, value=u, variable=self.unit).pack(side="left")
        ttk.Label(f, text="  list or start:stop:step; one value = a single case", foreground="#666").pack(side="left")
        ends = v.get("depths_end") or []
        self.depths_end = dlg.field("depths end (ramps)", tk.StringVar(value=_fmt_list(ends) if ends else ""),
                                    note="empty = constant Ap · else Ap at the END of the cut, one per depth (same "
                                         "unit): different = a ramp, equal = constant")
        a = v["ap_ref"] or {"mode": "none"}
        self.ap_mode = dlg.field("ap_ref (kappa = Ap / ap_ref)", tk.StringVar(value=a.get("mode", "none")),
                                 values=["none", "manual", "model", "model_at_spin"],
                                 note="manual: a depth · model: SLD minimum · model_at_spin: SLD limit at the n of each case")
        self.ap_manual = dlg.field("  ap_ref manual [mm]", tk.StringVar(
            value=f"{float(a['manual']) * 1e3:.10g}" if a.get("manual") else ""))
        self.ap_model = dlg.field("  SLD model", tk.StringVar(value=a.get("model", "")), values=ex.sld_models(),
                                  note="preset of DOE_plots/sld_model.py (also the simulated model's SLD)")
        var = dict(v["variables"])
        self.known = {}
        for key, label, note in KNOWN_VARS:
            self.known[key] = dlg.field(label, tk.StringVar(value=_fmt_list(var.pop(key)) if key in var else ""),
                                        note=note)
        self.other = dlg.field("other variables", tk.StringVar(
            value="; ".join(f"{k} = {_fmt_list(x)}" for k, x in var.items())), width=60,
            note="'$name$ = v1, v2; $other$ = v'")
        self.combine = tk.BooleanVar(value=False)
        ttk.Checkbutton(dlg.body, text="every combination of the lists (factorial); off = row by row (lists of the "
                        "same length, a single value repeats)", variable=self.combine).grid(
            row=dlg.row, column=0, columnspan=3, sticky="w")
        dlg.row += 1
        self.nb_proc = dlg.field("nb_proc", tk.StringVar(value=str(v["nb_proc"])), note="cases in parallel")
        self.signals = dlg.field("signals to extract", tk.StringVar(value=", ".join(v["extract_signals"])))
        self.force = dlg.field("force signal", tk.StringVar(value=v["force_signal"]))
        self.n2m = dlg.browse("n2m.bat", tk.StringVar(value=v.get("n2m_bat") or ""), kind="file")
        for w in dlg.body.grid_slaves(row=dlg.row - 1, column=1):
            w.configure(width=100)   # the whole path is visible (the Nessy2m folder is long)
        o = run_opts or {}
        self.timed, self.auto = tk.BooleanVar(value=bool(o.get("timed"))), tk.BooleanVar(value=bool(o.get("auto_extract")))
        rf = ttk.Frame(dlg.body)
        rf.grid(row=dlg.row, column=0, columnspan=3, sticky="w", pady=(4, 0))
        dlg.row += 1
        ttk.Label(rf, text="run options:").pack(side="left")
        ttk.Checkbutton(rf, text="--timed (cases one by one, nb_proc in parallel, time per case saved)",
                        variable=self.timed).pack(side="left", padx=6)
        ttk.Checkbutton(rf, text="--auto-extract (Extract right after)", variable=self.auto).pack(side="left")
        pf = ttk.Frame(dlg.body)
        pf.grid(row=dlg.row, column=0, columnspan=3, sticky="w", pady=(6, 0))
        dlg.row += 1
        ttk.Button(pf, text="Check / preview cases", command=self.preview).pack(side="left")
        ttk.Button(pf, text="Pick the depths on the SLD…", command=lambda: SldPicker(dlg.app, self)).pack(side="left", padx=6)
        ttk.Label(pf, text="  (also done when saving: nothing is saved while there is an error)", foreground="#666"
                  ).pack(side="left")
        self.out = tk.Text(dlg.body, width=120, height=11, font=("Consolas", 9))
        self.out.grid(row=dlg.row, column=0, columnspan=3, pady=4)
        dlg.row += 1
        for tag, col in (("bad", "#c62828"), ("warn", "#b26a00"), ("ok", "#2e7d32")):
            self.out.tag_configure(tag, foreground=col)

    def _cases(self):
        d = self.base_dir.get()
        try:
            cs = sorted(c for c in os.listdir(d) if os.path.isdir(os.path.join(d, c, "p")))
        except OSError:
            cs = []
        self.case_cb["values"] = cs

    def ap_ref(self) -> dict:
        m = self.ap_mode.get() or "none"
        if m == "manual":
            return {"mode": m, "manual": float(self.ap_manual.get()) * 1e-3}
        if m in ("model", "model_at_spin"):
            if not self.ap_model.get():
                raise ValueError("choose the SLD model of ap_ref")
            return {"mode": m, "model": self.ap_model.get()}
        return {"mode": "none"}

    def variables(self) -> dict:
        out = {k: v.get() for k, v in self.known.items() if v.get().strip()}
        for part in self.other.get().split(";"):
            if part.strip():
                if "=" not in part:
                    raise ValueError(f"other variables: '{part.strip()}' needs '$name$ = values'")
                k, val = part.split("=", 1)
                out[k.strip()] = val
        return out

    def build(self) -> dict:
        return ex.build_simulation(self.base_dir.get().strip(), self.case.get().strip(), self.doe_name.get().strip(),
                                   self.depths.get(), "kappa" if self.unit.get() == "kappa" else "mm", self.spins.get(),
                                   self.ap_ref(), self.variables(), self.combine.get(), _num(self.nb_proc.get(), int) or 1,
                                   self.n2m.get().strip(), [x.strip() for x in self.signals.get().split(",") if x.strip()],
                                   self.force.get().strip(), depths_end=self.depths_end.get().strip() or None)

    def run_opts(self) -> dict:
        return {k: True for k, v in (("timed", self.timed), ("auto_extract", self.auto)) if v.get()}

    def preview(self) -> bool:
        """Show the cases and the problems; True when there is no error."""
        t = self.out
        t.delete("1.0", "end")
        try:
            sim = self.build()
        except Exception as exc:
            t.insert("end", f"ERROR {exc}\n", "bad")
            return False
        rows, errs, warns = ex.check_simulation(sim)
        for x in errs:
            t.insert("end", f"ERROR {x}\n", "bad")
        for x in warns:
            t.insert("end", f"warning {x}\n", "warn")
        ref = getattr(self.dlg, "ref_exp", lambda: None)()
        for x in ex.kappa_overlap(rows, ref):
            t.insert("end", f"warning {x}\n", "warn")
        t.insert("end", f"{len(rows)} case(s) → {sim['base_dir']}/{sim['doe_name']}\n", "ok" if not errs else None)
        est = ex.estimate_time(sim)
        if est:
            t.insert("end", est + "\n")
        extra = [k for k in (rows[0] if rows else {}) if k not in ex.ROW_EXTRA]
        ramps = any(d.get("ramp") for d in rows)
        f3 = lambda x: "-" if x is None else f"{x:.3f}"   # noqa: E731
        t.insert("end", f"{'case':>5} {'type':>5} {'Ap [mm]':>10} " + (f"{'Ap end':>10} " if ramps else "")
                 + f"{'kappa':>8} " + (f"{'kappa end':>9} " if ramps else "") + f"{'n [rpm]':>10}  "
                 + "  ".join(extra) + "\n")
        for d in rows:
            r = bool(d.get("ramp"))
            k0, k1 = (d.get("kappa_start"), d.get("kappa_end")) if r else (d.get("kappa"), d.get("kappa"))
            t.insert("end", f"{d['case']:>5} {'ramp' if r else 'const':>5} {d['Ap_start'] * 1e3:>10.4f} "
                            + (f"{d['Ap_end'] * 1e3:>10.4f} " if ramps else "") + f"{f3(k0):>8} "
                            + (f"{f3(k1):>9} " if ramps else "")
                            + f"{d.get('spin_rate', float('nan')):>10g}  " + "  ".join(f"{d[x]:g}" for x in extra) + "\n")
        return not errs


class SldPicker:
    """Choose the depths of a simulation on the SLD of a model: the lobes, the line of n, its stability limit and
    the cases of the reference (coloured by label). Click = add an Ap at the n line (only the height counts),
    right click = remove the nearest; or fill a range in Ap [mm] or kappa (x the limit at n). 'Use these Ap'
    writes them, the n (and, if ticked, ap_ref = model_at_spin of that model) into the simulation form."""
    LAB_COL = {"stable": "#0072B2", "unstable": "#D55E00", "gray": "#999999", "": "#555555"}

    def __init__(self, app, frame):
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        tk, ttk = app.tk, app.ttk
        self.app, self.frame, self.tk = app, frame, tk
        models = ex.sld_models()
        if not models:
            app._msg("SLD", "sld_model / sld_tools cannot be loaded: no SLD to draw.", "error")
            return
        w = self.win = tk.Toplevel(app.root)
        w.title("Pick the depths on the SLD")
        top = ttk.Frame(w, padding=6)
        top.pack(fill=tk.X)
        self.model = tk.StringVar(value=frame.ap_model.get() if frame.ap_model.get() in models else models[0])
        spins = ex._values(frame.spins.get()) if frame.spins.get().strip() else []
        self.n = tk.StringVar(value=f"{spins[0]:g}" if spins else "12000")
        ttk.Label(top, text="SLD model").pack(side=tk.LEFT)
        ttk.Combobox(top, textvariable=self.model, values=models, state="readonly", width=16).pack(side=tk.LEFT, padx=4)
        ttk.Label(top, text="n [rpm]").pack(side=tk.LEFT, padx=(10, 0))
        e = ttk.Entry(top, textvariable=self.n, width=10)
        e.pack(side=tk.LEFT, padx=4)
        e.bind("<Return>", lambda _e: self.new_view())
        ttk.Button(top, text="Redraw", command=self.new_view).pack(side=tk.LEFT)
        self.scope = tk.StringVar(value=self.LOBE)
        self.scope_cb = ttk.Combobox(top, textvariable=self.scope, values=self.scopes(), state="readonly", width=34)
        self.scope_cb.pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(top, text="n at the minimum limit", command=self.set_min_n).pack(side=tk.LEFT, padx=4)
        self.model.trace_add("write", lambda *_: self.new_view())
        self.hint = tk.StringVar()
        ttk.Label(top, textvariable=self.hint, foreground="#555").pack(side=tk.LEFT, padx=6)
        row = ttk.Frame(w, padding=(6, 0))
        row.pack(fill=tk.X)
        self.r_from, self.r_to, self.r_n = tk.StringVar(), tk.StringVar(), tk.StringVar(value="5")
        self.r_unit = tk.StringVar(value="kappa")
        self.mode = tk.StringVar(value="points")   # points = constant Ap; ramps = Ap from a start to an end
        self.span = tk.StringVar(value="0.6")
        ttk.Label(row, text="add").pack(side=tk.LEFT)
        for m in ("points", "ramps"):
            ttk.Radiobutton(row, text=m, value=m, variable=self.mode, command=self._mode_changed).pack(side=tk.LEFT)
        ttk.Label(row, text="│ fill a range: from").pack(side=tk.LEFT, padx=(6, 0))
        ttk.Entry(row, textvariable=self.r_from, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="to").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.r_to, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="cases").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.r_n, width=5).pack(side=tk.LEFT, padx=2)
        self.span_lbl = ttk.Label(row, text="ramp span")
        self.span_lbl.pack(side=tk.LEFT)
        self.span_ent = ttk.Entry(row, textvariable=self.span, width=6)
        self.span_ent.pack(side=tk.LEFT, padx=2)
        for u in ("kappa", "Ap [mm]"):
            ttk.Radiobutton(row, text=u, value=u, variable=self.r_unit, command=self.new_view).pack(side=tk.LEFT)
        ttk.Button(row, text="Fill", command=self.fill).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Propose (as the validation planner)", command=self.propose).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="Clear", command=self.clear).pack(side=tk.LEFT)
        prow = ttk.Frame(w, padding=(6, 0))
        prow.pack(fill=tk.X)
        self.gap, self.jit, self.seed = tk.StringVar(value="0.02"), tk.StringVar(value="0.6"), tk.StringVar(value="1")
        ttk.Label(prow, text="proposal: min gap to the training kappa").pack(side=tk.LEFT)
        ttk.Entry(prow, textvariable=self.gap, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(prow, text="jitter (0-1)").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Entry(prow, textvariable=self.jit, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(prow, text="seed").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Entry(prow, textvariable=self.seed, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(prow, text="(the validation planner uses 0.02, 0.6, 1; ramps: no gap, a ramp covers a range; "
                             "span < 0 = Ap decreasing)", foreground="#666").pack(side=tk.LEFT, padx=8)
        body = ttk.Frame(w)
        body.pack(fill=tk.BOTH, expand=True)
        side = ttk.Frame(body, padding=6)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(side, text="chosen depths (points) and ramps", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.lst = tk.Listbox(side, width=76, height=24, font=("Consolas", 9), selectmode=tk.EXTENDED)
        self.lst.pack(fill=tk.Y, expand=True)
        self.proposed = []   # the Ap of the last proposal (replaced by the next one, removable from the list)
        self.accepted = []   # proposals accepted: kept apart from yours, never replaced by a new proposal
        self.ramps, self.r_proposed, self.r_accepted = [], [], []   # the same for ramps: (Ap start, Ap end) mm
        self._pending = None   # ramps mode: the start of a ramp clicked, waiting for its end
        self._items = []       # what each line of the list is: ("pt", Ap) or ("rp", (start, end))
        bt = ttk.Frame(side)
        bt.pack(anchor="w", pady=(4, 0))
        ttk.Button(bt, text="Remove selected", command=self.remove_selected).pack(side=tk.LEFT)
        ttk.Button(bt, text="Remove all proposed", command=self.remove_proposed).pack(side=tk.LEFT, padx=4)
        ttk.Button(bt, text="Accept proposed", command=self.accept_proposed).pack(side=tk.LEFT)
        ttk.Label(side, text="[proposed] = from Propose (blue, hollow circles / arrows)\n"
                             "[accepted] = proposals you accepted (purple, diamonds / arrows)\n"
                             "[yours] = added by you (green, crosses / arrows). Ctrl/Shift to select several.\n"
                             "A ramp is an arrow from its start to its end Ap (drawn side by side next to the n "
                             "line so they do not overlap; all are at that n).",
                  foreground="#666", wraplength=420, justify="left").pack(anchor="w")
        self.fig = Figure(figsize=(8.5, 5.5), constrained_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=body)
        self.toolbar = NavigationToolbar2Tk(self.canvas, body, pack_toolbar=False)
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        bot = ttk.Frame(w, padding=6)
        bot.pack(fill=tk.X)
        self.set_ref = tk.BooleanVar(value=True)
        ttk.Checkbutton(bot, text="also set ap_ref = model_at_spin of this model (kappa = Ap / limit at n)",
                        variable=self.set_ref).pack(side=tk.LEFT)
        ttk.Button(bot, text="Use these Ap", command=self.use).pack(side=tk.RIGHT)
        ttk.Button(bot, text="Cancel", command=w.destroy).pack(side=tk.RIGHT, padx=6)
        self.ref = getattr(frame.dlg, "ref_exp", lambda: None)()
        self.aps = []
        if frame.unit.get() == "Ap [mm]" and frame.depths.get().strip():   # the form's cases: points and ramps
            starts = [round(a, 6) for a in ex._values(frame.depths.get())]
            ends = [round(a, 6) for a in ex._values(frame.depths_end.get())] if frame.depths_end.get().strip() else []
            ends = ends * len(starts) if len(ends) == 1 else ends
            for i, a in enumerate(starts):
                b = ends[i] if i < len(ends) else a
                if abs(b - a) > 1e-9:
                    self.ramps.append((a, b))
                else:
                    self.aps.append(a)
            self.aps = sorted(set(self.aps))
            self.ramps = sorted(set(self.ramps))
        if self.ramps:
            self.mode.set("ramps")
        self._mode_changed(draw=False)
        self.draw()

    def _mode_changed(self, draw=True):
        ramps = self.mode.get() == "ramps"
        self.hint.set("two clicks = a ramp (start, then end Ap; y axis units) · right click = remove the nearest ramp"
                      if ramps else "click = add a depth at the n line (in the units of the y axis) · right click = "
                                    "remove the nearest")
        for wdg in (self.span_lbl, self.span_ent):
            wdg.state(["!disabled"] if ramps else ["disabled"])
        self._pending = None
        if draw:
            self.draw()

    def clear(self):
        self.aps, self.proposed, self.accepted = [], [], []
        self.ramps, self.r_proposed, self.r_accepted = [], [], []
        self._pending = None
        self.draw()

    def _n(self) -> float:
        return float(self.n.get())

    LOBE = "this lobe (where n is)"
    ALL = "whole SLD (all modes)"

    def scopes(self) -> list:
        """What 'n at the minimum limit' can look for: the lobe of the current n, the whole SLD, and the lowest
        point of each mode (one per natural frequency of the model)."""
        try:
            _, fp = ex._sld().lobes(self.model.get())
        except Exception:
            fp = []
        return [self.LOBE, self.ALL] + [f"lowest of the {f:g} Hz mode" for f in fp]

    def _segments(self, scope: str):
        """(mode index, lobe index) of the segments the scope covers."""
        lb, fp = ex._sld().lobes(self.model.get())
        if scope == self.ALL:
            return [(j, i) for j in range(lb.shape[0]) for i in range(lb.shape[1])]
        if scope in self.scopes()[2:]:
            j = self.scopes()[2:].index(scope)
            return [(j, i) for i in range(lb.shape[1])]
        return None

    @staticmethod
    def _crossings(lb, n: float) -> list:
        """[(limit at n, mode j, lobe i)] for every place where a lobe crosses the line of spin n. A lobe is not
        always sorted by spin (it can turn back on itself), so the crossings are looked for between consecutive
        points, as ap_lim does."""
        import math
        out = []
        for j in range(lb.shape[0]):
            for i in range(lb.shape[1]):
                x, y = lb[j, i, :, 0], lb[j, i, :, 1]
                for p in range(len(x) - 1):
                    a, b = x[p] - n, x[p + 1] - n
                    if not (all(math.isfinite(v) for v in (x[p], x[p + 1], y[p], y[p + 1])) and a * b <= 0 and a != b):
                        continue
                    t = a / (a - b)
                    out.append((float(y[p] + t * (y[p + 1] - y[p])), j, i))
        return out

    def _active_segment(self, n: float):
        """(limit at n, mode j, lobe i) of the lobe that sets the limit at n (the lowest crossing), or None."""
        lb, _ = ex._sld().lobes(self.model.get())
        cr = self._crossings(lb, n)
        return min(cr) if cr else None

    def min_limit_rpm(self, scope: str | None = None) -> float:
        """Spin [rpm] of the lowest point for the scope. 'this lobe': among the lobes whose spin range contains n,
        the one that sets the limit at n, and its lowest point (outside a lobe: the lowest of the whole SLD).
        'whole SLD' or one mode: the lowest point of those segments (each lobe is one segment, spin increasing)."""
        import math
        scope = scope or self.scope.get()
        lb, _ = ex._sld().lobes(self.model.get())
        if scope == self.LOBE:
            act = self._active_segment(self._n())
            if act is not None:
                _, j, i = act
                return self._lowest_point(lb, [(j, i)])
            scope = self.ALL   # n is not inside a lobe (pocket or outside the calculated range)
        segs = self._segments(scope)
        if not segs:
            raise ValueError(f"no lobes for '{scope}'")
        return self._lowest_point(lb, segs)

    @staticmethod
    def _lowest_point(lb, segs) -> float:
        import math
        best, rpm = math.inf, None
        for j, i in segs:
            for x, y in zip(lb[j, i, :, 0], lb[j, i, :, 1]):
                if math.isfinite(x) and math.isfinite(y) and y < best:
                    best, rpm = y, x
        if rpm is None:
            raise ValueError("the SLD has no finite lobe there")
        return float(rpm)

    def set_min_n(self):
        try:
            rpm = self.min_limit_rpm()
        except Exception as exc:
            self.app._msg("SLD", f"cannot find the minimum: {exc}", "warn")
            return
        self.n.set(f"{rpm:.1f}")
        self.new_view()

    def limit(self):
        try:
            return ex._sld().ap_lim(self.model.get(), self._n())
        except Exception:
            return None

    def draw(self):
        import math
        sm, ax = ex._sld(), self.ax
        if hasattr(self, "scope_cb"):   # the modes (and so the scopes) depend on the model
            vals = self.scopes()
            self.scope_cb["values"] = vals
            if self.scope.get() not in vals:
                self.scope.set(self.LOBE)
        keep = getattr(self, "_keep_view", False)   # zoom/pan survive adding or removing points
        if keep:
            view = (ax.get_xlim(), ax.get_ylim())
        ax.cla()
        try:
            lb, _ = sm.lobes(self.model.get())
            n = self._n()
        except Exception as exc:
            ax.text(0.5, 0.5, f"cannot draw: {exc}", ha="center", va="center", transform=ax.transAxes)
            self.canvas.draw()
            return
        lim = self.limit()
        # y axis: Ap [mm], or kappa = Ap / (limit at this n) when 'kappa' is chosen and that limit is finite
        self.div = div = lim if (self.r_unit.get() == "kappa" and lim is not None and math.isfinite(lim)) else 1.0
        kap = div != 1.0
        # one colour per MODE (all its lobes share it); the lobe the current n is in is drawn thicker and named
        import matplotlib.pyplot as _plt
        _, fp = sm.lobes(self.model.get())
        cmap = _plt.get_cmap("tab10")
        active = self._active_segment(n)
        for j in range(lb.shape[0]):
            col = cmap(j % 10)
            for i in range(lb.shape[1]):
                x, y = lb[j, i, :, 0], lb[j, i, :, 1]
                ok = [math.isfinite(a) and math.isfinite(b) for a, b in zip(x, y)]
                is_active = active is not None and (j, i) == active[1:]
                ax.plot(x[ok], y[ok] / div, color=col, lw=2.8 if is_active else 1.1,
                        label=(f"{fp[j]:g} Hz mode" if i == 0 else None))
                if is_active:
                    ax.plot([], [], color=col, lw=2.8, label=f"{fp[j]:g} Hz mode, lobe that sets the limit at n")
        pts = ex.case_points_of(self.ref)
        for lab in sorted({p[2] for p in pts}):
            sel = [p for p in pts if p[2] == lab and p[3] == p[1]]
            seg = [p for p in pts if p[2] == lab and p[3] != p[1]]   # a reference ramp: a vertical segment
            col = self.LAB_COL.get(lab, "#555")
            if sel:
                ax.scatter([p[0] for p in sel], [p[1] / div for p in sel], s=18, color=col,
                           label=f"reference {lab or 'case'} ({len(sel)})", zorder=3)
            if seg:
                ax.vlines([p[0] for p in seg], [p[1] / div for p in seg], [p[3] / div for p in seg], colors=col,
                          lw=2, label=f"reference ramps {lab or ''} ({len(seg)})", zorder=3)
        ax.axvline(n, color="#1565c0", ls="--", lw=1)
        if lim is not None and math.isfinite(lim):
            ax.plot([n], [lim / div], marker="_", markersize=22, color="#1565c0", mew=2,
                    label=f"limit at n: {lim:.3f} mm" + (" (kappa = 1)" if kap else ""))
        if kap:
            ax.axhline(1.0, color="#1565c0", lw=0.8, ls=":")
        if self.aps:
            acc = [a for a in self.aps if a in self.accepted]
            mine = [a for a in self.aps if a not in self.proposed and a not in self.accepted]
            prop = [a for a in self.aps if a in self.proposed]
            if acc:   # accepted: filled purple diamonds
                ax.scatter([n] * len(acc), [a / div for a in acc], marker="D", s=40, color="#6a1b9a", zorder=4,
                           label=f"accepted ({len(acc)})")
            if prop:   # proposed: hollow blue circles; yours: red crosses
                ax.scatter([n] * len(prop), [a / div for a in prop], marker="o", s=60, facecolors="none",
                           edgecolors="#1565c0", linewidths=1.6, zorder=4, label=f"proposed ({len(prop)})")
            if mine:
                ax.scatter([n] * len(mine), [a / div for a in mine], marker="x", s=50, color="#2e7d32", zorder=4,
                           label=f"yours ({len(mine)})")
        xs_all = [v for v in lb[..., 0].ravel() if math.isfinite(v)]
        dx = 0.006 * ((max(xs_all + [n]) - min(xs_all + [n])) if xs_all else n)   # ramps side by side
        counts = {}
        for i, r in enumerate(self.ramps):
            grp = "proposed" if r in self.r_proposed else ("accepted" if r in self.r_accepted else "yours")
            col = {"proposed": "#1565c0", "accepted": "#6a1b9a", "yours": "#2e7d32"}[grp]
            x = n + (i + 1) * dx
            ax.annotate("", xy=(x, r[1] / div), xytext=(x, r[0] / div), zorder=4,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8, shrinkA=0, shrinkB=0))
            ax.plot([x], [r[0] / div], marker="o", ms=4, color=col, zorder=4)
            counts[grp] = counts.get(grp, 0) + 1
        for grp, c in counts.items():
            ax.plot([], [], color={"proposed": "#1565c0", "accepted": "#6a1b9a", "yours": "#2e7d32"}[grp], lw=1.8,
                    marker="o", ms=4, label=f"ramps {grp} ({c}): dot = start, arrow = end")
        if self._pending is not None:
            ax.plot([n], [self._pending / div], marker="^", ms=10, color="#2e7d32", zorder=5,
                    label="ramp start: click its end")
        cap = sm.ap_crit(self.model.get())
        top = max([cap * 4] + [a * 1.15 for a in self.aps] + [max(r) * 1.15 for r in self.ramps]
                  + [max(p[1], p[3]) * 1.1 for p in pts]) / div
        ax.set_ylim(0, min(top, 4.0) if kap else top)
        xs = [v for v in lb[..., 0].ravel() if math.isfinite(v)]
        if xs:
            ax.set_xlim(min(xs + [n]) * 0.95, max(xs + [n]) * 1.02)
        ax.set_xlabel("n [rpm]")
        ax.set_ylabel(f"kappa = Ap / {lim:.3f} mm (limit at n)" if kap else "Ap [mm]")
        ax.set_title(f"{self.model.get()} · n = {n:g} rpm · " + (
            "pocket between lobes (no finite limit: axis stays in Ap)" if lim is not None and not math.isfinite(lim)
            else f"limit {lim:.3f} mm" if lim is not None else "outside the lobes computed (axis stays in Ap)"),
            fontsize=10)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, loc="upper right")
        if keep:
            ax.set_xlim(view[0])
            ax.set_ylim(view[1])
        self._keep_view = True
        self.canvas.draw()
        self.lst.delete(0, "end")
        self._items = [("pt", a) for a in self.aps] + [("rp", r) for r in self.ramps]
        ok_lim = lim and math.isfinite(lim)
        for idx, (kind, a) in enumerate(self._items):
            if kind == "pt":
                k = a / lim if ok_lim else None
                zone = "" if k is None else ("stable" if k < 1 else "UNSTABLE")
                prop, acc = a in self.proposed, a in self.accepted
                txt = f" Ap {a:8.4f} mm" + (f"   kappa {k:6.3f}  {zone}" if k is not None else "")
            else:
                k0, k1 = (a[0] / lim, a[1] / lim) if ok_lim else (None, None)
                zone = "" if k0 is None else ("crosses 1" if min(k0, k1) < 1 <= max(k0, k1) else
                                              ("stable" if max(k0, k1) < 1 else "UNSTABLE"))
                prop, acc = a in self.r_proposed, a in self.r_accepted
                txt = (f" ramp Ap {a[0]:.4f} -> {a[1]:.4f} mm"
                       + (f"  kappa {k0:.3f} -> {k1:.3f}  {zone}" if k0 is not None else ""))
            tag = "[proposed]" if prop else ("[accepted]" if acc else "[yours]   ")
            self.lst.insert("end", tag + txt)
            self.lst.itemconfig(idx, foreground="#1565c0" if prop else ("#6a1b9a" if acc else "#2e7d32"))

    def on_click(self, ev):
        if ev.inaxes is not self.ax or ev.ydata is None or self.toolbar.mode:
            return
        div = getattr(self, "div", 1.0)   # kappa axis: the click is a kappa, stored as Ap = kappa x limit
        if self.mode.get() == "ramps":   # two clicks = one ramp (start, end); right click = remove the nearest
            if ev.button == 1 and ev.ydata > 0:
                v = round(float(ev.ydata) * div, 4 if div != 1.0 else 3)
                if self._pending is None:
                    self._pending = v
                else:
                    if abs(v - self._pending) > 1e-6:
                        self.ramps = sorted(set(self.ramps + [(self._pending, v)]))
                    self._pending = None
            elif ev.button == 3:
                if self._pending is not None:
                    self._pending = None
                elif self.ramps:
                    y = float(ev.ydata) * div
                    gone = min(self.ramps, key=lambda r: (max(min(r) - y, 0, y - max(r)), abs((r[0] + r[1]) / 2 - y)))
                    self._drop_ramps([gone])
            self.draw()
            return
        if ev.button == 1 and ev.ydata > 0:
            self.aps = sorted(set(self.aps + [round(float(ev.ydata) * div, 4 if div != 1.0 else 3)]))
        elif ev.button == 3 and self.aps:
            self.aps.remove(min(self.aps, key=lambda a: abs(a / div - ev.ydata)))
        self.draw()

    def _drop_ramps(self, gone):
        self.ramps = [r for r in self.ramps if r not in gone]
        self.r_proposed = [r for r in self.r_proposed if r not in gone]
        self.r_accepted = [r for r in self.r_accepted if r not in gone]

    def _scale(self, what: str):
        """Factor from the range units to Ap [mm] (the limit at n for kappa), or None after telling why."""
        import math
        if self.r_unit.get() != "kappa":
            return 1.0
        lim = self.limit()
        if lim is None or not math.isfinite(lim):
            self.app._msg(what, "kappa needs a finite stability limit at this n (it is a pocket between lobes or "
                                "outside the lobes): choose another n or use Ap [mm]", "warn")
            return None
        return lim

    def _ramp_ends(self, starts, scale, what):
        """[(start, end)] Ap mm of ramps that start at `starts` (range units) with the span of the form."""
        try:
            span = float(self.span.get())
        except ValueError:
            self.app._msg(what, "give the ramp span (a number in the units of the range; < 0 = Ap decreasing)", "warn")
            return None
        if span == 0:
            self.app._msg(what, "a ramp span of 0 is a constant case: use the points mode", "warn")
            return None
        out = [(round(s * scale, 4), round((s + span) * scale, 4)) for s in starts]
        if any(min(r) <= 0 for r in out):
            self.app._msg(what, "a ramp would end at Ap <= 0: change the range or the span", "warn")
            return None
        return out

    def fill(self):
        try:
            a, b, n = float(self.r_from.get()), float(self.r_to.get()), int(self.r_n.get())
        except ValueError:
            self.app._msg("Fill", "give from, to (numbers) and the number of cases", "warn")
            return
        vals = [a + (b - a) * i / (n - 1) for i in range(n)] if n > 1 else [a]
        scale = self._scale("Fill")
        if scale is None:
            return
        if self.mode.get() == "ramps":   # ramps that START regularly in [from, to], each with the span
            new = self._ramp_ends(vals, scale, "Fill")
            if new is None:
                return
            self.ramps = sorted(set(self.ramps + new))
        else:
            self.aps = sorted(set(self.aps + [round(v * scale, 4) for v in vals if v > 0]))
        self.draw()

    def propose(self):
        """Same rule as doe_val_planner: the range [from, to] in kappa split into 'cases' strata, one case per
        stratum with a little jitter, kept at least 0.02 away from the kappa already used by the reference (its
        extracted cases; no labels needed to choose what to simulate). Kappa -> Ap with the limit at n."""
        import math
        sys.path.insert(0, ex.SIM) if ex.SIM not in sys.path else None
        import doe_val_planner as vp
        try:
            a, b, n = float(self.r_from.get()), float(self.r_to.get()), int(self.r_n.get())
        except ValueError:
            self.app._msg("Propose", "give from, to (kappa) and the number of cases", "warn")
            return
        if self.mode.get() == "ramps":
            self.propose_ramps(vp, a, b, n)
            return
        info = ex.h5_info(self.ref_h5()) if self.ref_h5() else None
        used = list(info["kappa"]) if info and info["kappa"] else []
        lim = self.limit()
        if lim is None or not math.isfinite(lim):
            self.app._msg("Propose", "kappa needs a finite stability limit at this n (pocket or outside the lobes)", "warn")
            return
        try:
            gap, jit, seed = float(self.gap.get()), float(self.jit.get()), int(self.seed.get())
            picked = vp.sample_zones([("propose", a, b, n)], used, gap, jit, seed)
        except ValueError as exc:
            self.app._msg("Propose", f"{exc} (gap, jitter and seed: numbers; seed an integer)", "warn")
            return
        # a new proposal replaces the previous one; the cases you added by hand stay
        self.aps = sorted(set(a for a in self.aps if a not in self.proposed) | {round(k * lim, 4) for _, k in picked})
        self.proposed = [round(k * lim, 4) for _, k in picked]
        self.r_unit.set("kappa")
        self.draw()
        self.app.status_msg.set(f"{len(picked)} cases proposed in kappa {a:g}-{b:g}, away from the {len(used)} kappa of "
                                f"the reference (gap {gap:g}, jitter {jit:g}, seed {seed})")

    def propose_ramps(self, vp, a, b, n):
        """N ramps whose START is stratified in [from, to] (one per stratum, jitter and seed as for the points; no
        gap to the training: a ramp covers a range), each with the span of the form (in the range units)."""
        scale = self._scale("Propose")
        if scale is None:
            return
        try:
            jit, seed = float(self.jit.get()), int(self.seed.get())
            picked = vp.sample_zones([("propose", a, b, n)], [], 0.0, jit, seed)
        except ValueError as exc:
            self.app._msg("Propose", f"{exc} (jitter and seed: numbers; seed an integer)", "warn")
            return
        new = self._ramp_ends([k for _, k in picked], scale, "Propose")
        if new is None:
            return
        self.ramps = sorted(set(r for r in self.ramps if r not in self.r_proposed) | set(new))
        self.r_proposed = new
        self.draw()
        self.app.status_msg.set(f"{len(new)} ramps proposed, starts in {a:g}-{b:g} {self.r_unit.get()}, span "
                                f"{self.span.get()} (jitter {jit:g}, seed {seed})")

    def new_view(self):
        """Model, n or units changed: the plot is drawn again with its own limits (not the zoom kept)."""
        self._keep_view = False
        self.draw()

    def accept_proposed(self):
        """The current proposal becomes ordinary cases ([yours]): the next Propose keeps them and proposes others."""
        n = len(self.proposed) + len(self.r_proposed)
        self.accepted = sorted(set(self.accepted + self.proposed))
        self.r_accepted = sorted(set(self.r_accepted + self.r_proposed))
        self.proposed, self.r_proposed = [], []
        self.draw()
        self.app.status_msg.set(f"{n} proposed case(s) accepted (own group): the next Propose will not replace them")

    def remove_proposed(self):
        """Removes the whole last proposal (points and ramps), keeping what you added by hand."""
        self.aps = [a for a in self.aps if a not in self.proposed]
        self.ramps = [r for r in self.ramps if r not in self.r_proposed]
        self.proposed, self.r_proposed = [], []
        self.draw()

    def remove_selected(self):
        """Removes the depths and ramps selected in the list (the list shows self._items in order)."""
        sel = set(self.lst.curselection())
        if not sel:
            return
        gone = [self._items[i] for i in sel if i < len(self._items)]
        pts = [a for k, a in gone if k == "pt"]
        self.aps = [a for a in self.aps if a not in pts]
        self.proposed = [a for a in self.proposed if a not in pts]
        self.accepted = [a for a in self.accepted if a not in pts]
        self._drop_ramps([a for k, a in gone if k == "rp"])
        self.draw()

    def ref_h5(self):
        """The reference's extracted signals file (its kappa are the ones to stay away from), or None."""
        ref = self.ref
        return ref.data_h5 if ref is not None and ref.data_h5 and os.path.isfile(ref.data_h5) else None

    def use(self):
        if not self.aps and not self.ramps:
            self.app._msg("SLD", "choose at least one depth or ramp (click on the plot or Fill)", "warn")
            return
        f = self.frame
        f.unit.set("Ap [mm]")   # points first (end = start: constant), then the ramps
        f.depths.set(", ".join(f"{a:.6g}" for a in self.aps + [r[0] for r in self.ramps]))
        f.depths_end.set(", ".join(f"{a:.6g}" for a in self.aps + [r[1] for r in self.ramps]) if self.ramps else "")
        f.spins.set(self.n.get())
        if self.set_ref.get():
            f.ap_mode.set("model_at_spin")
            f.ap_model.set(self.model.get())
        self.win.destroy()
        f.preview()


class SimulationForm(_Dialog):
    """Edit config of Simulate / Extract: the simulation of one run of the experiment."""

    def __init__(self, app, e, i: int = 0):
        super().__init__(app, f"Simulation — {e.name}")
        self.e, self.i = e, i
        if len(e.runs) > 1:
            self.note(f"Run {i + 1} of {len(e.runs)}: " + ", ".join(r.doe_name for r in e.runs))
        r = e.runs[i] if e.runs else None
        if r is not None and r.imported:
            self.note(f"Imported folder (already simulated, read-only): {r.doe_dir}\n"
                      "The app cannot re-run it. To plan new cases from it: 'New experiment…' with 'Load values from' "
                      "this experiment (its simulation is rebuilt from the folder), or open it in the planner below.",
                      "#b26a00")
            import yaml
            try:
                sim, src = ex.simulation_of_run(r), ("doe_config.yaml left by doe_runner" if r.folder_config()
                                                     else "rebuilt from the folder (var_val.py or the .h5)")
                txt = yaml.safe_dump(sim, sort_keys=False)
            except Exception as exc:
                sim, src, txt = None, "cannot be rebuilt", str(exc)
            self.section(f"Its simulation ({src})")
            t = self.tk.Text(self.body, width=100, height=18, font=("Consolas", 9))
            t.insert("1.0", txt)
            t.grid(row=self.row, column=0, columnspan=3)
            self.row += 1
            if sim is not None:
                self.ttk.Button(self.body, text="Open in the planner (cases on the SLD)", command=lambda: launch(
                    "gui", "DOE_simulacion/doe_planner.py", [ex.planner_config(e, i)])).grid(row=self.row, column=0,
                                                                                             sticky="w")
                self.row += 1
            self.frame = None
            self.buttons("Close")
            return
        vals = ex.simulation_form(ex.explicit_simulation(r.cfg)) if r is not None else {}
        if r is not None and r.existing:
            self.note("Simulated outside the app (imported without .h5). Extract reads the cases from the folder: "
                      "only ap_ref (kappa) and the signals to extract matter here. Simulate stays blocked so the "
                      "folder is never deleted.", "#b26a00")
        if r is not None and "config" in r.entry:
            self.note(f"This run reads {r.source}. Saving writes the whole simulation inside the experiment "
                      "(nothing inherited from base.yaml any more); the config file itself is not changed.", "#1565c0")
        self.frame = SimulationFrame(self, vals, e.section("simulate"))
        bf = self.ttk.Frame(self.body)
        bf.grid(row=self.row, column=0, columnspan=3, sticky="w")
        self.row += 1
        if r is not None:
            self.ttk.Button(bf, text="Open in the planner (cases on the SLD)",
                            command=lambda: launch("gui", "DOE_simulacion/doe_planner.py", [r.config])).pack(side="left")
        self.buttons()
        self.frame.preview()

    def ref_exp(self):
        return self.e.ref

    def save(self):
        if self.frame is None:
            return True
        if not self.frame.preview():
            raise ValueError("fix the errors shown in red first")
        ex.set_simulation(self.e.name, self.i, self.frame.build())
        ex.save_section(self.e.name, "simulate", self.frame.run_opts() or None)
        return True


def _experiment_names() -> list:
    return ex.list_experiments()


def _value_sources() -> list:
    """'Load values from' choices: every experiment with a run (also imported folders: their simulation is
    rebuilt from the folder or its .h5), and configs/*.yaml."""
    out = ["(defaults: base.yaml)"]
    for n in ex.list_experiments():
        try:
            if ex.load(n).runs:
                out.append(f"experiment: {n}")
        except Exception:
            pass
    return out + [f"config: {c}" for c in ex.list_configs()]


def _source_values(src: str) -> dict:
    if src.startswith("experiment: "):
        e = ex.load(src[12:])
        r = next((r for r in e.runs if r.cfg), e.runs[0])
        return ex.simulation_form(ex.simulation_of_run(r))
    if src.startswith("config: "):
        dr = ex._doe_runner()
        return ex.simulation_form(ex.explicit_simulation(dr.load_config(dr.find_config(src[8:]))))
    return {}


class NewExperimentDialog(_Dialog):
    """New experiment from scratch: what it is for (flow), an optional reference, and its simulation written in
    full. 'Load values from' only pre-fills the fields."""

    def __init__(self, app):
        tk, ttk = app.tk, app.ttk
        super().__init__(app, "New experiment")
        self.name = self.field("experiment name", tk.StringVar(), width=50, note="free (letters, digits, _ - .)")
        self.descr = self.field("description", tk.StringVar(), width=90)
        self.flow = self.field("what for (stages)", tk.StringVar(value="Labelled dataset (training)"),
                               values=list(ex.FLOWS), width=40,
                               note="the stages can be changed later in 'Experiment settings…'")
        self.ref = self.field("reference experiment", tk.StringVar(value="(none)"), values=["(none)"] + _experiment_names(),
                              width=40, note="optional: its labelled dataset trains the indicators (needed to validate)")
        self.src = self.field("load values from", tk.StringVar(value="(defaults: base.yaml)"), values=_value_sources(),
                              width=60, note="only pre-fills the fields below")
        self.last.configure(postcommand=lambda w=self.last: w.configure(values=_value_sources()))
        self.src.trace_add("write", lambda *_: self._load())
        bf = ttk.Frame(self.body)
        bf.grid(row=self.row, column=1, columnspan=2, sticky="w")
        self.row += 1
        ttk.Button(bf, text="Propose names", command=self._propose).pack(side="left")
        ttk.Button(bf, text="Pick kappa with the validation planner (constant cases)…", command=self._planner).pack(side="left", padx=6)
        self.frame_row = self.row
        self.frame = SimulationFrame(self, {})
        self.buttons("Create")

    def ref_exp(self):
        r = self.ref.get()
        try:
            return None if r in ("", "(none)") else ex.load(r)
        except Exception:
            return None

    def _load(self):
        try:
            vals = _source_values(self.src.get())
        except Exception as exc:
            self.app._msg("Load values", str(exc), "error")
            return
        for w in self.body.grid_slaves():
            if int(w.grid_info()["row"]) >= self.frame_row:
                w.destroy()
        self.row = self.frame_row
        self.frame = SimulationFrame(self, vals)

    def _propose(self):
        try:
            spins = ex._values(self.frame.spins.get())
            ks = (ex._values(self.frame.depths.get()) + ex._values(self.frame.depths_end.get())
                  if self.frame.unit.get() == "kappa" else None)
            name = ex.propose_name(self.flow.get(), self.frame.case.get(), spins[0] if len(set(spins)) == 1 else None, ks)
            self.name.set(name)
            self.frame.doe_name.set(name)
            if not self.descr.get():
                self.descr.set(f"{self.flow.get().lower()}, {self.frame.case.get()}, n = {self.frame.spins.get()} rpm, "
                               f"{self.frame.unit.get()} {self.frame.depths.get()}"
                               + (f" -> {self.frame.depths_end.get()} (ramps)" if self.frame.depths_end.get().strip() else ""))
        except Exception as exc:
            self.app._msg("Propose names", f"fill n and the depths first ({exc})", "warn")

    def _planner(self):
        ref = self.ref.get()
        if ref in ("", "(none)"):
            self.app._msg("Validation planner", "Choose the reference experiment first: the planner places the new "
                                                "kappa values around its labelled cases.", "warn")
            return
        doe = self.frame.doe_name.get().strip() or self.name.get().strip() or "validation_doe"
        launch("gui", "DOE_simulacion/doe_val_planner.py", [ex.load(ref).label["out"], "--name", doe])
        self.app._msg("Validation planner", f"In the planner: Generate cases, then 'Save YAML…' as '{doe}' in configs/.\n"
                                            f"Back here, choose 'config: {doe}' in 'load values from' (the list is "
                                            "refreshed when you open it).")

    def save(self):
        name = self.name.get().strip()
        flow, ref = self.flow.get(), (None if self.ref.get() in ("", "(none)") else self.ref.get())
        if flow == "Validation against a reference" and not ref:
            raise ValueError("a validation needs its reference experiment")
        if not self.frame.preview():
            raise ValueError("fix the errors shown in red first")
        sim = self.frame.build()
        model = self.frame.ap_model.get().strip()   # the SLD model chosen = the simulated machine
        ex.create_experiment(name, [{"simulation": sim, **({"model": model} if model else {})}],
                             stages_on=ex.FLOWS[flow], reference=ref,
                             description=self.descr.get(), simulate=self.frame.run_opts() or None)
        self.app.root.after(50, lambda: self.app.select(name))


class SettingsDialog(_Dialog):
    """What the experiment is: description, stages turned on, reference, output folder."""

    def __init__(self, app, e):
        tk, ttk = app.tk, app.ttk
        super().__init__(app, f"Experiment settings — {e.name}")
        self.e = e
        self.descr = self.field("description", tk.StringVar(value=e.cfg.get("description", "")), width=90)
        self.flow = self.field("flow template", tk.StringVar(value=e.flow), values=list(ex.FLOWS) + ["custom"],
                               note="sets the stage boxes below")
        self.flow.trace_add("write", lambda *_: self._apply_flow())
        box = ttk.LabelFrame(self.body, text="Stages turned on (a stage also brings the ones it needs)", padding=4)
        box.grid(row=self.row, column=0, columnspan=3, sticky="w", pady=4)
        self.row += 1
        self.on = {}
        for i, k in enumerate(k for k in ex.STAGES if k != "merge"):
            self.on[k] = tk.BooleanVar(value=k in e.enabled)
            ttk.Checkbutton(box, text=ex.TITLES[k] + ("  (optional)" if k in ex.OPTIONAL else ""),
                            variable=self.on[k]).grid(row=i // 4, column=i % 4, sticky="w", padx=6)
        others = [n for n in ex.list_experiments() if n != e.name]
        self.ref = self.field("reference experiment", tk.StringVar(value=e.ref.name if e.ref else "(none)"),
                              values=["(none)"] + others, width=40,
                              note="its labelled dataset trains the indicators; this one's labels use its parameters")
        self.out = self.browse("output folder", tk.StringVar(value=e.cfg.get("out_dir", "")),
                               note=f"empty = {os.path.join(e.data_dir, e.name)}")
        self.note("Labels, labelled dataset, indicator and validation results go to the output folder. The data "
                  "(doe_results.h5) stays in the DOE folder of the simulation: change it in Edit config of Simulate.")
        raw = ex.own_yaml(e.name)
        if "extends" in raw or any("config" in r for r in raw.get("runs") or []) or isinstance(
                (raw.get("indicators") or {}).get("variants"), list):
            self.note("This file still inherits (extends / a config of configs/ / preset names): 'Write it out in "
                      "full' copies everything it uses into the file.", "#1565c0")
            ttk.Button(self.body, text="Write it out in full", command=self._explicit).grid(row=self.row, column=1, sticky="w")
            self.row += 1
        self.buttons()

    def _apply_flow(self):
        f = self.flow.get()
        if f in ex.FLOWS:
            for k, v in self.on.items():
                if k in ex.MAIN:
                    v.set(k in ex.FLOWS[f])

    def _explicit(self):
        self.app._guard(lambda: ex.make_explicit(self.e.name))
        self.win.destroy()

    def save(self):
        d = ex.own_yaml(self.e.name)
        d["description"] = self.descr.get()
        d["stages"] = [k for k in ex.STAGES if k in self.on and self.on[k].get()]
        for k in ("kind", "training"):
            d.pop(k, None)
        ref = self.ref.get()
        if ref in ("", "(none)"):
            d.pop("reference", None)
        else:
            d["reference"] = ref
        if self.out.get().strip():
            d["out_dir"] = os.path.normpath(self.out.get().strip()).replace("\\", "/")
        else:
            d.pop("out_dir", None)
        if "validate" in d["stages"] and "validate" not in d:
            d["validate"] = {"channel": "Axial_disp"}
        if "indicators" in d["stages"] and "indicators" not in d:
            d["indicators"] = ex.indicators_for(None if ref in ("", "(none)") else ref)
        if "label_template" in d["stages"] and "label" not in d and ref in ("", "(none)"):
            d["label"] = ex.label_defaults(d.get("indicators"))
        order = ["name", "description", "stages", "reference", "out_dir"]
        ex.yaml_save({**{k: d[k] for k in order if k in d}, **{k: v for k, v in d.items() if k not in order}},
                     ex.exp_path(self.e.name))
        ex.reload()


class CopyDialog(_Dialog):
    def __init__(self, app, e):
        tk = app.tk
        super().__init__(app, f"Copy {e.name}")
        self.e = e
        self.name = self.field("name of the copy", tk.StringVar(value=e.name + "_copy"), width=50)
        self.descr = self.field("description", tk.StringVar(value=e.cfg.get("description", "")), width=90)
        self.suffix = self.field("new DOE folder suffix", tk.StringVar(), width=20,
                                 note="e.g. _n10000: give one if you will change the simulation (the copy then "
                                      "simulates into its own folder). Empty = same data")
        self.note("The copy is one explicit file (nothing inherited) with its own outputs folder. After copying, "
                  "change what differs: Edit config of Simulate (n, Ap / kappa…), Indicators, or Experiment settings.")
        self.buttons("Copy")

    def save(self):
        name = self.name.get().strip()
        ex.copy_experiment(self.e.name, name, self.descr.get(), self.suffix.get().strip())
        self.app.root.after(50, lambda: self.app.select(name, "simulate"))


NOT_EXTRACTED = "(not extracted yet)"


class ImportDialog(_Dialog):
    def __init__(self, app):
        from tkinter import filedialog
        tk = app.tk
        d = filedialog.askdirectory(title="DOE folder (already simulated)", parent=app.root)
        super().__init__(app, "Import folder")
        if not d:
            self.win.destroy()
            return
        self.dir = d
        h5s = sorted(f for f in os.listdir(d) if f.endswith(".h5"))
        self.field("folder", tk.StringVar(value=d), width=90, state="readonly")
        self.h5 = self.field("signals file", tk.StringVar(value="doe_results.h5" if "doe_results.h5" in h5s else
                                                          (h5s[0] if h5s else NOT_EXTRACTED)), values=h5s + [NOT_EXTRACTED],
                             note="the .h5 with case_* groups; 'not extracted yet' = simulated without .h5: Extract "
                                  "runs from the app")
        self.ap_mode = self.field("ap_ref (not extracted only)", tk.StringVar(value="none"),
                                  values=["none", "manual", "model", "model_at_spin"],
                                  note="for kappa = Ap / ap_ref, computed by Extract")
        self.ap_manual = self.field("  ap_ref manual [mm]", tk.StringVar())
        self.ap_model = self.field("  SLD model", tk.StringVar(), values=ex.sld_models())
        self.ref = self.field("reference experiment", tk.StringVar(value="(none)"), values=["(none)"] + _experiment_names(),
                              note="set it to validate these cases against it")
        refs = sorted(f for f in os.listdir(d) if f.startswith("reference_dataset") and f.endswith(".h5"))
        tagged = [f for f in refs if ex._label_attrs(os.path.join(d, f)).get("strategy")]
        self.lab = self.field("labelled dataset", tk.StringVar(value=(tagged or refs or ["(none)"])[0]),
                              values=["(none)"] + refs,
                              note="the ground truth of these cases (ignored with a reference: labels are redone)")
        # the name follows the folder (what the data are called), made unique; the user can change it
        base, nm, i = os.path.basename(os.path.normpath(d)), None, 2
        nm = base
        while nm in ex.list_experiments():
            nm, i = f"{base}_{i}", i + 1
        self.name = self.field("experiment name", tk.StringVar(value=nm), width=50,
                               note="by default the name of the folder")
        self.section("What is in the folder (what Import would turn on)")
        self.prev = tk.Text(self.body, width=120, height=16, font=("Consolas", 9))
        self.prev.grid(row=self.row, column=0, columnspan=3, pady=4)
        self.row += 1
        for tag, col in (("ok", "#2e7d32"), ("bad", "#c62828"), ("warn", "#b26a00")):
            self.prev.tag_configure(tag, foreground=col)
        for v in (self.h5, self.lab, self.ref, self.ap_mode, self.ap_manual, self.ap_model):
            v.trace_add("write", lambda *_: self.preview())
        self.preview()
        self.buttons("Import")

    def ap_ref(self):
        m = self.ap_mode.get() or "none"
        if m == "manual":
            return {"mode": m, "manual": float(self.ap_manual.get()) * 1e-3}
        return {"mode": m, "model": self.ap_model.get()} if m in ("model", "model_at_spin") else {"mode": "none"}

    def preview(self):
        """Contents of the folder, before importing: cases, datasets, results, and the stages that would be marked
        done (only what is really found)."""
        t, d, h5 = self.prev, self.dir, self.h5.get()
        t.delete("1.0", "end")

        def line(txt, tag=None):
            t.insert("end", txt + "\n", tag) if tag else t.insert("end", txt + "\n")
        idx = sorted((i for i in os.listdir(d) if i.isdigit()), key=int)
        sims = sum(os.path.isfile(os.path.join(d, i, ex.detect_case(d), "sens_out.hdf5")) for i in idx)
        line(f"{len(idx)} case folders, {sims} with sens_out.hdf5 (simulated)   case: {ex.detect_case(d)}",
             "ok" if sims else "warn")
        line("doe_config.yaml (simulation config left by doe_runner): " +
             ("yes" if os.path.isfile(os.path.join(d, "doe_config.yaml")) else "no (the simulation cannot be re-run from here)"))
        if h5 == NOT_EXTRACTED:
            try:
                sim = ex.simulation_from_folder(d, self.ap_ref())
                rows = ex.case_rows(ex._doe_runner().load_config(ex._tmp_config(sim)))
                ks = [k for r in rows for k in (r.get("kappa"), r.get("kappa_start"), r.get("kappa_end")) if k is not None]
                aps = [a * 1e3 for r in rows for a in (r.get("Ap_start"), r.get("Ap_end")) if a is not None]
                nr = sum(bool(r.get("ramp")) for r in rows)
                if nr:
                    line(f"  {nr} ramp case(s): " + "; ".join(f"Ap {ex.ap_text(r, '.3g')}, kappa {ex.kappa_text(r)}"
                                                             for r in rows if r.get("ramp"))[:300])
                ns = sorted({r["spin_rate"] for r in rows if "spin_rate" in r})
                line(f"not extracted yet: {len(rows)} cases rebuilt from their var_val.py → Extract will write "
                     f"doe_results.h5 here (Simulate stays blocked: it would delete this folder)", "ok")
                line("  n = " + ", ".join(f"{n:g}" for n in ns[:6]) + " rpm"
                     + (f"   Ap {min(aps):.4g}-{max(aps):.4g} mm" if aps else "")
                     + (f"   kappa {min(ks):.3g}-{max(ks):.3g}" if ks else "   kappa: none (choose an ap_ref)"))
            except Exception as exc:
                line(f"PROBLEM cannot rebuild the simulation: {exc}", "bad")
        elif h5:
            rep = ex.inspect_h5(os.path.join(d, h5))
            for x in rep["problems"]:
                line(f"PROBLEM {h5}: {x}", "bad")
            if rep["ok"]:
                info = ex.h5_info(os.path.join(d, h5))
                kap = ex.kappa_span(info)
                line(f"{h5}: {len(rep['cases'])} cases" + (f" ({info['ramps']} ramps)" if info["ramps"] else "")
                     + f", signals {', '.join(rep['signals'])}", "ok")
                line("  n = " + (f"{float(info['first']['$spin_rate$']):.10g} rpm" if "$spin_rate$" in info["first"] else "missing")
                     + (f"   kappa {kap[0]:g}-{kap[1]:g}" if kap else "   kappa missing") + f"   duration {info['duration']} s")
                for txt in info["ramp_text"][:4]:
                    line(f"  ramp {txt}")
                miss = [a for a in ex.STD_ATTRS if rep["missing"][a]]
                if miss:
                    line(f"  attributes missing in some cases: {', '.join(miss)} (Standardize an .h5… adds them)", "warn")
        found = {k: sorted(f for f in os.listdir(d) if f.startswith(k) and f.endswith(ext))
                 for k, ext in (("reference_dataset", ".h5"), ("reference_labels", ".yaml"),
                                ("doe_indicator_results", ".h5"), ("doe_validation_results", ".h5"))}
        for k, fs in found.items():
            line(f"{k}*: " + (", ".join(fs) if fs else "none"), "ok" if fs else None)
        lab = self.lab.get()
        if lab not in ("", "(none)") and self.ref.get() in ("", "(none)"):
            la = ex._label_attrs(os.path.join(d, lab))
            by = {}
            for _c, (lb, _k) in ex._label_cases(os.path.join(d, lab)).items():
                by[lb] = by.get(lb, 0) + 1
            line(f"chosen labelled dataset {lab}: " + "  ".join(f"{k} {v}" for k, v in by.items())
                 + (f"   labelled by {la.get('strategy')} on {la.get('amp_signal', '?')}" if la else
                    "   (no labelling attributes: how it was labelled is unknown)"), "ok" if la else "warn")
        ref = self.ref.get() not in ("", "(none)")
        on = ["Simulate", "Extract"] + (["Label template", "Label build", "Indicators", "Validate"] if ref else
                                         (["Label build"] if self.lab.get() not in ("", "(none)") else []) +
                                         (["Indicators"] if found["doe_indicator_results"] else []))
        line("Stages turned on: " + ", ".join(on) + ("   (a stage is marked done only if its file exists above)"), "warn")

    def save(self):
        h5 = self.h5.get()
        if h5 != NOT_EXTRACTED:
            rep = ex.inspect_h5(os.path.join(self.dir, h5)) if h5 else {"ok": False, "problems": ["no .h5 in the folder"]}
            if not rep["ok"]:
                raise ValueError("\n".join(rep["problems"]) + "\n\nUse 'Standardize an .h5…' first.")
        name = self.name.get().strip()
        ref = None if self.ref.get() in ("", "(none)") else self.ref.get()
        lab = self.lab.get()
        ex.import_dir(name, self.dir, ref, label_out=None if ref else ("-" if lab in ("", "(none)") else lab),
                      h5="" if h5 == NOT_EXTRACTED else h5, ap_ref=self.ap_ref() if h5 == NOT_EXTRACTED else None)
        if h5 == NOT_EXTRACTED:
            self.app.root.after(50, lambda: self.app.select(name, "extract"))
            return
        self.app.root.after(50, lambda: self.app.select(name))


class StandardizeDialog(_Dialog):
    """An .h5 made outside the app (other scripts, older runs): check what it has, add the attributes the app
    needs (simulated model, n, kappa) and create its experiment. Signals are never modified."""

    def __init__(self, app):
        from tkinter import filedialog
        tk = app.tk
        p = filedialog.askopenfilename(title=".h5 with the simulated cases", parent=app.root,
                                       filetypes=[("HDF5", "*.h5 *.hdf5"), ("all", "*.*")])
        super().__init__(app, "Standardize an .h5")
        if not p:
            self.win.destroy()
            return
        self.path = p
        self.rep = rep = ex.inspect_h5(p)
        self.field("file", tk.StringVar(value=p), width=100, state="readonly")
        t = tk.Text(self.body, width=110, height=10, font=("Consolas", 9))
        t.grid(row=self.row, column=0, columnspan=3, pady=4)
        self.row += 1
        t.tag_configure("bad", foreground="#c62828")
        t.tag_configure("ok", foreground="#2e7d32")
        if rep["problems"]:
            for x in rep["problems"]:
                t.insert("end", f"PROBLEM {x}\n", "bad")
            t.insert("end", "\nThe app reads case_000, case_001 … groups with <signal>/time and <signal>/values "
                            "(what doe_runner extract writes). Convert the file to that layout first.\n")
            self.buttons("Close")
            self.ok = False
            return
        self.ok = True
        t.insert("end", f"{len(rep['cases'])} cases, signals {', '.join(rep['signals'])}\n", "ok")
        for a in ex.STD_ATTRS:
            miss = rep["missing"][a]
            vals = [x for x in rep["values"][a] if x is not None]
            t.insert("end", f"  {a:<14s} " + ("present in every case" if not miss else
                                               f"missing in {len(miss)}/{len(rep['cases'])} cases") +
                     (f"  (e.g. {vals[0]})" if vals else "") + "\n", None if not miss else "bad")
        if rep.get("ramps"):
            t.insert("end", f"{len(rep['ramps'])} ramp case(s) (Ap_end != Ap_start): for them 'kappa' means "
                            "kappa_start and kappa_end (a single kappa is ignored)\n")
        t.configure(state="disabled")
        folder = os.path.dirname(p)
        self.section("Attributes to add (empty = leave as it is). Only these attributes are written to the file.")
        cur = lambda a: next((str(v) for v in rep["values"][a] if v is not None), "")   # noqa: E731
        self.sim_case = self.field("sim_case", tk.StringVar(value=cur("sim_case") or ex.detect_case(folder)),
                                   note="Nessy2m case (the model simulated); shown: the value in the file")
        self.sim_model = self.field("sim_model", tk.StringVar(value=cur("sim_model")), values=[""] + ex.sld_models(),
                                    editable=True, note="SLD preset of that model; also saved in the experiment(s)")
        self._orig = {"sim_case": self.sim_case.get(), "sim_model": self.sim_model.get()}
        self.spin = self.field("$spin_rate$ [rpm]", tk.StringVar(), state="normal" if rep["missing"]["$spin_rate$"] else "disabled",
                               note="only when missing" if rep["missing"]["$spin_rate$"] else "present")
        need_k = bool(rep["missing"]["kappa"]) and not rep["missing"]["$Ap_start$"]
        self.ap_mode = self.field("kappa from ap_ref", tk.StringVar(value="none"),
                                  values=["none", "manual", "model", "model_at_spin"],
                                  state="normal" if need_k else "disabled",
                                  note=("kappa = Ap / ap_ref (ramps: kappa_start and kappa_end)" if need_k else
                                        "kappa present (or no Ap to compute it)"))
        self.ap_manual = self.field("  ap_ref manual [mm]", tk.StringVar(), state="normal" if need_k else "disabled")
        self.ap_model = self.field("  SLD model", tk.StringVar(), values=ex.sld_models(),
                                   state="normal" if need_k else "disabled")
        self.section("Experiment")
        self.users = ex.experiments_using(p)
        if self.users:
            self.note(f"This file is already the data of: {', '.join(self.users)}. Usually you only want to add the "
                      "attributes (the box below is off).", "#1565c0")
        self.create = tk.BooleanVar(value=not self.users)
        ttk = app.ttk
        ttk.Checkbutton(self.body, text="also create an experiment for it (as 'Import folder…' would)",
                        variable=self.create).grid(row=self.row, column=0, columnspan=3, sticky="w")
        self.row += 1
        self.ref = self.field("reference experiment", tk.StringVar(value="(none)"), values=["(none)"] + _experiment_names())
        self.name = self.field("experiment name", tk.StringVar(value=os.path.basename(folder)), width=50,
                               note="by default the name of the folder")
        self.buttons("Apply")

    def save(self):
        from tkinter import messagebox
        if not self.ok:
            return True
        vals = {}
        for k, var in (("sim_case", self.sim_case), ("sim_model", self.sim_model)):
            v = var.get().strip()
            if v and (v != self._orig[k] or self.rep["missing"][k]):   # only what changes
                vals[k] = v
        if self.rep["missing"]["$spin_rate$"] and self.spin.get().strip():
            vals["$spin_rate$"] = float(self.spin.get())
        m = self.ap_mode.get()
        if m == "manual":
            vals["ap_ref"] = {"mode": m, "manual": float(self.ap_manual.get()) * 1e-3}
        elif m in ("model", "model_at_spin"):
            vals["ap_ref"] = {"mode": m, "model": self.ap_model.get()}
        if vals and not messagebox.askyesno(
                "Write attributes", f"Add to every case of\n{self.path}\n\n" +
                "\n".join(f"  {k} = {v}" for k, v in vals.items()) + "\n\n(signals are not touched) Continue?"):
            return False
        if vals:
            ex.add_case_attrs(self.path, vals)
        model = self.sim_model.get().strip()
        for n in self.users:   # the model is also written in the experiment file (survives a new Extract)
            ex.set_run_model(n, self.path, model or None)
        if not self.create.get():
            if self.users:
                self.app.root.after(50, lambda: self.app.select(self.users[0]))
            return True
        name = self.name.get().strip()
        ref = None if self.ref.get() in ("", "(none)") else self.ref.get()
        ex.import_dir(name, os.path.dirname(self.path), ref, h5=os.path.basename(self.path))
        ex.set_run_model(name, self.path, model or None)
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
    NL = chr(10)
    md = NL.join(["# T", "## S", "a **b** `c`", "| x | y |", "|---|---|", "| 1 | 2 |", "- item", "1. one",
                  "> note", "![cap](nope.png)", "```", "code line", "```"]) + NL
    assert os.path.isfile(TUTORIAL_FILE) and "## 3." in read_tutorial()
    imgs = [m.group(1) for m in __import__("re").finditer(r"!\[.*?\]\((.+?)\)", read_tutorial())]
    assert imgs and all(os.path.isfile(os.path.join(TUT_IMG_DIR, f)) for f in imgs), imgs
    # the window builds and draws every real experiment without errors (no mainloop)
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    try:
        app = App(root, auto_refresh=False)
        secs = render_markdown(app.tut_text, md)
        txt = app.tut_text.get("1.0", "end")
        assert secs == [("S", "sec0")] and "b c" in txt and "\u2022  item" in txt and "image missing" in txt, (secs, txt)
        assert "---" not in txt and "code line" in txt and "1. one" in txt
        app._render_tutorial()
        assert app.tut_toc.size() >= 6 and len(app._tut_images_ok()) == len(imgs)
        app._tut_zoom(1)
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
