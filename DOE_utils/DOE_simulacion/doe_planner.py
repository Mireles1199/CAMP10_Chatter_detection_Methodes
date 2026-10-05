#!/usr/bin/env python
# coding: utf-8
"""doe_planner.py — Planificador de DOE: ver QUE se va a lanzar antes de gastar horas simulando.

Elige uno o varios YAML de configs/ y muestra, sin simular nada:
  - la validacion (los mismos errores que daria doe_runner) y el resumen del DOE;
  - los casos planeados dibujados sobre el SLD (DOE_plots/sld_model.py) y, por caso, el limite de
    estabilidad a sus rpm y si queda en zona estable, inestable o cruza el limite;
  - botones para editar el YAML (la ventana se recarga sola al guardar) y para lanzar el runner.

"Lanzar" corre doe_runner.py en una consola APARTE (se queda abierta al terminar para ver el log; cerrar
el planificador no detiene la simulacion, cerrar esa consola si) con el Python correcto: el mismo con que
se abrio el planificador si trae h5py, numpy, yaml y sld_tools; si no, el de entorno_CAMP10; si tampoco,
no lanza y dice que falta. Nessy2m lo maneja el runner con su propio Python embebido.

Uso (con el Python de entorno_CAMP10):
    python doe_planner.py                  # lista configs/
    python doe_planner.py tube_ap_sweep    # con ese YAML ya seleccionado
"""
import math
import os
import subprocess
import sys

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "doe_runner.py")
ENV_PYTHON = "D:/Thesis/03-Code_Storage/02-Altintlas_Nessy2m_Storage/Env/entorno_CAMP10/Scripts/python.exe"
REQUIRED = ("h5py", "numpy", "yaml", "sld_tools")   # lo que necesita el runner (y el planificador)

try:
    import doe_runner as dr
except ImportError as exc:
    sys.exit(f"Falta un paquete en {sys.executable}: {exc}\n"
             f"Abre el planificador con el Python de entorno_CAMP10:\n  {ENV_PYTHON} doe_planner.py")


# ============================================================================== entorno y lanzamiento
def missing_modules(exe: str) -> list:
    """Paquetes de REQUIRED que le faltan a ese Python (lista vacia = tiene todo)."""
    code = "import importlib.util as u;print(','.join(n for n in %r if u.find_spec(n) is None))" % (REQUIRED,)
    try:
        r = subprocess.run([exe, "-c", code], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return [f"(no se pudo ejecutar {exe}: {exc})"]
    if r.returncode != 0:
        return [f"(error al ejecutar {exe}: {(r.stderr.strip().splitlines() or ['?'])[-1]})"]
    return [m for m in r.stdout.strip().split(",") if m]


def pick_python():
    """(exe, aviso). El Python del planificador si trae todo; si no el de entorno_CAMP10; si no, (None, motivo)."""
    miss = missing_modules(sys.executable)
    if not miss:
        return sys.executable, None
    same = os.path.abspath(ENV_PYTHON) == os.path.abspath(sys.executable)
    if os.path.isfile(ENV_PYTHON) and not same:
        miss2 = missing_modules(ENV_PYTHON)
        if not miss2:
            return ENV_PYTHON, f"{sys.executable} no tiene {miss}: se usa entorno_CAMP10"
        return None, f"faltan {miss} en {sys.executable} y {miss2} en {ENV_PYTHON}"
    return None, f"faltan {miss} en {sys.executable}"


def build_args(paths, command="n2m_sch", timed=False, dry=False, auto_extract=False, case="") -> list:
    """Argumentos de doe_runner.py. --yes: la confirmacion ya se pidio en el planificador."""
    args = ["--config"] + list(paths) + ["--yes", "--command", command]
    if timed:
        args.append("--timed")
    if dry:
        args.append("--dry-run")
    if auto_extract and command == "n2m_sch":
        args.append("--auto-extract")
    if case.strip():
        args += ["--case", case.strip()]
    return args


def build_command(py: str, args: list, runner: str = RUNNER) -> str:
    """Linea de comandos para una consola nueva que NO se cierra al terminar (cmd /k; /s: comillas seguras)."""
    return f'cmd /s /k "{subprocess.list2cmdline([py, runner] + list(args))}"'


def launch(py: str, args: list) -> str:
    """Abre doe_runner en su propia consola con el Python `py`; devuelve la linea usada."""
    cmd = build_command(py, args)
    subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE, cwd=HERE)
    return cmd


# ============================================================================== plan (sin simular)
def plan(path: str) -> dict:
    """Valida el YAML y arma lo que se lanzaria. Siempre devuelve un dict: con "error" si no es valido."""
    try:
        cfg = dr.load_config(path)
    except (ValueError, OSError, ImportError) as exc:
        return {"path": path, "name": os.path.splitext(os.path.basename(path))[0], "error": str(exc)}

    def info():
        lst, val = dr.build_doe_cases(dr.DOE_MODE)
        doe_dir = os.path.join(dr.SCRIPT_DIR, dr.DOE_NAME)
        case = cfg.get("case") or "1DOF_150Hz"
        refs = []   # AP_REF [m] de cada caso segun el ap_ref del YAML (None: sin kappa o spin fuera de los lobulos)
        for row in val:
            spin = row[lst.index("$spin_rate$")] if "$spin_rate$" in lst else None
            try:
                refs.append(dr.resolve_ap_ref(float(spin) if spin is not None else None))
            except Exception:   # model_at_spin sin spin o fuera de los lobulos; sld_tools ausente
                refs.append(None)
        return dict(text=dr.describe(doe_dir, os.path.join(dr.SCRIPT_DIR, case)), lst=lst, val=val, refs=refs,
                    doe=dr.DOE_NAME, exists=os.path.exists(doe_dir), ap_mode=dr.AP_REF_MODE,
                    ap_model=dr.AP_REF_MODEL, n2m=cfg.get("n2m_bat") or dr.DEFAULT_N2M_BAT, nb_proc=dr.NB_PROC)

    d = dr._with_config(cfg, info)
    d.update(path=path, name=os.path.splitext(os.path.basename(path))[0], cfg=cfg, error=None)
    d["cases"] = [{"var_val": dict(zip(d["lst"], row))} for row in d["val"]]
    return d


def zone(a0: float, a1: float, lim) -> str:
    """Estable / inestable / cruza el limite, segun el Ap del caso (mm) frente al limite del SLD a sus rpm."""
    if lim is None:
        return "sin lobulo (rpm fuera de rango)"
    if lim == float("inf"):
        return "estable (bolsillo)"
    lo, hi = min(a0, a1), max(a0, a1)
    if hi < lim:
        return "estable"
    if lo >= lim:
        return "inestable"
    return "cruza el limite"


def kappa_txt(a0: float, a1: float, ref) -> str:
    """kappa tal como lo guarda el extract: Ap/AP_REF (mm) truncado a 3 decimales (rampa: inicio -> fin)."""
    if ref is None or ref <= 0:
        return "—"
    k = lambda a: f"{math.trunc(a / ref * 1000) / 1000:g}"   # ref en mm; ref = inf -> kappa 0
    return k(a0) if a0 == a1 else f"{k(a0)} → {k(a1)}"


def plan_rows(plans: list, preset: str) -> list:
    """Una fila por caso: (config, n, spin, Ap ini, Ap fin, AP_REF del YAML [mm], kappa del YAML,
    limite SLD del modelo elegido [mm], Ap/limite, zona). AP_REF/kappa son los reales; limite/zona, informativos."""
    sm = dr._sld_model()
    rows = []
    for p in plans:
        if p.get("error"):
            continue
        for n, c in enumerate(p["cases"]):
            ref = p["refs"][n] * 1e3 if p["refs"][n] is not None else None
            v = c["var_val"]
            spin = v.get("$spin_rate$")
            a0, a1 = 1e3 * v.get("$Ap_start$", float("nan")), 1e3 * v.get("$Ap_end$", float("nan"))
            lim = None
            if spin is not None:
                try:
                    lim = sm.ap_lim(preset, float(spin))
                except ValueError:
                    lim = None
            ratio = (max(a0, a1) / lim) if lim not in (None, float("inf")) and lim > 0 else None
            rows.append((p["name"], n, spin, a0, a1, ref, kappa_txt(a0, a1, ref),
                         lim, ratio, zone(a0, a1, lim)))
    return rows


# ============================================================================== ventana
class PlannerApp:
    POLL_MS = 1000

    def __init__(self, root: tk.Tk, initial=()):
        self.root = root
        root.title("DOE planner")
        root.minsize(1100, 650)
        self.sm = dr._sld_model()
        self.plans: list = []
        self._mtimes: dict = {}
        self._canvas = self._toolbar = None
        self._env = pick_python()   # (exe, aviso): el Python con que se lanzaria el runner
        self._build()
        self._rescan()
        for name in initial:
            self.select_path(dr.find_config(name))
        root.after(self.POLL_MS, self._poll)

    # ------------------------------------------------------------------ construccion
    def _build(self):
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=5)
        paned.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(paned, padding=6)
        right = ttk.Frame(paned, padding=6)
        paned.add(left, minsize=150, width=170)
        paned.add(right, minsize=700)

        ttk.Label(left, text="configs/  (Ctrl/Shift: varias)", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.listbox = tk.Listbox(left, selectmode=tk.EXTENDED, exportselection=False, activestyle="none")
        self.listbox.pack(fill=tk.BOTH, expand=True, pady=4)
        self.listbox.bind("<<ListboxSelect>>", lambda _e: self.refresh())
        row = ttk.Frame(left)
        row.pack(fill=tk.X)
        ttk.Button(row, text="↻", width=3, command=self._rescan).pack(side=tk.LEFT)
        ttk.Button(row, text="Otro YAML…", command=self._open_other).pack(side=tk.LEFT, padx=4)

        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)
        box = ttk.Frame(right)   # resumen con barra de desplazamiento: con varios DOE no cabe entero
        box.grid(row=0, column=0, sticky="ew")
        self.text = tk.Text(box, height=11, wrap="word", font=("Consolas", 9), state="disabled")
        sb_txt = ttk.Scrollbar(box, orient=tk.VERTICAL, command=self.text.yview)
        self.text.configure(yscrollcommand=sb_txt.set)
        self.text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        sb_txt.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.tag_configure("err", foreground="#c62828")
        self.text.tag_configure("head", font=("Consolas", 9, "bold"))
        self.text.tag_configure("warn", foreground="#b26a00")
        self.text.tag_configure("ok", foreground="#2e7d32")

        mid = tk.PanedWindow(right, orient=tk.HORIZONTAL, sashwidth=5)
        mid.grid(row=1, column=0, sticky="nsew", pady=6)
        self.fig_frame = ttk.Frame(mid)
        tab = ttk.Frame(mid)
        mid.add(self.fig_frame, minsize=300, width=400)
        mid.add(tab, minsize=420)
        cols = ("cfg", "n", "spin", "ap", "ref", "kappa", "lim", "ratio", "zona")
        self.table = ttk.Treeview(tab, columns=cols, show="headings")
        # AP_REF y kappa: lo REAL (ap_ref del YAML, lo que guardará el extract). límite/Ap-límite/zona: informativo,
        # con el modelo del desplegable "Modelo SLD".
        for c, h, w in (("cfg", "config", 70), ("n", "#", 26), ("spin", "spin [rpm]", 62), ("ap", "Ap [mm]", 72),
                        ("ref", "AP_REF [mm] (YAML)", 122), ("kappa", "kappa (YAML)", 92),
                        ("lim", "límite SLD [mm] (info)", 135), ("ratio", "Ap/límite (info)", 100),
                        ("zona", "zona (info)", 125)):
            self.table.heading(c, text=h)
            self.table.column(c, width=w, minwidth=30, anchor=tk.W if c in ("cfg", "zona") else tk.CENTER)
        sb = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=self.table.yview)
        sbx = ttk.Scrollbar(tab, orient=tk.HORIZONTAL, command=self.table.xview)
        self.table.configure(yscrollcommand=sb.set, xscrollcommand=sbx.set)
        sbx.pack(side=tk.BOTTOM, fill=tk.X)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.table.tag_configure("estable", background="#e3f2fd")
        self.table.tag_configure("inestable", background="#fff3e0")
        self.table.tag_configure("cruza", background="#fff9c4")

        bar = ttk.Frame(right)
        bar.grid(row=2, column=0, sticky="ew")
        ttk.Label(bar, text="Modelo SLD:").pack(side=tk.LEFT)
        self.preset = tk.StringVar(value=next(iter(self.sm.MODELS)))
        cb = ttk.Combobox(bar, textvariable=self.preset, values=list(self.sm.MODELS), state="readonly", width=16)
        cb.pack(side=tk.LEFT, padx=(2, 12))
        cb.bind("<<ComboboxSelected>>", lambda _e: self.refresh(keep_preset=True))
        ttk.Button(bar, text="✎  Editar YAML", command=self.edit).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="💾  Export…", command=self.export).pack(side=tk.LEFT, padx=4)

        self.command = tk.StringVar(value="n2m_sch")
        self.timed, self.dry, self.auto = tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar()
        self.case = tk.StringVar()
        opt = ttk.Frame(right)
        opt.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(opt, text="Comando:").pack(side=tk.LEFT)
        ttk.Combobox(opt, textvariable=self.command, values=["n2m_sch", "extract"], state="readonly",
                     width=9).pack(side=tk.LEFT, padx=(2, 10))
        for txt, var in (("--timed", self.timed), ("--dry-run", self.dry), ("--auto-extract", self.auto)):
            ttk.Checkbutton(opt, text=txt + "  ", variable=var).pack(side=tk.LEFT, padx=4)   # espacios: Tk recorta el texto con DPI alto
        ttk.Label(opt, text="  caso:").pack(side=tk.LEFT)
        ttk.Entry(opt, textvariable=self.case, width=12).pack(side=tk.LEFT, padx=2)
        self.btn_launch = ttk.Button(opt, text="▶  Lanzar…", command=self.on_launch)
        self.btn_launch.pack(side=tk.RIGHT, padx=4)

    # ------------------------------------------------------------------ lista y recarga
    def _files(self) -> list:
        d = dr.CONFIGS_DIR
        base = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".yaml")) if os.path.isdir(d) else []
        extra = [p for p in getattr(self, "_extra", []) if p not in base]
        return base + extra

    def _rescan(self):
        keep = {self.listbox.get(i) for i in self.listbox.curselection()}
        self.files = self._files()
        self.listbox.delete(0, tk.END)
        for p in self.files:
            self.listbox.insert(tk.END, os.path.splitext(os.path.basename(p))[0])
        for i in range(self.listbox.size()):
            if self.listbox.get(i) in keep:
                self.listbox.selection_set(i)
        self._mtimes = self._stat()
        self.refresh()

    def _stat(self) -> dict:
        return {p: os.path.getmtime(p) for p in self._files() if os.path.exists(p)}

    def _poll(self):
        """Recarga sola si se guardo algun YAML de configs/ (incluye base.yaml, que heredan los demas)."""
        now = self._stat()
        if now != self._mtimes:
            self._rescan()
        self.root.after(self.POLL_MS, self._poll)

    def select_path(self, path: str):
        if path not in self.files:
            self._extra = getattr(self, "_extra", []) + [path]
            self._rescan()
        i = self.files.index(path)
        self.listbox.selection_set(i)
        self.refresh()

    def _open_other(self):
        f = filedialog.askopenfilename(parent=self.root, title="YAML de configuracion",
                                       filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")],
                                       initialdir=dr.CONFIGS_DIR)
        if f:
            self.select_path(os.path.abspath(f))

    def selected_paths(self) -> list:
        return [self.files[i] for i in self.listbox.curselection()]

    # ------------------------------------------------------------------ vista
    def _write(self, parts):
        self.text.config(state="normal")
        self.text.delete("1.0", tk.END)
        for txt, tag in parts:
            self.text.insert(tk.END, txt, tag)
        self.text.config(state="disabled")

    def _env_parts(self) -> list:
        """Primeras lineas del panel: con que Python se lanzaria el runner y si sirve."""
        exe, note = self._env
        if exe is None:
            return [("ENTORNO   ✖ NO se podrá lanzar: " + note + "\n", "err"), ("\n", "")]
        parts = [(f"ENTORNO   Python: {exe}\n", "head"),
                 (f"          ✔ tiene {', '.join(REQUIRED)}\n", "ok")]
        if note:
            parts.append((f"          ⚠ {note}\n", "warn"))
        parts.append(("          (Nessy2m lo maneja el runner con su propio Python embebido)\n\n", ""))
        return parts

    def refresh(self, keep_preset: bool = False):
        self.plans = [plan(p) for p in self.selected_paths()]
        ok = [p for p in self.plans if not p.get("error")]
        if ok and not keep_preset:   # modelo de la config si usa uno; si no, se respeta el elegido
            for p in ok:
                if p["ap_mode"] in ("model", "model_at_spin") and p["ap_model"] in self.sm.MODELS:
                    self.preset.set(p["ap_model"])
                    break
        parts = self._env_parts()
        if not self.plans:
            parts.append(("Elige uno o varios YAML de la lista para ver qué se lanzaría.\n", ""))
        for p in self.plans:
            parts.append((f"[{p['name']}]\n", "head"))
            if p.get("error"):
                parts.append((p["error"] + "\n", "err"))
            else:
                parts.append((p["text"] + "\n", "warn" if p["exists"] else ""))
                nb, n = p["nb_proc"], len(p["cases"])
                hilos = os.cpu_count() or 0
                parts.append((f"  Procesos  : nb_proc = {nb}  ({min(nb, n)} caso(s) a la vez; {n} caso(s) en {-(-n // nb)} tanda(s); "
                              f"este equipo tiene {hilos} hilos)" + ("  ⚠ más procesos que hilos" if hilos and nb > hilos else "") + "\n",
                              "warn" if hilos and nb > hilos else ""))
                found = os.path.isfile(p["n2m"])
                parts.append((f"  Nessy2m   : {p['n2m']}  " + ("(existe)" if found else "(NO EXISTE: el runner fallará)") + "\n",
                              "" if found else "err"))
            parts.append(("\n", ""))
        self._write(parts)

        rows = plan_rows(self.plans, self.preset.get())
        self.table.delete(*self.table.get_children())
        for name, n, spin, a0, a1, ref, kap, lim, ratio, z in rows:
            ap = f"{a0:g}" if a0 == a1 else f"{a0:g} → {a1:g}"
            reftxt = "—" if ref is None else ("∞" if ref == float("inf") else f"{ref:.4f}")
            limtxt = "—" if lim is None else ("∞" if lim == float("inf") else f"{lim:.3f}")
            tag = "estable" if z.startswith("estable") else "inestable" if z == "inestable" else "cruza"
            self.table.insert("", tk.END, values=(name, n, "—" if spin is None else f"{spin:g}", ap, reftxt, kap,
                                                  limtxt, "—" if ratio is None else f"{ratio:.2f}", z), tags=(tag,))
        self._draw([c for p in ok for c in p["cases"]])

    def _draw(self, cases):
        for w in (self._canvas.get_tk_widget() if self._canvas else None, self._toolbar):
            if w is not None:
                w.destroy()
        if cases:
            fig = self.sm.plot_sld(cases, self.preset.get())
            plt.close(fig)   # se desacopla de pyplot: lo dibuja el canvas de Tk
            fig.suptitle(f"{len(cases)} caso(s) planeado(s) sobre el SLD", fontsize=10)
        else:
            from matplotlib.figure import Figure
            fig = Figure(constrained_layout=True)
            fig.add_subplot(111).text(0.5, 0.5, "Sin casos que mostrar", ha="center", va="center")
        self.fig = fig
        self._canvas = FigureCanvasTkAgg(fig, master=self.fig_frame)
        self._toolbar = NavigationToolbar2Tk(self._canvas, self.fig_frame, pack_toolbar=False)
        self._toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._canvas.draw()

    # ------------------------------------------------------------------ acciones
    def export(self):
        """Export… (DOE_plots/figures_window.py): the figure on screen, as a copy, with size, language, dpi, format."""
        from figures_window import FiguresWindow, Item   # DOE_plots is on sys.path (doe_runner._sld_model)
        FiguresWindow(self.root, "Export — doe_planner",
                      [Item(f"SLD_plan_{self.preset.get()}", lambda: getattr(self, "fig", None), live=True)])

    def edit(self):
        paths = self.selected_paths()
        if not paths:
            messagebox.showinfo("Editar", "Elige un YAML de la lista.", parent=self.root)
            return
        for p in paths:
            try:
                os.startfile(p)
            except OSError:   # sin programa asociado a .yaml
                subprocess.Popen(["notepad.exe", p])

    def on_launch(self):
        paths = self.selected_paths()
        bad = [p["name"] for p in self.plans if p.get("error")]
        if not paths:
            messagebox.showinfo("Lanzar", "Elige al menos un YAML de la lista.", parent=self.root)
            return
        if bad:
            messagebox.showerror("Config inválida", f"Corrige antes de lanzar: {', '.join(bad)}\n(el detalle está arriba, en rojo)",
                                 parent=self.root)
            return
        py, aviso = pick_python()
        if py is None:
            messagebox.showerror("Entorno", f"No se lanza: {aviso}.\nAbre el planificador con entorno_CAMP10.", parent=self.root)
            return
        cmd, dry = self.command.get(), self.dry.get()
        lines = []
        for p in self.plans:
            borra = p["exists"] and cmd == "n2m_sch" and not dry
            lines.append(f"• {p['doe']}  ({len(p['cases'])} caso(s))" + ("   ⚠ YA EXISTE: SE BORRARÁ" if borra else ""))
        msg = (f"Comando: {cmd}{'  (dry-run)' if dry else ''}{'  --timed' if self.timed.get() else ''}\n\n"
               + "\n".join(lines) + f"\n\nPython: {py}" + (f"\n{aviso}" if aviso else "")
               + "\n\nSe abre una consola aparte con el log (cerrarla detiene la simulación). ¿Lanzar?")
        if not messagebox.askyesno("Lanzar DOE", msg, parent=self.root, icon="warning"):
            return
        launch(py, build_args(paths, cmd, self.timed.get(), dry, self.auto.get(), self.case.get()))


def main():
    root = tk.Tk()
    try:
        root.state("zoomed")
    except tk.TclError:
        pass
    PlannerApp(root, sys.argv[1:])
    root.mainloop()


if __name__ == "__main__":
    main()
