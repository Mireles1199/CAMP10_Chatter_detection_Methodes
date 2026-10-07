"""figures_window.py — the one export window of every viewer (PLAN_figuras.md): any figure of any panel, saved with the
controls of article-plot-style — size (the figure's own, FIGSIZE_SIMPLE, FIGSIZE_WIDE or a grid of them) x scale,
language EN / FR / both, dpi, format, folder and name — with the article proportions kept on screen.

It knows nothing about the figures. Each element is an `Item(name, render, native, live)`:
  - render() -> Figure (raises if the figure has no data);
  - native: the figure draws its own text in the chosen language and scale (validation_figures, sld_model:
    `style(language, scale, follow_text)` calls their set_style); the others are translated on the exported copy by
    fig_lang.translate_figure and, at a preset size, their text follows the scale (fig_lang.scale_text);
  - live: render() gives the figure of a panel on screen; an independent copy is exported (selection and zoom kept, the
    panel is not touched);
  - folder: where this item is saved (absolute), instead of the folder of the window (Save and Save all);
  - note: a text shown under the figure (e.g. a limitation of the language options for it);
  - skip_all: an alternative view of another item: it can be selected and saved, but 'Save all' leaves it out.
Read-only apart from the saved figures.

    python figures_window.py --selftest
"""
import os
import pickle
import re
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, NamedTuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_style as ps  # noqa: E402

LANGUAGES = ("EN", "FR", "both")
FORMATS = ("png", "pdf", "svg")
DPIS = ("200", "300", "600")   # 200 intermediate, 300 final (article-plot-style, sec. 10)
SIZES = ("own", "SIMPLE", "WIDE", "grid")   # own = the figure's article size, or a panel as it is on screen
# text: 'follows scale' = letters, lines and markers grow with the scale like a zoom (as at the default 1.5); 'fixed' = the
# points of the article-plot-style skill at any scale (PLAN_plot_style.md §3)
TEXT_MODES = ("follows scale", "fixed")


class Item(NamedTuple):
    name: str
    render: Callable
    native: bool = False
    live: bool = False
    folder: str = ""
    note: str = ""
    skip_all: bool = False   # an alternative view of another item (e.g. two of the three signals): selectable, not in Save all


def copy_figure(fig):
    """Independent copy of a live figure (selection, zoom and size kept); the panel keeps its own."""
    return pickle.loads(pickle.dumps(fig))


def translate(fig, language: str) -> list:
    """Translate the texts of `fig` (fig_lang); the texts with no translation are returned."""
    try:
        from fig_lang import translate_figure
    except ImportError:
        return []
    return translate_figure(fig, language)


def file_name(label: str, fmt: str = "") -> str:
    stem = re.sub(r"[^\w\-]+", "_", label).strip("_")
    return stem + ("." + fmt if fmt else "")


def layout(fig) -> tuple:
    """(columns, rows) of the grid of axes of a figure (colorbars and insets not counted); (1, 1) if unknown."""
    geos = []
    for ax in fig.axes:
        spec = ax.get_subplotspec() if hasattr(ax, "get_subplotspec") else None
        if spec is not None:
            rows, cols = spec.get_gridspec().get_geometry()
            geos.append((cols, rows))
    return max(geos, key=lambda g: g[0] * g[1]) if geos else (1, 1)


def target_size(fig, size: str, scale: float, grid=(2, 1)) -> tuple:
    """Size in inches for the export: a plot_style preset x scale, or the figure's own (_keep_size, else as it is)."""
    if size == "SIMPLE":
        return ps.figsize_from_scale(ps.FIGSIZE_SIMPLE, scale)
    if size == "WIDE":
        return ps.figsize_from_scale(ps.FIGSIZE_WIDE, scale)
    if size == "grid":
        return ps.figsize_from_scale(ps.figsize_grid(*grid), scale)
    keep = getattr(fig, "_keep_size", None)
    return tuple(float(v) for v in (keep if keep else fig.get_size_inches()))


class FiguresWindow:
    def __init__(self, parent, title: str, items: list, render=None, style=None, out_dir: str = "",
                 language: str = "EN", scale: float = 1.5, select: str | None = None):
        # items: Item, or names drawn by render(name) (native: the older call of the viewer)
        self.items = [it if isinstance(it, Item) else Item(it, (lambda n=it: render(n)), native=True) for it in items]
        self.labels = [it.name for it in self.items]
        self.style = style or (lambda language, scale, follow_text=True: None)
        self.fig = self.canvas = self.toolbar = None
        self.missing = []
        self.win = win = tk.Toplevel(parent)
        win.title(title)
        win.geometry("1360x880")
        v = lambda x: tk.StringVar(value=x)   # noqa: E731
        self.lang, self.scale, self.dpi, self.fmt = v(language), v(f"{scale:g}"), v("300"), v("png")
        self.text = v(TEXT_MODES[0])
        self.size, self.gcols, self.grows = v("own"), v("2"), v("1")
        self.folder, self.name = v(out_dir), v("")
        self.keep = tk.BooleanVar(value=True)
        self.status = v("")

        row1 = ttk.Frame(win, padding=(6, 4, 6, 0))
        row1.pack(side=tk.TOP, fill=tk.X)
        row2 = ttk.Frame(win, padding=(6, 2, 6, 4))
        row2.pack(side=tk.TOP, fill=tk.X)

        def add(bar, text, widget, redraw=True):
            if text:
                ttk.Label(bar, text=text).pack(side=tk.LEFT, padx=(10, 2))
            widget.pack(side=tk.LEFT)
            if redraw and isinstance(widget, ttk.Combobox):
                widget.bind("<<ComboboxSelected>>", lambda _e: self.draw())
            if redraw and isinstance(widget, ttk.Spinbox):
                widget.bind("<Return>", lambda _e: self.draw())
            return widget
        add(row1, "size", ttk.Combobox(row1, textvariable=self.size, values=SIZES, state="readonly", width=8))
        add(row1, "grid", ttk.Spinbox(row1, textvariable=self.gcols, from_=1, to=4, width=3, command=self.draw))
        add(row1, "×", ttk.Spinbox(row1, textvariable=self.grows, from_=1, to=4, width=3, command=self.draw))
        add(row1, "scale", ttk.Spinbox(row1, textvariable=self.scale, from_=0.5, to=3.0, increment=0.25, width=7,
                                       command=self.draw))
        add(row1, "language", ttk.Combobox(row1, textvariable=self.lang, values=LANGUAGES, state="readonly", width=8))
        add(row1, "text", ttk.Combobox(row1, textvariable=self.text, values=TEXT_MODES, state="readonly", width=13))
        ttk.Checkbutton(row1, text="article proportions", width=19, variable=self.keep,
                        command=self.fit).pack(side=tk.LEFT, padx=10)
        add(row2, "dpi", ttk.Combobox(row2, textvariable=self.dpi, values=DPIS, state="readonly", width=7), False)
        add(row2, "format", ttk.Combobox(row2, textvariable=self.fmt, values=FORMATS, state="readonly", width=7), False)
        add(row2, "folder", ttk.Entry(row2, textvariable=self.folder, width=48), False)
        ttk.Button(row2, text="…", width=3, command=self.browse).pack(side=tk.LEFT, padx=2)
        add(row2, "name", ttk.Entry(row2, textvariable=self.name, width=30), False)
        ttk.Button(row2, text="Save", command=self.save).pack(side=tk.LEFT, padx=(14, 2))
        ttk.Button(row2, text="Save all", command=self.save_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="Open folder", command=self.open_folder).pack(side=tk.LEFT, padx=2)

        body = ttk.Panedwindow(win, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(body)
        body.add(left, weight=1)
        self.lb = tk.Listbox(left, exportselection=False, activestyle="none", width=36)
        sb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.lb.yview)
        self.lb.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        for lab in self.labels:
            self.lb.insert(tk.END, lab)
        self.lb.bind("<<ListboxSelect>>", lambda _e: self.draw())
        right = ttk.Frame(body)
        body.add(right, weight=5)
        self.tbar = ttk.Frame(right)
        self.tbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(right, textvariable=self.status, foreground="#555", wraplength=1000,
                  justify="left").pack(side=tk.BOTTOM, fill=tk.X)
        self.holder = tk.Frame(right, background="white")
        self.holder.pack(fill=tk.BOTH, expand=True)
        self.holder.bind("<Configure>", lambda _e: self.fit())
        if self.items:
            i = self.labels.index(select) if select in self.labels else 0
            self.lb.selection_set(i)
            self.lb.see(i)
            win.after(100, self.draw)

    # ------------------------------------------------------------------ figures
    def select(self, name: str | None):
        """Bring the window up with `name` selected (if it is in the list) and draw it."""
        self.win.deiconify()
        self.win.lift()
        if name in self.labels:
            i = self.labels.index(name)
            self.lb.selection_clear(0, tk.END)
            self.lb.selection_set(i)
            self.lb.see(i)
            self.draw()

    def current(self):
        sel = self.lb.curselection()
        return self.items[sel[0]] if sel else None

    def make(self, item: Item):
        """Figure of `item` with the controls applied (language, size x scale); raises if it cannot be made."""
        try:
            scale = float(self.scale.get())
            grid = (int(self.gcols.get()), int(self.grows.get()))
        except ValueError:
            raise ValueError("scale and grid must be numbers (scale 1 = the article preset, 1.5 by default)")
        lang, follow = self.lang.get(), self.text.get() != "fixed"
        self.style(lang, scale, follow)
        fig = item.render()
        if fig is None:
            raise ValueError("nothing to export yet (draw this panel first)")
        if item.live:
            try:
                fig = copy_figure(fig)
            except Exception as exc:   # something in the panel cannot be copied (e.g. a local function)
                raise ValueError(f"this panel cannot be copied for export: {exc}") from exc
        else:
            plt.close(fig)   # detached from pyplot's windows; the object stays
            FigureCanvasAgg(fig)   # its pyplot canvas was a Tk widget, destroyed by close: resizing it would fail
        self.missing = [] if item.native else translate(fig, lang)
        if not item.native and self.size.get() != "own":   # a preset x scale: its text follows (or not) like a native one
            from fig_lang import scale_text
            scale_text(fig, ps.zoom(scale, follow))
        size = target_size(fig, self.size.get(), scale, grid)
        if abs(fig.get_size_inches()[0] - size[0]) > 1e-6 or abs(fig.get_size_inches()[1] - size[1]) > 1e-6:
            fig.set_size_inches(*size)
            if fig.get_layout_engine() is None:   # a panel laid out by hand: re-flowed for the new size
                fig.set_layout_engine("constrained")
        fig._keep_size = size
        fig.canvas.draw()   # errors that only show when drawing (e.g. a log axis without positive data) are raised here
        return fig

    def draw(self):
        item = self.current()
        if item is None:
            return
        if not self.name.get() or self.name.get() == getattr(self, "_auto_name", None):
            self._auto_name = file_name(item.name)
            self.name.set(self._auto_name)
        for w in list(self.holder.winfo_children()) + list(self.tbar.winfo_children()):
            w.destroy()
        self.fig = self.canvas = None
        try:
            self.fig = self.make(item)
            if item is not getattr(self, "_last_item", None):   # a new figure: the grid follows its layout of axes
                self._last_item = item
                geo = layout(self.fig)
                if geo != (int(self.gcols.get()), int(self.grows.get())):
                    self.gcols.set(str(geo[0]))
                    self.grows.set(str(geo[1]))
                    if self.size.get() == "grid":
                        self.fig = self.make(item)
        except Exception as exc:   # a figure without data raises: shown here
            self.status.set(f"{item.name}: {type(exc).__name__}: {exc}")
            return
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.holder)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.tbar, pack_toolbar=False)
        self.toolbar.pack(side=tk.LEFT)
        self.fit()
        w, h = self.fig._keep_size
        self.status.set(f"{item.name}   ·   exported at {w:.2f} x {h:.2f} in, {self.dpi.get()} dpi"
                        + (f"   ·   no translation for: {'; '.join(self.missing[:8])}" if self.missing else "")
                        + (f"\n{item.note}" if item.note else ""))

    def fit(self):
        """Place the canvas in the free space: with 'article proportions' at the aspect of the export, else filling it."""
        if self.fig is None or self.canvas is None:
            return
        W, H = self.holder.winfo_width(), self.holder.winfo_height()
        if W < 20 or H < 20:
            return
        keep = getattr(self.fig, "_keep_size", None) if self.keep.get() else None
        if keep:
            s = min(W / keep[0], H / keep[1])
            w, h = int(keep[0] * s), int(keep[1] * s)
        else:
            w, h = W, H
        self.canvas.get_tk_widget().place(x=(W - w) // 2, y=(H - h) // 2, width=w, height=h)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ saving
    def save_fig(self, fig, name: str, folder: str = "") -> str:
        folder = folder or self.folder.get().strip()
        if not folder:
            raise ValueError("choose a folder")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, file_name(name, self.fmt.get()))
        shown = fig.get_size_inches().copy()
        keep = getattr(fig, "_keep_size", None)
        if keep:   # the Tk canvas fits the figure to the widget: the file goes out at the export size
            fig.set_size_inches(*keep, forward=False)
        try:
            fig.savefig(path, dpi=int(self.dpi.get()), bbox_inches="tight")
        finally:
            fig.set_size_inches(*shown, forward=False)
        return path

    def save(self):
        if self.fig is None:
            messagebox.showinfo("Save", "There is no figure to save.", parent=self.win)
            return
        try:
            path = self.save_fig(self.fig, self.name.get() or self.current().name, self.current().folder)
        except Exception as exc:
            messagebox.showerror("Error saving", str(exc), parent=self.win)
            return
        self.status.set(f"saved {path}")

    def save_all(self):
        done, failed, missing, saved = 0, [], set(), set()
        for it in self.items:
            if it.skip_all:
                continue
            try:
                saved.add(os.path.dirname(self.save_fig(self.make(it), it.name, it.folder)))
                done += 1
                missing.update(self.missing)
            except Exception as exc:   # e.g. a figure without data: reported, the others are saved
                failed.append(f"{it.name}: {type(exc).__name__}: {exc}")
        messagebox.showinfo("Save all", f"{done} figures saved in\n" + "\n".join(sorted(saved) or [self.folder.get()])
                            + ("\n\nNot saved:\n" + "\n".join(failed) if failed else "")
                            + ("\n\nTexts with no translation (fig_lang table):\n" + "\n".join(sorted(missing)[:40])
                               if missing else ""), parent=self.win)

    def browse(self):
        d = filedialog.askdirectory(parent=self.win, initialdir=self.folder.get() or None)
        if d:
            self.folder.set(d)

    def open_folder(self):
        os.makedirs(self.folder.get(), exist_ok=True)
        os.startfile(self.folder.get())


def _selftest():
    import tempfile
    import matplotlib
    matplotlib.use("TkAgg")
    import numpy as np
    from matplotlib.figure import Figure
    root = tk.Tk()
    root.withdraw()
    msgs = []
    for n in ("showinfo", "showerror", "showwarning"):
        setattr(messagebox, n, lambda t, x, parent=None: msgs.append(x))
    # a panel on screen (embedded, zoomed) and a generated figure
    top = tk.Toplevel(root)
    live = Figure(figsize=(8, 5))
    ax = live.add_subplot(111)
    ax.plot(np.arange(100), np.sin(np.arange(100) / 5))
    ax.set_xlim(10, 40)
    ax.set_xlabel("Time (s)")
    FigureCanvasTkAgg(live, master=top).draw()

    def gen():
        f, a = plt.subplots(figsize=ps.figsize_from_scale(ps.FIGSIZE_SIMPLE, 1.5), constrained_layout=True)
        a.plot([0, 1], [0, 1])
        f._keep_size = tuple(f.get_size_inches())
        return f

    def bad():
        raise ValueError("no data")
    d = tempfile.mkdtemp(prefix="figwin_")
    w = FiguresWindow(root, "test", [Item("panel", lambda: live, live=True), Item("generated", gen, native=True),
                                     Item("empty", bad)], out_dir=d, select="generated")
    w.draw()
    assert w.current().name == "generated" and w.name.get() == "generated"
    f = w.make(w.items[0])                                         # panel: an independent copy, zoom kept
    assert f is not live and f.axes[0].get_xlim() == (10.0, 40.0) and tuple(f._keep_size) == (8.0, 5.0)
    w.lang.set("FR")                                               # fixed text translated on the copy only
    f = w.make(w.items[0])
    assert f.axes[0].get_xlabel() == "Temps (s)" and live.axes[0].get_xlabel() == "Time (s)"
    w.lang.set("EN")
    w.size.set("SIMPLE")
    w.scale.set("1")
    f = w.make(w.items[0])
    assert tuple(f._keep_size) == ps.FIGSIZE_SIMPLE and tuple(live.get_size_inches()) != ps.FIGSIZE_SIMPLE
    # 'text': follows the scale (default: as today at 1.5, x2 at 3) or fixed; native figures get it through style()
    calls = []
    w.style = lambda language, scale, follow_text=True: calls.append(follow_text)
    fs = lambda s, mode: (w.scale.set(s), w.text.set(mode), w.make(w.items[0]).axes[0].xaxis.label.get_fontsize())[2]   # noqa: E731
    base = fs("1.5", TEXT_MODES[0])
    assert fs("3", TEXT_MODES[0]) == 2 * base and fs("3", "fixed") == base and calls[-2:] == [True, False], calls
    assert live.axes[0].xaxis.label.get_fontsize() == base                    # the panel on screen is not touched
    w.size.set("own")
    assert fs("3", TEXT_MODES[0]) == base                                     # own size: the canvas does not grow, nor the text
    w.text.set(TEXT_MODES[0])
    w.scale.set("1")
    w.size.set("grid")
    w.gcols.set("2")
    w.grows.set("2")
    assert tuple(w.make(w.items[1])._keep_size) == ps.figsize_grid(2, 2)
    stacked = Figure()
    stacked.subplots(3, 1)
    assert layout(stacked) == (1, 3) and layout(Figure()) == (1, 1)   # the grid follows the axes of the panel
    w.size.set("WIDE")
    w.fmt.set("pdf")
    w.save_all()
    assert sorted(os.listdir(d)) == ["generated.pdf", "panel.pdf"], os.listdir(d)
    assert "empty: ValueError: no data" in msgs[-1]
    w.lb.selection_clear(0, tk.END)
    w.lb.selection_set(2)
    w.draw()
    assert w.fig is None and "no data" in w.status.get()
    d2 = tempfile.mkdtemp(prefix="figwin2_")   # an item with its own folder and note (the indicator figures)
    w2 = FiguresWindow(root, "test2", [Item("fixed", gen, native=True, folder=d2, note="English only")], out_dir=d)
    w2.draw()
    assert "English only" in w2.status.get()
    w2.save_all()
    assert os.listdir(d2) == ["fixed.png"] and "fixed.png" not in os.listdir(d) and d2 in msgs[-1]
    root.destroy()
    print("figures_window selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
