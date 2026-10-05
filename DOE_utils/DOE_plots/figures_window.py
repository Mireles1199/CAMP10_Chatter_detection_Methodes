"""figures_window.py — window of the viewer for the figures that are not reference curves (validation, convergence,
I(t) summaries...). The right panel of the viewer keeps the reference curves (SLD, outcome per indicator); this window
gives them room and the controls of article-plot-style: language EN / FR / both, scale (FIGSCALE x the FIGSIZE_*
presets), dpi and format of the saved file, and the article proportions kept on screen.

It knows nothing about the figures: the viewer passes `render(label) -> Figure` (raises if the figure has no data) and
`style(language, scale)` (sets LANGUAGE / FIGSCALE of the figure modules). Read-only apart from the saved figures.
"""
import os
import re
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

LANGUAGES = ("EN", "FR", "both")
FORMATS = ("png", "pdf", "svg")
DPIS = ("200", "300", "600")   # 200 intermediate, 300 final (article-plot-style, sec. 10)


def file_name(label: str, fmt: str) -> str:
    return re.sub(r"[^\w\-]+", "_", label).strip("_") + "." + fmt


class FiguresWindow:
    def __init__(self, parent, title: str, labels: list, render, style, out_dir: str, language: str = "EN",
                 scale: float = 1.5):
        self.labels, self.render, self.style, self.out_dir = labels, render, style, out_dir
        self.fig = self.canvas = self.toolbar = None
        self.win = win = tk.Toplevel(parent)
        win.title(title)
        win.geometry("1300x860")
        v = lambda x: tk.StringVar(value=x)   # noqa: E731
        self.lang, self.scale, self.dpi, self.fmt = v(language), v(f"{scale:g}"), v("300"), v("png")
        self.keep = tk.BooleanVar(value=True)
        self.status = v("")

        bar = ttk.Frame(win, padding=(6, 4))
        bar.pack(side=tk.TOP, fill=tk.X)

        def box(text, _var, widget):
            ttk.Label(bar, text=text).pack(side=tk.LEFT, padx=(10, 2))
            widget.pack(side=tk.LEFT)
        lg = ttk.Combobox(bar, textvariable=self.lang, values=LANGUAGES, state="readonly", width=8)
        lg.bind("<<ComboboxSelected>>", lambda _e: self.draw())
        box("language", self.lang, lg)
        sc = ttk.Spinbox(bar, textvariable=self.scale, from_=0.5, to=3.0, increment=0.25, width=7, command=self.draw)
        sc.bind("<Return>", lambda _e: self.draw())
        box("scale", self.scale, sc)
        ttk.Checkbutton(bar, text="article proportions", width=19, variable=self.keep, command=self.fit).pack(side=tk.LEFT, padx=10)
        box("dpi", self.dpi, ttk.Combobox(bar, textvariable=self.dpi, values=DPIS, state="readonly", width=7))
        box("format", self.fmt, ttk.Combobox(bar, textvariable=self.fmt, values=FORMATS, state="readonly", width=7))
        ttk.Button(bar, text="Save", command=self.save).pack(side=tk.LEFT, padx=(14, 2))
        ttk.Button(bar, text="Save all", command=self.save_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Open folder", command=self.open_folder).pack(side=tk.LEFT, padx=2)

        body = ttk.Panedwindow(win, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(body)
        body.add(left, weight=1)
        self.lb = tk.Listbox(left, exportselection=False, activestyle="none", width=34)
        sb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.lb.yview)
        self.lb.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        for lab in labels:
            self.lb.insert(tk.END, lab)
        self.lb.bind("<<ListboxSelect>>", lambda _e: self.draw())
        right = ttk.Frame(body)
        body.add(right, weight=5)
        self.tbar = ttk.Frame(right)
        self.tbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(right, textvariable=self.status, foreground="#555").pack(side=tk.BOTTOM, fill=tk.X)
        self.holder = tk.Frame(right, background="white")
        self.holder.pack(fill=tk.BOTH, expand=True)
        self.holder.bind("<Configure>", lambda _e: self.fit())
        if labels:
            self.lb.selection_set(0)
            win.after(100, self.draw)

    # ------------------------------------------------------------------ drawing
    def current(self):
        sel = self.lb.curselection()
        return self.labels[sel[0]] if sel else None

    def make(self, label: str):
        """Figure of `label` with the language / scale of the controls; raises if it cannot be made."""
        try:
            scale = float(self.scale.get())
        except ValueError:
            raise ValueError("scale must be a number (1 = the article preset, 1.5 by default)")
        self.style(self.lang.get(), scale)
        fig = self.render(label)
        if fig is None:
            raise ValueError("no figure")
        plt.close(fig)   # detached from pyplot's windows; the object stays
        fig.canvas.draw()   # errors that only show when drawing (e.g. a log axis without positive data) are raised here
        return fig

    def draw(self):
        label = self.current()
        if label is None:
            return
        for w in list(self.holder.winfo_children()) + list(self.tbar.winfo_children()):
            w.destroy()
        self.fig = self.canvas = None
        try:
            self.fig = self.make(label)
        except Exception as exc:   # a figure without data raises: shown here, like the right panel shows it
            self.status.set(f"{label}: {type(exc).__name__}: {exc}")
            return
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.holder)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.tbar, pack_toolbar=False)
        self.toolbar.pack(side=tk.LEFT)
        self.fit()
        self.status.set(f"{label}   ·   figure {self.fig.get_size_inches()[0]:.2f} x {self.fig.get_size_inches()[1]:.2f} in"
                        f"   ·   saved at {self.dpi.get()} dpi into {self.out_dir}")

    def fit(self):
        """Place the canvas in the free space: with 'article proportions' at the aspect of the figure, else filling it."""
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
    def save_fig(self, fig, label: str) -> str:
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, file_name(label, self.fmt.get()))
        shown = fig.get_size_inches().copy()
        keep = getattr(fig, "_keep_size", None)
        if keep:   # the Tk canvas fits the figure to the widget: the file goes out at the article size
            fig.set_size_inches(*keep, forward=False)
        try:
            fig.savefig(path, dpi=int(self.dpi.get()), bbox_inches="tight")
        finally:
            fig.set_size_inches(*shown, forward=False)
        return path

    def save(self):
        label = self.current()
        if self.fig is None or label is None:
            messagebox.showinfo("Save", "There is no figure to save.", parent=self.win)
            return
        try:
            path = self.save_fig(self.fig, label)
        except Exception as exc:
            messagebox.showerror("Error saving", str(exc), parent=self.win)
            return
        self.status.set(f"saved {path}")

    def save_all(self):
        done, failed = 0, []
        for lab in self.labels:
            try:
                self.save_fig(self.make(lab), lab)
                done += 1
            except Exception as exc:   # e.g. a figure without data: reported, the others are saved
                failed.append(f"{lab}: {type(exc).__name__}: {exc}")
        messagebox.showinfo("Save all", f"{done} figures saved in\n{self.out_dir}"
                            + ("\n\nNot saved:\n" + "\n".join(failed) if failed else ""), parent=self.win)

    def open_folder(self):
        os.makedirs(self.out_dir, exist_ok=True)
        os.startfile(self.out_dir)
