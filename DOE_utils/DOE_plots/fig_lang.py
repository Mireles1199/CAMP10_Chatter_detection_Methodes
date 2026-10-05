"""fig_lang.py — EN / FR / both for the figures whose text is fixed in the code (viewer panels, plotters), without
touching that code: the export window translates the texts of the exported COPY of a figure (titles, axis labels,
legends, annotations; never the tick labels) with the table figure_texts.yaml. The figures on screen keep their text.

A text is translated phrase by phrase in one pass (longest phrase first); values, channel names (Axial_disp...),
units in brackets and $math$ are kept. "both" follows plot_style.lang_text: "[EN] ...\\n[FR] ..." (" / " in legends).
translate_figure returns the texts that still have words with no translation, to complete the table.

    python fig_lang.py --selftest
"""
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_FILE = os.path.join(HERE, "figure_texts.yaml")
# words that need no translation (acronyms are kept anyway: any word in capitals)
KEEP = {"vs", "kappa", "rpm", "log", "max", "min", "rms", "dB", "mod", "ref", "per", "sld", "doe",
        "maxent", "sst", "svd", "ssq", "plot"}   # indicator names and the Plot button are names, not words
_table = None


def table() -> dict:
    """{phrase: (en, fr)} from figure_texts.yaml (read once)."""
    global _table
    if _table is None:
        with open(TABLE_FILE, encoding="utf-8") as fh:
            _table = {str(r[0]): (str(r[1]), str(r[2])) for r in yaml.safe_load(fh) or []}
    return _table


def _pattern():
    keys = sorted(table(), key=len, reverse=True)
    # a phrase matches as whole words (no match inside Axial_disp, case_000, ...)
    return re.compile("|".join(rf"(?<![\w$]){re.escape(k)}(?![\w$])" for k in keys))


def tr(text: str):
    """(english, french, words left with no translation) of one text."""
    t = table()
    pat = _pattern()
    en = pat.sub(lambda m: t[m.group(0)][0], text)
    fr = pat.sub(lambda m: t[m.group(0)][1], text)
    rest = pat.sub(" ", text)
    rest = re.sub(r"\$[^$]*\$|\[[^\]]*\]|\{[^}]*\}|\S*[_\d]\S*", " ", rest)   # math, units, values, identifiers
    left = [w for w in re.findall(r"[A-Za-zÀ-ÿ]{3,}", rest) if not w.isupper() and w.lower() not in KEEP]
    return en, fr, left


def convert(text: str, language: str, sep: str = "\n") -> tuple:
    """(text in `language`, translated?) — 'both' as plot_style.lang_text."""
    if not text.strip():
        return text, True
    en, fr, left = tr(text)
    if language == "EN":
        out = en
    elif language == "FR":
        out = fr
    else:
        out = en if en == fr else f"[EN] {en}{sep}[FR] {fr}"
    return out, not left


def _texts(fig):
    """(Text, sep) of a figure that carry words: titles, axis labels, annotations, legends (not the tick labels)."""
    out = [(fig._suptitle, "\n")] if getattr(fig, "_suptitle", None) else []
    out += [(t, "\n") for t in fig.texts]
    legends = list(getattr(fig, "legends", []))
    for ax in fig.axes:
        out += [(t, "\n") for t in (ax.title, getattr(ax, "_left_title", None), getattr(ax, "_right_title", None),
                                    ax.xaxis.label, ax.yaxis.label) if t is not None]
        out += [(t, "\n") for t in ax.texts]
        if ax.get_legend() is not None:
            legends.append(ax.get_legend())
    for lg in legends:
        out += [(t, " / ") for t in lg.get_texts()] + [(lg.get_title(), " / ")]
    return out


def translate_figure(fig, language: str) -> list:
    """Translate the texts of `fig` in place; returns the original texts that still have untranslated words."""
    missing, seen = [], set()
    for t, sep in _texts(fig):
        if id(t) in seen:   # the same Text can be reached twice (e.g. a title that is also a figure text)
            continue
        seen.add(id(t))
        s = t.get_text()
        new, ok = convert(s, language, sep)
        if new != s:
            t.set_text(new)
        if not ok:
            missing.append(s)
    return missing


def _selftest():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    en, fr, left = tr("Convergencia Axial_disp\nRMS vs $dt$  (RPM=12000)")
    assert en.startswith("Convergence Axial_disp") and fr.startswith("Convergence Axial_disp") and not left, (en, left)
    assert tr("Tiempo de cómputo vs $dt$")[1] == "Temps de calcul vs $dt$"
    assert tr("Tiempo (s)")[0] == "Time (s)"                          # longest phrase first, one pass
    assert tr("control (sin ruido)")[1] == "contrôle (sans bruit)"
    assert tr("Axial_disp")[0] == "Axial_disp"                         # identifiers untouched
    assert tr("Something unknown")[2] == ["Something", "unknown"]
    assert convert("Density", "both")[0] == "[EN] Density\n[FR] Densité"
    assert convert("Density", "both", " / ")[0] == "[EN] Density / [FR] Densité"
    assert convert("$t$ (s)", "FR") == ("$t$ (s)", True)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="Control")
    ax.set_title("Detection time vs SNR")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Something unknown")
    ax.legend()
    fig.texts.append(ax.title)                                # reached twice: translated once
    missing = translate_figure(fig, "FR")
    assert ax.get_title() == "Temps de détection vs SNR" and ax.get_xlabel() == "Temps (s)"
    assert ax.get_legend().get_texts()[0].get_text() == "Contrôle" and missing == ["Something unknown"]
    print("fig_lang selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
