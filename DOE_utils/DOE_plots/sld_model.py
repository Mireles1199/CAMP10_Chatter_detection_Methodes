"""sld_model.py — Lóbulos de estabilidad (SLD) con los casos del DOE encima.

Declarar la máquina = editar/añadir un preset en MODELS (como los ejemplos de
SLD_tools/Lobes.py). Los lóbulos los calcula sld_tools; el dibujo sigue
article-plot-style (plot_style.py): FIGSIZE_SIMPLE x FIGSCALE, constrained_layout,
idioma EN/FR, un color por modo.

Uso desde doe_unified_selector.py: las entradas "SLD — ..." del combo de resumen.
Autotest:  python sld_model.py
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt

import plot_style as ps

LANGUAGE = "EN"   # "EN" | "FR" | "both"
FIGSCALE = 1.5    # multiplicador de FIGSIZE_SIMPLE (mismo criterio que los plots de indicadores)
Y_AXIS = "Ap"     # eje vertical de plot_sld: "Ap" [mm] | "kappa" = Ap / límite del SLD a esa velocidad (casilla del visor)
DEFAULT_XLIM = (7000.0, 15000.0)   # rpm, si ningún caso trae spin_rate
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]   # Okabe-Ito: un color por modo
# resultado de validación por caso (attr $outcome_<run>$ de doe_validation_results.h5): color, marcador, texto
OUTCOMES = {"TP": ("#009E73", "o", "TP detected"), "TN": ("#56B4E9", "s", "TN no alarm"),
            "FN": ("#CC79A7", "X", "FN missed"), "FP": ("#F0E442", "^", "FP false alarm")}

# preset -> modos (f_n [Hz], k [N/m], zeta, theta [deg]) + Kf [N/m^2] + nº de lóbulos k
MODELS = {
    "1DOF_150": dict(modes=[(150.0, 2.13e8, 0.010, -45.0)], Kf=1000e6, num_lobes=5),
    "2DOF_150_250": dict(
        modes=[(250.0, 2.26e8, 0.012, 30.0), (150.0, 2.13e8, 0.010, -45.0)],
        Kf=1000e6, num_lobes=5,
    ),
}


@lru_cache(maxsize=None)
def lobes(preset: str):
    """(lobes[segmento, k, w, (rpm, a_lim mm)], f_peaks [Hz]). Un segmento por modo, de menor a mayor f."""
    from sld_tools import FRFModel, LobeCalculator, AltintasPhaseStrategy

    m = MODELS[preset]
    frf = FRFModel([(k, z, np.radians(th), lambda w, f=f: w / f) for f, k, z, th in m["modes"]])
    w = np.linspace(0.0, max(f for f, *_ in m["modes"]) * 1.5, 50_000)
    out = LobeCalculator(frf, AltintasPhaseStrategy()).compute_lobes(
        w=w, target=0.0, k_list=np.arange(m["num_lobes"]), Kf=m["Kf"])
    if not isinstance(out, tuple):   # compute_lobes devuelve solo un array vacío si G nunca es < 0
        raise ValueError(f"SLD '{preset}': la FRF no tiene regiones con G < 0, no hay lóbulos")
    return out


def _segs(preset: str, seg):
    n = len(lobes(preset)[0])
    return list(range(n)) if seg is None else [min(seg, n - 1)]


def ap_crit(preset: str, seg=None) -> float:
    """Profundidad crítica mínima [mm] (el 'AP_REF' del modelo)."""
    return float(np.nanmin(lobes(preset)[0][_segs(preset, seg)][..., 1]))


def ap_lim(preset: str, rpm: float) -> float:
    """Límite de estabilidad [mm] a esas rpm: el mínimo de a_lim entre todos los lóbulos (todos los modos y k)
    que pasan por Ω = rpm, o sea la envolvente del SLD en ese punto. A diferencia de ap_crit (el mínimo
    global, sin rpm), esto cambia con el spin: es el lóbulo en el que cae el caso.
    En un hueco entre lóbulos (bolsillo de estabilidad: el límite sube hacia infinito) devuelve inf; fuera
    del rango de rpm que cubren los lóbulos calculados (k < num_lobes) lanza ValueError.
    """
    lb, _ = lobes(preset)
    best, lo, hi = np.inf, np.inf, -np.inf
    for j in range(lb.shape[0]):
        for i in range(lb.shape[1]):
            x, y = lb[j, i, :, 0], lb[j, i, :, 1]
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            if x.size < 2:
                continue
            lo, hi = min(lo, x.min()), max(hi, x.max())
            a, b = x[:-1] - rpm, x[1:] - rpm
            m = (a * b <= 0) & (a != b)   # el tramo cruza la vertical Ω = rpm (puede haber varias ramas)
            if m.any():
                t = a[m] / (a[m] - b[m])
                best = min(best, float(np.min(y[:-1][m] + t * (y[1:][m] - y[:-1][m]))))
    if np.isfinite(best):
        return best
    if lo < rpm < hi:
        return float("inf")   # hueco entre lóbulos: estable a cualquier profundidad
    raise ValueError(f"SLD '{preset}': {rpm:g} rpm está fuera del rango de los lóbulos calculados "
                     f"({lo:.0f}-{hi:.0f} rpm, k < {lb.shape[1]})")


@lru_cache(maxsize=None)
def envelope(preset: str, n: int = 1500):
    """(rpm, límite [mm]) de la envolvente del SLD (ap_lim) en n velocidades sobre todo el rango de los lóbulos;
    inf en un hueco entre lóbulos. Tabla para dibujar los lóbulos en kappa (los casos usan ap_lim exacto)."""
    x = lobes(preset)[0][..., 0]
    x = x[np.isfinite(x)]
    rpm = np.linspace(x.min(), x.max(), n)
    lim = []
    for r in rpm:
        try:
            lim.append(ap_lim(preset, r))
        except ValueError:
            lim.append(np.nan)
    return rpm, np.array(lim)


def kappa_of(preset: str, rpm, ap_mm, exact: bool = False):
    """kappa = Ap / límite del SLD a esas rpm (nan en un hueco entre lóbulos o fuera de su rango).
    exact: ap_lim punto a punto (los casos); si no, interpolado en envelope() (los lóbulos, miles de puntos)."""
    rpm, ap_mm = np.asarray(rpm, float), np.asarray(ap_mm, float)
    if exact:
        def one(r):
            try:
                v = ap_lim(preset, float(r))
            except ValueError:
                return np.nan
            return v if np.isfinite(v) else np.nan   # hueco entre lóbulos: kappa sin definir, no 0
        lim = np.vectorize(one, otypes=[float])(rpm)
    else:
        g, e = envelope(preset)
        ok = np.isfinite(e)
        lim = np.interp(rpm, g[ok], e[ok], left=np.nan, right=np.nan)
        lim[~ok[np.clip(np.searchsorted(g, rpm), 0, len(g) - 1)]] = np.nan   # en un hueco: sin límite
    with np.errstate(divide="ignore", invalid="ignore"):
        k = ap_mm / lim
    k[~np.isfinite(k)] = np.nan
    return k


def case_points(cases, outcome_run=None):
    """[(rpm, ap_ini mm, ap_fin mm)] por caso; omite los que no traen spin_rate/Ap_*.
    Con outcome_run, cada tupla lleva un 4º elemento: el outcome de ese indicador ("" si el caso no lo trae)."""
    out = []
    for c in cases:
        v = c.get("var_val", {})
        g = lambda k: v.get(k, v.get(f"${k}$"))
        try:
            pt = (float(g("spin_rate")), 1e3 * float(g("Ap_start")), 1e3 * float(g("Ap_end")))
            out.append(pt + (str(v.get(f"outcome_{outcome_run}", "")),) if outcome_run else pt)
        except (TypeError, ValueError):
            pass
    return out


YCAP = 60.0   # mm: sobre esto un cruce no tiene sentido físico (mismo tope que el eje de Lobes.py)


@lru_cache(maxsize=None)
def intersections(preset: str):
    """((rpm, ap mm), ...) de TODOS los cruces entre curvas de modos distintos con a_p <= YCAP.

    No depende de la vista: plot_sld los dibuja todos, pero Matplotlib solo muestra los que caen
    dentro de los límites (que se fijan según los casos del DOE); al hacer zoom out aparecen el resto.
    Cruce exacto entre los segmentos de las dos polilíneas (sirve aunque una rama no sea monótona).
    ponytail: cada curva se diezma a ~6000 puntos y se prueban todos los pares de segmentos por bloques
    (O(n*m) por pareja de lóbulos); con >3 modos conviene un barrido por rpm.
    """
    lb, _ = lobes(preset)
    x1 = float(np.nanmax(lb[..., 0]))

    def segs(j, i):   # segmentos con x/x1 e y/YCAP en [0, 1]
        x, y = lb[j, i, :, 0], lb[j, i, :, 1]
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        st = max(1, x.size // 6000)
        x, y = x[::st] / x1, y[::st] / YCAP
        keep = y <= 1.0
        m = keep[:-1] & keep[1:]
        return x[:-1][m], y[:-1][m], (x[1:] - x[:-1])[m], (y[1:] - y[:-1])[m]

    S = {(j, i): segs(j, i) for j in range(len(lb)) for i in range(lb.shape[1])}

    def near(T, lo, hi):   # segmentos cuya caja (x, y) toca la caja común de las dos curvas
        px, py, rx, ry = T
        m = (np.maximum(px, px + rx) >= lo[0]) & (np.minimum(px, px + rx) <= hi[0])             & (np.maximum(py, py + ry) >= lo[1]) & (np.minimum(py, py + ry) <= hi[1])
        return tuple(v[m] for v in T)

    def box(T):   # (min, max) de la curva en x e y
        return (np.minimum(T[0], T[0] + T[2]).min(), np.minimum(T[1], T[1] + T[3]).min()),                (np.maximum(T[0], T[0] + T[2]).max(), np.maximum(T[1], T[1] + T[3]).max())

    found = []
    for j1 in range(len(lb)):
        for j2 in range(j1 + 1, len(lb)):
            for i1 in range(lb.shape[1]):
                for i2 in range(lb.shape[1]):
                    A, B = S[j1, i1], S[j2, i2]
                    if A[0].size == 0 or B[0].size == 0:
                        continue
                    (al, ah), (bl, bh) = box(A), box(B)
                    lo, hi = np.maximum(al, bl), np.minimum(ah, bh)   # caja común: fuera de ella no hay cruce
                    if (lo > hi).any():
                        continue
                    A, B = near(A, lo, hi), near(B, lo, hi)
                    qx, qy, sx, sy = (v[None, :] for v in B)
                    for a in range(0, A[0].size, 500):   # bloques de 500 filas: acota la memoria
                        px, py, rx, ry = (v[a:a + 500, None] for v in A)
                        den = rx * sy - ry * sx
                        with np.errstate(divide="ignore", invalid="ignore"):
                            t = ((qx - px) * sy - (qy - py) * sx) / den
                            u = ((qx - px) * ry - (qy - py) * rx) / den
                        hit = (np.abs(den) > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
                        found += list(zip((px + t * rx)[hit], (py + t * ry)[hit]))
    out = []
    for x, y in sorted(found):   # une los cruces repetidos (vértice compartido por 2 segmentos)
        if not out or abs(x - out[-1][0]) > 1e-6 or abs(y - out[-1][1]) > 1e-6:
            out.append((x, y))
    return tuple((float(x * x1), float(y * YCAP)) for x, y in out)


def plot_sld(cases, preset: str, seg=None, out_dir=None, language: str | None = None, outcome_run=None,
             y_axis: str | None = None):
    """Figura SLD del preset (seg=None: todos los modos; seg=j: solo el modo j) con los casos del DOE.

    Cada caso es un punto (rpm, Ap); si Ap_start != Ap_end, un segmento vertical. out_dir se ignora:
    el visor lo pasa a todas las figuras de resumen.
    outcome_run: nombre de un indicador de doe_validation_results.h5; los puntos se colorean por su resultado
    (TP/TN/FN/FP, ver OUTCOMES) en vez de todos del mismo color.
    y_axis: "Ap" [mm] o "kappa" (por defecto Y_AXIS). En kappa cada lóbulo se divide por el límite del SLD a su
    velocidad (la envolvente queda en kappa = 1) y cada caso va a Ap / límite a su velocidad (con estos lóbulos;
    puede diferir del kappa guardado en el .h5 si su ap_ref no fue model_at_spin de este modelo).
    """
    lang = language or LANGUAGE
    yk = (y_axis or Y_AXIS) == "kappa"
    lb, f_peaks = lobes(preset)
    pts = case_points(cases, outcome_run)
    if yk:   # (rpm, kappa_ini, kappa_fin[, outcome])
        pts = [(p[0], *(float(v) for v in kappa_of(preset, [p[0], p[0]], p[1:3], exact=True)), *p[3:]) for p in pts]

    with plt.rc_context(ps.ARTICLE_RCPARAMS):
        fig, ax = plt.subplots(figsize=ps.figsize_from_scale(ps.FIGSIZE_SIMPLE, FIGSCALE),
                               constrained_layout=True)
        fig._keep_size = tuple(fig.get_size_inches())   # el visor no la fuerza a 4.5x4.5 y la guarda a este tamaño
        # vista: alrededor de los casos (o rango por defecto)
        if pts:
            r = [p[0] for p in pts]
            pad = max(0.3 * (max(r) - min(r)), 3000.0)
            x0, x1 = max(min(r) - pad, 0.0), max(r) + (2 * pad if outcome_run else pad)   # outcome: margen a la derecha para la leyenda
        else:
            x0, x1 = DEFAULT_XLIM
        for j in _segs(preset, seg):
            c, a_j = COLORS[j % len(COLORS)], ap_crit(preset, j)
            for i in range(lb.shape[1]):   # un lóbulo por k, todos del color de su modo
                x, y = lb[j, i, :, 0], lb[j, i, :, 1]
                ok = np.isfinite(x) & np.isfinite(y)
                ax.plot(x[ok], kappa_of(preset, x[ok], y[ok]) if yk else y[ok], color=c,
                        label=rf"{f_peaks[j]:.0f} Hz  ($a_{{p,\min}}$ = {a_j:.3f} mm)" if i == 0 else None)
            if not yk:
                ax.axhline(a_j, color=c, linewidth=0.8, linestyle="--")
        a_min = ap_crit(preset, seg)
        vals = [v for p in pts for v in p[1:3] if np.isfinite(v)]
        top = max(vals) if vals else np.nan
        if yk:
            ax.axhline(1.0, color="k", linewidth=0.8, linestyle="--",
                       label=ps.lang_text(r"$\kappa$ = 1 (stability limit)", r"$\kappa$ = 1 (limite de stabilité)", lang,
                                          sep=" / "))
            ymax = max(1.3 * top, 2.0) if np.isfinite(top) else 2.0
        else:
            ymax = max(1.3 * top, 2.0 * a_min) if np.isfinite(top) else 3.0 * a_min

        # TODOS los cruces entre los modos (solo si se dibujan 2 o más): punto + nota con (rpm, Ap).
        # Los límites se fijan abajo según los casos; los cruces fuera de vista aparecen al hacer zoom out.
        if seg is None and len(lb) > 1:
            for n, (xi, yi) in enumerate(intersections(preset)):
                yv = float(kappa_of(preset, [xi], [yi], exact=True)[0]) if yk else yi
                ax.scatter([xi], [yv], marker="D", s=45, facecolor="white", edgecolor="k", linewidths=0.9,
                           zorder=6, label=ps.lang_text("Intersection", "Intersection", lang) if n == 0 else None)
                ax.annotate(f"{xi:.0f} rpm\n" + (f"κ = {yv:.2f}" if yk else f"{yi:.3f} mm"), (xi, yv),
                            xytext=(14, 14 if n % 2 == 0 else -34),
                            textcoords="offset points", fontsize=9, zorder=7,
                            arrowprops=dict(arrowstyle="-", linewidth=0.6))

        if pts:
            for r_, a0, a1, *_ in pts:
                if a0 != a1:
                    ax.plot([r_, r_], [a0, a1], color="crimson", linewidth=1.2)
            if outcome_run:
                for oc in [*OUTCOMES, ""]:   # "" = caso sin resultado para este indicador
                    sel = [p for p in pts if p[3] == oc]
                    if not sel:
                        continue
                    col, mk, txt = OUTCOMES.get(oc, ("crimson", "o", "no result"))
                    ax.scatter([p[0] for p in sel for _ in (0, 1)], [a for p in sel for a in p[1:3]],
                               color=col, marker=mk, s=45, edgecolor="k", linewidths=0.8, zorder=5,
                               label=f"{txt}  [{len(sel)}]")
            else:
                ax.scatter([p[0] for p in pts for _ in (0, 1)], [a for p in pts for a in p[1:3]],
                           color="crimson", s=30, edgecolor="k", linewidths=0.6, zorder=5,
                           label=ps.lang_text("DOE cases", "Cas du DOE", lang, sep=" / "))

        ax.set_xlim(x0, x1)
        ax.set_ylim(0, ymax)
        ax.set_xlabel(ps.lang_text(r"Spindle speed $\Omega$ [rpm]", r"Vitesse de broche $\Omega$ [tr/min]", lang))
        ax.set_ylabel(r"$\kappa = a_p\,/\,a_{p,\lim}(\Omega)$ [–]" if yk else
                      ps.lang_text(r"Width of cut $a_p$ [mm]", r"Largeur de coupe $a_p$ [mm]", lang))
        ax.set_title(f"SLD — {preset}" + (f" — {outcome_run}" if outcome_run else ""))
        # los lóbulos quedan sobre a_p,min: abajo no hay curvas (con outcome, a la derecha: los puntos de un DOE
        # a una sola velocidad caen en el centro)
        ax.legend(loc="lower right" if outcome_run else "lower left")
    return fig


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    # 1DOF: el mínimo tiene fórmula cerrada, 2 k zeta (1+zeta) / (Kf cos^2(theta)); con el modelo de
    # 150 Hz da 8.6052 mm, el AP_REF de doe_runner
    f, k, z, th = MODELS["1DOF_150"]["modes"][0]
    exact = 2 * k * z * (1 + z) / (MODELS["1DOF_150"]["Kf"] * np.cos(np.radians(th)) ** 2) * 1e3
    assert abs(ap_crit("1DOF_150") - exact) < 1e-3, (ap_crit("1DOF_150"), exact)
    # 2 modos: el mínimo global no puede ser mayor que el de un modo solo
    assert ap_crit("2DOF_150_250") <= ap_crit("2DOF_150_250", seg=0) + 1e-9
    # ap_lim (límite a unas rpm): nunca baja del mínimo global; en el rpm del mínimo lo iguala;
    # en el cruce de los dos modos de 2DOF (9989 rpm) vale el del cruce (15.557 mm); fuera de los lóbulos, error
    lb1, _ = lobes("1DOF_150")
    j, i, w = np.unravel_index(np.nanargmin(lb1[..., 1]), lb1[..., 1].shape)
    assert abs(ap_lim("1DOF_150", lb1[j, i, w, 0]) - ap_crit("1DOF_150")) < 1e-2
    for name in MODELS:
        assert all(ap_lim(name, w_) >= ap_crit(name) - 1e-9 for w_ in range(6000, 20001, 250)), name
    assert abs(ap_lim("2DOF_150_250", 9989.0) - 15.557) < 0.05, ap_lim("2DOF_150_250", 9989.0)
    assert ap_lim("1DOF_150", 8980.0) == float("inf")   # bolsillo entre los lóbulos k=1 (hasta 8954) y k=0 (desde 9003)
    try:
        ap_lim("1DOF_150", 1e6)
        raise AssertionError("debía fallar fuera de los lóbulos")
    except ValueError:
        pass
    cases = [{"var_val": {"spin_rate": 12100.0, "Ap_start": 5e-3, "Ap_end": 15e-3}},   # rampa
             {"var_val": {"$spin_rate$": 12100.0, "$Ap_start$": 9e-3, "$Ap_end$": 9e-3}},  # fijo, con $
             {"var_val": {}}]                                                          # sin datos: se omite
    assert len(case_points(cases)) == 2
    oc = [{"var_val": {"spin_rate": 12100.0, "Ap_start": 9e-3, "Ap_end": 9e-3, "outcome_run1": "TP"}},
          {"var_val": {"spin_rate": 12100.0, "Ap_start": 5e-3, "Ap_end": 5e-3}}]   # este no trae outcome
    assert [p[3] for p in case_points(oc, "run1")] == ["TP", ""] and len(case_points(oc)[0]) == 3
    leg = [t.get_text() for t in plot_sld(oc, "1DOF_150", outcome_run="run1").axes[0].get_legend().get_texts()]
    assert any(t.startswith("TP") and t.endswith("[1]") for t in leg) and any("no result" in t for t in leg), leg
    # cruces: con 1 modo no hay; con 2 modos cada punto cae sobre las dos curvas
    assert intersections("1DOF_150") == ()
    cr = intersections("2DOF_150_250")
    assert any(abs(x - 9989) < 5 and abs(y - 15.557) < 0.01 for x, y in cr), cr   # el cruce de la vista habitual
    lbs, _ = lobes("2DOF_150_250")
    for xi, yi in cr:   # cada cruce cae sobre las curvas de los dos modos (hay muestras cada ~9 rpm)
        for j in (0, 1):
            d = min(np.nanmin(np.hypot((lbs[j, i, :, 0] - xi) / 40.0, (lbs[j, i, :, 1] - yi) / 0.5))
                    for i in range(lbs.shape[1]))
            assert d < 1.0, (xi, yi, j, d)
    for p in MODELS:
        for s in (None, 0):
            fig = plot_sld(cases, p, seg=s)
            assert np.allclose(fig._keep_size, ps.figsize_from_scale(ps.FIGSIZE_SIMPLE, FIGSCALE))
    # eje Ap por defecto (llamadas de siempre) y rótulo pedido por el usuario
    assert plot_sld(cases, "1DOF_150", language="FR").axes[0].get_ylabel() == r"Largeur de coupe $a_p$ [mm]"
    assert plot_sld(cases, "1DOF_150").axes[0].get_ylabel() == r"Width of cut $a_p$ [mm]"
    # kappa: el caso a 9 mm y 12100 rpm va a 9 / ap_lim; la envolvente de los lóbulos queda en 1
    fig = plot_sld(cases, "1DOF_150", y_axis="kappa")
    ax = fig.axes[0]
    k9 = 9.0 / ap_lim("1DOF_150", 12100.0)
    pts_y = np.concatenate([c.get_offsets()[:, 1] for c in ax.collections])
    assert np.any(np.isclose(pts_y, k9, rtol=1e-6)), (pts_y, k9)
    g, e = envelope("1DOF_150")
    ok = np.isfinite(e)
    assert np.allclose(kappa_of("1DOF_150", g[ok][::50], e[ok][::50]), 1.0)                 # el límite en kappa = 1
    assert np.isnan(kappa_of("1DOF_150", [8980.0], [5.0], exact=True)[0])                  # hueco: sin kappa
    assert np.isnan(kappa_of("1DOF_150", [1e6], [5.0], exact=True)[0])                     # fuera de los lóbulos
    assert "kappa" in ax.get_ylabel() and any(ln.get_ydata()[0] == 1.0 for ln in ax.lines if len(ln.get_ydata()) == 2)
    Y_AXIS = "kappa"                                                                         # el global de la casilla
    assert "kappa" in plot_sld(cases, "2DOF_150_250").axes[0].get_ylabel()
    Y_AXIS = "Ap"
    print("sld_model self-test OK")
