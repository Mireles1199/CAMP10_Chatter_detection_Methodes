"""eta_compat.py — kappa -> eta (docs/planes/PLAN_eta_rename.md): readers accept both names, `eta` first.

Existing .h5 files (simulation, reference_dataset, validations) keep their `kappa` attrs / columns; new writers write
`eta`. Every reader goes through here, so "accept both, eta wins" lives in one place. stdlib only.

    get(d, name, default=nan)   d[name] if it exists, else d[<name with kappa>], else default
                                name starts with `eta` or `$eta`: "eta", "eta_start", "eta_t0", "$eta$", "$eta_end$"
                                d: h5py attrs / Group, dict, anything with `in` and `[]`
    has(d, name)                True if either spelling exists
    col(group, name)            the dataset / column, either spelling (KeyError if neither)
    old_name(name)              "eta_start" -> "kappa_start"
    canon(key)                  "kappa_range" -> "eta_range"  (the other way round)
    canon_keys(obj)             copy of dicts / lists with every key canon()-ed (eta wins when both exist): the hash of a
                                config is the same with the old and the new key
    LABEL / TEX                 "η" / r"$\\eta$"  for texts and figures
Only the leading token is renamed: meta, beta, delta, kappa_txt-like names inside other words are left alone.
"""
import re

LABEL = "η"
TEX = r"$\eta$"
_NEW = re.compile(r"^(\$?)eta(?=$|_|\$)")
_OLD = re.compile(r"^(\$?)kappa(?=$|_|\$)")
_NAN = float("nan")


def old_name(name: str) -> str:
    return _NEW.sub(r"\1kappa", name)


def canon(key):
    return _OLD.sub(r"\1eta", key) if isinstance(key, str) else key


def has(d, name: str) -> bool:
    return name in d or old_name(name) in d


def get(d, name: str, default=_NAN):
    if name in d:
        return d[name]
    old = old_name(name)
    return d[old] if old != name and old in d else default


def col(group, name: str):
    out = get(group, name, None)
    if out is None:
        raise KeyError(name)
    return out


def canon_keys(x):
    if isinstance(x, dict):
        return {canon(k): canon_keys(v) for k, v in x.items() if not (canon(k) != k and canon(k) in x)}   # eta wins
    if isinstance(x, (list, tuple)):
        return type(x)(canon_keys(v) for v in x)
    return x


def _selftest():
    old, new, both = {"kappa": 0.8, "kappa_start": 0.5, "$kappa$": 1.1}, {"eta": 0.9, "$eta$": 1.2}, {"eta": 1.0, "kappa": 2.0}
    assert get(old, "eta") == 0.8 and get(old, "eta_start") == 0.5 and get(old, "$eta$") == 1.1
    assert get(new, "eta") == 0.9 and get(new, "$eta$") == 1.2 and get(both, "eta") == 1.0   # eta wins
    assert get(old, "eta_end") != get(old, "eta_end") and get(old, "eta_end", 7) == 7        # NaN / the default
    assert has(old, "eta") and has(new, "eta") and not has(old, "eta_end")
    assert get({"meta": 3, "beta": 4}, "meta", None) == 3 and old_name("meta") == "meta" and old_name("beta_x") == "beta_x"
    assert old_name("eta_t0") == "kappa_t0" and old_name("$eta_end$") == "$kappa_end$" and old_name("etaX") == "etaX"
    assert canon("kappa_range") == "eta_range" and canon("$kappa$") == "$eta$" and canon("kappas") == "kappas" and canon(3) == 3
    assert col({"kappa": [1, 2]}, "eta") == [1, 2]
    try:
        col({}, "eta")
        raise SystemExit("col of nothing should raise")
    except KeyError:
        pass
    cfg_old = {"noise": {"kappa_range": [0.5, 2], "x": [{"kappa": 1}]}, "k": 1}
    cfg_new = {"noise": {"eta_range": [0.5, 2], "x": [{"eta": 1}]}, "k": 1}
    assert canon_keys(cfg_old) == cfg_new == canon_keys(cfg_new)                        # same hash with either spelling
    assert canon_keys({"eta_a": 1, "kappa_a": 2}) == {"eta_a": 1}                       # eta wins
    try:   # h5py attrs / groups work too
        import h5py
        import tempfile
        import os
        p = os.path.join(tempfile.mkdtemp(), "x.h5")
        with h5py.File(p, "w") as f:
            f.attrs["kappa"] = 0.8
            f["summary/kappa"] = [1.0, 2.0]
        with h5py.File(p, "r") as f:
            assert get(f.attrs, "eta") == 0.8 and list(col(f["summary"], "eta")[()]) == [1.0, 2.0]
    except ImportError:
        pass
    print("eta_compat selftest OK")


if __name__ == "__main__":
    _selftest()
