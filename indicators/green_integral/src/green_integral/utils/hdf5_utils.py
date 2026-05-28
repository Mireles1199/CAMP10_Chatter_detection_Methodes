"""Read-only HDF5 file reader with nested path navigation.

Loads an entire HDF5 file eagerly into nested Python dictionaries and
NumPy arrays at construction time, then exposes a rich query interface
for navigating the resulting tree.

:class:`HDF5Reader` is intentionally read-only: to keep memory
consumption manageable, arrays larger than a configurable threshold
should be loaded lazily (a future extension).  For the current research
use case all signals fit comfortably in RAM.

Usage example::

    reader = HDF5Reader("data/cono_Green_Integral_Method_results.h5")
    times  = reader.get_element("tool_dyn/time")
    vel    = reader.get_element("tool_dyn/Velocity")
    print(reader.list_paths()[:5])
"""

from typing import List, Optional, Dict, Any, Union
import h5py
import numpy as np
import os

class HDF5Reader:
    """Eager-loading, read-only HDF5 file reader.

    Reads the complete HDF5 file tree into a nested ``dict`` / NumPy
    array structure at construction time.  Once loaded the file handle is
    closed; all subsequent operations work on the in-memory copy.

    Groups become Python ``dict`` values; datasets are converted to NumPy
    arrays (with byte-string decoding applied automatically).

    Args:
        filepath (str): Absolute or relative path to the HDF5 file.

    Raises:
        FileNotFoundError: If *filepath* does not point to an existing file.
        OSError: If the file cannot be opened by :mod:`h5py` (corrupt,
            wrong format, locked, etc.).

    Attributes:
        filepath (str): The resolved path as passed to the constructor.
        data (Dict[str, Any]): Nested dictionary mirroring the HDF5 group/
            dataset tree.  Leaf values are NumPy arrays or Python scalars.

    Example::

        reader = HDF5Reader("results.h5")
        vel = reader.get_element("tool_dyn/Velocity")
        print(vel.shape)
    """

    _all_paths_cache: Optional[List[str]]

    def __init__(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        self.filepath = filepath
        self.data = self._read_file()
        self._all_paths_cache = None

    def _read_file(self) -> Dict[str, Any]:
        with h5py.File(self.filepath, "r") as hdf_file:
            return self._read_group(hdf_file)

    def _read_group(self, group: h5py.Group) -> Union[Dict[str, Any], List[Any], Any]:
        data = {}
        for key in group:
            item = group[key]
            if isinstance(item, h5py.Dataset):
                data[key] = self._read_dataset(item)
            elif isinstance(item, h5py.Group):
                data[key] = self._read_group(item)
        return data

    def _read_dataset(self, dataset: h5py.Dataset) -> Union[List[Any], Any]:
        data = dataset[()]
        if isinstance(data, np.ndarray):
            if data.dtype.kind in {"S", "a"}:
                decode = np.vectorize(
                    lambda x: x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else x,
                    otypes=[object],
                )
                decoded = decode(data)
                try:
                    return decoded.astype(str)
                except Exception:
                    return decoded
            if data.dtype == object:
                def _decode_obj(x: Any) -> Any:
                    if isinstance(x, (bytes, np.bytes_)):
                        return x.decode("utf-8")
                    if isinstance(x, np.ndarray):
                        return np.array([_decode_obj(y) for y in x.tolist()], dtype=object)
                    if isinstance(x, list):
                        return [_decode_obj(y) for y in x]
                    return x
                decoded_list = [_decode_obj(x) for x in data.tolist()]
                try:
                    return np.array(decoded_list)
                except Exception:
                    return np.array(decoded_list, dtype=object)
            return data
        if isinstance(data, (bytes, np.bytes_)):
            return data.decode("utf-8")
        return data

    def get_data(self) -> Dict[str, Any]:
        return self.data

    def get_element(self, *keys: str) -> Any:
        """Access a nested element using hierarchical keys or a slash-delimited path."""

        def _parse_slice(token: str):
            parts = token.split(":")
            if not 1 <= len(parts) <= 3:
                return None
            def _to_int(x):
                return int(x) if x != "" else None
            try:
                start = _to_int(parts[0]) if len(parts) >= 1 else None
                stop = _to_int(parts[1]) if len(parts) >= 2 else None
                step = _to_int(parts[2]) if len(parts) == 3 else None
            except ValueError:
                return None
            return slice(start, stop, step)

        def _parse_index(token: str):
            def _split_top_level_commas(s: str) -> List[str]:
                parts, buf, depth = [], [], 0
                for ch in s:
                    if ch == "[":
                        depth += 1
                    elif ch == "]":
                        depth = max(0, depth - 1)
                    if ch == "," and depth == 0:
                        parts.append("".join(buf).strip())
                        buf = []
                    else:
                        buf.append(ch)
                if buf:
                    parts.append("".join(buf).strip())
                return parts

            def _parse_list_token(t: str):
                if t.startswith("[") and t.endswith("]"):
                    inner = t[1:-1].strip()
                    if inner == "":
                        return []
                    try:
                        return [int(x.strip()) for x in inner.split(",")]
                    except ValueError:
                        return None
                return None

            if "," in token:
                idx_tokens = _split_top_level_commas(token)
                idx = []
                for t in idx_tokens:
                    lst = _parse_list_token(t)
                    if lst is not None:
                        idx.append(lst)
                        continue
                    s = _parse_slice(t) if ":" in t else None
                    if s is not None:
                        idx.append(s)
                    else:
                        try:
                            idx.append(int(t))
                        except ValueError:
                            return None
                return tuple(idx)
            if ":" in token:
                return _parse_slice(token)
            lst = _parse_list_token(token)
            if lst is not None:
                return lst
            try:
                return int(token)
            except ValueError:
                return None

        auto_search = False
        if len(keys) == 1 and isinstance(keys[0], str):
            if "/" in keys[0]:
                path_parts = [p for p in keys[0].split("/") if p != ""]
            else:
                path_parts = [keys[0]]
                auto_search = True
        else:
            path_parts = list(keys)

        current: Any = self.data
        for idx, key in enumerate(path_parts):
            if isinstance(current, dict):
                if key in current:
                    current = current[key]
                    continue
                if current is self.data:
                    if auto_search and len(path_parts) == 1:
                        found = self.find_first(key)
                        if found is None:
                            raise KeyError(f"Key not found (and no nested match): {key}")
                        return self.get_element(found)
                    if len(path_parts) > 1:
                        remaining = path_parts[idx + 1 :]
                        for base in self.find_all(key):
                            composed = base + (("/" + "/".join(remaining)) if remaining else "")
                            try:
                                return self.get_element(composed)
                            except KeyError:
                                continue
                raise KeyError(f"Key not found in group: {key}")
            if isinstance(current, (list, tuple)):
                parsed = _parse_index(key)
                if parsed is None:
                    raise KeyError(f"Invalid list/tuple index: {key}")
                try:
                    current = [current[i] for i in parsed] if isinstance(parsed, list) else current[parsed]
                except Exception as e:
                    raise KeyError(f"Index error for '{key}': {e}")
                continue
            if isinstance(current, np.ndarray):
                parsed = _parse_index(key)
                if parsed is None:
                    raise KeyError(f"Invalid numpy index: {key}")
                try:
                    current = current[parsed]
                except Exception as e:
                    raise KeyError(f"Index error for '{key}': {e}")
                continue
            raise KeyError(f"Cannot navigate into type {type(current).__name__} with key '{key}'")
        return current

    def list_paths(self) -> List[str]:
        if self._all_paths_cache is not None:
            return self._all_paths_cache
        paths: List[str] = []

        def _collect(node: Any, prefix: str = "") -> None:
            if isinstance(node, dict):
                if prefix:
                    paths.append(prefix)
                for k, v in node.items():
                    _collect(v, f"{prefix}/{k}" if prefix else k)
            else:
                if prefix:
                    paths.append(prefix)

        _collect(self.data)
        self._all_paths_cache = paths
        return paths

    def find_all(self, key: str) -> List[str]:
        return [p for p in self.list_paths() if p.split("/")[-1] == key]

    def find_first(self, key: str) -> Optional[str]:
        matches = self.find_all(key)
        return matches[0] if matches else None
