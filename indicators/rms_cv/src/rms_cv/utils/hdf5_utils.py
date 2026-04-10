
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

    # Internal cache; populated lazily by list_paths()
    _all_paths_cache: Optional[List[str]]

    def __init__(self, filepath: str):
        """Initialise the reader and eagerly load the entire HDF5 tree.

        Args:
            filepath (str): Path to the HDF5 file.

        Raises:
            FileNotFoundError: If the file does not exist at *filepath*.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        self.filepath = filepath
        self.data = self._read_file()
        # Index of all paths for quick search
        self._all_paths_cache = None

    def _read_file(self) -> Dict[str, Any]:
        """
        Reads the entire HDF5 file and converts it into a complete dictionary.
        """
        with h5py.File(self.filepath, "r") as hdf_file:
            return self._read_group(hdf_file)

    def _read_group(self, group: h5py.Group) -> Union[Dict[str, Any], List[Any], Any]:
        """
        Reads a group or dataset and converts it into a Python data structure.
        """
        data = {}
        for key in group:
            item = group[key]
            if isinstance(item, h5py.Dataset):
                data[key] = self._read_dataset(item)
            elif isinstance(item, h5py.Group):
                data[key] = self._read_group(item)
        return data

    def _read_dataset(self, dataset: h5py.Dataset) -> Union[List[Any], Any]:
        """
        Converts a dataset into a native Python type.
        """
        data = dataset[()]
    
        # Si es un array de numpy, devolver SIEMPRE un numpy.ndarray
        if isinstance(data, np.ndarray): 
            # Caso: array de bytes (cadenas en formato bytes) -> decodificar a str
            if data.dtype.kind in {'S', 'a'}:  # Byte strings
                # Decodificación elemento a elemento, preservando la forma
                decode = np.vectorize(lambda x: x.decode('utf-8') if isinstance(x, (bytes, np.bytes_)) else x,
                                      otypes=[object])
                decoded = decode(data)
                # Intentar convertir a dtype de cadenas nativas si es rectangular
                try:
                    return decoded.astype(str)  
                except Exception:
                    return decoded  # Mantener dtype=object si es irregular

            # Caso: array de objetos (posibles listas por fila, bytes anidados, etc.)
            if data.dtype == object:
                def _decode_obj(x: Any) -> Any:
                    if isinstance(x, (bytes, np.bytes_)):
                        return x.decode('utf-8')
                    if isinstance(x, np.ndarray):
                        # Decodificar recursivamente arrays anidados
                        return np.array([_decode_obj(y) for y in x.tolist()], dtype=object)
                    if isinstance(x, list):
                        return [_decode_obj(y) for y in x]
                    return x

                decoded_list = [_decode_obj(x) for x in data.tolist()]
                # Devolver como array; si las filas son listas de distinta longitud, será dtype=object
                try:
                    return np.array(decoded_list)
                except Exception:
                    return np.array(decoded_list, dtype=object)

            # Para arrays numéricos u otros tipos estándar, devolver tal cual
            return data

        # Si es un byte string, decodificarlo
        if isinstance(data, (bytes, np.bytes_)):
            return data.decode('utf-8')
        
        return data

    def get_data(self) -> Dict[str, Any]:
        """
        Returns the complete dictionary.
        """
        return self.data

    def get_element(self, *keys: str) -> Any:
        """Access a nested element using hierarchical keys or a slash-delimited path.

        *keys* can be passed in several equivalent forms:

        * Multiple positional strings: ``get_element("group", "sub", "dataset")``
        * A single slash-delimited path: ``get_element("group/sub/dataset")``
        * With indexing tokens appended: ``get_element("dataset", "0:10")``

        Indexing tokens (for the last key segment) support:

        * **Integer index:** ``"5"`` → ``data[5]``
        * **Slice:** ``"0:100"`` or ``"::2"`` → ``data[0:100]``
        * **Multi-dim:** ``"0,1"`` or ``"1:5,:"`` (CSV of int/slice/list tokens)
        * **List of indices:** ``"[0,2,4]"`` → ``data[[0,2,4]]``

        If the top-level key is not found in the root ``data`` dictionary
        the method attempts a recursive search via :meth:`find_first`.

        Args:
            *keys (str): One or more path segments (or a single slash-delimited
                string) identifying the target node, optionally followed by an
                indexing token.

        Returns:
            Any: The resolved node — a ``dict``, ``np.ndarray``, or scalar.

        Raises:
            KeyError: If any segment is not found or an indexing token is
                invalid.

        Example::

            v    = reader.get_element("tool_dyn/Velocity")
            clip = reader.get_element("tool_dyn", "Velocity", "0:500")
            rows = reader.get_element("matrix", "[0,2,4]")
        """

        def _parse_slice(token: str):
            # Supports 'start:end[:step]' with empty parts allowed (e.g., ':10', '5:')
            parts = token.split(':')
            if not 1 <= len(parts) <= 3:
                return None
            def _to_int(x):
                return int(x) if x != '' else None
            try:
                start = _to_int(parts[0]) if len(parts) >= 1 else None
                stop = _to_int(parts[1]) if len(parts) >= 2 else None
                step = _to_int(parts[2]) if len(parts) == 3 else None
            except ValueError:
                return None
            return slice(start, stop, step)

        def _parse_index(token: str):
            # Helper: split commas but ignore those inside brackets [...]
            def _split_top_level_commas(s: str) -> List[str]:
                parts = []
                buf = []
                depth = 0
                for ch in s:
                    if ch == '[':
                        depth += 1
                    elif ch == ']':
                        depth = max(0, depth - 1)
                    if ch == ',' and depth == 0:
                        parts.append(''.join(buf).strip())
                        buf = []
                    else:
                        buf.append(ch)
                if buf:
                    parts.append(''.join(buf).strip())
                return parts

            # Helper: parse list token like '[0,2,4]'
            def _parse_list_token(t: str):
                if t.startswith('[') and t.endswith(']'):
                    inner = t[1:-1].strip()
                    if inner == '':
                        return []
                    try:
                        return [int(x.strip()) for x in inner.split(',')]
                    except ValueError:
                        return None
                return None

            # Multi-dim index like 'i,j' or with slices '1:5, :' or lists ':[0,2,4]'
            if ',' in token:
                idx_tokens = _split_top_level_commas(token)
                idx = []
                for t in idx_tokens:
                    lst = _parse_list_token(t)
                    if lst is not None:
                        idx.append(lst)
                        continue
                    s = _parse_slice(t) if ':' in t else None
                    if s is not None:
                        idx.append(s)
                    else:
                        try:
                            idx.append(int(t))
                        except ValueError:
                            return None
                return tuple(idx)

            # Single-dim: slice or int
            if ':' in token:
                s = _parse_slice(token)
                return s
            # Single-dim: list-of-indices '[0,2,4]'
            lst = _parse_list_token(token)
            if lst is not None:
                return lst
            try:
                return int(token)
            except ValueError:
                return None

        # If a single path was provided, split it by '/'
        auto_search = False
        if len(keys) == 1 and isinstance(keys[0], str):
            if '/' in keys[0]:
                path_parts = [p for p in keys[0].split('/') if p != '']
            else:
                # Single token; we may need to auto-search nested keys if not present at root
                path_parts = [keys[0]]
                auto_search = True
        else:
            path_parts = list(keys)

        current: Any = self.data
        for idx, key in enumerate(path_parts):
            # Navigate dictionaries (HDF5 groups)
            if isinstance(current, dict):
                if key in current:
                    current = current[key]
                    continue
                else:
                    # If we are at root, try nested resolution for two cases:
                    # 1) Single-token auto_search (handled as before)
                    # 2) Multi-part path starting with a nested key (e.g., 'tool_dyn/subkey')
                    if current is self.data:
                        # Case 1: single token search
                        if auto_search and len(path_parts) == 1:
                            found = self.find_first(key)
                            if found is None:
                                raise KeyError(f"Key not found in group (and no nested match): {key}")
                            return self.get_element(found)

                        # Case 2: multi-part path; try to resolve base segment against all matches
                        if len(path_parts) > 1:
                            remaining = path_parts[idx+1:]
                            # Candidates whose last segment equals the missing key
                            candidates = self.find_all(key)
                            for base in candidates:
                                composed = base + (('/' + '/'.join(remaining)) if remaining else '')
                                try:
                                    return self.get_element(composed)
                                except KeyError:
                                    continue
                    # If not resolved, raise
                    raise KeyError(f"Key not found in group: {key}")

            # Index lists or tuples
            if isinstance(current, (list, tuple)):
                idx = _parse_index(key)
                if idx is None:
                    raise KeyError(f"Invalid list/tuple index: {key}")
                try:
                    if isinstance(idx, list):
                        # Manual advanced indexing for Python lists
                        current = [current[i] for i in idx]
                    else:
                        current = current[idx]
                except Exception as e:
                    raise KeyError(f"Index error for '{key}': {e}")
                continue

            # Index numpy arrays
            if isinstance(current, np.ndarray):
                idx = _parse_index(key)
                if idx is None:
                    raise KeyError(f"Invalid numpy index: {key}")
                try:
                    current = current[idx]
                except Exception as e:
                    raise KeyError(f"Index error for '{key}': {e}")
                continue

            # Unsupported type for further navigation
            raise KeyError(f"Cannot navigate into type {type(current).__name__} with key '{key}'")

        return current

    def list_paths(self) -> List[str]:
        """Return all node paths in the loaded HDF5 tree.

        Traverses the in-memory ``data`` dictionary recursively.
        Results are cached after the first call so that repeated
        invocations are O(1).

        Returns:
            List[str]: Slash-delimited paths for every group and dataset
            node, e.g. ``["tool_dyn", "tool_dyn/time", "tool_dyn/Velocity",
            ...]``.

        Example::

            paths = reader.list_paths()
            print([p for p in paths if "Velocity" in p])
        """
        if self._all_paths_cache is not None:
            return self._all_paths_cache

        paths: List[str] = []

        def _collect(node: Any, prefix: str = ""):
            if isinstance(node, dict):
                # Include group path itself so mid-path keys can be found
                if prefix:
                    paths.append(prefix)
                if not node:
                    return
                for k, v in node.items():
                    new_prefix = f"{prefix}/{k}" if prefix else k
                    _collect(v, new_prefix)
            else:
                if prefix:
                    paths.append(prefix)

        _collect(self.data)
        self._all_paths_cache = paths
        return paths

    def find_all(self, key: str) -> List[str]:
        """Find every path whose last segment matches *key*.

        Args:
            key (str): The exact final path segment to match (case-sensitive).

        Returns:
            List[str]: All matching full paths ordered by depth
            (breadth-first insertion order from :meth:`list_paths`).
            Returns an empty list if *key* is not found.

        Example::

            paths = reader.find_all("Velocity")
            # ["tool_dyn/Velocity", "workpiece/Velocity"] (example)
        """
        matches = []
        for p in self.list_paths():
            last = p.split('/')[-1]
            if last == key:
                matches.append(p)
        return matches

    def find_first(self, key: str) -> Optional[str]:
        """Return the first path whose last segment matches *key*, or ``None``.

        Calls :meth:`find_all` and returns its first element.  Intended as
        a convenience shortcut when only one match is expected.

        Args:
            key (str): Exact segment name to search for.

        Returns:
            Optional[str]: The first matching full path, or ``None`` when
            *key* is not present anywhere in the tree.

        Example::

            path = reader.find_first("Velocity")
            if path:
                vel = reader.get_element(path)
        """
        matches = self.find_all(key)
        return matches[0] if matches else None
