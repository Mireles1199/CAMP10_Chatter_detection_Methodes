 [Quick Start →](quickstart.md){ .md-button }
# Installation

## Requirements

| Requirement | Minimum version |
|---|---|
| Python | 3.9 |
| NumPy | 1.24 |
| Matplotlib | 3.6 |

---

## 1. Get the Source

The package lives in the `MaxEnt_SPRT/` subfolder of the project.  
Clone or download the repository so you have the folder locally:

```
MaxEnt_SPRT/
├── src/
│   └── MaxEnt_SPRT/     ← Python package
├── pyproject.toml
└── examples/
```

---

## 2. Install in Editable Mode (Recommended)

Open a terminal, navigate to the folder that contains `pyproject.toml`, and run:

```bash
cd path/to/MaxEnt_SPRT
pip install -e .
```

The `-e` flag means *editable* — any change you make to the source files in `src/` is immediately reflected without reinstalling.

### Verify the install

```python
import MaxEnt_SPRT
print(MaxEnt_SPRT.__version__)   # should print '0.2.0'
```

---

## 3. Install in a Virtual Environment (Best Practice)

```bash
# Create environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/macOS)
source .venv/bin/activate

# Install package
pip install -e .
```

---

## 4. Build the Documentation

The project uses two connected documentation layers:

- **MkDocs + Material** for theory, tutorials, and user-oriented navigation
- **Sphinx + AutoAPI** for detailed technical and developer documentation

Build the main site:

```bash
pip install -r requirements-docs.txt
mkdocs serve            # Local preview at http://127.0.0.1:8000
mkdocs build            # Build static site to site/
```

Build the technical site:

```bash
pip install -r requirements-sphinx.txt
sphinx-build -b html sphinx_docs/source docs/technical
```

Build both together:

```powershell
.\build_docs.ps1
```

---

## 5. Troubleshooting

### `ModuleNotFoundError: No module named 'MaxEnt_SPRT'`
You are running Python outside the virtual environment, or `pip install -e .` was not run.  
Re-activate the environment and re-run the install command.

### `ValueError: fs/fr must be an integer`
Your sampling frequency $f_s$ and rotational frequency $f_r = \text{rpm}/60$ do not yield an integer ratio.  
Adjust `rpm` or resample your signal so that $f_s \bmod f_r = 0$.

### Math does not render in the docs
Make sure you have internet access (MathJax is loaded from CDN) and that you installed `mkdocs-material>=9.0`.

---

[Quick Start →](quickstart.md)

