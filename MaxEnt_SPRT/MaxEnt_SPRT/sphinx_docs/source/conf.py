from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

project = "MaxEnt-SPRT Technical Documentation"
author = "Enrique Mireles"
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "autoapi.extension",
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_ivar = True

autoapi_type = "python"
autoapi_dirs = [str(SRC / "MaxEnt_SPRT")]
autoapi_root = "autoapi"
autoapi_keep_files = False
autoapi_member_order = "bysource"
autoapi_python_class_content = "both"
autoapi_ignore = [
    "*constants_user.py",
    "*functions_user.py",
    "*classes_user.py",
    "*original_imports.py",
]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

html_theme = "furo"
html_title = project
html_static_path = ["_static"]
html_css_files = ["custom.css"]
suppress_warnings = ["ref.python"]
