# Developer Hub

Entry point for contributors and maintainers. Use the cards below to navigate the technical documentation generated from source code and docstrings.

[Open Technical Docs](technical/index.html){ .md-button .md-button--primary }

---

## Quick Access

<div class="grid cards" markdown>

- :material-hammer-wrench: **Technical Docs Home**

    ---

    Landing page of the Sphinx technical documentation.

    [Open](technical/index.html)

- :material-map-outline: **Technical Overview**

    ---

    Suggested reading order for implementation-level understanding.

    [Open](technical/overview.html)

- :material-sitemap: **Internal Architecture**

    ---

    Package structure, module boundaries, and execution path.

    [Open](technical/architecture.html)

- :material-transit-connection-variant: **Execution Flow**

    ---

    Step-by-step runtime path from raw signal to chatter decision.

    [Open](technical/execution_flow.html)

- :material-layers-triple-outline: **Module Responsibilities**

    ---

    Clear ownership map of modules and where to implement changes.

    [Open](technical/module_responsibilities.html)

- :material-notebook-edit-outline: **Developer Notes**

    ---

    Build workflow, maintenance tips, and docstring conventions.

    [Open](technical/developer_notes.html)

- :material-code-json: **Developer Examples**

    ---

    Practical code snippets for extension and validation workflows.

    [Open](technical/developer_examples.html)

- :material-function-variant: **Full AutoAPI Reference**

    ---

    Complete technical API generated from your source and docstrings.

    [Open](technical/autoapi/index.html)

</div>

---

## Two Documentation Layers

| Site | Purpose |
|---|---|
| **MkDocs** (this site) | Theory, user guides, quickstart, and method explanation |
| **Sphinx** (technical) | Detailed module pages, internal architecture, and full API from docstrings |

---

## Build Commands

Build only Sphinx:

```bash
pip install -r requirements-sphinx.txt
python -m sphinx -b html sphinx_docs/source docs/technical
```

Build both sites at once:

```powershell
.\build_docs.ps1
```

> After changing docstrings, re-run `build_docs.ps1` to keep both sites in sync.
