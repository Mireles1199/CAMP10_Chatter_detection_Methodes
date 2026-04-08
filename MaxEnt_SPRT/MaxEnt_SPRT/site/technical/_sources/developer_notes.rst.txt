Developer Notes
===============

.. raw:: html

    <p>
       <a class="nav-link-btn" href="autoapi/index.html">Open Full API</a>
       <a class="nav-link-btn" href="../index.html">Back to User Docs</a>
    </p>

Build commands
--------------

Install the technical documentation dependencies:

.. code-block:: bash

   pip install -r requirements-sphinx.txt

Build the Sphinx technical site into the folder consumed by MkDocs:

.. code-block:: bash

   sphinx-build -b html sphinx_docs/source docs/technical

Build both sites together:

.. code-block:: powershell

   .\build_docs.ps1

Recommended workflow
--------------------

1. Update docstrings in ``src/MaxEnt_SPRT``.
2. Rebuild the Sphinx technical site.
3. Rebuild or serve MkDocs.
4. Verify the user-facing and technical pages are still aligned.

Docstring guidance
------------------

The current setup supports both Google-style and NumPy-style docstrings. For consistent rendering across MkDocs and Sphinx, keep these conventions:

- start with a short summary line
- document parameters and returns explicitly
- use meaningful names and avoid undocumented flags
- explain side effects or assumptions when the API is not obvious

Back to main site
-----------------

Return to the main user docs here: `Main documentation <../index.html>`_.

