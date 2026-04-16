Installation
============

Requirements
------------

- Python 3.10 or newer
- Linux or macOS (Windows via WSL2)
- An HPC cluster running SLURM or SGE, **or** a local machine for testing
- Apptainer / Singularity (only required for container-based pipelines)

----

For users — install from PyPI
------------------------------

The simplest way to get launchcontainers is from PyPI:

.. code-block:: console

   pip install launchcontainers

Verify the install:

.. code-block:: console

   lc --help
   checker --help

To upgrade to the latest release:

.. code-block:: console

   pip install --upgrade launchcontainers

----

For developers — install from source with Poetry
-------------------------------------------------

If you want to contribute or run the latest development version, use
`Poetry <https://python-poetry.org>`_ to manage the virtual environment.

**Step 1 — Install pipx and Poetry** (once per machine):

``pipx`` installs Poetry into its own isolated environment so it never
conflicts with project dependencies.

.. code-block:: console

   pip install pipx
   pipx install poetry
   poetry --version   # verify

**Step 2 — Clone the repository:**

.. code-block:: console

   git clone https://github.com/garikoitz/launchcontainers.git
   cd launchcontainers

**Step 3 — Create the virtual environment and install:**

.. code-block:: console

   poetry env use python3
   poetry install

**Step 4 — Activate the environment:**

.. code-block:: console

   poetry shell

All subsequent ``lc`` and ``checker`` commands can be run directly inside
the Poetry shell, or prefixed with ``poetry run`` without activating it:

.. code-block:: console

   poetry run lc --help

----

Entry points
------------

Both install paths register the same two command-line tools:

.. code-block:: console

   lc        # main pipeline launcher  (prepare / run / qc)
   checker   # analysis integrity checker

----

Optional: build the documentation locally
------------------------------------------

.. code-block:: console

   cd docs && make html
   open _build/html/index.html

Requires ``sphinx`` and ``pydata-sphinx-theme``, which are included in the
Poetry dev dependencies. If you installed from PyPI, install them separately:

.. code-block:: console

   pip install sphinx pydata-sphinx-theme
