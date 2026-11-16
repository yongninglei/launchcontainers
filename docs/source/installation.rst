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
`uv <https://docs.astral.sh/uv/>`_ to manage the virtual environment.

**Step 1 — Install uv** (once per machine):

.. code-block:: console

   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv --version   # verify

**Step 2 — Clone the repository:**

.. code-block:: console

   git clone https://github.com/garikoitz/launchcontainers.git
   cd launchcontainers

**Step 3 — Create the virtual environment and install:**

.. code-block:: console

   uv sync   # creates .venv from pyproject.toml + uv.lock

**Step 4 — Run the entry points:**

Prefix commands with ``uv run``; no activation is needed:

.. code-block:: console

   uv run lc --help
   uv run checker --help

From outside the repo, point uv at it with ``--project``:

.. code-block:: console

   uv run --project /path/to/launchcontainers lc --help

On HPC compute nodes, run ``uv sync`` once on the login node and call
``/path/to/launchcontainers/.venv/bin/lc`` directly in the jobs. Many jobs
running ``uv run`` in parallel would race on the same ``.venv``.

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
