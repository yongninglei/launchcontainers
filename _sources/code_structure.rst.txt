Code Structure
==============

Repository layout
-----------------

.. code-block:: text

   launchcontainers/
   ├── cli.py                       ← entry point; argparse subcommands
   ├── do_prepare.py                ← prepare orchestration (DWI legacy path)
   ├── do_launch.py                 ← job submission
   ├── do_qc.py                     ← quality control
   ├── utils.py                     ← shared utilities (read_yaml, read_df, …)
   ├── log_setup.py                 ← Rich console + Python logging setup
   │
   ├── prepare/
   │   ├── __init__.py              ← PREPARER_REGISTRY
   │   ├── base_preparer.py         ← BasePreparer, PrepIssue, PrepResult
   │   ├── glm_preparer.py          ← GLMPreparer
   │   ├── prepare_dwi.py           ← DWI container preparation (legacy)
   │   └── dwi_prepare_input.py
   │
   ├── gen_jobscript/               ← job-command generation (one sub-module per job type)
   │   ├── __init__.py              ← gen_launch_cmd() orchestrator; routes by container type
   │   ├── gen_container_cmd.py     ← Apptainer/Singularity command builder
   │   ├── gen_matlab_cmd.py        ← MATLAB script command builder (stub)
   │   └── gen_py_cmd.py            ← Python script command builder (stub)
   │
   ├── clusters/
   │   ├── slurm.py                 ← SLURM job submission helpers
   │   ├── sge.py                   ← SGE job submission helpers
   │   └── local.py                 ← local serial/parallel execution (concurrent.futures)
   │
   ├── check/
   │   ├── general_checks.py        ← shared file-existence checks
   │   └── check_dwi_pipelines.py   ← DWI-specific checks
   │
   └── helper_function/
       ├── gen_subses.py            ← subseslist generation
       ├── create_bids.py           ← fake BIDS structure for testing
       ├── copy_configs.py          ← copy example configs to working dir
       └── zip_example_config.py    ← archive configs (developer utility)

Logging architecture
---------------------

Every CLI command sets up two log files before doing any work:

.. code-block:: text

   analysis_dir/
   ├── prepare_log/
   │   ├── lc_prepare_<timestamp>.log   ← all messages (DEBUG and above)
   │   └── lc_prepare_<timestamp>.err   ← warnings and errors only
   ├── run_log/
   │   ├── lc_run_<timestamp>.log
   │   └── lc_run_<timestamp>.err
   └── qc_log/
       ├── qc_<timestamp>.log
       └── qc_<timestamp>.err

The key component is ``_LoggingConsole`` in ``log_setup.py``, a subclass of
Rich's ``Console`` that intercepts every ``console.print()`` call and
simultaneously forwards it to a Python ``logging.Logger``:

- ``style="red"`` or ``style="bold red"`` → ``logger.error()``
- ``style="yellow"`` → ``logger.warning()``
- everything else → ``logger.info()``

This means no code change is needed throughout the codebase — all existing
``console.print()`` calls are captured automatically. The ``set_log_files()``
function in ``log_setup.py`` attaches the two ``FileHandler`` instances; it is
called once per CLI command from ``cli.py`` before any work starts.

``gen_jobscript`` package
--------------------------

Command generation is organised as a package so that different job types
(Apptainer containers, Python scripts, MATLAB scripts) each have their own
module. The orchestrator in ``__init__.py`` reads ``lc_config["general"]["container"]``
and routes to the appropriate builder:

.. code-block:: python

   # gen_jobscript/__init__.py
   if container in _CONTAINER_JOBS:
       _gen_cmd = gen_RTP2_cmd          # apptainer
   elif container == "matlab":
       _gen_cmd = gen_matlab_cmd
   elif container == "python":
       _gen_cmd = gen_py_cmd

Adding support for a new job type means creating a new module and adding one
``elif`` branch — the orchestrator and ``do_launch.py`` need no other changes.

Entry points (``pyproject.toml``)
-----------------------------------

.. code-block:: toml

   [tool.poetry.scripts]
   lc      = "launchcontainers.cli:main"
   checker = "analysis_checker.check_analysis_integrity:main"

Dispatch logic in ``cli.py``
------------------------------

``lc prepare`` uses a two-path dispatch based on the ``container`` key:

.. code-block:: python

   if container in PREPARER_REGISTRY:
       # New-style: BasePreparer subclass owns all preparation logic
       cls = PREPARER_REGISTRY[container]
       preparer = cls(config=lc_config, subseslist=subsesrows, output_root=output_root)
       preparer.run(dry_run=False)
   else:
       # Legacy DWI path (untouched)
       do_prepare.main(parse_namespace, analysis_dir)

This means all DWI container pipelines use the original ``do_prepare`` code,
while analysis-based pipelines (``glm``, ``prf``, …) go through the
``BasePreparer`` class hierarchy. Adding a new analysis type never requires
modifying the legacy path.

BasePreparer class hierarchy
-----------------------------

.. code-block:: text

   BasePreparer  (abstract)
   ├── GLMPreparer
   └── PRFPreparer  (planned)

``BasePreparer`` owns all orchestration: directory creation, config
freezing, issue processing, Rich summary printing, and log writing.
Subclasses implement only two abstract methods:

- ``check_requirements(sub, ses) → list[PrepIssue]``
- ``generate_run_script(sub, ses, analysis_dir) → Path``

Data classes
------------

.. code-block:: python

   @dataclass
   class PrepIssue:
       sub:      str
       ses:      str
       category: str
       severity: str        # "blocking" | "warn" | "auto_fix"
       message:  str
       fix_fn:   Callable | None = None

   @dataclass
   class PrepResult:
       sub:    str
       ses:    str
       status: str          # "ready" | "fixed" | "warn" | "blocked"
       issues: list[PrepIssue]
