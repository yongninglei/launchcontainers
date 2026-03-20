CLI Reference
=============

All functionality is accessed through the ``lc`` command, which has several
subcommands. Run ``lc --help`` or ``lc <subcommand> --help`` for inline help.

Global options
--------------

.. option:: -q, --quiet

   Enable quiet mode. Log level is set to ``CRITICAL``; only fatal errors
   are printed.

----

lc prepare
----------

Set up the analysis directory structure and generate launch scripts.

.. code-block:: console

   lc prepare -lcc <config> -ssl <subseslist> [-cc <container_config>]

.. option:: -lcc, --lc_config <path>

   Path to the main ``lc_config.yaml`` file.

.. option:: -ssl, --sub_ses_list <path>

   Path to the ``subseslist.tsv`` file listing subjects and sessions to process.

.. option:: -cc, --container_specific_config <path>

   Path to the container-specific JSON config file (e.g. ``rtppreproc.json``).
   Required for container-based pipelines; not needed for analysis-based
   pipelines (``glm``, ``prf``).

**What it does:**

1. Reads and validates all input configs.
2. Checks that required input files exist for each subject/session.
3. Creates the analysis directory under ``BIDS/derivatives/``.
4. Writes frozen copies of all configs to ``analysis-dir/config/``.
5. Generates per-subject launch scripts in ``analysis-dir/scripts/``.
6. Prints a Rich summary table (ready / fixed / warn / blocked per subject).

----

lc run
------

Submit the prepared launch scripts to the HPC scheduler.

.. code-block:: console

   lc run -w <workdir> [--run_lc]

.. option:: -w, --workdir <path>

   Path to the prepared analysis directory (the one created by ``lc prepare``).

.. option:: -R, --run_lc

   Actually submit jobs. Without this flag, ``lc run`` performs a dry run and
   only prints the commands that would be executed.

----

lc qc
-----

Validate outputs after jobs have finished.

.. code-block:: console

   lc qc -w <workdir>

.. option:: -w, --workdir <path>

   Path to the analysis directory to check.

**What it produces:**

- A pass/fail summary table printed to the terminal.
- ``analysis-dir/logs/qc_TIMESTAMP.log`` with details of any missing files.
- ``analysis-dir/failed_subseslist.tsv`` listing all subject/sessions that
  failed, ready to be passed back to ``lc prepare``.

----

lc copy_configs
---------------

Copy the bundled example config files to a working directory.

.. code-block:: console

   lc copy_configs -o <output_path>

.. option:: -o, --output <path>

   Destination directory. Typically ``basedir/code/``.

----

lc gen_subses
-------------

Generate a ``subseslist.tsv`` from an existing directory tree.

.. code-block:: console

   lc gen_subses -b <basedir> -n <filename> [-o <output_dir>]

.. option:: -b, --basedir <path>

   Directory containing ``sub-*`` / ``ses-*`` folders to scan.

.. option:: -n, --name <filename>

   Output filename for the generated subseslist (e.g. ``subseslist.tsv``).

.. option:: -o, --output_dir <path>

   Directory in which to write the file. Defaults to ``basedir``.

----

lc create_bids
--------------

Create a minimal fake BIDS directory structure for testing.

.. code-block:: console

   lc create_bids -cbc <config> -ssl <subseslist>

.. option:: -cbc, --creat_bids_config <path>

   Path to the ``create_bids`` config YAML file.

.. option:: -ssl, --sub_ses_list <path>

   Path to the subseslist to use when generating the fake structure.

----

lc zip_configs
--------------

Archive the current example configs back into the repo (developer utility).

.. code-block:: console

   lc zip_configs
