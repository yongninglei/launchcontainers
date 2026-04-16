Configuration Reference
=======================

All configuration lives in a single YAML file (``lc_config.yaml``) with
three top-level sections, plus a separate ``subseslist.tsv``.

Copy the bundled examples to get started:

.. code-block:: console

   lc copy_configs --output /path/to/basedir/code/

----

general
-------

Project-level settings shared by all pipelines.

.. code-block:: yaml

   general:
     basedir: /scratch/tlei/VOTCLOC
     bidsdir_name: BIDS
     container: rtppreproc
     analysis_name: main
     host: DIPC
     deriv_layout: legacy
     force: True

.. list-table::
   :header-rows: 1
   :widths: 25 12 55

   * - Key
     - Type
     - Description
   * - ``basedir``
     - str
     - Absolute path to the project root directory.
   * - ``bidsdir_name``
     - str
     - Name of the BIDS directory (one level under ``basedir``).
   * - ``container``
     - str
     - Pipeline or analysis type to run. See :doc:`concepts` for the full list.
   * - ``analysis_name``
     - str
     - Name tag appended to the derivatives folder (e.g. ``main``, ``pilot``).
   * - ``host``
     - str
     - Compute host: ``DIPC``, ``BCBL``, or ``local``.
   * - ``deriv_layout``
     - str
     - ``legacy``: ``derivatives/<pipeline>-<version>/analysis-<name>/``.
       ``bids``: ``derivatives/<pipeline>-<version>_<name>/``.
   * - ``force``
     - bool
     - Overwrite existing files in the analysis directory.

----

container_specific
------------------

Parameters passed to the container or analysis pipeline.  Keyed by container
name; only the block matching ``general.container`` is used at runtime.

.. note::

   **Current state of this section:**

   * The **RTP2 DWI pipeline** blocks (``rtppreproc``, ``rtp2-preproc``,
     ``freesurferator``, ``anatrois``, etc.) are documented manually below.
     These configs are hand-written by the user and consumed directly by the
     container prepare logic in
     :mod:`~launchcontainers.prepare.dwi_prepare`.

   * The **fMRI-GLM prepare** pipeline (``fMRI-GLM``) already has a built-in
     config generator.  Running
     :func:`~launchcontainers.prepare.glm_prepare.run_glm_prepare` without a
     config (or calling ``lc prepare`` with no ``container_specific`` block)
     will write a fully annotated example ``lc_config_example.yaml`` to the
     current directory.  See :ref:`prepare_glm` for the full key reference.

   * In the near future, **every pipeline class** will expose the same
     ``write_example_config`` helper so that users never need to write
     ``container_specific`` blocks by hand.  The RTP2 DWI pipelines will be
     the next to receive this treatment.

rtppreproc / rtp2-preproc
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   container_specific:
     rtppreproc:
       version: 1.2.0-3.0.3
       precontainer_anat: freesurferator_0.2.1_7.4.1
       anat_analysis_name: main
       separated_shell_files: True
       rpe: True

.. list-table::
   :header-rows: 1
   :widths: 30 55

   * - Key
     - Description
   * - ``version``
     - Container version tag; used in the derivatives folder name.
   * - ``precontainer_anat``
     - Name of the anatomical derivatives folder (FreeSurfer / anatrois output).
   * - ``anat_analysis_name``
     - Analysis name inside the anatomical derivatives folder.
   * - ``separated_shell_files``
     - ``True`` if multi-shell DWI sequences are stored in separate NIfTI files.
   * - ``rpe``
     - ``True`` if a reverse phase-encoding (AP) acquisition was collected.

freesurferator / anatrois
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   container_specific:
     freesurferator:
       version: 0.2.1_7.4.1
       pre_fs: True
       prefs_dir_name: anatrois_4.5.3-7.3.2
       prefs_analysis_name: "01"
       prefs_zipname: '^freesurferator.*.zip$|^anatrois.*\.zip$'
       annotfile: /path/to/annots.zip
       use_src_session: T01

.. list-table::
   :header-rows: 1
   :widths: 30 55

   * - Key
     - Description
   * - ``pre_fs``
     - Whether a pre-run FreeSurfer result is available.
   * - ``prefs_dir_name``
     - Derivatives folder containing the pre-run FreeSurfer.
   * - ``prefs_analysis_name``
     - Analysis name inside that folder.
   * - ``prefs_zipname``
     - Regex pattern to locate the pre-run zip file.
   * - ``annotfile``
     - Optional path to a FreeSurfer annotation zip.
   * - ``use_src_session``
     - Session ID to use as the T1 source (e.g. ``T01``) for multi-session subjects.

fMRI-GLM
~~~~~~~~~

.. tip::

   Rather than writing this block by hand, let the pipeline generate an
   annotated example for you:

   .. code-block:: console

      lc prepare --lc_config lc_config.yaml   # omit container_specific.fMRI-GLM
      # → writes lc_config_example.yaml in the current directory

   See :ref:`prepare_glm` for a full explanation of every key and the
   prepare workflow.

.. code-block:: yaml

   container_specific:
     fMRI-GLM:
       is_WC: False
       output_bids: BIDS_WC          # WC mode only
       fmriprep_analysis_name: fmriprep-25.1.4
       task: null
       start_scans: 5
       space: fsnative
       contrast_yaml: /path/to/contrast.yaml
       output_name: glm_output
       slice_timing_ref: 0.5
       use_smoothed: False
       dry_run: False
       sm: null
       mask: null
       selected_runs: null
       power_analysis: False
       n_iterations: 10
       seed: 42
       total_runs: 10

----

host_options
------------

HPC scheduler settings. One block per host; only the block matching
``general.host`` is used.

DIPC — SLURM
~~~~~~~~~~~~~

.. code-block:: yaml

   host_options:
     DIPC:
       use_module: True
       apptainer: Apptainer
       manager: slurm
       system: scratch
       tmpdir: /scratch/tlei/tmp
       mount_options: ['/scratch']
       job_name: votcloc
       cores: 20
       memory: 32G
       partition: general
       qos: regular
       walltime: '10:00:00'

BCBL — SGE
~~~~~~~~~~

.. code-block:: yaml

   host_options:
     BCBL:
       use_module: True
       apptainer: apptainer/latest
       manager: sge
       mount_options: ['/bcbl', '/tmp', '/scratch']
       job_name: votcloc
       cores: 8
       memory: 32G
       queue: long.q
       walltime: '25:30:00'

local
~~~~~

.. code-block:: yaml

   host_options:
     local:
       use_module: True           # set False if apptainer is in PATH without module load
       apptainer: apptainer/latest
       mount_options: ['/bcbl', '/tmp', '/scratch', '/export']
       manager: local
       launch_mode: parallel      # 'serial' or 'parallel'
       max_workers: 2             # parallel only: max concurrent containers
       mem_per_job: 32g           # parallel only: memory cap per worker (e.g. 32g, 512m)

.. list-table::
   :header-rows: 1
   :widths: 25 12 55

   * - Key
     - Type
     - Description
   * - ``use_module``
     - bool
     - ``True`` if the ``module load`` command is needed to make ``apptainer``
       available (typical on BCBL workstations). ``False`` if apptainer is
       already in ``PATH``.
   * - ``apptainer``
     - str
     - Module name passed to ``module load``. Ignored when ``use_module: False``.
   * - ``mount_options``
     - list
     - Filesystem paths to bind-mount into every container.
   * - ``launch_mode``
     - str
     - ``serial``: containers run one after another.
       ``parallel``: containers run concurrently up to ``max_workers``.
   * - ``max_workers``
     - int
     - Maximum number of containers running at the same time (parallel mode
       only). Rule of thumb: ``floor(total_cores / cpus_needed_per_job)``.
   * - ``mem_per_job``
     - str
     - Memory ceiling per worker process enforced by the OS (``resource.setrlimit``).
       Accepts human-readable units: ``32g``, ``512m``, ``1t``. The apptainer
       container inherits this limit as a child process. Omit or set to ``null``
       for no limit.

----

subseslist.txt
--------------

A comma-separated file specifying which subjects and sessions to process.

.. code-block:: text

   sub,ses,RUN
   01,T01,True
   01,T02,True
   02,T01,False

.. list-table::
   :header-rows: 1
   :widths: 12 60

   * - Column
     - Description
   * - ``sub``
     - Subject ID without the ``sub-`` prefix.
   * - ``ses``
     - Session ID without the ``ses-`` prefix.
   * - ``RUN``
     - ``True`` to include this row in the current prepare/run cycle.

.. tip::

   Generate a subseslist from an existing BIDS directory automatically:

   .. code-block:: console

      lc gen_subses --basedir /path/to/BIDS --name subseslist.tsv
