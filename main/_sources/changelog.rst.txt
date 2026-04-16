Changelog
=========

0.4.8
-----

- **BasePrepare**: introduced ``launchcontainers/prepare/base_prepare.py`` with
  a ``BasePrepare`` base class shared by all prepare pipelines.  Provides
  ``basedir`` / ``bidsdir`` properties and the ``write_example_config`` /
  ``_example_config_dict`` classmethod pair so every subclass can auto-generate
  an annotated ``lc_config_example.yaml`` without duplicating boilerplate.

- **GLMPrepare** now subclasses ``BasePrepare``.  Removed the redundant
  ``basedir`` / ``bidsdir`` properties and replaced the ``@staticmethod
  write_example_config`` with ``@classmethod _example_config_dict``.

- **GLMPrepare — Word-Center (WC) mode**: new ``output_bids`` and
  ``output_bids_dir`` properties.  When ``is_WC=True`` the prepare workflow
  writes all outputs (events TSVs, BIDS bold symlinks, fMRIprep bold symlinks,
  and their JSON sidecars) into a separate BIDS-mirrored directory tree rooted
  at ``<basedir>/<output_bids>`` (e.g. ``BIDS_WC/``), including a
  ``derivatives/<fmriprep_analysis_name>/`` sub-tree.

- **Symlinks now include JSON sidecars**: both ``gen_bids_bold_symlinks`` and
  ``gen_fmriprep_bold_symlinks`` create a matching ``.json`` symlink alongside
  every NIfTI / GIFTI symlink.

- **Mapping TSV moved to vistadisplog dir**: ``PRFPrepare.parse_prf_mat`` now
  writes ``sub-<sub>_ses-<ses>_desc-mapping_PRF_acqtime.tsv`` to the
  vistadisplog source directory instead of the BIDS ``func/`` directory.
  ``GLMPrepare._load_mapping_tsv`` reads from the same location.

- **Bug fix**: ``run_glm_prepare`` was not forwarding the ``layout`` argument
  to ``run_prf_glm_prepare``, causing an ``AttributeError: 'NoneType' object
  has no attribute 'get'`` at runtime.

- **Type annotations**: added ``TYPE_CHECKING`` guard in ``glm_prepare.py``
  for ``pandas.DataFrame`` and ``bids.BIDSLayout``; all public wrapper
  functions now carry full type signatures.

- **Docs**: added tutorial sub-pages ``prepare_glm.rst``, ``prepare_dwi.rst``,
  ``prepare_ret.rst`` under the Tutorial section.  Rewrote ``extending.rst``
  to document the actual ``BasePrepare`` pattern with step-by-step guidance.
  Updated ``configuration.rst`` ``container_specific`` section and ``api.rst``
  fMRI-GLM preparation section.

0.4.7
-----

- **Logging**: replaced ad-hoc output redirection with a structured logging
  architecture. ``log_setup.py`` now provides ``_LoggingConsole``, a Rich
  ``Console`` subclass that auto-forwards every ``console.print()`` call to a
  Python ``logging.Logger``. Two log files are written per CLI command and
  copied into the analysis directory on completion:

  - ``<cmd>_<timestamp>.log`` — all messages (DEBUG and above)
  - ``<cmd>_<timestamp>.err`` — warnings and errors only

- **gen_jobscript package**: replaced the single ``gen_launch_cmd.py`` module
  with a ``gen_jobscript/`` package. The orchestrator (``__init__.py``) routes
  to per-type command builders:

  - ``gen_container_cmd.py`` — Apptainer/Singularity containers (existing logic)
  - ``gen_matlab_cmd.py`` — MATLAB script launcher (stub, raises ``NotImplementedError``)
  - ``gen_py_cmd.py`` — Python script launcher (stub, raises ``NotImplementedError``)

- **clusters/local.py**: renamed ``job_scheduler.py`` to ``local.py``. Added
  ``launch_serial()`` and ``launch_parallel()`` functions controlled by the new
  ``launch_mode`` yaml key. Each subprocess now runs inside ``bash -l`` so that
  ``module load`` commands work correctly. Parallel mode supports a
  ``mem_per_job`` memory ceiling enforced via ``resource.setrlimit`` on each
  worker process.

- **host_options.local** yaml keys updated: ``launch_mode`` (``serial`` /
  ``parallel``), ``max_workers`` (concurrent container limit), ``mem_per_job``
  (per-worker memory cap). Old Dask-era keys (``njobs``, ``memory_limit``,
  ``threads_per_worker``) are no longer used.

0.4.6
-----

- Added ``GLMPreparer`` and ``BasePreparer`` abstract class hierarchy for
  analysis-based (fMRI) pipelines.
- Introduced ``PREPARER_REGISTRY`` for zero-touch extension of new analysis types.
- ``cli.py``: two-path dispatch — legacy DWI path untouched; new registry path
  for analysis-based pipelines.
- Added ``glm_specific`` config section and ``glm_config_template.yaml``.

0.4.3
-----

- Improved temporal proximity matching for sbref↔bold pairing (180-second window).
- Fixed DWI file mapping for PA/AP direction naming.
- Fixed datetime parsing for ISO timestamp formats in scans.tsv validation.
- Fixed bidirectional file matching to always produce ``_dwi.nii.gz`` suffixes.

0.4.0
-----

- Initial public release of the ``lc`` CLI with ``prepare``, ``run``, ``qc``,
  ``copy_configs``, ``gen_subses``, and ``create_bids`` subcommands.
- SLURM (DIPC) and SGE (BCBL) scheduler support.
- Container-based pipeline support: ``anatrois``, ``freesurferator``,
  ``rtppreproc``, ``rtp-pipeline``, ``rtp2-preproc``, ``rtp2-pipeline``.
