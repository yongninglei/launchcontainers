.. _prepare_glm:

Prepare — fMRI GLM
==================

What are we preparing?
-----------------------

Before the GLM container can run, two things must be in place for every
subject and session:

1. **A BIDS** ``events.tsv`` **file for each run** — telling the model *when*
   each stimulus appeared, *how long* it lasted, and *what condition* it
   belonged to.
2. **Bold NIfTIs (and their JSON sidecars) with consistent, normalised
   filenames** — both the raw BIDS bold and the fMRIprep preprocessed bold,
   renamed so that the task-run label used by the GLM (e.g.
   ``task-fixnonstop_run-01``) is identical across the BIDS tree and the
   derivatives tree.

The challenge is that the experiment-recording software (vistadisplog) stores
stimulus timing in ``.mat`` log files, not in BIDS, and the original BIDS bold
filenames use the raw task labels from the scanner (e.g.
``task-fixRW_run-03``).  The prepare stage bridges this gap entirely through
symlinks and TSV files — **no data are ever copied or modified**.

How the pipeline works
-----------------------

Step 1 — Parse the log files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~launchcontainers.prepare.prf_prepare.PRFPrepare` scans the
vistadisplog directory
``<bidsdir>/sourcedata/vistadisplog/sub-<sub>/ses-<ses>/`` for ``20*.mat``
files.  For each log file it:

* reads ``params.loadMatrix`` to identify the stimulus file and therefore the
  task and condition;
* subtracts 6 minutes from the log filename timestamp to recover the run
  *start* time (the log filename records the *end* time);
* assigns a per-task run counter and a normalised GLM task-run label
  (``fixnonstop`` or ``fixblock``).

The result is written as a mapping TSV
(``sub-<sub>_ses-<ses>_desc-mapping_PRF_acqtime.tsv``) into the same
vistadisplog directory.  This file is the authoritative record of which log
belongs to which run.

Step 2 — Write ``events.tsv`` files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~launchcontainers.prepare.glm_prepare.GLMPrepare.gen_events_tsv_vistadisplog`
iterates over the same log files and writes one BIDS ``events.tsv`` per run
with ``onset``, ``duration``, and ``trial_type`` columns derived from the
stimulus filename.

For block designs (task label contains ``block``) the run is split into
alternating ``baseline`` / ``<lgn>_fixblock`` epochs of ``block_time``
seconds each.  For event designs the whole run is a single event.

Step 3 — Match logs → NIfTIs via acquisition time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~launchcontainers.prepare.glm_prepare.GLMPrepare.gen_bids_bold_symlinks`
uses pyBIDS to list all ``*_bold.nii.gz`` files for the subject/session,
reads the ``AcquisitionTime`` field from each JSON sidecar, and matches it
against the ``acq_time`` column in the mapping TSV (within a ±120 s window).
Once matched, it creates a symlink with the normalised GLM task-run name and
also symlinks the accompanying JSON sidecar.

Step 4 — Mirror fMRIprep bold with the same names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~launchcontainers.prepare.glm_prepare.GLMPrepare.gen_fmriprep_bold_symlinks`
uses the matched list from Step 3 to locate the corresponding preprocessed
bold under the fMRIprep derivatives directory (filtered by ``space`` and file
suffix) and symlinks it — again using the same GLM task-run name.  JSON
sidecars are symlinked whenever they exist.

Word-Center (WC) mode
----------------------

When ``is_WC: True`` the output is written into a separate directory tree
rooted at ``<basedir>/<output_bids>`` (e.g. ``BIDS_WC``) instead of the
original BIDS tree.  The structure mirrors standard BIDS:

.. code-block:: text

   <basedir>/
     <output_bids>/                               # e.g. BIDS_WC/
       sub-<sub>/ses-<ses>/func/
         *_task-fixnonstop_run-01_bold.nii.gz      → symlink → original BIDS
         *_task-fixnonstop_run-01_bold.json         → symlink → original BIDS JSON
         *_task-fixnonstop_run-01_events.tsv        (written directly)
       derivatives/
         <fmriprep_analysis_name>/
           sub-<sub>/ses-<ses>/func/
             *_task-fixnonstop_run-01_*_bold.func.gii  → symlink → fMRIprep
             *_task-fixnonstop_run-01_*_bold.json       → symlink → fMRIprep JSON

Relevant config keys
---------------------

All keys live under ``container_specific.fMRI-GLM`` in the launchcontainers
YAML:

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Key
     - Default
     - Description
   * - ``is_WC``
     - —
     - Enable Word-Center mode.
   * - ``output_bids``
     - ``None``
     - Name of the output BIDS root directory (WC mode only).
   * - ``block_time``
     - ``10.0``
     - Duration in seconds of each block epoch.
   * - ``fmriprep_analysis_name``
     - —
     - Derivatives subdirectory name of the fMRIprep run.
   * - ``task``
     - ``None``
     - Task label filter (``None`` = all tasks).
   * - ``space``
     - ``fsnative``
     - Output space for the preprocessed bold (``fsnative``, ``T1w``, …).
   * - ``start_scans``
     - ``0``
     - Number of non-steady TRs to discard at the start of each run.
   * - ``contrast_yaml``
     - ``None``
     - Path to the contrast definition YAML consumed by the GLM container.

API reference
--------------

See :ref:`api_ref` for the full auto-generated documentation of all classes
and functions in this module:

* :class:`launchcontainers.prepare.glm_prepare.GLMPrepare`
* :func:`launchcontainers.prepare.glm_prepare.run_glm_prepare`
* :func:`launchcontainers.prepare.glm_prepare.run_prf_glm_prepare`
* :class:`launchcontainers.prepare.prf_prepare.PRFPrepare`
