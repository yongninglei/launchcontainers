.. _prepare_prf:

PRF Pipeline — Prepare and Run
===============================

Overview
--------

The population receptive field (PRF) pipeline chains three containers in
sequence. Each container reads from the output of the previous one:

.. code-block:: text

   fMRIPrep output
        ↓
   prfprepare       — projects volumetric BOLD to surface, averages runs
        ↓
   prfanalyze-vista — fits the pRF model (Benson / vista-based)
        ↓
   prfresult        — extracts ROI-level pRF parameters and plots

**Important constraints:**

* ``prfprepare`` and ``prfanalyze-vista`` run on **DIPC**.
* ``prfresult`` only works on the **BCBL server** — do not submit it on DIPC,
  the container will fail.
* fMRIPrep must be completed and its output symlinked under
  ``BIDS/derivatives/fmriprep/analysis-{fp}/`` before any PRF step is run.

----

Pipeline scripts
----------------

Scripts live in ``tests/prepare_prf/`` (preparation) and ``tests/run_prf/``
(SLURM submission).  They must be run **in the numbered order** shown below.

Preparation scripts
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Script
     - What it does
   * - ``00_dcm2niix_ret.py``
     - Converts retinotopy DICOM series (retCB, retRW, retFF, retfixRW, …) to
       BIDS NIfTI.  Classifies each folder as *magnitude*, *phase*, *sbref*, or
       *skip* by folder-name rules.
   * - ``00_drop_ret_run.py``
     - Drops a bad ret run from BIDS func/ and renames its vistadisplog ``.mat``
       target to ``wrongrun_*`` so it is excluded from later steps.  Renumbers
       surviving runs of the same task sequentially.
   * - ``01_prepare_prf_log_and_bold.py``
     - Reads vistadisplog ``.mat`` files, determines task identity by
       **reading the file content** (``params.loadMatrix`` / stimName field),
       creates named ``params.mat`` symlinks, and positionally matches each run
       to its BIDS bold/sbref by acquisition time.  Outputs a per-session table
       for visual inspection.
   * - ``01b_special_fix_sub01_block02_rename.py``
     - **Session-specific script** (sub-01 ses-09 and ses-10 only).  Atomically
       renames ``retfixRWblock02_run-01`` files (BIDS func, BIDS fmap,
       fmriprep func, fmriprep fmap, and sourcedata ``_params.mat``) to the
       correct label.  Each session has its own rule — see
       :ref:`prf_special_notes` for context.
   * - ``02_correct_wc_bold_naming.py``
     - Fixes BIDS file naming for the wordcentre (WC) condition.  Reads the
       task/run label from the ``_params.mat`` **filename** (not the file
       content) and the AcquisitionTime from the symlink target filename.
       Matches mats to BIDS bold files by positional time-order and renames
       mismatched files atomically.
   * - ``03_prepare_prf_subses_json.py``
     - Generates per-subject/session JSON config files for each pipeline step
       (``prfprepare``, ``prfanalyze-vista``, or ``prfresult``).
   * - ``04_clean_orphan_events_tsv.py``
     - Removes ``*_events.tsv`` files from BIDS func directories that have no
       matching ``bold.nii.gz``.  Prevents containers from trying to process
       tasks that were never acquired.

Run scripts (DIPC)
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Script
     - What it does
   * - ``sbatch_prfprepare.sh``
     - Submits one SLURM job per sub/ses pair for ``prfprepare``.  Checks that
       the JSON exists before submitting.
   * - ``sbatch_prfanalyze.sh``
     - Submits one SLURM job per task per sub/ses pair for ``prfanalyze-vista``.
       Auto-detects tasks by globbing existing JSONs — no manual task list
       needed.
   * - ``prfprepare_dipc.sh``
     - SLURM job script called by ``sbatch_prfprepare.sh`` on the compute node.
       After the container finishes, captures the exit code and appends one
       summary line to ``job_results.tsv`` in the log directory.
   * - ``prfanalyze-vista_dipc.sh``
     - SLURM job script called by ``sbatch_prfanalyze.sh`` on the compute node.
       After the container finishes, captures the exit code and appends one
       summary line (including task label) to ``job_results.tsv`` in the log
       directory.

----

Step-by-step workflow
---------------------

Step 0 — DICOM conversion
~~~~~~~~~~~~~~~~~~~~~~~~~

Script: ``00_dcm2niix_ret.py``

Converts raw retinotopy DICOM folders to BIDS NIfTI.  Each run produces four
DICOM folders; the script classifies them by name:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Folder pattern
     - Output label
   * - Neither ``SBRef`` nor ``Pha``
     - ``*_bold.nii.gz``
   * - ``Pha`` only
     - ``*_phase.nii.gz``
   * - ``SBRef`` only
     - ``*_sbref.nii.gz``
   * - Both ``SBRef`` and ``Pha``
     - skipped

.. code-block:: console

   python 00_dcm2niix_ret.py -s 01 --ses 09 -d /path/dicom -o /path/BIDS --dry-run
   python 00_dcm2niix_ret.py -s 01 --ses 09 -d /path/dicom -o /path/BIDS

----

Step 0b — Drop a bad run
~~~~~~~~~~~~~~~~~~~~~~~~~

Script: ``00_drop_ret_run.py``

Use this when a run was aborted or is otherwise unusable.  It:

1. Removes all BIDS func files for the specified ``task``/``run``.
2. Renames the vistadisplog ``.mat`` target to ``wrongrun_*.mat`` (preserves
   the data while excluding it from later steps).
3. Renumbers surviving runs of the same task (e.g. ``run-03`` → ``run-02``).

.. code-block:: console

   # Dry-run — inspect what would change
   python 00_drop_ret_run.py -s 01,09 --task retRW --run 01 --bidsdir /path/BIDS

   # Apply
   python 00_drop_ret_run.py -s 01,09 --task retRW --run 01 --bidsdir /path/BIDS --execute

----

Step 1 — Prepare vistadisplog links and verify BIDS pairing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Script: ``01_prepare_prf_log_and_bold.py``

Reads each vistadisplog ``.mat`` file and identifies the task by **reading
the file content**: it opens the ``.mat`` with ``scipy.io.loadmat`` and
inspects the ``params.loadMatrix`` (stimName) field for keywords such as
``fixRWblock01_``, ``fixRWblock02_``, ``FF_``, ``RW_``, etc.  This is the
**only step that opens the .mat binary** — subsequent scripts work from the
symlink filenames only.

After task identification it creates a named symlink
``sub-{sub}_ses-{ses}_task-{task}_run-{N}_params.mat`` pointing to the
original timestamp-named file, then matches each ``.mat`` to its BIDS
bold/sbref by **positional ordering**: both lists are sorted by acquisition
time and the i-th mat is paired with the i-th bold.  A table is printed for
visual inspection.

.. code-block:: console

   # Dry-run (default): inspect the table
   python 01_prepare_prf_log_and_bold.py -s 01,09 --logdir /path/vistadisplog -b /path/BIDS -v

   # Create symlinks
   python 01_prepare_prf_log_and_bold.py -s 01,09 --logdir /path/vistadisplog -b /path/BIDS --execute

   # Batch
   python 01_prepare_prf_log_and_bold.py -f subseslist.tsv --logdir /path/vistadisplog -b /path/BIDS --execute

**Things to check**

* The ``Mats`` and ``Bolds`` columns in the output table should be equal.
  A mismatch means a run was dropped from BIDS but its ``.mat`` was not renamed
  (or vice versa) — fix with ``00_drop_ret_run.py`` before continuing.
* Verify task labels match the expected stimuli for each session.

----

Step 2 — Fix wordcentre bold naming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Script: ``02_correct_wc_bold_naming.py``

For sessions that contain the wordcentre (WC) condition, BIDS files may need
renaming to reflect the correct task label.

**How it reads the params.mat files** — unlike step 1, this script does
*not* open the ``.mat`` binary.  Instead it reads:

* The **task/run label** from the ``_params.mat`` symlink filename via regex
  (``task-(\w+)_run-(\d+)``).
* The **AcquisitionTime** from the symlink target filename (e.g.
  ``20230531T172442.mat`` → ``17:24:42``).

Both params.mat and BIDS bold lists are sorted by acquisition time and
matched positionally (same approach as step 1).  If a BIDS file's task/run
label differs from its paired params.mat label, the BIDS file is renamed
atomically.

.. code-block:: console

   python 02_correct_wc_bold_naming.py -s 01,09 -b /path/BIDS --dry-run
   python 02_correct_wc_bold_naming.py -s 01,09 -b /path/BIDS --execute

----

Step 3 — Clean orphan events.tsv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Script: ``04_clean_orphan_events_tsv.py``

BIDS validation tools (and some containers) scan for ``*_events.tsv`` files
and infer the existence of a corresponding task from the filename.  If a task
was listed in the protocol but never acquired (or was later dropped), an
orphaned events file will cause containers to search for bold data that does
not exist.

This script identifies events.tsv files with no matching ``bold.nii.gz`` and
moves them to a ``func_backup_orphan_events/`` directory.

.. code-block:: console

   # Dry-run — whole dataset
   python 04_clean_orphan_events_tsv.py -b /path/BIDS

   # Apply for one session
   python 04_clean_orphan_events_tsv.py -b /path/BIDS -s 11 --ses 10 --no-dry-run

----

Step 4 — Generate per-subject JSON configs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Script: ``03_prepare_prf_subses_json.py``

Generates the JSON config file that each container reads at runtime.  There is
one JSON format per pipeline step; the step is selected with ``--step``.

**Step-specific required options:**

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Step
     - Required option
     - Meaning
   * - ``prfprepare``
     - ``--fp``
     - Name of the fMRIPrep analysis, e.g. ``25.1.4_t2w_fmapsbref_newest``
   * - ``prfanalyze-vista``
     - ``--prepid``
     - prfprepare analysis ID string, e.g. ``01``
   * - ``prfresult``
     - ``--analyzeid``
     - prfanalyze analysis ID string, e.g. ``03``

The ``--fp`` / ``--prepid`` / ``--analyzeid`` values are written **into the JSON
content only** — they do not appear in the output filename.

**Output filename conventions** (must match what the container launcher expects):

.. code-block:: text

   prfprepare       →  all_sub-{sub}_ses-{ses}.json
   prfanalyze-vista →  {task}_{model}_sub-{sub}_ses-{ses}.json
   prfresult        →  all_sub-{sub}_ses-{ses}.json

.. code-block:: console

   # prfprepare JSONs
   python 03_prepare_prf_subses_json.py \
       -b /path/BIDS -f subseslist.tsv \
       --step prfprepare --fp 25.1.4_t2w_fmapsbref_newest \
       -o /path/code/prfprepare_jsons

   # prfanalyze-vista JSONs (one per task per sub/ses)
   python 03_prepare_prf_subses_json.py \
       -b /path/BIDS -f subseslist.tsv \
       --step prfanalyze-vista --prepid 01 --model css \
       -o /path/code/prfanalyze-vista_jsons

   # prfresult JSONs
   python 03_prepare_prf_subses_json.py \
       -b /path/BIDS -f subseslist.tsv \
       --step prfresult --analyzeid 03 \
       -o /path/code/prfresult_jsons

The script also checks whether the fmriprep analysis directory exists and
whether the FreeSurfer symlink inside it is valid.

----

Step 5 — Submit prfprepare jobs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Script: ``sbatch_prfprepare.sh``

Configure the hardcoded variables at the top of the script (``baseP``,
``version``, ``qos``, ``json_dir``, ``sif_path``), then submit:

**Required options:**

* ``-n <log_note>`` — short label included in the log directory name (e.g. ``firstbatch``, ``rerun_sub03``).
* ``-s <sub>,<ses>`` **or** ``-f <subseslist>`` — target subject/session(s).

.. code-block:: console

   # Single sub/ses
   bash sbatch_prfprepare.sh -n firstbatch -s 01,09

   # Batch from file
   bash sbatch_prfprepare.sh -n firstbatch -f /path/subseslist.tsv

SLURM ``.o``/``.e`` log files and the submission script are written to:

.. code-block:: text

   $baseP/dipc_prfprepare_logs/{date}_{log_note}_{analysis_name}/

After each job finishes, ``prfprepare_dipc.sh`` appends one line to
``job_results.tsv`` in that directory:

.. code-block:: text

   2026-04-16 14:10:05   sub-01   ses-09   1_all_prfprepare   12345   exit=0

----

Step 6 — Submit prfanalyze-vista jobs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Script: ``sbatch_prfanalyze.sh``

**Required options:**

* ``-n <log_note>`` — short label included in the log directory name.
* ``-m <model>`` — model label: ``og`` (one gaussian) or ``css``.  Must match
  the ``--model`` value used in step 4.
* ``-s <sub>,<ses>`` **or** ``-f <subseslist>`` — target subject/session(s).

**Task auto-detection:** the script globs existing JSONs in ``json_dir``
matching ``*_{model}_sub-{sub}_ses-{ses}.json`` to determine which tasks to
run — no manual task list is needed.  One SLURM job is submitted per detected
task.

.. code-block:: console

   bash sbatch_prfanalyze.sh -n firstbatch -m og -s 01,09
   bash sbatch_prfanalyze.sh -n firstbatch -m og -f /path/subseslist.tsv

SLURM ``.o``/``.e`` log files are written to:

.. code-block:: text

   $baseP/dipc_prfanalyze-vista_logs/{date}_{log_note}_{analysis_name}/

After each job finishes, ``prfanalyze-vista_dipc.sh`` appends one line to
``job_results.tsv`` in that directory:

.. code-block:: text

   2026-04-16 14:23:01   sub-03   ses-01   task-retRW   3_retRW_prfanalyze-vista   12346   exit=0

----

Step 7 — prfresult (BCBL only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   ``prfresult`` only works on the **BCBL server**.  Do not submit it on DIPC
   — the container will fail.  Use the equivalent script on the BCBL cluster.

----

.. _prf_special_notes:

Special notes
-------------

fMRIPrep analysis directory layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The prfprepare container expects the fMRIPrep output under:

.. code-block:: text

   BIDS/derivatives/fmriprep/analysis-{fp}/sub-{sub}/ses-{ses}/func/

If the fMRIPrep output was produced with a different naming scheme (e.g.
``fmriprep-{fp}/``), a symlink is needed:

.. code-block:: text

   BIDS/derivatives/fmriprep/analysis-{fp}  →  BIDS/derivatives/fmriprep-{fp}

``03_prepare_prf_subses_json.py`` checks for this symlink automatically when
``--step prfprepare`` is used and prints a warning if it is missing or broken.

FreeSurfer license
~~~~~~~~~~~~~~~~~~

prfprepare requires a FreeSurfer license file.  The path is currently
hardcoded in the SLURM job scripts (``sbatch_prfprepare.sh``).  Ensure
``license_path`` at the top of the script points to a valid ``.license`` file
before submitting.

prfanalyze-vista output folder structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The folder structure written by ``prfanalyze-vista`` is **hardcoded in the
original Benson MATLAB code** and cannot be changed at runtime.  The LC
workflow must therefore respect that structure when referencing prfanalyze
outputs in subsequent steps.  Do not rename or reorganise the prfanalyze
output directories.

events.tsv and phantom tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BIDS ``*_events.tsv`` files are sometimes created during protocol design for
tasks that were never actually acquired (e.g. ``task-fixblock``).  Because
containers scan ``events.tsv`` filenames to infer available tasks, an orphaned
events file causes the container to look for bold data that does not exist,
producing a ``FileNotFoundError``.

Fix: run ``04_clean_orphan_events_tsv.py`` to move orphaned events files to a
backup directory before submitting any containers.

sub-01 ses-09/10 — retfixRWblock renaming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sub-01 sessions 09 and 10 contain both ``retfixRWblock01`` and
``retfixRWblock02`` within the same session.  The two sessions have
**different rename rules**:

.. list-table::
   :header-rows: 1
   :widths: 15 35 35 15

   * - Session
     - Old name
     - New name
     - Notes
   * - ses-09
     - ``task-retfixRWblock02_run-01``
     - ``task-retfixRWblock01_run-01``
     - task renamed; run number unchanged
   * - ses-10
     - ``task-retfixRWblock02_run-01``
     - ``task-retfixRWblock01_run-02``
     - task renamed; run-01 → run-02

``01b_special_fix_sub01_block02_rename.py`` applies the rename across five
locations per session: BIDS/func, BIDS/fmap (IntendedFor), fmriprep/func,
fmriprep/fmap (IntendedFor), and ``sourcedata/`` (``_params.mat`` symlinks).
All renames are **atomic** (via UUID tmp dir) — no backup directories are
created.  Run in dry-run mode first to verify:

.. code-block:: console

   python 01b_special_fix_sub01_block02_rename.py          # dry-run
   python 01b_special_fix_sub01_block02_rename.py --no-dry-run

JSON path matching
~~~~~~~~~~~~~~~~~~~

The container launcher binds the JSON directory into the container at a fixed
path.  If the JSON filename or location does not match what the launcher
expects, the job will fail silently or with a confusing bind error.  Always
verify that the output directory passed to ``03_prepare_prf_subses_json.py``
matches the ``json_dir`` variable in the SLURM submission scripts.

----

End-to-end example (single session)
-------------------------------------

.. code-block:: console

   cd tests/prepare_prf

   # 1. Convert DICOMs
   python 00_dcm2niix_ret.py -s 01 --ses 09 -d /dicom -o /BIDS

   # 2. Inspect vistadisplog pairing table
   python 01_prepare_prf_log_and_bold.py -s 01,09 --logdir /vistadisplog -b /BIDS -v

   # 3. Create params.mat symlinks
   python 01_prepare_prf_log_and_bold.py -s 01,09 --logdir /vistadisplog -b /BIDS --execute

   # 4. Clean orphan events.tsv (dry-run then apply)
   python 04_clean_orphan_events_tsv.py -b /BIDS -s 01 --ses 09
   python 04_clean_orphan_events_tsv.py -b /BIDS -s 01 --ses 09 --no-dry-run

   # 5. Generate prfprepare JSON
   python 03_prepare_prf_subses_json.py \
       -b /BIDS -s 01 --ses 09 \
       --step prfprepare --fp 25.1.4_t2w_fmapsbref_newest \
       -o /VOTCLOC/code/prfprepare_jsons

   # 6. Submit prfprepare
   cd ../run_prf
   bash sbatch_prfprepare.sh -n ses09_test -s 01,09

   # 7. Generate prfanalyze-vista JSONs
   cd ../prepare_prf
   python 03_prepare_prf_subses_json.py \
       -b /BIDS -s 01 --ses 09 \
       --step prfanalyze-vista --prepid 01 --model og \
       -o /VOTCLOC/code/prfanalyze-vista_jsons

   # 8. Submit prfanalyze (tasks auto-detected from existing JSONs)
   cd ../run_prf
   bash sbatch_prfanalyze.sh -n ses09_test -m og -s 01,09

   # 9. prfresult — run on BCBL, not DIPC
