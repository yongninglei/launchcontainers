.. _prepare_fmriprep:

Prepare — fMRIPrep (fmap + sbref)
==================================

Why this matters
----------------

**fMRIPrep** performs susceptibility distortion correction (SDC) on functional
EPI data using field maps (fmaps) and uses single-band reference images (sbref)
as the EPI reference for motion correction and co-registration.  For this to
work correctly, the BIDS dataset must satisfy two conditions **before** fMRIPrep
is run:

1. Every fmap JSON sidecar must contain a valid ``IntendedFor`` field listing
   exactly the func runs it should correct.
2. Every func run must be paired with its matching sbref, and both must have
   consistent run indices.

Even a single mismatched or missing ``IntendedFor`` entry will cause fMRIPrep to
either skip SDC for that run entirely, or apply the wrong field map — silently
producing incorrect distortion correction.  Similarly, an orphaned or
mis-numbered sbref will cause fMRIPrep to use the wrong reference image for
motion correction.

This pipeline provides a fully automated, auditable way to clean and annotate
your raw BIDS data so it is ready to pass into fMRIPrep.

----

Overview of the pipeline
-------------------------

The pipeline consists of four scripts that must be run **in order**:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Script
     - What it does
   * - ``00_merge_split_ses_reording_runs.py``
     - Merges split session directories and fixes run-number gaps / non-1
       starting indices across **all modalities** (func, fmap, dwi, anat).
   * - ``00b_copy_fmap_to_bids.py``
     - Copies fmap folders and scans.tsv from the raw_nifti source tree into the
       target BIDS directory, then ``chmod 755`` the destination fmap folder.
   * - ``01_drop_duplicated_sbrefs.py``
     - Drops orphaned sbref files (no matching func within 30 s) and renumbers
       kept sbrefs sequentially per task.
   * - ``02_prepare_fmap_intendedfor.py``
     - Prunes invalid fmaps, renumbers survivors, and writes (or repairs) the
       ``IntendedFor`` field in every fmap JSON sidecar.

All scripts share the same CLI conventions:

* ``-s 10,02`` — single sub/ses pair (for testing)
* ``-f subseslist.tsv`` — batch mode over a TSV/CSV file with ``sub``, ``ses`` columns
* Default mode is always **dry-run** (nothing is changed).  Pass ``--execute`` to apply.

----

Step 0 — Merge split sessions and reorder runs
-----------------------------------------------

Script: ``00_merge_split_ses_reording_runs.py``

**Problem it solves**

Scanners sometimes split a long session across two directories
(e.g. ``ses-02`` and ``ses-02part2``).  Files in the secondary directory
restart their run indices from ``run-01``, which creates BIDS conflicts.
Additionally, if a run was aborted and re-acquired, gaps in run numbering
(e.g. ``run-08`` → ``run-10``) can appear even in a single-directory session.

**How it works**

For every modality (func, fmap, dwi, anat) the script:

1. Collects all files across **all** matching ``ses-{label}*`` directories.
2. Groups files by logical run unit (see below) and sorts each group by
   ``AcquisitionTime``.
3. Assigns new run numbers 1, 2, 3 … sequentially — this both merges split
   sessions and fills any gaps.
4. Moves secondary-directory files into the primary ``ses-{label}`` directory
   with updated names (two-phase rename: ``→ temp → final`` to avoid conflicts).
5. Merges ``scans.tsv`` files and removes empty secondary directories.

**Run-number formatting**

.. code-block:: text

   fmap   → 1-digit   (run-1, run-2, …)
   others → 2-digit   (run-01, run-02, …)

**Grouping rules** (what counts as "one run sequence")

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Modality
     - Group key
   * - ``func``
     - ``(mod, task, suffix)`` — sbref, magnitude, and phase are counted
       **independently** per task (``task-fLoc_run-01_sbref`` and
       ``task-fLoc_run-01_magnitude`` are separate sequences).
   * - ``fmap``
     - ``(mod, None, None)`` — AP and PA of the same acquisition share a run
       number; pairs are identified by ``(ses_label, run_int)`` so that
       ``ses-02 run-1`` and ``ses-02part2 run-1`` are treated as distinct runs.
   * - ``dwi``
     - ``(mod, acq, None)`` — AP and PA share a run number; different ``acq-``
       labels (e.g. ``acq-b0`` vs ``acq-b1000``) are independent sequences.
   * - ``anat``
     - ``(mod, None, suffix)``

**Things to check**

* Run in dry-run mode first and inspect the timeline table — verify that
  secondary-session files get the expected new run numbers.
* Watch for ``ses only`` entries (session label changes but run number is
  already correct) vs ``run-01 → run-02`` entries (actual renumber).

----

Step 0b — Copy fmap to BIDS
----------------------------

Script: ``00b_copy_fmap_to_bids.py``

Copies the cleaned fmap folder (and scans.tsv) from the raw_nifti source
tree into the target BIDS directory.  After copying, ``chmod 755`` is applied
to the destination fmap directory and all files inside it to ensure fMRIPrep
can read them.

Options:

* ``--force`` — overwrite existing destination files.
* ``--sync-bids-components`` — also copy top-level BIDS files
  (``dataset_description.json``, ``participants.tsv``, ``README``) once.

----

Step 1 — Drop orphaned sbrefs
-------------------------------

Script: ``01_drop_duplicated_sbrefs.py``

**The sbref ↔ func pairing rule**

For susceptibility distortion correction, fMRIPrep uses the sbref as the
EPI reference image.  Each sbref must be acquired **immediately before** its
corresponding func run (the scanner acquires it as the first volume of the
EPI sequence, then aborts and starts the full run).  A sbref that is not
followed by a func run within ``--max-gap`` seconds (default: 30 s) is
**orphaned** — it has no matching run and must be dropped.

**Algorithm**

1. For each sbref (in chronological order), find the nearest func run that
   comes **after** it in time.
2. If that func is within ``max_gap`` seconds → **keep**.  Otherwise → **drop**.
3. After all drops are decided, renumber kept sbrefs sequentially
   (``run-01``, ``run-02``, …) **per task** so indices remain gapless.

**Important**: drops happen before renames to avoid a renamed file being
immediately deleted by the path it replaced.

**Things to check**

* The ``Δs`` column shows the time gap between each sbref and its matched func —
  values around 14 s are normal (the sbref takes ~14 s to acquire).
* A gap > 30 s indicates a broken pairing.  Inspect these manually before
  executing.
* The sbref run index is task-specific: ``task-fLoc_run-01_sbref`` and
  ``task-retRW_run-01_sbref`` are independent sequences.

----

Step 2 — Prepare fmap IntendedFor
-----------------------------------

Script: ``02_prepare_fmap_intendedfor.py``

**Why IntendedFor?**

BIDS requires that each fmap JSON contains an ``IntendedFor`` list of NIfTI
paths (relative to the subject directory) for all func runs it should correct.
fMRIPrep reads this field to decide which fmap to apply to which run.

**How IntendedFor windows are assigned**

Fmaps are sorted by ``AcquisitionTime``.  Each fmap "owns" all func files
whose ``AcquisitionTime`` falls **after that fmap and before the next fmap**.
This reflects the physical reality that a fmap is acquired once per scanning
block and corrects all the runs that follow it.

**Fmap pruning (two passes)**

.. code-block:: text

   Pass 1 — consecutive fmaps with no func between them:
       The earlier fmap has an empty window → drop it, keep the later one
       (which is closer to the actual func block).

   Pass 2 — tail orphans:
       Any fmap with an empty IntendedFor after pass 1 (no func follows it
       at all) → drop.

After pruning, surviving fmaps are renumbered 1-digit sequentially (``run-1``,
``run-2``, …).

**Checks and warnings**

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Check
     - Severity
     - Meaning
   * - SESSION GAP
     - Warning
     - A fmap → first-func-after-it gap > 45 min.  This often indicates
       a genuine session break but can also be a metadata error.
   * - FMAP→FUNC GAP
     - Warning
     - The gap between a kept fmap and the nearest func after it is > 3 min.
       Suspicious — may indicate the wrong fmap is being used for that block.
   * - FMAP→FIRST-FUNC GAP
     - Warning
     - The fmap immediately before the first func run has a gap > 3 min.
   * - NO FMAP BEFORE FIRST FUNC
     - **Error**
     - The earliest func run has no fmap preceding it in time.  IntendedFor
       cannot be correctly assigned.  Session is **blocked** in ``--execute``.
   * - NOT COVERED
     - **Error**
     - One or more func runs fall outside all fmap time windows and will have no
       ``IntendedFor`` pointing to them.  Session is **blocked** in ``--execute``.

**IntendedFor status per fmap JSON**

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Status
     - Meaning
   * - ``GOOD``
     - The JSON already has the correct ``IntendedFor`` — nothing written.
   * - ``WRONG``
     - The JSON has an ``IntendedFor`` field but it points to wrong/stale runs —
       overwritten on ``--execute``.
   * - ``MISSING``
     - The JSON has no ``IntendedFor`` field — written on ``--execute``.
   * - ``ERROR``
     - One of the blocking checks above fired.  ``--execute`` is refused for
       this session until the data issue is resolved manually.

**Things to check**

* Run in dry-run mode (default) and read every ``ERROR`` session carefully
  before proceeding.
* A ``NOT COVERED`` error for the first few func runs of a session usually means
  the fmap was acquired too late — the scan protocol needs to be fixed for
  future sessions.
* ``WRONG`` status is common after re-running ``00_`` or ``01_`` (run indices
  changed, so old ``IntendedFor`` paths are stale).  ``--execute`` will fix them
  automatically.
* ``IntendedFor`` paths are written relative to the **subject directory**::

      ses-02/func/sub-10_ses-02_task-fLoc_run-01_bold.nii.gz

----

End-to-end workflow example
----------------------------

.. code-block:: console

   # 1. Merge split sessions and fix run-number gaps (inspect output first)
   python 00_merge_split_ses_reording_runs.py -b /raw_nifti -s 10,02
   python 00_merge_split_ses_reording_runs.py -b /raw_nifti -s 10,02 --execute

   # 2. Copy cleaned fmaps to the target BIDS directory
   python 00b_copy_fmap_to_bids.py -b /raw_nifti -t /BIDS -s 10,02
   python 00b_copy_fmap_to_bids.py -b /raw_nifti -t /BIDS -s 10,02 --execute

   # 3. Drop orphaned sbrefs
   python 01_drop_duplicated_sbrefs.py -b /BIDS -s 10,02
   python 01_drop_duplicated_sbrefs.py -b /BIDS -s 10,02 --execute

   # 4. Write IntendedFor
   python 02_prepare_fmap_intendedfor.py -b /BIDS -s 10,02
   python 02_prepare_fmap_intendedfor.py -b /BIDS -s 10,02 --execute

   # Batch run over a full subject list
   python 00_merge_split_ses_reording_runs.py -b /raw_nifti -f subseslist.tsv --execute
   python 00b_copy_fmap_to_bids.py  -b /raw_nifti -t /BIDS -f subseslist.tsv --execute
   python 01_drop_duplicated_sbrefs.py -b /BIDS -f subseslist.tsv --execute
   python 02_prepare_fmap_intendedfor.py -b /BIDS -f subseslist.tsv --execute

----

Common failure scenarios
-------------------------

**fLoc runs acquired before the first fmap**

The fLoc task is sometimes run at the start of a session before the fieldmap
is acquired.  These runs will trigger ``NOT COVERED`` errors.  There is no
automatic fix — either the scan protocol must be changed (acquire a fmap before
fLoc) or fMRIPrep must be configured to run without SDC for those runs.

**Split session with colliding fmap run-1**

Both ``ses-02`` and ``ses-02part2`` contain a ``run-1`` fmap.  ``00_`` will
correctly identify these as distinct logical runs (keyed by ``ses_label +
run_int``) and renumber them to ``run-1`` and ``run-2``.

**Orphaned sbref after abort**

If a run was aborted immediately after the sbref was acquired (before any func
volumes), the sbref will have no matching func within 30 s and be dropped by
``01_``.  The run-number gap left by the drop is closed by ``01_``'s sequential
renumbering.

**WRONG IntendedFor after re-running 00_ or 01_**

Any time run indices change, previously written ``IntendedFor`` paths become
stale.  Re-running ``02_`` with ``--execute`` will detect the ``WRONG`` status
and overwrite all affected fmap JSONs.
