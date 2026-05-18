# DCM → BIDS Workflow — Standard Operating Procedure

## Overview

The workflow has two phases:

1. **raw_nifti cleanup** — fix filenames and sbref issues *before* any data are
   copied to BIDS.
2. **BIDS preparation** — run presurfer (anat), NORDIC (func), copy fmaps,
   then populate `IntendedFor`.

---

## Phase 0 — raw_nifti cleanup (run once, before any BIDS step)

These scripts operate on the **raw_nifti** tree (already in BIDS format from
heudiconv).  They must finish before Phase 1 begins.  **Run in order.**

| Step | Script | Purpose |
|------|--------|---------|
| 00 | `00_merge_split_ses_reording_runs.py -b /raw_nifti -s sub,ses` | Merge split sessions (ses-02 + ses-02part2 → ses-02) and fix run-number gaps |
| 01 | `01_rename_ME_magnitude.py -b /raw_nifti -s sub,ses` | Rename ME magnitude files from heudiconv `_magnitudeN` → BIDS `_echo-N_magnitude` |
| 02 | `02_copy_bids_component.py -b /raw_nifti -t /BIDS` | Copy top-level BIDS metadata (dataset_description.json, participants.\*, README) |
| 03 | `03_drop_duplicated_sbrefs.py -b /raw_nifti -s sub,ses` | Drop orphaned sbrefs and renumber kept ones (ME-aware) |

> **Why this order matters:**
> - Step 01 (ME rename) must precede step 03 (sbref drop) because the drop
>   script detects multi-echo files by their `_echo-N_` entity.  If the rename
>   has not run yet, the echo entity is absent and ME deduplication is skipped.
> - Step 03 must finish before Phase 1 because `nordic_fmri.m` copies sbrefs
>   1-to-1 from raw_nifti based on magnitude run numbers.  Cleaning raw_nifti
>   first ensures that whatever NORDIC copies is already correct.
> - Step 02 (top-level metadata copy) is independent of steps 01/03 — it only
>   touches dataset_description.json, participants.\*, and README.

---

## Multi-echo (ME) implementation

### What heudiconv produces

heudiconv outputs echo-specific magnitude files with a suffix-embedded echo
number rather than a BIDS `echo-` entity:

```
sub-01_ses-01_task-BfLocVideo_acq-ME_run-01_magnitude1.nii.gz
sub-01_ses-01_task-BfLocVideo_acq-ME_run-01_magnitude2.nii.gz
sub-01_ses-01_task-BfLocVideo_acq-ME_run-01_magnitude3.nii.gz
```

### Step 01 — `01_rename_ME_magnitude.py`

Inserts the BIDS `echo-N` entity and normalises the suffix:

```
sub-01_ses-01_task-BfLocVideo_acq-ME_run-01_echo-1_magnitude.nii.gz
sub-01_ses-01_task-BfLocVideo_acq-ME_run-01_echo-2_magnitude.nii.gz
sub-01_ses-01_task-BfLocVideo_acq-ME_run-01_echo-3_magnitude.nii.gz
```

Consistent with phase and sbref files that already carry the echo entity:

```
sub-01_ses-01_task-BfLocVideo_acq-ME_run-01_echo-1_phase.nii.gz
sub-01_ses-01_task-BfLocVideo_acq-ME_run-01_echo-1_sbref.nii.gz
```

Detection: regex `_magnitude(\d+)$` on the stem.  Files with an existing
`_echo-\w+_` entity are skipped (idempotent).  Two-phase atomic rename (src →
temp → final) is used throughout.

### Step 03 — `03_drop_duplicated_sbrefs.py` (ME-aware)

Without ME handling, N echo files per run would appear as N separate func
entries, and a drop/rename decision on an sbref would only move the echo-1
file, leaving echoes 2…N stranded under their old name.

**`_dedup_me_by_run(entries)`** — called on both funcs and sbrefs after
collection:

- Detects ME entries by `_echo-(\d+)_` in the basename.
- Strips the echo entity (`_echo_canonical`) to get a run-level key.
- Keeps one **representative** entry per run (first echo encountered).
- Stores all echo entries (including the representative itself) in
  `representative["me_echoes"]`.
- Non-ME entries pass through with an empty `me_echoes` list.

**Sbref renumbering key — `(task, acq)`**

After matching, kept sbrefs are renumbered sequentially (run-01, run-02, …).
The counter is keyed by `(task, acq)`, not by `task` alone.  This is critical
because different acquisition sequences (e.g. `acq-SE` and `acq-ME`) have
independent run numbering — a session with three SE runs and three ME runs
should produce `acq-SE run-01…03` and `acq-ME run-01…03`, not a shared
sequence `run-01…06` that would propose spurious renames.

Effect on matching: the time-gap matcher sees one entry per run, so an ME
sbref correctly matches its single func run rather than three.

Effect on execute:

- **Drop**: iterates `me_echoes` and removes every echo's `.nii.gz` + `.json`.
- **Rename**: expands the sbref group into per-echo rename tasks, applying the
  same `_run-XX_` → `_run-YY_` substitution to each echo basename.  The
  all-phase-1-before-all-phase-2 strategy is preserved to avoid collisions
  (e.g. run-03→run-02 must not clobber the original run-02 before it moves).

Display: the acq-time table appends `(ME×N)` to the name column so you can
see at a glance which entries are multi-echo.

---

## Phase 1 — BIDS preparation

Steps 1a and 1b are independent and can run in parallel.

| Step | Script | Location | Purpose |
|------|--------|----------|---------|
| 1a | `local_run_presurfer.sh -b <basedir> -i <rel_src> -s sub,ses` | `prepare_anat/` | T1w presurfer → BIDS `anat/` |
| 1b | `local_nordic.sh -b <basedir> -i <rel_src> -s sub,ses` | `prepare_func/` | NORDIC denoising → BIDS `func/` (copies clean sbrefs from raw_nifti) |
| 2  | `01_copy_fmap_to_bids.py -b /raw_nifti -t /BIDS -s sub,ses --execute` | `prepare_fmap/` | Copy fmap/ and scans.tsv from raw_nifti → BIDS |
| 3  | `02_prepare_fmap_intendedfor.py -b /BIDS -s sub,ses --execute -v` | `prepare_fmap/` | Populate `IntendedFor` in BIDS fmap JSONs based on AcquisitionTime |

> Steps 2 and 3 must run **after** both 1a and 1b are complete.

---

## Full command sequence (single session example)

```bash
# ── Phase 0: raw_nifti cleanup ──────────────────────────────────────────────
python 00_merge_split_ses_reording_runs.py  -b /raw_nifti -s pilot01,01
python 01_rename_ME_magnitude.py            -b /raw_nifti -s pilot01,01 --execute
python 02_copy_bids_component.py            -b /raw_nifti -t /BIDS      --execute
python 03_drop_duplicated_sbrefs.py         -b /raw_nifti -s pilot01,01 --execute

# ── Phase 1a + 1b: anat and func (can run in parallel) ──────────────────────
bash prepare_anat/local_run_presurfer.sh -b /basedir -i raw_nifti -s pilot01,01
bash prepare_func/local_nordic.sh        -b /basedir -i raw_nifti -s pilot01,01

# ── Phase 1 fmap: copy then IntendedFor (after 1a + 1b finish) ──────────────
python prepare_fmap/01_copy_fmap_to_bids.py       -b /raw_nifti -t /BIDS -s pilot01,01 --execute
python prepare_fmap/02_prepare_fmap_intendedfor.py -b /BIDS     -s pilot01,01 --execute -v
```

---

## Notes

- All `_*.py` scripts default to **dry-run**.  Always check the dry-run output
  before adding `--execute`.
- `02_copy_bids_component.py` will **not** overwrite existing files unless
  `--force` is passed.  Safe to re-run.
- `03_drop_duplicated_sbrefs.py` uses a 30-second time-gap threshold by
  default to decide if an sbref is orphaned.  Pass `--max-gap N` to override.
- `02_prepare_fmap_intendedfor.py` includes func files up to 10 minutes
  **before** the fmap in `IntendedFor` (lookback window).  Pass
  `--lookback N` (minutes) to override.
- Do **not** modify the subseslist file while `local_run_presurfer.sh` or
  `local_nordic.sh` are still running — the running job reads it at launch
  time but any batch-mode restart will pick up the modified list.
