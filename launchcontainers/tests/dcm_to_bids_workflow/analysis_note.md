# DCM → BIDS Workflow — Standard Operating Procedure

## Overview

The workflow has two phases:

1. **raw_nifti cleanup** — fix run numbering and sbref issues *before* any
   data are copied to BIDS.
2. **BIDS preparation** — run presurfer (anat), NORDIC (func), copy fmaps,
   then populate `IntendedFor`.

---

## Phase 0 — raw_nifti cleanup (run once, before any BIDS step)

These scripts operate on the **raw_nifti** tree (which is already in BIDS
format from heudiconv).  They must finish before Phase 1 begins.

| Step | Script | Purpose |
|------|--------|---------|
| 00 | `00_merge_split_ses_reording_runs.py -b /raw_nifti -s sub,ses` | Merge split sessions (ses-02 + ses-02part2 → ses-02) and fix run-number gaps |
| 01 | `01_drop_duplicated_sbrefs.py -b /raw_nifti -s sub,ses` | Drop orphaned sbrefs and renumber kept ones **in raw_nifti** before any copy |
| 02 | `02_copy_bids_component.py -b /raw_nifti -t /BIDS` | Copy top-level BIDS files (dataset_description.json, participants.*, README) to BIDS root |

> **Why 00+01 must run on raw_nifti:**
> `nordic_fmri.m` copies sbrefs 1-to-1 from raw_nifti based on magnitude
> run numbers (`run-N_magnitude → run-N_sbref`).  It has no awareness of
> orphaned or mismatched sbrefs.  Cleaning raw_nifti first ensures that
> whatever NORDIC copies is already correct.

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
python 00_merge_split_ses_reording_runs.py -b /raw_nifti -s pilot01,01
python 01_drop_duplicated_sbrefs.py        -b /raw_nifti -s pilot01,01 --execute
python 02_copy_bids_component.py           -b /raw_nifti -t /BIDS      --execute

# ── Phase 1a + 1b: anat and func (can run in parallel) ──────────────────────
bash prepare_anat/local_run_presurfer.sh -b /basedir -i raw_nifti -s pilot01,01
bash prepare_func/local_nordic.sh        -b /basedir -i raw_nifti -s pilot01,01

# ── Phase 1 fmap: copy then IntendedFor (after 1a + 1b finish) ──────────────
python prepare_fmap/01_copy_fmap_to_bids.py      -b /raw_nifti -t /BIDS -s pilot01,01 --execute
python prepare_fmap/02_prepare_fmap_intendedfor.py -b /BIDS     -s pilot01,01 --execute -v
```

---

## Notes

- All `_*.py` scripts default to **dry-run**.  Always check the dry-run output
  before adding `--execute`.
- `01_drop_duplicated_sbrefs.py` uses a 30-second time-gap threshold by
  default to decide if an sbref is orphaned.  Pass `--max-gap N` to override.
- `02_prepare_fmap_intendedfor.py` includes func files up to 10 minutes
  **before** the fmap in `IntendedFor` (lookback window).  Pass
  `--lookback N` (minutes) to override.
- Do **not** modify the subseslist file while `local_run_presurfer.sh` or
  `local_nordic.sh` are still running — the running job reads it at launch
  time but any batch-mode restart will pick up the modified list.
