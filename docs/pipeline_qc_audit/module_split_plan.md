# Notes: taking the DICOM → BIDS → fMRIPrep steps out of launchcontainers

Status: **notes only, nothing moved yet** (2026-11-17). Input: the review
`2026-11-17_votcloc_missing3_fmri_review.md`.

## Why

Today these steps live in lc as loose scripts (`MR_pipelines/00_dicom_to_nifti`,
`MR_pipelines/01_prepare_nifti` / `launchcontainers/tests/dcm_to_bids_workflow`,
`MR_pipelines/03_fmriprep`), each with its own bash launcher, subses parsing and
log layout. The November run showed what that costs:

- people run per-user, untracked copies (siwang's `notes` branch), so nobody can say
  which code produced BIDS;
- the scripts moved (`tests/prepare_fmriprep` → `tests/dcm_to_bids_workflow`) and the
  old copies kept being used;
- launchers do not record themselves, the list, the commit or the tool version;
- success was judged by "empty `.e`", which does not detect failures (NORDIC prints
  "finished" with 0 runs);
- no step had a QC whose output a person or lc could read back.

lc's design already has the answer (`Claude.md`, `docs/plan_protocol.md`): an
analysis is a **standalone program** with a single-session CLI
(`--config --sub --ses`) and `--plan` (JSON: analysis dirs, session outputs, inputs,
problems). lc prepares, launches with `lc-session`, records provenance and run logs,
and does launch QC. The loop over sessions lives only in lc.

## Proposed modules (one repo / entry point each, registered as `ExternalTool`)

| Module | Steps it takes over | `--plan` problems (blocks the session) | Session outputs + QC it writes |
|---|---|---|---|
| `dcm2nii` (heudiconv wrapper) | keep-date DICOM selection (`link_keepdate_dcm.sh` logic, header-date fallback), heudiconv step2, the series-map QC (`qc_heudiconv_series_map.py`) | DICOM dir missing / more than one scan date / 0 files; heuristic missing | `raw_nifti/sub/ses/**`, `.heudiconv/` records, `qc/sub-XX_ses-YY_series_map.csv`; `ok=False` on orphan sbref, missing output, label mismatch, lab-note mismatch |
| `nifti-tidy` (Phase 0) | `00_merge_split_ses_reording_runs`, `01_rename_ME_magnitude`, `03_drop_duplicated_sbrefs` — on raw_nifti, **before** NORDIC | orphan sbref that cannot be matched; run gaps | renames logged to TSV (old → new); never deletes, moves dropped files aside |
| `nordic-func` | `local_nordic.sh` / `src_nordic_fmri.sh` / `nordic_fmri.m` / `prepare_nordic_bold_nifti.m` | raw run without phase or sbref; `_orig` not uint16; volume count ≠ expected; bold already present and `force` false | BIDS `_bold`, `_gfactor`, sbref copy; QC CSV per run (nvol = raw − 1, bold.json = magnitude.json, sbref byte-identical, 14 s / SN − 2) |
| `fmap-prep` | `01_copy_fmap_to_bids`, `02_prepare_fmap_intendedfor`, scans.tsv regeneration | AP/PA not paired; any func covered 0 or > 1 times; fmap affine ≠ func affine | BIDS fmap + IntendedFor; QC CSV (multiplicity, assigned pair, Δaffine) |
| `anat-prep`, `dwi-prep` | presurfer + T2w; DWI NORDIC / magonly | (to review) | (to review) |
| fMRIPrep (container tool in lc) + `fmriprep-qc` | run fMRIPrep with BIDS filter + precomputed anatomy; `qc_fmap_boldref_xfm.py` | BIDS session incomplete (from the modules above) | per-run fieldmap → boldref displacement; flags failed SDC (sub-10) |

Rules for every module (from lc's conventions):

- typer + rich, exit 2 on config / input errors; dry run unless told to execute.
- never delete: moved-aside files go to a dated folder next to the session.
- each writes a per-session QC CSV that lc's launch QC can check exists
  (`session_outputs`) and that `lc qc` can summarise; image QC stays in
  QC-registration.
- heavy dependencies (MATLAB toolboxes, heudiconv container) are named in the config
  with versions; lc records them in the run log.

## Order of work

1. Fix the two code problems that can corrupt data now: the IntendedFor window
   overlap (`02_prepare_fmap_intendedfor.py`) and NORDIC `force` /
   backup handling. Branch per fix, CHANGELOG entry each.
2. Run the QCs dataset-wide (commands in the review), decide sub-10 SDC.
3. Extract `fmap-prep` first (pure Python, smallest, has the worst bug), then
   `nifti-tidy`, `dcm2nii`, `nordic-func`.
4. Delete the old copies from `MR_pipelines/` / `tests/` only once the module is
   released and the maintainer agrees.

## Open questions for the maintainer

- One repo per module, or one `bids-prep` repo with several entry points?
- `--minmeta`: on or off for the whole dataset?
- scans.tsv: regenerate for BIDS names, or drop it from BIDS?
- sub-10 SDC: rerun without fieldmaps, with `--use-syn-sdc`, or debug sdcflows?
