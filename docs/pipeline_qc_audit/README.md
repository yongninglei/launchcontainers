# Pipeline QC audit (branch `pipeline-qc-audit`)

Step-by-step review of how the DICOM → BIDS → fMRIPrep preparation was run, what
QC each step needs, and which parts of launchcontainers should become independent
modules. Read-only reviews: nothing in the data was changed.

| File | What |
|---|---|
| `2026-11-17_votcloc_missing3_fmri_review.md` | review of the VOTCLOC sub-05 ses-08, sub-08 ses-04, sub-10 ses-05 fMRI conversion (siwang, 2026-11-09…16): per-step verdict, findings with severity and the lc file to fix |
| `module_split_plan.md` | notes on taking the steps out of lc as standalone modules (`--plan` contract) |

QC scripts added on this branch (compute → CSV; plotting reads the CSV only):

| Script | Step | Output |
|---|---|---|
| `MR_pipelines/00_dicom_to_nifti/quality_control/qc_heudiconv_series_map.py` | heudiconv | `sub-XX_ses-YY_series_map.csv`, `heudiconv_qc_summary.csv` |
| `MR_pipelines/03_fmriprep/quanlity_control/qc_fmap_boldref_xfm.py` | fMRIPrep SDC | `fmap_boldref_xfm_runs.csv`, `fmap_boldref_xfm_sessions.csv`, `fmap_boldref_xfm.png` |

Still to review: DWI (NORDIC dwi / magonly, RTP inputs), anat (presurfer, T2w).
