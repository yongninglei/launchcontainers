# Review: VOTCLOC missing-3 fMRI conversion (DICOM → BIDS → fMRIPrep)

**Date:** 2026-11-17. **Reviewer:** Claude (for tlei), branch `pipeline-qc-audit`.
**Scope:** the functional data of the three sessions converted 2026-11-09 … 11-16 by
siwang: **sub-05 ses-08, sub-08 ses-04, sub-10 ses-05**. DWI is reviewed separately.
**Her account:** `/bcbl/home/public/Gari/VOTCLOC/main_exp/code/claude_fMRI_conversion_workflow/workflow_fMRI.md`.
**Method:** read-only. Outputs, logs and headers were read on disk (no image data
loaded except the fMRIPrep report PNGs). siwang's checkout
(`/bcbl/home/home_n-z/siwang/projects/launchcontainers/launchcontainers`, branch
`notes`) is not readable by tlei, so the code was reviewed from this repo's
history and identified from strings in her logs.

`M=/bcbl/home/public/Gari/VOTCLOC/main_exp`, `B=$M/BIDS`,
`F=$B/derivatives/fmriprep-25.1.4_t2w_fmapsbref_newest`.

Severity used below: **error** = a wrong output; **risk** = the output is right
here but the code or process can produce a wrong output elsewhere;
**provenance** = cannot be traced or reproduced; **cosmetic**.

---

## Verdict

| Step | Output for the 3 sessions | Process / code |
|---|---|---|
| 1. DICOM selection (keep-date link trees) | correct | provenance gaps |
| 2. heudiconv → `raw_nifti_missing3` | correct | differs from repo; provenance gaps |
| 3. rerun / run-gap fix | not needed (no reruns, no splits) | not run; SOP step skipped |
| 4. func NORDIC → BIDS | correct (48/48 runs) | several risks; provenance gaps |
| 5. fmap copy, sbref↔bold, IntendedFor | correct (IntendedFor exact, incl. sub-10's 5 pairs) | ran **outdated scripts**; the **current** SOP IntendedFor script has a window-overlap bug that would break sub-10 ses-05 |
| 6. events.tsv | correct (right dates/stimset) | manual `ln -s`, no log |
| 7. fMRIPrep 25.1.4 | sub-05 ses-08, sub-08 ses-04 OK. **sub-10 ses-05: fieldmap→boldref registration failed in 16/16 runs** (also in tlei's April sub-10 ses-04, 15/16) | settings match April |

The headline problem is not siwang's: **SDC for sub-10 is wrong in fMRIPrep**,
in her new session and in the April run of ses-04. It needs a dataset-wide check
(command in §7) and a decision on how to rerun.

---

## 1. DICOM selection

She built per-file symlink trees `dicom/sub-XX/ses-YY/<series>/<file>.dcm →
dicom_manual/sub-XX_ses-YY/...` keeping only files whose name contains
`.167004.<keep_date>` (`link_keepdate_dcm.sh`, untracked in her checkout).

| | sub-05 ses-08 | sub-08 ses-04 | sub-10 ses-05 |
|---|---|---|---|
| links | 5346 | 5346 | 5352 |
| dates in file names | 20250820 only | 20250204 only | 20250703 only |
| broken links / duplicate file names | 0 / 0 | 0 / 0 | 0 / 0 |

The other study in each mixed folder (3509 files of 20250626 still sit in
`dicom_manual/sub-10_ses-05`) never reached heudiconv. File names are SOP
instance UIDs, so no DICOM was linked twice. **Correct.**

Findings:
- **[provenance]** `link_keepdate_dcm.sh` is not in this repo; no log of the
  link step. The selection rule only works for UIDs of the form
  `...167004.<YYYYMMDD>...`; `dicom_manual/MEMORY.md` lists exports where the
  date is not in the name (`...167004.30000025062610...`) — the script would link
  0 files there. → add it to `MR_pipelines/00_dicom_to_nifti/prepare/` with a
  header-based date fallback and a log (count per date kept / skipped).
- **[risk]** `dicom_manual/MEMORY.md` still says the mixed folders are
  "not cleaned / not converted"; the cleaning script was never applied (the link
  tree made it unnecessary). Update that note.

## 2. heudiconv → raw_nifti_missing3

Joined heudiconv's own records (`raw_nifti_missing3/.heudiconv/*/info/`:
`dicominfo`, `auto.txt`, `edit.txt`, `heuristic.py`) with the files and
`scans.tsv`, and with the lab note (`VOTCLOC_subses_list.xlsx`, per-subject sheet).

| | sub-05 ses-08 | sub-08 ses-04 | sub-10 ses-05 |
|---|---|---|---|
| series → converted | 88 → 60 | 88 → 60 | 94 → 66 |
| DICOMs processed | 5346 | 5346 | 5352 |
| wanted series not converted | none | none | none |
| converted file missing | none | none | none |
| scanner label (`fLoc_run03`) = output run | all | all | all |
| runs 1..N in acq-time order | yes | yes | yes |
| orphan sbref | none | none | none |
| lab-note runs = NIfTI runs | 18 = 18 | 18 = 18 | 21 = 21 (5 fmap pairs) |
| max lab-note vs acq_time | 3.5 min | 1.4 min | 3.8 min |

Not converted, by design: scouts, `SBRef_Pha`, DWI SBRefs. `edit.txt == auto.txt`
(no hand edits). The heuristic used is byte-identical to this repo's
`heuristic/heuristic_all_VOTCLOC.py`. **Correct.**

sub-10 ses-05 really has 5 AP/PA pairs: the lab note says the subject moved
before retRW run-02 and the operator added a fieldmap before each later run.

Findings:
- **[provenance]** Her `src_heudiconv_step2.sh` differs from this repo's: it had
  `--minmeta` hard-coded and the input glob `*/*.dcm` (repo: `*`). Effect:
  missing3 `_sbref.json` lack the `dcmmeta_*` / `global` blocks (3 KB vs 176 KB in
  older sessions). `_bold.json` are unaffected (no session has dcmmeta there).
  Harmless for fMRIPrep/GLM, but a QC that diffs JSON keys must allow it.
  → make `--minmeta` an explicit, logged option in
  `00_dicom_to_nifti/src_heudiconv_step2.sh`; decide one value for the dataset.
- **[provenance]** The launcher, heuristic path and step2 script came from her
  untracked checkout (`heuristic_all_votcloc.py`, lower case). The copy saved in
  `.heudiconv/` is what proves which heuristic ran. → the launcher should record
  `git rev-parse HEAD` and the heuristic's checksum in the log dir.
- **[risk]** The heuristic numbers runs with `{item}` (order within a key) and
  drops a bold series only when it has fewer than 159 (fLoc) / 155 (ret) files,
  while it keeps that run's 1-file SBRef. Any interrupted run therefore yields an
  **orphan sbref and an sbref/bold run-number shift**. Correct conversion of such
  sessions depends entirely on the sbref-drop step (§3, §5). → the new QC
  (`qc_heudiconv_series_map.py`, column `orphan_sbref`) catches it right after
  conversion.
- **[cosmetic]** "PROCESSING DONE" is written to `.e`, not `.o`; the fixes run has
  ~72 harmless `KeyError: 'PixelSpacing'` tracebacks per log, so "no traceback"
  is not a usable success test.

## 3. Rerun / run-gap fix

SOP (`launchcontainers/tests/dcm_to_bids_workflow/analysis_note.md`, Phase 0)
runs on `raw_nifti` before NORDIC: `00_merge_split_ses_reording_runs.py`,
`01_rename_ME_magnitude.py`, `02_copy_bids_component.py`,
`03_drop_duplicated_sbrefs.py`. siwang ran none of them on `raw_nifti_missing3`.

For these sessions none was needed: no split session folder, no interrupted
series, no run-number gaps, no orphan sbref, single-echo (§2). **No effect on
the output.**

- **[risk]** Skipping Phase 0 is only safe because these sessions are clean. Her
  account presents the BIDS-side dry run of the old sbref script as the sbref
  check (§5); for a session with a rerun, NORDIC copies sbrefs 1-to-1 by
  magnitude run number, so an un-dropped orphan sbref would land in BIDS with the
  wrong run. → the launcher of NORDIC should refuse a session whose raw_nifti has
  an orphan sbref (cheap check, see QC).

## 4. func NORDIC → BIDS

Code that ran: the log strings (`this is sub, 05`, `Gfactor orig file is
here…`, `sbref copied to %s`) exist only in `nordic_fmri.m` **before** a7e5678
(`git show a7e5678^:MR_pipelines/01_prepare_nifti/prepare_func/nordic_fmri.m`,
last changed 40bc64b, 2026-03-05), loaded from her checkout, with an a7e5678-style
launcher (`-f`, log dir named after the list). Same code family and the same NORDIC
parameters as tlei's May-2025 runs (temporal_phase 1, phase_filter_width 10,
noise_volume_last 1, kernel [14 14 1], NVR_threshold 1).

Outputs, 48/48 runs:
- 90 files per session = 16 × (bold, bold.json, gfactor, sbref, sbref.json) + 10
  events links; no phase/magnitude/temp files left in BIDS func.
- bold = raw − 1 volume (noise scan dropped): fLoc 160 → 159, ret 156 → 155; same as
  neighbours and as fMRIPrep confound rows.
- bold geometry = raw geometry (affine, 1.5 mm, TR 2 s); qform 0 / sform 1, same as
  every April session.
- `bold.json` dict-identical to the raw `_magnitude.json` of the same run;
  SeriesDescription matches the file name.
- sbref byte-identical to the raw sbref of the same run; 14.0 s before its bold;
  SeriesNumber = bold − 2.
- `.e` empty, 16 × each expected log line. **Correct.**

`raw_nifti_missing3/**/*_orig.nii.gz` are the pristine heudiconv files kept on
purpose (the working `_magnitude/_phase` were converted to float32 in place).

Findings (files: `MR_pipelines/01_prepare_nifti/prepare_func/*` in the old
layout; `launchcontainers/tests/dcm_to_bids_workflow/prepare_func/*` now):
- **[provenance]** `local_nordic.sh` does not copy itself, the subses list, the git
  commit or the NORDIC toolbox version into the log dir (the dwi and presurfer
  launchers copy themselves). Her checkout mixed an a7e5678 launcher with an older
  `nordic_fmri.m`, all untracked; which code made BIDS had to be reconstructed from
  log strings.
- **[risk]** `FORCE` never re-runs NORDIC: Step 2 skips a run whose `_bold.nii.gz`
  exists. In a7e5678 and the current code force *does* overwrite JSON/sbref → after
  a raw-data change a rerun gives a stale bold with a new sidecar/sbref.
- **[risk]** `prepare_nordic_bold_nifti.m`: if only one of `mag_orig` / `phase_orig`
  exists, the current (possibly float/trimmed) working files are copied over both
  backups; a non-uint16 input only prints `WARNINIG` and continues.
- **[risk]** No failure detection: `system()` return codes ignored; shell errors go
  to `.o`; "0 runs found" still prints "NORDIC finished!!" (tlei's
  `nordic_08_03_05;20.o`, May 2025). An empty `.e` is **not** a success criterion.
  `src_nordic_fmri.sh` runs MATLAB without `-batch` / exit status.
- **[risk]** tSNR branch (`dotsnr`) pairs BIDS bolds with raw magnitudes by index of
  two `dir()` listings (still in the current SOP). Not used here.
- **[risk]** `strrep` on full paths (`_magnitude→_phase`, `_bold→_gfactor`,
  `gfactor_→''`): a directory name containing those strings breaks pairing.
- **[risk]** `genpath(<src>/../code)` puts all of `$M/code` on the MATLAB path
  (nothing shadows today) plus a dead hard-coded tlei path.
- **[risk]** exactly one noise volume is dropped, never checked against the
  expected raw volume count.
- **[risk]** `chmod -R 777` on source and BIDS func; sidecars set 755.
- **[cosmetic]** `sbref copied to W_run-01_sbref.nii.gz` prints only the last 21
  characters (the copy itself uses the full path); numbers through `%s` (`^P`);
  `unset $SUB` unsets the wrong name; gfactor has qform/sform code 0.
- Outside scope, noted: neighbour **sub-05 ses-07** `bold.json` SeriesDescription is
  swapped between `task-retCB_run-01` (`retFF_run02`) and `task-retFF_run-02`
  (`retCB_run01`) — probably a deliberate rename after a console mislabel;
  **check against the lab note**.

## 5. fmap copy, sbref↔bold match, IntendedFor

She ran `launchcontainers/tests/prepare_fmriprep/{00b_copy_fmap_to_bids,
01_drop_duplicated_sbrefs,02_prepare_fmap_intendedfor}.py`. Those files were
deleted from `main` on 2026-05-13 (cda8d48) / 2026-06-30 (9cab781) and replaced by
`launchcontainers/tests/dcm_to_bids_workflow/` (`03_drop_duplicated_sbrefs.py` on
raw_nifti before NORDIC; `prepare_fmap/01_copy_fmap_to_bids.py`;
`prepare_fmap/02_prepare_fmap_intendedfor.py`, ME- and acq-aware). Her `notes`
branch predates that refactor. The old scripts (identical from 8d30164 to a7e5678)
were run here from `git show a7e5678:…` in dry-run mode: the old 02 reproduces what
is on disk exactly, so that is very likely what she ran (her checkout is not
readable, so not proven).

**Outputs: correct in all 3 sessions.**
- fmap copy: raw and BIDS fmap files identical (md5); JSONs differ only by the
  added `IntendedFor`. AP `j-`, PA `j`, bold `j-`. fmap TotalReadoutTime 0.0882679
  vs bold 0.0846279 — a protocol property (each = EffectiveEchoSpacing × 91; same in
  the neighbours), sdcflows uses each file's own value. No `B0Field*` anywhere
  (as in the neighbours); fMRIPrep built its estimators from `IntendedFor`.
- sbref↔bold: 16 = 16 per session, same task/run keys, bold − sbref = 14.0 s and
  SeriesNumber + 2 in all 48 pairs, same affine. BIDS sbref byte-identical to raw.
  Old 01 (BIDS), new 03 (raw) and new 03 (BIDS): 16/16 OK, 0 renames, 0 drops.
- IntendedFor: AP list = PA list; all 32 func files (16 bold + 16 sbref) covered
  **exactly once**; every path exists; every run is assigned the latest fmap pair
  before it; each bold's affine equals its fmap's affine (the slab was re-prescribed
  once per session — 3.5 / 7.6 / 5.5 mm — and the fmap assignment follows it).
  sub-05 ses-08 and sub-08 ses-04: fmap1 → fLoc 01–05 + ret*-01, fmap2 → ret*-02 +
  fLoc 06–10. sub-10 ses-05: fmap1 → fLoc 01–05, ret*-01 (8 runs, up to 50 min
  later); fmap2 → retRW-02; fmap3 → retFF-02; fmap4 → retCB-02; fmap5 → fLoc 06–10.
  fMRIPrep's `auto_00000…00004` fieldmap JSONs carry the same lists, so **the sub-10
  SDC failure (§7) is not caused by the BIDS input**.
- `00_merge_split_ses_reording_runs.py` dry run on raw_nifti_missing3: "OK (no
  changes)" in all 3; no `ses-XX*part*` folders.

Process vs SOP: she checked sbrefs on BIDS after NORDIC (SOP: on raw_nifti before
NORDIC). Here raw and BIDS sbrefs are identical, so no effect. Old vs new copy
script: byte-identical. Old vs new sbref script: only ME dedup and (task, acq)
renumbering, neither applies. Old vs new IntendedFor script: different time window
(see the error below), acq grouping, `--match` (timing / FOV), `--suffix`.

Her account vs disk: "each run assigned the fmap taken 38 s to 2 min before it" is
wrong as written (that is fmap → first sbref; fmap1 serves runs up to 50 min later),
but the assignment is right. "every session had IntendedFor" at 21:53 cannot be true
for missing3 (fmap JSON mtime 22:50:40). Every file of the 3 BIDS sessions and of
raw_nifti_missing3 has ctime 2026-11-17 18:19–18:22 with unchanged mtime: an
unrecorded permission sweep (probably the `chmod -R go+rwX` her note lists as open).

Findings:
- **[error, in the current SOP code — not on disk]**
  `dcm_to_bids_workflow/prepare_fmap/02_prepare_fmap_intendedfor.py`,
  `_assign_intendedfor` (l. 295–318): the window of fmap *i* is
  `[max(t_i − lookback, t_{i−1}), t_{i+1})`, which overlaps fmap *i−1*'s window
  `[…, t_i)`. Every run acquired within `--lookback` (10 min) **before** a later
  fmap lands in two IntendedFor lists. Dry run on sub-10 ses-05: fmap3 would get
  retRW-02 + retFF-02, fmap4 retFF-02 + retCB-02, fmap5 retCB-02 + fLoc 06–10, with
  no error. **Do not run the current 02 with `--execute` on sub-10 ses-05** (or any
  session with an fmap added mid-session). Fix: apply the lookback only to the first
  fmap (or only to funcs no other fmap covers). The old script (her version) is
  right here.
- **[risk]** coverage check (`_check_func_coverage`, old and new) only tests
  "covered at least once", so the error above passes silently → require exactly
  once and refuse to write otherwise.
- **[risk]** IntendedFor is written even when the session status is ERROR; in the new
  script `has_coverage_error` (l. 881) looks for `"NO FMAP BEFORE FIRST FUNC"`
  while the check now emits `"FIRST FUNC OUTSIDE LOOKBACK"` (l. 426) → BLOCKED
  never fires (dead code).
- **[risk]** FOV fallback (`_read_fov`, `--match both`) always triggers in VOTCLOC
  (`acq-fMRI` fmaps, func without `acq`), and its fingerprint (n_slices,
  ReconMatrixPE) ignores position/orientation, so it cannot see a slab
  re-prescription (which happened in all 3 sessions) → compare affines.
- **[risk]** fmap pairs not validated (`_read_fmap_files`): grouped by run number
  only; no check for exactly 1 AP + 1 PA with opposite PE.
- **[risk]** sbref matching (`_match_sbrefs`, old 01 / new 03) ignores task and
  renumbers by position per task instead of taking the matched bold's run number —
  wrong renames when bold numbering has gaps (her sub-01 ses-09 "do not apply"
  case); a bold without sbref or two sbrefs for one bold is not reported.
- **[risk]** `--execute` of the IntendedFor script deletes fmaps in BIDS
  (`_prune_fmaps`, `_prune_session_gap_fmaps`) with no backup; the copy script skips
  an existing destination without comparing content.
- **[risk, dataset-wide]** `scans.tsv` is copied verbatim from raw_nifti: it lists
  `_magnitude/_phase`, `T1_inv*/uni` that do not exist in BIDS and omits `_bold`,
  `_T1w`, nordic/magonly dwi. Same in every session; fMRIPrep ignores it, a
  validator would not → regenerate scans.tsv after NORDIC/presurfer.
- **[provenance]** copy, sbref check and IntendedFor printed to the terminal only
  (01/03 has `-o` for TSV logs, unused; 02 has no log option); script version
  unverifiable (untracked branch).
- **[cosmetic]** `_chmod755` makes data files executable; the 02 timeline regex
  shows `AP_run+PA_run`; `analysis_note.md` describes a `(bold|sbref|magnitude)`
  regex the code replaced with `--suffix`; `03_drop_duplicated_sbrefs.py` docstring
  still names `01_drop_duplicated_sbref_and_fmaps.py`.

## 6. events.tsv

10 fLoc links per session into `BIDS/sourcedata/sub-XX/ses-YY/<log folder>/`;
folder dates match the scan dates (20-Aug-2025, 04-Feb-2025, 03-Jul-2025), none
broken. sub-05 ses-08 uses Stimset2 labels (`face1`, `ES_RW1`) — like ~40 other
sessions; the GLM must use the matching condition names. **Correct.**
- **[provenance]** made by hand (`ln -sv`), no log; launchcontainers'
  `l1_surface` prepare (`prepare/glm_prepare.py`) builds events from the vistadisp
  logs and is the tracked way.

## 7. fMRIPrep 25.1.4

Run setup: same options as the April run (`fmriprep.toml` diff: only cpu/mem,
`notrack`, `stop_on_first_crash`, the BIDS filter pinning anat to ses-01 and
func/fmap to the session, and `--derivatives anat=/prev_fmriprep`). The reused
anatomy is the April ses-01 anatomy, which was not modified (only two report
SVGs per subject were redrawn). All 16 runs per session: PEPOLAR SDC, BBR 6 dof (no
fallback), 576 files in `func/`, file names identical to the neighbour session
after normalising `fmapid`. Crashes in the logs: sub-05 11-13 (HDF5 lock, earlier
attempt), sub-08 / sub-10 11-15 22:43 (`No space left on device`), all followed by
clean reruns; sub-05's final run finished 22:35, before the disk filled.

### sub-10: fieldmap → boldref registration failed

fMRIPrep (sdcflows `init_coeff2epi_wf`) registers each fieldmap reference to the
run's boldref (`func/*_from-boldref_to-<fmapid>_mode-image_xfm.txt`). In VOTCLOC
the SE-EPI fieldmaps, SBRef and BOLD have **the same grid, orientation and shim**
(checked in the headers) and are acquired seconds to minutes apart, so the true
transform is near identity.

Mean displacement over an 80 mm cube around the boldref centre, per run:

| session | median | max | max rotation | runs flagged (>5 mm or >3°) |
|---|---|---|---|---|
| sub-05 ses-08 | 1.8 mm | 3.7 mm | 2.2° | 0/16 |
| sub-08 ses-04 | 2.0 mm | 2.8 mm | 2.1° | 0/16 |
| **sub-10 ses-05** | **10.3 mm** | **18.7 mm** | **14.6°** | **16/16** |
| **sub-10 ses-04 (April, tlei)** | **11.3 mm** | **27.2 mm** | **19.9°** | **15/16** |

A 10–20° rotation between images acquired minutes apart in a head coil is not
physical. The `desc-fmapCoreg` report for sub-10 ses-05 retFF run-02 shows the
boldref resampled with a visible tilt (rotated black wedges at the tile corners),
while sub-05 ses-08 fLoc run-01 overlays cleanly. Even right after fmap 1
(runs 1–5, 1–2 min later) the offset is a consistent ~9 mm. So the fieldmap was
applied in the wrong place: **SDC for sub-10 is wrong** (smooth field, so the
corrected images still look plausible at a glance).

Not caused by siwang's run (ses-04 from April has it too) and not by BIDS
geometry. Cause inside sdcflows still open (e.g. boldref/fmap-ref masks or contrast
for this subject). What to do:
1. Dataset-wide numbers (heavy: ~1,800 header reads; run it yourself):
   ```bash
   ~/micromamba/envs/lc/bin/python MR_pipelines/03_fmriprep/quanlity_control/qc_fmap_boldref_xfm.py compute \
       -p $F -o $M/derivatives_qc/fmap_boldref_xfm   # choose the output dir
   ~/micromamba/envs/lc/bin/python MR_pipelines/03_fmriprep/quanlity_control/qc_fmap_boldref_xfm.py plot \
       -o $M/derivatives_qc/fmap_boldref_xfm
   ```
2. Decide a rerun for flagged sessions (options to test on one session: fMRIPrep
   without fieldmaps vs `--use-syn-sdc` vs a fixed sdcflows registration); the
   GLM / pRF results for flagged sessions should be marked until then.

Other fMRIPrep notes:
- **[provenance]** her fMRIPrep launcher (`launch_fmriprep_votcloc_missing.sh`) and
  `merge_fmriprep_missing.sh` are untracked in her checkout; the BIDS filters are in
  `$M/code/bids_filter_fmriprep_missing/`.
- non-steady-state volumes detected by fMRIPrep: up to 6 in one sub-05 ses-08 run,
  1–2 in several sub-10 ses-05 runs — relevant to the GLM, not an error.
- `fmriprep-25.1.4_t2w_fmapsbref_newest_missing` now holds only duplicates of the
  April anatomy (cleanup is the user's decision).

---

## What launchcontainers needs to change (summary)

| # | Component | Change | Severity |
|---|---|---|---|
| 1 | every prepare launcher (heudiconv, NORDIC func, fmap scripts, fMRIPrep) | record launcher copy, subses list, `git rev-parse HEAD`, container / toolbox version in the log dir; untracked per-user copies must not be the production path | provenance |
| 2 | `00_dicom_to_nifti/prepare/` | add the keep-date link-tree script with a header-date fallback and a log | provenance |
| 3 | `00_dicom_to_nifti/src_heudiconv_step2.sh` | `--minmeta` as an explicit option, one value per dataset | provenance |
| 4 | `00_dicom_to_nifti/quality_control/qc_heudiconv_series_map.py` (new) | run after every conversion; blocks NORDIC on `orphan_sbref` / `unexpected_unconverted` | risk |
| 5 | `prepare_func/nordic_fmri.m`, `prepare_nordic_bold_nifti.m`, `src_nordic_fmri.sh`, `local_nordic.sh` | real FORCE, safe backups, exit codes, basename-only string replace, no `chmod 777`, drop index pairing in tSNR, check volume counts | risk |
| 6a | `dcm_to_bids_workflow/prepare_fmap/02_prepare_fmap_intendedfor.py` | fix overlapping lookback window; coverage must be exactly once; gate writes on session status (fix dead `has_coverage_error`); FOV match by affine; validate AP/PA pairs; back up before pruning; add a TSV log | error / risk |
| 6b | `dcm_to_bids_workflow/03_drop_duplicated_sbrefs.py` | match within task; take the matched bold's run number; report bold-without-sbref; always write the TSV log | risk |
| 6c | `prepare_fmap/01_copy_fmap_to_bids.py` | regenerate `scans.tsv` for BIDS names instead of copying; compare content before skipping; no `chmod 755` | risk |
| 7 | `03_fmriprep/quanlity_control/qc_fmap_boldref_xfm.py` (new) | run after every fMRIPrep; flags failed SDC registration | error (sub-10) |
| 8 | workflow doc | Phase 0 is mandatory before NORDIC; success = QC CSV `ok`, not "empty .e" | risk |
