# Changelog

Every change to launchcontainers gets an entry here, in its own branch, merged
into the fork (`yongninglei/launchcontainers` `main`) through a PR as soon as
the tests pass (no review wait, 2026-11-19); the PR upstream stays the
maintainer's. Each entry says **what**
changed, the **problem** it solves and **why** it was done this way — most lc
changes exist to match a program lc runs (its CLI, `--plan` JSON, outputs), so
name that program. Newest first. History before this file: `git log`.

## [Unreleased]

### Changed — docs: image QC split (branch `qc-split-docs`)

- **What:** `ARCHITECTURE.md` §6–7 and `Claude.md` Upstream / downstream: QC-registration
  QCs registration only; DWI segmentation QC is the new repo dwi-qc, SWI stage montages
  are swi-preproc's `swi-stage-qc`.
- **Problem / why:** maintainer rule (2026-11-19); lc's docs named the old commands.

### Removed
- The Sphinx source files (`docs/source/`, `Makefile`, `requirements.txt`, `_static/`,
  `_templates/`, …), deleted by the maintainer after the docs job was removed
  (2026-11-19: no Sphinx in any repo).

### Removed — Sphinx documentation (branch `remove-sphinx`)

- **What:** the CI `build-docs` job (Sphinx build + GitHub Pages deploy) is replaced by a
  `test` job (`uv run pytest -q tests`); the `docs` dependency group (sphinx,
  pydata-sphinx-theme, sphinx-copybutton, nbsphinx, myst-parser) is gone from
  `pyproject.toml` / `uv.lock`; `Claude.md`, `README.md`, `HANDOFF.md` no longer describe
  a docs site. The Sphinx source files are left for the maintainer to delete (agents do
  not delete files): `docs/source/`, `docs/Makefile`, `docs/make.bat`,
  `docs/requirements.txt`, `docs/README.md`, `docs/_static/`, `docs/_templates/`,
  `docs/_build/` (untracked build output). `docs/plan_protocol.md` and
  `docs/pipeline_qc_audit/` stay.
- **Problem:** the per-repo Sphinx automation cost more than it gave (jobs failing or
  disabled, pages out of date since 0.5.0).
- **Why this way:** maintainer decision (2026-11-19): no Sphinx in any repo; if rendered
  docs come back, they are built in one place for all repos. The docs are `Claude.md`,
  `ARCHITECTURE.md`, `docs/plan_protocol.md`.

### Changed — `AGENTS.md` names the PR + auto-merge rule (branch `agents-merge-rule`)

- **What:** `AGENTS.md` item 3 now says what `Claude.md` "How changes land" says since
  PR #19: PR to the fork's `main`, merged by the agent once the tests pass.
- **Problem:** it still said "merged into the fork's `main`; the maintainer reviews", so an
  agent reading only `AGENTS.md` would follow the old rule.
- **Why this way:** one rule in every file an agent reads first.

### Added — `bold2surf` as an external tool; changes land by PR + auto-merge (branch `bold2surf-tool`)

- **What:** `tools/external.py` registers `ExternalTool("bold2surf", "mrireg-bold2surf")`;
  `Claude.md`: its row in the tool table, an `## Upstream / downstream` section, and the new
  "How changes land" (PR to the fork's `main`, merged by the agent once the tests pass).
- **Problem:** mri-registration's `mrireg-bold2surf` (fMRIPrep T1w BOLD → dense x1 mesh,
  `derivatives/bold_dense/`) already speaks the `--plan` protocol (`general.tool: bold2surf`),
  but lc did not know it, so dense BOLD could only be made with its own batch runner, without
  lc's run logs and launch QC. The maintainer asked to register it (and not avg-timeseries).
  Merges waited for a manual review that the maintainer no longer wants for the fork.
- **Why this way:** a plain `ExternalTool`, like qmri-preproc; nothing else changes in lc.
  The merge rule is the one in `~/soft/CLAUDE.md` for every MRI-hub repo.

### Added — `qmri-preproc` as an external tool (branch `qmri-preproc-tool`)

- **What:** `tools/external.py` registers `ExternalTool("qmri-preproc", "qmri-preproc")`;
  `Claude.md`'s tool table has its row.
- **Problem:** qMRI-preproc 0.2.0 replaced its five `mmr-*` CLIs, batch runners and qsub
  launchers with one single-session program, `qmri-preproc --config --sub --ses [--plan]`
  (register, apply, surface sampling of one MRF session into
  `derivatives/qmri_preproc/analysis-<name>/sub-XX/ses-YY/`). lc did not know it, so the
  sessions could only run by hand.
- **Why this way:** it speaks the `--plan` protocol (`docs/plan_protocol.md`) like
  swi-preproc, so it is a plain `ExternalTool`; nothing else changes in lc. Its input step,
  qMRI-to-bids (MRF recon outputs -> the raw-style dataset `BIDS_qMRI`), is deliberately NOT
  registered: it writes a BIDS dataset, not an analysis dir, and lc prepare would put its own
  `dataset_description.json` and logs there.

### Changed — `swi-analysis` replaces `pial-vein` (branch `swi-analysis-tool`)

- **What:** `tools/external.py` registers `ExternalTool("swi-analysis", "swi-analysis")` and no
  longer `pial-vein`; `Claude.md`'s tool table updated (also MRI-hub/swi_preproc →
  MRI-hub/swi-preproc, the repo's new name).
- **Problem:** swi-analysis 0.3.0 replaced its three CLIs (`pial-vein`, `pial-vein-display`,
  `swi-vessels`) with one, `swi-analysis`, that runs every step from one config section
  (`container_specific.swi-analysis`) into one flat folder `derivatives/swi_analysis/
  analysis-<name>/sub-XX/ses-YY/`; lc still knew only `pial-vein`, and `swi-vessels` was never
  registered (it ran by hand).
- **Why this way:** the maintainer's output rule (one derivatives folder per repo, flat
  sessions, every parameter in one YAML; `~/soft/CLAUDE.md` §6). `swi-analysis` speaks the same
  `--config / --sub / --ses / --plan` protocol, so it is a plain `ExternalTool`, nothing else
  changes in lc. Existing `pial_vein/` analyses were moved into `swi_analysis/analysis-test`.

### Fixed — `Claude.md` named the wrong changelog (branch `docs-changelog-path`)

- **What:** the header of `Claude.md` now says every change goes into
  `CHANGELOG.md` under `## [Unreleased]`, and that `docs/source/changelog.rst`
  is the older Sphinx changelog not to be added to.
- **Problem:** the header still pointed agents at `docs/source/changelog.rst`,
  while "How changes land" in the same file and `AGENTS.md` say `CHANGELOG.md`;
  agents got two contradictory instructions.
- **Why this way:** `CHANGELOG.md` is the one in use since 2026-11-17; one place
  for entries.

### Removed — pre-commit (branch `remove-pre-commit`)

- **What:** deleted `.pre-commit-config.yaml` (trailing-whitespace, end-of-file,
  debug-statements, ruff, ruff-format hooks) and the local `.git/hooks/pre-commit`;
  dropped `pre-commit` from the `dev` dependency group (`uv.lock` relocked).
  The ruff settings in `pyproject.toml` stay, so `ruff check` / `ruff format` can
  still be run by hand or by an agent.
- **Problem:** the installed hook pointed to a Python under `/export`, which does
  not exist on brodmann, and `pre-commit` was not on PATH there, so every commit
  from brodmann failed until the hook was run from `.venv` by hand (hit committing
  `pipeline-qc-audit`, 2026-11-18).
- **Why this way:** the maintainer's decision: changes are now written and
  checked with AI agents, so a commit-time hook adds friction without catching
  anything those reviews and the test suite do not. Nothing else (CI included)
  ran pre-commit.

### Added — pipeline QC audit: review of the VOTCLOC missing-3 fMRI conversion + two QC scripts (branch `pipeline-qc-audit`)

- **What:** `docs/pipeline_qc_audit/` (review log, module-split notes, index);
  `MR_pipelines/00_dicom_to_nifti/quality_control/qc_heudiconv_series_map.py`
  (joins heudiconv's `dicominfo` / `auto.txt` with the converted files, the DICOM
  input tree and the lab note; one CSV per session + a summary with an `ok`
  column); `MR_pipelines/03_fmriprep/quanlity_control/qc_fmap_boldref_xfm.py`
  (`compute`: size of fMRIPrep's fieldmap → boldref rigid transform per run,
  flags > 5 mm or > 3°; `plot`: reads that CSV). No existing code changed.
- **Problem:** three VOTCLOC sessions (sub-05 ses-08, sub-08 ses-04, sub-10
  ses-05) were converted DICOM → BIDS → fMRIPrep in Nov 2026 from a per-user,
  untracked checkout with scripts that had been removed from `main`; no step had a
  QC that could show the result was right. The review found the outputs correct up
  to BIDS, but (1) **fMRIPrep's fieldmap → boldref registration failed for sub-10**
  (ses-05 16/16 runs, and tlei's April ses-04 15/16: 10–27 mm, up to 20°, between
  images on the same grid acquired minutes apart), so SDC is wrong there; and
  (2) the **current** SOP script
  `tests/dcm_to_bids_workflow/prepare_fmap/02_prepare_fmap_intendedfor.py` has
  overlapping lookback windows (`_assign_intendedfor`) and dead BLOCKED logic, and
  would put sub-10 ses-05's ret run-02s in two fieldmaps if run with `--execute`.
  Plus NORDIC `force` / backup / exit-code risks and missing provenance in every
  launcher (full list with files: the review's last table).
- **Why this way:** the review only records and measures; fixes go on their own
  branches (one per component, as `Claude.md` asks). The QC scripts follow the
  lab's pattern (compute → CSV, plotting from the CSV) so they can later become the
  QC of standalone modules (`docs/pipeline_qc_audit/module_split_plan.md`).
  The fMRIPrep QC reads only text transforms and NIfTI headers, but dataset-wide
  that is ~1,800 header reads, so it takes `-s` for a few sessions and the full run
  is left to the user.

### Added — external tools `swi-preproc`, `pial-vein`; provenance description (branches `swi-preproc-tool`, `provenance-description`, `pial-vein-tool`)

- **What:** `swi-preproc` (MRI-hub/swi_preproc main stage) and `pial-vein`
  (yongninglei/pial-vein-sampling) registered as ExternalTools next to
  `swi-prep` and `eyetrack`; `general.description` and the `--plan`
  `source_datasets` are written into `dataset_description.json`
  (`Description`, `SourceDatasets` / `DatasetLinks`).
- **Problem / why:** detailed in `docs/source/changelog.rst` "Unreleased
  (0.5.1)", written on these branches before this file existed.

### Fixed — `lc run` crashed on external tools' session folders (branch `fix-run-example-folder`)

- **What:** `check/general_checks.cli_show_folder_struc` (the "example session
  folder" `lc run` prints before launching) shows `sub-XX/ses-YY/input` when it
  exists, else `sub-XX/ses-YY`, runs `tree` with `check=False`, and falls back
  to printing the path when `tree` is not installed. Test:
  `test_example_session_folder_never_stops_a_run`.
- **Problem:** the helper assumed the DWI-container layout (`ses-YY/input`) and
  ran `tree` with `check=True`. External tools (eyetrack, pyGLM, autoROI) write
  no `input/` folder, so once the first session folder existed — i.e. on every
  rerun — `tree` exited 2 and `lc run` died with `CalledProcessError` before
  launching anything. Found rerunning the VOTCLOC eyetrack analysis
  (2026-11-17).
- **Why this way:** the listing is display only; a run must never stop on it,
  and matching the external tools' layout (session folder, no `input/`) keeps
  it useful for them.

### Fixed — test suite: 3 failures + 22 errors (branch `fix-tests`)

- **What:** `setup.cfg` no longer sets `[tool:pytest] log_cli = true`;
  `tests/fakedata.py` gains `require_extra()`, called by the fake-dataset
  fixtures of `test_golden.py`, `test_qc.py`, `test_run.py` (dwi extra) and by
  `test_l1_surface_prepare_still_prepares_glm_inputs` (fmri extra).
- **Problem 1 (2 failures,** `test_l1_surface_run_and_qc_point_to_pyglm[run|qc]`**):**
  `ValueError: I/O operation on closed file` inside typer's `CliRunner`. With
  `log_cli`, pytest suspends/resumes output capture around every log record and
  swaps `sys.stdout`; lc's console forwards each print to the
  `launchcontainers` logger, so the swap drops the last reference to the
  runner's stdout wrapper, whose `__del__` closes the buffer before the runner
  reads it. The same commands work outside pytest — a test-config bug, not an
  lc bug. Live logging only echoed lc's messages a second time.
- **Problem 2 (22 errors + 1 failure):** the tests that build the fake DWI
  dataset need nibabel / pybids (the `dwi` extra), and the l1_surface prepare
  test needs the `fmri` extra; in a core-only install (`uv sync`, supported on
  purpose — Claude.md) they crashed with `ModuleNotFoundError`. They now skip
  with the `uv sync --extra …` to run.
- **Result:** core install 49 passed, 23 skipped; with
  `uv sync --extra dwi --extra fmri` all 72 pass (so the skipped tests are
  green, not hidden). Run the full suite with the extras before merging lc
  changes that touch DWI tools or l1_surface prepare.

### Added — `eyetrack` external tool (branch `eyetrack-tool`)

- **What:** `tools/external.py` registers `EYETRACK = ExternalTool("eyetrack",
  "eyetrack")`; `Claude.md` lists it in the tool table.
- **Problem:** VOTCLOC needs the EyeLink eye-tracking QC of every fLoc run
  (MRI-hub/eyetracker_analysis: EDF ↔ BOLD pairing by clock, gaze
  reconstruction, per-run verdict pass / fail / unreliable / no_data for the
  GLM) run session by session like the other analyses, with provenance — so
  far only a plain bash loop could run it.
- **Why this way:** `eyetrack` already follows the plan protocol
  (`eyetrack --config lc_config.yaml --sub S --ses E [--plan]`, plan JSON with
  inputs / outputs / problems), exactly like pyGLM and autoROI, so the generic
  `ExternalTool` is enough — no eyetrack-specific code in lc. The analysis-wide
  table (`eyetrack-summary`) is not a session unit and stays outside lc.
- **Note:** branch `swi-preproc-tool` (not merged yet) also edits the
  `EXTERNAL` tuple: merging it after this one is a one-line conflict — keep
  both `SWI_PREPROC` and `EYETRACK`.
