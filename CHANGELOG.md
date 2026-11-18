# Changelog

Every change to launchcontainers gets an entry here, in its own branch, before
it is merged into the fork (`yongninglei/launchcontainers` `main`); the
maintainer reviews it there and opens the PR upstream. Each entry says **what**
changed, the **problem** it solves and **why** it was done this way — most lc
changes exist to match a program lc runs (its CLI, `--plan` JSON, outputs), so
name that program. Newest first. History before this file: `git log`.

## [Unreleased]

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
