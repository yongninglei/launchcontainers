# Changelog

Every change to launchcontainers gets an entry here, in its own branch, before
it is merged into the fork (`yongninglei/launchcontainers` `main`); the
maintainer reviews it there and opens the PR upstream. Each entry says **what**
changed, the **problem** it solves and **why** it was done this way — most lc
changes exist to match a program lc runs (its CLI, `--plan` JSON, outputs), so
name that program. Newest first. History before this file: `git log`.

## [Unreleased]

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
