# CLAUDE.md — launchcontainers project context

Read automatically by Claude Code. Shared conventions (uv, src layout,
typer + rich, one sub/ses per call, `subseslist.txt`, launchers, release /
CITATION rules) are in `~/soft/CLAUDE.md`. The design contract for MRI-hub —
the Tool interface, provenance, run logs, QC — is
`ARCHITECTURE.md` (repo root); it wins where this file disagrees.

**Open / handed-off work is tracked in `HANDOFF.md`. Read it first.**
**Every change is written into `CHANGELOG.md` (under `## [Unreleased]` at the
top) in the same commit: what changed, the problem it solves, and why this way.**
That file is how the next person or agent learns what happened
(`docs/source/changelog.rst` is the older Sphinx changelog; do not add to it).

---

## Project overview

**launchcontainers** (`lc`) is the launcher of MRI-hub: it prepares, runs and
QCs neuroimaging analyses on local machines and HPC clusters, one
subject × session at a time, BIDS-compliant. Developed at BCBL.

- **Repos**: working fork `github.com/yongninglei/launchcontainers` (`origin`,
  branch `main`); shared repo `github.com/MRI-hub/launchcontainers`
  (`master`, protected: 1 approving review); `upstream` =
  `garikoitz/launchcontainers`.
- **How changes land (every agent, every change).** Never commit to `main`
  directly. (1) Branch from `origin/main` (one topic per branch, e.g.
  `eyetrack-tool`). (2) Add an entry under `## [Unreleased]` in `CHANGELOG.md`:
  what changed, the problem it solves, and why this way — usually to match a
  program lc runs (its CLI, `--plan` JSON, outputs). (3) Commit, merge into the
  fork's `main` (`--no-ff`) and push `origin main` + the branch. (4) The
  maintainer reviews the fork and opens the PR to upstream. The changelog is how
  the next agent learns what was done and why — read it before changing lc.
- **Division of work.** launchcontainers prepares inputs, records provenance,
  launches, and checks that each launch worked. The analyses it runs (pyGLM,
  autoROI, …) are standalone programs that assume their inputs are ready:
  they do not import launchcontainers and lc does not import them. lc holds
  the rules for running each one (`tools/`). Image QC (rendering slices,
  rating pages) is not lc's job: that is the image-QC module, QC-registration.
- **The shared core is inside this repo**: `src/launchcontainers/core/`
  (`launchcontainers.core`) — work list, config loader + guard, provenance,
  run logs, Tool interface, session wrapper (`lc-session`), launch-status
  report. Until 0.5.0 (2026-11-16) it was the separate package
  `analysis-core` (MRI-hub/analysis-core); see ARCHITECTURE.md for the renames.
- **Core install is light** (pyyaml, rich, typer). Heavy dependencies are
  extras: `dwi` (DWI tools), `fmri` (l1_surface input prep), `checker`,
  `heudiconv`, `scripts` (study scripts: tedana, nilearn, …), `all`. A DWI
  tool or l1_surface prepare without its extra stops with the
  `uv sync --extra …` to run.
- **Entry points**: `lc` (`launchcontainers.cli:main`), `checker`
  (`analysis_checker.check_analysis_integrity:main`).
- **Python** 3.10+ (uv-managed 3.11 in `.python-version`).

---

## Install (development)

```bash
uv sync                  # core only (lc, lc-session); enough for pyGLM / autoROI
uv sync --extra dwi      # + the DWI container tools
uv sync --extra all      # everything
uv run lc --help
```

pyGLM, autoROI, … are installed in **their own** environments (their repos'
`uv sync`); lc runs their programs. Tell lc where each program is, per host:

```yaml
host_options:
  BCBL:
    manager: sge
    exe:
      pyGLM: /home/tlei/soft/MRI-hub/pyGLM/.venv/bin/pyglm
      autoROI-scaling: /home/tlei/soft/MRI-hub/autoROI/.venv/bin/autoroi-scaling
```

- On SGE / Slurm: `uv sync` once on the submit host; jobs call
  `<repo>/.venv/bin/...` directly (uv cannot take file locks on cajal03, and
  parallel `uv run`s race on one `.venv`). `lc run` does this already: every
  job line uses the `lc-session` next to the interpreter running `lc`.
- `requirements.txt` / `environment.yml` are derived from `pyproject.toml`
  (core deps only; extras are listed in their comments); update all three
  together.
- CI (`.github/workflows/main.yml`, docs job) needs no token: lc depends on
  no private repo. It was disabled with `gh workflow disable`; re-enable with
  `gh workflow enable main.yml`.

---

## Host paths — `/export/home` vs `/home`

| Machine | Home prefix |
|---|---|
| ips, lmc02, cajal03, cajal04 | `/export/home/tlei/...` |
| brodmann | `/home/tlei/...` (`/export` does not exist) |

- Swap the prefix when moving a path between these machines; `/bcbl/...`
  paths are the same everywhere. On brodmann `/home/tlei/tlei` resolves to
  `/bcbl/home/home_n-z/tlei`.
- Lab DICOMs: links in `VOTCLOC/main_exp/dicom` written on the cluster point to
  `/export/home/tlei/lab/MRI/VOTCLOC_22324/...`; on brodmann the data is at
  `/bcbl/data/MRI/VOTCLOC_22324/...` (use this form for new links).
- The heudiconv `src_heudiconv_step*.sh` scripts bind `/export` only when it
  exists, so they run on both kinds of host.

---

## Workflow: prepare → run → qc

```
lc prepare -lcc lc_config.yaml -ssl subseslist.txt [-cc container.json]
lc run     -w <analysis_dir> [-R] [-ssl list] [-y]
lc qc      -w <analysis_dir> [-ssl list]
```

The **analysis directory** is the source of truth: `run` and `qc` read only
its `lc_config.yaml` / `subseslist.txt`, never `basedir/code/`.

Every command resolves `general.tool` (alias `general.container`) to a
**Tool** — built-in (`tools/dwi.py`, `tools/external.py`), or any package
registered under the `launchcontainers.tools` entry point — and loops over the
work list; the tool does one sub/ses per call.

| Step | Module | What it does |
|---|---|---|
| prepare | `lc_prepare.py` | config guard (a changed config is refused unless `general.force`; runtime-only keys `force`, `host`, `description` exempt), `dataset_description.json` (Description from `general.description`, SourceDatasets = raw + the tool's `source_datasets`; old file archived when either changes), `tool.prepare_analysis()` once, `tool.prepare()` per session; run record in `logs/<analysis>_prepare_<stamp>/`, rows in `logs/runs.tsv` + `logs/sessions.tsv` (status `prepared` / `failed`) |
| run | `lc_run.py` | one line per session: `lc-session --analysis-dir … --tool … --sub --ses -- <tool.command()>`, so each session records its own start/end/exit code/log (`logs/<dir>_sub-XX_ses-YY_<ts>/`, `sessions.tsv`). Backend from `host_options.<host>.manager` (`local` / `sge` / `slurm`), never the host's name. Writes `logs/run_<stamp>[_dryrun]/`: `commands.txt`, `jobs.tsv`, job script, scheduler .out/.err, `exit_codes.tsv`. Skips sessions whose last *prepare* status is not `prepared`. Without `-R`: dry run. |
| qc | `lc_qc.py` | launch checks only, per session: `last run` (exit status from `sessions.tsv`), `run log` (the lc-session log scanned for tracebacks / error lines), and `tool.qc()` (expected outputs exist and are not empty; DWI pipelines: `RTP_log` exit line) → `qc/index.html`, `qc/sub-XX_ses-YY.html`, `qc/qc.tsv`, `qc/failed_subseslist.txt` (feed back to `lc run -ssl`). No images: image QC is QC-registration's job. |

### Tools

| `general.tool` | Where | Notes |
|---|---|---|
| `anatrois`, `freesurferator`, `rtppreproc`, `rtp2-preproc`, `rtp-pipeline`, `rtp2-pipeline` | `tools/dwi.py` (`DWIContainer`) | wraps the staging code in `prepare/dwi_prepare.py`, `prepare/RTP2_prepare_input.py`, `gen_jobscript/gen_container_cmd.py`; `container_specific.<name>.version` required. LC-only hook `before_run()` (container JSON has `inputs`; back up `RTP_log.txt`). QC: outputs present, `.err` log, pipelines' `RTP_log` exit line |
| `pyGLM` | `tools/external.py` → program `pyglm` (MRI-hub/pyGLM, own env) | the GLM; one analysis dir per strategy, from its `--plan` |
| `autoROI-scaling`, `autoROI-clustering` | `tools/external.py` → `autoroi-scaling` / `autoroi-clustering` (MRI-hub/autoROI, own env) | a work-list `sources` column becomes `--sources a,b` (pseudo-sessions) |
| `swi-prep` | `tools/external.py` → `swi-prep` (MRI-hub/swi_preproc, own env) | SWI raw QC: echo average + rigid MC of every run to run-01; greedy must be on PATH (cajal: `module load vtk/9.1 itk/5.4.5 greedy/latest`) |
| `eyetrack` | `tools/external.py` → `eyetrack` (MRI-hub/eyetracker_analysis, own env) | EyeLink EDF ↔ BOLD pairing by clock, gaze reconstruction, per-run eye QC verdict; then `eyetrack-summary` (not lc) for the analysis-wide run table |
| `swi-preproc` | `tools/external.py` → `swi-preproc` (MRI-hub/swi_preproc, own env) | SWI main stage (Faruk): crop, 2×, MC at 2×, composite of the chosen runs, N4, T1w registration; needs c3d, greedy, ANTs on PATH (cajal: `module load vtk/9.1 itk/5.4.5 c3d/latest greedy/latest ants/2.5.1`) |
| `pial-vein` | `tools/external.py` → `pial-vein` (yongninglei/pial-vein-sampling, own env) | swi_preproc's SWI on dense (twice-subdivided) fMRIPrep surfaces: A outward from pial, B subdural envelope. No modules needed |
| `l1_surface` | `do_prepare.py` → `prepare/glm_prepare.py` | **prepare only**: GLM *input* preparation (events.tsv from vistadisp logs, BIDS / fMRIPrep BOLD symlinks), which pyGLM does not have. `lc run` / `lc qc` on it exit 2 and point to pyGLM |

External analyses follow one contract, `docs/plan_protocol.md`: a
single-session CLI (`--config --sub --ses`) plus `--plan`, which prints JSON
(analysis dirs, session outputs, input files, problems) and computes nothing.
lc uses it for prepare (problems → `blocked`; inputs recorded in
`<analysis_dir>/logs/inputs/sub-S_ses-E.tsv`), run and launch QC. To add an
analysis: give its CLI `--plan` and add one `ExternalTool(name, default_exe)`
line in `tools/external.py`.

---

## Code structure

```
src/launchcontainers/
├── cli.py               lc prepare / run / qc (typer); routes l1_surface prepare to do_prepare
├── core/               the shared core (formerly analysis-core): subseslist, config, provenance,
│                        runlog, tool, session (lc-session), qc (launch-status report)
├── lc_prepare.py        prepare loop (resolve_tool, load, prepare)
├── lc_run.py            run plan, wrapper lines, local / sge / slurm backends
├── lc_qc.py             qc loop, report, failed_subseslist
├── tools/dwi.py         the six DWI containers as Tools
├── tools/external.py    pyGLM, autoROI-*: their programs via CLI + --plan
├── do_prepare.py        legacy prepare, now only for l1_surface (and golden capture)
├── prepare/             dwi_prepare.py, RTP2_prepare_input.py (DWI staging, used by tools/dwi.py);
│                        glm_prepare.py + base_prepare.py (l1_surface input prep);
│                        prf_prepare.py, gen_bids_derivatives.py, zip_freesurfer.py
├── gen_jobscript/       gen_container_cmd.py (apptainer command per session)
├── check/               general_checks.py (folder tree print), check_dwi_pipelines.py
├── helper_function/     copy_configs, create_bids, gen_subses, zip_example_config, mrhelp
├── quality_control/     standalone RTP2 / stats-map QC scripts (not wired into lc)
├── log_setup.py, utils.py
src/analysis_checker/    `checker`: output-integrity specs (bids, fmriprep, prf, qmri, rtp);
                         the glm spec was removed (pyGLM's qc covers it)
tests/                   test_golden.py, test_run.py, test_qc.py, test_cli_routing.py; core/ (the core's own tests);
                         fakedata.py (fake BIDS + derivatives), golden/ (snapshots)
launchcontainers/tests/  study scripts (VOTCLOC GLM strategy, FIR, power analysis, prepare_glm, …);
                         to move to VOTCLOC — not part of the package
MR_pipelines/            per-stage pipeline scripts (dicom→nifti, freesurfer, fmriprep, GLM inputs, pRF)
```

---

## Tests

```bash
uv run pytest -q tests
```

- `tests/test_golden.py`: the Tool-based prepare must stage exactly what the
  legacy code staged for 13 scenarios (`tests/golden/scenarios.py`; snapshots
  of commit a7448d9 in `tests/golden/data/`). Any change to DWI staging must
  keep it green or re-capture deliberately — `tests/golden/capture_legacy.py`
  drives the legacy code, so run it from a checkout of a7448d9.
- Tests use the fake dataset in `tests/fakedata.py` only, never real data.

---

## Configuration (`lc_config.yaml`)

```yaml
general:
  basedir: /bcbl/home/public/Gari/VOTCLOC/main_exp
  bidsdir_name: BIDS
  tool: rtp2-pipeline          # key under container_specific (alias: container)
  analysis_name: main
  host: BCBL                   # key under host_options
  deriv_layout: legacy         # legacy: <tool>_<ver>/analysis-<name>; new: <tool>-<ver>_<name>
  force: false
container_specific:
  rtp2-pipeline: {version: 0.2.1_3.0.4, precontainer_anat: …, anat_analysis_name: …, …}
host_options:
  BCBL:  {manager: sge,   queue: long.q, walltime: "24:00:00", job_name: lc, sge_extra: ["-pe <pe_name> 6"], …}   # BCBL's PE name: check `qconf -spl`
  DIPC:  {manager: slurm, partition: general, qos: regular, cores: 6, memory: 32G, walltime: …}
  local: {manager: local, n_workers: 2, mem_per_job: 32g}
```

- `manager` picks the backend. Optional: `sge_extra` / `slurm_extra` (extra
  directive lines), `n_workers` / `mem_per_job` (local; the old
  `launch_mode: parallel` + `max_workers` still works).
- `host`, `force` and `host_options` are runtime-only: changing them never
  trips the config guard.

---

## Sphinx documentation

- `docs/source/` (lower-case file names: `installation.rst`, `concepts.rst`,
  `configuration.rst`, `prepare_*.rst`, `api.rst`, …); theme
  `pydata_sphinx_theme`; build `cd docs && make html`.
- `pandas` is out of `intersphinx_mapping` (server returns 522);
  `suppress_warnings = ["intersphinx.fetch_inventory"]`; all toctrees in
  `index.rst` are `:hidden:`.
- `code_structure.rst`, `extending.rst` and parts of `configuration.rst`
  still describe the pre-restructure code; this file is current.

---

## Key conventions

Rules for `src/launchcontainers/core/` (from the former analysis-core):
- Imports only pyyaml, rich, typer and the standard library. Never add a
  heavy dependency to the core install.
- It is the only copy of runlog, subseslist and the generic provenance code;
  tools import it, never copy it.
- No analysis imports the core; the contract with them is their CLI and
  `--plan` (docs/plan_protocol.md). File formats (runs.tsv / sessions.tsv
  columns, `subseslist.txt`, `logs/inputs/*.tsv`) are a contract with users.

- New analyses are Tools (in their own repo, registered by entry point), not
  new branches in `cli.py`. The loop over sessions lives in `lc` only.
- Launching, provenance and run logs live only here; analyses never carry them. pyGLM is the
  reference for the GLM — no GLM fitting code in this repo.
- Nothing in an analysis dir is deleted: superseded configs go to
  `logs/history/`.
- typer + rich `console` for all output; colours `[green]saved`,
  `[yellow]SKIP`, `[red]ERROR` (exit 2 for config / input errors).
