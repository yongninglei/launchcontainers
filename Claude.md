# CLAUDE.md — launchcontainers project context

Read automatically by Claude Code. Shared conventions (uv, src layout,
typer + rich, one sub/ses per call, `subseslist.txt`, launchers, release /
CITATION rules) are in `~/soft/CLAUDE.md`. The design contract for MRI-hub —
the Tool interface, provenance, run logs, QC — is
`../analysis-core/ARCHITECTURE.md`; it wins where this file disagrees.

**Open / handed-off work is tracked in `HANDOFF.md`. Read it first.**

---

## Project overview

**launchcontainers** (`lc`) is the launcher of MRI-hub: it prepares, runs and
QCs neuroimaging analyses on local machines and HPC clusters, one
subject × session at a time, BIDS-compliant. Developed at BCBL.

- **Repos**: working fork `github.com/yongninglei/launchcontainers` (`origin`,
  branch `main`); shared repo `github.com/MRI-hub/launchcontainers`
  (`master`, protected: 1 approving review); `upstream` =
  `garikoitz/launchcontainers`.
- **Built on** `analysis-core` (`github.com/MRI-hub/analysis-core`, private):
  work list, config loader + guard, provenance, run logs, Tool interface, QC
  report, slice renderer, rating page.
- **Entry points**: `lc` (`launchcontainers.cli:main`), `checker`
  (`analysis_checker.check_analysis_integrity:main`).
- **Python** 3.10+ (uv-managed 3.11 in `.python-version`).

---

## Install (development)

Clone the sibling repos next to this one — `pyproject.toml` takes them from
`../` via `[tool.uv.sources]`:

```
MRI-hub/
├── launchcontainers/     this repo
├── analysis-core/        required
└── pyGLM/                for the GLM (optional extra "glm")
```

```bash
uv sync                  # .venv from pyproject.toml + uv.lock (needs ../analysis-core)
uv sync --extra glm      # + pyGLM from ../pyGLM (Python 3.11–3.12)
uv run lc --help
```

- On SGE / Slurm: `uv sync` once on the submit host; jobs call
  `<repo>/.venv/bin/...` directly (uv cannot take file locks on cajal03, and
  parallel `uv run`s race on one `.venv`). `lc run` does this already: every
  job line uses the `analysis-session` next to the interpreter running `lc`.
- `requirements.txt` / `environment.yml` are derived from `pyproject.toml`
  (analysis-core as `git+https://github.com/MRI-hub/analysis-core`); update
  all three together.
- CI (`.github/workflows/main.yml`, docs job) checks out analysis-core and
  pyGLM next to the repo with the secret `ANALYSIS_CORE_TOKEN` (read-only PAT).

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
lc qc      -w <analysis_dir> [-ssl list] [--overwrite] [--serve --port --host]
```

The **analysis directory** is the source of truth: `run` and `qc` read only
its `lc_config.yaml` / `subseslist.txt`, never `basedir/code/`.

Every command resolves `general.tool` (alias `general.container`) to a
**Tool** — built-in (`tools/dwi.py`) or any package registered under the
`analysis_core.tools` entry point (pyGLM, autoROI, QC-registration) — and
loops over the work list; the tool does one sub/ses per call.

| Step | Module | What it does |
|---|---|---|
| prepare | `lc_prepare.py` | config guard (a changed config is refused unless `general.force`; runtime-only keys exempt), `dataset_description.json`, `tool.prepare_analysis()` once, `tool.prepare()` per session; run record in `logs/<analysis>_prepare_<stamp>/`, rows in `logs/runs.tsv` + `logs/sessions.tsv` (status `prepared` / `failed`) |
| run | `lc_run.py` | one line per session: `analysis-session --analysis-dir … --tool … --sub --ses -- <tool.command()>`, so each session records its own start/end/exit code/log (`logs/<dir>_sub-XX_ses-YY_<ts>/`, `sessions.tsv`). Backend from `host_options.<host>.manager` (`local` / `sge` / `slurm`), never the host's name. Writes `logs/run_<stamp>[_dryrun]/`: `commands.txt`, `jobs.tsv`, job script, scheduler .out/.err, `exit_codes.tsv`. Skips sessions whose last *prepare* status is not `prepared`. Without `-R`: dry run. |
| qc | `lc_qc.py` | per session `tool.qc()` (checks + pre-rendered PNGs via `analysis_core.slices`, fingerprinted so unchanged items are skipped) plus a `last run` check from `sessions.tsv` → `qc/index.html`, `qc/sub-XX_ses-YY.html`, `qc/qc.tsv`, `qc/failed_subseslist.txt` (feed back to `lc run -ssl`). `--serve` opens core's rating page; verdicts go to `qc/qc_rater-<name>.tsv` and a False makes the session `fail`. |

### Tools

| `general.tool` | Where | Notes |
|---|---|---|
| `anatrois`, `freesurferator`, `rtppreproc`, `rtp2-preproc`, `rtp-pipeline`, `rtp2-pipeline` | `tools/dwi.py` (`DWIContainer`) | wraps the staging code in `prepare/dwi_prepare.py`, `prepare/RTP2_prepare_input.py`, `gen_jobscript/gen_container_cmd.py`; `container_specific.<name>.version` required. LC-only hook `before_run()` (container JSON has `inputs`; back up `RTP_log.txt`). QC items: `segmentation` (anatrois/freesurferator), `dwi_on_t1` (preproc); pipelines: `RTP_log` exit-line check |
| `pyGLM` | MRI-hub/pyGLM | the GLM (`uv sync --extra glm`) |
| `autoROI-*` | MRI-hub/autoROI | |
| `qc-registration` | MRI-hub/QC-registration | registration QC; rendering is its `lc run` |
| `l1_surface` | `do_prepare.py` → `prepare/glm_prepare.py` | **prepare only**: GLM *input* preparation (events.tsv from vistadisp logs, BIDS / fMRIPrep BOLD symlinks), which pyGLM does not have. `lc run` / `lc qc` on it exit 2 and point to pyGLM |

To add an analysis: write a Tool in its own repo (ARCHITECTURE.md §2) and
register it under `analysis_core.tools`; launchcontainers needs no change.

---

## Code structure

```
src/launchcontainers/
├── cli.py               lc prepare / run / qc (typer); routes l1_surface prepare to do_prepare
├── lc_prepare.py        prepare loop on analysis-core (resolve_tool, load, prepare)
├── lc_run.py            run plan, wrapper lines, local / sge / slurm backends
├── lc_qc.py             qc loop, report, failed_subseslist
├── tools/dwi.py         the six DWI containers as Tools
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
tests/                   test_golden.py, test_run.py, test_qc.py, test_cli_routing.py;
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

- New analyses are Tools (in their own repo, registered by entry point), not
  new branches in `cli.py`. The loop over sessions lives in `lc` only.
- Shared code goes to analysis-core, never copied here; pyGLM is the
  reference for the GLM — no GLM fitting code in this repo.
- Nothing in an analysis dir is deleted: superseded configs go to
  `logs/history/`.
- typer + rich `console` for all output; colours `[green]saved`,
  `[yellow]SKIP`, `[red]ERROR` (exit 2 for config / input errors).
