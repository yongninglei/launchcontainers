# CLAUDE.md — launchcontainers project context

This file is read automatically by Claude Code. It provides full context
about the project so you can contribute effectively without needing to
re-explain the architecture each session.

Shared neuroimaging script conventions (typer/rich CLI, one-sub-per-call,
parallel option, timing, bash launcher skeleton) are in `../CLAUDE.md`.

---

## Project overview

**launchcontainers** is an open-source Python framework for launching
neuroimaging analysis pipelines on HPC clusters in a reproducible,
BIDS-compliant way. Developed at BCBL (Basque Center on Cognition, Brain
and Language). Right know it is under development.

- **Repo**: https://github.com/garikoitz/launchcontainers
- **Package manager**: Poetry (`pyproject.toml`)
- **Entry points**: `lc` (pipeline launcher), `checker` (integrity checker)
- **Python**: 3.10+

---

## Install (development)

```bash
pip install pipx && pipx install poetry
git clone https://github.com/garikoitz/launchcontainers.git
cd launchcontainers
poetry env use python3 && poetry install
poetry shell
```

---

## Core architecture — three-phase workflow

```
lc prepare  →  lc run  →  lc qc (not implemented yet, maybe move the checker funciton to here)
```

- **prepare**: validates inputs, creates analysis directory under
  `BIDS/derivatives/`, freezes configs, generates per-subject HPC scripts
- **run**: submits scripts to SLURM / SGE / local
- **qc**: checks outputs, writes `failed_subseslist.tsv` for re-runs

The **analysis directory** is the source of truth — `run` and `qc` read
only from it, never from the original `basedir/code/` files.

---

## Code structure

```
launchcontainers/
├── cli.py                        ← argparse entry point; dispatches all subcommands
├── do_prepare.py                 ← DWI container prepare (legacy path, do not modify)
├── do_launch.py                  ← job submission
├── do_qc.py                      ← quality control
├── utils.py                      ← shared utilities (read_yaml, read_df, …)
├── config_logger.py              ← This one maybe not useful, check and maybe remove it, since we are moving
                                    to Rich console and typer
│
├── prepare/
│   ├── __init__.py               ← PREPARER_REGISTRY dict
│   ├── base_preparer.py          ← BasePreparer ABC, PrepIssue, PrepResult dataclasses
│   ├── glm_preparer.py           ← GLMPreparer (IN PROGRESS — see below)
│   ├── prepare_dwi.py            ← legacy DWI preparation
│   └── dwi_prepare_input.py
│
├── clusters/
│   ├── slurm.py
│   ├── sge.py
│   └── dask_scheduler.py
│
├── check/
│   ├── general_checks.py
│   └── check_dwi_pipelines.py
│
└── helper_function/
    ├── gen_subses.py
    ├── create_bids.py
    ├── copy_configs.py
    └── zip_example_config.py
```

---

## CLI subcommands (`lc`)

| Subcommand | Key flags | Purpose |
|---|---|---|
| `prepare` | `-lcc`, `-ssl`, `-cc` | Validate + create analysis dir + generate scripts |
| `run` | `-w`, `-R/--run_lc` | Submit scripts (dry-run without `--run_lc`) |
| `qc` | `-w` | Check outputs, write `failed_subseslist.tsv` |
| `copy_configs` | `-o` | Copy example configs to working dir |
| `gen_subses` | `-b`, `-n`, `-o` | Generate subseslist.tsv from directory |
| `create_bids` | `-cbc`, `-ssl` | Create fake BIDS structure for testing |
| `zip_configs` | — | Archive example configs (developer utility) |

---

## Dispatch logic in `cli.py`

```python
if container in PREPARER_REGISTRY:
    # New-style: BasePreparer subclass (glm, prf, ...)
    cls = PREPARER_REGISTRY[container]
    preparer = cls(config=lc_config, subseslist=subsesrows, output_root=output_root)
    preparer.run(dry_run=False)
else:
    # Legacy DWI path — do NOT modify
    do_prepare.main(parse_namespace, analysis_dir)
```

---

## Supported pipelines

### Container-based (DWI/structural) — legacy path
`anatrois`, `freesurferator`, `rtppreproc`, `rtp-pipeline`, `rtp2-preproc`, `rtp2-pipeline`

### Analysis-based (fMRI) — PREPARER_REGISTRY
`glm`, `prf`

---

## Supported hosts

| Host key | Scheduler | Notes |
|---|---|---|
| `DIPC` | SLURM | `sbatch`, `/scratch` filesystem |
| `BCBL` | SGE | `qsub`, `long.q` queue |
| `local` | bash | serial or Dask parallel |

---

## BasePreparer class hierarchy

```
BasePreparer  (ABC — base_preparer.py)
└── GLMPreparer  (glm_preparer.py)
```

### Key dataclasses

```python
@dataclass
class PrepIssue:
    sub: str; ses: str; category: str
    severity: str   # "blocking" | "warn" | "auto_fix"
    message: str;   fix_fn: Callable | None = None

@dataclass
class PrepResult:
    sub: str; ses: str
    status: str     # "ready" | "fixed" | "warn" | "blocked"
    issues: list[PrepIssue]
```

### Abstract methods every subclass must implement

```python
def check_requirements(self, sub: str, ses: str) -> list[PrepIssue]: ...
def generate_run_script(self, sub: str, ses: str, analysis_dir: Path) -> Path: ...
```

### Adding a new analysis type (3 steps)
1. Create `launchcontainers/prepare/<type>_preparer.py` subclassing `BasePreparer`
2. Register it in `prepare/__init__.py`: `PREPARER_REGISTRY["type"] = TypePreparer`
3. Add a `<type>_specific` block to `lc_config.yaml`

---

## GLMPreparer — work in progress

**File**: `launchcontainers/prepare/glm_preparer.py`
**Status**: class skeleton exists; all methods raise `NotImplementedError`

### Still to implement
- `_check_fmriprep_bold(sub, ses)`
- `_check_fmriprep_confounds(sub, ses)`
- `_check_fmriprep_mask(sub, ses)`
- `_check_events_file(sub, ses)`
- `_check_bold_json(sub, ses)`
- `_slurm_script(sub, ses, analysis_dir)`
- `_sge_script(sub, ses, analysis_dir)`
- `_local_script(sub, ses, analysis_dir)`

### Config section (`glm_specific`)
```yaml
glm_specific:
  version: "1.0.0"
  fmriprep_dir: fmriprep-23.2.0
  fmriprep_analysis_name: main
  space: T1w
  tasks: [floc]
  n_runs: 1
  tr: 1.5
  hrf_model: spm
  smoothing_fwhm: 6
  confounds: [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z, framewise_displacement]
  python_path: python
  glm_script_path: /path/to/run_glm.py
```

---

## Configuration (`lc_config.yaml`) structure

```yaml
general:          # project-level: basedir, container, host, analysis_name, ...
container_specific:
  <pipeline>:     # DWI containers — version, precontainer_anat, rpe, ...
glm_specific:     # fMRI GLM — see above
host_options:
  DIPC:           # SLURM: cores, memory, partition, qos, walltime
  BCBL:           # SGE: cores, memory, queue, walltime
  local:          # njobs, memory_limit
```

---

## Sphinx documentation

- **Location**: `docs/source/`
- **Theme**: `pydata_sphinx_theme`
- **Build**: `cd docs && make html`
- **Pages**: `index.rst` (hidden toctrees), `Installation.rst`, `Concepts.rst`,
  `Tutorial.rst`, `CLI reference.rst`, `Configuration.rst`, `Extending.rst`,
  `Code structure.rst`, `Changelog.rst`, `api.rst`

### Known conf.py settings
- `pandas` removed from `intersphinx_mapping` (server returns 522 errors)
- All toctrees in `index.rst` use `:hidden:` to avoid rendering on the homepage
- `suppress_warnings = ["intersphinx.fetch_inventory"]` is set
- File names are **Title Case** (e.g. `Installation.rst`) — toctree refs must match exactly

---

## Key conventions

- DWI legacy code (`do_prepare.py`, `prepare_dwi.py`) is **never modified**
- All new analysis types go through `PREPARER_REGISTRY`
- `BasePreparer` owns all orchestration; subclasses implement only domain logic
- Dry-run support is required for all file operations
- Prefer functional over class-based code where possible
- Rich console output with color/progress bars via the `rich` library
- Three output file pattern for checker: brief CSV, detailed log, pivot matrix
