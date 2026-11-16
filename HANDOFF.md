# HANDOFF — open work for the next person or agent

Read this before starting. Update it when you finish or hand off a task, and
delete an entry once it is done. Repo conventions are in `Claude.md`.

---

## sub-02 ses-07…10 DWI heudiconv conversion (VOTCLOC) — 2026-11-04

**Goal:** convert only the DWI DICOMs of sub-02 ses-07, 08, 09, 10 to BIDS NIfTI
with heudiconv on IPS (SGE).

| | |
|---|---|
| Data | `/bcbl/home/public/Gari/VOTCLOC/main_exp/dicom/sub-02/ses-XX` (symlinks to the lab DICOMs) |
| Output | `/bcbl/home/public/Gari/VOTCLOC/main_exp/raw_nifti` |
| Criteria | `MR_pipelines/00_dicom_to_nifti/heuristic/heuristic_dwi_VOTCLOC.py`: `dMRI`/`1d5iso` series, no SBRef; PA = 6 files, AP = 105 files; magnitude + phase |
| Work list | `/bcbl/home/public/Gari/VOTCLOC/main_exp/code/subseslist_heudiconv_sub02_dwi.txt` |

### Done
- Wrote the DWI-only heuristic. In the lab folders, all four sessions have
  exactly one PA (6 files) and one AP (105 files) series, so it matches them.
- `run_ips/qsub_heurdiconv.sh`: added optional `-H <heuristic>` and
  `-o <output dir under basedir>` flags (the defaults are unchanged), and
  `script_dir` now uses the `/bcbl/home/...` path.
- `src_heudiconv_step{1,2}.sh`: `--bind /export` is added only when `/export`
  exists on the host.
- Created `dicom/sub-02/ses-09` and `ses-10` with
  `prepare/03_create_dicom_symlink.py`, pointing to
  `/bcbl/data/MRI/VOTCLOC_22324/DATA/images/sub-02_ses-{09,10}`.

### Left to do
1. **Run the conversion** (it has not been run yet):
   ```bash
   bash MR_pipelines/00_dicom_to_nifti/run_ips/qsub_heurdiconv.sh -t step2 \
     -f /bcbl/home/public/Gari/VOTCLOC/main_exp/code/subseslist_heudiconv_sub02_dwi.txt \
     -H heuristic_dwi_VOTCLOC.py -o raw_nifti
   ```
   Then check `raw_nifti/sub-02/ses-XX/dwi/` and the logs in
   `raw_nifti/log_heudiconv/subseslist_heudiconv_sub02_dwi_<date>/step2/`.
2. **ses-07 and ses-08 links still point to `/export/home/tlei/lab/...`**, which
   works only on hosts that have `/export` (IPS is fine). Optionally repoint them
   to `/bcbl/data/MRI/...` with `03_create_dicom_symlink.py --force`.
3. **The ses-08 source folder needs confirming.** The lab has both
   `sub-02_ses-08` and `sub-02_ses-08-mar-24`, and both contain DWI. The link
   uses `-mar-24`. For ses-07, `sub-02_ses-07re` is correct, because
   `sub-02_ses-07` has no DWI.

---

## Host-path check over SSH — 2026-11-04

`Claude.md` → "Host paths" says that `/export/home/tlei` (ips, lmc02,
cajal03/04) and `/home/tlei` (brodmann) are the same home, and that
`/export/home/tlei/lab/MRI` is the same folder as `/bcbl/data/MRI`. This has
**not been verified** on the cluster yet.

- From brodmann, key-based SSH to cajal03, cajal04 and lmc02 is refused, and
  `ips` does not answer on port 22 (`ips-0-3` does).
- **To do:** run `ssh-copy-id -i ~/.ssh/id_ed25519.pub tlei@{cajal03,lmc02,ips-0-3}`
  and add `Host cajal03` / `Host ips` (→ `ips-0-3`) entries to `~/.ssh/config`.
  Then run on one host:
  ```bash
  ls -d /bcbl/data/MRI/VOTCLOC_22324 && readlink -f /export/home/tlei/lab/MRI
  ```
  and mark the Claude.md section as verified.

---

## Repo cleanup and conventions review — 2026-11-04

A status review and a comparison of this repo against `soft/CLAUDE.md`. No code
was changed. **The maintainer has not yet decided which of these fixes to do,**
so confirm before starting items 2 and 3.

### 1. Uncommitted uv migration — the committed repo is inconsistent
`Claude.md` at HEAD says uv + `uv.lock` + hatchling, but `pyproject.toml` at HEAD
is still Poetry and `uv.lock` is untracked. **`uv sync` from a fresh clone will
not work as documented until this is committed.** Only the `poetry.lock` deletion
is staged. The rest is in the working tree of the maintainer's checkout:
- Migration: `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`,
  `environment.yml`, `requirements.txt`, `.github/workflows/main.yml`,
  `README.md`, `docs/source/{index,installation}.rst`,
  `launchcontainers/__init__.py`.
- GLM scripts: edits in `launchcontainers/tests/glm_strategy/`, and two contrast
  YAMLs moved there from `tests/run_glm/`.
- New power analysis: `launchcontainers/tests/power_analysis_ohbm/` (untracked).

Suggested: commit the migration and the GLM/power-analysis work separately. Check
each diff first; other sessions may be editing this tree.

### 2. `Claude.md` describes code that no longer exists
Checked against the code on 2026-11-04. Rewrite these sections from the code:

| `Claude.md` says | The code has |
|---|---|
| `cli.py` is argparse with 7 subcommands | typer, with only `prepare`, `run`, `qc` |
| `base_preparer.py`, `glm_preparer.py`, `PREPARER_REGISTRY`, `PrepIssue`/`PrepResult` | `base_prepare.py`, `glm_prepare.py`, `prf_prepare.py`, `dwi_prepare.py`; no registry |
| GLM preparer: all methods raise `NotImplementedError` | `glm_prepare.py` is implemented (926 lines, no `NotImplementedError`) |
| `config_logger.py` | `log_setup.py` |
| Docs file names are Title Case | lowercase (`installation.rst`) |
| (not mentioned) | `gen_jobscript/`, `quality_control/` |

Until this is done, trust the code over those sections.

### 3. Conventions
- **Shared rules are not in this repo.** `Claude.md` points to `../CLAUDE.md`, but
  `soft/` is not a git repository, so anyone working from a fresh clone does not
  have that file. Decide whether to copy the rules that matter into this repo.
- **File name:** Claude Code looks for `CLAUDE.md`, so `Claude.md` may not load
  automatically on a case-sensitive filesystem (not tested). Consider renaming it.
- **Proposed fixes to `soft/CLAUDE.md` and `soft/AGENTS.md`** (outside this repo):
  - "Launchers call `uv run`" is wrong for cluster jobs: uv cannot lock files on
    cajal03. Use `uv sync` on the submit host, then `.venv/bin/python` in jobs.
  - The bash launcher skeleton contradicts §1 and §1b: it splits on tabs, does not
    skip the header or `RUN=False` rows, calls `python`, and loops in bash.
  - `mmr-<name>` entry-point naming belongs to multi_modal_mri_registration.
- **typer + rich only:** about 1,500 `print(` calls and 8 files using argparse
  remain. Suggested: apply the rule to new or touched code, not a mass rewrite.
- **`CITATION.cff` is missing.** The template lists one author; this repo has
  several (see `pyproject.toml`).
- **Pre-commit hook:** `.git/hooks/pre-commit` points to a Python under `/export`,
  which does not exist on brodmann, and `pre-commit` is not on PATH there.

### 4. Housekeeping (low priority)
- `launchcontainers/tests/` is mostly analysis scripts and session lists, not tests.
- 12 old stashes, and local branches `WC_matching_mat` and `data_cleaning` whose
  remotes are gone. Ask before deleting; they may hold unmerged work.
