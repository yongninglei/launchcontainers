# MRI-hub architecture

MRI-hub is a set of repos that work as one system, with one division of work:

- **launchcontainers** prepares inputs, records provenance, launches, and
  checks that each launch worked. It holds the rules every analysis shares —
  work list, config, provenance, run logs, launch report — in
  `launchcontainers.core`, and the rules for running each analysis in
  `launchcontainers.tools`.
- **Analyses** (DWI containers, pyGLM, autoROI, any future tool) only compute
  one subject × session and assume their inputs are ready. pyGLM and autoROI
  are standalone programs: they do not import launchcontainers, and
  launchcontainers does not import them. Each can also be run by hand.
- **Image QC** (render slices once, rate them in a browser page) is its own
  module, **QC-registration**, also standalone.

```
   ┌──────────────────────────────────────────────────────────────┐
   │ launchcontainers  (`lc`, `lc-session`)                       │
   │  core: subseslist · config · runlog · provenance · Tool      │
   │        interface · session wrapper · launch-status report    │
   │        (needs only pyyaml, rich, typer)                      │
   │  lc prepare / lc run / lc qc · backends local · sge · slurm  │
   │  tools/dwi.py       DWI containers (apptainer images)        │
   │  tools/external.py  pyGLM, autoROI-*: their programs         │
   └───────────────┬──────────────────────────────────────────────┘
                   │ runs  <exe> --config C --sub S --ses E   (and --plan)
   ┌───────────────┴──────────────┐     ┌─────────────────────────────┐
   │ pyGLM, autoROI, …            │     │ QC-registration             │
   │  standalone programs, own    │     │  image QC: slices, rating   │
   │  environments, no lc imports │     │  page, ratings TSV; DWI QC  │
   └──────────────────────────────┘     └─────────────────────────────┘
```

The contract between lc and an analysis program is its command line and the
`--plan` JSON (`docs/plan_protocol.md`): analysis dirs, session outputs, input
files, problems — computed from the config without running anything.

Until 2026-11-16 the core was a separate package, `analysis-core`
(MRI-hub/analysis-core, last version 0.1.4); it moved into launchcontainers
0.5.0 unchanged except for names: `analysis_core.*` → `launchcontainers.core.*`,
`AnalysisCoreError` → `LCError`, `analysis-session` → `lc-session`, entry-point
group `analysis_core.tools` → `launchcontainers.tools`. The run-log protocol
(`ANALYSIS_RUN_ID` / `ANALYSIS_RUN_DIR`) and all file formats are unchanged.

The heavy dependencies of lc's own code are extras (`dwi`, `fmri`, `checker`,
`heudiconv`, `scripts`); pyGLM / autoROI are not dependencies of lc at all.

---

## 1. One workflow: prepare → run → qc

| Step | Command | What it does | Writes |
|---|---|---|---|
| prepare | `lc prepare -c lc_config.yaml -f subseslist.txt` | load + validate config, create/guard the analysis dir(s), per session: `tool.prepare()` (stage inputs, check requirements — never compute) | analysis dir skeleton, `lc_config.yaml`, `dataset_description.json`, `logs/prepare_<stamp>/`, `sessions.tsv` rows with status `prepared` / `blocked` |
| run | `lc run -w <analysis_dir> [-R]` | per session: `tool.command()` wrapped by `lc-session`, then submit with the backend from `host_options.<host>.manager` | `logs/run_<stamp>/` (job script, commands, scheduler .out/.err), per-session run record + `sessions.tsv` row with the real exit code |
| qc | `lc qc -w <analysis_dir>` | launch checks per session: exit status, run log, `tool.qc()` (outputs exist, not empty) | `qc/index.html` (sub × ses table), `qc/sub-XX_ses-YY.html`, `qc/qc.tsv`, `qc/failed_subseslist.txt` |

Each step works on **one sub/ses per call** in the tool; the loop over the
work list lives in launchcontainers only. Without `-R`, `lc run` is a dry run
and prints the job script and the first command.

## 2. The Tool interface (`launchcontainers.core.tool`)

```python
class Tool(Protocol):
    name: str                      # key under container_specific in lc_config.yaml
    version: str

    def load(self, cfg: LCConfig) -> ToolConfig:
        """Validate container_specific.<name>; unknown or missing keys are errors."""

    def analysis_dirs(self, tc: ToolConfig) -> list[Path]:
        """Where this analysis writes (pyGLM: one per strategy)."""

    def prepare(self, tc: ToolConfig, sub: str, ses: str) -> SessionResult:
        """Stage inputs for one session. Idempotent; raises nothing — returns status."""

    def command(self, tc: ToolConfig, sub: str, ses: str) -> list[str]:
        """The one command that processes one session (apptainer run …, or the
        tool's own single-session CLI)."""

    def qc(self, tc: ToolConfig, sub: str, ses: str) -> QCResult:
        """Checks + figures for one session's outputs."""
```

Optional sixth method, `prepare_analysis(tc, pairs)`: analysis-level staging
called once after the analysis dir is guarded and before any per-session
`prepare` (e.g. write a container's `config.json` once). It may return the
sessions reordered (anatrois/freesurferator stage source sessions before the
retest sessions that link to them). The launcher calls it through
`launchcontainers.core.tool.prepare_analysis(tool, tc, pairs)`.

A new software joins MRI-hub by implementing these five methods and adding
one line to its `pyproject.toml`:

```toml
[project.entry-points."launchcontainers.tools"]
pyGLM = "pyglm.lc_tool:TOOL"
```

`ToolConfig` holds the parsed `general` section plus the tool's own validated
parameters, and `rows`: the launcher's work list, so `tc.row(sub, ses)` gives a
tool its extra columns (e.g. autoROI's pseudo-session `sources`).
`SessionResult` and `QCResult` are small dataclasses in core
(sub, ses, status, messages, outputs/figures).

## 3. Config — one file, one shape

```yaml
general:                 # where the data are, which tool, which analysis
  basedir: /bcbl/home/public/Gari/VOTCLOC/main_exp
  bidsdir_name: BIDS
  tool: pyGLM            # key into container_specific (`container:` accepted as alias)
  analysis_name: main
  host: BCBL             # key into host_options
  force: false
container_specific:
  pyGLM: { … }           # validated by the tool's load()
host_options:
  BCBL:  { manager: sge,   queue: long.q, cores: 4, memory: 16G, walltime: "24:00:00", mounts: [/bcbl] }
  DIPC:  { manager: slurm, partition: general, … }
  local: { manager: local, n_workers: 2 }
```

- The backend is chosen by `manager`, never by the host's name.
- **Config guard** (core): an analysis dir keeps the `lc_config.yaml` it was
  made with. Re-running with a config whose *computational* keys differ is
  refused unless `general.force: true`; then the old copy moves to
  `logs/history/`. Runtime-only keys (`host`, `force`, `host_options`) never
  trigger the guard. Nothing is ever deleted.

## 4. Work list — `subseslist.txt`

`sub,ses,RUN` with a header; optional extra columns (e.g. autoROI's
`sources`). Blank lines and `#` comments are ignored, `sub-`/`ses-` prefixes
stripped, numeric labels zero-padded to two digits. A malformed row, a RUN value
other than True/False, or a duplicate sub/ses is a hard error. One reader, in
core: `read_subseslist(path, extra=()) -> list[Row]`. `write_subseslist` seeds a
list (`--write-list`).

## 5. Provenance — what every analysis dir contains

```
<analysis dir>/
├── dataset_description.json   BIDS derivative: Description (general.description: why
│                              the analysis exists, + tool and version), GeneratedBy (tool, version, commit,
│                              dirty), SourceDatasets / DatasetLinks (bids: URIs)
├── lc_config.yaml             the config it was made with (guarded, §3)
├── subseslist.txt             union of every session ever run here
├── .bidsignore
├── logs/
│   ├── runs.tsv               one row per prepare/run/qc batch
│   ├── sessions.tsv           one row per session per batch; last row = current state
│   ├── <step>_<stamp>/        run.json (command, host, user, versions, checksums,
│   │                          per-session status) + per-session logs + scheduler .out/.err
│   └── history/               earlier configs / descriptions
├── qc/                        index.html, per-session pages, qc.tsv
└── sub-XX/ses-YY/…            outputs; sidecars with Sources (bids: URIs + sha256)
```

- **Why and from what.** `general.description` (free text, not computational, so
  rewording it never trips the guard) becomes `Description`; the datasets a tool
  reads besides raw BIDS (its `--plan` `source_datasets`, e.g. the upstream analysis
  a registration-only rerun reads) become `SourceDatasets` / `DatasetLinks`. A
  changed Description or source list archives the old file to `logs/history/`.
- **No Markdown or summary pages in an analysis dir.** Everything a reader needs is
  in the files above, the tool's `analysis_parameters.yaml` + sidecars, `qc/`
  (launch QC) and `qc/qc_rater-<name>.tsv` (visual QC verdicts with a comment: what
  failed and why, QC-registration's format).
- `runlog.start_run / finish_run` (moved from pyGLM, now the only copy).
- **Session wrapper** `lc-session --analysis-dir D --tool T --sub S --ses E -- <cmd…>`:
  every job line `lc run` submits goes through it, so each array task records
  its own start, end, exit code and log — the scheduler no longer needs to.
  When a tool's own CLI runs inside the wrapper, core's `start_run` sees
  `ANALYSIS_RUN_ID` and records into the wrapper's run instead of opening a
  second one.
- For container tools, prepare also records every staged input: link path,
  target, and sha256 of the target (`inputs.tsv` per session).
- Raw BIDS is read-only: anything generated (mock bval/bvec, merged shells)
  goes inside the analysis dir.

## 6. QC — two jobs, two places

1. **Launch QC — launchcontainers (`lc qc`).** Did each session's launch
   work? Per session: the last run's exit status (`sessions.tsv`), the
   lc-session log scanned for tracebacks / error lines, and `tool.qc()` —
   the expected outputs exist and are not empty (DWI pipelines also: the
   `RTP_log` exit line). Written to `qc/index.html` (sub × ses matrix),
   `qc/sub-XX_ses-YY.html`, `qc/qc.tsv`, and `qc/failed_subseslist.txt` for
   `lc run -ssl`. lc never looks at the images.
2. **Image QC — QC-registration.** Renders PNG slices once per sub/ses/item
   (fingerprinted, `qc/<item>/sub-XX/ses-YY[/<run>]/` with `session.json`),
   serves a browser page where raters answer True/False + comment, and
   writes `qc/qc_rater-<name>.tsv`. Registration QC (`qc-registration-render`),
   DWI QC (`qc-registration-dwi`: segmentation on T1, mean b0 on T1), SWI
   montages; `qc-registration-viewer` serves the page.

## 7. Repos and what moves where

| Repo | Keeps | Gets | Gives away |
|---|---|---|---|
| launchcontainers | CLI, backends, `core` (was analysis-core: runlog, subseslist, config + guard, provenance, Tool interface, session wrapper, launch report), DWI container tools, the rules for running pyGLM / autoROI (`tools/external.py`) | — | image QC (→ QC-registration), its GLM copies (→ pyGLM) |
| pyGLM | the GLM: single-session CLI + `--plan` | — | batch runner, run logs, provenance, `lc_tool.py` (→ launchcontainers) |
| autoROI | scaling, clustering: single-session CLIs + `--plan` | — | batch runners, run logs, analysis-dir provenance, `lc_tool.py` (→ launchcontainers) |
| QC-registration | registration QC | slice renderer, rating page, ratings, DWI QC | — |

## 8. Migration order

1. **analysis-core** with tests on a fake dataset (pytest `tmp_path`).
2. **pyGLM, autoROI** import core; delete their copies; their test suites must still pass.
3. **launchcontainers**: src layout, `lc` on core; containers ported to Tool
   classes one by one, each checked against a golden snapshot (staged symlink
   map + generated command) taken from the current code on the fake dataset.
4. Backends + session wrapper; `lc run` submits pyGLM/autoROI too.
5. `lc qc` + HTML; `analysis_checker` specs become `tool.qc()` implementations.
6. Move study-specific scripts out; the maintainer deletes what is listed.
