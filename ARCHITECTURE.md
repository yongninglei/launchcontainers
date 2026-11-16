# MRI-hub architecture

MRI-hub is a set of repos that work as one system. **launchcontainers** is the
only launcher, and it also holds the rules that every analysis shares — the
work list, the config, provenance, run logs and the QC report — in its
`launchcontainers.core` subpackage, so they exist exactly once. Every analysis
software (containers, pyGLM, autoROI, QC-registration, any future tool) plugs
into it through one interface, and can still be run on its own.

```
   ┌──────────────────────────────────────────────────────────────┐
   │ launchcontainers  (`lc`, `lc-session`)                       │
   │                                                              │
   │  launchcontainers.core   (needs only pyyaml, rich, typer)    │
   │   subseslist · config · runlog · provenance · Tool interface │
   │   session wrapper · QC report · slices · rating page         │
   │                                                              │
   │  lc prepare / lc run / lc qc · backends local · sge · slurm  │
   │  built-in tools: anatrois, freesurferator, rtp(2)-preproc,   │
   │  rtp(2)-pipeline        heavy deps = extras: [dwi] [render]… │
   └──────────────────────────────▲───────────────────────────────┘
                     depends on   │   (core only; no heavy extras)
   ┌──────────────────────────────┴───────────────────────────────┐
   │ tools: pyGLM, autoROI, QC-registration, …                    │
   │  each exposes Tool objects via the entry-point group         │
   │  "launchcontainers.tools" and keeps its own standalone CLI   │
   │  (pyglm, autoroi-scaling, qc-registration-render, …)         │
   └──────────────────────────────────────────────────────────────┘
        lc finds tools at run time through entry points — it never
        imports pyGLM/autoROI directly.
```

Until 2026-11-16 the core was a separate package, `analysis-core`
(MRI-hub/analysis-core, last version 0.1.4); it moved into launchcontainers
0.5.0 unchanged except for names: `analysis_core.*` → `launchcontainers.core.*`,
`AnalysisCoreError` → `LCError`, `analysis-session` → `lc-session`, entry-point
group `analysis_core.tools` → `launchcontainers.tools`. The run-log protocol
(`ANALYSIS_RUN_ID` / `ANALYSIS_RUN_DIR`) and all file formats are unchanged.

Tools depend on launchcontainers (core install: pyyaml, rich, typer); the
heavy dependencies of lc's own tools are extras (`dwi`, `render`, `fmri`,
`checker`, `heudiconv`, `scripts`, `glm`), so a tool never pulls them in.
The one loop is optional: lc's `glm` extra installs pyGLM, which depends on lc.

---

## 1. One workflow: prepare → run → qc

| Step | Command | What it does | Writes |
|---|---|---|---|
| prepare | `lc prepare -c lc_config.yaml -f subseslist.txt` | load + validate config, create/guard the analysis dir(s), per session: `tool.prepare()` (stage inputs, check requirements — never compute) | analysis dir skeleton, `lc_config.yaml`, `dataset_description.json`, `logs/prepare_<stamp>/`, `sessions.tsv` rows with status `prepared` / `blocked` |
| run | `lc run -w <analysis_dir> [-R]` | per session: `tool.command()` wrapped by `lc-session`, then submit with the backend from `host_options.<host>.manager` | `logs/run_<stamp>/` (job script, commands, scheduler .out/.err), per-session run record + `sessions.tsv` row with the real exit code |
| qc | `lc qc -w <analysis_dir>` | per session: `tool.qc()`; render the HTML report | `qc/index.html` (sub × ses table), `qc/sub-XX_ses-YY.html`, `qc/qc.tsv` |

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
├── dataset_description.json   BIDS derivative: GeneratedBy (tool, version, commit,
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

## 6. QC — HTML everywhere, two layers

The model is the one QC-registration already uses (github.com/MRI-hub/QC-registration):
render images once per session, then a browser page where a person rates them.

1. **Automatic checks.** `tool.qc()` returns checks (name, status
   ok/warn/fail, message): expected outputs present, logs without errors,
   numbers in range. Core writes them to `qc/qc.tsv` and a static
   `qc/index.html` (sub × ses status matrix, per-session pages).
2. **Visual rating.** `tool.qc()` also returns figures: PNGs rendered once
   per sub/ses/item (item = run, map, tract…), stored under
   `qc/<item>/sub-XX/ses-YY/` with a `session.json` fingerprint of the
   inputs, so nothing is re-rendered without a reason. One shared viewer —
   QC-registration's stdlib HTTP server + static page (web-MRI look), moved
   into core — serves any analysis's `qc/` folder. Raters answer True/False +
   comment per item; answers go to `qc/qc_rater-<name>.tsv`
   (`item sub ses run ok comment rater time`, one file per rater, written
   atomically).

Implemented (0.1.3): `launchcontainers.core.slices.render_item(analysis_dir, item,
sub, ses, base=…, layers=[Layer(key, label, kind, source)…])` — kinds `map`,
`labels`, `outline`, `surface`; one PNG per view × slice × layer plus
`preview.png`; `session.json` with the fingerprint. `launchcontainers.core.viewer`
(`lc qc --serve -w D`) serves every rendered item of `D/qc/`; layers and
the question come from each item's `session.json`. Rendering happens in
`tool.qc()`; a viewer given a `render` hook and the work list (`expected`)
renders a missing / stale item when it is first opened. QC-registration 0.3.0
is a tool on this renderer and viewer (`qc-registration`, `lc_tool.py`).

`lc qc -w <analysis_dir>` runs layer 1 and pre-renders layer 2;
`lc qc -w <analysis_dir> --serve` opens the viewer. QC-registration itself
becomes a tool (its `render_session` is its `qc()`), so registration QC runs
through the same commands.

## 7. Repos and what moves where

| Repo | Keeps | Gets | Gives away |
|---|---|---|---|
| launchcontainers.core (was analysis-core) | — | runlog (master, from pyGLM), subseslist + labels, config loader + guard, provenance helpers, atomic write / sha256, session logging, Tool interface + discovery, session wrapper, QC HTML renderer | — |
| launchcontainers | CLI, backends, built-in container tools | Tool classes for each container (one file each, replacing the if/elif chains) | its subseslist parser, logging setup, study-specific scripts (→ VOTCLOC) |
| pyGLM | GLM | `lc_tool.py` | `runlog.py`, subseslist code (→ core) |
| autoROI | scaling, clustering | `lc_tool.py` | `runlog.py`, subseslist + generic provenance code (→ core) |

## 8. Migration order

1. **analysis-core** with tests on a fake dataset (pytest `tmp_path`).
2. **pyGLM, autoROI** import core; delete their copies; their test suites must still pass.
3. **launchcontainers**: src layout, `lc` on core; containers ported to Tool
   classes one by one, each checked against a golden snapshot (staged symlink
   map + generated command) taken from the current code on the fake dataset.
4. Backends + session wrapper; `lc run` submits pyGLM/autoROI too.
5. `lc qc` + HTML; `analysis_checker` specs become `tool.qc()` implementations.
6. Move study-specific scripts out; the maintainer deletes what is listed.
