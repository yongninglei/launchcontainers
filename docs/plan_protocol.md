# The `--plan` protocol between launchcontainers and an analysis

launchcontainers prepares inputs, records provenance, launches, and checks
that the launch worked. The analyses (pyGLM, autoROI, …) are standalone
programs that assume their inputs are ready; they do not import
launchcontainers and launchcontainers does not import them. The only contract
is the analysis's single-session command line:

```
<exe> --config <lc_config.yaml> --sub <S> --ses <E> [tool flags]          # run one session
<exe> --config <lc_config.yaml> [--sub <S> --ses <E>] [tool flags] --plan  # describe, compute nothing
```

`--plan` reads the config (and, with `--sub/--ses`, looks for that session's
input files), prints **one JSON object on stdout** and exits 0 — also when
inputs are missing (they go into `problems`). Exit code 2 only when the config
itself is invalid (message on stderr). It never writes a file.

```json
{
  "tool": "pyGLM",
  "version": "0.1.0",
  "analysis_dirs": ["/abs/.../l1_surface/analysis-main_base", "..."],
  "sub": "01",
  "ses": "02",
  "session_outputs": ["/abs/.../analysis-main_base/sub-01/ses-02", "..."],
  "inputs": ["/abs/path/of/every/file/the/run/will/read", "..."],
  "problems": ["human-readable reason this session cannot run", "..."],
  "notes": ["informational, e.g. '4 run(s) 01,02,03,04'"]
}
```

- `analysis_dirs`: always present; where this config writes (pyGLM: one per strategy).
- `source_datasets` (optional, since 0.5.1): `{"<link>": "/abs/root"}`, the datasets the
  analysis reads besides raw BIDS: an upstream analysis, fMRIPrep, … e.g.
  `{"upstream": "/abs/.../swi_preproc/analysis-v1", "fmriprep": "/abs/.../fmriprep-25.1.4"}`.
  lc prepare writes them next to `raw` into `dataset_description.json`
  (`SourceDatasets` `bids:<link>:` + relative `DatasetLinks`). Same answer with or
  without `--sub/--ses`; lc reads it from the analysis-level call.
- `sub`, `ses`, `session_outputs`, `inputs`, `problems`, `notes`: present when
  `--sub/--ses` are given; otherwise `sub`/`ses` are null and the lists empty.
- All paths absolute. Labels bare (`01`, not `sub-01`), zero-padded like the run itself.
- `problems` empty ⇔ the session is ready to run.

launchcontainers uses it as follows (`src/launchcontainers/tools/external.py`):

| lc step | uses |
|---|---|
| `load` / `analysis_dirs` | `--plan` without sub/ses → `analysis_dirs`, `source_datasets` |
| `prepare` | `--plan --sub --ses` → `problems` (blocked if any), `inputs` recorded (path, size, mtime) in `<analysis_dir>/logs/inputs/sub-S_ses-E.tsv` |
| `run` | the run command, wrapped by `lc-session` |
| `qc` (launch only) | each of `session_outputs` exists and is not empty; exit code from `sessions.tsv`; error lines in the session log |
