# Rerun logic — what, why, and where in the code

## Background: what is a rerun?

During fMRI scanning a run can fail mid-way (scanner artefact, excessive head motion, subject discomfort, button-box drop-out, etc.).
When this happens the session continues and the operator acquires an **extra run** to replace the failed one.
The Siemens protocol name for that extra acquisition encodes the relationship:

```
fLoc_run-11_rerun-04
```

This means **run-11 is the replacement** that was scanned to compensate for the aborted **run-04**.

After BIDS conversion, *both* runs land in the `func/` directory.  
The dataset therefore contains:

| run | what happened |
|-----|---------------|
| run-04 | aborted — incomplete data, must be **excluded** from GLM |
| run-11 | redo — complete data, must be **included** instead |

Without explicit tracking, both runs would enter the GLM, corrupting the design matrix and the events alignment.

---

## The rerun_check.tsv

`<bids_dir>/sourcedata/qc/rerun_check.tsv` is the canonical map of every known rerun event.  
It is **tab-separated** with a mandatory header row.

| column | meaning |
|--------|---------|
| `sub` | subject ID, zero-padded to 2 digits |
| `ses` | session ID, zero-padded to 2 digits |
| `task` | task label (e.g. `fLoc`, `IRAKEINU`) |
| `extra_run` | the redo run number (zero-padded) |
| `compensates_run` | the aborted run number it replaces (zero-padded) |
| `protocol_name` | auto-generated label `{task}_run-{extra}_rerun-{comp}` |
| `found_in_bids` | `True` / `False` — does `extra_run` actually exist in BIDS? |
| `is_within_range` | `True` / `False` — is `compensates_run` within the standard run count? |
| `status` | free-text note: `OK`, `MISMATCH`, `NO_FUNC_DIR`, or any memo |

**The `found_in_bids` flag is critical.**  
If the redo run was never transferred to BIDS there is no replacement, so the original must stay.  
All GLM scripts guard on this before excluding anything.

### Chained reruns

A redo run can itself be aborted and replaced again:

```
run-11 → compensates run-10 → compensates run-04
```

`02_prepare_floc_events_tsv.py` handles this via `_resolve_original_run()`, which follows the
compensation chain until it reaches a standard (non-extra) run, returning all intermediate hops.

---

## Generator scripts (in `prepare_glm/`)

| script | when to use |
|--------|-------------|
| `01_generate_rerun_check_from_labnote.py` | authoritative — parses the lab-note Excel; catches within-range reruns the BIDS structure alone cannot detect |
| `generate_rerun_check_from_bids.py` | use when the lab-note Excel is not accessible (e.g. running on DIPC without BCBL mount) |
| `03_add_rerun_check_interactive.py` | use when you remember a rerun from scan day but have no lab note; prompts field-by-field, appends to an existing file |

All three write the same 9-column TSV schema.

---

## How the code uses rerun_check.tsv

### 1. Events TSV preparation — `02_prepare_floc_events_tsv.py`

```
_load_rerun_map(rerun_tsv)
  → RerunMap: (sub, ses, task, extra_run) → compensates_run
```

When building the events TSV for a run, the script checks whether the current run is an `extra_run`.
If it is, it looks up the `compensates_run` and **copies or symlinks the events file** from the original run
to the redo run — because the task design is identical and no separate events file was saved for the redo.

Chained reruns are resolved with `_resolve_original_run()` before the lookup.

### 2. GLM run exclusion — `run_glm_VOTCLOC*.py`, `run_glm_IRAKEINU.py`

```
_load_rerun_exclusions(rerun_tsv)
  → dict[(sub, ses, task)] → {compensates_run, ...}
```

Only rows where `found_in_bids == "True"` are loaded.

`generate_run_groups()` receives the resulting `excl_runs` set and removes those run labels from the
run list **before** the design matrix is built.  
The redo run (`extra_run`) stays in; the original aborted run (`compensates_run`) is dropped.

### 3. Batch GLM launcher — `run_allses_glm.py`

Delegates to whichever `run_glm_*.py` module is active; calls its `_load_rerun_exclusions` the same way.

---

## Typical workflow

```
                  scan day
                     │
         operator notes rerun in lab-note Excel
                     │
         ┌───────────▼────────────────────────────────────┐
         │  01_generate_rerun_check_from_labnote.py       │  ← preferred
         │  (or 01b_add_rerun_check_interactive.py        │  ← if no lab note
         │   or 01c_generate_rerun_check_from_bids.py)        │  ← on DIPC
         └───────────┬────────────────────────────────────┘
                     │ writes / appends
                     ▼
          sourcedata/qc/rerun_check.tsv
                     │
         ┌───────────┴──────────────────────┐
         │                                  │
         ▼                                  ▼
02_prepare_floc_events_tsv.py      run_glm_*.py --rerun-map
 events TSV symlink for redo         exclude aborted run from GLM
```
