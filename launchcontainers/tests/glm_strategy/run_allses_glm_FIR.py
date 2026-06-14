# -----------------------------------------------------------------------------
# Copyright (c) Yongning Lei 2024-2025
# All rights reserved.
#
# This script is distributed under the Apache-2.0 license.
# You may use, distribute, and modify this code under the terms of the Apache-2.0 license.
# See the LICENSE file for details.
#
# THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT.
#
# Author: Yongning Lei
# Email: yl4874@nyu.edu
# GitHub: https://github.com/yongninglei
#
# run_allses_glm_FIR.py
# ----------------------
# Fit ONE FIR GLM per subject by concatenating data across ALL sessions and
# their runs (same concat logic as run_allses_glm.py).  Two design matrices
# are built and fit against the same data:
#
#   1. "cond"    — one FIR set (fir_delays) per named condition
#   2. "allstim" — one FIR set (fir_delays) for ALL stimulus events pooled
#                   into a single "AllStim" condition
#
# Outputs (per hemisphere) under:
#
#     <bids_dir>/derivatives/l1_surface_fir/analysis-<output_name>/sub-<sub>/allses/
#
#   - sub-XX_ses-allses_task-<task>_hemi-<H>_space-<space>_desc-FIR<cond>_delays.func.gii
#       10-frame GIFTI, one per named condition (frame = delay)
#   - sub-XX_ses-allses_task-<task>_hemi-<H>_space-<space>_desc-AllStimdelay<d>_statmap.func.gii
#       single-frame GIFTI per delay d = 0..n_delays-1  ("AllStim_delay_0" ... )
#   - sub-XX_ses-allses_task-<task>_hemi-<H>_space-<space>_desc-AllStimFIR0-18s_timeseries.func.gii
#       n_delays-frame GIFTI stack of the AllStim FIR curve ("4D" surface analog)
#   - betas TSVs + design matrix CSV/plots + confounds TSV
#
# Usage::
#
#     python run_allses_glm_FIR.py \
#         --base /bcbl/home/public/Gari/VOTCLOC/main_exp --sub 03 \
#         --sessions 01,02,03,04,05,06,07,08,09 \
#         --fp-ana-name 25.1.4_t2w_fmapsbref_newest --task fLoc \
#         --space fsnative --start-scans 6 \
#         --output-name FIR_v1
# -----------------------------------------------------------------------------

from __future__ import annotations

import csv
import gc
import os
import os.path as op
import time
from os import makedirs
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
import yaml
from bids import BIDSLayout
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.first_level.first_level import run_glm as nilearn_run_glm
from nilearn.surface import load_surf_data
from rich import box
from rich.console import Console
from rich.table import Table
from scipy import stats

# Import shared helpers from glm_surface_check_model_strategy.py by explicit file path.
import importlib.util as _ilu
_helpers_path = op.join(op.dirname(op.abspath(__file__)), "glm_surface_check_model_strategy.py")
_spec = _ilu.spec_from_file_location("_glm_helpers_mod", _helpers_path)
_glm_helpers = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_glm_helpers)

save_statmap_to_gifti       = _glm_helpers.save_statmap_to_gifti
save_timeseries_to_gifti    = _glm_helpers.save_timeseries_to_gifti
save_array_as_dataframe     = _glm_helpers.save_array_as_dataframe
replace_prefix_and_suffix   = _glm_helpers.replace_prefix_and_suffix
_load_rerun_exclusions      = _glm_helpers._load_rerun_exclusions
plot_design_matrix_to_file  = _glm_helpers.plot_design_matrix_to_file

console = Console()
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score each column, leaving constant columns (std == 0) untouched
    (avoids NaNs from FIR delay regressors that happen to be all-zero).
    """
    std = df.std(axis=0)
    mean = df.mean(axis=0)
    safe_std = std.replace(0, 1)
    out = (df - mean) / safe_std
    out.loc[:, std == 0] = df.loc[:, std == 0]
    return out


def _reconstruct_betas(n_reg, n_vtx, labels, estimates) -> np.ndarray:
    """Rebuild a (n_vertices, n_regressors) beta array from nilearn run_glm output."""
    out = np.zeros((n_vtx, n_reg), dtype=np.float32)
    for label, result in estimates.items():
        mask = labels == label
        out[mask, :] = result.theta.T.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Core: accumulate data across all sessions and runs, build FIR design matrices
# ---------------------------------------------------------------------------

def prepare_allses_fir_input(
    bids_dir: str,
    fmriprep_dir: str,
    fp_layout: BIDSLayout,
    layout: BIDSLayout,
    label_dir: str,
    subject: str,
    sessions: list[str],
    task: str,
    start_scans: int,
    space: str,
    slice_time_ref: float,
    use_smoothed: bool,
    sm: str,
    apply_label_as_mask: str,
    rerun_excl: dict,
    conditions: list[str],
    n_delays: int,
    hemi: str | None = None,
):
    """
    Load, z-score, and concatenate functional data across ALL sessions and
    their runs for *subject*, building TWO FIR design matrices:

      - design_matrix_cond_std    : FIR per named condition (fir_delays = range(n_delays))
      - design_matrix_allstim_std : FIR for a single pooled "AllStim" condition

    Returns
    -------
    tuple
        (conc_data_std, design_matrix_cond_std, design_matrix_allstim_std, nonan_confounds)
    """
    is_surface = space in ["fsnative", "fsaverage"]

    data_all: list[np.ndarray] = []
    frame_times_all: list[np.ndarray] = []
    events_all: list[pd.DataFrame] = []
    confounds_all: list[pd.DataFrame] = []

    total_scans_so_far = 0
    run_step_times: dict[str, dict[str, float]] = {}

    for ses in sessions:
        excl_runs = rerun_excl.get((subject, ses, task), set())
        raw_runs = sorted(set(layout.get_runs(subject=subject, session=ses, task=task)))

        if not raw_runs:
            console.print(
                f"  [yellow]WARNING[/yellow]: no runs for sub-{subject} ses-{ses} "
                f"task-{task} — skipping session"
            )
            continue

        run_list = [f"{r:02d}" for r in raw_runs]
        if excl_runs:
            before = set(run_list)
            run_list = [r for r in run_list if r not in excl_runs]
            excluded = sorted(before - set(run_list))
            if excluded:
                console.print(
                    f"  ses-{ses}: [yellow]excluded compensated runs:[/yellow] {excluded}"
                )

        console.print(f"\n  [bold]ses-{ses}[/bold]  runs: {run_list}")

        for run_num in run_list:
            run_label = f"ses-{ses}_run-{run_num}"
            run_step_times[run_label] = {}

            # ── Step 1: load functional data ─────────────────────────────────
            _t = time.time()
            query = {
                "subject": subject,
                "session": ses,
                "task": task,
                "run": run_num,
                "space": space,
                "suffix": "bold",
                "extension": ".func.gii" if is_surface else ".nii.gz",
            }
            if is_surface and hemi:
                query["hemi"] = hemi
            if use_smoothed:
                query["desc"] = f"smoothed{sm}"
            elif not is_surface:
                query["desc"] = "preproc"

            func_files = fp_layout.get(**query)
            if not func_files:
                console.print(
                    f"  [yellow]WARNING[/yellow]: no func file for {run_label} "
                    f"(query: {query}) — skipping"
                )
                continue

            func_file = func_files[0].path
            console.print(f"  Found: [dim]{func_file}[/dim]")

            if is_surface:
                data = load_surf_data(func_file)
                data_float = np.vstack(data[:, :]).astype(float)
            else:
                import nibabel as nib
                img = nib.load(func_file)
                arr = img.get_fdata()
                data_float = arr.reshape(-1, arr.shape[3]).astype(float)

            run_step_times[run_label]["load_func"] = time.time() - _t

            # ── Step 2: z-score + trim ────────────────────────────────────────
            _t = time.time()
            data_trimmed = data_float[:, start_scans:]
            data_std = stats.zscore(data_trimmed, axis=1).astype(np.float32)
            del data_float, data_trimmed
            n_features = data_std.shape[0]

            if apply_label_as_mask and is_surface:
                label_path = f"{label_dir}/{apply_label_as_mask}"
                surf_mask = load_surf_data(label_path)
                mask_arr = np.zeros((n_features, 1))
                mask_arr[surf_mask] = 1
                data_std = data_std * mask_arr

            n_scans = data_std.shape[1]
            data_all.append(data_std)
            run_step_times[run_label]["zscore_mask"] = time.time() - _t
            console.print(
                f"  Trimmed length: {n_scans}  "
                f"[dim](load+zscore: "
                f"{run_step_times[run_label]['load_func'] + run_step_times[run_label]['zscore_mask']:.1f} s)[/dim]"
            )

            # ── Step 3: TR from BOLD sidecar JSON ────────────────────────────
            _t = time.time()
            json_files = fp_layout.get(
                subject=subject, session=ses, task=task, run=run_num,
                suffix="bold", extension=".json",
            )
            if not json_files:
                console.print(
                    f"  [yellow]WARNING[/yellow]: no sidecar JSON for {run_label} — skipping"
                )
                data_all.pop()
                continue
            import json as _json
            with open(json_files[0].path) as _jf:
                t_r = float(_json.load(_jf)["RepetitionTime"])
            run_step_times[run_label]["read_tr"] = time.time() - _t

            # ── Step 4: events.tsv from BIDS layout ───────────────────────────
            _t = time.time()
            ev_files = layout.get(
                subject=subject, session=ses, task=task, run=run_num,
                suffix="events", extension=".tsv",
            )
            if not ev_files:
                console.print(
                    f"  [yellow]WARNING[/yellow]: no events.tsv for {run_label} — skipping"
                )
                data_all.pop()
                continue
            events = pd.read_csv(ev_files[0].path, sep="\t")

            # ── Step 5: confounds from fMRIprep layout ────────────────────────
            conf_files = fp_layout.get(
                subject=subject, session=ses, task=task, run=run_num,
                desc="confounds", suffix="timeseries", extension=".tsv",
            )
            if not conf_files:
                console.print(
                    f"  [yellow]WARNING[/yellow]: no confounds TSV for {run_label} — skipping"
                )
                data_all.pop()
                continue
            confounds = pd.read_csv(conf_files[0].path, sep="\t")
            run_step_times[run_label]["read_events_confounds"] = time.time() - _t

            # ── Step 6: confounds + frame_times ──────────────────────────────
            _t = time.time()

            events = events.copy()
            events.loc[:, "onset"] = events["onset"] + total_scans_so_far * t_r
            events_nobaseline = events[events["trial_type"] != "baseline"]
            events_all.append(events_nobaseline)

            frame_times = t_r * (
                (np.arange(n_scans) + slice_time_ref) + total_scans_so_far
            )
            frame_times_all.append(frame_times)

            motion_keys = ["framewise_displacement", "rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
            a_compcor_keys = [k for k in confounds.keys() if "a_comp_cor" in k]
            non_steady_keys = [k for k in confounds.keys() if "non_steady" in k]
            cosine_keys = [k for k in confounds.keys() if "cosine" in k]
            keep_keys = motion_keys + a_compcor_keys + cosine_keys + non_steady_keys
            confounds_keep = confounds[keep_keys].copy()
            confounds_keep["framewise_displacement"].iloc[0] = np.nanmean(
                confounds_keep["framewise_displacement"]
            )
            confounds_keep = confounds_keep.iloc[start_scans:]
            confounds_all.append(confounds_keep)

            total_scans_so_far += n_scans
            run_step_times[run_label]["confounds"] = time.time() - _t

            console.print(
                f"  Confounds length: {len(confounds_keep)}  "
                f"total_scans_so_far: {total_scans_so_far}  "
                f"[dim](confounds: {run_step_times[run_label]['confounds']:.1f} s)[/dim]"
            )

    if not data_all:
        raise RuntimeError(
            f"No data collected for sub-{subject} over sessions {sessions}."
        )

    # ── Per-run timing summary ────────────────────────────────────────────────
    step_cols = ["load_func", "zscore_mask", "read_tr", "read_events_confounds", "confounds"]
    tbl = Table(title="Per-run step timing (s)", box=box.SIMPLE_HEAD)
    tbl.add_column("ses_run")
    for s in step_cols:
        tbl.add_column(s, justify="right")
    tbl.add_column("total", justify="right")
    for rl, steps in run_step_times.items():
        vals = [steps.get(s, 0.0) for s in step_cols]
        tbl.add_row(rl, *[f"{v:.2f}" for v in vals], f"{sum(vals):.2f}")
    console.print(tbl)

    # ── Build concatenated inputs ─────────────────────────────────────────────
    _t = time.time()
    conc_data_std = np.concatenate(data_all, axis=1)
    del data_all
    concat_frame_times = np.concatenate(frame_times_all, axis=0)
    concat_events = pd.concat(events_all, axis=0)
    concat_events = concat_events.applymap(replace_prefix_and_suffix)
    concat_confounds = pd.concat(confounds_all, axis=0)

    nonan_confounds = concat_confounds.dropna(axis=1, how="any")
    console.print(f"\n  Confound columns after dropna: {list(nonan_confounds.columns)}")

    # Keep only the named conditions (drop anything unexpected so it doesn't
    # pollute the FIR design / AllStim pooling).
    before_n = len(concat_events)
    unexpected = sorted(set(concat_events["trial_type"]) - set(conditions))
    if unexpected:
        console.print(
            f"  [yellow]WARNING[/yellow]: dropping events with unexpected trial_type "
            f"not in --conditions: {unexpected}"
        )
    concat_events = concat_events[concat_events["trial_type"].isin(conditions)].copy()
    console.print(f"  Events: {before_n} → {len(concat_events)} after condition filter")

    fir_delays = list(range(n_delays))

    # ── Design matrix 1: FIR per named condition ───────────────────────────────
    design_matrix_cond = make_first_level_design_matrix(
        concat_frame_times,
        events=concat_events,
        hrf_model="fir",
        fir_delays=fir_delays,
        drift_model=None,
        add_regs=nonan_confounds,
    )
    design_matrix_cond_std = _safe_zscore(design_matrix_cond)
    design_matrix_cond_std["constant"] = np.ones(len(design_matrix_cond_std)).astype(int)

    # ── Design matrix 2: FIR for pooled "AllStim" condition ────────────────────
    concat_events_allstim = concat_events.copy()
    concat_events_allstim["trial_type"] = "AllStim"

    design_matrix_allstim = make_first_level_design_matrix(
        concat_frame_times,
        events=concat_events_allstim,
        hrf_model="fir",
        fir_delays=fir_delays,
        drift_model=None,
        add_regs=nonan_confounds,
    )
    design_matrix_allstim_std = _safe_zscore(design_matrix_allstim)
    design_matrix_allstim_std["constant"] = np.ones(len(design_matrix_allstim_std)).astype(int)

    console.print(
        f"  Design matrix (cond):    {conc_data_std.shape[1]} timepoints × "
        f"{design_matrix_cond_std.shape[1]} regressors\n"
        f"  Design matrix (allstim): {conc_data_std.shape[1]} timepoints × "
        f"{design_matrix_allstim_std.shape[1]} regressors  "
        f"[dim]({time.time() - _t:.2f} s)[/dim]"
    )

    return conc_data_std, design_matrix_cond_std, design_matrix_allstim_std, nonan_confounds


# ---------------------------------------------------------------------------
# GLM fit + output
# ---------------------------------------------------------------------------

def glm_fir_allses(
    conc_data_std: np.ndarray,
    design_matrix_cond_std: pd.DataFrame,
    design_matrix_allstim_std: pd.DataFrame,
    bids_dir: str,
    task: str,
    space: str,
    subject: str,
    output_name: str,
    conditions: list[str],
    n_delays: int,
    use_smoothed: bool = False,
    sm: str = "",
    hemi: str | None = None,
    n_glm_jobs: int = 1,
) -> dict[str, float]:
    """
    Fit both FIR design matrices and save beta maps under
    ``<bids_dir>/derivatives/l1_surface_fir/analysis-<output_name>/sub-<sub>/allses/``.
    """
    ses_label = "allses"

    outdir = op.join(
        bids_dir, "derivatives", "l1_surface_fir",
        f"analysis-{output_name}", f"sub-{subject}", ses_label,
    )
    makedirs(outdir, exist_ok=True)

    console.print(f"[bold]------- FIR GLM start  ({ses_label}, hemi-{hemi})[/bold]")

    desc_suffix = f"_desc-smoothed{sm}" if use_smoothed else ""

    # ── Save design matrices + plots + Y diagnostics ────────────────────────────
    for label, dm in (("cond", design_matrix_cond_std), ("allstim", design_matrix_allstim_std)):
        dm_csv = op.join(outdir, f"design_matrix_{task}_{label}{('_hemi-' + hemi) if hemi else ''}.csv")
        dm.to_csv(dm_csv)
        console.print(f"  [dim]Design matrix CSV → {op.basename(dm_csv)}[/dim]")
        plot_design_matrix_to_file(
            dm, outdir, subject, ses_label, f"{task}_{label}{('_hemi-' + hemi) if hemi else ''}"
        )

    Y = np.transpose(conc_data_std)
    del conc_data_std
    n_tp, n_vtx = Y.shape
    console.print(
        f"  [DIAG] Y (vertices × timepoints): shape={Y.shape}  "
        f"min={np.nanmin(Y):.4f}  max={np.nanmax(Y):.4f}  std={np.nanstd(Y):.4f}"
    )

    timing: dict[str, float] = {}

    # ── Fit 1: per-condition FIR ─────────────────────────────────────────────────
    _t = time.time()
    X_cond = np.asarray(design_matrix_cond_std)
    console.print(
        f"  [DIAG] Design matrix (cond): shape={X_cond.shape}  "
        f"rank={np.linalg.matrix_rank(X_cond)}  (expected full rank={X_cond.shape[1]})"
    )
    labels_c, estimates_c = nilearn_run_glm(Y, X_cond, n_jobs=n_glm_jobs)
    betas_cond = _reconstruct_betas(X_cond.shape[1], n_vtx, labels_c, estimates_c)
    timing["fit_cond"] = time.time() - _t
    console.print(f"  [dim]run_glm (cond): {timing['fit_cond']:.2f} s[/dim]")

    cond_cols = list(design_matrix_cond_std.columns)
    betas_cond_tsv = op.join(
        outdir,
        f"sub-{subject}_ses-{ses_label}_task-{task}"
        f"{('_hemi-' + hemi) if hemi else ''}_space-{space}{desc_suffix}_desc-FIRcond_betas.tsv.gz",
    )
    save_array_as_dataframe(betas_cond, betas_cond_tsv, col_labels=cond_cols)

    _t = time.time()
    for cond in conditions:
        delay_cols = [f"{cond}_delay_{d}" for d in range(n_delays)]
        missing = [c for c in delay_cols if c not in cond_cols]
        if missing:
            console.print(
                f"  [yellow]WARNING[/yellow]: condition '{cond}' missing columns "
                f"{missing} — skipping (no events for this condition?)"
            )
            continue
        idx = [cond_cols.index(c) for c in delay_cols]
        cond_arr = betas_cond[:, idx]  # (n_vtx, n_delays)

        if hemi:
            outname = op.join(
                outdir,
                f"sub-{subject}_ses-{ses_label}_task-{task}"
                f"_hemi-{hemi}_space-{space}{desc_suffix}_desc-FIR{cond}_delays.func.gii",
            )
        else:
            outname = op.join(
                outdir,
                f"sub-{subject}_ses-{ses_label}_task-{task}"
                f"_space-{space}{desc_suffix}_desc-FIR{cond}_delays.func.gii",
            )
        save_timeseries_to_gifti(cond_arr, outname)
    timing["save_cond_maps"] = time.time() - _t

    # ── Fit 2: pooled "AllStim" FIR ──────────────────────────────────────────────
    _t = time.time()
    X_allstim = np.asarray(design_matrix_allstim_std)
    console.print(
        f"  [DIAG] Design matrix (allstim): shape={X_allstim.shape}  "
        f"rank={np.linalg.matrix_rank(X_allstim)}  (expected full rank={X_allstim.shape[1]})"
    )
    labels_a, estimates_a = nilearn_run_glm(Y, X_allstim, n_jobs=n_glm_jobs)
    betas_allstim = _reconstruct_betas(X_allstim.shape[1], n_vtx, labels_a, estimates_a)
    timing["fit_allstim"] = time.time() - _t
    console.print(f"  [dim]run_glm (allstim): {timing['fit_allstim']:.2f} s[/dim]")

    allstim_cols = list(design_matrix_allstim_std.columns)
    betas_allstim_tsv = op.join(
        outdir,
        f"sub-{subject}_ses-{ses_label}_task-{task}"
        f"{('_hemi-' + hemi) if hemi else ''}_space-{space}{desc_suffix}_desc-AllStimFIR_betas.tsv.gz",
    )
    save_array_as_dataframe(betas_allstim, betas_allstim_tsv, col_labels=allstim_cols)

    _t = time.time()
    delay_cols = [f"AllStim_delay_{d}" for d in range(n_delays)]
    missing = [c for c in delay_cols if c not in allstim_cols]
    if missing:
        raise RuntimeError(f"AllStim design matrix missing expected columns: {missing}")
    idx = [allstim_cols.index(c) for c in delay_cols]
    allstim_arr = betas_allstim[:, idx]  # (n_vtx, n_delays)

    hemi_tag = f"_hemi-{hemi}" if hemi else ""

    # 10 single-frame "AllStim_delay_d" maps
    for d in range(n_delays):
        outname = op.join(
            outdir,
            f"sub-{subject}_ses-{ses_label}_task-{task}"
            f"{hemi_tag}_space-{space}{desc_suffix}_desc-AllStimdelay{d}_statmap.func.gii",
        )
        save_statmap_to_gifti(allstim_arr[:, d], outname)

    # Stacked "4D" surface map (n_delays frames)
    outname_stack = op.join(
        outdir,
        f"sub-{subject}_ses-{ses_label}_task-{task}"
        f"{hemi_tag}_space-{space}{desc_suffix}_desc-AllStimFIR0-18s_timeseries.func.gii",
    )
    save_timeseries_to_gifti(allstim_arr, outname_stack)
    timing["save_allstim_maps"] = time.time() - _t

    console.print(f"  [green]FIR GLM done[/green]  (hemi-{hemi if hemi else 'volumetric'})")
    return timing


# ---------------------------------------------------------------------------
# Helpers: parse sessions (same as run_allses_glm.py)
# ---------------------------------------------------------------------------

def _parse_subject_sessions(
    sub: str,
    sessions_arg: Optional[str],
    file_arg: Optional[str],
) -> list[str] | None:
    if sessions_arg:
        return [s.strip().zfill(2) for s in sessions_arg.split(",") if s.strip()]

    if file_arg:
        from pathlib import Path
        path = Path(file_arg)
        if not path.exists():
            console.print(f"[red]ERROR[/red]: file not found: {file_arg}")
            raise typer.Exit(1)
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        seen: dict[str, None] = {}
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh, delimiter=delimiter):
                row_sub = str(row["sub"]).strip().zfill(2)
                if row_sub != sub:
                    continue
                if "RUN" in row and str(row["RUN"]).strip() != "True":
                    continue
                ses = str(row["ses"]).strip().zfill(2)
                seen[ses] = None
        if not seen:
            console.print(
                f"[red]ERROR[/red]: no sessions found for sub-{sub} in {file_arg}"
            )
            raise typer.Exit(1)
        return list(seen)

    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    base: str = typer.Option(..., "--base", help="Base directory, e.g. /bcbl/home/public/Gari/VOTCLOC/main_exp"),
    sub: str = typer.Option(..., "--sub", help="Subject label without 'sub-', e.g. 03"),
    sessions_arg: Optional[str] = typer.Option(
        None, "--sessions",
        help="Comma-separated session labels, e.g. '01,02,03,04,05,06,07,08,09'",
    ),
    file_arg: Optional[str] = typer.Option(
        None, "-f",
        help="Subseslist CSV/TSV; rows for this --sub are used (RUN==True filter applied)",
    ),
    fp_ana_name: str = typer.Option(..., "--fp-ana-name", help="fMRIPrep analysis name"),
    task: str = typer.Option(..., "--task", help="Task name, e.g. fLoc"),
    start_scans: int = typer.Option(..., "--start-scans", help="Non-steady-state TRs to drop"),
    space: str = typer.Option("fsnative", "--space", help="Space: fsnative | fsaverage"),
    output_name: str = typer.Option(..., "--output-name", help="Output folder label"),
    input_dirname: str = typer.Option("BIDS", "--input-dir", "-i", help="BIDS dir name under base"),
    slice_time_ref: float = typer.Option(0.5, "--slice-time-ref"),
    use_smoothed: bool = typer.Option(False, "--use-smoothed"),
    sm: str = typer.Option("", "--sm", help="FreeSurfer FWHM label, e.g. 05"),
    mask: str = typer.Option("", "--mask", help="FreeSurfer label file to apply as mask"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Build design matrices but do not write outputs"),
    rerun_map: Optional[str] = typer.Option(
        None, "--rerun-map",
        help="Path to rerun_check.tsv for compensated-run exclusion",
    ),
    conditions: str = typer.Option(
        "RW,CS,FF,SC,bodylimb,face", "--conditions",
        help="Comma-separated list of trial_type names to model with FIR.",
    ),
    n_delays: int = typer.Option(10, "--n-delays", help="Number of FIR delays (TR units)."),
    n_glm_jobs: int = typer.Option(1, "--n-glm-jobs", help="n_jobs for nilearn run_glm."),
) -> None:
    t0 = time.time()

    sub = sub.strip().zfill(2)
    sessions = _parse_subject_sessions(sub, sessions_arg, file_arg)
    conditions_list = [c.strip() for c in conditions.split(",") if c.strip()]

    bids_dir = op.join(base, input_dirname)
    fsdir = op.join(bids_dir, "derivatives", "freesurfer")
    fmriprep_dir = op.join(bids_dir, "derivatives", f"fmriprep-{fp_ana_name}")
    label_dir = f"{fsdir}/sub-{sub}/label"
    is_surface = space in ["fsnative", "fsaverage"]

    if not is_surface:
        console.print(f"[red]ERROR[/red]: this script only supports surface spaces (fsnative/fsaverage), got '{space}'")
        raise typer.Exit(1)

    rerun_excl = _load_rerun_exclusions(rerun_map) if rerun_map else {}

    console.print("Creating BIDS layout …")
    layout = BIDSLayout(bids_dir, validate=False)
    console.print("Creating fMRIPrep layout …")
    fp_layout = BIDSLayout(fmriprep_dir, validate=False)
    console.print("[green]Layouts ready.[/green]\n")

    # Auto-detect sessions if neither --sessions nor -f was given
    if sessions is None:
        raw = sorted(s.zfill(2) for s in layout.get_sessions(subject=sub, task=task))
        if not raw:
            console.print(f"[red]ERROR[/red]: no sessions found in BIDS for sub-{sub} task-{task}")
            raise typer.Exit(1)

        ext = ".func.gii"
        sessions = []
        skipped = []
        for s in raw:
            func_dir = op.join(fmriprep_dir, f"sub-{sub}", f"ses-{s}", "func")
            has_func = op.isdir(func_dir) and any(
                task in f and f.endswith(ext) for f in os.listdir(func_dir)
            )
            if has_func:
                sessions.append(s)
            else:
                skipped.append(s)

        if skipped:
            console.print(
                f"  [yellow]Skipped (no fMRIprep func for task-{task}):[/yellow] "
                f"{', '.join('ses-' + s for s in skipped)}"
            )
        if not sessions:
            console.print(f"[red]ERROR[/red]: no sessions with fMRIprep func data found for sub-{sub} task-{task}")
            raise typer.Exit(1)

        console.print(
            f"  [dim]Auto-detected {len(sessions)} valid session(s): "
            f"{', '.join('ses-' + s for s in sessions)}[/dim]\n"
        )

    # ── Launch summary ────────────────────────────────────────────────────────
    console.rule("[bold cyan]All-Sessions FIR GLM Launch[/bold cyan]")
    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    tbl.add_column("key", style="dim")
    tbl.add_column("value", style="bold")
    tbl.add_row("Subject", f"sub-{sub}")
    tbl.add_row("Sessions", f"({len(sessions)})  {', '.join('ses-' + s for s in sessions)}")
    tbl.add_row("Task", task)
    tbl.add_row("Space", space)
    tbl.add_row("fMRIPrep", fp_ana_name)
    tbl.add_row("Output name", output_name)
    tbl.add_row("Start scans", str(start_scans))
    tbl.add_row("Conditions", ", ".join(conditions_list))
    tbl.add_row("FIR delays", f"0..{n_delays - 1}  (TR units)")
    tbl.add_row("Smoothed", f"Yes (sm={sm})" if use_smoothed else "No")
    tbl.add_row("Mask", mask or "—")
    tbl.add_row("Rerun map", rerun_map or "— (no exclusions)")
    tbl.add_row("Mode", "[yellow]DRY-RUN[/yellow]" if dry_run else "[green]EXECUTE[/green]")
    console.print(tbl)

    timing_per_hemi: dict[str, dict[str, float]] = {}
    hemis = ["L", "R"]

    for hemi in hemis:
        label = f"hemi-{hemi}"
        console.rule(f"[bold]Processing {label}[/bold]")

        conc_data_std, dm_cond_std, dm_allstim_std, nonan_confounds = prepare_allses_fir_input(
            bids_dir=bids_dir,
            fmriprep_dir=fmriprep_dir,
            fp_layout=fp_layout,
            layout=layout,
            label_dir=label_dir,
            subject=sub,
            sessions=sessions,
            task=task,
            start_scans=start_scans,
            space=space,
            slice_time_ref=slice_time_ref,
            use_smoothed=use_smoothed,
            sm=sm,
            apply_label_as_mask=mask,
            rerun_excl=rerun_excl,
            conditions=conditions_list,
            n_delays=n_delays,
            hemi=hemi,
        )
        console.print(f"  Design matrix (cond) columns: {list(dm_cond_std.columns)}")
        console.print(f"  Design matrix (allstim) columns: {list(dm_allstim_std.columns)}")

        if dry_run:
            console.print(
                "  [dim]Dry-run — design matrices printed above, nothing written.[/dim]"
            )
            del conc_data_std, dm_cond_std, dm_allstim_std, nonan_confounds
            gc.collect()
            timing_per_hemi[label] = {}
            continue

        timing = glm_fir_allses(
            conc_data_std=conc_data_std,
            design_matrix_cond_std=dm_cond_std,
            design_matrix_allstim_std=dm_allstim_std,
            bids_dir=bids_dir,
            task=task,
            space=space,
            subject=sub,
            output_name=output_name,
            conditions=conditions_list,
            n_delays=n_delays,
            use_smoothed=use_smoothed,
            sm=sm,
            hemi=hemi,
            n_glm_jobs=n_glm_jobs,
        )
        del conc_data_std, dm_cond_std, dm_allstim_std, nonan_confounds
        gc.collect()
        timing_per_hemi[label] = timing

    total_elapsed = time.time() - t0
    console.rule("[bold cyan]Done[/bold cyan]")
    if any(v for v in timing_per_hemi.values()):
        tbl_t = Table(title="Step timing (s)", box=box.SIMPLE_HEAD, show_footer=True)
        tbl_t.add_column("step", footer="[bold]total[/bold]")
        hemi_totals = []
        for hl in timing_per_hemi:
            col_total = sum(timing_per_hemi[hl].values())
            hemi_totals.append(col_total)
            tbl_t.add_column(hl, justify="right", footer=f"[bold]{col_total:.2f}[/bold]")
        all_steps = list({k for t in timing_per_hemi.values() for k in t})
        for step in all_steps:
            tbl_t.add_row(step, *[f"{timing_per_hemi[hl].get(step, 0.0):.2f}" for hl in timing_per_hemi])
        console.print(tbl_t)
    else:
        console.print(f"  [dim]Dry-run completed.[/dim]  ({total_elapsed:.1f} s)")

    console.print(
        f"  [bold]Total program time:[/bold]  {total_elapsed:.1f} s  ({total_elapsed / 60:.1f} min)"
    )
    console.print(
        f"  Output: [bold]{bids_dir}/derivatives/l1_surface_fir/"
        f"analysis-{output_name}/sub-{sub}/allses/[/bold]"
    )


if __name__ == "__main__":
    app()
