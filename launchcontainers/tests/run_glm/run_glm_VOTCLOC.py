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
# -----------------------------------------------------------------------------
from __future__ import annotations

import logging
import os
import os.path as op
import random
import time
from os import makedirs
from typing import List, Optional

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import typer
import yaml
from bids import BIDSLayout
from nilearn.glm.contrasts import compute_contrast
from nilearn.glm.first_level import first_level_from_bids
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.first_level.first_level import run_glm
from nilearn.plotting import plot_design_matrix, plot_contrast_matrix
from nilearn.surface import load_surf_data
from rich import box
from rich.console import Console
from rich.table import Table
from scipy import stats

console = Console()
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
logger = logging.getLogger("GENERAL")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_random_run_combinations(total_runs, num_runs, n_iterations, seed=None):
    """
    Generate random combinations of runs for power analysis.

    Parameters
    ----------
    total_runs : int
        Total number of available runs.
    num_runs : int
        Number of runs to select in each iteration.
    n_iterations : int
        Number of random combinations to generate.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list of lists
        Each sublist contains num_runs randomly selected run numbers.
    """
    if seed is not None:
        random.seed(seed + num_runs)

    available_runs = list(range(1, total_runs + 1))
    combinations = []
    for _ in range(n_iterations):
        selected = random.sample(available_runs, num_runs)
        selected.sort()
        combinations.append(selected)
    return combinations


def save_statmap_to_gifti(data, outname):
    """Save a stat-map array to a GIFTI file."""
    gii = nib.gifti.gifti.GiftiImage()
    gii.add_gifti_data_array(
        nib.gifti.gifti.GiftiDataArray(data=data, datatype="NIFTI_TYPE_FLOAT32"),
    )
    nib.save(gii, outname)


def replace_prefix_and_suffix(val):
    if isinstance(val, str) and (val.endswith("1") or val.endswith("2")):
        val = val[:-1]
    if isinstance(val, str) and val[:3] in {
        "EU_", "ES_", "AT_", "EN_", "FR_", "IT_", "CN_", "ZH_", "JP_",
    }:
        return val[3:]
    return val


def load_contrasts(yaml_file, design_matrix):
    """
    Load contrast definitions from a YAML file and convert to contrast vectors.

    Parameters
    ----------
    yaml_file : str
        Path to the YAML contrast definition file.
    design_matrix : pd.DataFrame
        The design matrix (used to align contrast vectors to columns).

    Returns
    -------
    dict
        {contrast_name: contrast_vector (np.ndarray)}.
    """
    with open(yaml_file) as f:
        contrast_definitions = yaml.safe_load(f)

    contrast_matrix = np.eye(design_matrix.shape[1])
    basic_contrasts = {
        col: contrast_matrix[i] for i, col in enumerate(design_matrix.columns)
    }

    contrasts = {}
    for name, conditions in contrast_definitions.items():
        vec = np.zeros(design_matrix.shape[1])
        pos_terms = conditions.get("positive", [])
        neg_terms = conditions.get("negative", [])
        pos_w = 1 / len(pos_terms) if pos_terms else 0
        neg_w = -1 / len(neg_terms) if neg_terms else 0
        for term in pos_terms:
            if term in basic_contrasts:
                vec += pos_w * basic_contrasts[term]
        for term in neg_terms:
            if term in basic_contrasts:
                vec += neg_w * basic_contrasts[term]
        contrasts[name] = vec

    return contrasts


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_design_matrix_to_file(design_matrix, outdir, subject, session, task):
    """Save the design matrix plot to <outdir>/design_matrix_{task}.png."""
    ax = plot_design_matrix(design_matrix)
    fig = ax.get_figure()
    fig.suptitle(f"sub-{subject}  ses-{session}  task-{task}")
    outpath = op.join(outdir, f"design_matrix_{task}.png")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  [dim]Design matrix saved → {outpath}[/dim]")
    return outpath


def plot_contrast_matrices(contrasts, design_matrix, outdir, subject, session, task):
    """Save one contrast-matrix plot per contrast."""
    for key, values in contrasts.items():
        ax = plot_contrast_matrix(values, design_matrix=design_matrix)
        fig = ax.get_figure()
        fig.suptitle(f"sub-{subject}  ses-{session}  task-{task}  {key}")
        outpath = op.join(outdir, f"contrast_matrix_{task}_{key}.png")
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)
    console.print(f"  [dim]Contrast matrices saved → {outpath}[/dim]")
    return outpath


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def glm_l1(
    conc_data_std, design_matrix_std, contrasts,
    bids_dir, task, space, subject, session,
    analysis_name, use_smoothed=False, sm=None, randrun_idx=None, hemi=None,
) -> dict[str, float]:
    """
    Fit the GLM and compute contrasts.

    Returns
    -------
    dict[str, float]
        Wall-clock seconds spent computing each contrast.
    """
    console.print("[bold]------- GLM start running[/bold]")

    outdir = op.join(
        bids_dir, "derivatives", "l1_surface",
        f"analysis-{analysis_name}", f"sub-{subject}", f"ses-{session}",
    )
    if not op.exists(outdir):
        makedirs(outdir)

    plot_design_matrix_to_file(design_matrix_std, outdir, subject, session, task)
    plot_contrast_matrices(contrasts, design_matrix_std, outdir, subject, session, task)

    Y = np.transpose(conc_data_std)
    X = np.asarray(design_matrix_std)

    console.print(
        f"  [DIAG] Y (vertices × timepoints): shape={Y.shape}  "
        f"min={np.nanmin(Y):.4f}  max={np.nanmax(Y):.4f}  std={np.nanstd(Y):.4f}"
    )
    console.print(
        f"  [DIAG] Design matrix: shape={X.shape}  "
        f"rank={np.linalg.matrix_rank(X)}  (expected full rank={X.shape[1]})"
    )

    labels, estimates = run_glm(Y, X, n_jobs=1)

    timing: dict[str, float] = {}

    for contrast_id, contrast_val in contrasts.items():
        t_c = time.time()

        # Build output filename template
        if hemi:
            outname_base = (
                f"sub-{subject}_ses-{session}_task-{task}"
                f"_hemi-{hemi}_space-{space}_contrast-{contrast_id}"
                f"_stat-X_statmap.func.gii"
            )
        else:
            outname_base = (
                f"sub-{subject}_ses-{session}_task-{task}"
                f"_space-{space}_contrast-{contrast_id}"
                f"_stat-X_statmap.nii.gz"
            )
        if use_smoothed:
            outname_base = outname_base.replace("_statmap", f"_desc-smoothed{sm}_statmap")
        if randrun_idx:
            outname_base = outname_base.replace("_statmap", f"{randrun_idx}_statmap")
        outname_base = op.join(outdir, outname_base)

        contrast_obj = compute_contrast(labels, estimates, contrast_val)

        betas     = contrast_obj.effect_size()
        t_value   = contrast_obj.stat()
        z_score   = contrast_obj.z_score()
        p_value   = contrast_obj.p_value()
        variance  = contrast_obj.effect_variance()

        console.print(
            f"  [DIAG] contrast={contrast_id}  "
            f"z: min={np.nanmin(z_score):.4f}  max={np.nanmax(z_score):.4f}  "
            f"std={np.nanstd(z_score):.4f}  n_nan={np.sum(np.isnan(z_score))}"
        )

        if hemi:
            save_statmap_to_gifti(betas,   outname_base.replace("stat-X", "stat-effect"))
            save_statmap_to_gifti(t_value, outname_base.replace("stat-X", "stat-t"))
            if not randrun_idx:
                save_statmap_to_gifti(z_score,  outname_base.replace("stat-X", "stat-z"))
                save_statmap_to_gifti(p_value,  outname_base.replace("stat-X", "stat-p"))
                save_statmap_to_gifti(variance, outname_base.replace("stat-X", "stat-variance"))
        else:
            console.print(
                f"  [yellow]WARNING[/yellow]: volumetric output not implemented, skipping {outname_base}"
            )

        timing[contrast_id] = time.time() - t_c

    label = f"hemi-{hemi}" if hemi else "volumetric"
    console.print(f"  [green]GLM done[/green] ({label})")
    return timing


def prepare_glm_input(
    bids_dir, fmriprep_dir, fp_layout, label_dir, contrast_fpath,
    subject, session, analysis_name, task, start_scans, space,
    slice_time_ref, run_list, use_smoothed, sm, apply_label_as_mask, hemi=None,
):
    """
    Gather per-run timeseries, events, and confounds; build concatenated
    design matrix and contrast dict for a single GLM call.

    Returns
    -------
    tuple
        (conc_data_std, design_matrix_std, contrasts)
    """
    is_surface = space in ["fsnative", "fsaverage"]

    data_allrun      = []
    frame_time_allrun = []
    events_allrun    = []
    confounds_allrun = []
    store_l1         = []

    # Per-run step timing: {run_num: {step: seconds}}
    run_step_times: dict[str, dict[str, float]] = {}

    for idx, run_num in enumerate(run_list):
        console.print(f"  Processing run [cyan]{run_num}[/cyan]")
        run_step_times[run_num] = {}

        # ── Step 1: find + load functional data ─────────────────────────────
        _t = time.time()
        query_params = {
            "subject":   subject,
            "session":   session,
            "task":      task,
            "run":       run_num,
            "space":     space,
            "suffix":    "bold",
            "extension": ".func.gii" if is_surface else ".nii.gz",
        }
        if is_surface and hemi:
            query_params["hemi"] = hemi
        if use_smoothed:
            query_params["desc"] = f"smoothed{sm}"
        elif not is_surface:
            query_params["desc"] = "preproc"

        func_files = fp_layout.get(**query_params, invalid_filters="allow")
        if not func_files:
            console.print(
                f"  [yellow]WARNING[/yellow]: no functional file for run {run_num} "
                f"(query: {query_params})"
            )
            continue

        func_file = func_files[0].path
        console.print(f"  Found: [dim]{func_file}[/dim]")

        if is_surface:
            data = load_surf_data(func_file)
            data_float = np.vstack(data[:, :]).astype(float)
        else:
            img = nib.load(func_file)
            data_array = img.get_fdata()
            original_shape = data_array.shape[:3]
            n_timepoints = data_array.shape[3]
            data_float = data_array.reshape(-1, n_timepoints).astype(float)
            console.print(
                f"  Volumetric shape: {original_shape}, timepoints: {n_timepoints}"
            )
        run_step_times[run_num]["load_func"] = time.time() - _t
        console.print(
            f"  Length original data: {np.shape(data_float)[1]}  "
            f"[dim](load_func: {run_step_times[run_num]['load_func']:.1f} s)[/dim]"
        )

        # ── Step 2: z-score + trim ────────────────────────────────────────────
        _t = time.time()
        data_remove_first = data_float[:, start_scans:]
        console.print(f"  Length after removing {start_scans} prescan TRs: {np.shape(data_remove_first)[1]}")

        data_std  = stats.zscore(data_remove_first, axis=1)
        n_features = np.shape(data_std)[0]

        if apply_label_as_mask:
            if is_surface:
                label_path = f"{label_dir}/{apply_label_as_mask}"
                surf_mask  = load_surf_data(label_path)
                mask = np.zeros((n_features, 1))
                mask[surf_mask] = 1
                data_std   = data_std * mask
                data_float = data_float * mask
            else:
                console.print("  [yellow]WARNING[/yellow]: volumetric masking not implemented")

        n_scans = np.shape(data_std)[1]
        data_allrun.append(data_std)
        run_step_times[run_num]["zscore_mask"] = time.time() - _t

        # ── Step 3: first_level_from_bids (events + confounds + TR) ──────────
        _t = time.time()
        img_filters = [("desc", "preproc"), ("ses", session), ("run", run_num)]
        try:
            l1 = first_level_from_bids(
                bids_dir,
                task,
                space_label="T1w",
                sub_labels=[subject],
                slice_time_ref=slice_time_ref,
                hrf_model="spm",
                drift_model=None,
                drift_order=0,
                high_pass=None,
                img_filters=img_filters,
                derivatives_folder=fmriprep_dir,
            )
        except (TypeError, FileNotFoundError, IndexError) as e:
            console.print(f"  [yellow]WARNING[/yellow]: error processing run {run_num}: {e} — skipping")
            continue
        run_step_times[run_num]["first_level_from_bids"] = time.time() - _t
        console.print(
            f"  [dim]first_level_from_bids: {run_step_times[run_num]['first_level_from_bids']:.1f} s[/dim]"
        )

        # ── Step 4: confound processing ───────────────────────────────────────
        _t = time.time()
        t_r      = l1[0][0].t_r
        events   = l1[2][0][0]
        confounds = l1[3][0][0]
        events.loc[:, "onset"] = events["onset"] + idx * n_scans * t_r

        events_nobaseline = events[events.loc[:, "trial_type"] != "baseline"]
        events_allrun.append(events_nobaseline)
        store_l1.append(l1)

        motion_keys = [
            "framewise_displacement",
            "rot_x", "rot_y", "rot_z",
            "trans_x", "trans_y", "trans_z",
        ]
        a_compcor_keys     = [k for k in confounds.keys() if "a_comp_cor"  in k]
        non_steady_keys    = [k for k in confounds.keys() if "non_steady"  in k]
        cosine_keys        = [k for k in confounds.keys() if "cosine"      in k]
        confound_keys_keep = motion_keys + a_compcor_keys + cosine_keys + non_steady_keys
        confounds_keep = confounds[confound_keys_keep]

        confounds_keep["framewise_displacement"][0] = np.nanmean(
            confounds_keep["framewise_displacement"]
        )
        confounds_keep = confounds_keep.iloc[start_scans:]
        confounds_allrun.append(confounds_keep)

        frame_times = t_r * ((np.arange(n_scans) + slice_time_ref) + idx * n_scans)
        frame_time_allrun.append(frame_times)
        run_step_times[run_num]["confounds"] = time.time() - _t
        console.print(
            f"  Confounds length: {len(confounds_keep)}  "
            f"[dim](confound processing: {run_step_times[run_num]['confounds']:.1f} s)[/dim]"
        )

    # ── Per-run timing summary table ──────────────────────────────────────────
    if run_step_times:
        step_cols = ["load_func", "zscore_mask", "first_level_from_bids", "confounds"]
        tbl_run = Table(title="Per-run step timing (s)", box=box.SIMPLE_HEAD)
        tbl_run.add_column("run")
        for s in step_cols:
            tbl_run.add_column(s, justify="right")
        tbl_run.add_column("run_total", justify="right")
        for rn, steps in run_step_times.items():
            vals = [steps.get(s, 0.0) for s in step_cols]
            tbl_run.add_row(
                rn,
                *[f"{v:.2f}" for v in vals],
                f"{sum(vals):.2f}",
            )
        console.print(tbl_run)

    # ── Step 5: build design matrix ───────────────────────────────────────────
    _t = time.time()
    conc_data_std   = np.concatenate(data_allrun, axis=1)
    concat_frame_times = np.concatenate(frame_time_allrun, axis=0)
    concat_events   = pd.concat(events_allrun, axis=0)
    concat_events   = concat_events.applymap(replace_prefix_and_suffix)
    concat_confounds = pd.concat(confounds_allrun, axis=0)

    console.print(f"\n  Confound columns:\n  {list(concat_confounds.columns)}")
    nonan_confounds = concat_confounds.dropna(axis=1, how="any")
    console.print(f"  Confound columns after dropna: {list(nonan_confounds.columns)}")

    design_matrix = make_first_level_design_matrix(
        concat_frame_times,
        events=concat_events,
        hrf_model="spm",
        drift_model=None,
        add_regs=nonan_confounds,
    )

    design_matrix_std = design_matrix.apply(stats.zscore, axis=0)
    design_matrix_std["constant"] = np.ones(len(design_matrix_std)).astype(int)

    contrasts = load_contrasts(contrast_fpath, design_matrix)
    t_design = time.time() - _t
    console.print(
        f"  Design matrix columns: {list(design_matrix.columns)}\n"
        f"  [dim]design_matrix build: {t_design:.2f} s[/dim]"
    )

    return conc_data_std, design_matrix_std, contrasts


def _load_rerun_exclusions(rerun_tsv: str) -> dict[tuple[str, str, str], set[str]]:
    """
    Read rerun_check.tsv and return the set of compensates_run values to
    exclude per (sub, ses, task).

    Only rows where ``found_in_bids == "True"`` are included — if the redo
    run is not in BIDS there is nothing to replace the original with, so the
    original must stay.

    Sub, ses, and run values are zero-padded to two digits to match BIDS
    run labels.

    Returns
    -------
    dict[tuple[str, str, str], set[str]]
        ``{(sub, ses, task): {compensates_run_str, ...}}``
    """
    import csv as _csv

    excl: dict[tuple[str, str, str], set[str]] = {}
    with open(rerun_tsv, newline="") as fh:
        for row in _csv.DictReader(fh, delimiter="\t"):
            if str(row.get("found_in_bids", "")).strip() != "True":
                continue
            sub  = str(row["sub"]).strip().zfill(2)
            ses  = str(row["ses"]).strip().zfill(2)
            task = str(row["task"]).strip()
            crun = str(row["compensates_run"]).strip().zfill(2)
            excl.setdefault((sub, ses, task), set()).add(crun)
    return excl


def generate_run_groups(
    layout, subject, session, task,
    selected_runs=None,
    excl_runs: set[str] | None = None,
):
    """
    Return the run list for a task and a run-label string for filenames.

    Parameters
    ----------
    excl_runs : set[str] | None
        Zero-padded run strings to drop (compensated/aborted runs from
        rerun_check.tsv).  Ignored when *selected_runs* is given explicitly.

    Returns
    -------
    tuple[list[str], str | None]
        (run_list, randrun_idx)
    """
    if not selected_runs:
        runs = sorted(set(layout.get_runs(subject=subject, session=session, task=task)))
        randrun_idx = None
    else:
        runs = selected_runs
        randrun_idx = f"_run-{''.join(map(str, runs))}"

    if not runs:
        raise ValueError(f"No runs found for task '{task}' in BIDS dataset.")

    run_list = [f"{r:02d}" for r in runs]

    # Filter out compensated (aborted) runs — only when not using explicit selected_runs
    if excl_runs and not selected_runs:
        before     = set(run_list)
        run_list   = [r for r in run_list if r not in excl_runs]
        excluded   = sorted(before - set(run_list))
        kept_extra = sorted(excl_runs - before)  # extra_runs already in list
        if excluded:
            console.print(
                f"  [yellow]Excluded compensated runs:[/yellow] {excluded}  "
                f"(replaced by extra runs in BIDS)"
            )
        if kept_extra:
            console.print(
                f"  [dim]Note: extra/redo runs not in BIDS (no exclusion needed): {kept_extra}[/dim]"
            )

    console.print(f"  Run list: {run_list}")
    return run_list, randrun_idx


def process_run_list(
    bids_dir, fmriprep_dir, fp_layout, label_dir, contrast_fpath,
    subject, session, analysis_name, task, start_scans, space, slice_time_ref,
    run_list, use_smoothed, sm, apply_label_as_mask, dry_run,
    randrun_idx=None, hemi=None,
) -> dict[str, float]:
    """
    Build GLM inputs and run the GLM for one run-list / hemisphere combination.

    Returns
    -------
    dict[str, float]
        Per-contrast wall-clock seconds (empty dict in dry-run mode).
    """
    label = f"hemi-{hemi}" if hemi else "volumetric"
    console.print(f"\n[bold]Processing {label}[/bold]  runs: {run_list}")

    conc_data_std, design_matrix_std, contrasts = prepare_glm_input(
        bids_dir, fmriprep_dir, fp_layout, label_dir, contrast_fpath,
        subject, session, analysis_name, task, start_scans, space, slice_time_ref,
        run_list, use_smoothed, sm, apply_label_as_mask, hemi,
    )
    console.print(f"  Contrasts: {list(contrasts.keys())}")

    if dry_run:
        console.print("  [dim]Dry-run — design matrix and confounds printed above, nothing written.[/dim]")
        return {}

    return glm_l1(
        conc_data_std, design_matrix_std, contrasts,
        bids_dir, task, space, subject, session,
        analysis_name, use_smoothed, sm, randrun_idx, hemi,
    )


def run_power_analysis(
    bids_dir, fmriprep_dir, fp_layout, label_dir, contrast_fpath,
    subject, session, base_analysis_name, task, start_scans, space, slice_time_ref,
    use_smoothed, sm, apply_label_as_mask, dry_run,
    total_runs, n_iterations, seed, hemi=None,
) -> None:
    """Run power analysis: total_runs × n_iterations GLMs."""
    label = f"hemi-{hemi}" if hemi else "volumetric"
    console.rule(f"[bold cyan]POWER ANALYSIS — {label}[/bold cyan]")
    console.print(
        f"  Subject: [bold]{subject}[/bold]  Session: [bold]{session}[/bold]\n"
        f"  Configurations: 1 … {total_runs} run(s)  ×  {n_iterations} iterations"
        f"  = [bold]{total_runs * n_iterations}[/bold] GLMs\n"
        f"  Seed: {seed}"
    )

    total_glms       = total_runs * n_iterations
    glms_done        = 0
    iter_times: list[float] = []

    for num_of_runs in range(1, total_runs + 1):
        console.rule(f"[cyan]Configuration: {num_of_runs} run(s)[/cyan]", style="dim")
        combinations = generate_random_run_combinations(total_runs, num_of_runs, n_iterations, seed)

        for iter_num, selected_runs in enumerate(combinations, start=1):
            run_list    = [f"{r:02d}" for r in selected_runs]
            randrun_idx = f"_run-{''.join(map(str, selected_runs))}"
            iter_output = f"{base_analysis_name}/power_analysis_{num_of_runs}_run/iter_{iter_num:02d}"

            console.print(f"  Iteration {iter_num}/{n_iterations}: runs {selected_runs}")
            t_iter = time.time()

            process_run_list(
                bids_dir, fmriprep_dir, fp_layout, label_dir, contrast_fpath,
                subject, session, iter_output, task, start_scans,
                space, slice_time_ref,
                run_list, use_smoothed, sm, apply_label_as_mask, dry_run,
                randrun_idx, hemi,
            )

            elapsed = time.time() - t_iter
            iter_times.append(elapsed)
            glms_done += 1
            pct = glms_done / total_glms * 100
            console.print(
                f"  Progress: {glms_done}/{total_glms} ({pct:.1f}%)  "
                f"this iter: {elapsed:.1f} s"
            )

    # Power-analysis summary
    console.rule(f"[bold green]POWER ANALYSIS DONE — {label}[/bold green]")
    console.print(
        f"  GLMs completed : {glms_done}\n"
        f"  Total time     : {sum(iter_times):.1f} s\n"
        f"  Avg / GLM      : {sum(iter_times)/len(iter_times):.1f} s"
    )


# ---------------------------------------------------------------------------
# Timing summary table
# ---------------------------------------------------------------------------

def _print_timing_table(
    timing_per_hemi: dict[str, dict[str, float]],
    total_elapsed: float,
) -> None:
    """
    Print a Rich table of per-contrast, per-hemi wall-clock seconds.

    Parameters
    ----------
    timing_per_hemi : dict
        {hemi_label: {contrast_id: elapsed_seconds}}
        hemi_label is e.g. "hemi-L", "hemi-R", or "volumetric".
    total_elapsed : float
        Total program wall-clock time in seconds.
    """
    hemi_labels  = list(timing_per_hemi.keys())
    # Collect all contrast IDs in insertion order
    all_contrasts: list[str] = []
    for timing in timing_per_hemi.values():
        for c in timing:
            if c not in all_contrasts:
                all_contrasts.append(c)

    tbl = Table(
        title="Contrast Timing Summary",
        box=box.SIMPLE_HEAD,
        show_footer=True,
    )
    tbl.add_column("Contrast", style="cyan", footer="[bold]TOTAL[/bold]")
    hemi_totals: list[float] = []
    for hl in hemi_labels:
        col_total = sum(timing_per_hemi[hl].get(c, 0.0) for c in all_contrasts)
        hemi_totals.append(col_total)
        tbl.add_column(
            hl,
            justify="right",
            footer=f"[bold]{col_total:.2f} s[/bold]",
        )
    grand_total_contrasts = sum(hemi_totals)
    tbl.add_column(
        "Sum",
        justify="right",
        footer=f"[bold]{grand_total_contrasts:.2f} s[/bold]",
    )

    for c in all_contrasts:
        row_vals = [timing_per_hemi[hl].get(c, 0.0) for hl in hemi_labels]
        row_sum  = sum(row_vals)
        tbl.add_row(c, *[f"{v:.2f}" for v in row_vals], f"{row_sum:.2f}")

    console.print(tbl)
    console.print(
        f"  [bold]Total program time:[/bold]  {total_elapsed:.1f} s  "
        f"({total_elapsed/60:.1f} min)"
    )


# ---------------------------------------------------------------------------
# Helpers: sub/ses pair parsing
# ---------------------------------------------------------------------------

def _parse_pairs(subses_arg: Optional[str], file_arg: Optional[str]) -> list[tuple[str, str]]:
    """
    Parse sub/ses pairs from either a single ``sub,ses`` string or a TSV/CSV file.

    The file must have a header row with ``sub`` and ``ses`` columns.
    If a ``RUN`` column is present, only rows where ``RUN == "True"`` are included.
    Values are zero-padded to two digits.

    Returns
    -------
    list[tuple[str, str]]
        Ordered list of (sub, ses) string pairs.
    """
    import csv
    from pathlib import Path

    if subses_arg:
        parts = subses_arg.split(",")
        if len(parts) != 2:
            console.print(f"[red]ERROR[/red]: -s expects 'sub,ses' (e.g. 01,09), got: {subses_arg!r}")
            raise typer.Exit(1)
        sub = parts[0].strip().zfill(2)
        ses = parts[1].strip().zfill(2)
        return [(sub, ses)]

    if file_arg:
        path = Path(file_arg)
        if not path.exists():
            console.print(f"[red]ERROR[/red]: subseslist not found: {path}")
            raise typer.Exit(1)
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        pairs: list[tuple[str, str]] = []
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh, delimiter=delimiter):
                if "RUN" in row and str(row["RUN"]).strip() != "True":
                    continue
                sub = str(row["sub"]).strip().zfill(2)
                ses = str(row["ses"]).strip().zfill(2)
                pairs.append((sub, ses))
        return pairs

    console.print("[red]ERROR[/red]: provide either -s <sub,ses> or -f <subseslist>")
    raise typer.Exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    base: str = typer.Option(..., "--base", help="Base directory, e.g. /scratch/tlei/VOTCLOC"),
    subses_arg: Optional[str] = typer.Option(None, "-s", help="Single sub,ses pair, e.g. 01,09"),
    file_arg: Optional[str] = typer.Option(None, "-f", help="Path to subseslist TSV/CSV file"),
    fp_ana_name: str = typer.Option(..., "--fp-ana-name", help="fMRIPrep analysis name"),
    task: str = typer.Option(..., "--task", help="Task name, e.g. fLoc"),
    start_scans: int = typer.Option(..., "--start-scans", help="Number of non-steady-state TRs to drop"),
    space: str = typer.Option(..., "--space", help="Space: T1w | fsnative | fsaverage | MNI152NLin2009cAsym"),
    contrast: str = typer.Option(..., "--contrast", help="Path to YAML contrast definition file"),
    analysis_name: str = typer.Option(..., "--analysis-name", help="Analysis name (output folder label)"),
    input_dirname: str = typer.Option("BIDS", "--input-dir", "-i", help="Input BIDS dir name under base"),
    slice_time_ref: float = typer.Option(0.5, "--slice-time-ref", help="Slice timing reference (fMRIPrep default 0.5)"),
    use_smoothed: bool = typer.Option(False, "--use-smoothed", help="Use smoothed functional files"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print design matrix / confounds; do not write outputs"),
    sm: str = typer.Option("", "--sm", help="FreeSurfer FWHM smoothing label, e.g. 05"),
    mask: str = typer.Option("", "--mask", help="FreeSurfer label file to apply as mask"),
    selected_runs: Optional[str] = typer.Option(
        None, "--selected-runs",
        help="Comma-separated run numbers to use, e.g. '1,3,5'. Default: all runs.",
    ),
    power_analysis: bool = typer.Option(False, "--power-analysis", help="Run power analysis mode"),
    n_iterations: int = typer.Option(10, "--n-iterations", help="Iterations per run count in power analysis"),
    seed: int = typer.Option(42, "--seed", help="Random seed for power analysis"),
    total_runs: int = typer.Option(10, "--total-runs", help="Total runs available (power analysis)"),
    rerun_map: Optional[str] = typer.Option(
        None, "--rerun-map",
        help=(
            "Path to rerun_check.tsv.  Compensated (aborted) runs are automatically "
            "excluded from the run list; their replacement extra runs are kept.  "
            "Ignored when --selected-runs is given."
        ),
    ),
) -> None:
    t0 = time.time()

    # ── Parse inputs ─────────────────────────────────────────────────────────
    pairs = _parse_pairs(subses_arg, file_arg)

    selected_runs_list: Optional[List[int]] = None
    if selected_runs:
        selected_runs_list = [int(r.strip()) for r in selected_runs.split(",")]

    # Shared directories (layout-level, not per-subject)
    bids_dir     = op.join(base, input_dirname)
    fsdir        = op.join(bids_dir, "derivatives", "freesurfer")
    fmriprep_dir = op.join(bids_dir, "derivatives", f"fmriprep-{fp_ana_name}")
    is_surface   = space in ["fsnative", "fsaverage"]

    # Count contrasts early (YAML only, no design matrix needed)
    with open(contrast) as _f:
        _contrast_defs = yaml.safe_load(_f)
    n_contrasts = len(_contrast_defs)

    # Load rerun exclusion map (once, shared across all sessions)
    rerun_excl: dict[tuple[str, str, str], set[str]] = {}
    if rerun_map:
        rerun_excl = _load_rerun_exclusions(rerun_map)

    # ── Launch summary ───────────────────────────────────────────────────────
    console.rule("[bold cyan]GLM Launch[/bold cyan]")
    tbl_launch = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    tbl_launch.add_column("key",   style="dim")
    tbl_launch.add_column("value", style="bold")

    mode_str = "[yellow]DRY-RUN[/yellow]" if dry_run else "[green]EXECUTE[/green]"
    runs_str = (
        ", ".join(map(str, selected_runs_list)) if selected_runs_list else "all (from BIDS layout)"
    )
    pairs_str = "  ".join(f"sub-{s} ses-{e}" for s, e in pairs)
    tbl_launch.add_row("Subjects/Sessions", f"({len(pairs)})  {pairs_str}")
    tbl_launch.add_row("Task",              task)
    tbl_launch.add_row("Space",             space)
    tbl_launch.add_row("fMRIPrep",          fp_ana_name)
    tbl_launch.add_row("Analysis name",      analysis_name)
    tbl_launch.add_row("Runs",              runs_str)
    tbl_launch.add_row("Contrasts",         f"{n_contrasts}  ({contrast})")
    tbl_launch.add_row("Start scans",       str(start_scans))
    tbl_launch.add_row("Slice time ref",    str(slice_time_ref))
    tbl_launch.add_row("Smoothed",          f"Yes (sm={sm})" if use_smoothed else "No")
    tbl_launch.add_row("Mask",              mask or "—")
    tbl_launch.add_row(
        "Rerun map",
        rerun_map if rerun_map else "[dim]— (no exclusions)[/dim]",
    )
    tbl_launch.add_row("Mode",              mode_str)
    if power_analysis:
        tbl_launch.add_row(
            "Power analysis",
            f"Yes  ({total_runs} runs × {n_iterations} iter = {total_runs * n_iterations} GLMs)",
        )
    console.print(tbl_launch)
    # ────────────────────────────────────────────────────────────────────────

    # Build BIDS layouts ONCE — reused across all sub/ses pairs
    console.print("Creating BIDS layout …")
    layout = BIDSLayout(bids_dir, validate=False)
    console.print("Creating fMRIPrep layout …")
    fp_layout = BIDSLayout(fmriprep_dir, validate=False)
    console.print("[green]Layouts ready.[/green]  (shared across all sessions)\n")

    # ── Loop over sub/ses pairs ───────────────────────────────────────────────
    # Records: {(sub, ses): total_seconds}
    session_times: dict[tuple[str, str], float] = {}

    for sub, ses in pairs:
        t_ses = time.time()
        label_dir = f"{fsdir}/sub-{sub}/label"

        console.rule(f"[bold magenta]sub-{sub}  ses-{ses}[/bold magenta]")

        # ── Power analysis mode ──────────────────────────────────────────────
        if power_analysis:
            hemis = ["L", "R"] if is_surface else [None]
            for hemi in hemis:
                run_power_analysis(
                    bids_dir, fmriprep_dir, fp_layout, label_dir, contrast,
                    sub, ses, analysis_name, task, start_scans, space, slice_time_ref,
                    use_smoothed, sm, mask, dry_run,
                    total_runs, n_iterations, seed, hemi,
                )
            session_times[(sub, ses)] = time.time() - t_ses
            continue

        # ── Regular mode ─────────────────────────────────────────────────────
        excl_runs = rerun_excl.get((sub, ses, task), set())
        run_list, randrun_idx = generate_run_groups(
            layout, sub, ses, task, selected_runs_list, excl_runs or None,
        )
        timing_per_hemi: dict[str, dict[str, float]] = {}

        if is_surface:
            for hemi in ["L", "R"]:
                console.rule(f"[bold]Hemisphere {hemi}[/bold]", style="dim")
                timing = process_run_list(
                    bids_dir, fmriprep_dir, fp_layout, label_dir, contrast,
                    sub, ses, analysis_name, task, start_scans, space, slice_time_ref,
                    run_list, use_smoothed, sm, mask, dry_run,
                    randrun_idx, hemi,
                )
                timing_per_hemi[f"hemi-{hemi}"] = timing
        else:
            console.rule("[bold]Volumetric[/bold]", style="dim")
            timing = process_run_list(
                bids_dir, fmriprep_dir, fp_layout, label_dir, contrast,
                sub, ses, analysis_name, task, start_scans, space, slice_time_ref,
                run_list, use_smoothed, sm, mask, dry_run,
                randrun_idx, hemi=None,
            )
            timing_per_hemi["volumetric"] = timing

        ses_elapsed = time.time() - t_ses
        session_times[(sub, ses)] = ses_elapsed

        # Per-session contrast timing table
        console.rule(f"[cyan]sub-{sub} ses-{ses} — Contrast Timing[/cyan]", style="dim")
        if any(v for v in timing_per_hemi.values()):
            _print_timing_table(timing_per_hemi, ses_elapsed)
        else:
            console.print(
                f"  [dim]Dry-run — no outputs written.[/dim]  ({ses_elapsed:.1f} s)"
            )

    # ── Final summary across all sessions ────────────────────────────────────
    console.rule("[bold cyan]Run Summary[/bold cyan]")
    tbl_sum = Table(box=box.SIMPLE_HEAD, show_footer=True)
    tbl_sum.add_column("sub",     style="bold", footer="[bold]Total[/bold]")
    tbl_sum.add_column("ses",     style="bold")
    tbl_sum.add_column("time (s)", justify="right",
                       footer=f"[bold]{sum(session_times.values()):.1f}[/bold]")
    tbl_sum.add_column("time (min)", justify="right",
                       footer=f"[bold]{sum(session_times.values())/60:.1f}[/bold]")
    for (s, e), t in session_times.items():
        tbl_sum.add_row(s, e, f"{t:.1f}", f"{t/60:.1f}")
    console.print(tbl_sum)

    total_elapsed = time.time() - t0
    console.print(
        f"  [bold]Total program time:[/bold]  {total_elapsed:.1f} s  ({total_elapsed/60:.1f} min)"
    )


if __name__ == "__main__":
    app()
