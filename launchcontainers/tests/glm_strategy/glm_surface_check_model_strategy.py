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
import os.path as op
import random
import time
from os import makedirs
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
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


def save_statmap_to_nifti(data: np.ndarray, outname: str, affine: np.ndarray, shape: tuple) -> None:
    """Save a flat (n_voxels,) stat-map array to a NIfTI file, reshaping to ``shape``."""
    data_3d = data.reshape(shape)
    nib.save(nib.Nifti1Image(data_3d.astype(np.float32), affine), outname)


def save_timeseries_to_gifti(data: np.ndarray, outname: str) -> None:
    """
    Save a (n_vertices, n_frames) array as a multi-frame GIFTI.

    Used for residuals and fitted timeseries (frame = timepoint) and for
    per-regressor beta maps (frame = regressor column).
    """
    gii = nib.gifti.gifti.GiftiImage()
    intent = nib.nifti1.intent_codes["NIFTI_INTENT_TIME_SERIES"]
    for i in range(data.shape[1]):
        gii.add_gifti_data_array(
            nib.gifti.gifti.GiftiDataArray(
                data=data[:, i].astype(np.float32),
                intent=intent,
                datatype="NIFTI_TYPE_FLOAT32",
            )
        )
    nib.save(gii, outname)
    console.print(f"  [dim]  → {op.basename(outname)}[/dim]")


def save_timeseries_to_nifti(
    data: np.ndarray, outname: str, affine: np.ndarray, shape: tuple
) -> None:
    """
    Save a (n_voxels, n_frames) array as a 4D NIfTI, reshaping each frame to ``shape``.

    Used for residuals/fitted timeseries (frame = timepoint) and for
    per-regressor beta maps (frame = regressor column), volumetric space.
    """
    n_frames = data.shape[1]
    data_4d = np.moveaxis(data.T.reshape((n_frames,) + shape), 0, -1)
    nib.save(nib.Nifti1Image(data_4d.astype(np.float32), affine), outname)
    console.print(f"  [dim]  → {op.basename(outname)}[/dim]")


def _reconstruct_vertex_array(
    n_tp: int,
    n_vtx: int,
    labels: np.ndarray,
    estimates: dict,
    attr: str,
) -> np.ndarray:
    """
    Rebuild a (n_vertices, n_timepoints) array from nilearn run_glm output.

    ``attr`` is "resid"/"residuals" (ε = y − Ŷ) or "predicted" (Ŷ = Xθ).
    Handles nilearn versions that renamed the attribute from 'resid' to 'residuals'.
    """
    out = np.zeros((n_vtx, n_tp), dtype=np.float32)
    for label, result in estimates.items():
        mask = labels == label
        val = getattr(result, attr, None)
        if val is None:  # nilearn renamed 'resid' → 'residuals' across versions
            val = getattr(result, "residuals" if attr == "resid" else attr)
        out[mask, :] = val.T.astype(np.float32)
    return out


def _reconstruct_betas(
    n_reg: int,
    n_vtx: int,
    labels: np.ndarray,
    estimates: dict,
) -> np.ndarray:
    """
    Rebuild a (n_vertices, n_regressors) beta array from nilearn run_glm output.

    result.theta has shape (n_regressors, n_verts_for_label).
    Frame order matches columns of the design matrix.
    """
    out = np.zeros((n_vtx, n_reg), dtype=np.float32)
    for label, result in estimates.items():
        mask = labels == label
        out[mask, :] = result.theta.T.astype(np.float32)
    return out


def save_array_as_dataframe(
    data: np.ndarray,
    outname: str,
    row_labels=None,
    col_labels=None,
) -> None:
    """
    Save a 2-D array as a tab-separated DataFrame.

    Timeseries (residuals / fitted): shape (n_timepoints, n_vertices),
    rows = timepoints, columns = vertex indices.  Saved as .tsv.gz.

    Betas: shape (n_vertices, n_regressors), rows = vertices,
    columns = regressor names.  Saved as plain .tsv.

    The gzip extension is detected from ``outname`` automatically.
    """
    df = pd.DataFrame(data, index=row_labels, columns=col_labels)
    kw = (
        {"sep": "\t", "compression": "gzip"}
        if outname.endswith(".gz")
        else {"sep": "\t"}
    )
    df.to_csv(outname, **kw)
    console.print(f"  [dim]  → {op.basename(outname)}[/dim]")


def replace_prefix_and_suffix(val):
    if isinstance(val, str) and (val.endswith("1") or val.endswith("2")):
        val = val[:-1]
    if isinstance(val, str) and val[:3] in {
        "EU_",
        "ES_",
        "AT_",
        "FR_",
        "IT_",
        "CN_",
        "JP_",
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


def load_confound_strategy(strategy_yaml: str, strategy_name: str) -> dict:
    """
    Load the confound strategy YAML and return a bundle dict with keys:
    ``name``, ``cfg``, ``col_groups``, ``regex_groups``.

    The YAML has three top-level keys: ``strategies``, ``column_groups``,
    ``regex_groups``.  ``strategy_name`` selects which entry under
    ``strategies`` to use.
    """
    with open(strategy_yaml) as f:
        raw = yaml.safe_load(f)
    strategies = raw["strategies"]
    col_groups = raw.get("column_groups", {})
    regex_groups = raw.get("regex_groups", {})
    if strategy_name not in strategies:
        raise ValueError(
            f"Strategy '{strategy_name}' not defined in YAML. "
            f"Available: {list(strategies.keys())}"
        )
    return {
        "name": strategy_name,
        "cfg": strategies[strategy_name],
        "col_groups": col_groups,
        "regex_groups": regex_groups,
    }


def _resolve_confound_columns(cs: dict, available_cols: list) -> list[str]:
    """
    Return the ordered subset of fMRIPrep confound columns to keep, driven
    by the active confound strategy bundle ``cs`` from ``load_confound_strategy``.

    ``framewise_displacement`` is always included so the first-TR NaN fix
    runs regardless of strategy.
    """
    import re as _re

    cfg = cs["cfg"]
    col_groups = cs["col_groups"]
    regex_groups = cs["regex_groups"]

    explicit: list[str] = ["framewise_displacement"]
    patterns: list[str] = []

    # ── Explicit column groups ─────────────────────────────────────────────
    motion_level = cfg.get("motion", "none")
    if motion_level != "none":
        explicit.extend(col_groups.get(f"motion_{motion_level}", []))

    wm_csf_level = cfg.get("wm_csf", "none")
    if wm_csf_level != "none":
        explicit.extend(col_groups.get(f"wm_csf_{wm_csf_level}", []))

    gs_level = cfg.get("global_signal", "none")
    if gs_level != "none":
        explicit.extend(col_groups.get(f"global_signal_{gs_level}", []))

    # ── Regex-matched groups ───────────────────────────────────────────────
    if cfg.get("high_pass", False):
        patterns.extend(regex_groups.get("cosine", []))

    if cfg.get("include_non_steady_state", False) or cfg.get(
        "demean", False
    ):  # demean kept for back-compat
        patterns.extend(regex_groups.get("non_steady_state", []))

    if cfg.get("model_spiking", False) or cfg.get(
        "scrub", False
    ):  # scrub kept for back-compat
        prefix = cfg.get("scrub_regressor_prefix", "motion_outlier")
        patterns.extend(regex_groups.get(prefix, [f"^{prefix}"]))

    # ── CompCor ───────────────────────────────────────────────────────────
    compcor_type = cfg.get("compcor", "none")
    if compcor_type is True:  # YAML `compcor: true` → default to anat_combined
        compcor_type = "anat_combined"
    if compcor_type not in (False, "none", "false"):
        comp_pats = regex_groups.get(f"compcor_{compcor_type}", [])
        n_compcor = cfg.get("n_compcor", "all")
        if n_compcor == "all":
            patterns.extend(comp_pats)
        else:
            matched = [
                col
                for col in available_cols
                if any(_re.search(p, col) for p in comp_pats)
            ]
            explicit.extend(matched[: int(n_compcor)])

    # ── ICA-AROMA ─────────────────────────────────────────────────────────
    aroma_type = cfg.get("ica_aroma", "none")
    if aroma_type != "none":
        patterns.extend(regex_groups.get("ica_aroma", []))

    # ── Resolve regex patterns against available columns ───────────────────
    regex_matched = [
        col for col in available_cols if any(_re.search(p, col) for p in patterns)
    ]

    # ── Merge, deduplicate, filter to available ────────────────────────────
    avail_set = set(available_cols)
    seen: set[str] = set()
    result: list[str] = []
    for col in explicit + regex_matched:
        if col not in seen and col in avail_set:
            seen.add(col)
            result.append(col)
    return result


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_design_matrix_to_file(design_matrix, outdir, subject, session, task, tag=""):
    """Save the design matrix plot to <outdir>/design_matrix_{task}{tag}.png."""
    ax = plot_design_matrix(design_matrix)
    fig = ax.get_figure()
    fig.suptitle(f"sub-{subject}  ses-{session}  task-{task}")
    outpath = op.join(outdir, f"design_matrix_{task}{tag}.png")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  [dim]Design matrix saved → {outpath}[/dim]")
    return outpath


def plot_contrast_matrices(contrasts, design_matrix, outdir, subject, session, task, tag=""):
    """Save one contrast-matrix plot per contrast."""
    for key, values in contrasts.items():
        ax = plot_contrast_matrix(values, design_matrix=design_matrix)
        fig = ax.get_figure()
        fig.suptitle(f"sub-{subject}  ses-{session}  task-{task}  {key}")
        outpath = op.join(outdir, f"contrast_matrix_{task}{tag}_{key}.png")
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)
    console.print(f"  [dim]Contrast matrices saved → {outpath}[/dim]")
    return outpath


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def glm_l1(
    conc_data_std,
    design_matrix_std,
    contrasts,
    bids_dir,
    task,
    space,
    subject,
    session,
    analysis_name,
    confounds_df,
    strategy_name,
    use_smoothed=False,
    sm=None,
    randrun_idx=None,
    hemi=None,
    n_glm_jobs=1,
    vol_affine=None,
    vol_shape=None,
    save_betas=False,
    acq=None,
    bold_desc=None,
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
        bids_dir,
        "derivatives",
        "l1_surface",
        f"analysis-{analysis_name}",
        f"sub-{subject}",
        f"ses-{session}",
    )
    if not op.exists(outdir):
        makedirs(outdir)

    # acq/desc tags — keep outputs from different acq/bold_desc combos
    # (e.g. acq-SE vs acq-ME desc-denoised vs acq-ME desc-optcom) from
    # overwriting each other.
    acq_tok = f"_acq-{acq}" if acq else ""
    desc_tok = f"_desc-{bold_desc}" if bold_desc else ""
    tag = f"{acq_tok}{desc_tok}"

    confounds_tsv = op.join(
        outdir,
        f"sub-{subject}_ses-{session}_task-{task}{tag}_desc-confounds_timeseries.tsv",
    )
    # ── Save metadata: confounds TSV + design matrix CSV + plots ─────────────
    _t = time.time()
    confounds_df.to_csv(confounds_tsv, sep="\t")
    console.print(f"  [dim]Confounds TSV → {op.basename(confounds_tsv)}[/dim]")
    plot_design_matrix_to_file(design_matrix_std, outdir, subject, session, task, tag=tag)
    plot_contrast_matrices(contrasts, design_matrix_std, outdir, subject, session, task, tag=tag)
    dm_csv = op.join(outdir, f"design_matrix_{task}{tag}_strategy-{strategy_name}.csv")
    design_matrix_std.to_csv(dm_csv)
    console.print(f"  [dim]Design matrix CSV → {op.basename(dm_csv)}[/dim]")
    t_save_meta = time.time() - _t

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

    # ── GLM fitting ────────────────────────────────────────────────────────────
    _t = time.time()
    labels, estimates = run_glm(Y, X, n_jobs=n_glm_jobs)
    t_glm_fit = time.time() - _t
    console.print(f"  [dim]run_glm: {t_glm_fit:.2f} s[/dim]")

    # ── Reconstruct vertex arrays + save components ────────────────────────────
    # Y = Xθ + ε   (fitted = Xθ,  residuals = ε,  betas = θ per regressor)
    t_reconstruct = 0.0
    t_save_components = 0.0
    if save_betas:
        _t = time.time()
        n_vtx, n_tp = Y.shape
        resid = _reconstruct_vertex_array(n_tp, n_vtx, labels, estimates, "resid")
        fitted = _reconstruct_vertex_array(n_tp, n_vtx, labels, estimates, "predicted")
        beta_arr = _reconstruct_betas(X.shape[1], n_vtx, labels, estimates)
        t_reconstruct = time.time() - _t

        def _component_name(desc: str, ext: str) -> str:
            desc_label = f"{bold_desc}{desc.capitalize()}" if bold_desc else desc
            if hemi:
                base = (
                    f"sub-{subject}_ses-{session}_task-{task}{acq_tok}"
                    f"_hemi-{hemi}_space-{space}_desc-{desc_label}_timeseries{ext}"
                )
            else:
                base = (
                    f"sub-{subject}_ses-{session}_task-{task}{acq_tok}"
                    f"_space-{space}_desc-{desc_label}_timeseries{ext}"
                )
            if use_smoothed:
                base = base.replace(f"_desc-{desc_label}", f"_desc-{desc_label}sm{sm}")
            if randrun_idx:
                base = base.replace("_timeseries", f"{randrun_idx}_timeseries")
            return op.join(outdir, base)

        _t = time.time()
        if hemi:
            save_timeseries_to_gifti(resid, _component_name("residuals", ".func.gii"))
            save_timeseries_to_gifti(fitted, _component_name("fitted", ".func.gii"))
            save_timeseries_to_gifti(beta_arr, _component_name("betas", ".func.gii"))
            save_array_as_dataframe(
                beta_arr,
                _component_name("betas", ".tsv"),
                col_labels=design_matrix_std.columns.tolist(),
            )
        else:
            save_timeseries_to_nifti(resid, _component_name("residuals", ".nii.gz"), vol_affine, vol_shape)
            save_timeseries_to_nifti(fitted, _component_name("fitted", ".nii.gz"), vol_affine, vol_shape)
            save_timeseries_to_nifti(beta_arr, _component_name("betas", ".nii.gz"), vol_affine, vol_shape)
        t_save_components = time.time() - _t

    # ── Contrasts: compute then save stat maps ─────────────────────────────────
    timing: dict[str, float] = {}
    t_compute_total = 0.0
    t_save_maps_total = 0.0

    desc_parts = []
    if bold_desc:
        desc_parts.append(bold_desc)
    if use_smoothed:
        desc_parts.append(f"smoothed{sm}")
    stat_desc_tok = f"_desc-{''.join(desc_parts)}" if desc_parts else ""

    for contrast_id, contrast_val in contrasts.items():
        if hemi:
            outname_base = op.join(
                outdir,
                f"sub-{subject}_ses-{session}_task-{task}{acq_tok}"
                f"_hemi-{hemi}_space-{space}{stat_desc_tok}_contrast-{contrast_id}"
                f"_stat-X_statmap.func.gii",
            )
        else:
            outname_base = op.join(
                outdir,
                f"sub-{subject}_ses-{session}_task-{task}{acq_tok}"
                f"_space-{space}{stat_desc_tok}_contrast-{contrast_id}"
                f"_stat-X_statmap.nii.gz",
            )
        if randrun_idx:
            outname_base = outname_base.replace("_statmap", f"{randrun_idx}_statmap")

        _t = time.time()
        contrast_obj = compute_contrast(labels, estimates, contrast_val)
        effect = contrast_obj.effect_size()
        t_value = contrast_obj.stat()
        z_score = contrast_obj.z_score()
        p_value = contrast_obj.p_value()
        variance = contrast_obj.effect_variance()
        t_compute = time.time() - _t
        t_compute_total += t_compute

        console.print(
            f"  [DIAG] contrast={contrast_id}  "
            f"z: min={np.nanmin(z_score):.4f}  max={np.nanmax(z_score):.4f}  "
            f"std={np.nanstd(z_score):.4f}  n_nan={np.sum(np.isnan(z_score))}"
        )

        _t = time.time()
        if hemi:
            save_statmap_to_gifti(effect, outname_base.replace("stat-X", "stat-effect"))
            save_statmap_to_gifti(t_value, outname_base.replace("stat-X", "stat-t"))
            if not randrun_idx:
                save_statmap_to_gifti(z_score, outname_base.replace("stat-X", "stat-z"))
                save_statmap_to_gifti(p_value, outname_base.replace("stat-X", "stat-p"))
                save_statmap_to_gifti(
                    variance, outname_base.replace("stat-X", "stat-variance")
                )
        else:
            save_statmap_to_nifti(effect, outname_base.replace("stat-X", "stat-effect"), vol_affine, vol_shape)
            save_statmap_to_nifti(t_value, outname_base.replace("stat-X", "stat-t"), vol_affine, vol_shape)
            if not randrun_idx:
                save_statmap_to_nifti(z_score, outname_base.replace("stat-X", "stat-z"), vol_affine, vol_shape)
                save_statmap_to_nifti(p_value, outname_base.replace("stat-X", "stat-p"), vol_affine, vol_shape)
                save_statmap_to_nifti(variance, outname_base.replace("stat-X", "stat-variance"), vol_affine, vol_shape)
        t_save = time.time() - _t
        t_save_maps_total += t_save

        timing[contrast_id] = t_compute + t_save

    # ── Step timing summary table ──────────────────────────────────────────────
    hemi_label = f"hemi-{hemi}" if hemi else "volumetric"
    tbl_steps = Table(
        title=f"GLM step timing — {hemi_label}", box=box.SIMPLE_HEAD, show_footer=True
    )
    tbl_steps.add_column("step", footer="[bold]total[/bold]")
    tbl_steps.add_column(
        "time (s)",
        justify="right",
        footer=f"[bold]{t_save_meta + t_glm_fit + t_reconstruct + t_save_components + t_compute_total + t_save_maps_total:.2f}[/bold]",
    )
    steps = [
        ("save_metadata  (confounds TSV + DM CSV + plots)", t_save_meta),
        ("glm_fit        (run_glm, all vertices)", t_glm_fit),
    ]
    if save_betas:
        steps += [
            ("reconstruct    (resid / fitted / beta arrays)", t_reconstruct),
            ("save_components (residuals + fitted + betas GIFTI)", t_save_components),
        ]
    steps += [
        ("compute_contrasts", t_compute_total),
        ("save_maps      (stat-map GIFTIs)", t_save_maps_total),
    ]
    for step, t in steps:
        tbl_steps.add_row(step, f"{t:.2f}")
    console.print(tbl_steps)
    console.print(f"  [green]GLM done[/green] ({hemi_label})")

    return timing


def _load_nilearn_confounds(func_file: str, cfg: dict, start_scans: int):
    """
    Load confounds via nilearn.interfaces.fmriprep.load_confounds_strategy.

    Returns
    -------
    confounds_df : pd.DataFrame
        Confounds trimmed to the post-start_scans TRs (and scrubbed rows removed
        for scrubbing strategies).
    rel_sample_mask : np.ndarray | None
        For scrubbing strategies: 0-based TR indices into the post-start_scans
        data array that should be kept.  None when no scrubbing is applied.
    """
    _LOAD_CONFOUNDS_PARAMS = (
        "motion",
        "wm_csf",
        "global_signal",
        "compcor",
        "n_compcor",
        "fd_threshold",
        "std_dvars_threshold",
        "scrub",
        "demean",
        "include_non_steady_state",
    )

    if "nilearn_strategy" in cfg:
        # Preset strategy — load_confounds_strategy
        from nilearn.interfaces.fmriprep import load_confounds_strategy

        extra_kw = {
            k: cfg[k]
            for k in ("fd_threshold", "std_dvars_threshold", "n_compcor", "scrub")
            if k in cfg
        }
        confounds, sample_mask = load_confounds_strategy(
            func_file, denoise_strategy=cfg["nilearn_strategy"], **extra_kw
        )
    elif "strategy" in cfg:
        # Custom strategy — load_confounds with explicit component tuple
        from nilearn.interfaces.fmriprep import load_confounds

        extra_kw = {k: cfg[k] for k in _LOAD_CONFOUNDS_PARAMS if k in cfg}
        confounds, sample_mask = load_confounds(
            func_file, strategy=tuple(cfg["strategy"]), **extra_kw
        )
    else:
        raise ValueError(
            "nilearn backend requires either 'nilearn_strategy' (preset) "
            "or 'strategy' (custom component list) in the strategy config."
        )

    if sample_mask is not None:
        # sample_mask: integer indices of kept TRs in the FULL run (not pre-filtered).
        # confounds has the full run length; use sample_mask as integer row selector,
        # then additionally drop TRs before start_scans.
        keep = sample_mask >= start_scans  # boolean mask over sample_mask
        kept_indices = sample_mask[keep]  # integer indices into full confounds
        confounds = confounds.iloc[kept_indices].reset_index(drop=True)
        rel_sample_mask = kept_indices - start_scans
    else:
        confounds = confounds.iloc[start_scans:].reset_index(drop=True)
        rel_sample_mask = None

    return confounds, rel_sample_mask


def _fetch_bids_run_metadata(
    bids_dir,
    fmriprep_dir,
    task,
    subject,
    session,
    run_list,
    slice_time_ref,
    acq: str | None = None,
    bold_desc: str | None = None,
    bids_layout=None,
    fp_bids_layout=None,
) -> dict:
    """
    Return {run_num: (t_r, events_df, confounds_df)} for each run.

    SE path (acq=None): uses ``first_level_from_bids`` with ``desc-preproc``
    and ``space-T1w`` — the standard fmriprep output for single-echo data.

    ME path (acq provided): fmriprep outputs per-echo T1w files
    (``echo-N_desc-preproc``) *and* a native-BOLD-space optimal combination
    (``desc-preproc``, no echo, no space entity).  ``first_level_from_bids``
    cannot disambiguate these, so the ME path queries the pre-built layouts
    directly:
      - TR from the native BOLD OC JSON sidecar
      - events from the raw BIDS layout (acq-specific, with inheritance fallback)
      - confounds from the fmriprep layout (acq-specific TSV)

    ``bids_layout`` and ``fp_bids_layout`` must be passed when ``acq`` is set.
    Runs that fail are omitted from the dict.
    """
    meta: dict = {}

    for run_num in run_list:
        try:
            if acq:
                # ── ME: direct layout queries ─────────────────────────────────
                assert bids_layout is not None and fp_bids_layout is not None, (
                    "bids_layout and fp_bids_layout are required when acq is set"
                )

                # TR — from native BOLD OC (desc-preproc, no echo, no space entity)
                bold_candidates = fp_bids_layout.get(
                    subject=subject,
                    session=session,
                    task=task,
                    run=run_num,
                    acquisition=acq,
                    desc="preproc",
                    suffix="bold",
                    extension=".nii.gz",
                    invalid_filters="allow",
                )
                oc_files = [
                    f
                    for f in bold_candidates
                    if not f.entities.get("echo") and not f.entities.get("space")
                ]
                if not oc_files:
                    raise FileNotFoundError(
                        f"No native BOLD OC (desc-preproc, no echo/space) "
                        f"for acq-{acq} run-{run_num}"
                    )
                t_r = oc_files[0].get_metadata()["RepetitionTime"]
                console.print(
                    f"  [dim]ME metadata anchor: {oc_files[0].filename}  TR={t_r}[/dim]"
                )

                # Events — acq-specific, with inheritance fallback
                ev_files = bids_layout.get(
                    subject=subject,
                    session=session,
                    task=task,
                    run=run_num,
                    acquisition=acq,
                    suffix="events",
                    extension=".tsv",
                    invalid_filters="allow",
                )
                if not ev_files:
                    ev_files = bids_layout.get(
                        subject=subject,
                        session=session,
                        task=task,
                        run=run_num,
                        suffix="events",
                        extension=".tsv",
                    )
                if not ev_files:
                    raise FileNotFoundError(
                        f"No events.tsv for acq-{acq} run-{run_num}"
                    )
                events_df = pd.read_csv(ev_files[0].path, sep="\t")

                # Confounds — acq-specific TSV from fmriprep
                conf_files = fp_bids_layout.get(
                    subject=subject,
                    session=session,
                    task=task,
                    run=run_num,
                    acquisition=acq,
                    desc="confounds",
                    suffix="timeseries",
                    extension=".tsv",
                    invalid_filters="allow",
                )
                if not conf_files:
                    raise FileNotFoundError(
                        f"No confounds TSV for acq-{acq} run-{run_num}"
                    )
                confounds_df = pd.read_csv(conf_files[0].path, sep="\t")

                meta[run_num] = (t_r, events_df, confounds_df)

            else:
                # ── SE: first_level_from_bids with desc-preproc + space-T1w ──
                img_filters = [("desc", "preproc"), ("ses", session), ("run", run_num)]
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
                meta[run_num] = (l1[0][0].t_r, l1[2][0][0], l1[3][0][0])

        except (TypeError, FileNotFoundError, IndexError, AssertionError) as e:
            console.print(
                f"  [yellow]WARNING[/yellow]: error fetching metadata for run {run_num}: {e} — skipping"
            )
    return meta


def prepare_glm_input(
    bids_dir,
    fmriprep_dir,
    fp_layout,
    label_dir,
    contrast_fpath,
    subject,
    session,
    analysis_name,
    task,
    start_scans,
    space,
    slice_time_ref,
    run_list,
    use_smoothed,
    sm,
    apply_label_as_mask,
    confound_strategy,
    run_metadata,
    hemi=None,
    bold_desc=None,
    n_vols=0,
    acq=None,
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

    data_allrun = []
    frame_time_allrun = []
    events_allrun = []
    confounds_allrun = []
    vol_affine = None
    vol_shape = None

    # Per-run step timing: {run_num: {step: seconds}}
    run_step_times: dict[str, dict[str, float]] = {}

    # Per-run scrubbing stats (populated only when rel_sample_mask is not None)
    scrub_stats_per_run: list[dict] = []

    for idx, run_num in enumerate(run_list):
        console.print(f"  Processing run [cyan]{run_num}[/cyan]")
        run_step_times[run_num] = {}

        # ── Step 1: find + load functional data ─────────────────────────────
        _t = time.time()
        query_params = {
            "subject": subject,
            "session": session,
            "task": task,
            "run": run_num,
            "space": space,
            "suffix": "bold",
            "extension": ".func.gii" if is_surface else ".nii.gz",
        }
        if is_surface and hemi:
            query_params["hemi"] = hemi
        if acq:
            query_params["acquisition"] = acq
        if bold_desc:
            # explicit desc override (e.g. 'denoised', 'optcom' for tedana outputs)
            query_params["desc"] = bold_desc
        elif use_smoothed:
            query_params["desc"] = f"smoothed{sm}"
        elif not is_surface:
            query_params["desc"] = "preproc"

        func_files = fp_layout.get(**query_params, invalid_filters="allow")
        # When no bold_desc is requested, prefer the file without a desc entity
        # (fMRIprep's native projection). Without this, tedana symlinks (desc-denoised,
        # desc-optcom) would contaminate the result set and [0] would be arbitrary.
        if func_files and not bold_desc:
            no_desc = [f for f in func_files if not f.entities.get("desc")]
            if no_desc:
                func_files = no_desc
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
            if vol_affine is None:
                vol_affine = img.affine
                vol_shape = original_shape
        run_step_times[run_num]["load_func"] = time.time() - _t
        console.print(
            f"  Length original data: {np.shape(data_float)[1]}  "
            f"[dim](load_func: {run_step_times[run_num]['load_func']:.1f} s)[/dim]"
        )

        # ── Step 1b: truncate to n_vols before start_scans removal ───────────
        if n_vols > 0 and data_float.shape[1] > n_vols:
            console.print(
                f"  [dim]Truncating {data_float.shape[1]} → {n_vols} volumes "
                f"(--n-vols)[/dim]"
            )
            data_float = data_float[:, :n_vols]

        # ── Step 2: z-score + trim ────────────────────────────────────────────
        _t = time.time()
        data_remove_first = data_float[:, start_scans:]
        console.print(
            f"  Length after removing {start_scans} prescan TRs: {np.shape(data_remove_first)[1]}"
        )

        data_std = stats.zscore(data_remove_first, axis=1)
        n_features = np.shape(data_std)[0]

        if apply_label_as_mask:
            if is_surface:
                label_path = f"{label_dir}/{apply_label_as_mask}"
                surf_mask = load_surf_data(label_path)
                mask = np.zeros((n_features, 1))
                mask[surf_mask] = 1
                data_std = data_std * mask
                data_float = data_float * mask
            else:
                console.print(
                    "  [yellow]WARNING[/yellow]: volumetric masking not implemented"
                )

        n_scans_full = np.shape(data_std)[1]
        run_step_times[run_num]["zscore_mask"] = time.time() - _t

        # ── Step 3: look up prefetched events + confounds + TR ───────────────
        if run_num not in run_metadata:
            console.print(
                f"  [yellow]WARNING[/yellow]: no prefetched metadata for run {run_num} — skipping"
            )
            continue

        # ── Step 4: confound processing ───────────────────────────────────────
        _t = time.time()
        t_r, events, confounds = run_metadata[run_num]
        events = events.copy()  # don't mutate the shared cache (thread-safety)

        # Clip events to n_vols × TR so design matrix matches truncated data
        if n_vols > 0:
            max_time = n_vols * t_r
            n_before = len(events)
            events = events[events["onset"] < max_time].copy()
            # Clip durations that extend beyond the truncated window
            events["duration"] = events.apply(
                lambda r: min(r["duration"], max_time - r["onset"]), axis=1
            )
            if len(events) < n_before:
                console.print(
                    f"  [dim]Events clipped: {n_before} → {len(events)} rows "
                    f"(max_time={max_time:.1f} s = {n_vols} × {t_r:.3f} s)[/dim]"
                )
        # Use full (pre-scrub) run duration for onset offset so that events
        # from later runs remain correctly timed even when volumes are censored.
        events.loc[:, "onset"] = events["onset"] + idx * n_scans_full * t_r

        events_nobaseline = events[events.loc[:, "trial_type"] != "baseline"]
        events_allrun.append(events_nobaseline)

        backend = confound_strategy["cfg"].get("backend", "yaml")
        if backend == "nilearn":
            confounds_keep, rel_sample_mask = _load_nilearn_confounds(
                func_file, confound_strategy["cfg"], start_scans
            )
            # _load_nilearn_confounds already removed start_scans; clip to n_scans_full
            # in case the full run is longer than the n_vols-truncated data.
            if rel_sample_mask is None and len(confounds_keep) > n_scans_full:
                confounds_keep = confounds_keep.iloc[:n_scans_full].reset_index(
                    drop=True
                )
            scrub_note = (
                f", {len(confounds_keep)} vols kept after scrubbing"
                if rel_sample_mask is not None
                else ""
            )
            console.print(
                f"  [dim]Strategy [cyan]{confound_strategy['name']}[/cyan] (nilearn): "
                f"{len(confounds_keep.columns)} confound cols{scrub_note}[/dim]"
            )
        else:
            confound_keys_keep = _resolve_confound_columns(
                confound_strategy, list(confounds.columns)
            )
            confounds_keep = confounds[confound_keys_keep].copy()
            if "framewise_displacement" in confounds_keep.columns:
                confounds_keep.loc[
                    confounds_keep.index[0], "framewise_displacement"
                ] = np.nanmean(confounds_keep["framewise_displacement"])
            confounds_keep = confounds_keep.iloc[start_scans:].reset_index(drop=True)
            # Clip to n_scans_full: confounds come from the full run but data was
            # already truncated to n_vols, so they may be 1+ rows too long.
            if len(confounds_keep) > n_scans_full:
                confounds_keep = confounds_keep.iloc[:n_scans_full].reset_index(
                    drop=True
                )
            rel_sample_mask = None
            console.print(
                f"  [dim]Strategy [cyan]{confound_strategy['name']}[/cyan] (yaml): "
                f"selected {len(confound_keys_keep)} confound cols[/dim]"
            )

        # Apply scrubbing mask — censors volumes before appending to run list.
        # Frame times use the kept TR indices so events stay correctly aligned.
        if rel_sample_mask is not None:
            data_final = data_std[:, rel_sample_mask]
            frame_times = t_r * (
                (rel_sample_mask + slice_time_ref) + idx * n_scans_full
            )
        else:
            data_final = data_std
            frame_times = t_r * (
                (np.arange(n_scans_full) + slice_time_ref) + idx * n_scans_full
            )

        # ── Scrubbing accounting ──────────────────────────────────────────────
        kept_vols = data_final.shape[1]
        pct_removed = (
            100.0 * (1.0 - kept_vols / n_scans_full)
            if rel_sample_mask is not None
            else 0.0
        )
        scrub_stats_per_run.append(
            {
                "run": run_num,
                "total_vols": n_scans_full,
                "kept_vols": kept_vols,
                "removed_vols": n_scans_full - kept_vols,
                "pct_removed": round(pct_removed, 1),
                "scrubbed": rel_sample_mask is not None,
                "flagged": pct_removed > 15.0,
            }
        )

        data_allrun.append(data_final)
        confounds_allrun.append(confounds_keep)
        frame_time_allrun.append(frame_times)
        run_step_times[run_num]["confounds"] = time.time() - _t
        console.print(
            f"  Confounds length: {len(confounds_keep)}  "
            f"[dim](confound processing: {run_step_times[run_num]['confounds']:.1f} s)[/dim]"
        )

    # ── Per-run timing summary table ──────────────────────────────────────────
    if run_step_times:
        step_cols = ["load_func", "zscore_mask", "confounds"]
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

    # ── Scrubbing report ──────────────────────────────────────────────────────
    any_scrubbed = any(r["scrubbed"] for r in scrub_stats_per_run)
    if any_scrubbed:
        scrub_df = pd.DataFrame(scrub_stats_per_run)
        outdir_scrub = op.join(
            bids_dir,
            "derivatives",
            "l1_surface",
            f"analysis-{analysis_name}",
            f"sub-{subject}",
            f"ses-{session}",
        )
        if not op.exists(outdir_scrub):
            makedirs(outdir_scrub)
        strategy_name_safe = confound_strategy["name"]
        scrub_tsv = op.join(
            outdir_scrub,
            f"sub-{subject}_ses-{session}_task-{task}_strategy-{strategy_name_safe}_desc-scrubbing_report.tsv",
        )
        scrub_df.to_csv(scrub_tsv, sep="\t", index=False)
        console.print(f"  [dim]Scrubbing report → {op.basename(scrub_tsv)}[/dim]")

        tbl_scrub = Table(title="Scrubbing summary", box=box.SIMPLE_HEAD)
        tbl_scrub.add_column("run")
        tbl_scrub.add_column("total", justify="right")
        tbl_scrub.add_column("kept", justify="right")
        tbl_scrub.add_column("removed", justify="right")
        tbl_scrub.add_column("% removed", justify="right")
        tbl_scrub.add_column("flag", justify="center")
        for r in scrub_stats_per_run:
            flag = "[bold red]FLAG >15%[/bold red]" if r["flagged"] else ""
            tbl_scrub.add_row(
                r["run"],
                str(r["total_vols"]),
                str(r["kept_vols"]),
                str(r["removed_vols"]),
                f"{r['pct_removed']:.1f}%",
                flag,
            )
        console.print(tbl_scrub)

        flagged_runs = [r["run"] for r in scrub_stats_per_run if r["flagged"]]
        if flagged_runs:
            console.print(
                f"  [bold red]WARNING[/bold red]: runs {flagged_runs} have >15% volumes removed "
                f"(sub-{subject} ses-{session} strategy={strategy_name_safe}). "
                f"Consider excluding this session or switching strategy."
            )

    # ── Step 5: build design matrix ───────────────────────────────────────────
    _t = time.time()
    conc_data_std = np.concatenate(data_allrun, axis=1)
    concat_frame_times = np.concatenate(frame_time_allrun, axis=0)
    concat_events = pd.concat(events_allrun, axis=0)
    concat_events = concat_events.applymap(replace_prefix_and_suffix)
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

    return conc_data_std, design_matrix_std, contrasts, nonan_confounds, vol_affine, vol_shape


def _load_rerun_exclusions(
    rerun_tsv: str, acq: str | None = None
) -> dict[tuple[str, str, str], set[str]]:
    """
    Read rerun_check.tsv/csv and return the set of compensates_run values to
    exclude per (sub, ses, task).

    Only rows where ``found_in_bids`` is truthy are included.
    When ``acq`` is given, only rows whose ``acq`` column matches (or is "None")
    are included — so acq-SE reruns do not pollute an acq-ME run list.

    Sub, ses, and run values are zero-padded to two digits when numeric.

    Returns
    -------
    dict[tuple[str, str, str], set[str]]
        ``{(sub, ses, task): {compensates_run_str, ...}}``
    """
    import csv as _csv

    excl: dict[tuple[str, str, str], set[str]] = {}
    with open(rerun_tsv, newline="") as fh:
        sample = fh.read(2048)
        fh.seek(0)
        delimiter = "\t" if "\t" in sample else ","
        for row in _csv.DictReader(fh, delimiter=delimiter):
            if str(row.get("found_in_bids", "")).strip().lower() not in (
                "true",
                "1",
                "yes",
            ):
                continue
            row_acq = str(row.get("acq", "None")).strip()
            if acq and row_acq not in ("None", "", acq):
                continue
            raw_sub = str(row["sub"]).strip()
            sub = raw_sub.zfill(2) if raw_sub.isdigit() else raw_sub
            raw_ses = str(row["ses"]).strip()
            ses = raw_ses.zfill(2) if raw_ses.isdigit() else raw_ses
            task = str(row["task"]).strip()
            raw_crun = str(row["compensates_run"]).strip()
            crun = raw_crun.zfill(2) if raw_crun.isdigit() else raw_crun
            excl.setdefault((sub, ses, task), set()).add(crun)
    return excl


def generate_run_groups(
    layout,
    subject,
    session,
    task,
    selected_runs=None,
    excl_runs: set[str] | None = None,
    acq: str | None = None,
):
    """
    Return the run list for a task and a run-label string for filenames.

    Parameters
    ----------
    excl_runs : set[str] | None
        Zero-padded run strings to drop (compensated/aborted runs from
        rerun_check.tsv).  Ignored when *selected_runs* is given explicitly.
    acq : str | None
        Filter runs by acquisition entity (e.g. ``"ME"`` or ``"SE"``).
        When given, only runs that have files with that acquisition are returned.

    Returns
    -------
    tuple[list[str], str | None]
        (run_list, randrun_idx)
    """
    if not selected_runs:
        get_kw: dict = dict(subject=subject, session=session, task=task)
        if acq:
            get_kw["acquisition"] = acq
        runs = sorted(set(layout.get_runs(**get_kw)))
        randrun_idx = None
    else:
        runs = selected_runs
        randrun_idx = f"_run-{''.join(map(str, runs))}"

    if not runs:
        raise ValueError(f"No runs found for task '{task}' in BIDS dataset.")

    run_list = [f"{r:02d}" for r in runs]

    # Filter out compensated (aborted) runs — only when not using explicit selected_runs
    if excl_runs and not selected_runs:
        before = set(run_list)
        run_list = [r for r in run_list if r not in excl_runs]
        excluded = sorted(before - set(run_list))
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
    bids_dir,
    fmriprep_dir,
    fp_layout,
    label_dir,
    contrast_fpath,
    subject,
    session,
    analysis_name,
    task,
    start_scans,
    space,
    slice_time_ref,
    run_list,
    use_smoothed,
    sm,
    apply_label_as_mask,
    dry_run,
    confound_strategy,
    run_metadata,
    randrun_idx=None,
    hemi=None,
    n_glm_jobs=1,
    bold_desc=None,
    n_vols=0,
    acq=None,
    save_betas=False,
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

    conc_data_std, design_matrix_std, contrasts, nonan_confounds, vol_affine, vol_shape = prepare_glm_input(
        bids_dir,
        fmriprep_dir,
        fp_layout,
        label_dir,
        contrast_fpath,
        subject,
        session,
        analysis_name,
        task,
        start_scans,
        space,
        slice_time_ref,
        run_list,
        use_smoothed,
        sm,
        apply_label_as_mask,
        confound_strategy,
        run_metadata,
        hemi,
        bold_desc=bold_desc,
        n_vols=n_vols,
        acq=acq,
    )
    console.print(f"  Contrasts: {list(contrasts.keys())}")

    if dry_run:
        console.print(
            "  [dim]Dry-run — design matrix and confounds printed above, nothing written.[/dim]"
        )
        return {}

    return glm_l1(
        conc_data_std,
        design_matrix_std,
        contrasts,
        bids_dir,
        task,
        space,
        subject,
        session,
        analysis_name,
        nonan_confounds,
        confound_strategy["name"],
        use_smoothed,
        sm,
        randrun_idx,
        hemi,
        n_glm_jobs,
        vol_affine,
        vol_shape,
        save_betas=save_betas,
        acq=acq,
        bold_desc=bold_desc,
    )


def run_power_analysis(
    bids_dir,
    fmriprep_dir,
    fp_layout,
    label_dir,
    contrast_fpath,
    subject,
    session,
    base_analysis_name,
    task,
    start_scans,
    space,
    slice_time_ref,
    use_smoothed,
    sm,
    apply_label_as_mask,
    dry_run,
    total_runs,
    n_iterations,
    seed,
    confound_strategy,
    run_metadata,
    hemi=None,
    bold_desc=None,
    n_vols=0,
    acq=None,
    save_betas=False,
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

    total_glms = total_runs * n_iterations
    glms_done = 0
    iter_times: list[float] = []

    for num_of_runs in range(1, total_runs + 1):
        console.rule(f"[cyan]Configuration: {num_of_runs} run(s)[/cyan]", style="dim")
        combinations = generate_random_run_combinations(
            total_runs, num_of_runs, n_iterations, seed
        )

        for iter_num, selected_runs in enumerate(combinations, start=1):
            run_list = [f"{r:02d}" for r in selected_runs]
            randrun_idx = f"_run-{''.join(map(str, selected_runs))}"
            iter_output = f"{base_analysis_name}/power_analysis_{num_of_runs}_run/iter_{iter_num:02d}"

            console.print(
                f"  Iteration {iter_num}/{n_iterations}: runs {selected_runs}"
            )
            t_iter = time.time()

            process_run_list(
                bids_dir,
                fmriprep_dir,
                fp_layout,
                label_dir,
                contrast_fpath,
                subject,
                session,
                iter_output,
                task,
                start_scans,
                space,
                slice_time_ref,
                run_list,
                use_smoothed,
                sm,
                apply_label_as_mask,
                dry_run,
                confound_strategy,
                run_metadata,
                randrun_idx,
                hemi,
                bold_desc=bold_desc,
                n_vols=n_vols,
                acq=acq,
                save_betas=save_betas,
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
        f"  Avg / GLM      : {sum(iter_times) / len(iter_times):.1f} s"
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
    hemi_labels = list(timing_per_hemi.keys())
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
        row_sum = sum(row_vals)
        tbl.add_row(c, *[f"{v:.2f}" for v in row_vals], f"{row_sum:.2f}")

    console.print(tbl)
    console.print(
        f"  [bold]Total program time:[/bold]  {total_elapsed:.1f} s  "
        f"({total_elapsed / 60:.1f} min)"
    )


# ---------------------------------------------------------------------------
# Helpers: sub/ses pair parsing
# ---------------------------------------------------------------------------


def _parse_pairs(
    subses_arg: Optional[str], file_arg: Optional[str]
) -> list[tuple[str, str]]:
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
            console.print(
                f"[red]ERROR[/red]: -s expects 'sub,ses' (e.g. 01,09), got: {subses_arg!r}"
            )
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
    base: str = typer.Option(
        ..., "--base", help="Base directory, e.g. /scratch/tlei/VOTCLOC"
    ),
    subses_arg: Optional[str] = typer.Option(
        None, "-s", help="Single sub,ses pair, e.g. 01,09"
    ),
    file_arg: Optional[str] = typer.Option(
        None, "-f", help="Path to subseslist TSV/CSV file"
    ),
    fp_ana_name: str = typer.Option(
        ..., "--fp-ana-name", help="fMRIPrep analysis name"
    ),
    task: str = typer.Option(..., "--task", help="Task name, e.g. fLoc"),
    start_scans: int = typer.Option(
        ..., "--start-scans", help="Number of non-steady-state TRs to drop"
    ),
    space: str = typer.Option(
        ..., "--space", help="Space: T1w | fsnative | fsaverage | MNI152NLin2009cAsym"
    ),
    contrast: str = typer.Option(
        ..., "--contrast", help="Path to YAML contrast definition file"
    ),
    analysis_name: str = typer.Option(
        ..., "--analysis-name", help="Analysis name (output folder label)"
    ),
    input_dirname: str = typer.Option(
        "BIDS", "--input-dir", "-i", help="Input BIDS dir name under base"
    ),
    slice_time_ref: float = typer.Option(
        0.5, "--slice-time-ref", help="Slice timing reference (fMRIPrep default 0.5)"
    ),
    use_smoothed: bool = typer.Option(
        False, "--use-smoothed", help="Use smoothed functional files"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print design matrix / confounds; do not write outputs"
    ),
    sm: str = typer.Option("", "--sm", help="FreeSurfer FWHM smoothing label, e.g. 05"),
    mask: str = typer.Option(
        "", "--mask", help="FreeSurfer label file to apply as mask"
    ),
    selected_runs: Optional[str] = typer.Option(
        None,
        "--selected-runs",
        help="Comma-separated run numbers to use, e.g. '1,3,5'. Default: all runs.",
    ),
    power_analysis: bool = typer.Option(
        False, "--power-analysis", help="Run power analysis mode"
    ),
    n_iterations: int = typer.Option(
        10, "--n-iterations", help="Iterations per run count in power analysis"
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for power analysis"),
    total_runs: int = typer.Option(
        10, "--total-runs", help="Total runs available (power analysis)"
    ),
    rerun_map: Optional[str] = typer.Option(
        None,
        "--rerun-map",
        help=(
            "Path to rerun_check.tsv.  Compensated (aborted) runs are automatically "
            "excluded from the run list; their replacement extra runs are kept.  "
            "Ignored when --selected-runs is given."
        ),
    ),
    strategy_yaml: str = typer.Option(
        ..., "--strategy-yaml", help="Path to confound strategy YAML file"
    ),
    strategy: Optional[str] = typer.Option(
        None,
        "--strategy",
        help="Run a single named strategy from the YAML (default: run all strategies)",
    ),
    n_workers: int = typer.Option(
        1,
        "--n-workers",
        help=(
            "Number of strategies to run in parallel. "
            "GLM per-vertex jobs are set to max(1, 4 // n_workers) automatically."
        ),
    ),
    bold_desc: Optional[str] = typer.Option(
        None,
        "--bold-desc",
        help=(
            "desc entity to filter bold files in BIDSLayout query.  "
            "e.g. 'denoised' for tedana output, 'optcom' for optimal combination.  "
            "Default (None): no desc filter for surface, 'preproc' for volumetric."
        ),
    ),
    acq: Optional[str] = typer.Option(
        None,
        "--acq",
        help=(
            "acquisition entity to filter bold files, e.g. 'ME' or 'SE'.  "
            "Default (None): no acq filter — BIDSLayout returns all acquisitions."
        ),
    ),
    n_vols: int = typer.Option(
        0,
        "--n-vols",
        help=(
            "Truncate timeseries to first N volumes before removing start_scans.  "
            "Events.tsv onsets beyond N × TR are also dropped.  "
            "0 = no truncation (default)."
        ),
    ),
    save_betas: bool = typer.Option(
        False,
        "--save-betas",
        help=(
            "Also save per-regressor betas, residuals, and fitted timeseries "
            "(GIFTI for surface, NIfTI for volumetric, each tagged with "
            "space-{space}). Default: only confounds TSV, design matrix, and "
            "contrast stat maps are saved."
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
    bids_dir = op.join(base, input_dirname)
    fsdir = op.join(bids_dir, "derivatives", "freesurfer")
    fmriprep_dir = op.join(bids_dir, "derivatives", f"fmriprep-{fp_ana_name}")
    is_surface = space in ["fsnative", "fsaverage"]

    # Count contrasts early (YAML only, no design matrix needed)
    with open(contrast) as _f:
        _contrast_defs = yaml.safe_load(_f)
    n_contrasts = len(_contrast_defs)

    # Resolve which strategies to run (all by default; one if --strategy is given)
    with open(strategy_yaml) as _f:
        all_strategy_names = list(yaml.safe_load(_f)["strategies"].keys())
    strategy_names = [strategy] if strategy else all_strategy_names

    # Load rerun exclusion map (once, shared across all sessions)
    rerun_excl: dict[tuple[str, str, str], set[str]] = {}
    if rerun_map:
        rerun_excl = _load_rerun_exclusions(rerun_map, acq=acq)

    # ── Launch summary ───────────────────────────────────────────────────────
    console.rule("[bold cyan]GLM Launch[/bold cyan]")
    tbl_launch = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    tbl_launch.add_column("key", style="dim")
    tbl_launch.add_column("value", style="bold")

    mode_str = "[yellow]DRY-RUN[/yellow]" if dry_run else "[green]EXECUTE[/green]"
    runs_str = (
        ", ".join(map(str, selected_runs_list))
        if selected_runs_list
        else "all (from BIDS layout)"
    )
    pairs_str = "  ".join(f"sub-{s} ses-{e}" for s, e in pairs)
    tbl_launch.add_row("Subjects/Sessions", f"({len(pairs)})  {pairs_str}")
    tbl_launch.add_row("Task", task)
    tbl_launch.add_row("Space", space)
    tbl_launch.add_row("fMRIPrep", fp_ana_name)
    tbl_launch.add_row("Analysis name", analysis_name)
    tbl_launch.add_row("Runs", runs_str)
    tbl_launch.add_row("Contrasts", f"{n_contrasts}  ({contrast})")
    tbl_launch.add_row("Start scans", str(start_scans))
    tbl_launch.add_row("Slice time ref", str(slice_time_ref))
    tbl_launch.add_row("Smoothed", f"Yes (sm={sm})" if use_smoothed else "No")
    tbl_launch.add_row("Mask", mask or "—")
    tbl_launch.add_row("Strategy YAML", strategy_yaml)
    tbl_launch.add_row(
        "Strategies",
        ", ".join(strategy_names)
        + (f"  (all {len(all_strategy_names)})" if not strategy else ""),
    )
    tbl_launch.add_row(
        "Rerun map",
        rerun_map if rerun_map else "[dim]— (no exclusions)[/dim]",
    )
    tbl_launch.add_row("Mode", mode_str)
    if power_analysis:
        tbl_launch.add_row(
            "Power analysis",
            f"Yes  ({total_runs} runs × {n_iterations} iter = {total_runs * n_iterations} GLMs)",
        )
    console.print(tbl_launch)
    # ────────────────────────────────────────────────────────────────────────

    # Build BIDS layouts ONCE — reused across all sub/ses pairs
    console.print("Creating BIDS layout …")
    _t = time.time()
    layout = BIDSLayout(bids_dir, validate=False)
    console.print(f"  [dim]BIDS layout ready in {time.time() - _t:.1f} s[/dim]")
    console.print("Creating fMRIPrep layout …")
    _t = time.time()
    fp_layout = BIDSLayout(fmriprep_dir, validate=False)
    console.print(f"  [dim]fMRIPrep layout ready in {time.time() - _t:.1f} s[/dim]")
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
            all_run_list = [f"{r:02d}" for r in range(1, total_runs + 1)]
            console.print("  Prefetching BIDS run metadata for power analysis…")
            _t_fetch = time.time()
            run_metadata = _fetch_bids_run_metadata(
                bids_dir,
                fmriprep_dir,
                task,
                sub,
                ses,
                all_run_list,
                slice_time_ref,
                acq=acq,
                bold_desc=bold_desc,
                bids_layout=layout,
                fp_bids_layout=fp_layout,
            )
            console.print(
                f"  [dim]Metadata prefetched in {time.time() - _t_fetch:.1f} s[/dim]"
            )
            hemis = ["L", "R"] if is_surface else [None]
            for strategy_name in strategy_names:
                confound_strategy = load_confound_strategy(strategy_yaml, strategy_name)
                per_strategy_analysis_name = f"{analysis_name}_{strategy_name}"
                console.rule(
                    f"[bold yellow]Strategy: {strategy_name}[/bold yellow]", style="dim"
                )
                for hemi in hemis:
                    run_power_analysis(
                        bids_dir,
                        fmriprep_dir,
                        fp_layout,
                        label_dir,
                        contrast,
                        sub,
                        ses,
                        per_strategy_analysis_name,
                        task,
                        start_scans,
                        space,
                        slice_time_ref,
                        use_smoothed,
                        sm,
                        mask,
                        dry_run,
                        total_runs,
                        n_iterations,
                        seed,
                        confound_strategy,
                        run_metadata,
                        hemi,
                        bold_desc=bold_desc,
                        n_vols=n_vols,
                        acq=acq,
                        save_betas=save_betas,
                    )
            session_times[(sub, ses)] = time.time() - t_ses
            continue

        # ── Regular mode ─────────────────────────────────────────────────────
        excl_runs = rerun_excl.get((sub, ses, task), set())
        run_list, randrun_idx = generate_run_groups(
            layout,
            sub,
            ses,
            task,
            selected_runs_list,
            excl_runs or None,
            acq=acq,
        )

        # Prefetch events/confounds/TR once — shared across ALL strategies and hemispheres
        console.print("  Prefetching BIDS run metadata (once for all strategies)…")
        _t_fetch = time.time()
        run_metadata = _fetch_bids_run_metadata(
            bids_dir,
            fmriprep_dir,
            task,
            sub,
            ses,
            run_list,
            slice_time_ref,
            acq=acq,
            bold_desc=bold_desc,
            bids_layout=layout,
            fp_bids_layout=fp_layout,
        )
        console.print(
            f"  [dim]Metadata prefetched in {time.time() - _t_fetch:.1f} s[/dim]"
        )

        # n_glm_jobs: split the vertex-level parallelism budget across concurrent strategies
        n_glm_jobs = max(1, 4 // n_workers)

        def _run_one_strategy(strategy_name: str):
            """Run the full GLM pipeline for one strategy. Safe to call from a thread."""
            _t = time.time()
            cs = load_confound_strategy(strategy_yaml, strategy_name)
            per_name = f"{analysis_name}_{strategy_name}"
            console.rule(
                f"[bold yellow]Strategy: {strategy_name}[/bold yellow]  "
                f"→  analysis-{per_name}",
                style="dim",
            )
            t_per_hemi: dict[str, dict[str, float]] = {}
            if is_surface:
                for hemi in ["L", "R"]:
                    console.rule(f"[bold]Hemisphere {hemi}[/bold]", style="dim")
                    t_per_hemi[f"hemi-{hemi}"] = process_run_list(
                        bids_dir,
                        fmriprep_dir,
                        fp_layout,
                        label_dir,
                        contrast,
                        sub,
                        ses,
                        per_name,
                        task,
                        start_scans,
                        space,
                        slice_time_ref,
                        run_list,
                        use_smoothed,
                        sm,
                        mask,
                        dry_run,
                        cs,
                        run_metadata,
                        randrun_idx,
                        hemi,
                        n_glm_jobs,
                        bold_desc=bold_desc,
                        n_vols=n_vols,
                        acq=acq,
                        save_betas=save_betas,
                    )
            else:
                console.rule("[bold]Volumetric[/bold]", style="dim")
                t_per_hemi["volumetric"] = process_run_list(
                    bids_dir,
                    fmriprep_dir,
                    fp_layout,
                    label_dir,
                    contrast,
                    sub,
                    ses,
                    per_name,
                    task,
                    start_scans,
                    space,
                    slice_time_ref,
                    run_list,
                    use_smoothed,
                    sm,
                    mask,
                    dry_run,
                    cs,
                    run_metadata,
                    randrun_idx,
                    hemi=None,
                    n_glm_jobs=n_glm_jobs,
                    bold_desc=bold_desc,
                    n_vols=n_vols,
                    acq=acq,
                    save_betas=save_betas,
                )
            return strategy_name, t_per_hemi, time.time() - _t

        if n_workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            console.print(
                f"  [dim]Running {len(strategy_names)} strategies in parallel "
                f"(n_workers={n_workers}, n_glm_jobs={n_glm_jobs})[/dim]"
            )
            results: dict[str, tuple] = {}
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futs = {pool.submit(_run_one_strategy, sn): sn for sn in strategy_names}
                for fut in as_completed(futs):
                    sn, tph, elapsed = fut.result()
                    results[sn] = (tph, elapsed)
            # Print timing tables in original order after all threads finish
            for sn in strategy_names:
                tph, elapsed = results[sn]
                console.rule(
                    f"[cyan]Strategy: {sn} — Contrast Timing[/cyan]", style="dim"
                )
                if any(v for v in tph.values()):
                    _print_timing_table(tph, elapsed)
                else:
                    console.print(
                        f"  [dim]Dry-run — no outputs written.[/dim]  ({elapsed:.1f} s)"
                    )
        else:
            for sn in strategy_names:
                _, tph, elapsed = _run_one_strategy(sn)
                console.rule(
                    f"[cyan]Strategy: {sn} — Contrast Timing[/cyan]", style="dim"
                )
                if any(v for v in tph.values()):
                    _print_timing_table(tph, elapsed)
                else:
                    console.print(
                        f"  [dim]Dry-run — no outputs written.[/dim]  ({elapsed:.1f} s)"
                    )

        ses_elapsed = time.time() - t_ses
        session_times[(sub, ses)] = ses_elapsed

    # ── Final summary across all sessions ────────────────────────────────────
    console.rule("[bold cyan]Run Summary[/bold cyan]")
    tbl_sum = Table(box=box.SIMPLE_HEAD, show_footer=True)
    tbl_sum.add_column("sub", style="bold", footer="[bold]Total[/bold]")
    tbl_sum.add_column("ses", style="bold")
    tbl_sum.add_column(
        "time (s)",
        justify="right",
        footer=f"[bold]{sum(session_times.values()):.1f}[/bold]",
    )
    tbl_sum.add_column(
        "time (min)",
        justify="right",
        footer=f"[bold]{sum(session_times.values()) / 60:.1f}[/bold]",
    )
    for (s, e), t in session_times.items():
        tbl_sum.add_row(s, e, f"{t:.1f}", f"{t / 60:.1f}")
    console.print(tbl_sum)

    total_elapsed = time.time() - t0
    console.print(
        f"  [bold]Total program time:[/bold]  {total_elapsed:.1f} s  ({total_elapsed / 60:.1f} min)"
    )


if __name__ == "__main__":
    app()
