# -----------------------------------------------------------------------------
# Copyright (c) Yongning Lei 2024-2025
# All rights reserved.
#
# This script is distributed under the Apache-2.0 license.
#
# power_analysis_ohbm.py
# ----------------------
# Cross-session power analysis for OHBM.
#
# 1. Loads ALL sessions for one subject into a single concatenated design
#    matrix (same approach as run_allses_glm.py, with strategy-YAML confounds).
# 2. For n_runs in 1..total_runs, draws n_iter random subsets of runs,
#    fits a GLM on each subset (row-slicing the pre-built design matrix),
#    and records mean T per ROI per contrast.
# 3. Saves: results.npz  summary.tsv  power_plot.png
#
# Storage note: no whole-brain GIFTIs are saved for the 900 iterations — only
# the scalar mean-T values per ROI.  One set of full-brain GIFTIs is saved for
# the "all runs" baseline (max n_runs).
# -----------------------------------------------------------------------------
from __future__ import annotations

import csv
import gc
import importlib.util as _ilu
import json as _json
import os
import os.path as op
import time
from os import makedirs
from typing import Optional

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
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.first_level.first_level import run_glm
from nilearn.plotting import plot_contrast_matrix, plot_design_matrix
from nilearn.surface import load_surf_data
from rich import box
from rich.console import Console
from rich.table import Table
from scipy import stats

console = Console()
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

# ---------------------------------------------------------------------------
# Import shared helpers from glm_surface_check_model_strategy.py
# ---------------------------------------------------------------------------
_THIS_DIR = op.dirname(op.abspath(__file__))
_GLM_STRATEGY = op.join(_THIS_DIR, "..", "glm_strategy", "glm_surface_check_model_strategy.py")
_spec = _ilu.spec_from_file_location("_glm_strategy", _GLM_STRATEGY)
_glm_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_glm_mod)

save_statmap_to_gifti     = _glm_mod.save_statmap_to_gifti
replace_prefix_and_suffix = _glm_mod.replace_prefix_and_suffix
load_contrasts            = _glm_mod.load_contrasts
load_confound_strategy    = _glm_mod.load_confound_strategy
_resolve_confound_columns = _glm_mod._resolve_confound_columns
_load_rerun_exclusions    = _glm_mod._load_rerun_exclusions


# ---------------------------------------------------------------------------
# Session parser
# ---------------------------------------------------------------------------

def _parse_sessions(
    sub: str,
    sessions_arg: Optional[str],
    file_arg: Optional[str],
) -> list[str] | None:
    if sessions_arg:
        return [s.strip().zfill(2) for s in sessions_arg.split(",") if s.strip()]
    if file_arg:
        from pathlib import Path
        path = Path(file_arg)
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        seen: dict[str, None] = {}
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh, delimiter=delimiter):
                if str(row.get("sub", "")).strip().zfill(2) != sub:
                    continue
                if "RUN" in row and str(row["RUN"]).strip() != "True":
                    continue
                seen[str(row["ses"]).strip().zfill(2)] = None
        if not seen:
            console.print(f"[red]ERROR[/red]: no sessions for sub-{sub} in {file_arg}")
            raise typer.Exit(1)
        return list(seen)
    return None


# ---------------------------------------------------------------------------
# ROI loading
# ---------------------------------------------------------------------------

def load_roi_masks(
    roi_yaml_path: str,
    hemi: str,
    label_dir: str,
) -> dict[str, np.ndarray]:
    """Load FreeSurfer label files for each ROI for the given hemisphere.

    Returns {roi_name: integer vertex-index array}.
    """
    with open(roi_yaml_path) as f:
        cfg = yaml.safe_load(f)

    masks: dict[str, np.ndarray] = {}
    for roi_name, paths in cfg.get("rois", {}).items():
        label_file = (paths or {}).get(hemi)
        if not label_file:
            continue
        full_path = op.join(label_dir, label_file)
        if not op.exists(full_path):
            console.print(
                f"  [yellow]WARNING[/yellow]: ROI label not found: {full_path} — skipping {roi_name}"
            )
            continue
        verts = load_surf_data(full_path).astype(int)
        masks[roi_name] = verts
        console.print(
            f"  ROI [cyan]{roi_name}[/cyan] hemi-{hemi}: {len(verts)} vertices  ({full_path})"
        )

    if not masks:
        console.print(f"  [yellow]WARNING[/yellow]: no ROI masks loaded for hemi-{hemi}")
    return masks


# ---------------------------------------------------------------------------
# Data loading: all sessions → one concatenated design matrix
# ---------------------------------------------------------------------------

def prepare_allses_data(
    bids_dir: str,
    fmriprep_dir: str,
    fp_layout: BIDSLayout,
    layout: BIDSLayout,
    contrast_fpath: str,
    subject: str,
    sessions: list[str],
    task: str,
    start_scans: int,
    space: str,
    slice_time_ref: float,
    confound_strategy: dict,
    rerun_excl: dict,
    hemi: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict, list[tuple[int, int]], list[str]]:
    """Load all sessions / runs, z-score, and build the full concatenated design matrix.

    Returns
    -------
    Y_full          : np.ndarray (n_verts × total_TRs)
    X_full          : np.ndarray (total_TRs × n_regs)
    contrasts       : dict[str, np.ndarray]  — contrast vectors over X columns
    run_boundaries  : list[(start_tr, end_tr)] — TR slice per run in Y_full / X_full
    run_labels      : list[str] — human-readable label per run (for diagnostics)
    """
    is_surface = space in ["fsnative", "fsaverage"]

    data_all: list[np.ndarray] = []
    frame_times_all: list[np.ndarray] = []
    events_all: list[pd.DataFrame] = []
    confounds_all: list[pd.DataFrame] = []
    run_boundaries: list[tuple[int, int]] = []
    run_labels: list[str] = []

    total_trs_so_far = 0
    t_r_global: float | None = None
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
            before   = set(run_list)
            run_list = [r for r in run_list if r not in excl_runs]
            excluded = sorted(before - set(run_list))
            if excluded:
                console.print(f"  ses-{ses}: [yellow]excluded compensated runs:[/yellow] {excluded}")

        console.print(f"\n  [bold]ses-{ses}[/bold]  runs: {run_list}")

        for run_num in run_list:
            run_label = f"ses-{ses}_run-{run_num}"
            run_step_times[run_label] = {}

            # ── Load functional data ──────────────────────────────────────────
            _t = time.time()
            query: dict = {
                "subject": subject, "session": ses, "task": task, "run": run_num,
                "space": space, "suffix": "bold",
                "extension": ".func.gii" if is_surface else ".nii.gz",
            }
            if is_surface and hemi:
                query["hemi"] = hemi

            func_files = fp_layout.get(**query)
            # Prefer files without a desc entity (native fMRIprep projection)
            if func_files:
                no_desc = [f for f in func_files if not f.entities.get("desc")]
                if no_desc:
                    func_files = no_desc
            if not func_files:
                console.print(
                    f"  [yellow]WARNING[/yellow]: no func file for {run_label} "
                    f"(query: {query}) — skipping"
                )
                continue

            func_file = func_files[0].path
            console.print(f"  Found: [dim]{func_file}[/dim]")
            if is_surface:
                data_float = np.vstack(load_surf_data(func_file)[:, :]).astype(float)
            else:
                arr = nib.load(func_file).get_fdata()
                data_float = arr.reshape(-1, arr.shape[3]).astype(float)
            run_step_times[run_label]["load_func"] = time.time() - _t

            # ── Z-score + trim ────────────────────────────────────────────────
            _t = time.time()
            data_std = stats.zscore(data_float[:, start_scans:], axis=1).astype(np.float32)
            del data_float
            n_scans = data_std.shape[1]
            run_step_times[run_label]["zscore"] = time.time() - _t

            # ── TR ────────────────────────────────────────────────────────────
            json_files = fp_layout.get(
                subject=subject, session=ses, task=task, run=run_num,
                suffix="bold", extension=".json",
            )
            if not json_files:
                console.print(
                    f"  [yellow]WARNING[/yellow]: no sidecar JSON for {run_label} — skipping"
                )
                continue
            with open(json_files[0].path) as jf:
                t_r = float(_json.load(jf)["RepetitionTime"])
            if t_r_global is None:
                t_r_global = t_r

            # ── Events ────────────────────────────────────────────────────────
            ev_files = layout.get(
                subject=subject, session=ses, task=task, run=run_num,
                suffix="events", extension=".tsv",
            )
            if not ev_files:
                console.print(
                    f"  [yellow]WARNING[/yellow]: no events.tsv for {run_label} — skipping"
                )
                continue
            events = pd.read_csv(ev_files[0].path, sep="\t").copy()
            events.loc[:, "onset"] = events["onset"] + total_trs_so_far * t_r
            events_nobaseline = events[events["trial_type"] != "baseline"]

            # ── Confounds ─────────────────────────────────────────────────────
            _t = time.time()
            conf_files = fp_layout.get(
                subject=subject, session=ses, task=task, run=run_num,
                desc="confounds", suffix="timeseries", extension=".tsv",
            )
            if not conf_files:
                console.print(
                    f"  [yellow]WARNING[/yellow]: no confounds TSV for {run_label} — skipping"
                )
                continue
            confounds = pd.read_csv(conf_files[0].path, sep="\t")
            conf_keys = _resolve_confound_columns(confound_strategy, list(confounds.columns))
            confounds_keep = confounds[conf_keys].copy()
            if "framewise_displacement" in confounds_keep.columns:
                confounds_keep.loc[confounds_keep.index[0], "framewise_displacement"] = (
                    np.nanmean(confounds_keep["framewise_displacement"])
                )
            confounds_keep = confounds_keep.iloc[start_scans:].reset_index(drop=True)
            run_step_times[run_label]["confounds"] = time.time() - _t

            # ── Frame times + accumulate ──────────────────────────────────────
            frame_times = t_r * ((np.arange(n_scans) + slice_time_ref) + total_trs_so_far)

            run_boundaries.append((total_trs_so_far, total_trs_so_far + n_scans))
            run_labels.append(run_label)
            total_trs_so_far += n_scans

            data_all.append(data_std)
            frame_times_all.append(frame_times)
            events_all.append(events_nobaseline)
            confounds_all.append(confounds_keep)

            console.print(
                f"  {run_label}: {n_scans} TRs  "
                f"total_TRs_so_far={total_trs_so_far}  "
                f"[dim](load={run_step_times[run_label]['load_func']:.1f}s "
                f"zscore={run_step_times[run_label]['zscore']:.2f}s)[/dim]"
            )

    if not data_all:
        raise RuntimeError(
            f"No data collected for sub-{subject} over sessions {sessions}."
        )

    # ── Per-run timing summary ────────────────────────────────────────────────
    step_cols = ["load_func", "zscore", "confounds"]
    tbl = Table(title="Per-run step timing (s)", box=box.SIMPLE_HEAD)
    tbl.add_column("ses_run")
    for s in step_cols:
        tbl.add_column(s, justify="right")
    tbl.add_column("total", justify="right")
    for rl, steps in run_step_times.items():
        vals = [steps.get(s, 0.0) for s in step_cols]
        tbl.add_row(rl, *[f"{v:.2f}" for v in vals], f"{sum(vals):.2f}")
    console.print(tbl)

    # ── Build concatenated design matrix ──────────────────────────────────────
    console.print(f"\n  Building design matrix from {len(run_labels)} runs …")
    _t = time.time()

    Y_full = np.concatenate(data_all, axis=1)
    del data_all
    gc.collect()

    concat_frame_times = np.concatenate(frame_times_all)
    concat_events = pd.concat(events_all).applymap(replace_prefix_and_suffix)
    concat_confounds = pd.concat(confounds_all, axis=0)
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
    X_full = np.asarray(design_matrix_std, dtype=np.float32)

    console.print(
        f"  Y_full: {Y_full.shape}  X_full: {X_full.shape}  "
        f"runs: {len(run_boundaries)}  "
        f"[dim]DM build: {time.time() - _t:.2f} s[/dim]"
    )
    return Y_full, X_full, design_matrix_std, contrasts, run_boundaries, run_labels


# ---------------------------------------------------------------------------
# Design-matrix diagnostics (CSV + PNGs)  — same pattern as glm_surface_check_model_strategy.py
# ---------------------------------------------------------------------------

def save_design_matrix_files(
    design_matrix_std: pd.DataFrame,
    contrasts: dict,
    ses_dir: str,
    sub: str,
    task: str,
    hemi: str,
) -> None:
    """Save DM CSV, DM PNG, and one contrast PNG per contrast.

    ses_dir is the allses subdir — matches existing l1_surface layout.
    """
    makedirs(ses_dir, exist_ok=True)

    # design_matrix.png — plain name, matches existing analysis-final_v3_allses
    ax = plot_design_matrix(design_matrix_std)
    fig = ax.get_figure()
    fig.suptitle(f"sub-{sub}  all-sessions  task-{task}  hemi-{hemi}", fontsize=9)
    dm_png = op.join(ses_dir, "design_matrix.png")
    fig.savefig(dm_png, bbox_inches="tight", dpi=120)
    plt.close(fig)
    console.print(f"  [dim]DM PNG  → {dm_png}[/dim]")

    # DM CSV — BIDS-ish name for clarity
    csv_path = op.join(ses_dir, f"sub-{sub}_ses-allses_task-{task}_hemi-{hemi}_design_matrix.csv")
    design_matrix_std.to_csv(csv_path)
    console.print(f"  [dim]DM CSV  → {csv_path}[/dim]")

    for key, vec in contrasts.items():
        ax = plot_contrast_matrix(vec, design_matrix=design_matrix_std)
        fig = ax.get_figure()
        fig.suptitle(f"sub-{sub}  {key}  hemi-{hemi}", fontsize=9)
        c_png = op.join(ses_dir, f"sub-{sub}_ses-allses_task-{task}_hemi-{hemi}_contrast-{key}_contrast_matrix.png")
        fig.savefig(c_png, bbox_inches="tight", dpi=100)
        plt.close(fig)
    console.print(f"  [dim]Contrast PNGs saved ({len(contrasts)} files)[/dim]")


# ---------------------------------------------------------------------------
# Cache helpers  (Y_full is large ~10 GB; X metadata is tiny)
# ---------------------------------------------------------------------------

def _cache_paths(outdir: str, sub: str, hemi: str) -> tuple[str, str]:
    """Return (y_cache_path, x_cache_path) for this subject/hemisphere."""
    y = op.join(outdir, f"sub-{sub}_hemi-{hemi}_Yfull_cache.npy")
    x = op.join(outdir, f"sub-{sub}_hemi-{hemi}_Xmeta_cache.npz")
    return y, x


def save_cache(
    Y_full: np.ndarray,
    X_full: np.ndarray,
    contrasts: dict,
    run_boundaries: list[tuple[int, int]],
    run_labels: list[str],
    outdir: str,
    sub: str,
    hemi: str,
) -> None:
    makedirs(outdir, exist_ok=True)
    y_path, x_path = _cache_paths(outdir, sub, hemi)

    console.print(
        f"  Saving Y_full cache ({Y_full.nbytes / 1e9:.1f} GB) … this may take a moment"
    )
    np.save(y_path, Y_full)
    console.print(f"  [dim]Y cache → {y_path}[/dim]")

    rb_starts = np.array([s for s, _ in run_boundaries], dtype=np.int32)
    rb_ends   = np.array([e for _, e in run_boundaries], dtype=np.int32)
    np.savez_compressed(
        x_path,
        X_full          = X_full,
        contrast_names  = np.array(list(contrasts.keys())),
        contrast_vecs   = np.array(list(contrasts.values()), dtype=np.float32),
        rb_starts       = rb_starts,
        rb_ends         = rb_ends,
        run_labels      = np.array(run_labels),
    )
    console.print(f"  [dim]X cache → {x_path}[/dim]")


def load_cache(
    outdir: str,
    sub: str,
    hemi: str,
) -> tuple[np.ndarray, np.ndarray, dict, list[tuple[int, int]], list[str]] | None:
    """Return (Y_full, X_full, contrasts, run_boundaries, run_labels) or None if cache missing."""
    y_path, x_path = _cache_paths(outdir, sub, hemi)
    if not (op.exists(y_path) and op.exists(x_path)):
        return None

    console.print(f"  [green]Cache found — loading (skips all BOLD/BIDS loading)[/green]")
    console.print(f"  [dim]{y_path}[/dim]")
    console.print(f"  [dim]{x_path}[/dim]")

    _t = time.time()
    Y_full = np.load(y_path)
    xdata  = np.load(x_path, allow_pickle=False)
    X_full = xdata["X_full"]
    contrasts = {
        str(k): v
        for k, v in zip(xdata["contrast_names"], xdata["contrast_vecs"])
    }
    run_boundaries = list(zip(
        xdata["rb_starts"].tolist(),
        xdata["rb_ends"].tolist(),
    ))
    run_labels = xdata["run_labels"].tolist()
    console.print(
        f"  [dim]Cache loaded in {time.time() - _t:.1f} s  "
        f"Y_full={Y_full.shape}  X_full={X_full.shape}[/dim]"
    )
    return Y_full, X_full, contrasts, run_boundaries, run_labels


# ---------------------------------------------------------------------------
# Power analysis loop
# ---------------------------------------------------------------------------

def run_power_loop(
    Y_full: np.ndarray,
    X_full: np.ndarray,
    contrasts: dict,
    run_boundaries: list[tuple[int, int]],
    roi_masks: dict[str, np.ndarray],
    n_iter: int,
    seed: int,
) -> tuple[np.ndarray, list[str], list[str]]:
    """For each n_runs in 1..total_runs, draw n_iter random subsets and compute
    mean T per ROI per contrast.

    Row-slices the pre-built Y_full and X_full — no data reloading, no DM rebuild.

    Returns
    -------
    results        : np.ndarray (total_runs, n_iter, n_contrasts, n_rois)  float32
    contrast_names : list[str]
    roi_names      : list[str]
    """
    total_runs     = len(run_boundaries)
    contrast_names = list(contrasts.keys())
    contrast_vecs  = list(contrasts.values())
    roi_names      = list(roi_masks.keys())
    roi_verts_list = [roi_masks[r] for r in roi_names]
    n_contrasts    = len(contrast_names)
    n_rois         = len(roi_names)

    # Restrict to the union of ROI vertices — run_glm cost scales per-voxel,
    # and only ROI-mean stats are ever read out, so fitting all ~150k
    # vertices per GLM is wasted work. Remap ROI indices into the subset.
    roi_union = np.unique(np.concatenate(roi_verts_list))
    Y_roi = Y_full[roi_union, :]
    roi_verts_list = [np.searchsorted(roi_union, rv) for rv in roi_verts_list]
    console.print(f"  ROI-restricted Y: {len(roi_union)} / {Y_full.shape[0]} vertices")

    results = np.full((total_runs, n_iter, n_contrasts, n_rois), np.nan, dtype=np.float32)

    total_glms = total_runs * n_iter
    done = 0
    iter_times: list[float] = []
    rng = np.random.default_rng(seed)

    console.rule("[bold cyan]Power analysis loop[/bold cyan]")
    console.print(
        f"  {total_runs} run counts × {n_iter} iters = [bold]{total_glms}[/bold] GLMs\n"
        f"  {n_contrasts} contrasts  ×  {n_rois} ROIs  →  {n_contrasts * n_rois} scalars per GLM"
    )

    report_every = max(1, total_glms // 20)

    for n_run_idx in range(total_runs):
        n_runs = n_run_idx + 1
        subsets = [
            sorted(rng.choice(total_runs, size=n_runs, replace=False).tolist())
            for _ in range(n_iter)
        ]

        for iter_idx, selected in enumerate(subsets):
            t_iter = time.time()

            # Row indices in the concatenated timeline for selected runs
            rows = np.concatenate(
                [np.arange(run_boundaries[i][0], run_boundaries[i][1]) for i in selected]
            )
            Y_sub = Y_roi[:, rows].T    # (n_trs, n_verts)  — run_glm expects (n_trs, n_verts)
            X_sub = X_full[rows, :]     # (n_trs, n_regs)

            labels, estimates = run_glm(Y_sub, X_sub)

            for c_idx, c_vec in enumerate(contrast_vecs):
                contrast_obj = compute_contrast(labels, estimates, c_vec)
                t_map = contrast_obj.stat()  # (n_verts,)
                for r_idx, roi_verts in enumerate(roi_verts_list):
                    results[n_run_idx, iter_idx, c_idx, r_idx] = np.nanmean(t_map[roi_verts])

            elapsed = time.time() - t_iter
            iter_times.append(elapsed)
            done += 1

            if done % report_every == 0 or done == total_glms:
                avg = np.mean(iter_times)
                eta = avg * (total_glms - done)
                console.print(
                    f"  [{done}/{total_glms}]  n_runs={n_runs}  iter={iter_idx + 1}  "
                    f"this={elapsed:.2f}s  avg={avg:.2f}s  ETA={eta / 60:.1f} min"
                )

    total_elapsed = sum(iter_times)
    console.print(
        f"\n  [green]Loop done[/green]  {total_glms} GLMs  "
        f"total={total_elapsed / 60:.1f} min  "
        f"avg={total_elapsed / total_glms:.2f} s/GLM"
    )
    return results, contrast_names, roi_names


# ---------------------------------------------------------------------------
# Save all-runs full-brain maps (baseline)
# ---------------------------------------------------------------------------

def save_allruns_maps(
    Y_full: np.ndarray,
    X_full: np.ndarray,
    contrasts: dict,
    ses_dir: str,
    sub: str,
    task: str,
    space: str,
    hemi: str,
) -> None:
    """Fit GLM on ALL runs and save full-brain stat maps (effect + t only).

    Files go into ses_dir (the allses subdir) with names matching
    analysis-final_v3_allses convention: ses-allses, stat-effect / stat-t.
    """
    makedirs(ses_dir, exist_ok=True)
    console.print(f"\n  [bold]Fitting all-runs baseline GLM (hemi-{hemi}) …[/bold]")

    labels, estimates = run_glm(Y_full.T, X_full)

    for contrast_id, contrast_val in contrasts.items():
        contrast_obj = compute_contrast(labels, estimates, contrast_val)
        effect  = contrast_obj.effect_size()
        t_value = contrast_obj.stat()

        for stat_label, arr in [("stat-effect", effect), ("stat-t", t_value)]:
            fname = (
                f"sub-{sub}_ses-allses_task-{task}"
                f"_hemi-{hemi}_space-{space}"
                f"_contrast-{contrast_id}_{stat_label}_statmap.func.gii"
            )
            save_statmap_to_gifti(arr, op.join(ses_dir, fname))

    console.print(f"  All-runs maps saved → {ses_dir}")


# ---------------------------------------------------------------------------
# Save power-analysis results
# ---------------------------------------------------------------------------

def save_results(
    results: np.ndarray,
    contrast_names: list[str],
    roi_names: list[str],
    outdir: str,
    sub: str,
    hemi: str,
    seed: int,
    n_iter: int,
) -> None:
    makedirs(outdir, exist_ok=True)
    total_runs = results.shape[0]
    run_counts = np.arange(1, total_runs + 1)

    # ── Compact NPZ — everything needed to re-plot ────────────────────────────
    npz_path = op.join(outdir, f"sub-{sub}_hemi-{hemi}_power_results.npz")
    np.savez_compressed(
        npz_path,
        mean_t=results,
        run_counts=run_counts,
        roi_names=np.array(roi_names),
        contrast_names=np.array(contrast_names),
        seed=np.array(seed),
    )
    console.print(f"  [dim]NPZ → {npz_path}[/dim]")

    # ── Long-format TSV — human readable, importable into R/Python ────────────
    rows = []
    for n_idx, n_runs in enumerate(run_counts):
        for c_idx, c_name in enumerate(contrast_names):
            for r_idx, roi_name in enumerate(roi_names):
                vals = results[n_idx, :, c_idx, r_idx]
                rows.append({
                    "n_runs":      int(n_runs),
                    "contrast":    c_name,
                    "roi":         roi_name,
                    "hemi":        hemi,
                    "mean_t_mean": float(np.nanmean(vals)),
                    "mean_t_std":  float(np.nanstd(vals)),
                    "mean_t_median": float(np.nanmedian(vals)),
                    "n_iter":      n_iter,
                    "seed":        seed,
                })
    tsv_path = op.join(outdir, f"sub-{sub}_hemi-{hemi}_power_summary.tsv")
    pd.DataFrame(rows).to_csv(tsv_path, sep="\t", index=False)
    console.print(f"  [dim]TSV  → {tsv_path}[/dim]")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_power_results(
    results: np.ndarray,
    contrast_names: list[str],
    roi_names: list[str],
    outdir: str,
    sub: str,
    hemi: str,
) -> None:
    """One subplot per ROI, one line per contrast; x = n_runs, y = mean T ± std."""
    total_runs = results.shape[0]
    run_counts = np.arange(1, total_runs + 1)
    n_rois     = len(roi_names)
    n_contrasts = len(contrast_names)

    ncols = min(n_rois, 3)
    nrows = (n_rois + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, n_contrasts))

    for r_idx, roi_name in enumerate(roi_names):
        ax = axes[r_idx // ncols][r_idx % ncols]
        for c_idx, c_name in enumerate(contrast_names):
            vals = results[:, :, c_idx, r_idx]    # (n_runs, n_iter)
            mu = np.nanmean(vals, axis=1)
            sd = np.nanstd(vals, axis=1)
            ax.plot(run_counts, mu, label=c_name, color=colors[c_idx], lw=1.5)
            ax.fill_between(run_counts, mu - sd, mu + sd, alpha=0.15, color=colors[c_idx])
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_title(f"{roi_name}  (hemi-{hemi})", fontsize=10)
        ax.set_xlabel("n runs")
        ax.set_ylabel("mean T")
        ax.legend(fontsize=6, ncol=2)

    for idx in range(n_rois, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f"sub-{sub}  power analysis — hemi-{hemi}", fontsize=12)
    fig.tight_layout()
    plot_path = op.join(outdir, f"sub-{sub}_hemi-{hemi}_power_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  [dim]Plot → {plot_path}[/dim]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    base: str = typer.Option(
        ..., "--base", help="Base directory, e.g. /scratch/tlei/VOTCLOC"
    ),
    sub: str = typer.Option(..., "--sub", help="Subject label without 'sub-', e.g. 09"),
    sessions_arg: Optional[str] = typer.Option(
        None, "--sessions",
        help="Comma-separated session labels, e.g. '01,02,03,04,05,06,07,08,09'",
    ),
    file_arg: Optional[str] = typer.Option(
        None, "-f",
        help="Subseslist CSV/TSV; rows for --sub (RUN==True filter applied)",
    ),
    fp_ana_name: str = typer.Option(..., "--fp-ana-name", help="fMRIPrep analysis name"),
    task: str = typer.Option(..., "--task", help="Task name, e.g. fLoc"),
    start_scans: int = typer.Option(..., "--start-scans", help="Non-steady-state TRs to drop"),
    space: str = typer.Option(..., "--space", help="Space: fsnative | fsaverage | T1w"),
    contrast: str = typer.Option(..., "--contrast", help="Path to YAML contrast definition file"),
    output_name: str = typer.Option(..., "--output-name", help="Output folder label"),
    roi_yaml: str = typer.Option(..., "--roi-yaml", help="Path to ROI config YAML"),
    strategy_yaml: str = typer.Option(..., "--strategy-yaml", help="Path to confound strategy YAML"),
    strategy: str = typer.Option("basic_MC", "--strategy", help="Strategy name from YAML"),
    n_iter: int = typer.Option(10, "--n-iter", help="Random draws per run count"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    input_dirname: str = typer.Option("BIDS", "--input-dir", "-i"),
    fs_ana_name: str = typer.Option(
        "freesurfer-with_t2", "--fs-ana-name",
        help="FreeSurfer derivatives folder name under BASE/derivatives/",
    ),
    label_subdir: str = typer.Option(
        "manual_label_clusters_analysis_12_v3", "--label-subdir",
        help="Subdirectory under <fsdir>/sub-<sub>/label/ containing the ROI label files",
    ),
    slice_time_ref: float = typer.Option(0.5, "--slice-time-ref"),
    rerun_map: Optional[str] = typer.Option(None, "--rerun-map"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(
        False, "--force",
        help="Ignore existing cache and rebuild Y_full/X_full from scratch",
    ),
    save_allruns: bool = typer.Option(
        False, "--save-allruns",
        help="Also save full-brain GIFTI maps for the all-runs baseline GLM",
    ),
) -> None:
    t0 = time.time()
    sub = sub.strip().zfill(2)

    bids_dir     = op.join(base, input_dirname)
    fsdir        = op.join(base, "derivatives", fs_ana_name)
    fmriprep_dir = op.join(bids_dir, "derivatives", f"fmriprep-{fp_ana_name}")
    label_dir    = op.join(fsdir, f"sub-{sub}", "label", label_subdir)
    is_surface   = space in ["fsnative", "fsaverage"]

    sessions          = _parse_sessions(sub, sessions_arg, file_arg)
    rerun_excl        = _load_rerun_exclusions(rerun_map) if rerun_map else {}
    confound_strategy = load_confound_strategy(strategy_yaml, strategy)

    outdir = op.join(
        base, "derivatives", "power_analysis_ohbm",
        f"analysis-{output_name}", f"sub-{sub}",
    )
    ses_dir = op.join(outdir, "allses")

    # ── Launch summary ────────────────────────────────────────────────────────
    console.rule("[bold cyan]Power Analysis OHBM[/bold cyan]")
    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    tbl.add_column("key",   style="dim")
    tbl.add_column("value", style="bold")
    tbl.add_row("Subject",     f"sub-{sub}")
    tbl.add_row("Sessions",    sessions_arg or (file_arg and f"from {file_arg}") or "auto-detect")
    tbl.add_row("Task",        task)
    tbl.add_row("Space",       space)
    tbl.add_row("fMRIPrep",    fp_ana_name)
    tbl.add_row("FreeSurfer",  fsdir)
    tbl.add_row("Label dir",   label_dir)
    tbl.add_row("Output name", output_name)
    tbl.add_row("ROI config",  roi_yaml)
    tbl.add_row("Strategy",    strategy)
    tbl.add_row("n_iter",      str(n_iter))
    tbl.add_row("Seed",        str(seed))
    tbl.add_row("Output dir",  outdir)
    tbl.add_row(
        "Mode",
        "[yellow]DRY-RUN[/yellow]" if dry_run else "[green]EXECUTE[/green]",
    )
    console.print(tbl)

    hemis = ["L", "R"] if is_surface else [None]

    # ── Skip layouts entirely when every hemisphere is already cached ─────────
    _all_cached = (not force) and all(
        op.exists(_cache_paths(outdir, sub, h)[0]) and op.exists(_cache_paths(outdir, sub, h)[1])
        for h in (hemis if hemis[0] is not None else [])
    )

    layout = fp_layout = None
    if _all_cached:
        console.print(
            "  [green]Cache found for all hemispheres — skipping BIDS/fMRIPrep layout.[/green]\n"
        )
    else:
        # ── Build BIDS layouts (in-memory — no shared on-disk index) ──────────
        # A persisted SQLite index shared across parallel per-subject jobs
        # causes "disk I/O error" / corrupted reads when jobs race to build
        # or read it. Build in-memory per job instead; the design-matrix
        # cache above is what makes repeat runs fast.
        console.print("Creating BIDS layout …")
        _t = time.time()
        layout = BIDSLayout(bids_dir, validate=False)
        console.print(f"  [dim]BIDS ready in {time.time() - _t:.1f} s[/dim]")

        console.print("Creating fMRIPrep layout …")
        _t = time.time()
        fp_layout = BIDSLayout(fmriprep_dir, validate=False)
        console.print(f"  [dim]fMRIPrep ready in {time.time() - _t:.1f} s[/dim]\n")

        # ── Auto-detect sessions if not supplied ──────────────────────────────
        if sessions is None:
            ext  = ".func.gii" if is_surface else ".nii.gz"
            raw  = sorted(s.zfill(2) for s in layout.get_sessions(subject=sub, task=task))
            sessions = []
            for s in raw:
                func_dir = op.join(fmriprep_dir, f"sub-{sub}", f"ses-{s}", "func")
                if op.isdir(func_dir) and any(
                    task in f and f.endswith(ext) for f in os.listdir(func_dir)
                ):
                    sessions.append(s)
            if not sessions:
                console.print(f"[red]ERROR[/red]: no sessions with fMRIprep data for sub-{sub}")
                raise typer.Exit(1)
            console.print(
                f"  Auto-detected {len(sessions)} sessions: "
                f"{', '.join('ses-' + s for s in sessions)}\n"
            )

    for hemi in hemis:
        hemi_label = f"hemi-{hemi}" if hemi else "volumetric"
        console.rule(f"[bold magenta]{hemi_label}[/bold magenta]")

        # ── Load ROI masks ────────────────────────────────────────────────────
        if is_surface and hemi:
            roi_masks = load_roi_masks(roi_yaml, hemi, label_dir)
            if not roi_masks:
                console.print(f"  [yellow]No ROI masks for {hemi_label} — skipping[/yellow]")
                continue
        else:
            console.print("  [yellow]Volumetric ROI masking not implemented — skipping[/yellow]")
            continue

        # ── Load data + build design matrix (or restore from cache) ──────────
        makedirs(outdir, exist_ok=True)
        cached = None if force else load_cache(outdir, sub, hemi)

        if cached is not None:
            Y_full, X_full, contrasts, run_boundaries, run_labels = cached
        else:
            if force:
                console.print("  [yellow]--force: ignoring cache, rebuilding from scratch[/yellow]")
            Y_full, X_full, design_matrix_std, contrasts, run_boundaries, run_labels = (
                prepare_allses_data(
                    bids_dir=bids_dir,
                    fmriprep_dir=fmriprep_dir,
                    fp_layout=fp_layout,
                    layout=layout,
                    contrast_fpath=contrast,
                    subject=sub,
                    sessions=sessions,
                    task=task,
                    start_scans=start_scans,
                    space=space,
                    slice_time_ref=slice_time_ref,
                    confound_strategy=confound_strategy,
                    rerun_excl=rerun_excl,
                    hemi=hemi,
                )
            )
            # Save DM diagnostics into allses/ (matches l1_surface layout)
            save_design_matrix_files(design_matrix_std, contrasts, ses_dir, sub, task, hemi)
            del design_matrix_std

            # Save big cache so next run skips all BOLD/BIDS loading
            save_cache(Y_full, X_full, contrasts, run_boundaries, run_labels, outdir, sub, hemi)

        # ── Restrict to the contrasts requested via --contrast ────────────────
        # A cached run may carry contrasts from a previous --contrast file
        # (contrast vectors only depend on design-matrix columns, which are
        # unchanged, so cached vectors for these names remain valid).
        with open(contrast) as _f:
            wanted = list(yaml.safe_load(_f).keys())
        missing = [c for c in wanted if c not in contrasts]
        if missing:
            console.print(
                f"  [red]ERROR[/red]: contrasts {missing} not found in cache for "
                f"sub-{sub} hemi-{hemi}. Re-run with --force to rebuild the cache."
            )
            raise typer.Exit(1)
        contrasts = {c: contrasts[c] for c in wanted}

        total_runs = len(run_boundaries)
        console.print(
            f"\n  [bold]{total_runs}[/bold] runs  "
            f"Y_full={Y_full.shape}  X_full={X_full.shape}"
        )

        if dry_run:
            console.print(
                "  [dim]Dry-run — data loaded and design matrix built; not running GLM loop.[/dim]"
            )
            del Y_full, X_full, contrasts
            gc.collect()
            continue

        # ── Optional: all-runs full-brain baseline ────────────────────────────
        if save_allruns:
            save_allruns_maps(Y_full, X_full, contrasts, ses_dir, sub, task, space, hemi)

        # ── Power analysis loop ───────────────────────────────────────────────
        results, c_names, r_names = run_power_loop(
            Y_full, X_full, contrasts, run_boundaries, roi_masks, n_iter, seed,
        )
        del Y_full, X_full, contrasts
        gc.collect()

        # ── Save results ──────────────────────────────────────────────────────
        save_results(results, c_names, r_names, outdir, sub, hemi, seed, n_iter)
        plot_power_results(results, c_names, r_names, outdir, sub, hemi)

    total_elapsed = time.time() - t0
    console.rule("[bold cyan]Done[/bold cyan]")
    console.print(
        f"  [bold]Total wall time:[/bold] {total_elapsed:.1f} s  "
        f"({total_elapsed / 60:.1f} min)"
    )
    console.print(f"  Output: [bold]{outdir}[/bold]")


if __name__ == "__main__":
    app()
