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
# roi_psc_analysis.py
# -------------------
# ROI-level percent signal change (PSC) from GLM stat-effect maps.
#
# The GLM in glm_surface_*.py z-scores each vertex's timeseries before fitting:
#
#   z = (x - μ) / σ
#
# So stat-effect betas (β_z) are in z-score units.  To recover BOLD PSC:
#
#   PSC = β_z × (σ_BOLD / μ_BOLD) × 100
#
# This script loads the original fMRIPrep BOLD runs (to compute σ and μ per
# vertex), loads the stat-effect GIFTI from the GLM output, applies an ROI
# mask, and summarises PSC within the ROI.
#
# ROI can be defined as:
#   --roi-gifti  path/to/hemi-{hemi}_roi.func.gii  (binary 0/1 GIFTI, one per hemi)
#   --roi-label  path/to/hemi-{hemi}.label          (FreeSurfer .label file)
#   --roi-thresh <float>  applied to a --roi-stat-gifti to threshold a stat map
#
# Usage examples:
#   # Single session, binary GIFTI ROI
#   python roi_psc_analysis.py \
#       --base /bcbl/home/public/Gari/VOTCLOC/main_exp \
#       --fp-ana-name 25.1.4_t2w_fmapsbref_newest \
#       --analysis-name final_v3_basic_MC \
#       --task fLoc --space fsnative --start-scans 6 \
#       --contrast RWvsFF \
#       --roi-gifti /path/to/roi/hemi-{hemi}_VOTC.func.gii \
#       -s 10,06 \
#       --out-tsv results/psc_VOTC.tsv
#
#   # Batch from file, FreeSurfer label ROI
#   python roi_psc_analysis.py \
#       --base /bcbl/home/public/Gari/VOTCLOC/main_exp \
#       --fp-ana-name 25.1.4_t2w_fmapsbref_newest \
#       --analysis-name final_v3_basic_MC \
#       --task fLoc --space fsnative --start-scans 6 \
#       --contrast RWvsFF \
#       --roi-label /path/to/freesurfer/sub-{sub}/label/hemi-{hemi}_VOTC.label \
#       -f subseslist_HMC.txt \
#       --out-tsv results/psc_VOTC.tsv
# -----------------------------------------------------------------------------
from __future__ import annotations

import csv
import os.path as op
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import typer
from bids import BIDSLayout
from nilearn.surface import load_surf_data
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

HEMIS = ["L", "R"]


# ---------------------------------------------------------------------------
# ROI loading
# ---------------------------------------------------------------------------


def load_roi_mask(
    n_vertices: int,
    roi_gifti: Optional[str],
    roi_label: Optional[str],
    roi_stat_gifti: Optional[str],
    roi_thresh: Optional[float],
    hemi: str,
    sub: str,
) -> np.ndarray:
    """Return a boolean array of shape (n_vertices,) marking ROI vertices."""

    def _resolve(template: str) -> str:
        return template.format(hemi=hemi, sub=sub)

    if roi_gifti:
        path = _resolve(roi_gifti)
        data = load_surf_data(path).astype(float).ravel()
        if data.shape[0] != n_vertices:
            raise ValueError(
                f"ROI GIFTI has {data.shape[0]} vertices, expected {n_vertices} (hemi-{hemi})"
            )
        return data > 0

    if roi_label:
        path = _resolve(roi_label)
        # FreeSurfer .label: first line is comment, second is count, rest are
        # "vertex_idx  R  A  S  value"
        with open(path) as fh:
            lines = fh.read().strip().splitlines()
        # skip comment (#...) and count lines
        vertex_lines = [ln for ln in lines if not ln.startswith("#")]
        if not vertex_lines:
            raise ValueError(f"Empty label file: {path}")
        try:
            int(vertex_lines[0].strip())
            vertex_lines = vertex_lines[1:]  # first non-comment line was the count
        except ValueError:
            pass
        indices = np.array([int(ln.split()[0]) for ln in vertex_lines if ln.strip()])
        mask = np.zeros(n_vertices, dtype=bool)
        mask[indices] = True
        return mask

    if roi_stat_gifti and roi_thresh is not None:
        path = _resolve(roi_stat_gifti)
        data = load_surf_data(path).astype(float).ravel()
        if data.shape[0] != n_vertices:
            raise ValueError(
                f"ROI stat GIFTI has {data.shape[0]} vertices, expected {n_vertices}"
            )
        return data >= roi_thresh

    raise ValueError(
        "Provide one of --roi-gifti, --roi-label, or --roi-stat-gifti + --roi-thresh"
    )


# ---------------------------------------------------------------------------
# BOLD mean/std from fMRIPrep runs
# ---------------------------------------------------------------------------


def compute_bold_mean_std(
    fp_layout: BIDSLayout,
    subject: str,
    session: str,
    task: str,
    space: str,
    hemi: str,
    start_scans: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all fMRIPrep BOLD runs for the sub/ses/task/hemi and compute
    per-vertex mean and std across the concatenated timeseries (after
    removing start_scans non-steady-state TRs).

    Returns
    -------
    mean_bold, std_bold : np.ndarray, shape (n_vertices,)
    """
    query = {
        "subject": subject,
        "session": session,
        "task": task,
        "hemi": hemi,
        "space": space,
        "suffix": "bold",
        "extension": ".func.gii",
    }
    files = fp_layout.get(**query, invalid_filters="allow")
    if not files:
        raise FileNotFoundError(
            f"No BOLD files found for sub-{subject} ses-{session} hemi-{hemi}: {query}"
        )

    all_data = []
    for f in sorted(files, key=lambda x: x.path):
        d = load_surf_data(f.path).astype(float)  # (n_vertices, n_timepoints)
        if d.ndim == 1:
            d = d[:, np.newaxis]
        d = d[:, start_scans:]
        all_data.append(d)

    concat = np.concatenate(all_data, axis=1)  # (n_vertices, total_timepoints)
    mean_bold = np.mean(concat, axis=1)
    std_bold = np.std(concat, axis=1, ddof=1)

    console.print(
        f"  [dim]BOLD loaded: {len(files)} run(s), {concat.shape[1]} TPs, "
        f"{concat.shape[0]} vertices (hemi-{hemi})[/dim]"
    )
    return mean_bold, std_bold


# ---------------------------------------------------------------------------
# PSC computation
# ---------------------------------------------------------------------------


def compute_psc_from_effect(
    effect: np.ndarray,
    mean_bold: np.ndarray,
    std_bold: np.ndarray,
) -> np.ndarray:
    """
    Convert z-scored GLM effect (β_z) to PSC.

    PSC = β_z × (σ_BOLD / μ_BOLD) × 100

    Vertices where mean_bold ≈ 0 (background / masked) are set to NaN.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(np.abs(mean_bold) > 1e-6, std_bold / mean_bold, np.nan)
    return effect * cv * 100.0


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _save_gifti(data: np.ndarray, path: str) -> None:
    gii = nib.gifti.gifti.GiftiImage()
    gii.add_gifti_data_array(
        nib.gifti.gifti.GiftiDataArray(
            data=data.astype(np.float32), datatype="NIFTI_TYPE_FLOAT32"
        )
    )
    nib.save(gii, path)


def _plot_psc_histogram(psc_roi: np.ndarray, outpath: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(psc_roi[~np.isnan(psc_roi)], bins=30, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("PSC (%)")
    ax.set_ylabel("Vertices")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sub/ses pair parsing (shared pattern across the project)
# ---------------------------------------------------------------------------


def _parse_pairs(
    subses_arg: Optional[str],
    file_arg: Optional[str],
) -> list[tuple[str, str]]:
    if subses_arg:
        parts = subses_arg.split(",")
        if len(parts) != 2:
            console.print(
                f"[red]ERROR[/red]: -s expects 'sub,ses', got: {subses_arg!r}"
            )
            raise typer.Exit(1)
        return [(parts[0].strip().zfill(2), parts[1].strip().zfill(2))]

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

    console.print("[red]ERROR[/red]: provide -s <sub,ses> or -f <subseslist>")
    raise typer.Exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    base: str = typer.Option(..., "--base", help="Project base dir (contains BIDS/)"),
    fp_ana_name: str = typer.Option(
        ..., "--fp-ana-name", help="fMRIPrep analysis name"
    ),
    analysis_name: str = typer.Option(
        ..., "--analysis-name", help="GLM analysis name (folder label)"
    ),
    task: str = typer.Option(..., "--task", help="Task name, e.g. fLoc"),
    space: str = typer.Option(..., "--space", help="Space: fsnative | fsaverage | T1w"),
    start_scans: int = typer.Option(
        ..., "--start-scans", help="Non-steady-state TRs dropped in the GLM"
    ),
    contrast: str = typer.Option(..., "--contrast", help="Contrast name, e.g. RWvsFF"),
    roi_name: str = typer.Option(
        "ROI", "--roi-name", help="Label for the ROI (used in output)"
    ),
    # ROI definition (exactly one required)
    roi_gifti: Optional[str] = typer.Option(
        None,
        "--roi-gifti",
        help="Binary GIFTI mask; use {hemi} and/or {sub} placeholders, e.g. /path/hemi-{hemi}_roi.func.gii",
    ),
    roi_label: Optional[str] = typer.Option(
        None,
        "--roi-label",
        help="FreeSurfer .label file; supports {hemi} and {sub} placeholders",
    ),
    roi_stat_gifti: Optional[str] = typer.Option(
        None,
        "--roi-stat-gifti",
        help="Stat-map GIFTI to threshold into ROI; use {hemi}/{sub} placeholders",
    ),
    roi_thresh: Optional[float] = typer.Option(
        None,
        "--roi-thresh",
        help="Threshold for --roi-stat-gifti (e.g. 3.0 for t > 3)",
    ),
    # sub/ses
    subses_arg: Optional[str] = typer.Option(
        None, "-s", help="Single sub,ses, e.g. 10,06"
    ),
    file_arg: Optional[str] = typer.Option(
        None, "-f", help="Path to subseslist CSV/TSV"
    ),
    # output
    out_tsv: str = typer.Option(
        "psc_roi_results.tsv", "--out-tsv", help="Output TSV path"
    ),
    save_psc_gifti: bool = typer.Option(
        False,
        "--save-psc-gifti",
        help="Save whole-surface PSC map as GIFTI alongside the stat-effect files",
    ),
    save_histogram: bool = typer.Option(
        False,
        "--save-histogram",
        help="Save a per-session ROI PSC histogram PNG",
    ),
    input_dirname: str = typer.Option(
        "BIDS", "--input-dir", help="BIDS dir name under base"
    ),
) -> None:

    pairs = _parse_pairs(subses_arg, file_arg)

    bids_dir = op.join(base, input_dirname)
    fmriprep_dir = op.join(bids_dir, "derivatives", f"fmriprep-{fp_ana_name}")
    glm_dir = op.join(
        bids_dir, "derivatives", "l1_surface", f"analysis-{analysis_name}"
    )
    is_surface = space in ["fsnative", "fsaverage"]

    console.rule("[bold cyan]ROI PSC Analysis[/bold cyan]")
    console.print(
        f"  Analysis : [bold]{analysis_name}[/bold]\n"
        f"  Contrast : [bold]{contrast}[/bold]\n"
        f"  ROI      : [bold]{roi_name}[/bold]\n"
        f"  Space    : {space}  |  Task: {task}  |  Start scans: {start_scans}\n"
        f"  Sessions : {len(pairs)}\n"
    )

    console.print("Building fMRIPrep layout …")
    fp_layout = BIDSLayout(fmriprep_dir, validate=False)
    console.print("[green]Layout ready.[/green]\n")

    records: list[dict] = []

    for sub, ses in pairs:
        console.rule(f"[magenta]sub-{sub}  ses-{ses}[/magenta]")
        ses_dir = op.join(glm_dir, f"sub-{sub}", f"ses-{ses}")

        hemis = HEMIS if is_surface else [None]
        for hemi in hemis:
            hemi_label = f"hemi-{hemi}" if hemi else "volumetric"
            console.print(f"  [bold]{hemi_label}[/bold]")

            # ── 1. Load GLM stat-effect map ──────────────────────────────────
            if hemi:
                effect_fname = (
                    f"sub-{sub}_ses-{ses}_task-{task}"
                    f"_hemi-{hemi}_space-{space}_contrast-{contrast}"
                    f"_stat-effect_statmap.func.gii"
                )
            else:
                console.print(
                    "  [yellow]WARNING[/yellow]: volumetric PSC not implemented — skipping"
                )
                continue

            effect_path = op.join(ses_dir, effect_fname)
            if not op.exists(effect_path):
                console.print(
                    f"  [yellow]WARNING[/yellow]: stat-effect not found: {effect_path}"
                )
                continue

            effect = load_surf_data(effect_path).astype(float).ravel()
            n_vertices = effect.shape[0]
            console.print(f"  Effect map: {n_vertices} vertices  ({effect_fname})")

            # ── 2. Load ROI mask ─────────────────────────────────────────────
            try:
                roi_mask = load_roi_mask(
                    n_vertices,
                    roi_gifti,
                    roi_label,
                    roi_stat_gifti,
                    roi_thresh,
                    hemi,
                    sub,
                )
            except (FileNotFoundError, ValueError) as e:
                console.print(f"  [red]ERROR[/red] loading ROI: {e}")
                continue

            n_roi = int(roi_mask.sum())
            console.print(f"  ROI size: {n_roi} vertices")
            if n_roi == 0:
                console.print("  [yellow]WARNING[/yellow]: empty ROI — skipping")
                continue

            # ── 3. Load BOLD to compute mean / std ───────────────────────────
            try:
                mean_bold, std_bold = compute_bold_mean_std(
                    fp_layout, sub, ses, task, space, hemi, start_scans
                )
            except FileNotFoundError as e:
                console.print(f"  [red]ERROR[/red]: {e}")
                continue

            # ── 4. Compute PSC ───────────────────────────────────────────────
            psc_full = compute_psc_from_effect(effect, mean_bold, std_bold)
            psc_roi = psc_full[roi_mask]

            psc_mean = float(np.nanmean(psc_roi))
            psc_std = float(np.nanstd(psc_roi, ddof=1))
            psc_median = float(np.nanmedian(psc_roi))
            n_valid = int(np.sum(~np.isnan(psc_roi)))

            console.print(
                f"  PSC  mean={psc_mean:.3f}%  std={psc_std:.3f}%  "
                f"median={psc_median:.3f}%  valid_vertices={n_valid}"
            )

            records.append(
                {
                    "sub": sub,
                    "ses": ses,
                    "hemi": hemi,
                    "contrast": contrast,
                    "roi": roi_name,
                    "space": space,
                    "n_roi_vertices": n_roi,
                    "n_valid_vertices": n_valid,
                    "PSC_mean": round(psc_mean, 4),
                    "PSC_std": round(psc_std, 4),
                    "PSC_median": round(psc_median, 4),
                }
            )

            # ── Optional outputs ─────────────────────────────────────────────
            if save_psc_gifti:
                psc_outname = effect_path.replace("_stat-effect_", "_stat-PSC_")
                _save_gifti(psc_full.astype(np.float32), psc_outname)
                console.print(f"  [dim]PSC GIFTI → {op.basename(psc_outname)}[/dim]")

            if save_histogram:
                hist_outname = effect_path.replace(
                    "_stat-effect_statmap.func.gii",
                    f"_stat-PSC_roi-{roi_name}_histogram.png",
                )
                _plot_psc_histogram(
                    psc_roi,
                    hist_outname,
                    f"sub-{sub} ses-{ses} hemi-{hemi} — {contrast} PSC in {roi_name}",
                )
                console.print(f"  [dim]Histogram → {op.basename(hist_outname)}[/dim]")

    # ── Save results ─────────────────────────────────────────────────────────
    if not records:
        console.print("[yellow]No records to save.[/yellow]")
        raise typer.Exit(0)

    out_path = Path(out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(out_path, sep="\t", index=False)
    console.print(f"\n[green]Results saved → {out_path}[/green]  ({len(df)} rows)")

    # Summary table
    console.rule("[cyan]Summary[/cyan]")
    tbl = Table(box=box.SIMPLE_HEAD)
    for col in df.columns:
        tbl.add_column(col)
    for _, row in df.iterrows():
        tbl.add_row(*[str(v) for v in row])
    console.print(tbl)


if __name__ == "__main__":
    app()
