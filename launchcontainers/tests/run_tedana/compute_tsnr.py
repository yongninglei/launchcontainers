#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) Yongning Lei 2025-2026
# Apache-2.0 license
# -----------------------------------------------------------------------------
"""
compute_tsnr.py

Compare tSNR between ME_denoised, ME_optcom, and SE data in T1w space.

Sequences compared
------------------
  ME_denoised : tedana output  …_desc-denoised_bold.nii.gz   (native BOLD space)
  ME_optcom   : tedana output  …_desc-optcom_bold.nii.gz     (native BOLD space)
  SE          : fmriprep       …_acq-SE_space-T1w_desc-preproc_bold.nii.gz

tSNR computed per run (quadratic detrending + MAD variant), then averaged
across runs.  All maps are saved as NIfTI and plotted with jet colormap.

Figures
-------
  {out}/figures/{sub}_{ses}_{seq}_per_run.png   — one ortho panel per run
  {out}/figures/{sub}_{ses}_comparison.png      — mean-across-runs, all seqs
  {out}/figures/{sub}_{ses}_summary_barplot.png — median tSNR bar chart
  {out}/tsnr_summary.csv                        — numeric summary

Usage
-----
    python compute_tsnr.py \\
        -b /data/BIDS \\
        -tedana tedana-26.0.3 \\
        -n default_ica \\
        -fp /data/BIDS/derivatives/fmriprep-25.1.4 \\
        -s pilot02,01 \\
        --tasks BfLocVideo \\
        -o /data/results/tsnr_comparison
"""

from __future__ import annotations

import glob
import os
import os.path as op
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from scipy.ndimage import center_of_mass

console = Console()
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


# ---------------------------------------------------------------------------
# tSNR computation  (logic from 01_tSNR.py — VOTCLOC)
# ---------------------------------------------------------------------------


def _detrend_quadratic(data_2d: np.ndarray) -> np.ndarray:
    n_voxels, n_tp = data_2d.shape
    t = np.linspace(-1, 1, n_tp, dtype=np.float64)
    X = np.column_stack([t**d for d in (0, 1, 2)])
    betas, _, _, _ = np.linalg.lstsq(X, data_2d.T, rcond=None)
    return (data_2d.T - X @ betas).T


def _brain_mask(mean_vol: np.ndarray) -> np.ndarray:
    p99 = np.percentile(mean_vol.ravel(), 99)
    return mean_vol > (0.1 * p99)


def compute_tsnr(data: np.ndarray) -> dict:
    """Compute tSNR metrics from a 4D array (X, Y, Z, T).

    Matches 01_tSNR.py exactly: brain mask = mean > 0.1 × p99(mean),
    used only for the median scalar — the full map is returned unmasked.
    """
    x, y, z, t = data.shape
    data = data.astype(np.float32)
    mean_vol = data.mean(axis=3)
    mask = _brain_mask(mean_vol)

    data_2d = data.reshape(-1, t)
    mean_1d = mean_vol.reshape(-1)
    mask_1d = mask.reshape(-1)

    # Standard tSNR (quadratic detrending)
    det = _detrend_quadratic(data_2d)
    std_1d = det.std(axis=1, ddof=0)
    tsnr_std = np.full(len(mean_1d), np.nan, dtype=np.float32)
    ok = std_1d > 0
    tsnr_std[ok] = mean_1d[ok] / std_1d[ok]
    tsnr_std = tsnr_std.reshape(x, y, z)

    # MAD-based tSNR
    mad_1d = np.mean(np.abs(np.diff(data_2d, axis=1)), axis=1)
    tsnr_mad = np.full(len(mean_1d), np.nan, dtype=np.float32)
    ok2 = mad_1d > 0
    tsnr_mad[ok2] = mean_1d[ok2] / mad_1d[ok2]
    tsnr_mad = tsnr_mad.reshape(x, y, z)

    return {
        "mean_vol": mean_vol,
        "mask": mask,
        "tsnr_std": tsnr_std,
        "tsnr_mad": tsnr_mad,
        "median_std": float(np.nanmedian(tsnr_std[mask])),
        "median_mad": float(np.nanmedian(tsnr_mad[mask])),
    }


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def _find_files(base_dir: str, pattern: str) -> list[tuple[str, str]]:
    """Return sorted [(run_label, path), ...] matching glob pattern."""
    found = sorted(glob.glob(op.join(base_dir, pattern)))
    pairs = []
    for f in found:
        m = re.search(r"_run-(\w+)[_.]", op.basename(f))
        pairs.append((m.group(1) if m else "01", f))
    return pairs


def _find_mask_path(base_dir: str, pattern: str) -> str | None:
    found = sorted(glob.glob(op.join(base_dir, pattern)))
    return found[0] if found else None


def _get_sequences(
    sub: str,
    ses: str,
    task: str,
    tedana_func: str,
    fp_func: str,
) -> dict[str, dict]:
    """Return {seq_name: {files, mask_pattern, mask_dir}} for the three sequences.

    mask_pattern is used to find a brain mask to apply to the raw data before
    computing tSNR — non-brain voxels are zeroed out so the 0.1×p99 threshold
    in compute_tsnr() operates only on brain tissue.
    """
    return {
        "ME_denoised": {
            "files": _find_files(
                tedana_func,
                f"sub-{sub}_ses-{ses}_task-{task}_acq-ME_run-*_desc-denoised_bold.nii.gz",
            ),
            "mask_pattern": f"sub-{sub}_ses-{ses}_task-{task}_acq-ME_run-{{run}}_desc-adaptiveGoodSignal_mask.nii.gz",
            "mask_dir": tedana_func,
        },
        "ME_optcom": {
            "files": _find_files(
                tedana_func,
                f"sub-{sub}_ses-{ses}_task-{task}_acq-ME_run-*_desc-optcom_bold.nii.gz",
            ),
            "mask_pattern": f"sub-{sub}_ses-{ses}_task-{task}_acq-ME_run-{{run}}_desc-adaptiveGoodSignal_mask.nii.gz",
            "mask_dir": tedana_func,
        },
        "SE": {
            "files": _find_files(
                fp_func,
                f"sub-{sub}_ses-{ses}_task-{task}_acq-SE_run-*_space-T1w_desc-preproc_bold.nii.gz",
            ),
            "mask_pattern": f"sub-{sub}_ses-{ses}_task-{task}_acq-SE_run-{{run}}_space-T1w_desc-brain_mask.nii.gz",
            "mask_dir": fp_func,
        },
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

# Sequence display colours (for bar chart)
SEQ_COLORS = {
    "ME_denoised": "#2196F3",  # blue
    "ME_optcom": "#FF9800",  # orange
    "SE": "#4CAF50",  # green
}


def _com_slices(mask: np.ndarray) -> tuple[int, int, int]:
    """Center-of-mass voxel coordinates within the brain mask."""
    com = center_of_mass(mask.astype(float))
    return tuple(int(round(c)) for c in com)


def _apply_mask_nan(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Set non-brain voxels to NaN for cleaner display."""
    out = vol.astype(np.float32).copy()
    out[~mask] = np.nan
    return out


def _plot_ortho_panels(
    axes: list,  # list of 3 matplotlib Axes
    tsnr_vol: np.ndarray,
    mask: np.ndarray,  # used only for center-of-mass slice selection
    vmin: float,
    vmax: float,
    title: str,
    median: float,
) -> None:
    """Fill three axes with axial / coronal / sagittal tSNR slices.

    The mask is only used to find the brain centre-of-mass for slice
    selection — the tSNR map is plotted unmasked, matching the original
    01_tSNR.py approach.
    """
    ci, cj, ck = _com_slices(mask)
    vol = tsnr_vol  # plot the full map; background zeros show as dark blue
    kw = dict(
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        aspect="equal",
        interpolation="nearest",
    )

    axial = axes[0].imshow(np.rot90(vol[:, :, ck]), **kw)
    axes[0].set_title(f"Axial  z={ck}", fontsize=8)

    axes[1].imshow(np.rot90(vol[:, cj, :]), **kw)
    axes[1].set_title(f"Coronal  y={cj}", fontsize=8)

    axes[2].imshow(np.rot90(vol[ci, :, :]), **kw)
    axes[2].set_title(f"Sagittal  x={ci}", fontsize=8)

    for ax in axes:
        ax.axis("off")

    axes[0].set_xlabel(f"{title}\nmedian={median:.1f}", fontsize=8)

    # Colorbar on rightmost axis
    plt.colorbar(axial, ax=axes[2], fraction=0.046, pad=0.04, label="tSNR")


# ---------------------------------------------------------------------------
# Core processing per subject/session
# ---------------------------------------------------------------------------


def _process_subses(
    sub: str,
    ses: str,
    task: str,
    tedana_func: str,
    fp_func: str,
    out_dir: str,
    save_nifti: bool,
    n_vols: int = 213,
) -> pd.DataFrame:
    seqs = _get_sequences(sub, ses, task, tedana_func, fp_func)

    rows = []
    seq_results: dict[str, list] = {}  # seq → [{run, metrics, img, ...}]
    all_tsnr_values: list[float] = []

    fig_dir = op.join(out_dir, "figures")
    nii_dir = op.join(out_dir, "tsnr_niftis")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(nii_dir, exist_ok=True)

    # ── Pass 1: compute tSNR for every run ────────────────────────────────
    for seq_name, seq_cfg in seqs.items():
        run_files = seq_cfg["files"]
        if not run_files:
            console.print(f"  [yellow]⚠ {seq_name}: no files found — skipping[/yellow]")
            continue
        console.print(f"  [bold]{seq_name}[/bold]  ({len(run_files)} run(s))")
        seq_results[seq_name] = []

        for run_label, path in run_files:
            console.print(f"    run-{run_label}  {op.basename(path)}")
            img = nib.load(path)
            data = img.get_fdata(dtype=np.float32)
            if n_vols > 0:
                data = data[:, :, :, :n_vols]

            # Apply brain mask to data before computing tSNR so that non-brain
            # voxels (skull, neck, coil) are zeroed out.  The 0.1×p99 threshold
            # inside compute_tsnr() then operates on brain tissue only.
            mask_pat = seq_cfg["mask_pattern"].format(run=run_label)
            mask_path = _find_mask_path(seq_cfg["mask_dir"], mask_pat)
            if mask_path:
                brain_mask = nib.load(mask_path).get_fdata().astype(bool)
                data[~brain_mask] = 0.0
                console.print(f"      Mask applied: {op.basename(mask_path)}")
            else:
                console.print(
                    "      [yellow]⚠ mask not found — computing on full FOV[/yellow]"
                )

            metrics = compute_tsnr(data)
            all_tsnr_values.append(metrics["median_std"])
            console.print(
                f"      median tSNR (std)={metrics['median_std']:.1f}  "
                f"(mad)={metrics['median_mad']:.1f}"
            )

            if save_nifti:
                stem = op.basename(path).replace("_bold.nii.gz", "")
                for key, arr in [
                    ("tsnr_std", metrics["tsnr_std"]),
                    ("tsnr_mad", metrics["tsnr_mad"]),
                ]:
                    nib.save(
                        nib.Nifti1Image(arr, img.affine, img.header),
                        op.join(nii_dir, f"{stem}_desc-{key}.nii.gz"),
                    )

            rows.append(
                {
                    "sub": sub,
                    "ses": ses,
                    "task": task,
                    "seq": seq_name,
                    "run": run_label,
                    "median_tsnr_std": metrics["median_std"],
                    "median_tsnr_mad": metrics["median_mad"],
                    "n_tp": data.shape[-1],
                }
            )
            seq_results[seq_name].append(
                {
                    "run": run_label,
                    "path": path,
                    "img": img,
                    "metrics": metrics,
                }
            )

    if not seq_results:
        console.print("  [red]No data found — skipping figures.[/red]")
        return pd.DataFrame(rows)

    # Shared colour scale: 0 → 95th percentile of all median values
    vmax = float(np.percentile(all_tsnr_values, 95)) * 2.0
    vmax = max(vmax, 50.0)  # floor so scale isn't degenerate
    vmin = 0.0
    console.print(f"  Shared tSNR colour scale: {vmin:.0f} – {vmax:.0f}")

    # ── Pass 2: per-sequence per-run figure ───────────────────────────────
    for seq_name, runs in seq_results.items():
        n_runs = len(runs)
        fig, axes = plt.subplots(
            n_runs,
            3,
            figsize=(10, 3.2 * n_runs),
            squeeze=False,
        )
        fig.suptitle(
            f"{seq_name}  —  sub-{sub} ses-{ses} task-{task}",
            fontsize=11,
            fontweight="bold",
        )
        for row_idx, run_info in enumerate(runs):
            _plot_ortho_panels(
                axes[row_idx],
                run_info["metrics"]["tsnr_std"],
                run_info["metrics"]["mask"],
                vmin,
                vmax,
                f"run-{run_info['run']}",
                run_info["metrics"]["median_std"],
            )
        plt.tight_layout()
        fig_path = op.join(
            fig_dir, f"sub-{sub}_ses-{ses}_task-{task}_{seq_name}_per_run.png"
        )
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        console.print(f"  [green]✓[/green] [dim]{op.basename(fig_path)}[/dim]")

    # ── Pass 3: comparison figure (mean-across-runs tSNR) ─────────────────
    n_seqs = len(seq_results)
    fig, axes = plt.subplots(n_seqs, 3, figsize=(10, 3.2 * n_seqs), squeeze=False)
    fig.suptitle(
        f"tSNR comparison (mean across runs)  —  sub-{sub} ses-{ses} task-{task}",
        fontsize=11,
        fontweight="bold",
    )
    for row_idx, (seq_name, runs) in enumerate(seq_results.items()):
        # Average tSNR across runs (use first run's mask/affine as reference)
        tsnr_stack = np.stack([r["metrics"]["tsnr_std"] for r in runs], axis=-1)
        mean_tsnr = np.nanmean(tsnr_stack, axis=-1)
        mask_ref = runs[0]["metrics"]["mask"]
        overall_median = float(np.nanmedian(mean_tsnr[mask_ref]))

        if save_nifti:
            stem = op.basename(runs[0]["path"]).replace(
                f"_run-{runs[0]['run']}_", "_mean_"
            )
            stem = re.sub(r"_desc-.*_bold", "_desc-tsnr_std_mean_bold", stem)
            nib.save(
                nib.Nifti1Image(
                    mean_tsnr, runs[0]["img"].affine, runs[0]["img"].header
                ),
                op.join(nii_dir, stem.replace(".nii.gz", "") + ".nii.gz"),
            )

        _plot_ortho_panels(
            axes[row_idx],
            mean_tsnr,
            mask_ref,
            vmin,
            vmax,
            seq_name,
            overall_median,
        )

    plt.tight_layout()
    fig_path = op.join(fig_dir, f"sub-{sub}_ses-{ses}_task-{task}_comparison.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  [green]✓[/green] [dim]{op.basename(fig_path)}[/dim]")

    # ── Pass 4: bar chart summary ──────────────────────────────────────────
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    seq_names = list(seq_results.keys())
    x = np.arange(len(seq_names))
    width = 0.6 / max(max(len(v) for v in seq_results.values()), 1)

    for seq_idx, seq_name in enumerate(seq_names):
        seq_df = df[df["seq"] == seq_name].sort_values("run")
        for run_idx, (_, row) in enumerate(seq_df.iterrows()):
            offset = (run_idx - (len(seq_df) - 1) / 2) * width
            ax.bar(
                x[seq_idx] + offset,
                row["median_tsnr_std"],
                width=width * 0.9,
                color=SEQ_COLORS.get(seq_name, "grey"),
                alpha=0.7 + 0.3 * run_idx / max(len(seq_df) - 1, 1),
                label=f"{seq_name} run-{row['run']}" if seq_idx == 0 else None,
            )

    # Overlay sequence mean as horizontal tick
    for seq_idx, seq_name in enumerate(seq_names):
        mean_val = df[df["seq"] == seq_name]["median_tsnr_std"].mean()
        ax.plot(
            [x[seq_idx] - 0.35, x[seq_idx] + 0.35],
            [mean_val, mean_val],
            color="black",
            linewidth=2.5,
            zorder=5,
        )
        ax.text(
            x[seq_idx],
            mean_val + 1.5,
            f"{mean_val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(seq_names, fontsize=10)
    ax.set_ylabel("Median tSNR (standard)", fontsize=10)
    ax.set_title(
        f"tSNR per run  —  sub-{sub} ses-{ses} task-{task}\n"
        f"(black bar = mean across runs)",
        fontsize=10,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    plt.tight_layout()
    fig_path = op.join(fig_dir, f"sub-{sub}_ses-{ses}_task-{task}_summary_barplot.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  [green]✓[/green] [dim]{op.basename(fig_path)}[/dim]")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    bids_dir: str = typer.Option(..., "-b", help="BIDS root directory"),
    tedana_dir_name: str = typer.Option(
        ..., "-tedana", help="tedana derivatives dir name under bids_dir/derivatives/"
    ),
    analysis_name: str = typer.Option(
        ..., "-n", "--analysis-name", help="tedana analysis label (analysis-{name})"
    ),
    fp_dir: str = typer.Option(
        ..., "-fp", help="fMRIprep derivatives directory (absolute path)"
    ),
    out_dir: str = typer.Option(
        ..., "-o", help="Output directory for figures and CSVs"
    ),
    single: Optional[str] = typer.Option(None, "-s", help="sub,ses  e.g.  pilot02,01"),
    batch_file: Optional[str] = typer.Option(
        None, "-f", help="Batch file: one sub,ses per line"
    ),
    tasks: str = typer.Option(..., "--tasks", "-t", help="Comma-separated task names"),
    save_nifti: bool = typer.Option(
        True, "--save-nifti/--no-nifti", help="Save per-run tSNR nifti maps"
    ),
    n_vols: int = typer.Option(
        213, "--n-vols", help="Use only the first N volumes from each run (0 = use all)"
    ),
    n_jobs: int = typer.Option(
        1,
        "-j",
        "--n-jobs",
        help="Parallel workers across (sub, ses, task) combinations",
    ),
) -> None:
    """Compare tSNR across ME_denoised, ME_optcom, and SE acquisitions."""

    task_list = [t.strip() for t in tasks.split(",") if t.strip()]

    # Sub/ses pairs
    subses_pairs: list[tuple[str, str]] = []
    if single:
        p = single.split(",")
        if len(p) != 2:
            console.print("[red]✗ -s expects sub,ses[/red]")
            raise typer.Exit(1)
        subses_pairs.append((p[0].strip(), p[1].strip()))
    elif batch_file:
        if not op.isfile(batch_file):
            console.print(f"[red]✗ batch file not found: {batch_file}[/red]")
            raise typer.Exit(1)
        with open(batch_file) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p = line.split(",")
                if len(p) < 2:
                    p = line.split()
                if len(p) >= 2:
                    subses_pairs.append((p[0].strip(), p[1].strip()))
    else:
        console.print("[red]✗ provide -s sub,ses  or  -f batch_file[/red]")
        raise typer.Exit(1)

    tedana_root = op.join(
        bids_dir, "derivatives", tedana_dir_name, f"analysis-{analysis_name}"
    )
    os.makedirs(out_dir, exist_ok=True)

    tbl = Table(title="compute_tsnr", show_lines=False)
    tbl.add_column("Parameter", style="cyan")
    tbl.add_column("Value")
    tbl.add_row("tedana_root", tedana_root)
    tbl.add_row("fp_dir", fp_dir)
    tbl.add_row("out_dir", out_dir)
    tbl.add_row("tasks", ", ".join(task_list))
    tbl.add_row("n_vols", str(n_vols) if n_vols > 0 else "all")
    tbl.add_row("n_jobs", str(n_jobs))
    tbl.add_row("subjects", str(len(subses_pairs)))
    console.print(tbl)

    # Build all (sub, ses, task) jobs
    jobs = [
        dict(
            sub=sub,
            ses=ses,
            task=task,
            tedana_func=op.join(tedana_root, f"sub-{sub}", f"ses-{ses}", "func"),
            fp_func=op.join(fp_dir, f"sub-{sub}", f"ses-{ses}", "func"),
            out_dir=out_dir,
            save_nifti=save_nifti,
            n_vols=n_vols,
        )
        for sub, ses in subses_pairs
        for task in task_list
    ]

    all_rows: list[pd.DataFrame] = []
    workers = min(n_jobs, len(jobs)) if n_jobs > 1 else 1

    if workers == 1:
        for kw in jobs:
            console.print(
                f"\n[bold cyan]sub-{kw['sub']}  ses-{kw['ses']}  task-{kw['task']}[/bold cyan]"
            )
            all_rows.append(_process_subses(**kw))
    else:
        console.print(
            f"\n[dim]Launching {len(jobs)} job(s) across {workers} worker(s) …[/dim]"
        )
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_kw = {pool.submit(_process_subses, **kw): kw for kw in jobs}
            for future in as_completed(future_to_kw):
                kw = future_to_kw[future]
                label = f"sub-{kw['sub']} ses-{kw['ses']} task-{kw['task']}"
                try:
                    all_rows.append(future.result())
                    console.print(f"  [green]✓[/green] {label}")
                except Exception as exc:
                    console.print(f"  [red]✗ {label}: {exc}[/red]")

    if all_rows:
        csv_path = op.join(out_dir, "tsnr_summary.csv")
        pd.concat(all_rows, ignore_index=True).to_csv(csv_path, index=False)
        console.print(f"\n[green]✓ summary CSV → {csv_path}[/green]")


if __name__ == "__main__":
    app()
