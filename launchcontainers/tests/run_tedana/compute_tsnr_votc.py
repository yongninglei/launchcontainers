#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) Yongning Lei 2025-2026  —  Apache-2.0 license
# -----------------------------------------------------------------------------
"""
compute_tsnr_votc.py

Compare tSNR across ME_denoised, ME_optcom, and SE on the fsnative surface,
restricted to cortical ROIs defined by the Destrieux atlas (aparc.a2009s).

ROI groups
----------
  VOTC      : fusiform, lingual, parahippocampal
  IFG       : pars opercularis, pars orbitalis, pars triangularis
  Premotor  : precentral gyrus + precentral sulci (BA4/BA6)
  Temporal  : entire temporal lobe (STG, MTG, ITG, STS, plana, pole)
  V1        : calcarine sulcus + cuneus

All three sequences live on the same fsnative vertex grid → no FOV mismatch.
Parallelism: within session, runs × hemispheres are processed in parallel (-j).
The summary figure shows mean tSNR per region (averaged across runs × hemis
× parcels within region) — not per directory.

Input surface timeseries
-------------------------
  ME_denoised : {tedana_func}/…_hemi-{LR}_space-fsnative_desc-denoised_bold.func.gii
  ME_optcom   : {tedana_func}/…_hemi-{LR}_space-fsnative_desc-optcom_bold.func.gii
  SE          : {fp_func}/…_acq-SE_run-*_hemi-{LR}_space-fsnative_bold.func.gii

Usage
-----
    python compute_tsnr_votc.py \\
        -b /data/BIDS \\
        -tedana tedana-26.0.3 \\
        -n default_ica \\
        -fp /data/BIDS/derivatives/fmriprep-25.1.4 \\
        --fs-subjects-dir /data/BIDS/derivatives/freesurfer \\
        -s pilot02,01 \\
        --tasks BfLocVideo \\
        -o /data/results/tsnr_roi \\
        -j 6
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

console = Console()
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

# ---------------------------------------------------------------------------
# Destrieux ROI group definitions  (parcel_index: parcel_name)
# ---------------------------------------------------------------------------

ROI_GROUPS: dict[str, dict[int, str]] = {
    "VOTC": {
        21: "G_oc-temp_lat-fusifor",
        22: "G_oc-temp_med-Lingual",
        23: "G_oc-temp_med-Parahip",
    },
    "IFG": {
        12: "G_front_inf-Opercular",
        13: "G_front_inf-Orbital",
        14: "G_front_inf-Triangul",
    },
    "Premotor": {
        29: "G_precentral",
        69: "S_precentral-inf-part",
        70: "S_precentral-sup-part",
    },
    "Temporal": {
        33: "G_temp_sup-G_T_transv",
        34: "G_temp_sup-Lateral",
        35: "G_temp_sup-Plan_polar",
        36: "G_temp_sup-Plan_tempo",
        37: "G_temporal_inf",
        38: "G_temporal_middle",
        44: "Pole_temporal",
        73: "S_temporal_inf",
        74: "S_temporal_sup",
        75: "S_temporal_transverse",
    },
    "V1": {
        11: "G_cuneus",
        45: "S_calcarine",
    },
}

SEQ_COLORS = {
    "ME_denoised": "#2196F3",
    "ME_optcom": "#FF9800",
    "SE": "#4CAF50",
}


# ---------------------------------------------------------------------------
# tSNR on surface data  (matches 01_tSNR.py logic exactly)
# ---------------------------------------------------------------------------


def _detrend_quadratic(data_2d: np.ndarray) -> np.ndarray:
    n_verts, n_tp = data_2d.shape
    t = np.linspace(-1, 1, n_tp, dtype=np.float64)
    X = np.column_stack([t**d for d in (0, 1, 2)])
    betas, _, _, _ = np.linalg.lstsq(X, data_2d.T, rcond=None)
    return (data_2d.T - X @ betas).T


def compute_tsnr_surface(data: np.ndarray) -> np.ndarray:
    """Return per-vertex tSNR from (n_verts, n_tp). Brain mask used for
    the threshold only (0.1 × p99); map returned for all vertices."""
    data = data.astype(np.float32)
    mean_1d = data.mean(axis=1)
    det = _detrend_quadratic(data)
    std_1d = det.std(axis=1, ddof=0)
    tsnr = np.full(len(mean_1d), np.nan, dtype=np.float32)
    ok = std_1d > 0
    tsnr[ok] = mean_1d[ok] / std_1d[ok]
    return tsnr


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------


def load_roi_masks(
    fs_subjects_dir: str, subject: str
) -> dict[str, dict[str, dict[int, np.ndarray]]]:
    """Return {hemi: {group_name: {parcel_idx: bool_array}}}.

    The inner bool_array has length n_verts (fsnative vertex count).
    """
    result: dict = {}
    fs_hemi_map = {"L": "lh", "R": "rh"}

    for hemi, fsh in fs_hemi_map.items():
        annot_path = op.join(
            fs_subjects_dir, subject, "label", f"{fsh}.aparc.a2009s.annot"
        )
        if not op.isfile(annot_path):
            console.print(f"  [red]✗ annotation not found: {annot_path}[/red]")
            result[hemi] = {}
            continue

        labels, _, names = nib.freesurfer.read_annot(annot_path)
        names_str = [n.decode() if isinstance(n, bytes) else n for n in names]

        hemi_groups: dict[str, dict[int, np.ndarray]] = {}
        for group_name, parcel_dict in ROI_GROUPS.items():
            parcel_masks: dict[int, np.ndarray] = {}
            for idx, expected_name in parcel_dict.items():
                if idx < len(names_str) and names_str[idx] == expected_name:
                    parcel_masks[idx] = labels == idx
                else:
                    console.print(
                        f"  [yellow]⚠ hemi-{hemi} {group_name} "
                        f"idx={idx} not found (expected '{expected_name}')[/yellow]"
                    )
            if parcel_masks:
                hemi_groups[group_name] = parcel_masks
        result[hemi] = hemi_groups

    return result


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def _find_giftis(base_dir: str, pattern: str) -> list[tuple[str, str, str]]:
    """Return sorted [(run, hemi, path)] matching glob pattern."""
    triples = []
    for f in sorted(glob.glob(op.join(base_dir, pattern))):
        bn = op.basename(f)
        run = (
            re.search(r"_run-(\w+)[_.]", bn)
            or type("", (), {"group": lambda s, _: "01"})()
        ).group(1)
        hemi = (
            re.search(r"_hemi-([LR])[_.]", bn)
            or type("", (), {"group": lambda s, _: "L"})()
        ).group(1)
        triples.append((run, hemi, f))
    return triples


def _get_surface_sequences(
    sub: str,
    ses: str,
    task: str,
    tedana_func: str,
    fp_func: str,
) -> dict[str, list[tuple[str, str, str]]]:
    return {
        "ME_denoised": _find_giftis(
            tedana_func,
            f"sub-{sub}_ses-{ses}_task-{task}_acq-ME_run-*"
            f"_hemi-*_space-fsnative_desc-denoised_bold.func.gii",
        ),
        "ME_optcom": _find_giftis(
            tedana_func,
            f"sub-{sub}_ses-{ses}_task-{task}_acq-ME_run-*"
            f"_hemi-*_space-fsnative_desc-optcom_bold.func.gii",
        ),
        "SE": _find_giftis(
            fp_func,
            f"sub-{sub}_ses-{ses}_task-{task}_acq-SE_run-*"
            f"_hemi-*_space-fsnative_bold.func.gii",
        ),
    }


# ---------------------------------------------------------------------------
# Per-file worker  (runs in parallel)
# ---------------------------------------------------------------------------


def _worker(
    seq_name: str,
    run: str,
    hemi: str,
    path: str,
    roi_masks: dict,  # {group: {idx: bool_array}} — already hemi-specific
    n_vols: int,
) -> dict:
    """Module-level function so ProcessPoolExecutor can pickle it."""
    """Load one gifti, compute tSNR, extract ROI medians."""
    img = nib.load(path)
    data = np.column_stack([da.data for da in img.darrays]).astype(np.float32)
    if n_vols > 0:
        data = data[:, :n_vols]

    tsnr = compute_tsnr_surface(data)

    roi_rows = []
    for group_name, parcel_dict in roi_masks.items():
        # Union of all parcel masks in this group
        group_mask = np.zeros(len(tsnr), dtype=bool)
        for pm in parcel_dict.values():
            group_mask |= pm

        vals = tsnr[group_mask]
        vals = vals[~np.isnan(vals)]
        median = float(np.nanmedian(vals)) if len(vals) > 0 else np.nan

        # Per-parcel medians
        for idx, pm in parcel_dict.items():
            pv = tsnr[pm]
            pv = pv[~np.isnan(pv)]
            roi_rows.append(
                {
                    "seq": seq_name,
                    "run": run,
                    "hemi": hemi,
                    "roi_group": group_name,
                    "parcel_idx": idx,
                    "parcel_name": ROI_GROUPS[group_name][idx],
                    "group_median": median,
                    "parcel_median": float(np.nanmedian(pv)) if len(pv) > 0 else np.nan,
                    "n_verts": int(pm.sum()),
                    "n_tp": data.shape[1],
                }
            )

    return {"rows": roi_rows, "tsnr": tsnr, "seq": seq_name, "run": run, "hemi": hemi}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_region_summary(
    df: pd.DataFrame, sub: str, ses: str, task: str, fig_dir: str
) -> None:
    """One figure: mean tSNR per region × sequence.
    Mean is taken across runs × hemispheres × parcels within region."""
    region_order = list(ROI_GROUPS.keys())
    seq_names = [
        s for s in ["ME_denoised", "ME_optcom", "SE"] if s in df["seq"].unique()
    ]

    # Aggregate: mean of group_median across (run, hemi) per (seq, roi_group)
    agg = (
        df.groupby(["seq", "roi_group"])["group_median"]
        .mean()
        .reset_index()
        .rename(columns={"group_median": "mean_tsnr"})
    )

    n_regs = len(region_order)
    n_seqs = len(seq_names)
    width = 0.7 / n_seqs
    x = np.arange(n_regs)

    fig, ax = plt.subplots(figsize=(max(8, n_regs * 2.5), 4.5))
    for si, seq in enumerate(seq_names):
        seq_df = agg[agg["seq"] == seq]
        vals = [
            seq_df.loc[seq_df["roi_group"] == r, "mean_tsnr"].values[0]
            if r in seq_df["roi_group"].values
            else np.nan
            for r in region_order
        ]
        offset = (si - (n_seqs - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width=width * 0.92,
            color=SEQ_COLORS.get(seq, "grey"),
            label=seq,
            alpha=0.87,
        )
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 0.4,
                    f"{v:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(region_order, fontsize=11)
    ax.set_ylabel("Mean tSNR  (avg across runs, hemis, parcels)", fontsize=10)
    ax.set_title(
        f"tSNR comparison by cortical region\nsub-{sub}  ses-{ses}  task-{task}",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    plt.tight_layout()

    path = op.join(fig_dir, f"sub-{sub}_ses-{ses}_task-{task}_region_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  [green]✓[/green] [dim]{op.basename(path)}[/dim]")


def _plot_per_hemi(
    df: pd.DataFrame, sub: str, ses: str, task: str, fig_dir: str
) -> None:
    """Grid: rows = regions, cols = L / R, bars = sequences × runs."""
    region_order = list(ROI_GROUPS.keys())
    seq_names = [
        s for s in ["ME_denoised", "ME_optcom", "SE"] if s in df["seq"].unique()
    ]

    fig, axes = plt.subplots(
        len(region_order),
        2,
        figsize=(12, 3.5 * len(region_order)),
        squeeze=False,
    )
    fig.suptitle(
        f"tSNR per region × hemisphere  —  sub-{sub} ses-{ses} task-{task}",
        fontsize=11,
        fontweight="bold",
    )

    for ri, region in enumerate(region_order):
        for hi, hemi in enumerate(["L", "R"]):
            ax = axes[ri][hi]
            sub_df = df[(df["roi_group"] == region) & (df["hemi"] == hemi)]
            if sub_df.empty:
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.axis("off")
                continue

            run_seq = (
                sub_df.groupby(["seq", "run"])["group_median"].mean().reset_index()
            )
            seq_list = [s for s in seq_names if s in run_seq["seq"].unique()]
            runs = sorted(run_seq["run"].unique())
            x = np.arange(len(seq_list))
            w = 0.7 / max(len(runs), 1)

            for rri, run in enumerate(runs):
                rdf = run_seq[run_seq["run"] == run]
                vals = [
                    rdf.loc[rdf["seq"] == s, "group_median"].values[0]
                    if s in rdf["seq"].values
                    else np.nan
                    for s in seq_list
                ]
                offset = (rri - (len(runs) - 1) / 2) * w
                alpha = 0.5 + 0.5 * rri / max(len(runs) - 1, 1)
                ax.bar(
                    x + offset,
                    vals,
                    width=w * 0.9,
                    color=[SEQ_COLORS.get(s, "grey") for s in seq_list],
                    alpha=alpha,
                    label=f"run-{run}",
                )

            ax.set_xticks(x)
            ax.set_xticklabels(seq_list, fontsize=7, rotation=15, ha="right")
            ax.set_title(f"{region}  hemi-{hemi}", fontsize=8)
            ax.set_ylabel("tSNR", fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if ri == 0 and hi == 0:
                ax.legend(fontsize=7, frameon=False)

    plt.tight_layout()
    path = op.join(fig_dir, f"sub-{sub}_ses-{ses}_task-{task}_per_hemi.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  [green]✓[/green] [dim]{op.basename(path)}[/dim]")


def _plot_surface_maps(
    tsnr_store: dict,  # {(seq, run, hemi): tsnr_array}
    roi_masks: dict,  # {hemi: {group: {idx: mask}}}
    fp_anat_dir: str,
    sub: str,
    ses: str,
    task: str,
    fig_dir: str,
) -> None:
    """Ventral surface views of VOTC tSNR per sequence."""
    try:
        from nilearn import plotting as nplot
    except ImportError:
        console.print(
            "  [yellow]nilearn not available — skipping surface plot[/yellow]"
        )
        return

    seq_names = sorted({k[0] for k in tsnr_store})

    for hemi in ("L", "R"):
        surf_path = sorted(
            glob.glob(op.join(fp_anat_dir, f"*hemi-{hemi}_pial.surf.gii"))
        )
        if not surf_path:
            continue
        surf_path = surf_path[0]
        n_verts = nib.load(surf_path).darrays[0].data.shape[0]

        # Combined VOTC mask
        votc_union = np.zeros(n_verts, dtype=bool)
        for pm in roi_masks.get(hemi, {}).get("VOTC", {}).values():
            votc_union |= pm

        fig, axes = plt.subplots(1, len(seq_names), figsize=(5 * len(seq_names), 4))
        if len(seq_names) == 1:
            axes = [axes]
        fig.suptitle(
            f"VOTC tSNR (mean across runs)  hemi-{hemi}  "
            f"sub-{sub} ses-{ses} task-{task}",
            fontsize=10,
        )

        all_vals_votc = []
        mean_maps: dict[str, np.ndarray] = {}
        for seq in seq_names:
            seq_maps = [
                v for k, v in tsnr_store.items() if k[0] == seq and k[2] == hemi
            ]
            if not seq_maps:
                continue
            mean_tsnr = np.nanmean(np.stack(seq_maps, axis=0), axis=0)
            masked = mean_tsnr.copy()
            masked[~votc_union] = np.nan
            mean_maps[seq] = masked
            all_vals_votc.extend(
                masked[votc_union][~np.isnan(masked[votc_union])].tolist()
            )

        vmax = float(np.percentile(all_vals_votc, 95)) if all_vals_votc else 100.0

        import tempfile

        for ax, seq in zip(axes, seq_names):
            if seq not in mean_maps:
                ax.axis("off")
                continue
            tsnr_arr = mean_maps[seq]
            med = float(np.nanmedian(tsnr_arr[votc_union]))
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            nplot.plot_surf_stat_map(
                surf_mesh=surf_path,
                stat_map=tsnr_arr,
                hemi="left" if hemi == "L" else "right",
                view="ventral",
                cmap="jet",
                vmin=0,
                vmax=vmax,
                colorbar=True,
                title=f"{seq}\nVOTC median={med:.0f}",
                output_file=tmp.name,
            )
            plt.close("all")
            ax.imshow(plt.imread(tmp.name))
            ax.axis("off")
            os.unlink(tmp.name)

        plt.tight_layout()
        path = op.join(
            fig_dir, f"sub-{sub}_ses-{ses}_task-{task}_VOTC_surface_hemi-{hemi}.png"
        )
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        console.print(f"  [green]✓[/green] [dim]{op.basename(path)}[/dim]")


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def _process_subses(
    sub: str,
    ses: str,
    task: str,
    tedana_func: str,
    fp_func: str,
    roi_masks: dict,
    out_dir: str,
    n_vols: int,
    n_jobs: int,
    fp_anat_dir: str,
) -> pd.DataFrame:
    seqs = _get_surface_sequences(sub, ses, task, tedana_func, fp_func)

    # Build flat job list: (seq_name, run, hemi, path)
    job_list = [
        (seq_name, run, hemi, path)
        for seq_name, triples in seqs.items()
        for run, hemi, path in triples
    ]

    if not job_list:
        console.print("  [red]No surface files found.[/red]")
        return pd.DataFrame()

    console.print(
        f"  {len(job_list)} (seq, run, hemi) jobs  "
        f"→ {min(n_jobs, len(job_list))} workers"
    )

    all_rows: list[dict] = []
    tsnr_store: dict[tuple, np.ndarray] = {}  # (seq, run, hemi) → tsnr array

    # Build kwargs per job so _worker (module-level, picklable) receives everything
    def _make_kwargs(seq_name, run, hemi, path):
        return dict(
            seq_name=seq_name,
            run=run,
            hemi=hemi,
            path=path,
            roi_masks=roi_masks.get(hemi, {}),
            n_vols=n_vols,
        )

    if n_jobs == 1:
        for seq_name, run, hemi, path in job_list:
            res = _worker(**_make_kwargs(seq_name, run, hemi, path))
            all_rows.extend(res["rows"])
            tsnr_store[(res["seq"], res["run"], res["hemi"])] = res["tsnr"]
            console.print(
                f"  [dim]{res['seq']} run-{res['run']} hemi-{res['hemi']} done[/dim]"
            )
    else:
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(job_list))) as pool:
            futures = {
                pool.submit(_worker, **_make_kwargs(sn, ru, he, pa)): (sn, ru, he, pa)
                for sn, ru, he, pa in job_list
            }
            for future in as_completed(futures):
                job_id = futures[future]
                try:
                    res = future.result()
                    all_rows.extend(res["rows"])
                    tsnr_store[(res["seq"], res["run"], res["hemi"])] = res["tsnr"]
                    console.print(
                        f"  [green]✓[/green] [dim]{res['seq']} "
                        f"run-{res['run']} hemi-{res['hemi']}[/dim]"
                    )
                except Exception as exc:
                    console.print(f"  [red]✗ {job_id}: {exc}[/red]")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    fig_dir = op.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    _plot_region_summary(df, sub, ses, task, fig_dir)
    _plot_per_hemi(df, sub, ses, task, fig_dir)
    _plot_surface_maps(tsnr_store, roi_masks, fp_anat_dir, sub, ses, task, fig_dir)

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    bids_dir: str = typer.Option(..., "-b"),
    tedana_dir_name: str = typer.Option(..., "-tedana"),
    analysis_name: str = typer.Option(..., "-n", "--analysis-name"),
    fp_dir: str = typer.Option(..., "-fp"),
    fs_subjects_dir: str = typer.Option(..., "--fs-subjects-dir", "--fs-sd"),
    out_dir: str = typer.Option(..., "-o"),
    single: Optional[str] = typer.Option(None, "-s", help="sub,ses  e.g.  pilot02,01"),
    batch_file: Optional[str] = typer.Option(None, "-f"),
    tasks: str = typer.Option(..., "--tasks", "-t"),
    n_vols: int = typer.Option(213, "--n-vols", help="First N volumes (0=all)"),
    n_jobs: int = typer.Option(
        1,
        "-j",
        "--n-jobs",
        help="Parallel workers within session (across runs × hemis)",
    ),
) -> None:
    """Compare tSNR in cortical ROIs (VOTC / IFG / Premotor / Temporal / V1)."""

    task_list = [t.strip() for t in tasks.split(",") if t.strip()]

    subses_pairs: list[tuple[str, str]] = []
    if single:
        p = single.split(",")
        subses_pairs.append((p[0].strip(), p[1].strip()))
    elif batch_file:
        with open(batch_file) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p = line.split(",")
                if len(p) >= 2:
                    subses_pairs.append((p[0].strip(), p[1].strip()))
    else:
        console.print("[red]✗ provide -s sub,ses  or  -f file[/red]")
        raise typer.Exit(1)

    tedana_root = op.join(
        bids_dir, "derivatives", tedana_dir_name, f"analysis-{analysis_name}"
    )
    os.makedirs(out_dir, exist_ok=True)

    tbl = Table(title="compute_tsnr_votc", show_lines=False)
    tbl.add_column("Parameter", style="cyan")
    tbl.add_column("Value")
    tbl.add_row("tedana_root", tedana_root)
    tbl.add_row("fp_dir", fp_dir)
    tbl.add_row("fs_subjects_dir", fs_subjects_dir)
    tbl.add_row("out_dir", out_dir)
    tbl.add_row("n_vols", str(n_vols) if n_vols > 0 else "all")
    tbl.add_row("n_jobs", str(n_jobs))
    tbl.add_row("ROI groups", ", ".join(ROI_GROUPS.keys()))
    console.print(tbl)

    all_rows: list[pd.DataFrame] = []

    for sub, ses in subses_pairs:
        for task in task_list:
            console.print(f"\n[bold cyan]sub-{sub}  ses-{ses}  task-{task}[/bold cyan]")

            roi_masks = load_roi_masks(fs_subjects_dir, f"sub-{sub}")
            # Log vertex counts per group
            for hemi in ("L", "R"):
                for grp, parcels in roi_masks.get(hemi, {}).items():
                    total = sum(pm.sum() for pm in parcels.values())
                    console.print(
                        f"  hemi-{hemi}  {grp:<10s}  {total:5d} verts  "
                        f"({len(parcels)} parcels)"
                    )

            tedana_func = op.join(tedana_root, f"sub-{sub}", f"ses-{ses}", "func")
            fp_func = op.join(fp_dir, f"sub-{sub}", f"ses-{ses}", "func")
            fp_anat_dir = op.join(fp_dir, f"sub-{sub}", f"ses-{ses}", "anat")

            df = _process_subses(
                sub=sub,
                ses=ses,
                task=task,
                tedana_func=tedana_func,
                fp_func=fp_func,
                roi_masks=roi_masks,
                out_dir=out_dir,
                n_vols=n_vols,
                n_jobs=n_jobs,
                fp_anat_dir=fp_anat_dir,
            )
            all_rows.append(df)

    non_empty = [d for d in all_rows if not d.empty]
    if non_empty:
        csv_path = op.join(out_dir, "tsnr_roi_summary.csv")
        pd.concat(non_empty, ignore_index=True).to_csv(csv_path, index=False)
        console.print(f"\n[green]✓ CSV → {csv_path}[/green]")


if __name__ == "__main__":
    app()
