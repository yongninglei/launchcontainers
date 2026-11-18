#!/usr/bin/env python3
"""QC of fMRIPrep's fieldmap -> boldref registration (SDC), per run.

fMRIPrep (sdcflows ``init_coeff2epi_wf``) registers each fieldmap reference to
the run's boldref with a rigid transform, saved as
``func/*_from-boldref_to-<fmapid>_mode-image_xfm.txt``. In VOTCLOC the
SE-EPI fieldmaps and the BOLD/SBRef share the same grid and are acquired
seconds to minutes apart, so the true transform is close to identity
(typical: < 3 mm, < 2.5 deg). A large rotation is not physical in a head coil:
it means the registration failed and the fieldmap was applied in the wrong
place (found in sub-10 ses-04 and ses-05, up to 27 mm / 20 deg).

``compute`` writes ``fmap_boldref_xfm_runs.csv`` (one row per run) and
``fmap_boldref_xfm_sessions.csv`` (one row per session) in ``--out-dir``.
``plot`` reads the runs CSV only and draws one strip per session.

Per run: rotation angle (deg), translation (mm, ITK/LPS, about the origin),
displacement of the boldref grid centre, and mean displacement of the corners
of an 80 mm cube around it (FD-like). ``flag`` = rot > --max-rot or
disp_cube > --max-disp.

Reads only small text transforms and NIfTI headers (no image data), but over
the whole dataset that is ~1,800 header reads: run it yourself for all
sessions; for a check, give -s for a few sessions.

Usage::

    python qc_fmap_boldref_xfm.py compute -p <fmriprep_dir> -o <out_dir> [-s sub-10/ses-05 ...]
    python qc_fmap_boldref_xfm.py plot -o <out_dir>
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import typer
from rich.console import Console

app = typer.Typer(add_completion=False)
console = Console()
XFM_RE = re.compile(
    r"(sub-[^_]+)_(ses-[^_]+)_task-(\w+?)_run-(\d+)_from-boldref_to-(\w+?)_mode-image_xfm\.txt$"
)


def read_itk_affine(path: str) -> tuple[np.ndarray, np.ndarray]:
    txt = open(path).read()
    p = [float(v) for v in txt.split("Parameters:")[1].split("\n")[0].split()]
    fixed = (
        txt.split("FixedParameters:")[1].split("\n")[0].split()
        if "FixedParameters:" in txt
        else []
    )
    A, t = np.array(p[:9]).reshape(3, 3), np.array(p[9:12])
    if fixed and any(
        float(v) != 0 for v in fixed
    ):  # centre of rotation -> fold into translation
        c = np.array([float(v) for v in fixed[:3]])
        t = t + c - A @ c
    return A, t


def run_row(xfm: str) -> dict:
    import nibabel as nib

    sub, ses, task, run, fmapid = XFM_RE.search(os.path.basename(xfm)).groups()
    A, t = read_itk_affine(xfm)
    stem = xfm.split("_from-boldref")[0]
    ref = next(
        (
            r
            for r in (
                stem + "_desc-coreg_boldref.nii.gz",
                stem + "_desc-hmc_boldref.nii.gz",
            )
            if os.path.exists(r)
        ),
        None,
    )
    row = dict(
        sub=sub,
        ses=ses,
        task=task,
        run=int(run),
        fmapid=fmapid,
        rot_deg=float(np.degrees(np.arccos(np.clip((np.trace(A) - 1) / 2, -1, 1)))),
        tx=t[0],
        ty=t[1],
        tz=t[2],
        disp_centre_mm=np.nan,
        disp_cube_mm=np.nan,
        boldref=ref or "",
    )
    if ref:
        img = nib.load(ref)
        c = (img.affine @ np.r_[(np.array(img.shape[:3]) - 1) / 2, 1])[:3] * np.array(
            [-1, -1, 1]
        )  # RAS -> LPS
        corners = c + 40 * np.array(
            [[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]
        )
        row["disp_centre_mm"] = float(np.linalg.norm(A @ c + t - c))
        row["disp_cube_mm"] = float(
            np.mean(np.linalg.norm(corners @ A.T + t - corners, axis=1))
        )
    return row


@app.command()
def compute(
    fmriprep_dir: Path = typer.Option(..., "-p", "--fmriprep-dir"),
    out_dir: Path = typer.Option(..., "-o", "--out-dir"),
    sessions: Optional[List[str]] = typer.Option(
        None, "-s", "--session", help="sub-XX/ses-YY; repeatable; default all"
    ),
    max_rot: float = typer.Option(3.0, help="flag rotation above this (deg)"),
    max_disp: float = typer.Option(
        5.0, help="flag mean cube displacement above this (mm)"
    ),
):
    pats = sessions or ["sub-*/ses-*"]
    files = sorted(
        f
        for s in pats
        for f in glob.glob(
            str(fmriprep_dir / s / "func" / "*_from-boldref_to-*_mode-image_xfm.txt")
        )
    )
    if not files:
        console.print(
            f"[red]ERROR[/red] no boldref->fieldmap transforms under {fmriprep_dir}"
        )
        raise typer.Exit(2)
    df = pd.DataFrame([run_row(f) for f in files])
    df["flag"] = (df["rot_deg"] > max_rot) | (df["disp_cube_mm"] > max_disp)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.round(3).to_csv(out_dir / "fmap_boldref_xfm_runs.csv", index=False)
    ses = (
        df.groupby(["sub", "ses"])
        .agg(
            n_runs=("run", "size"),
            n_fmaps=("fmapid", "nunique"),
            n_flagged=("flag", "sum"),
            median_disp_mm=("disp_cube_mm", "median"),
            max_disp_mm=("disp_cube_mm", "max"),
            max_rot_deg=("rot_deg", "max"),
        )
        .reset_index()
    )
    ses.round(2).to_csv(out_dir / "fmap_boldref_xfm_sessions.csv", index=False)
    for _, r in ses.iterrows():
        tag = "[red]FLAG[/red]" if r["n_flagged"] else "[green]OK[/green]"
        console.print(
            f"{tag} {r['sub']} {r['ses']}: {int(r['n_flagged'])}/{int(r['n_runs'])} runs flagged; "
            f"median {r['median_disp_mm']:.1f} mm, max {r['max_disp_mm']:.1f} mm, max rot {r['max_rot_deg']:.1f} deg"
        )
    console.print(
        f"[green]saved[/green] {out_dir / 'fmap_boldref_xfm_runs.csv'} and fmap_boldref_xfm_sessions.csv"
    )


@app.command()
def plot(
    out_dir: Path = typer.Option(
        ..., "-o", "--out-dir", help="dir holding fmap_boldref_xfm_runs.csv"
    ),
    max_disp: float = typer.Option(5.0),
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(out_dir / "fmap_boldref_xfm_runs.csv")
    df["session"] = df["sub"] + " " + df["ses"]
    order = sorted(df["session"].unique())
    fig, ax = plt.subplots(figsize=(max(6, 0.12 * len(order)), 4))
    for i, s in enumerate(order):
        d = df[df["session"] == s]
        ax.scatter(
            np.full(len(d), i),
            d["disp_cube_mm"],
            s=8,
            c=np.where(d["flag"], "#c0392b", "#7f8c8d"),
        )
    ax.axhline(max_disp, ls="--", lw=0.8, c="#c0392b")
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=6)
    ax.set_ylabel("fieldmap -> boldref displacement (mm)")
    ax.set_title("fMRIPrep fieldmap registration per run")
    fig.tight_layout()
    out = out_dir / "fmap_boldref_xfm.png"
    fig.savefig(out, dpi=150)
    console.print(f"[green]saved[/green] {out}")


if __name__ == "__main__":
    app()
