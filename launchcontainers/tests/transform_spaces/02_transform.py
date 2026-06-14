#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) Yongning Lei 2025-2026
# Apache-2.0 license
# -----------------------------------------------------------------------------
"""
02_transform.py

Generic transformation utility: project one or more volumes to the surface
with `mri_vol2surf`, given an affine produced by 01_register.py (or any other
tool).

`--affine` may be:
    *.lta   - a FreeSurfer registration, used as-is
    *.mat   - an ANTs `*0GenericAffine.mat`, converted via ConvertTransformFile + lta_convert
    *.txt   - an ITK-format affine (e.g. fMRIprep's *_xfm.txt), converted via lta_convert

If conversion is needed, `--affine-src` (defaulting to the first matched
`-i` input) defines the affine's source geometry, and `--affine-trg`
(defaulting to `{fs-subjects-dir}/{fs-subject}/mri/orig.mgz`) defines its
target geometry. The converted .lta is cached next to `--affine` (or in
`--out-dir`).

For every input file, projects both hemispheres using a single --surf/--interp
combination. Output filenames are derived from the input by:

    ..._space-{space_label_in}_..._<suffix>{ext}
        -> ..._hemi-{L,R}_space-{space_label_out}_..._desc-{surf}{Interp}_<suffix>{out_ext}

e.g. --surf white --interp nearest   -> desc-whiteNearest
     --surf pial  --interp trilinear -> desc-pialTrilinear

If `_space-{space_label_in}_` is not found in the filename, `_hemi-{H}_space-{space_label_out}`
is appended before the extension.

Usage
-----
    python 02_transform.py \\
        --affine .../sub-pilot01_ses-01_from-boldref_to-fsnative_desc-bbregister_reg.lta \\
        --fs-subjects-dir .../derivatives/freesurfer --fs-subject sub-pilot01 \\
        -i ".../sub-pilot01_ses-01_task-x_*_statmap.nii.gz" \\
        --space-label-in T1w --space-label-out fsnative \\
        --execute
"""

from __future__ import annotations

import glob
import os.path as op
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

import transform_common as tc

console = Console()
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

HEMIS = {"L": "lh", "R": "rh"}


@app.command()
def main(
    affine: str = typer.Option(..., "--affine", help="Affine from 01_register.py (or any tool): .lta | .mat | .txt"),
    fs_subjects_dir: str = typer.Option(..., "--fs-subjects-dir", "--fs-sd", help="FreeSurfer SUBJECTS_DIR"),
    fs_subject: str = typer.Option(..., "-s", "--fs-subject", help="FreeSurfer subject name (e.g. sub-pilot01)"),
    inputs: list[str] = typer.Option(..., "-i", "--input", help="Input volume path(s) or glob pattern(s)"),
    affine_src: Optional[str] = typer.Option(None, "--affine-src", help="Source-geometry volume for affine->lta conversion (default: first matched -i file)"),
    affine_trg: Optional[str] = typer.Option(None, "--affine-trg", help="Target-geometry volume for affine->lta conversion (default: {fs-subjects-dir}/{fs-subject}/mri/orig.mgz)"),
    out_dir: Optional[str] = typer.Option(None, "--out-dir", help="Directory for a converted .lta (default: same directory as --affine)"),
    hemi: str = typer.Option("LR", "--hemi", help="Hemispheres to project: L | R | LR"),
    surf: str = typer.Option("white", "--surf", help="Surface to sample: white | pial | midthickness"),
    interp: str = typer.Option("trilinear", "--interp", help="Interpolation: nearest | trilinear"),
    proj_frac: Optional[float] = typer.Option(None, "--proj-frac", help="--projfrac value"),
    proj_frac_avg: Optional[str] = typer.Option(None, "--proj-frac-avg", help="start,stop,step for --projfrac-avg (overrides --proj-frac)"),
    trgsubject: Optional[str] = typer.Option(None, "--trgsubject", help="Resample to this FS subject (e.g. fsaverage)"),
    space_label_in: str = typer.Option("T1w", "--space-label-in", help="space-* label to replace in input filenames"),
    space_label_out: str = typer.Option("fsnative", "--space-label-out", help="space-* label to use in output filenames (overridden by --trgsubject)"),
    out_ext: str = typer.Option(".func.gii", "--out-ext", help="Output file extension"),
    fs_module: str = typer.Option("freesurfer/7.3.2", "--fs-module"),
    ants_module: str = typer.Option("ants", "--ants-module"),
    execute: bool = typer.Option(False, "--execute", help="Run mri_vol2surf. Without this flag only the plan is printed."),
    overwrite: bool = typer.Option(False, "--overwrite"),
    n_jobs: int = typer.Option(1, "--n-jobs", "-J", help="Files to project in parallel (1=sequential, -1=all)"),
) -> None:
    """Resolve `--affine` to a .lta (converting if needed) and project volume(s) to the surface."""

    hemi = hemi.upper()
    hemis = [h for h in ("L", "R") if h in hemi]
    if not hemis:
        console.print(f"[red]✗ --hemi must be L, R, or LR (got {hemi})[/red]")
        raise typer.Exit(1)

    pfa: Optional[tuple[float, float, float]] = None
    if proj_frac_avg:
        parts = proj_frac_avg.split(",")
        if len(parts) != 3:
            console.print("[red]✗ --proj-frac-avg expects start,stop,step[/red]")
            raise typer.Exit(1)
        pfa = tuple(float(p) for p in parts)  # type: ignore[assignment]

    files: list[str] = []
    for pat in inputs:
        pat = op.expanduser(pat)
        matches = sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
        files.extend(matches)
    if not files:
        console.print("[red]✗ no input files matched[/red]")
        raise typer.Exit(1)

    tbl = Table(title="02_transform", show_lines=False)
    tbl.add_column("Parameter", style="cyan")
    tbl.add_column("Value")
    tbl.add_row("affine", affine)
    tbl.add_row("fs_subjects_dir", fs_subjects_dir)
    tbl.add_row("fs_subject", fs_subject)
    tbl.add_row("hemi", "".join(hemis))
    tbl.add_row("surf", surf)
    tbl.add_row("interp", interp)
    tbl.add_row("desc", tc.desc_label(surf, interp))
    tbl.add_row("proj_frac", str(proj_frac) if proj_frac is not None else "(none)")
    tbl.add_row("proj_frac_avg", str(pfa) if pfa else "(none)")
    tbl.add_row("trgsubject", trgsubject or "(none)")
    tbl.add_row("space_label_in", space_label_in)
    tbl.add_row("space_label_out", trgsubject or space_label_out)
    tbl.add_row("out_ext", out_ext)
    tbl.add_row("n_files", str(len(files)))
    tbl.add_row("execute", str(execute))
    console.print(tbl)

    if not execute:
        console.print("\n[yellow bold]DRY RUN — pass --execute to run mri_vol2surf[/yellow bold]\n")

    # ── Step 1: resolve --affine to a .lta (converting if needed) ──────────────
    console.print("[bold]Resolve affine -> .lta[/bold]")
    affine_out_dir = out_dir or op.dirname(op.abspath(affine))
    resolved = tc.resolve_affine_to_lta(
        affine=affine,
        out_dir=affine_out_dir,
        src_vol=affine_src or files[0],
        trg_vol=affine_trg,
        fs_subject=fs_subject,
        fs_subjects_dir=fs_subjects_dir,
        fs_module=fs_module,
        ants_module=ants_module,
        execute=execute,
        overwrite=overwrite,
    )
    for line in resolved["log"]:
        console.print(f"  {line}")
    reg = resolved["lta_path"]

    if execute and not op.isfile(reg):
        console.print(f"\n[red]✗ resolved .lta not found: {reg}[/red]")
        raise typer.Exit(1)

    # ── Step 2: project files ───────────────────────────────────────────────────
    console.print("\n[bold]vol2surf[/bold]")

    run_kwargs = [
        dict(
            src=f, reg=reg, fs_subject=fs_subject, fs_subjects_dir=fs_subjects_dir,
            hemis=hemis, surf=surf, interp=interp, proj_frac=proj_frac, proj_frac_avg=pfa,
            trgsubject=trgsubject, space_label_in=space_label_in,
            space_label_out=space_label_out, out_ext=out_ext, fs_module=fs_module,
            execute=execute, overwrite=overwrite,
        )
        for f in files
    ]

    results: list[tuple[str, bool, float]] = []
    workers = len(run_kwargs) if n_jobs == -1 else n_jobs

    if workers <= 1 or len(run_kwargs) == 1:
        for kw in run_kwargs:
            console.print(f"\n[bold]{op.basename(kw['src'])}[/bold]")
            ok, elapsed, log = tc.project_file(**kw)
            for line in log:
                console.print(line)
            results.append((op.basename(kw["src"]), ok, elapsed))
    else:
        console.print(f"\n[dim]Projecting {len(run_kwargs)} file(s) across {min(workers, len(run_kwargs))} worker(s) …[/dim]")
        with ProcessPoolExecutor(max_workers=min(workers, len(run_kwargs))) as pool:
            future_to_f = {pool.submit(tc.project_file, **kw): kw["src"] for kw in run_kwargs}
            for future in as_completed(future_to_f):
                f = future_to_f[future]
                try:
                    ok, elapsed, log = future.result()
                except Exception as exc:
                    console.print(f"  [red]✗ {op.basename(f)}: {exc}[/red]")
                    ok, elapsed, log = False, 0.0, []
                console.print(f"\n[bold]{op.basename(f)}[/bold]")
                for line in log:
                    console.print(line)
                results.append((op.basename(f), ok, elapsed))

    console.print("\n")
    summary = Table(title="Results", show_lines=False)
    summary.add_column("file")
    summary.add_column("status")
    summary.add_column("wall time", justify="right")
    for fname, ok, elapsed in results:
        summary.add_row(
            fname,
            "[green]✓[/green]" if ok else "[red]✗[/red]",
            tc.fmt_time(elapsed) if elapsed > 0 else "—",
        )
    console.print(summary)

    if not execute:
        console.print("[yellow]Above is the plan. Re-run with --execute to apply.[/yellow]")
        return

    n_failed = sum(1 for _f, ok, _e in results if not ok)
    if n_failed:
        console.print(f"\n[red]{n_failed} item(s) failed.[/red]")
        raise typer.Exit(1)
    console.print("\n[green]All projections completed successfully.[/green]")


if __name__ == "__main__":
    app()
