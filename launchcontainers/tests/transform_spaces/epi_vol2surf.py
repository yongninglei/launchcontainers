#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) Yongning Lei 2025-2026
# Apache-2.0 license
# -----------------------------------------------------------------------------
"""
epi_vol2surf.py

Orchestrator for the EPI vol2surf pipeline. For each sub/ses, runs:

    0_check_affine - reuse a cached .lta, or convert fMRIprep's
                      from-T1w_to-fsnative xfm (--src-space T1w only)
    1_do_register  - if step 0 found nothing, compute a new .lta
                      via bbregister or ANTs (--reg-method / --mov)
    2_transform    - project every file matching --input-glob to the
                      surface with mri_vol2surf

This is the batch entry point; the individual steps (0_check_affine.py,
1_do_register.py, 2_do_transformation.py) can also be run standalone.

Usage
-----
Dry-run (print plan, no execution)::

    python epi_vol2surf.py \\
        --src-space T1w \\
        --fp-dir .../derivatives/fmriprep-25.1.4_IRpilot \\
        --fs-subjects-dir .../derivatives/freesurfer \\
        --cache-dir .../analysis-NAME/sub-{sub}/ses-{ses} \\
        --input-glob ".../analysis-NAME/sub-{sub}/ses-{ses}/sub-{sub}_ses-{ses}_*_space-T1w_*_statmap.nii.gz" \\
        -s pilot01,01

Execute, with native-space EPI registered via bbregister::

    python epi_vol2surf.py \\
        --src-space native --reg-method bbregister \\
        --mov ".../sub-{sub}_ses-{ses}_task-x_boldref.nii.gz" \\
        --fs-subjects-dir .../derivatives/freesurfer \\
        --cache-dir .../analysis-NAME/sub-{sub}/ses-{ses} \\
        --input-glob ".../sub-{sub}_ses-{ses}_task-x_space-native_*.nii.gz" \\
        --space-label-in native \\
        -s pilot01,01 --execute
"""

from __future__ import annotations

import os.path as op
import time
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

import transform_common as tc

console = Console()
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def main(
    src_space: str = typer.Option(..., "--src-space", help="Source space of the EPI data: T1w | native"),
    fp_dir: str = typer.Option("", "--fp-dir", help="fMRIprep derivatives directory (required for --src-space T1w, or ANTs+native)"),
    fs_subjects_dir: str = typer.Option(..., "--fs-subjects-dir", "--fs-sd", help="FreeSurfer SUBJECTS_DIR"),
    fs_subject_template: str = typer.Option("sub-{sub}", "--fs-subject", help="FreeSurfer subject name template. Placeholders: {sub} {ses}"),
    cache_dir: str = typer.Option(..., "--cache-dir", help="Directory (template, placeholders {sub} {ses}) where the resolved .lta is cached"),
    input_glob: str = typer.Option(..., "--input-glob", help="Glob pattern (placeholders {sub} {ses}) for volumes to project"),
    single: Optional[str] = typer.Option(None, "-s", help="sub,ses  e.g.  pilot01,01"),
    batch_file: Optional[str] = typer.Option(None, "-f", help="Batch file: one sub,ses per line"),
    reg_method: str = typer.Option("bbregister", "--reg-method", help="Registration method if step 0 finds nothing: bbregister | ants"),
    mov: str = typer.Option("", "--mov", help="Moving volume template (placeholders {sub} {ses}) for 1_do_register. Required if --src-space native, or if step 0 finds nothing for T1w."),
    bbr_contrast: Optional[str] = typer.Option(None, "--bbr-contrast", help="bbregister contrast: t1 | t2 | bold (default: t1 for T1w, bold for native)"),
    bbr_init: str = typer.Option("header", "--bbr-init", help="bbregister init: header | fsl | coreg"),
    hemi: str = typer.Option("LR", "--hemi", help="Hemispheres to project: L | R | LR"),
    surf: str = typer.Option("white", "--surf", help="Surface to sample: white | pial | midthickness"),
    interp: str = typer.Option("trilinear", "--interp", help="Interpolation: nearest | trilinear"),
    proj_frac: Optional[float] = typer.Option(None, "--proj-frac", help="--projfrac value"),
    proj_frac_avg: Optional[str] = typer.Option(None, "--proj-frac-avg", help="start,stop,step for --projfrac-avg (overrides --proj-frac)"),
    trgsubject: Optional[str] = typer.Option(None, "--trgsubject", help="Resample to this FS subject (e.g. fsaverage)"),
    space_label_in: Optional[str] = typer.Option(None, "--space-label-in", help="space-* label to replace in input filenames (default: --src-space)"),
    space_label_out: str = typer.Option("fsnative", "--space-label-out", help="space-* label to use in output filenames (overridden by --trgsubject)"),
    out_ext: str = typer.Option(".func.gii", "--out-ext", help="Output file extension"),
    fs_module: str = typer.Option("freesurfer/7.3.2", "--fs-module"),
    ants_module: str = typer.Option("ants", "--ants-module"),
    execute: bool = typer.Option(False, "--execute", help="Run all steps. Without this flag only the plan is printed."),
    overwrite: bool = typer.Option(False, "--overwrite"),
    n_jobs: int = typer.Option(1, "--n-jobs", "-J", help="Files to project in parallel per sub/ses (1=sequential, -1=all)"),
) -> None:
    """Run check_affine -> do_register (if needed) -> vol2surf for each sub/ses."""

    if src_space not in tc.SRC_SPACES:
        console.print(f"[red]✗ Unknown --src-space: {src_space}  Valid: {tc.SRC_SPACES}[/red]")
        raise typer.Exit(1)
    if reg_method not in tc.REG_METHODS:
        console.print(f"[red]✗ Unknown --reg-method: {reg_method}  Valid: {tc.REG_METHODS}[/red]")
        raise typer.Exit(1)

    space_label_in = space_label_in or src_space

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

    pfa: Optional[tuple[float, float, float]] = None
    if proj_frac_avg:
        parts = proj_frac_avg.split(",")
        if len(parts) != 3:
            console.print("[red]✗ --proj-frac-avg expects start,stop,step[/red]")
            raise typer.Exit(1)
        pfa = tuple(float(p) for p in parts)  # type: ignore[assignment]

    hemi = hemi.upper()
    hemis = [h for h in ("L", "R") if h in hemi]
    if not hemis:
        console.print(f"[red]✗ --hemi must be L, R, or LR (got {hemi})[/red]")
        raise typer.Exit(1)

    tbl = Table(title="epi_vol2surf", show_lines=False)
    tbl.add_column("Parameter", style="cyan")
    tbl.add_column("Value")
    tbl.add_row("src_space", src_space)
    tbl.add_row("reg_method (fallback)", reg_method)
    tbl.add_row("fs_subjects_dir", fs_subjects_dir)
    tbl.add_row("cache_dir", cache_dir)
    tbl.add_row("input_glob", input_glob)
    tbl.add_row("hemi", "".join(hemis))
    tbl.add_row("surf", surf)
    tbl.add_row("interp", interp)
    tbl.add_row("desc", tc.desc_label(surf, interp))
    tbl.add_row("proj_frac", str(proj_frac) if proj_frac is not None else "(none)")
    tbl.add_row("proj_frac_avg", str(pfa) if pfa else "(none)")
    tbl.add_row("trgsubject", trgsubject or "(none)")
    tbl.add_row("space_label_in", space_label_in)
    tbl.add_row("space_label_out", trgsubject or space_label_out)
    tbl.add_row("execute", str(execute))
    console.print(tbl)

    if not execute:
        console.print("\n[yellow bold]DRY RUN — pass --execute to run all steps[/yellow bold]\n")

    import glob as globmod

    results: list[tuple[str, str, str, bool, float]] = []
    t_total_start = time.perf_counter()

    for sub, ses in subses_pairs:
        console.print(f"\n[bold cyan]sub-{sub}  ses-{ses}[/bold cyan]")
        fs_subject = fs_subject_template.format(sub=sub, ses=ses)
        sub_cache_dir = cache_dir.format(sub=sub, ses=ses)

        # ── Step 0: check for an existing affine ────────────────────────────
        console.print("  [bold]Step 0 - check_affine[/bold]")
        chk = tc.check_affine(
            sub, ses, src_space, fp_dir, fs_subjects_dir, fs_subject,
            sub_cache_dir, fs_module, execute, overwrite,
        )
        for line in chk["log"]:
            console.print(f"    {line}")

        reg_lta = chk["lta_path"]

        # ── Step 1: register if needed ───────────────────────────────────────
        if chk["needs_register"]:
            console.print("  [bold]Step 1 - do_register[/bold]")
            if not mov:
                console.print("    [red]✗ no affine found and --mov not given; cannot register[/red]")
                results.append((sub, ses, "(reg)", False, 0.0))
                continue
            mov_path = mov.format(sub=sub, ses=ses)
            reg = tc.do_register(
                sub, ses, src_space, reg_method, mov_path, fp_dir, fs_subjects_dir,
                fs_subject, sub_cache_dir, bbr_contrast, bbr_init, fs_module,
                ants_module, execute, overwrite,
            )
            for line in reg["log"]:
                console.print(f"    {line}")
            reg_lta = reg["lta_path"]

        if execute and not op.isfile(reg_lta):
            console.print(f"  [red]✗ registration .lta not found after step 0/1: {reg_lta}[/red]")
            results.append((sub, ses, "(reg)", False, 0.0))
            continue

        # ── Step 2: project files ────────────────────────────────────────────
        console.print("  [bold]Step 2 - vol2surf[/bold]")
        pattern = op.expanduser(input_glob.format(sub=sub, ses=ses))
        files = sorted(globmod.glob(pattern))
        if not files:
            console.print(f"    [yellow]⚠ no files matched: {pattern}[/yellow]")
            continue
        console.print(f"    Found {len(files)} file(s)")

        run_kwargs = [
            dict(
                src=f, reg=reg_lta, fs_subject=fs_subject, fs_subjects_dir=fs_subjects_dir,
                hemis=hemis, surf=surf, interp=interp, proj_frac=proj_frac, proj_frac_avg=pfa,
                trgsubject=trgsubject, space_label_in=space_label_in,
                space_label_out=space_label_out, out_ext=out_ext, fs_module=fs_module,
                execute=execute, overwrite=overwrite,
            )
            for f in files
        ]

        workers = len(run_kwargs) if n_jobs == -1 else n_jobs
        if workers <= 1 or len(run_kwargs) == 1:
            for kw in run_kwargs:
                console.print(f"    [bold]{op.basename(kw['src'])}[/bold]")
                ok, elapsed, log = tc.project_file(**kw)
                for line in log:
                    console.print(line)
                results.append((sub, ses, op.basename(kw["src"]), ok, elapsed))
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            console.print(f"    [dim]Projecting {len(run_kwargs)} file(s) across {min(workers, len(run_kwargs))} worker(s) …[/dim]")
            with ProcessPoolExecutor(max_workers=min(workers, len(run_kwargs))) as pool:
                future_to_f = {pool.submit(tc.project_file, **kw): kw["src"] for kw in run_kwargs}
                for future in as_completed(future_to_f):
                    f = future_to_f[future]
                    try:
                        ok, elapsed, log = future.result()
                    except Exception as exc:
                        console.print(f"    [red]✗ {op.basename(f)}: {exc}[/red]")
                        ok, elapsed, log = False, 0.0, []
                    console.print(f"    [bold]{op.basename(f)}[/bold]")
                    for line in log:
                        console.print(line)
                    results.append((sub, ses, op.basename(f), ok, elapsed))

    t_total = time.perf_counter() - t_total_start

    console.print("\n")
    summary = Table(title="Results", show_lines=False)
    summary.add_column("sub")
    summary.add_column("ses")
    summary.add_column("file")
    summary.add_column("status")
    summary.add_column("wall time", justify="right")
    for sub, ses, fname, ok, elapsed in results:
        summary.add_row(
            sub, ses, fname,
            "[green]✓[/green]" if ok else "[red]✗[/red]",
            tc.fmt_time(elapsed) if elapsed > 0 else "—",
        )
    console.print(summary)

    if not execute:
        console.print("[yellow]Above is the plan. Re-run with --execute to apply.[/yellow]")
        return

    console.print(f"Total wall time: [bold]{tc.fmt_time(t_total)}[/bold]")
    n_failed = sum(1 for *_, ok, _e in results if not ok)
    if n_failed:
        console.print(f"\n[red]{n_failed} item(s) failed.[/red]")
        raise typer.Exit(1)
    console.print("\n[green]All projections completed successfully.[/green]")


if __name__ == "__main__":
    app()
