#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) Yongning Lei 2025-2026
# Apache-2.0 license
# -----------------------------------------------------------------------------
"""
01_register.py

Generic registration utility: register one moving volume onto a target,
producing an affine/.lta tagged with `_desc-{method}_`.

  --method bbregister
      Direct registration of `--mov` onto a FreeSurfer subject.
        requires: --fs-subject, --fs-subjects-dir
        output  : {out-prefix}_desc-bbregister_reg.lta

  --method ants
      Rigid ANTs registration of `--mov` onto `--fixed` (any volume), then
      converted to .lta.
        requires: --fixed
        output  : {out-prefix}_desc-ants_0GenericAffine.mat
                  {out-prefix}_desc-ants_reg.lta
      If --fs-subject/--fs-subjects-dir are also given, the .lta's `subject`
      field is set (needed by mri_vol2surf when `--fixed` is a FreeSurfer
      volume, e.g. orig.mgz).

Usage
-----
    # BOLD reference -> fsnative, direct BBR
    python 01_register.py \\
        --method bbregister --bbr-contrast bold \\
        --mov sub-pilot01_ses-01_task-x_boldref.nii.gz \\
        --fs-subject sub-pilot01 --fs-subjects-dir .../derivatives/freesurfer \\
        --out-prefix .../sub-pilot01_ses-01_from-boldref_to-fsnative \\
        --execute

    # T1w -> fsnative, ANTs rigid
    python 01_register.py \\
        --method ants \\
        --mov sub-pilot01_ses-01_desc-preproc_T1w.nii.gz \\
        --fixed .../freesurfer/sub-pilot01/mri/orig.mgz \\
        --fs-subject sub-pilot01 --fs-subjects-dir .../derivatives/freesurfer \\
        --out-prefix .../sub-pilot01_ses-01_from-T1w_to-fsnative \\
        --execute
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

import transform_common as tc

console = Console()
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def main(
    mov: str = typer.Option(..., "--mov", help="Moving volume to register"),
    method: str = typer.Option(..., "--method", help="Registration method: bbregister | ants"),
    out_prefix: str = typer.Option(..., "--out-prefix", help="Output path prefix (directory + basename stem, no extension)"),
    fixed: Optional[str] = typer.Option(None, "--fixed", help="Fixed/target volume (required for --method ants)"),
    fs_subject: Optional[str] = typer.Option(None, "--fs-subject", help="FreeSurfer subject name (required for bbregister; optional for ants to set the .lta subject field)"),
    fs_subjects_dir: Optional[str] = typer.Option(None, "--fs-subjects-dir", "--fs-sd", help="FreeSurfer SUBJECTS_DIR"),
    bbr_contrast: str = typer.Option("t1", "--bbr-contrast", help="bbregister contrast: t1 | t2 | bold"),
    bbr_init: str = typer.Option("header", "--bbr-init", help="bbregister init: header | fsl | coreg"),
    fs_module: str = typer.Option("freesurfer/7.3.2", "--fs-module"),
    ants_module: str = typer.Option("ants", "--ants-module"),
    execute: bool = typer.Option(False, "--execute", help="Run registration. Without this flag only the plan is printed."),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Register `--mov` and write an affine/.lta tagged `_desc-{method}_`."""

    if method not in tc.REG_METHODS:
        console.print(f"[red]✗ Unknown --method: {method}  Valid: {tc.REG_METHODS}[/red]")
        raise typer.Exit(1)
    if method == "bbregister" and not (fs_subject and fs_subjects_dir):
        console.print("[red]✗ --method bbregister requires --fs-subject and --fs-subjects-dir[/red]")
        raise typer.Exit(1)
    if method == "ants" and not fixed:
        console.print("[red]✗ --method ants requires --fixed[/red]")
        raise typer.Exit(1)

    tbl = Table(title="01_register", show_lines=False)
    tbl.add_column("Parameter", style="cyan")
    tbl.add_column("Value")
    tbl.add_row("mov", mov)
    tbl.add_row("method", method)
    tbl.add_row("fixed", fixed or "(none)")
    tbl.add_row("fs_subject", fs_subject or "(none)")
    tbl.add_row("fs_subjects_dir", fs_subjects_dir or "(none)")
    if method == "bbregister":
        tbl.add_row("bbr_contrast", bbr_contrast)
        tbl.add_row("bbr_init", bbr_init)
    tbl.add_row("out_prefix", out_prefix)
    tbl.add_row("execute", str(execute))
    console.print(tbl)

    if not execute:
        console.print("\n[yellow bold]DRY RUN — pass --execute to run registration[/yellow bold]\n")

    result = tc.register_generic(
        mov=mov,
        method=method,
        out_prefix=out_prefix,
        fixed=fixed,
        fs_subject=fs_subject,
        fs_subjects_dir=fs_subjects_dir,
        bbr_contrast=bbr_contrast,
        bbr_init=bbr_init,
        fs_module=fs_module,
        ants_module=ants_module,
        execute=execute,
        overwrite=overwrite,
    )
    for line in result["log"]:
        console.print(f"  {line}")

    console.print(f"\n[bold]lta_path:[/bold] {result['lta_path']}")


if __name__ == "__main__":
    app()
