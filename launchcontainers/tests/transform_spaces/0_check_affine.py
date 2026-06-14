#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) Yongning Lei 2025-2026
# Apache-2.0 license
# -----------------------------------------------------------------------------
"""
0_check_affine.py

Step 0 of the EPI vol2surf pipeline: determine whether a usable
volume->fsnative .lta registration already exists for a given
sub/ses + source space.

  --src-space T1w
      fMRIprep already computes a sub-*_from-T1w_to-fsnative_mode-image_xfm.txt
      (ITK affine). If found (and not already cached as .lta), this step
      converts it to .lta via `lta_convert` and caches it - this is just a
      format conversion of an existing transform, not a new registration.

  --src-space native
      There is no fMRIprep-provided transform from EPI-native space to
      fsnative. Unless a .lta is already cached from a previous run, this
      step reports "not found" and 1_do_register.py must be run.

In all cases, a previously-cached .lta (sub-*_from-{src_space}_to-fsnative_reg.lta
in --cache-dir) is reused unless --overwrite is given.

Usage
-----
    python 0_check_affine.py \\
        --src-space T1w \\
        --fp-dir .../derivatives/fmriprep-25.1.4_IRpilot \\
        --fs-subjects-dir .../derivatives/freesurfer \\
        --cache-dir .../derivatives/l1_surface/analysis-NAME/sub-pilot01/ses-01 \\
        -s pilot01,01 --execute
"""

from __future__ import annotations

import os.path as op
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
    fp_dir: str = typer.Option(..., "--fp-dir", help="fMRIprep derivatives directory (absolute path)"),
    fs_subjects_dir: str = typer.Option(..., "--fs-subjects-dir", "--fs-sd", help="FreeSurfer SUBJECTS_DIR"),
    fs_subject_template: str = typer.Option("sub-{sub}", "--fs-subject", help="FreeSurfer subject name template. Placeholders: {sub} {ses}"),
    cache_dir: str = typer.Option(..., "--cache-dir", help="Directory where the resolved .lta is cached / looked up"),
    single: Optional[str] = typer.Option(None, "-s", help="sub,ses  e.g.  pilot01,01"),
    batch_file: Optional[str] = typer.Option(None, "-f", help="Batch file: one sub,ses per line"),
    fs_module: str = typer.Option("freesurfer/7.3.2", "--fs-module"),
    execute: bool = typer.Option(False, "--execute", help="Perform format conversions. Without this flag only the plan is printed."),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Check (and where possible resolve) a volume->fsnative .lta."""

    if src_space not in tc.SRC_SPACES:
        console.print(f"[red]✗ Unknown --src-space: {src_space}  Valid: {tc.SRC_SPACES}[/red]")
        raise typer.Exit(1)

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

    if not execute:
        console.print("\n[yellow bold]DRY RUN — pass --execute to perform conversions[/yellow bold]\n")

    summary = Table(title="0_check_affine", show_lines=False)
    summary.add_column("sub")
    summary.add_column("ses")
    summary.add_column("status")
    summary.add_column("source")
    summary.add_column("lta_path")

    for sub, ses in subses_pairs:
        console.print(f"\n[bold cyan]sub-{sub}  ses-{ses}[/bold cyan]")
        fs_subject = fs_subject_template.format(sub=sub, ses=ses)
        result = tc.check_affine(
            sub, ses, src_space, fp_dir, fs_subjects_dir, fs_subject,
            cache_dir, fs_module, execute, overwrite,
        )
        for line in result["log"]:
            console.print(f"  {line}")
        status = "[green]found[/green]" if result["found"] else "[yellow]needs register[/yellow]"
        summary.add_row(sub, ses, status, result["source"], result["lta_path"])

    console.print("\n")
    console.print(summary)


if __name__ == "__main__":
    app()
