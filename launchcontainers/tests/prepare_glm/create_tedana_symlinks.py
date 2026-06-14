#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) Yongning Lei 2025-2026  —  Apache-2.0 license
# -----------------------------------------------------------------------------
"""
create_tedana_symlinks.py

Create symlinks in the fMRIprep func directory pointing to tedana outputs
so the BIDSLayout-based GLM code can find them via ``--bold-desc``.

Symlinks created
----------------
For each requested space × desc × run:

  fsnative / fsaverage (from project_to_spaces.py output):
    {fp_func}/sub-{sub}_ses-{ses}_task-{task}_acq-ME_run-{run}
              _hemi-{LR}_space-{space}_desc-{desc}_bold.func.gii
    → {tedana_func}/same_filename

  T1w (from project_to_spaces.py output, used for volumetric GLM):
    {fp_func}/sub-{sub}_ses-{ses}_task-{task}_acq-ME_run-{run}
              _space-T1w_desc-{desc}_bold.nii.gz
    → {tedana_func}/same_filename

Usage
-----
Dry-run (print what would be linked)::

    python create_tedana_symlinks.py \\
        -fp .../fmriprep-25.1.4_IRpilot \\
        -tedana .../tedana-26.0.3/analysis-default_ica \\
        -s pilot02,01 --tasks BfLocVideo

Create links::

    python create_tedana_symlinks.py ... --execute
"""

from __future__ import annotations

import glob
import os
import os.path as op
import re
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

SURFACE_EXTS = {"fsnative": ".func.gii", "fsaverage": ".func.gii"}
VOL_EXT = ".nii.gz"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bids_prefix(sub, ses, task, acq, run) -> str:
    parts = [f"sub-{sub}", f"ses-{ses}", f"task-{task}"]
    if acq:
        parts.append(f"acq-{acq}")
    if run:
        parts.append(f"run-{run}")
    return "_".join(parts) + "_"


def _find_runs(tedana_func: str, sub: str, ses: str, task: str, acq: str) -> list[str]:
    pat = op.join(
        tedana_func,
        f"sub-{sub}_ses-{ses}_task-{task}_acq-{acq}_run-*_desc-denoised_bold.nii.gz",
    )
    runs = set()
    for f in glob.glob(pat):
        m = re.search(r"_run-(\w+)[_.]", op.basename(f))
        if m:
            runs.add(m.group(1))
    return sorted(runs)


def _make_symlink(src: str, dst: str, execute: bool, overwrite: bool) -> str:
    """Create (or report) one symlink. Returns status string."""
    if not op.isfile(src):
        return "[red]src missing[/red]"
    if op.lexists(dst):
        if not overwrite:
            return "[yellow]exists[/yellow]"
        os.remove(dst)
    if not execute:
        return "[dim]dry-run[/dim]"
    os.symlink(src, dst)
    return "[green]✓ linked[/green]"


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def _process_subses(
    sub: str,
    ses: str,
    task: str,
    acq: str,
    tedana_func: str,
    fp_func: str,
    descs: list[str],
    spaces: list[str],
    execute: bool,
    overwrite: bool,
) -> list[dict]:
    runs = _find_runs(tedana_func, sub, ses, task, acq)
    if not runs:
        console.print(f"  [yellow]⚠ no tedana runs found for acq-{acq}[/yellow]")
        return []

    rows = []
    for run in runs:
        prefix = _bids_prefix(sub, ses, task, acq, run)

        for desc in descs:
            # ── Volumetric (T1w space, from project_to_spaces.py) ────────
            src_vol = op.join(tedana_func, f"{prefix}space-T1w_desc-{desc}_bold{VOL_EXT}")
            dst_vol = op.join(fp_func, f"{prefix}space-T1w_desc-{desc}_bold{VOL_EXT}")
            status = _make_symlink(src_vol, dst_vol, execute, overwrite)
            console.print(
                f"  run-{run}  desc-{desc}  vol   {status}\n"
                f"    src: {op.basename(src_vol)}\n"
                f"    dst: {dst_vol}"
            )
            rows.append(
                {"run": run, "desc": desc, "space": "T1w", "status": status}
            )

            # ── Surface spaces ────────────────────────────────────────────
            for space in spaces:
                if space not in SURFACE_EXTS:
                    continue
                ext = SURFACE_EXTS[space]
                for hemi in ("L", "R"):
                    fname = f"{prefix}hemi-{hemi}_space-{space}_desc-{desc}_bold{ext}"
                    src = op.join(tedana_func, fname)
                    dst = op.join(fp_func, fname)
                    status = _make_symlink(src, dst, execute, overwrite)
                    console.print(
                        f"  run-{run}  desc-{desc}  {space} hemi-{hemi}  {status}"
                    )
                    rows.append(
                        {
                            "run": run,
                            "desc": desc,
                            "space": f"{space} hemi-{hemi}",
                            "status": status,
                        }
                    )
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    fp_dir: str = typer.Option(
        ..., "-fp", help="fMRIprep derivatives directory (absolute path)"
    ),
    tedana_dir: str = typer.Option(
        ..., "-tedana", help="tedana analysis root (analysis-{name}/)"
    ),
    single: Optional[str] = typer.Option(None, "-s", help="sub,ses  e.g.  pilot02,01"),
    batch_file: Optional[str] = typer.Option(
        None, "-f", help="Batch file: sub,ses per line"
    ),
    tasks: str = typer.Option(..., "--tasks", "-t", help="Comma-separated task names"),
    acq: str = typer.Option("ME", "--acq", help="Acquisition label"),
    descs: str = typer.Option(
        "denoised,optcom",
        "--descs",
        help="Comma-separated desc labels to symlink",
    ),
    spaces: str = typer.Option(
        "fsnative,fsaverage",
        "--spaces",
        help="Surface spaces to symlink (fsnative, fsaverage)",
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Create symlinks. Without this flag only the plan is printed.",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Remove and re-create existing symlinks"
    ),
) -> None:
    """Create symlinks in fmriprep func dir pointing to tedana outputs."""

    desc_list = [d.strip() for d in descs.split(",") if d.strip()]
    space_list = [s.strip() for s in spaces.split(",") if s.strip()]
    task_list = [t.strip() for t in tasks.split(",") if t.strip()]

    subses_pairs: list[tuple[str, str]] = []
    if single:
        p = single.split(",")
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
        console.print("[red]✗ provide -s sub,ses  or  -f file[/red]")
        raise typer.Exit(1)

    tbl = Table(title="create_tedana_symlinks", show_lines=False)
    tbl.add_column("Parameter", style="cyan")
    tbl.add_column("Value")
    tbl.add_row("fp_dir", fp_dir)
    tbl.add_row("tedana_dir", tedana_dir)
    tbl.add_row("tasks", ", ".join(task_list))
    tbl.add_row("acq", acq)
    tbl.add_row("descs", ", ".join(desc_list))
    tbl.add_row("spaces", ", ".join(space_list))
    tbl.add_row("execute", str(execute))
    console.print(tbl)

    if not execute:
        console.print(
            "\n[yellow bold]DRY RUN — pass --execute to create symlinks[/yellow bold]\n"
        )

    all_rows = []
    for sub, ses in subses_pairs:
        for task in task_list:
            console.print(f"\n[bold cyan]sub-{sub}  ses-{ses}  task-{task}[/bold cyan]")
            tedana_func = op.join(tedana_dir, f"sub-{sub}", f"ses-{ses}", "func")
            fp_func = op.join(fp_dir, f"sub-{sub}", f"ses-{ses}", "func")

            if not op.isdir(tedana_func):
                console.print(
                    f"  [red]✗ tedana func dir not found: {tedana_func}[/red]"
                )
                continue
            if not op.isdir(fp_func):
                console.print(f"  [red]✗ fmriprep func dir not found: {fp_func}[/red]")
                continue

            rows = _process_subses(
                sub=sub,
                ses=ses,
                task=task,
                acq=acq,
                tedana_func=tedana_func,
                fp_func=fp_func,
                descs=desc_list,
                spaces=space_list,
                execute=execute,
                overwrite=overwrite,
            )
            all_rows.extend(rows)

    n_linked = sum(1 for r in all_rows if "✓" in r["status"])
    n_missing = sum(1 for r in all_rows if "missing" in r["status"])
    console.print(
        f"\n[bold]Total: {n_linked} linked, {n_missing} src missing, "
        f"{len(all_rows) - n_linked - n_missing} skipped[/bold]"
    )

    if not execute:
        console.print("[yellow]Re-run with --execute to create the symlinks.[/yellow]")


if __name__ == "__main__":
    app()
