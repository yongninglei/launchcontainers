#!/usr/bin/env python3
"""
Copy T2w NIfTI (+JSON sidecar) from raw_nifti to BIDS for each sub/ses.

Looks for files matching:
    <raw_nifti>/sub-{sub}/ses-{ses}/anat/*_T2w.nii.gz

and copies each pair (nii.gz + json) to:
    <bids>/sub-{sub}/ses-{ses}/anat/

Skips files that already exist in BIDS unless --force is passed.

Usage
-----
    # dry-run, single session
    python prepare_T2w.py --raw-nifti /path/raw_nifti --bids /path/BIDS -s 01,02

    # batch from subseslist, execute
    python prepare_T2w.py --raw-nifti /path/raw_nifti --bids /path/BIDS \\
        -f subseslist.txt --execute

    # overwrite existing files
    python prepare_T2w.py --raw-nifti /path/raw_nifti --bids /path/BIDS \\
        -s 01,02 --execute --force
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from launchcontainers.utils import parse_subses_list

app = typer.Typer(add_completion=False)
console = Console()


def copy_t2w_session(
    raw_nifti: Path,
    bids: Path,
    sub: str,
    ses: str,
    dry_run: bool,
    force: bool,
) -> dict:
    """Copy T2w files for one session. Returns a result summary dict."""
    src_anat = raw_nifti / f"sub-{sub}" / f"ses-{ses}" / "anat"
    dst_anat = bids / f"sub-{sub}" / f"ses-{ses}" / "anat"

    result = {"sub": sub, "ses": ses, "copied": 0, "skipped": 0, "missing": 0}

    if not src_anat.is_dir():
        console.print(
            f"  [yellow]sub-{sub} ses-{ses}:[/] anat dir not found in raw_nifti — skip"
        )
        result["missing"] = 1
        return result

    niftis = sorted(src_anat.glob("*_T2w.nii.gz"))
    if not niftis:
        console.print(f"  [yellow]sub-{sub} ses-{ses}:[/] no T2w NIfTI found — skip")
        result["missing"] = 1
        return result

    for src_nii in niftis:
        src_json = src_nii.with_suffix("").with_suffix(".json")
        dst_nii = dst_anat / src_nii.name
        dst_json = dst_anat / src_json.name

        for src, dst in [(src_nii, dst_nii), (src_json, dst_json)]:
            if not src.exists():
                continue
            if dst.exists() and not force:
                console.print(f"  [dim]SKIP[/]  {dst.name} (already exists)")
                result["skipped"] += 1
                continue
            tag = "[dim][DRY][/]" if dry_run else "[green][COPY][/]"
            console.print(f"  {tag} {src.name}  →  {dst.relative_to(bids)}")
            if not dry_run:
                dst_anat.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            result["copied"] += 1

    return result


@app.command()
def main(
    raw_nifti: Path = typer.Option(
        ..., "--raw-nifti", "-r", help="Raw NIfTI root directory."
    ),
    bids: Path = typer.Option(..., "--bids", "-b", help="BIDS root directory."),
    subses: Optional[str] = typer.Option(
        None, "--subses", "-s", help="Single sub,ses pair e.g. 01,02"
    ),
    subses_file: Optional[Path] = typer.Option(
        None, "--file", "-f", help="Subseslist file (CSV/TSV with sub,ses columns)."
    ),
    execute: bool = typer.Option(
        False, "--execute", help="Apply changes. Without this flag runs as dry-run."
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite files that already exist in BIDS."
    ),
) -> None:
    """Copy T2w NIfTI (+JSON sidecar) from raw_nifti to BIDS."""

    if subses_file is not None:
        pairs = parse_subses_list(subses_file)
        console.print(f"[dim]Loaded {len(pairs)} session(s) from {subses_file.name}[/]")
    elif subses is not None:
        parts = [p.strip().zfill(2) for p in subses.split(",")]
        if len(parts) != 2:
            console.print("[red]--subses must be sub,ses e.g. 01,02[/]")
            raise typer.Exit(1)
        pairs = [(parts[0], parts[1])]
    else:
        console.print("[red]Provide --subses or --file.[/]")
        raise typer.Exit(1)

    dry_run = not execute
    mode = "[bold yellow]DRY-RUN[/]" if dry_run else "[bold green]EXECUTE[/]"
    console.rule(f"prepare_T2w  {mode}")

    total_copied = total_skipped = total_missing = 0

    for sub, ses in pairs:
        console.print(f"\n[bold cyan]sub-{sub}  ses-{ses}[/]")
        r = copy_t2w_session(raw_nifti, bids, sub, ses, dry_run, force)
        total_copied += r["copied"]
        total_skipped += r["skipped"]
        total_missing += r["missing"]

    console.rule("Summary")
    console.print(
        f"  sessions: {len(pairs)}  |  "
        f"copied: {total_copied}  |  "
        f"skipped: {total_skipped}  |  "
        f"not found: {total_missing}"
    )
    if dry_run:
        console.print("\n[dim]Dry-run — pass [bold]--execute[/bold] to apply.[/]")


if __name__ == "__main__":
    app()
