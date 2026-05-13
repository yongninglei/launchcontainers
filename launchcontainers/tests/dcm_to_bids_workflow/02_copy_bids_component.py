"""
02_copy_bids_component.py
-------------------------
Copy top-level BIDS components from a raw_nifti source tree into the target
BIDS directory.

Components copied
-----------------
  dataset_description.json
  participants.json
  participants.tsv
  README

Pre-requisite
-------------
Run after ``00_merge_split_ses_reording_runs.py`` and
``01_drop_duplicated_sbrefs.py`` so that the source tree is already clean.
Run this BEFORE prepare_anat / prepare_func / prepare_fmap.

Usage
-----
  # dry-run (default):
  python 02_copy_bids_component.py -b /raw_nifti -t /BIDS

  # execute:
  python 02_copy_bids_component.py -b /raw_nifti -t /BIDS --execute

  # overwrite existing files:
  python 02_copy_bids_component.py -b /raw_nifti -t /BIDS --execute --force
"""

from __future__ import annotations

import os
import os.path as op
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(pretty_exceptions_show_locals=False)

BIDS_COMPONENTS = [
    "dataset_description.json",
    "participants.json",
    "participants.tsv",
    "README",
]


def _copy_component(
    src_root: str, dst_root: str, name: str, force: bool, dry_run: bool
) -> str:
    """
    Copy a single top-level BIDS component file.
    Returns status string: "ok" | "skip" | "dry" | "no_src" | "error".
    """
    src = op.join(src_root, name)
    dst = op.join(dst_root, name)

    if not op.exists(src):
        return "no_src"

    if op.exists(dst) and not force:
        return "skip"

    if dry_run:
        return "dry"

    try:
        os.makedirs(dst_root, exist_ok=True)
        shutil.copy2(src, dst)
        os.chmod(dst, 0o755)
        return "ok"
    except Exception as exc:
        console.print(f"  [red][ERROR][/] {name}: {exc}")
        return "error"


@app.command()
def main(
    src_bids: Path = typer.Option(
        ..., "--src", "-b", help="Source BIDS directory (e.g. raw_nifti root)."
    ),
    dst_bids: Path = typer.Option(..., "--dst", "-t", help="Target BIDS directory."),
    force: bool = typer.Option(
        False, "--force", help="Overwrite existing files in the target."
    ),
    dry_run: bool = typer.Option(
        True, "--dry-run/--execute", help="Print plan only (default) or copy files."
    ),
) -> None:
    """
    Copy top-level BIDS components (dataset_description.json, participants.*,
    README) from a raw_nifti source into a target BIDS directory.

    Default: dry-run.  Use --execute to apply.

    Examples:
        python 02_copy_bids_component.py -b /raw_nifti -t /BIDS
        python 02_copy_bids_component.py -b /raw_nifti -t /BIDS --execute
        python 02_copy_bids_component.py -b /raw_nifti -t /BIDS --execute --force
    """
    mode_str = "[yellow]DRY-RUN[/]" if dry_run else "[bold green]EXECUTE[/]"
    console.print(
        f"\n[bold]02_copy_bids_component[/]  src={src_bids}  dst={dst_bids}  mode={mode_str}\n"
    )

    t = Table(
        title="BIDS component copy plan",
        show_header=True,
        header_style="bold magenta",
        box=None,
        padding=(0, 1),
    )
    t.add_column("file")
    t.add_column("status", justify="center")

    results = {}
    for name in BIDS_COMPONENTS:
        status = _copy_component(str(src_bids), str(dst_bids), name, force, dry_run)
        results[name] = status

    for name, status in results.items():
        cell = {
            "ok": "[green]copied[/]",
            "skip": "[dim]skip (exists)[/]",
            "dry": "[dim]dry — would copy[/]",
            "no_src": "[yellow]no source[/]",
            "error": "[red]ERROR[/]",
        }.get(status, status)
        t.add_row(name, cell)

    console.print(t)

    n_ok = sum(1 for s in results.values() if s == "ok")
    n_dry = sum(1 for s in results.values() if s == "dry")
    n_err = sum(1 for s in results.values() if s == "error")

    console.print()
    if dry_run:
        console.print(
            f"[yellow]Dry-run complete.[/]  {n_dry} file(s) would be copied."
            "  Pass [bold]--execute[/bold] to apply."
        )
    else:
        console.print(f"[green]Done.[/]  copied={n_ok}  errors={n_err}")


if __name__ == "__main__":
    app()
