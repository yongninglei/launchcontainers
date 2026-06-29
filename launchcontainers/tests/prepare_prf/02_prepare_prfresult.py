"""
02_prepare_prfresult.py
-----------------------
Check the two symlinks required before running prfresult:

  1. derivatives/fmriprep/analysis-<xx>
       must be a symlink pointing to derivatives/fmriprep-<xx>

  2. derivatives/fmriprep-<xx>/sourcedata/freesurfer
       must be a symlink pointing to the freesurfer derivatives dir

If a symlink is missing or wrong, the script will create / fix it
(with --fix). Without --fix it only reports the status.

Usage
-----
    # check only
    python 02_prepare_prfresult.py --bidsdir /path/BIDS --version 23

    # create / fix missing symlinks
    python 02_prepare_prfresult.py --bidsdir /path/BIDS --version 23 --fix

    # point freesurfer to a custom path
    python 02_prepare_prfresult.py --bidsdir /path/BIDS --version 23 --fix \\
        --freesurfer-dir /path/BIDS/derivatives/freesurfer
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

console = Console()
app = typer.Typer(pretty_exceptions_show_locals=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_symlink(
    link_path: Path,
    expected_target: Path,
    label: str,
    fix: bool,
) -> bool:
    """
    Check (and optionally fix) one symlink.

    Returns True if the symlink is correct after the call.
    """
    if link_path.is_symlink():
        actual_target = Path(os.readlink(link_path))
        # resolve to absolute for fair comparison
        actual_abs = (link_path.parent / actual_target).resolve()
        expected_abs = expected_target.resolve()

        if actual_abs == expected_abs:
            console.print(f"  [green]✓[/] {label}: symlink OK")
            console.print(f"    {link_path} → {actual_target}")
            return True
        else:
            console.print(
                f"  [red]✗[/] {label}: symlink exists but points to wrong target"
            )
            console.print(f"    link    : {link_path}")
            console.print(f"    actual  : {actual_target}")
            console.print(f"    expected: {expected_target}")
            if fix:
                link_path.unlink()
                link_path.symlink_to(expected_target)
                console.print(f"  [green]→ fixed[/] {link_path} → {expected_target}")
                return True
            return False

    elif link_path.exists():
        console.print(
            f"  [red]✗[/] {label}: path exists but is NOT a symlink: {link_path}"
        )
        return False

    else:
        console.print(f"  [yellow]✗[/] {label}: symlink missing")
        console.print(f"    expected: {link_path} → {expected_target}")
        if fix:
            if not expected_target.exists():
                console.print(
                    f"  [red]  cannot fix:[/] target does not exist: {expected_target}"
                )
                return False
            link_path.parent.mkdir(parents=True, exist_ok=True)
            link_path.symlink_to(expected_target)
            console.print(f"  [green]→ created[/] {link_path} → {expected_target}")
            return True
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    bidsdir: Path = typer.Option(..., "--bidsdir", "-b", help="BIDS root directory."),
    version: str = typer.Option(
        ...,
        "--version",
        "-v",
        help="fMRIprep version label, e.g. '23' for fmriprep-23.",
    ),
    fix: bool = typer.Option(
        False, "--fix", help="Create or fix missing/wrong symlinks."
    ),
    freesurfer_dir: Optional[Path] = typer.Option(
        None,
        "--freesurfer-dir",
        help="Path to the FreeSurfer derivatives dir. "
        "Default: <bidsdir>/derivatives/freesurfer",
    ),
):
    """Check (and optionally fix) symlinks required for prfresult."""
    derivatives = bidsdir / "derivatives"

    # 1. derivatives/fmriprep/analysis-<version>  →  derivatives/fmriprep-<version>
    fmriprep_versioned = derivatives / f"fmriprep-{version}"
    fmriprep_base = derivatives / "fmriprep"
    analysis_link = fmriprep_base / f"analysis-{version}"

    # 2. derivatives/fmriprep-<version>/sourcedata/freesurfer  →  <freesurfer_dir>
    fs_target = freesurfer_dir if freesurfer_dir else derivatives / "freesurfer"
    fs_link = fmriprep_versioned / "sourcedata" / "freesurfer"

    console.print(f"\n[bold cyan]prfresult symlink check — version {version}[/]")
    console.print(f"  BIDS root       : {bidsdir}")
    console.print(f"  fmriprep-{version}  : {fmriprep_versioned}")
    console.print(f"  freesurfer target: {fs_target}")
    if fix:
        console.print("  [bold yellow]--fix enabled: will create/repair symlinks[/]")
    console.print()

    ok1 = _check_symlink(
        link_path=analysis_link,
        expected_target=fmriprep_versioned,
        label=f"fmriprep/analysis-{version} → fmriprep-{version}",
        fix=fix,
    )

    console.print()

    ok2 = _check_symlink(
        link_path=fs_link,
        expected_target=fs_target,
        label=f"fmriprep-{version}/sourcedata/freesurfer → freesurfer",
        fix=fix,
    )

    console.print()
    if ok1 and ok2:
        console.print("[bold green]All symlinks OK — ready for prfresult.[/]")
    else:
        if not fix:
            console.print(
                "[bold yellow]Issues found. Re-run with --fix to create/repair symlinks.[/]"
            )
        else:
            console.print("[bold red]Some symlinks could not be fixed — see above.[/]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
