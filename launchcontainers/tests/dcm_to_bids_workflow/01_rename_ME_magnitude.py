"""
01_rename_ME_magnitude.py
--------------------------
Rename multi-echo magnitude files from heudiconv's suffix-embedded numbering to
the BIDS echo-entity format, consistent with phase and sbref files.

heudiconv output:
    sub-01_ses-01_task-BfLocVideo_acq-ME_run-01_magnitude1
    sub-01_ses-01_task-BfLocVideo_acq-ME_run-01_magnitude2
    sub-01_ses-01_task-BfLocVideo_acq-ME_run-01_magnitude3

After this script:
    sub-01_ses-01_task-BfLocVideo_acq-ME_run-01_echo-1_magnitude
    sub-01_ses-01_task-BfLocVideo_acq-ME_run-01_echo-2_magnitude
    sub-01_ses-01_task-BfLocVideo_acq-ME_run-01_echo-3_magnitude

Consistent with:
    sub-01_ses-01_task-BfLocVideo_acq-ME_run-01_echo-1_phase
    sub-01_ses-01_task-BfLocVideo_acq-ME_run-01_echo-1_sbref

Pre-requisite
-------------
Run ``00_merge_split_ses_reording_runs.py`` first.

Pipeline order
--------------
  00_merge_split_ses_reording_runs.py
  01_rename_ME_magnitude.py          ← this script
  01_drop_duplicated_sbrefs.py
  02_copy_bids_component.py

Usage
-----
  # dry-run (default):
  python 01_rename_ME_magnitude.py -b /BIDS -s 01,01

  # execute:
  python 01_rename_ME_magnitude.py -b /BIDS -s 01,01 --execute

  # batch:
  python 01_rename_ME_magnitude.py -b /BIDS -f subseslist.tsv --execute -v
"""

from __future__ import annotations

import glob
import os
import os.path as op
import re
import shutil
import uuid
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from launchcontainers.utils import parse_subses_list

console = Console()
app = typer.Typer(pretty_exceptions_show_locals=False)

# Matches stems ending in _magnitudeN (digit(s)), with no prior echo entity
_MAGNITUDE_RE = re.compile(r"_magnitude(\d+)$")
_ECHO_RE = re.compile(r"_echo-\w+_")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_rename_pairs(bidsdir: str, sub: str, ses: str) -> list[dict]:
    """
    Return one entry per magnitude file that needs renaming.

    Each entry:
      func_dir  — full path to func directory
      stem      — original stem (no extension)
      new_stem  — target stem after echo-entity insertion
      echo_num  — echo number string (e.g. "1", "2", "3")
      files     — list of (src_path, dst_path) for every companion that exists
    """
    func_dir = op.join(bidsdir, f"sub-{sub}", f"ses-{ses}", "func")
    if not op.isdir(func_dir):
        return []

    pairs: list[dict] = []
    for jf in sorted(glob.glob(op.join(func_dir, "*.json"))):
        stem = re.sub(r"\.json$", "", op.basename(jf))

        # Only touch files that end in _magnitudeN and have no echo entity yet
        m = _MAGNITUDE_RE.search(stem)
        if not m or _ECHO_RE.search(stem):
            continue

        echo_num = m.group(1)
        new_stem = _MAGNITUDE_RE.sub(f"_echo-{echo_num}_magnitude", stem)

        file_pairs: list[tuple[str, str]] = []
        for ext in (".json", ".nii.gz"):
            src = op.join(func_dir, stem + ext)
            if op.exists(src):
                file_pairs.append((src, op.join(func_dir, new_stem + ext)))

        if file_pairs:
            pairs.append(
                {
                    "func_dir": func_dir,
                    "stem": stem,
                    "new_stem": new_stem,
                    "echo_num": echo_num,
                    "files": file_pairs,
                }
            )

    return pairs


def _execute_pairs(pairs: list[dict], dry_run: bool) -> dict:
    """Two-phase atomic rename: src → temp → final."""
    counts = {"renamed": 0, "errors": 0}

    for entry in pairs:
        file_pairs = entry["files"]

        if dry_run:
            counts["renamed"] += 1
            continue

        # Phase 1: rename each src to a temp name in the same directory
        temp_id = uuid.uuid4().hex[:8]
        temps: list[tuple[str, str]] = []
        phase1_ok = True
        for idx, (src, _dst) in enumerate(file_pairs):
            ext = ".nii.gz" if src.endswith(".nii.gz") else op.splitext(src)[1]
            tmp = op.join(entry["func_dir"], f"._tmp_{temp_id}_{idx}{ext}")
            try:
                os.rename(src, tmp)
                temps.append((tmp, _dst))
            except Exception as exc:
                console.print(f"  [red][ERROR] phase-1 rename {op.basename(src)}: {exc}[/]")
                counts["errors"] += 1
                phase1_ok = False
                break

        if not phase1_ok:
            continue

        # Phase 2: move temp → final destination
        phase2_ok = True
        for tmp, final in temps:
            try:
                shutil.move(tmp, final)
            except Exception as exc:
                console.print(f"  [red][ERROR] phase-2 rename {op.basename(tmp)}: {exc}[/]")
                counts["errors"] += 1
                phase2_ok = False

        if phase2_ok:
            counts["renamed"] += 1

    return counts


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _print_pairs(
    sub: str,
    ses: str,
    pairs: list[dict],
    verbose: bool,
) -> None:
    n = len(pairs)
    if n == 0:
        console.print(f"  sub-{sub}  ses-{ses}  [green]OK (no magnitude files to rename)[/]")
        return

    console.print(f"  sub-{sub}  ses-{ses}  [yellow]{n} magnitude file(s) to rename[/]")

    if not verbose:
        return

    t = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 1))
    t.add_column("echo", justify="center", style="cyan")
    t.add_column("old suffix", style="yellow")
    t.add_column("", justify="center", style="dim")
    t.add_column("new suffix", style="green")

    for entry in pairs:
        # Show only the suffix portion (strip sub/ses prefix) for readability
        old_short = re.sub(r"^sub-\w+_ses-\w+_", "", entry["stem"])
        new_short = re.sub(r"^sub-\w+_ses-\w+_", "", entry["new_stem"])
        t.add_row(entry["echo_num"], old_short, "→", new_short)

    console.print(t)


def _print_summary(results: list[dict]) -> None:
    t = Table(
        title="rename-ME-magnitude summary",
        show_header=True,
        header_style="bold magenta",
        box=None,
        padding=(0, 1),
    )
    t.add_column("sub", justify="right", style="cyan")
    t.add_column("ses", justify="right", style="cyan")
    t.add_column("renamed", justify="right", style="yellow")
    t.add_column("errors", justify="right", style="red")
    t.add_column("status", justify="center")

    for r in results:
        if r.get("error"):
            st = f"[red]{r['error']}[/]"
        elif r.get("counts", {}).get("errors", 0):
            st = "[red]ERRORS[/]"
        elif r.get("counts", {}).get("renamed", 0) == 0:
            st = "[green]OK (nothing to do)[/]"
        else:
            st = "[green]OK[/]"
        t.add_row(
            r["sub"],
            r["ses"],
            str(r.get("counts", {}).get("renamed", 0)),
            str(r.get("counts", {}).get("errors", 0)),
            st,
        )

    console.print(t)


# ---------------------------------------------------------------------------
# Per-session driver
# ---------------------------------------------------------------------------


def _process_session(
    bidsdir: str,
    sub: str,
    ses: str,
    dry_run: bool,
    verbose: bool,
) -> dict:
    func_dir = op.join(bidsdir, f"sub-{sub}", f"ses-{ses}", "func")
    if not op.isdir(func_dir):
        console.print(f"[yellow]sub-{sub} ses-{ses}: func dir not found — skipped[/]")
        return {"sub": sub, "ses": ses, "error": "func dir not found"}

    pairs = _find_rename_pairs(bidsdir, sub, ses)
    _print_pairs(sub, ses, pairs, verbose=verbose)
    counts = _execute_pairs(pairs, dry_run=dry_run)

    return {"sub": sub, "ses": ses, "counts": counts, "error": None}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    bidsdir: Path = typer.Option(..., "--bidsdir", "-b", help="BIDS root directory."),
    subses: Optional[list[str]] = typer.Option(
        None,
        "--subses",
        "-s",
        help="sub,ses pair e.g. -s 01,01  (repeatable: -s 01,01 -s 02,01).",
    ),
    subses_file: Optional[Path] = typer.Option(
        None, "--file", "-f", help="TSV/CSV with columns sub, ses."
    ),
    dry_run: bool = typer.Option(
        True, "--dry-run/--execute", help="Print plan only (default) or execute renames."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show per-file rename table."
    ),
):
    """
    Rename multi-echo magnitude files from heudiconv's _magnitudeN format to
    BIDS _echo-N_magnitude format, consistent with phase and sbref naming.

    Run after 00_merge_split_ses_reording_runs.py and before
    01_drop_duplicated_sbrefs.py.

    Default: dry-run.  Use --execute to apply.

    Examples:
        python 01_rename_ME_magnitude.py -b /BIDS -s 01,01
        python 01_rename_ME_magnitude.py -b /BIDS -s 01,01 -s 02,01 --execute -v
        python 01_rename_ME_magnitude.py -b /BIDS -f subseslist.tsv --execute
    """
    if subses_file is not None:
        pairs = parse_subses_list(subses_file)
    elif subses:
        pairs = []
        for token in subses:
            parts = token.split(",")
            if len(parts) != 2:
                console.print(f"[red]Error:[/] -s expects 'sub,ses' (got '{token}').")
                raise typer.Exit(1)
            pairs.append((parts[0].strip().zfill(2), parts[1].strip().zfill(2)))
    else:
        console.print("[bold red]Error:[/] provide -s sub,ses  or  -f <subseslist>.")
        raise typer.Exit(1)

    mode_str = "[yellow]DRY-RUN[/]" if dry_run else "[bold red]EXECUTE[/]"
    console.print(f"\n[bold]01_rename_ME_magnitude[/]  bidsdir={bidsdir}  mode={mode_str}\n")

    results = []
    for sub, ses in pairs:
        r = _process_session(str(bidsdir), sub, ses, dry_run=dry_run, verbose=verbose)
        results.append(r)

    console.print()
    _print_summary(results)

    if dry_run:
        console.print("\n[yellow]Dry-run complete.  Use --execute to apply.[/]")
    else:
        total = sum(r.get("counts", {}).get("renamed", 0) for r in results)
        errors = sum(r.get("counts", {}).get("errors", 0) for r in results)
        console.print(f"\n[green]Done.[/]  renamed={total}  errors={errors}")


if __name__ == "__main__":
    app()
