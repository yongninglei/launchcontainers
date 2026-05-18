#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) Yongning Lei 2024-2025
# MIT License
# -----------------------------------------------------------------------------
"""
Generate (or regenerate) rerun_check.tsv from BIDS func directories.

Background
----------
When a run is aborted and reacquired, the scanner produces an "extra" run
beyond the standard count (e.g. fLoc run-11 when only 10 are expected).
``prepare_floc_events_tsv.py`` creates a symlink for each extra run's
events.tsv pointing to the original events.tsv of the run it compensates for::

    func/sub-04_ses-09_task-fLoc_run-11_events.tsv
        → sourcedata/.../sub-04_ses-09_task-fLoc_run-02_events.tsv

This script reads those symlinks to reconstruct the full mapping and writes
the canonical ``rerun_check.tsv`` to ``<bids_dir>/sourcedata/qc/``.

This is equivalent to running ``handle_reruns.py check --lab-note <xlsx>``
but works without the lab note Excel file (which lives on the BCBL server
and is not accessible on DIPC).

Standard run ranges used to identify "extra" runs
--------------------------------------------------
- fLoc          : runs 01–10  (extra = run ≥ 11)
- ret* tasks    : runs 01–02  (extra = run ≥ 03)

Output columns (same as handle_reruns.py)
-----------------------------------------
sub, ses, task, extra_run, compensates_run,
protocol_name, found_in_bids, is_within_range, status

Usage
-----
Single session (dry-run preview)::

    python generate_rerun_check.py -s 04,09 \\
        --bids-dir /scratch/tlei/VOTCLOC/BIDS

Batch from file (write output)::

    python generate_rerun_check.py -f subseslist.tsv \\
        --bids-dir /scratch/tlei/VOTCLOC/BIDS --execute
"""
from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BIDS_DIR = Path("/scratch/tlei/VOTCLOC/BIDS")
_DEFAULT_OUT = _BIDS_DIR / "sourcedata" / "qc" / "rerun_check.tsv"

# Only fLoc reruns are of interest for the GLM exclusion map
_FLOC_STANDARD_MAX = 10

_BOLD_RE = re.compile(r"task-(\w+)_run-(\d+)_bold\.nii\.gz$")
_RUN_IN_NAME_RE = re.compile(r"_run-(\d+)_events\.tsv$")

_TSV_FIELDS = [
    "sub", "ses", "task", "extra_run", "compensates_run",
    "protocol_name", "found_in_bids", "is_within_range", "status",
]

# ---------------------------------------------------------------------------
# Sub/ses pair parsing
# ---------------------------------------------------------------------------

def _parse_pairs(subses_arg: Optional[str], file_arg: Optional[str]) -> list[tuple[str, str]]:
    if subses_arg:
        parts = subses_arg.split(",")
        if len(parts) != 2:
            console.print(f"[red]ERROR[/red]: -s expects 'sub,ses' e.g. 01,09, got: {subses_arg!r}")
            raise typer.Exit(1)
        return [(parts[0].strip().zfill(2), parts[1].strip().zfill(2))]

    if file_arg:
        path = Path(file_arg)
        if not path.exists():
            console.print(f"[red]ERROR[/red]: subseslist not found: {path}")
            raise typer.Exit(1)
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        pairs: list[tuple[str, str]] = []
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh, delimiter=delimiter):
                if "RUN" in row and str(row["RUN"]).strip() != "True":
                    continue
                pairs.append((str(row["sub"]).strip().zfill(2), str(row["ses"]).strip().zfill(2)))
        return pairs

    console.print("[red]ERROR[/red]: provide either -s <sub,ses> or -f <subseslist>")
    raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Per-session logic
# ---------------------------------------------------------------------------

def _standard_max(task: str) -> int:
    """Return the maximum standard run number for fLoc; 0 for all other tasks."""
    if task == "fLoc":
        return _FLOC_STANDARD_MAX
    return 0  # only fLoc is of interest


def _compensates_run_from_symlink(events_path: Path) -> str | None:
    """
    If *events_path* is a symlink, read the target and extract the run
    number from the target filename.

    Returns zero-padded run string (e.g. ``"02"``) or ``None``.
    """
    if not events_path.is_symlink():
        return None
    target = Path(os.readlink(events_path))
    m = _RUN_IN_NAME_RE.search(target.name)
    if m:
        return m.group(1).zfill(2)
    return None


def _scan_session(bids_dir: Path, sub: str, ses: str) -> list[dict]:
    """
    Scan the BIDS func directory for a single session and return a list of
    rerun-record dicts (one per extra run found).

    Returns an empty list if the func dir is missing or no extra runs exist.
    """
    func_dir = bids_dir / f"sub-{sub}" / f"ses-{ses}" / "func"
    if not func_dir.is_dir():
        return []

    records: list[dict] = []

    for bold in sorted(func_dir.glob(f"sub-{sub}_ses-{ses}_task-*_run-*_bold.nii.gz")):
        m = _BOLD_RE.search(bold.name)
        if not m:
            continue
        task = m.group(1)
        run_int = int(m.group(2))
        run_str = f"{run_int:02d}"

        std_max = _standard_max(task)
        if std_max == 0:
            continue  # task not of interest

        is_within_range = run_int <= std_max
        found_in_bids   = True  # we found the bold file

        events_path = func_dir / f"sub-{sub}_ses-{ses}_task-{task}_run-{run_str}_events.tsv"
        compensates_run = _compensates_run_from_symlink(events_path)

        if is_within_range:
            # Only flag a within-range run if its events.tsv points to a
            # DIFFERENT run — that means this slot is a rerun of something else.
            # (e.g. run-02 symlink → run-05 events.tsv)
            # Normal runs have no symlink, or point to their own run number.
            if compensates_run is None or compensates_run == run_str:
                continue

        # Determine protocol_name and compensates_run
        if compensates_run is None:
            # Extra run exists but events.tsv symlink is absent/not-yet-created
            compensates_run_display = "?"
            protocol_name = f"{task}_run-{run_str}_rerun-?"
            status = "NO_EVENTS_LINK"
        else:
            compensates_run_display = compensates_run
            protocol_name = f"{task}_run-{run_str}_rerun-{compensates_run}"
            status = "OK"

        records.append({
            "sub":              sub,
            "ses":              ses,
            "task":             task,
            "extra_run":        run_str,
            "compensates_run":  compensates_run_display,
            "protocol_name":    protocol_name,
            "found_in_bids":    str(found_in_bids),
            "is_within_range":  str(is_within_range),
            "status":           status,
        })

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    bids_dir: Path = typer.Option(_BIDS_DIR, "--bids-dir", "-b", help="BIDS root directory"),
    subses_arg: Optional[str] = typer.Option(None, "-s", help="Single sub,ses pair e.g. 04,09"),
    file_arg: Optional[str] = typer.Option(None, "-f", help="Path to subseslist TSV/CSV"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output TSV path. Default: <bids_dir>/sourcedata/qc/rerun_check.tsv",
    ),
    execute: bool = typer.Option(False, "--execute", help="Write the output file (default: dry-run preview only)"),
) -> None:
    """
    Generate rerun_check.tsv from BIDS func directory symlinks.

    Reads events.tsv symlinks created by prepare_floc_events_tsv.py to
    determine which extra runs compensate for which original runs, then
    writes the canonical rerun_check.tsv.

    Default is dry-run (preview only). Pass --execute to write the file.
    """
    pairs = _parse_pairs(subses_arg, file_arg)

    out_path = output or (bids_dir / "sourcedata" / "qc" / "rerun_check.tsv")
    mode_str = "[green]EXECUTE[/green]" if execute else "[yellow]DRY-RUN[/yellow]"

    console.rule("[bold cyan]generate_rerun_check[/bold cyan]")
    console.print(
        f"  BIDS dir   : {bids_dir}\n"
        f"  Sessions   : {len(pairs)}\n"
        f"  Output     : {out_path}\n"
        f"  Mode       : {mode_str}\n"
    )

    all_records: list[dict] = []

    for sub, ses in pairs:
        records = _scan_session(bids_dir, sub, ses)
        if not records:
            console.print(f"  [dim]sub-{sub} ses-{ses}  — no extra runs[/dim]")
        else:
            for r in records:
                color = "green" if r["status"] == "OK" else "red"
                console.print(
                    f"  sub-{sub} ses-{ses}  "
                    f"task-{r['task']}  "
                    f"extra=run-{r['extra_run']}  "
                    f"compensates=run-{r['compensates_run']}  "
                    f"[{color}]{r['status']}[/{color}]"
                )
        all_records.extend(records)

    # Summary table
    if all_records:
        console.print()
        tbl = Table(title="Rerun records found", box=box.SIMPLE_HEAD)
        for col in _TSV_FIELDS:
            tbl.add_column(col)
        for r in all_records:
            color = "green" if r["status"] == "OK" else "red" if r["status"] == "NO_EVENTS_LINK" else "yellow"
            tbl.add_row(*[f"[{color}]{r[c]}[/{color}]" if c == "status" else r[c] for c in _TSV_FIELDS])
        console.print(tbl)
    else:
        console.print("\n  [dim]No extra runs found across all sessions.[/dim]")

    console.print(f"\n  Total extra runs: [bold]{len(all_records)}[/bold]")
    n_ok      = sum(1 for r in all_records if r["status"] == "OK")
    n_missing = sum(1 for r in all_records if r["status"] == "NO_EVENTS_LINK")
    if n_missing:
        console.print(
            f"  [yellow]WARNING[/yellow]: {n_missing} extra run(s) have no events.tsv symlink "
            f"— run prepare_floc_events_tsv.py first, then re-run this script."
        )

    if not execute:
        console.print(f"\n  [dim]Dry-run — nothing written. Pass --execute to write {out_path}[/dim]")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_TSV_FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(all_records)
    console.print(f"\n  [green]Written →[/green] {out_path}  ({len(all_records)} rows)")


if __name__ == "__main__":
    app()
