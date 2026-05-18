#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) Yongning Lei 2024-2025
# Apache-2.0 License
# -----------------------------------------------------------------------------
"""
Interactively create or append entries to rerun_check.tsv.

Use this when you remember a rerun from scanning day but don't have (or can't
access) the lab-note Excel — a complement to 01_generate_rerun_check_from_labnote.py.

Output columns
--------------
sub, ses, task, extra_run, compensates_run,
protocol_name, found_in_bids, is_within_range, status

Usage
-----
    python 03_add_rerun_check_interactive.py --out /path/to/rerun_check.tsv
    python 03_add_rerun_check_interactive.py --out /path/to/rerun_check.tsv \\
        --bids-dir /bcbl/home/public/Gari/VOTCLOC/main_exp/BIDS
"""
from __future__ import annotations

import csv
import glob
import os.path as op
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

_TSV_FIELDS = [
    "sub", "ses", "task", "extra_run", "compensates_run",
    "protocol_name", "found_in_bids", "is_within_range", "status",
]

# Standard run count per task — used to auto-fill is_within_range.
# A compensates_run within [1, N] → True.  Add tasks as needed.
_STANDARD_N_RUNS: dict[str, int] = {
    "fLoc": 10,
    "BfLocVideo": 10,
    "IRAKEINU": 8,
}


# ---------------------------------------------------------------------------
# Interactive helpers
# ---------------------------------------------------------------------------

def _ask(label: str, default: str = "", required: bool = True) -> str:
    """Prompt the user; return default on empty input when one is provided."""
    hint = f"[dim]{default}[/dim]" if default else "[dim]required[/dim]"
    while True:
        val = console.input(f"  [cyan]{label}[/cyan] ({hint}): ").strip()
        if val:
            return val
        if default:
            return default
        if not required:
            return ""
        console.print("  [yellow]This field is required.[/yellow]")


def _ask_bool(label: str, auto: Optional[str] = None) -> str:
    """Return 'True' or 'False', with optional auto-detected default."""
    if auto is not None:
        console.print(f"  [dim]{label} auto-detected: {auto}[/dim]")
        override = console.input(f"  [cyan]Override {label}?[/cyan] (leave blank to keep [bold]{auto}[/bold]): ").strip()
        if override.lower() in ("true", "t", "1", "yes"):
            return "True"
        if override.lower() in ("false", "f", "0", "no"):
            return "False"
        return auto
    raw = _ask(label, default="True")
    return "True" if raw.lower() in ("true", "t", "1", "yes", "y", "") else "False"


# ---------------------------------------------------------------------------
# BIDS checks
# ---------------------------------------------------------------------------

def _detect_found_in_bids(bids_dir: Path, sub: str, ses: str, task: str, extra_run: str) -> Optional[str]:
    func_dir = bids_dir / f"sub-{sub}" / f"ses-{ses}" / "func"
    pattern = str(func_dir / f"sub-{sub}_ses-{ses}_task-{task}_run-{extra_run}_bold*")
    return "True" if glob.glob(pattern) else "False"


def _detect_within_range(task: str, compensates_run: str) -> Optional[str]:
    n = _STANDARD_N_RUNS.get(task)
    if n is None:
        return None
    try:
        return "True" if 1 <= int(compensates_run) <= n else "False"
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------

@app.command()
def main(
    out: Path = typer.Option(
        ..., "--out", "-o",
        help="Path to rerun_check.tsv (created if absent, appended if present)",
    ),
    bids_dir: Optional[Path] = typer.Option(
        None, "--bids-dir",
        help="BIDS root — used to auto-detect found_in_bids from the func/ tree",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview entries without writing"),
):
    """Interactively add rerun entries to rerun_check.tsv."""

    # ---- load existing rows ------------------------------------------------
    existing: list[dict] = []
    if out.exists() and out.stat().st_size > 0:
        with open(out) as f:
            existing = list(csv.DictReader(f, delimiter="\t"))
        console.print(f"\n[green]Appending to:[/green] {out}  ([dim]{len(existing)} existing rows[/dim])")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        console.print(f"\n[green]Creating:[/green] {out}")

    if bids_dir:
        if bids_dir.is_dir():
            console.print(f"[green]BIDS dir:[/green] {bids_dir}  [dim](found_in_bids will be auto-checked)[/dim]")
        else:
            console.print(f"[yellow]BIDS dir not found:[/yellow] {bids_dir}  [dim](skipping auto-check)[/dim]")
            bids_dir = None

    new_rows: list[dict] = []

    # ---- entry loop --------------------------------------------------------
    while True:
        console.rule("[bold]New entry")

        sub             = _ask("sub  (e.g. 04)").zfill(2)
        ses             = _ask("ses  (e.g. 01)").zfill(2)
        task            = _ask("task  (e.g. fLoc)")
        extra_run       = _ask("extra_run       — the redo/bonus run number (e.g. 11)").zfill(2)
        compensates_run = _ask("compensates_run — the aborted run it replaces (e.g. 04)").zfill(2)

        protocol_name   = f"{task}_run-{extra_run}_rerun-{compensates_run}"
        console.print(f"  [dim]protocol_name → {protocol_name}[/dim]")

        auto_bids   = _detect_found_in_bids(bids_dir, sub, ses, task, extra_run) if bids_dir else None
        auto_range  = _detect_within_range(task, compensates_run)
        found_in_bids   = _ask_bool("found_in_bids",   auto=auto_bids)
        is_within_range = _ask_bool("is_within_range", auto=auto_range)

        status = _ask(
            "status  (free text — note why you know this, e.g. 'remembered from scan day')",
            default="OK",
            required=False,
        ) or "OK"

        row = {
            "sub": sub, "ses": ses, "task": task,
            "extra_run": extra_run, "compensates_run": compensates_run,
            "protocol_name": protocol_name,
            "found_in_bids": found_in_bids,
            "is_within_range": is_within_range,
            "status": status,
        }

        # duplicate check
        key = (sub, ses, task, extra_run)
        dup_src = [r for r in existing + new_rows
                   if (r["sub"], r["ses"], r["task"], r["extra_run"]) == key]
        if dup_src:
            console.print(
                f"  [yellow]Warning:[/yellow] entry (sub-{sub} ses-{ses} {task} run-{extra_run}) already exists."
            )
            if console.input("  Overwrite? [y/N]: ").strip().lower() != "y":
                console.print("  [dim]Skipped.[/dim]")
                if console.input("\nAdd another entry? [y/N]: ").strip().lower() != "y":
                    break
                continue

        new_rows.append(row)
        console.print(
            f"  [green]✓[/green]  sub-{sub} ses-{ses}  {task}  "
            f"extra_run={extra_run}  compensates={compensates_run}  status={status!r}"
        )

        if console.input("\nAdd another entry? [y/N]: ").strip().lower() != "y":
            break

    if not new_rows:
        console.print("\n[dim]No new entries — nothing written.[/dim]")
        raise typer.Exit()

    # ---- preview table -----------------------------------------------------
    console.print()
    t = Table(
        title=f"{'[DRY RUN] ' if dry_run else ''}{'Append' if existing else 'Create'}  →  {out}",
        box=box.SIMPLE_HEAD, show_lines=False,
    )
    for col in _TSV_FIELDS:
        t.add_column(col, style="bold cyan" if col in ("sub", "ses") else "")
    for r in new_rows:
        t.add_row(*[r[c] for c in _TSV_FIELDS])
    console.print(t)

    if dry_run:
        console.print("[yellow]Dry run — nothing written.[/yellow]")
        raise typer.Exit()

    # ---- write -------------------------------------------------------------
    need_header = not out.exists() or out.stat().st_size == 0
    with open(out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_TSV_FIELDS, delimiter="\t")
        if need_header:
            writer.writeheader()
        writer.writerows(new_rows)

    console.print(f"[green]Wrote {len(new_rows)} row(s) to {out}[/green]")


if __name__ == "__main__":
    app()
