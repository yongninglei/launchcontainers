#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) Yongning Lei 2024-2025
# MIT License
# -----------------------------------------------------------------------------
"""
Generate rerun_check.tsv from the lab-note Excel file.

Background
----------
The lab note records rerun acquisitions in the ``protocol_name`` column
with entries like::

    fLoc_run-11_rerun-04

meaning run-11 is the extra acquisition that compensates for run-04.

This script parses those entries, cross-checks them against the BIDS func
directory, and writes ``rerun_check.tsv`` to
``<bids_dir>/sourcedata/qc/rerun_check.tsv``.

This is the authoritative generator — it has access to the complete lab note
and captures within-range reruns (e.g. run-02 compensates for run-02) that
cannot be detected from BIDS alone.

Use ``generate_rerun_check_from_bids.py`` when the lab-note Excel is not
accessible (e.g. BCBL server not mounted on DIPC).

Output columns
--------------
sub, ses, task, extra_run, compensates_run,
protocol_name, found_in_bids, is_within_range, status

Usage
-----
Single session::

    python generate_rerun_check_from_labnote.py \\
        --lab-note /path/to/VOTCLOC_subses_list.xlsx \\
        --bids-dir /scratch/tlei/VOTCLOC/BIDS \\
        -s 04,09

Batch from file::

    python generate_rerun_check_from_labnote.py \\
        --lab-note /path/to/VOTCLOC_subses_list.xlsx \\
        --bids-dir /scratch/tlei/VOTCLOC/BIDS \\
        -f subseslist.tsv --execute
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional

import pandas as pd
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
_TSV_FIELDS = [
    "sub",
    "ses",
    "task",
    "acq",
    "extra_run",
    "compensates_run",
    "protocol_name",
    "found_in_bids",
    "is_within_range",
    "status",
]

# Only fLoc reruns are of interest for the GLM exclusion map
FLOC_STANDARD_RUNS: set[str] = {f"{i:02d}" for i in range(1, 11)}  # 01–10


# ---------------------------------------------------------------------------
# Lab-note parsing
# Follows the logic from check_lab_note_and_fix_events_tsv.py exactly,
# with minor robustness additions noted inline.
# ---------------------------------------------------------------------------

_BAD_SES_PATTERNS = r"-|wrong|failed|lost|ME|bad|00|test|-t"

# Columns that mark a sheet as a real data sheet (not a summary/legend tab)
_REQUIRED_SHEET_COLS = ["sub", "ses", "protocol_name"]
# Optional columns that exist in the original xlsx; we skip sheets that
# lack even the minimum three above.
_EXTRA_SHEET_COLS = ["date", "quality_mark"]


def _convert_sub(series: pd.Series) -> pd.Series:
    """
    Convert the sub column to zero-padded 2-digit strings.

    Excel stores sub as floats (1.0, 2.0 …); original code uses
    ``astype(int)`` after reading.  We add ``astype(float)`` first
    to handle both float and string representations cleanly.
    """
    return series.astype(float).astype(int).astype(str).str.zfill(2)


def _convert_ses(series: pd.Series, sheet_name: str = "") -> pd.Series:
    """
    Convert the ses column to zero-padded 2-digit strings.

    Mirrors the try/except pattern from the original scripts:
    - If ses is already a string dtype, apply element-wise conversion.
    - Otherwise try astype(int) (works for numeric Excel cells).
    - If that fails, leave as-is and warn.
    """
    if pd.api.types.is_string_dtype(series):
        # Already strings — try to normalise each value individually
        def _zfill_one(v):
            s = str(v).strip()
            try:
                return str(int(float(s))).zfill(2)
            except (ValueError, TypeError):
                return s  # non-numeric session label (e.g. "01rr") — keep as-is

        return series.apply(_zfill_one)
    else:
        try:
            return series.astype(float).astype(int).astype(str).str.zfill(2)
        except Exception:
            console.print(
                f"  [dim]ses column could not be converted to int "
                f"for sheet {sheet_name!r} — leaving as-is[/dim]"
            )
            return series.astype(str)


def _extract_rerun_rows(df: pd.DataFrame) -> list[dict]:
    """
    From a cleaned sub/ses/protocol_name DataFrame, return rows where
    protocol_name encodes a rerun relationship (e.g. ``fLoc_run-11_rerun-04``).

    Uses the same simple string-split approach as the original scripts, with
    a regex fallback for the rerun number to handle ``rerun04`` vs ``rerun-04``.
    """
    rerun_df = df[
        df["protocol_name"].str.contains("rerun", case=False, na=False)
        & ~df["protocol_name"].str.contains(r"qmri|T1", case=False, na=False)
    ]

    rows: list[dict] = []
    for _, row in rerun_df.iterrows():
        proto = str(row["protocol_name"]).strip()
        parts = proto.split("_")

        # run part: segment like "run-11"
        run_part = [p for p in parts if "run-" in p and "rerun" not in p.lower()]
        # rerun part: segment containing "rerun" (with or without dash)
        rerun_part = [p for p in parts if "rerun" in p.lower()]
        # task part: fLoc or ret*
        # Only fLoc is of interest
        task_part = [p for p in parts if "fLoc" in p]

        if not (run_part and rerun_part and task_part):
            continue

        extra_run_raw = run_part[0].split("-")[-1]  # "11"
        rerun_digits = re.search(r"\d+", rerun_part[0])  # "04" from "rerun-04"
        if not rerun_digits:
            continue

        task = "fLoc"

        rows.append(
            {
                "sub": str(row["sub"]),
                "ses": str(row["ses"]),
                "task": task,
                "acq": "None",  # lab-note does not record acq; fill manually via 01b
                "extra_run": extra_run_raw.zfill(2),
                "compensates_run": rerun_digits.group(0).zfill(2),
                "protocol_name": proto,
            }
        )
    return rows


def parse_lab_note(lab_note_path: Path) -> pd.DataFrame:
    """
    Read the lab-note Excel (or flat TSV/CSV) and return all rerun rows.

    Follows the exact sheet-processing logic from
    ``check_lab_note_and_fix_events_tsv.py``:

    1. Loop over sheets named ``sub-*``.
    2. Skip sheets missing any of the minimum required columns.
    3. Replace empty strings with NaN and drop rows where sub or ses is NaN.
    4. Convert sub to int → zero-padded string.
    5. Filter out bad session labels (wrong / failed / ME / …).
    6. Convert ses to int → zero-padded string (with graceful fallback).
    7. Keep rows whose protocol_name contains "rerun" but not "qmri" or "T1".
    """
    ext = lab_note_path.suffix.lower()
    rows: list[dict] = []

    if ext in (".xlsx", ".xls"):
        xls = pd.ExcelFile(lab_note_path)

        for sheet_name in xls.sheet_names:
            if not sheet_name.startswith("sub-"):
                continue

            df = pd.read_excel(xls, sheet_name=sheet_name, header=0)

            # Normalise column names (strip whitespace, lowercase)
            df.columns = [str(c).strip().lower() for c in df.columns]

            # Skip sheets that don't have the minimum required columns
            if not all(c in df.columns for c in _REQUIRED_SHEET_COLS):
                continue

            # Keep only the columns we care about
            keep_cols = _REQUIRED_SHEET_COLS + [
                c for c in _EXTRA_SHEET_COLS if c in df.columns
            ]
            df = df[keep_cols].copy()

            # ── Step 3: clean sub / ses ───────────────────────────────────
            df[["sub", "ses"]] = df[["sub", "ses"]].replace("", pd.NA)
            df = df.dropna(subset=["sub", "ses"], how="any")
            if df.empty:
                continue

            # ── Step 4: convert sub ───────────────────────────────────────
            try:
                df["sub"] = _convert_sub(df["sub"])
            except Exception as exc:
                console.print(
                    f"  [yellow]WARNING[/yellow]: sub conversion failed for "
                    f"sheet {sheet_name!r}: {exc} — skipping sheet"
                )
                continue

            # ── Step 5: bad session filter ────────────────────────────────
            df = df[~df["ses"].astype(str).str.contains(_BAD_SES_PATTERNS, na=False)]
            if df.empty:
                continue

            # ── Step 6: convert ses ───────────────────────────────────────
            df["ses"] = _convert_ses(df["ses"], sheet_name)

            # ── Step 7: extract rerun rows ────────────────────────────────
            rows.extend(_extract_rerun_rows(df))

    else:
        # Flat TSV / CSV fallback
        delimiter = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(lab_note_path, sep=delimiter, dtype=str)
        df.columns = [str(c).strip().lower() for c in df.columns]

        if not all(c in df.columns for c in _REQUIRED_SHEET_COLS):
            raise ValueError(
                f"Lab note must have columns {_REQUIRED_SHEET_COLS}. "
                f"Found: {list(df.columns)}"
            )

        df = df[_REQUIRED_SHEET_COLS].copy()
        df[["sub", "ses"]] = df[["sub", "ses"]].replace("", pd.NA)
        df = df.dropna(subset=["sub", "ses"], how="any")
        df = df[~df["ses"].astype(str).str.contains(_BAD_SES_PATTERNS, na=False)]
        try:
            df["sub"] = _convert_sub(df["sub"])
        except Exception:
            df["sub"] = df["sub"].astype(str).str.strip().str.zfill(2)
        df["ses"] = _convert_ses(df["ses"])
        rows.extend(_extract_rerun_rows(df))

    if not rows:
        return pd.DataFrame(
            columns=[
                "sub",
                "ses",
                "task",
                "extra_run",
                "compensates_run",
                "protocol_name",
            ]
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# BIDS scanning  (extracted from handle_reruns.py)
# ---------------------------------------------------------------------------


def _is_task_of_interest(task: str) -> bool:
    return task == "fLoc"


def _is_standard_run(task: str, run: str) -> bool:
    if task == "fLoc":
        return run in FLOC_STANDARD_RUNS
    return False


def _bids_extras_for_session(
    bids_dir: Path,
    sub: str,
    ses: str,
) -> tuple[list[tuple[str, str]], bool]:
    """
    Return (extras, func_exists).

    extras      — (task, run) pairs beyond the standard range found in BIDS.
    func_exists — False when the BIDS func folder is absent.
    """
    func_dir = bids_dir / f"sub-{sub}" / f"ses-{ses}" / "func"
    if not func_dir.is_dir():
        return [], False

    extras = []
    for bold in sorted(func_dir.glob(f"sub-{sub}_ses-{ses}_task-*_run-*_bold.nii.gz")):
        m = re.search(r"task-(\w+)_run-(\d+)", bold.name)
        if not m:
            continue
        task, run = m.group(1), m.group(2).zfill(2)
        if not _is_task_of_interest(task):
            continue
        if not _is_standard_run(task, run):
            extras.append((task, run))
    return extras, True


# ---------------------------------------------------------------------------
# Per-session check
# ---------------------------------------------------------------------------


def check_single_session(
    bids_dir: Path,
    sub: str,
    ses: str,
    df_note: pd.DataFrame,
) -> list[dict]:
    """
    Compare lab-note reruns with BIDS extra runs for one sub/ses.

    Returns a list of record dicts ready for writing to the TSV.
    """
    note_rows = df_note[(df_note["sub"] == sub) & (df_note["ses"] == ses)]
    bids_extras, func_exists = _bids_extras_for_session(bids_dir, sub, ses)

    if not func_exists:
        # No BIDS func dir — still emit lab-note rows with found_in_bids=False
        records = []
        for _, nr in note_rows.iterrows():
            records.append(
                {
                    "sub": sub,
                    "ses": ses,
                    "task": nr.task,
                    "extra_run": nr.extra_run,
                    "compensates_run": nr.compensates_run,
                    "protocol_name": nr.protocol_name,
                    "found_in_bids": "False",
                    "is_within_range": str(_is_standard_run(nr.task, nr.extra_run)),
                    "status": "NO_FUNC_DIR",
                }
            )
        return records

    bids_set = set(bids_extras)

    # Classify lab-note rows
    within_range: set[tuple[str, str]] = set()
    note_set: set[tuple[str, str]] = set()
    for _, nr in note_rows.iterrows():
        key = (nr.task, nr.extra_run)
        if _is_standard_run(nr.task, nr.extra_run):
            within_range.add(key)
        else:
            note_set.add(key)

    matched = note_set & bids_set
    only_note_extra = note_set - bids_set
    only_bids = bids_set - note_set

    if only_note_extra or only_bids:
        session_status = "MISMATCH"
    else:
        session_status = "OK"

    records = []
    for _, nr in note_rows.iterrows():
        key = (nr.task, nr.extra_run)
        found_in_bids = key in bids_set
        is_within_range = key in within_range

        if is_within_range and not found_in_bids:
            row_status = "OK"  # within-range rerun not visible in BIDS count
        elif not found_in_bids:
            row_status = "MISMATCH"
        elif key in only_bids:
            row_status = "MISMATCH"
        else:
            row_status = session_status

        records.append(
            {
                "sub": sub,
                "ses": ses,
                "task": nr.task,
                "extra_run": nr.extra_run,
                "compensates_run": nr.compensates_run,
                "protocol_name": nr.protocol_name,
                "found_in_bids": str(found_in_bids),
                "is_within_range": str(is_within_range),
                "status": row_status,
            }
        )

    # BIDS extras not in lab note → emit with unknown compensates_run
    for task, run in sorted(only_bids):
        records.append(
            {
                "sub": sub,
                "ses": ses,
                "task": task,
                "extra_run": run,
                "compensates_run": "?",
                "protocol_name": f"{task}_run-{run}_rerun-?",
                "found_in_bids": "True",
                "is_within_range": "False",
                "status": "MISMATCH",
            }
        )

    return records


# ---------------------------------------------------------------------------
# Sub/ses pair parsing
# ---------------------------------------------------------------------------


def _parse_pairs(
    subses_arg: Optional[str], file_arg: Optional[str]
) -> list[tuple[str, str]]:
    if subses_arg:
        parts = subses_arg.split(",")
        if len(parts) != 2:
            console.print(
                f"[red]ERROR[/red]: -s expects 'sub,ses' e.g. 01,09, got: {subses_arg!r}"
            )
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
                pairs.append(
                    (str(row["sub"]).strip().zfill(2), str(row["ses"]).strip().zfill(2))
                )
        return pairs

    console.print("[red]ERROR[/red]: provide either -s <sub,ses> or -f <subseslist>")
    raise typer.Exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    lab_note: Path = typer.Option(
        ..., "--lab-note", help="Path to the lab-note Excel (.xlsx) or TSV"
    ),
    bids_dir: Path = typer.Option(
        _BIDS_DIR, "--bids-dir", "-b", help="BIDS root directory"
    ),
    subses_arg: Optional[str] = typer.Option(
        None, "-s", help="Single sub,ses pair e.g. 04,09"
    ),
    file_arg: Optional[str] = typer.Option(
        None, "-f", help="Path to subseslist TSV/CSV"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output TSV path. Default: <bids_dir>/sourcedata/qc/rerun_check.tsv",
    ),
    execute: bool = typer.Option(
        False, "--execute", help="Write the output file (default: dry-run preview only)"
    ),
) -> None:
    """
    Generate rerun_check.tsv by parsing the lab-note Excel and cross-checking
    against BIDS.

    Dry-run by default — pass --execute to write the file.
    """
    pairs = _parse_pairs(subses_arg, file_arg)
    out_path = output or (bids_dir / "sourcedata" / "qc" / "rerun_check.tsv")
    mode_str = "[green]EXECUTE[/green]" if execute else "[yellow]DRY-RUN[/yellow]"

    console.rule("[bold cyan]generate_rerun_check_from_labnote[/bold cyan]")
    console.print(
        f"  Lab note   : {lab_note}\n"
        f"  BIDS dir   : {bids_dir}\n"
        f"  Sessions   : {len(pairs)}\n"
        f"  Output     : {out_path}\n"
        f"  Mode       : {mode_str}\n"
    )

    # Parse lab note once
    console.print("Reading lab note …")
    df_note = parse_lab_note(lab_note)
    console.print(
        f"  [green]{len(df_note)} rerun row(s)[/green] across "
        f"{df_note[['sub', 'ses']].drop_duplicates().shape[0]} session(s) in lab note.\n"
    )

    all_records: list[dict] = []

    for sub, ses in pairs:
        records = check_single_session(bids_dir, sub, ses, df_note)
        if not records:
            console.print(f"  [dim]sub-{sub} ses-{ses}  — no reruns in lab note[/dim]")
        else:
            for r in records:
                color = {
                    "OK": "green",
                    "MISMATCH": "bold red",
                    "NO_FUNC_DIR": "yellow",
                }.get(r["status"], "white")
                console.print(
                    f"  sub-{sub} ses-{ses}  "
                    f"task-{r['task']}  "
                    f"extra=run-{r['extra_run']}  "
                    f"compensates=run-{r['compensates_run']}  "
                    f"[{color}]{r['status']}[/{color}]"
                )
        all_records.extend(records)

    # Summary table
    console.print()
    if all_records:
        tbl = Table(title="Rerun records", box=box.SIMPLE_HEAD)
        for col in _TSV_FIELDS:
            tbl.add_column(col)
        _color = {"OK": "green", "MISMATCH": "bold red", "NO_FUNC_DIR": "yellow"}
        for r in all_records:
            c = _color.get(r["status"], "white")
            tbl.add_row(
                *[
                    f"[{c}]{r[col]}[/{c}]" if col == "status" else r[col]
                    for col in _TSV_FIELDS
                ]
            )
        console.print(tbl)

    n_ok = sum(1 for r in all_records if r["status"] == "OK")
    n_mismatch = sum(1 for r in all_records if r["status"] == "MISMATCH")
    n_nofunc = sum(1 for r in all_records if r["status"] == "NO_FUNC_DIR")
    console.print(
        f"  Total: [bold]{len(all_records)}[/bold] rows  "
        f"([green]OK={n_ok}[/green]  "
        f"[red]MISMATCH={n_mismatch}[/red]  "
        f"[yellow]NO_FUNC_DIR={n_nofunc}[/yellow])"
    )

    if not execute:
        console.print(
            f"\n  [dim]Dry-run — nothing written. Pass --execute to write {out_path}[/dim]"
        )
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_TSV_FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(all_records)
    console.print(f"\n  [green]Written →[/green] {out_path}  ({len(all_records)} rows)")


if __name__ == "__main__":
    app()
