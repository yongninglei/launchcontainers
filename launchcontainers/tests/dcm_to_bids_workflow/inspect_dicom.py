#!/usr/bin/env python3
"""
inspect_dicom.py
----------------
Print key DICOM header fields from one or more DICOM files.

Usage
-----
    python inspect_dicom.py path/to/file.dcm
    python inspect_dicom.py path/to/dicom_dir/          # reads first file found
    python inspect_dicom.py path/to/dicom_dir/ --all    # reads one file per series
    python inspect_dicom.py path/to/file.dcm --tags     # dump ALL tags
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pydicom
import typer
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()
app = typer.Typer(add_completion=False)

# Fields to show by default: (tag_keyword, label)
_FIELDS = [
    ("PatientID",                  "PatientID"),
    ("PatientName",                "PatientName"),
    ("StudyDate",                  "StudyDate"),
    ("StudyTime",                  "StudyTime"),
    ("AcquisitionDate",            "AcquisitionDate"),
    ("AcquisitionTime",            "AcquisitionTime"),
    ("ContentDate",                "ContentDate"),
    ("ContentTime",                "ContentTime"),
    ("SeriesNumber",               "SeriesNumber"),
    ("SeriesDescription",          "SeriesDescription"),
    ("ProtocolName",               "ProtocolName"),
    ("Modality",                   "Modality"),
    ("ImageType",                  "ImageType"),
    ("SequenceName",               "SequenceName"),
    ("ScanningSequence",           "ScanningSequence"),
    ("MRAcquisitionType",          "MRAcquisitionType"),
    ("SliceThickness",             "SliceThickness"),
    ("RepetitionTime",             "RepetitionTime (TR)"),
    ("EchoTime",                   "EchoTime (TE)"),
    ("FlipAngle",                  "FlipAngle"),
    ("NumberOfTemporalPositions",  "Volumes (nVols)"),
    ("Rows",                       "Rows"),
    ("Columns",                    "Columns"),
    ("PixelSpacing",               "PixelSpacing"),
    ("SliceLocation",              "SliceLocation"),
    ("InstanceNumber",             "InstanceNumber"),
    ("InPlanePhaseEncodingDirection", "PhaseEncDir"),
    ("Manufacturer",               "Manufacturer"),
    ("ManufacturerModelName",      "Scanner model"),
    ("MagneticFieldStrength",      "FieldStrength (T)"),
    ("InstitutionName",            "Institution"),
    ("SOPInstanceUID",             "SOPInstanceUID"),
    ("SeriesInstanceUID",          "SeriesInstanceUID"),
]


def _val(ds: pydicom.Dataset, keyword: str) -> str:
    v = getattr(ds, keyword, None)
    if v is None:
        return "[dim]—[/dim]"
    if isinstance(v, pydicom.sequence.Sequence):
        return f"<Sequence len={len(v)}>"
    if isinstance(v, pydicom.multival.MultiValue):
        return "  ".join(str(x) for x in v)
    if isinstance(v, bytes):
        return v.hex()
    return str(v)


def _collect_dcm_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = []
    for ext in ("*.dcm", "*.IMA", "*.ima", "*"):
        hits = [f for f in path.rglob(ext) if f.is_file()]
        if hits:
            files = hits
            break
    return sorted(files)


def _read(f: Path) -> pydicom.Dataset | None:
    try:
        return pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
    except Exception as e:
        console.print(f"[red]Cannot read {f}: {e}[/red]")
        return None


def _print_ds(ds: pydicom.Dataset, filepath: Path, show_all_tags: bool) -> None:
    console.print(f"\n[bold cyan]File:[/bold cyan] {filepath}")

    if show_all_tags:
        console.print(ds)
        return

    t = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
    t.add_column("Field", style="bold", no_wrap=True)
    t.add_column("Value")
    for keyword, label in _FIELDS:
        t.add_row(label, _val(ds, keyword))
    console.print(t)


@app.command()
def main(
    path: Path = typer.Argument(..., help="DICOM file or directory"),
    all_series: bool = typer.Option(
        False, "--all", "-a",
        help="When a directory is given, print one file per unique SeriesNumber.",
    ),
    show_all_tags: bool = typer.Option(
        False, "--tags", "-t",
        help="Dump every DICOM tag (verbose).",
    ),
) -> None:
    """Print key DICOM header fields."""
    if not path.exists():
        console.print(f"[red]Not found:[/red] {path}")
        raise typer.Exit(1)

    files = _collect_dcm_files(path)
    if not files:
        console.print(f"[red]No DICOM files found in {path}[/red]")
        raise typer.Exit(1)

    if path.is_file() or not all_series:
        ds = _read(files[0])
        if ds:
            _print_ds(ds, files[0], show_all_tags)
        return

    # --all: one file per SeriesNumber
    seen: dict[str, Path] = {}
    for f in files:
        ds = _read(f)
        if ds is None:
            continue
        key = str(ds.get("SeriesNumber", "?"))
        if key not in seen:
            seen[key] = f
    for series_num in sorted(seen, key=lambda x: int(x) if x.isdigit() else 0):
        ds = _read(seen[series_num])
        if ds:
            _print_ds(ds, seen[series_num], show_all_tags)


if __name__ == "__main__":
    app()
