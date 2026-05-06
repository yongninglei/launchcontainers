"""
04_check_sourcedata_dates.py
----------------------------
Walk BIDS/sourcedata, find per-session 20*.mat files and *10runs dirs,
parse their acquisition dates, and print a summary table.

Usage
-----
    python 04_check_sourcedata_dates.py --sourcedir /path/to/BIDS/sourcedata
    python 04_check_sourcedata_dates.py --sourcedir /path/to/BIDS/sourcedata --output dates.csv
"""

import csv
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(pretty_exceptions_show_locals=False)


def parse_mat_datetime(mat_filename: str) -> Optional[datetime]:
    """Extract datetime from .mat filename 20250804T111657.mat, adjusted -6 min."""
    try:
        dt_str = mat_filename.replace(".mat", "")
        dt = datetime.strptime(dt_str, "%Y%m%dT%H%M%S")
        return dt - timedelta(minutes=6)
    except Exception:
        return None


def parse_floc_datetime(dirname: str) -> Optional[datetime]:
    """Extract date from *10runs dirname, e.g. sub-11_ses-08_task-fLoc_01-Sep-2025_..._10runs."""
    m = re.search(r"(\d{2}-[A-Za-z]{3}-\d{4})", dirname)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%d-%b-%Y")
    except Exception:
        return None


@app.command()
def main(
    sourcedir: Path = typer.Option(
        ..., "--sourcedir", "-s", help="Path to BIDS/sourcedata"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Optional CSV output path"
    ),
):
    rows = []

    for sub_dir in sorted(sourcedir.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        sub_id = sub_dir.name.replace("sub-", "")

        for ses_dir in sorted(sub_dir.glob("ses-*")):
            if not ses_dir.is_dir():
                continue
            ses_id = ses_dir.name.replace("ses-", "")

            mat_dates = []
            for mf in sorted(ses_dir.glob("20*.mat")):
                dt = parse_mat_datetime(mf.name)
                if dt:
                    mat_dates.append(dt.strftime("%Y-%m-%d %H:%M"))

            floc_dates = []
            for fd in sorted(
                d for d in ses_dir.iterdir() if d.is_dir() and d.name.endswith("10runs")
            ):
                dt = parse_floc_datetime(fd.name)
                if dt:
                    floc_dates.append(dt.strftime("%Y-%m-%d"))

            rows.append(
                {
                    "sub": sub_id,
                    "ses": ses_id,
                    "mat_date": ", ".join(mat_dates) if mat_dates else "-",
                    "floc_date": ", ".join(floc_dates) if floc_dates else "-",
                }
            )

    table = Table(title="Sourcedata Date Summary", show_lines=True)
    table.add_column("sub", style="cyan")
    table.add_column("ses", style="cyan")
    table.add_column("mat_date", style="green")
    table.add_column("floc_date", style="yellow")

    for r in rows:
        table.add_row(r["sub"], r["ses"], r["mat_date"], r["floc_date"])

    console.print(table)

    if output:
        with open(output, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["sub", "ses", "mat_date", "floc_date"])
            w.writeheader()
            w.writerows(rows)
        console.print(f"[bold green]{len(rows)} rows written to {output}[/]")


if __name__ == "__main__":
    app()
