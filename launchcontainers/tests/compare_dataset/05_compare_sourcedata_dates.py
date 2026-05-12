"""
05_compare_sourcedata_dates.py
------------------------------
Compare two sourcedata CSVs (produced by 04_check_sourcedata_dates.py).
For each (sub, ses) pair, report any mismatch in mat_date or floc_date.

Usage
-----
    python 05_compare_sourcedata_dates.py --csv1 bcbl_sourcedata.csv --csv2 nyu_sourcedata.csv
    python 05_compare_sourcedata_dates.py --csv1 a.csv --csv2 b.csv --output mismatches.csv
"""

import csv
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(pretty_exceptions_show_locals=False)


def _date_set(date_str: str) -> set[str]:
    """Extract unique YYYY-MM-DD dates from a comma-separated datetime string."""
    if not date_str or date_str.strip() == "-":
        return set()
    return {entry.strip()[:10] for entry in date_str.split(",") if entry.strip()}


def _load(path: Path) -> dict[tuple, dict]:
    """Return {(sub, ses): row} from a sourcedata CSV."""
    data = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            data[(row["sub"], row["ses"])] = row
    return data


@app.command()
def main(
    csv1: Path = typer.Option(..., "--csv1"),
    csv2: Path = typer.Option(..., "--csv2"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write mismatch CSV. Default: print only."
    ),
):
    d1 = _load(csv1)
    d2 = _load(csv2)

    all_keys = sorted(set(d1) | set(d2))

    mismatches = []

    for key in all_keys:
        sub, ses = key
        in1, in2 = key in d1, key in d2

        if not in1:
            mismatches.append(
                {
                    "sub": sub,
                    "ses": ses,
                    "issue": f"missing_in_{csv1.name}",
                    "field": "-",
                    "csv1_value": "-",
                    "csv2_value": d2[key]["mat_date"] + " / " + d2[key]["floc_date"],
                }
            )
            continue

        if not in2:
            mismatches.append(
                {
                    "sub": sub,
                    "ses": ses,
                    "issue": f"missing_in_{csv2.name}",
                    "field": "-",
                    "csv1_value": d1[key]["mat_date"] + " / " + d1[key]["floc_date"],
                    "csv2_value": "-",
                }
            )
            continue

        # compare mat_date (by calendar date, ignoring time)
        mat1 = _date_set(d1[key]["mat_date"])
        mat2 = _date_set(d2[key]["mat_date"])
        if mat1 != mat2:
            mismatches.append(
                {
                    "sub": sub,
                    "ses": ses,
                    "issue": "mat_date_mismatch",
                    "field": "mat_date",
                    "csv1_value": d1[key]["mat_date"],
                    "csv2_value": d2[key]["mat_date"],
                }
            )

        # compare floc_date
        floc1 = d1[key]["floc_date"].strip()
        floc2 = d2[key]["floc_date"].strip()
        if floc1 != floc2:
            mismatches.append(
                {
                    "sub": sub,
                    "ses": ses,
                    "issue": "floc_date_mismatch",
                    "field": "floc_date",
                    "csv1_value": floc1,
                    "csv2_value": floc2,
                }
            )

    # --- summary ---
    n_checked = len(all_keys)
    n_match = len(
        [k for k in all_keys if k in d1 and k in d2 and k not in {(m["sub"], m["ses"]) for m in mismatches}]
    )
    console.print(f"\nchecked  : {n_checked} (sub, ses) pairs")
    console.print(f"clean    : {n_match}")
    console.print(f"issues   : {len(mismatches)}")

    if mismatches:
        table = Table(title="Date Mismatches", show_lines=True)
        table.add_column("sub", style="cyan", no_wrap=True)
        table.add_column("ses", style="cyan", no_wrap=True)
        table.add_column("issue", style="red")
        table.add_column(f"csv1  ({csv1.name})", style="yellow")
        table.add_column(f"csv2  ({csv2.name})", style="green")

        for m in mismatches:
            table.add_row(m["sub"], m["ses"], m["issue"], m["csv1_value"], m["csv2_value"])

        console.print(table)
    else:
        console.print("[bold green]All sessions match.[/]")

    if output and mismatches:
        with open(output, "w", newline="") as fh:
            w = csv.DictWriter(
                fh, fieldnames=["sub", "ses", "issue", "field", "csv1_value", "csv2_value"]
            )
            w.writeheader()
            w.writerows(mismatches)
        console.print(f"[bold green]{len(mismatches)} rows written to {output}[/]")


if __name__ == "__main__":
    app()
