"""
05_fix_bold_dtype.py
--------------------------------------
Fix BOLD NIfTI files whose header dtype is int16 but whose raw bytes are
actually uint16 data (dcm2niix conversion bug).

The fix reinterprets the raw bytes in-place (no data values change) and
rewrites the NIfTI header to say uint16, scl_slope=1, scl_inter=0.
After the fix the BOLD intensities match the sbref (also uint16), so
fmriprep HMC can register them correctly.

Only operates on BOLD files where the sbref for that session is uint16
and the BOLD is int16 (the dangerous mismatch pattern).  All other
dtype combinations are skipped with an explanation.

Usage
-----
    python 05_fix_bold_dtype.py -b /BIDS -s 05,day5BCBL           # dry-run
    python 05_fix_bold_dtype.py -b /BIDS -s 05,day5BCBL --execute # apply
    python 05_fix_bold_dtype.py -b /BIDS -f subseslist.tsv --execute
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from launchcontainers.utils import parse_subses_list

console = Console()
app = typer.Typer(pretty_exceptions_show_locals=False)

_DTYPE_NAMES = {4: "int16", 16: "float32", 512: "uint16"}


# ---------------------------------------------------------------------------
# Header inspection
# ---------------------------------------------------------------------------


def _dtype_code(path: Path) -> int:
    return int(nib.load(str(path)).header["datatype"])


def _find_sbref_dtype(func_dir: Path, sub: str, ses: str) -> int | None:
    """Return the dtype code of the first sbref found in func_dir, or None."""
    for p in sorted(func_dir.glob(f"sub-{sub}_ses-{ses}_*_sbref.nii.gz")):
        return _dtype_code(p)
    return None


def _collect_bold(func_dir: Path, sub: str, ses: str) -> list[Path]:
    """All bold NIfTI files (excluding ME echo duplicates: keep run-level only)."""
    seen: set[str] = set()
    result: list[Path] = []
    for p in sorted(func_dir.glob(f"sub-{sub}_ses-{ses}_*_bold.nii.gz")):
        key = re.sub(r"_echo-\d+", "", p.name)
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def _fix_bold(path: Path, dry_run: bool) -> dict:
    """
    Reinterpret raw bytes as uint16 and rewrite the NIfTI header.
    Returns a result dict describing what happened.
    """
    code = _dtype_code(path)
    if code != 4:  # not int16 — nothing to do
        return {"path": path, "action": "skip", "reason": f"dtype={_DTYPE_NAMES.get(code, code)} not int16"}

    if dry_run:
        return {"path": path, "action": "would-fix", "reason": "int16 → uint16"}

    img = nib.load(str(path))
    raw_int16 = np.asarray(img.dataobj)          # read bytes as int16 (wrong label)
    data_uint16 = raw_int16.view(np.uint16)      # reinterpret same bytes as uint16

    new_hdr = img.header.copy()
    new_hdr.set_data_dtype(np.uint16)
    new_hdr["scl_slope"] = 1.0
    new_hdr["scl_inter"] = 0.0

    new_img = nib.Nifti1Image(data_uint16, img.affine, new_hdr)

    # Write to temp file in same directory, then atomically replace
    tmp = Path(tempfile.mktemp(dir=path.parent, suffix=".nii.gz"))
    try:
        nib.save(new_img, str(tmp))
        shutil.move(str(tmp), str(path))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    return {"path": path, "action": "fixed", "reason": "int16 → uint16"}


# ---------------------------------------------------------------------------
# Per-session driver
# ---------------------------------------------------------------------------


def _process_session(bidsdir: Path, sub: str, ses: str, dry_run: bool) -> dict:
    func_dir = bidsdir / f"sub-{sub}" / f"ses-{ses}" / "func"
    if not func_dir.is_dir():
        return {"sub": sub, "ses": ses, "found": False, "results": [], "skipped_reason": "session not found"}

    sbref_code = _find_sbref_dtype(func_dir, sub, ses)
    if sbref_code is None:
        return {"sub": sub, "ses": ses, "found": True, "results": [], "skipped_reason": "no sbref found"}

    if sbref_code != 512:  # sbref not uint16 — this fix does not apply
        name = _DTYPE_NAMES.get(sbref_code, str(sbref_code))
        return {
            "sub": sub, "ses": ses, "found": True, "results": [],
            "skipped_reason": f"sbref dtype={name} (fix only targets uint16 sbref + int16 bold)",
        }

    bold_files = _collect_bold(func_dir, sub, ses)
    results = []
    for p in bold_files:
        r = _fix_bold(p, dry_run=dry_run)
        results.append(r)

    return {"sub": sub, "ses": ses, "found": True, "results": results, "skipped_reason": None}


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _print_session(result: dict, verbose: bool) -> None:
    sub, ses = result["sub"], result["ses"]

    if not result["found"] or result.get("skipped_reason"):
        reason = result.get("skipped_reason", "not found")
        if verbose:
            console.print(f"  [dim]sub-{sub} ses-{ses}: skipped ({reason})[/]")
        return

    results = result["results"]
    n_fixed = sum(1 for r in results if r["action"] in ("fixed", "would-fix"))
    n_skip = sum(1 for r in results if r["action"] == "skip")

    parts = []
    if n_fixed:
        parts.append(f"[green]{n_fixed} fixed[/]")
    if n_skip:
        parts.append(f"[dim]{n_skip} skipped[/]")
    flag = "  " + "  ".join(parts) if parts else " [dim]nothing to do[/]"
    console.print(f"\n[bold cyan]sub-{sub}  ses-{ses}[/]{flag}")

    if not verbose and n_fixed == 0:
        return

    t = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 1))
    t.add_column("file", style="dim")
    t.add_column("action", justify="center")
    t.add_column("note")

    for r in results:
        fname = r["path"].name.removesuffix(".nii.gz")
        fname = re.sub(rf"^sub-{sub}_ses-{ses}_", "", fname)
        action = r["action"]
        if action == "fixed":
            action_str = "[green]fixed[/]"
        elif action == "would-fix":
            action_str = "[yellow]would-fix[/]"
        else:
            action_str = "[dim]skip[/]"
        t.add_row(fname, action_str, r["reason"])

    console.print(t)


def _print_summary(session_results: list[dict], dry_run: bool) -> None:
    t = Table(
        title="bold dtype fix summary",
        show_header=True,
        header_style="bold magenta",
        box=None,
        padding=(0, 1),
    )
    t.add_column("sub", justify="right", style="cyan")
    t.add_column("ses", justify="right", style="cyan")
    t.add_column("bold files", justify="right")
    t.add_column("fixed" if not dry_run else "would-fix", justify="right")
    t.add_column("skipped", justify="right")
    t.add_column("status", justify="center")

    for r in session_results:
        if not r["found"] or r.get("skipped_reason"):
            t.add_row(r["sub"], r["ses"], "—", "—", "—", f"[dim]{r.get('skipped_reason', 'not found')}[/]")
            continue
        results = r["results"]
        n_total = len(results)
        n_fixed = sum(1 for x in results if x["action"] in ("fixed", "would-fix"))
        n_skip = sum(1 for x in results if x["action"] == "skip")
        st = "[green]done[/]" if n_fixed and not dry_run else ("[yellow]dry-run[/]" if n_fixed else "[dim]nothing to do[/]")
        t.add_row(r["sub"], r["ses"], str(n_total), str(n_fixed), str(n_skip), st)

    console.print(t)


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
        help="sub,ses pair e.g. -s 05,day5BCBL  (repeatable).",
    ),
    subses_file: Optional[Path] = typer.Option(
        None, "--file", "-f", help="TSV/CSV with columns sub, ses."
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--execute",
        help="Print plan only (default) or apply fixes.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show per-file table for every session."
    ),
):
    """
    Fix BOLD NIfTI files where the header says int16 but the raw bytes are
    uint16 (dcm2niix bug).  Only targets sessions where sbref=uint16 and
    bold=int16 (the combination that causes catastrophic HMC failure in
    fmriprep).

    The fix: reinterpret the raw bytes as uint16 and update the header.
    No data values are changed; file size is unchanged.

    Default: dry-run — prints what would be done without touching any files.
    Use --execute to apply.

    Examples:
        python 05_fix_bold_dtype.py -b /BIDS -s 05,day5BCBL
        python 05_fix_bold_dtype.py -b /BIDS -s 05,day5BCBL -s 05,day6BCBL --execute
        python 05_fix_bold_dtype.py -b /BIDS -f subseslist.tsv --execute
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
            pairs.append((parts[0].strip(), parts[1].strip()))
    else:
        console.print("[bold red]Error:[/] provide -s sub,ses  or  -f <subseslist>.")
        raise typer.Exit(1)

    mode_str = "[yellow]DRY-RUN[/]" if dry_run else "[bold red]EXECUTE[/]"
    console.print(f"\n[bold]05_fix_bold_dtype[/]  bidsdir={bidsdir}  mode={mode_str}\n")

    session_results = []
    for sub, ses in pairs:
        r = _process_session(bidsdir, sub, ses, dry_run=dry_run)
        _print_session(r, verbose)
        session_results.append(r)

    console.print()
    _print_summary(session_results, dry_run)

    if dry_run:
        console.print("\n[yellow]Dry-run complete.  Use --execute to apply fixes.[/]")


if __name__ == "__main__":
    app()
