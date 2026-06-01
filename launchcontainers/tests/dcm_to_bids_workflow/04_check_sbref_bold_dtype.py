"""
04_check_sbref_bold_dtype.py
--------------------------------------
Check that sbref and bold NIfTI files have matching dtypes for each sub/ses.

For each run, reads the NIfTI header of the sbref and its paired bold and
compares the datatype code.  A mismatch means fmriprep will build hmc_boldref
from the sbref (correct sign) but motion-correct the bold (wrong sign) against
it, producing catastrophically wrong motion parameters and FD.

Also flags: scl_slope==0 (raw values unscaled — reader may misinterpret range).

Usage
-----
    python 04_check_sbref_bold_dtype.py -b /BIDS -s 05,day5BCBL
    python 04_check_sbref_bold_dtype.py -b /BIDS -f subseslist.tsv
    python 04_check_sbref_bold_dtype.py -b /BIDS -f subseslist.tsv --verbose
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import nibabel as nib
import typer
from rich.console import Console
from rich.table import Table

from launchcontainers.utils import parse_subses_list

console = Console()
app = typer.Typer(pretty_exceptions_show_locals=False)

# NIfTI dtype code → human-readable name
_DTYPE_NAMES = {
    2: "uint8",
    4: "int16",
    8: "int32",
    16: "float32",
    64: "float64",
    256: "int8",
    512: "uint16",
    768: "uint32",
}

# (sbref_dtype_code, bold_dtype_code) pairs that are safe for fmriprep HMC.
# Matching dtypes are always safe; mixed pairs are safe when the contrast
# polarity is preserved (both positive), so MCFLIRT's NMI/CC cost function
# can still register correctly.
_SAFE_PAIRS: set[tuple[int, int]] = {
    (512, 512),  # uint16 + uint16
    (4,   4),    # int16  + int16   (same sign, same scale → HMC works)
    (16,  16),   # float32 + float32
    (512, 16),   # uint16 sbref + float32 bold (benign: both positive, scale only)
}


def _pair_status(sb_code: int, bold_code: int) -> str:
    """Return 'ok', 'benign', or 'danger' for an sbref/bold dtype combination."""
    if sb_code == bold_code:
        return "ok"
    if (sb_code, bold_code) in _SAFE_PAIRS:
        return "benign"
    return "danger"


# ---------------------------------------------------------------------------
# NIfTI header reading
# ---------------------------------------------------------------------------


def _nii_info(path: Path) -> dict:
    hdr = nib.load(str(path)).header
    code = int(hdr["datatype"])
    slope = float(hdr["scl_slope"])
    freq, phase, slc = hdr.get_dim_info()
    return {
        "dtype_code": code,
        "dtype_name": _DTYPE_NAMES.get(code, f"unknown({code})"),
        "scl_slope": slope,
        "freq": freq,
        "phase": phase,
        "slice": slc,
    }


# ---------------------------------------------------------------------------
# Collect and match
# ---------------------------------------------------------------------------


def _run_key(stem: str) -> str:
    """Strip _echo-N so each run has one key regardless of ME echoes."""
    return re.sub(r"_echo-\d+", "", stem)


# Dtypes expected for raw scanner images (anat, fmap).
# float32 from dcm2niix is unusual and worth flagging.
_RAW_EXPECTED_CODES: set[int] = {4, 512}  # int16, uint16


def _collect_modality(bidsdir: Path, sub: str, ses: str, modality: str) -> list[dict]:
    """Return one entry per NIfTI in sub/ses/<modality>/ with header info."""
    mod_dir = bidsdir / f"sub-{sub}" / f"ses-{ses}" / modality
    if not mod_dir.is_dir():
        return []
    entries = []
    for nii in sorted(mod_dir.glob(f"sub-{sub}_ses-{ses}_*.nii.gz")):
        info = _nii_info(nii)
        unusual = info["dtype_code"] not in _RAW_EXPECTED_CODES
        entries.append(
            {
                "name": re.sub(
                    rf"^sub-{sub}_ses-{ses}_", "", nii.name.removesuffix(".nii.gz")
                ),
                "path": nii,
                "info": info,
                "unusual": unusual,
            }
        )
    return entries


def _collect(bidsdir: Path, sub: str, ses: str) -> dict:
    func_dir = bidsdir / f"sub-{sub}" / f"ses-{ses}" / "func"
    if not func_dir.is_dir():
        return {"found": False, "sbrefs": {}, "bolds": {}}

    sbrefs: dict[str, Path] = {}
    bolds: dict[str, Path] = {}

    for nii in sorted(func_dir.glob(f"sub-{sub}_ses-{ses}_*.nii.gz")):
        stem = nii.name.removesuffix(".nii.gz")
        if stem.endswith("_sbref"):
            key = stem.removesuffix("_sbref")
            sbrefs[key] = nii
        elif stem.endswith("_bold"):
            key = _run_key(stem.removesuffix("_bold"))
            bolds.setdefault(key, nii)  # keep first echo only

    return {"found": True, "sbrefs": sbrefs, "bolds": bolds}


def _check_session(bidsdir: Path, sub: str, ses: str) -> dict:
    collected = _collect(bidsdir, sub, ses)
    anat_entries = _collect_modality(bidsdir, sub, ses, "anat")
    fmap_entries = _collect_modality(bidsdir, sub, ses, "fmap")
    if not collected["found"]:
        return {
            "sub": sub, "ses": ses, "found": False,
            "pairs": [], "n_danger": 0, "n_benign": 0, "n_slope0": 0,
            "anat": anat_entries, "fmap": fmap_entries,
            "n_unusual_anat": sum(1 for e in anat_entries if e["unusual"]),
            "n_unusual_fmap": sum(1 for e in fmap_entries if e["unusual"]),
        }

    sbrefs = collected["sbrefs"]
    bolds = collected["bolds"]

    pairs = []
    for key, sb_path in sorted(sbrefs.items()):
        bold_path = bolds.get(key)
        sb_info = _nii_info(sb_path)
        bold_info = _nii_info(bold_path) if bold_path else None

        status = (
            _pair_status(sb_info["dtype_code"], bold_info["dtype_code"])
            if bold_info is not None
            else "no_bold"
        )
        slope0_bold = bold_info is not None and bold_info["scl_slope"] == 0.0
        slope0_sbref = sb_info["scl_slope"] == 0.0

        pairs.append(
            {
                "key": key,
                "sb_path": sb_path,
                "bold_path": bold_path,
                "sb_info": sb_info,
                "bold_info": bold_info,
                "status": status,
                "slope0_bold": slope0_bold,
                "slope0_sbref": slope0_sbref,
            }
        )

    n_danger = sum(1 for p in pairs if p["status"] == "danger")
    n_benign = sum(1 for p in pairs if p["status"] == "benign")
    n_slope0 = sum(1 for p in pairs if p["slope0_bold"])
    n_unusual_anat = sum(1 for e in anat_entries if e["unusual"])
    n_unusual_fmap = sum(1 for e in fmap_entries if e["unusual"])

    return {
        "sub": sub,
        "ses": ses,
        "found": True,
        "pairs": pairs,
        "n_danger": n_danger,
        "n_benign": n_benign,
        "n_slope0": n_slope0,
        "anat": anat_entries,
        "fmap": fmap_entries,
        "n_unusual_anat": n_unusual_anat,
        "n_unusual_fmap": n_unusual_fmap,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _run_label(key: str, sub: str, ses: str) -> str:
    return re.sub(rf"^sub-{sub}_ses-{ses}_", "", key)


_STATUS_DISPLAY = {
    "ok":      ("[green]ok[/]",      lambda sb, bd: (sb, bd)),
    "benign":  ("[yellow]ok (benign)[/]", lambda sb, bd: (f"[cyan]{sb}[/]", f"[yellow]{bd}[/]")),
    "danger":  ("[red]DANGER[/]",    lambda sb, bd: (f"[cyan]{sb}[/]", f"[red]{bd}[/]")),
    "no_bold": ("[dim]—[/]",         lambda sb, bd: (sb, bd)),
}


def _print_modality_table(entries: list[dict], label: str) -> None:
    if not entries:
        return
    t = Table(
        title=f"[bold]{label}[/]",
        show_header=True,
        header_style="bold magenta",
        box=None,
        padding=(0, 1),
    )
    t.add_column("file", style="dim")
    t.add_column("dtype", justify="center")
    t.add_column("note", justify="center")
    for e in entries:
        dtype_str = (
            f"[yellow]{e['info']['dtype_name']}[/]"
            if e["unusual"]
            else e["info"]["dtype_name"]
        )
        note = "[yellow]unusual for raw scanner data[/]" if e["unusual"] else ""
        t.add_row(e["name"], dtype_str, note)
    console.print(t)


def _print_session(result: dict, verbose: bool) -> None:
    sub, ses = result["sub"], result["ses"]
    n_danger = result["n_danger"]
    n_benign = result["n_benign"]
    n_slope0 = result["n_slope0"]
    n_unusual_anat = result["n_unusual_anat"]
    n_unusual_fmap = result["n_unusual_fmap"]

    has_issues = n_danger or n_slope0 or n_unusual_anat or n_unusual_fmap
    if not verbose and not has_issues:
        return

    parts = []
    if n_danger:
        parts.append(f"[red]DANGER dtype-mismatch {n_danger}[/]")
    if n_benign:
        parts.append(f"[yellow]benign mismatch {n_benign}[/]")
    if n_slope0:
        parts.append(f"[yellow]scl_slope=0 {n_slope0}[/]")
    if n_unusual_anat:
        parts.append(f"[yellow]unusual anat dtype {n_unusual_anat}[/]")
    if n_unusual_fmap:
        parts.append(f"[yellow]unusual fmap dtype {n_unusual_fmap}[/]")
    flag = "  " + "  ".join(parts) if parts else " [green]OK[/]"
    console.print(f"\n[bold cyan]sub-{sub}  ses-{ses}[/]{flag}")

    t = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 1))
    t.add_column("run", style="dim")
    t.add_column("sbref dtype", justify="center")
    t.add_column("bold dtype", justify="center")
    t.add_column("hmc safe", justify="center")
    t.add_column("scl_slope bold", justify="center")

    for p in result["pairs"]:
        run_lbl = _run_label(p["key"], sub, ses)
        sb_dtype = p["sb_info"]["dtype_name"]
        bold_dtype = p["bold_info"]["dtype_name"] if p["bold_info"] else "[dim]no bold[/]"

        status_str, colour_fn = _STATUS_DISPLAY.get(p["status"], ("[dim]?[/]", lambda s, b: (s, b)))
        sb_dtype, bold_dtype = colour_fn(sb_dtype, bold_dtype)

        slope_str = (
            "[dim]—[/]"
            if p["bold_info"] is None
            else ("[red]0.0[/]" if p["slope0_bold"] else f"{p['bold_info']['scl_slope']:.1f}")
        )

        t.add_row(run_lbl, sb_dtype, bold_dtype, status_str, slope_str)

    console.print(t)

    if result["anat"] or verbose:
        _print_modality_table(result["anat"], "anat")
    if result["fmap"] or verbose:
        _print_modality_table(result["fmap"], "fmap")


def _print_summary(results: list[dict]) -> None:
    t = Table(
        title="sbref / bold dtype check summary",
        show_header=True,
        header_style="bold magenta",
        box=None,
        padding=(0, 1),
    )
    t.add_column("sub", justify="right", style="cyan")
    t.add_column("ses", justify="right", style="cyan")
    t.add_column("pairs", justify="right")
    t.add_column("danger", justify="right")
    t.add_column("benign", justify="right")
    t.add_column("anat unusual", justify="right")
    t.add_column("fmap unusual", justify="right")
    t.add_column("status", justify="center")

    for r in results:
        if not r["found"]:
            t.add_row(r["sub"], r["ses"], "—", "—", "—", "—", "—", "[yellow]not found[/]")
            continue
        n = len(r["pairs"])
        nd = r["n_danger"]
        nb = r["n_benign"]
        na = r["n_unusual_anat"]
        nf = r["n_unusual_fmap"]
        if nd:
            st = "[red]DANGER[/]"
        elif na or nf:
            st = "[yellow]unusual dtype[/]"
        else:
            st = "[green]OK[/]"
        t.add_row(
            r["sub"],
            r["ses"],
            str(n),
            f"[red]{nd}[/]" if nd else "0",
            f"[yellow]{nb}[/]" if nb else "0",
            f"[yellow]{na}[/]" if na else "0",
            f"[yellow]{nf}[/]" if nf else "0",
            st,
        )

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
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print per-run table for every session, not just problem ones.",
    ),
):
    """
    Check that sbref and bold NIfTI dtypes match for every run in each session.

    A dtype mismatch (e.g. sbref=uint16, bold=int16) means fmriprep will
    motion-correct sign-inverted BOLD volumes against the sbref reference,
    producing catastrophically wrong motion parameters (~30–200 mm FD).

    Examples:
        python 04_check_sbref_bold_dtype.py -b /BIDS -s 05,day5BCBL
        python 04_check_sbref_bold_dtype.py -b /BIDS -f subseslist.tsv
        python 04_check_sbref_bold_dtype.py -b /BIDS -f subseslist.tsv -v
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

    console.print(f"\n[bold]04_check_sbref_bold_dtype[/]  bidsdir={bidsdir}\n")

    results = []
    for sub, ses in pairs:
        r = _check_session(bidsdir, sub, ses)
        _print_session(r, verbose)
        results.append(r)

    console.print()
    _print_summary(results)


if __name__ == "__main__":
    app()
