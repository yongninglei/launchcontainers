"""
01_drop_duplicated_sbref_and_fmaps.py
--------------------------------------
Drop/rename orphaned or mismatched sbrefs.

Pre-requisite
-------------
Run ``00_merge_split_ses_reording_runs.py`` first to consolidate split session
directories (ses-02 + ses-02part2 → ses-02).  This script assumes a single
session dir.

Fmap pruning is handled by ``02_prepare_fmap_intendedfor.py`` in the next step.

Sbref matching rule
-------------------
For each sbref (in chronological order):
  - Find the nearest func (bold or magnitude) that comes AFTER it in time.
  - If that func is within ``--max-gap`` seconds (default 30 s) → KEEP.
  - Otherwise → DROP.

After deciding which sbrefs to keep, renumber them sequentially (run-01,
run-02, …) by position.  A kept sbref whose current run number already matches
its position → "ok".  One that differs → "rename".

Rename/drop strategy
--------------------
Two-phase rename: old → temp (``._tmp_<hex8>_*``) → final.
Drops happen BEFORE renames to avoid clobbering a freshly renamed file.
Both ``.nii.gz`` and ``.json`` are handled together.

Usage
-----
    python 01_drop_duplicated_sbref_and_fmaps.py -b /BIDS -s 10,02
    python 01_drop_duplicated_sbref_and_fmaps.py -b /BIDS -s 10,02 -s 11,03 --execute
    python 01_drop_duplicated_sbref_and_fmaps.py -b /BIDS -f subseslist.tsv --execute -v
"""

from __future__ import annotations

import csv
import glob
import json
import os
import os.path as op
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from launchcontainers.utils import parse_subses_list

console = Console()
app = typer.Typer(pretty_exceptions_show_locals=False)

DEFAULT_MAX_GAP = 30  # seconds — sbrefs farther than this from any func are orphans


# ---------------------------------------------------------------------------
# Generic helpers  (shared pattern with show_bids_acqtimes.py)
# ---------------------------------------------------------------------------


def _to_sec(t: str | None) -> float:
    """HH:MM:SS[.xxx] → seconds since midnight.  Returns inf on failure."""
    if not t:
        return float("inf")
    try:
        parts = str(t).split(":")
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except Exception:
        return float("inf")


def _fmt_time(t: str | None) -> str:
    if not t:
        return ""
    return str(t).split(".")[0]


def _short_name(basename: str, sub: str, ses_label: str) -> str:
    name = re.sub(rf"^sub-{sub}_ses-{ses_label}_", "", basename)
    return re.sub(r"\.(nii\.gz|json)$", "", name)


def _find_ses_dirs(bidsdir: str, sub: str, ses: str) -> list[str]:
    """Return all session directories matching ses-{ses}* (handles ses-02part2 etc.)."""
    sub_dir = op.join(bidsdir, f"sub-{sub}")
    matches = sorted(glob.glob(op.join(sub_dir, f"ses-{ses}*")))
    return [d for d in matches if op.isdir(d)]


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def _collect_func_and_sbref(bidsdir: str, sub: str, ses: str) -> dict:
    """
    Collect func (bold/magnitude) and sbref files from the single merged
    session directory ses-{ses}.

    Assumes 00_merge_split_ses_reording_runs.py has already been run; only the
    primary ses-{ses} directory is read (split parts are ignored here).

    Returns:
      funcs    — sorted by acq_sec
      sbrefs   — sorted by acq_sec
      ses_dirs — list of the single matching directory (empty if not found)
    """
    ses_dirs = _find_ses_dirs(bidsdir, sub, ses)
    # Use only the exact primary directory (ses-{ses}), ignore split parts
    primary = op.join(bidsdir, f"sub-{sub}", f"ses-{ses}")
    ses_dirs = [primary] if op.isdir(primary) else []

    funcs: list[dict] = []
    sbrefs: list[dict] = []

    for ses_dir in ses_dirs:
        ses_label = ses
        func_dir = op.join(ses_dir, "func")

        if op.isdir(func_dir):
            for jf in sorted(
                glob.glob(op.join(func_dir, f"sub-{sub}_ses-{ses_label}_*.json"))
            ):
                basename = op.basename(jf)
                stem = re.sub(r"\.json$", "", basename)
                short = _short_name(basename, sub, ses_label)
                nii_path = op.join(func_dir, stem + ".nii.gz")

                acq_time = ""
                try:
                    with open(jf) as fh:
                        acq_time = json.load(fh).get("AcquisitionTime", "")
                except Exception:
                    pass

                entry = {
                    "ses_label": ses_label,
                    "name": short,
                    "basename": stem,  # without extension
                    "json_path": jf,
                    "nii_path": nii_path,
                    "acq_time": acq_time,
                    "acq_sec": _to_sec(acq_time),
                    "func_dir": func_dir,
                }

                if short.endswith("_sbref"):
                    sbrefs.append(entry)
                elif short.endswith("_bold") or short.endswith("_magnitude"):
                    funcs.append(entry)
                # else: phase, gfactor, etc. — ignore

    funcs.sort(key=lambda r: r["acq_sec"])
    sbrefs.sort(key=lambda r: r["acq_sec"])

    return {"funcs": funcs, "sbrefs": sbrefs, "ses_dirs": ses_dirs}


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _match_sbrefs(funcs: list[dict], sbrefs: list[dict], max_gap: int) -> list[dict]:
    """
    Match sbrefs to funcs and determine rename/drop actions.

    Logic
    -----
    1. For each sbref (in time order) find the nearest func that comes AFTER
       it within max_gap seconds.  If none → DROP.
    2. After all matches are determined, renumber the KEPT sbrefs sequentially
       (run-01, run-02, …) by position in time order.  A kept sbref whose
       current run number already matches its new position → "ok".
       One whose run number differs → "rename".

    This means dropping run-01_sbref automatically causes run-02_sbref to be
    renamed run-01_sbref, run-03_sbref → run-02_sbref, etc.

    Returns a list of match records:
      sbref       — the sbref entry
      func        — nearest following func (set even for DROP, for display)
      delta_sec   — time gap to that func (None if no following func at all)
      action      — "ok" | "rename" | "drop"
      new_name    — new target basename (only set when action == "rename")
    """
    results = []

    for sbref in sbrefs:
        sb_sec = sbref["acq_sec"]

        # Find the nearest func that follows this sbref
        nearest_func = None
        nearest_delta = float("inf")

        for func in funcs:
            fn_sec = func["acq_sec"]
            if fn_sec == float("inf") or sb_sec == float("inf"):
                continue
            if fn_sec < sb_sec:
                continue  # must come after the sbref
            delta = fn_sec - sb_sec
            if delta < nearest_delta:
                nearest_delta = delta
                nearest_func = func

        within_gap = nearest_func is not None and nearest_delta <= max_gap
        results.append(
            {
                "sbref": sbref,
                "func": nearest_func,  # shown in display even for DROP
                "delta_sec": nearest_delta if nearest_func else None,
                "action": "keep_pending" if within_gap else "drop",
                "new_name": None,
            }
        )

    # Renumber kept sbrefs sequentially PER TASK, in time order.
    # e.g. retRW sbrefs get run-01, run-02 independently of retFF sbrefs.
    task_counters: dict[str, int] = {}  # task_label → next run number
    kept = [r for r in results if r["action"] == "keep_pending"]
    for r in kept:
        sbref = r["sbref"]
        task_m = re.search(r"_task-(\w+)_", sbref["basename"])
        task = task_m.group(1) if task_m else "_notask_"

        task_counters[task] = task_counters.get(task, 0) + 1
        new_run_int = task_counters[task]

        run_m = re.search(r"_run-(\d+)_", sbref["basename"])
        old_run_int = int(run_m.group(1)) if run_m else None

        if old_run_int is not None and old_run_int != new_run_int:
            old_tag = f"_run-{run_m.group(1)}_"
            new_tag = f"_run-{new_run_int:02d}_"
            r["action"] = "rename"
            r["new_name"] = sbref["basename"].replace(old_tag, new_tag, 1)
        else:
            r["action"] = "ok"

    return results


# ---------------------------------------------------------------------------
# Execute: two-phase rename + drop
# ---------------------------------------------------------------------------


def _execute(matches: list[dict], dry_run: bool) -> dict:
    """
    Apply renames and drops.  Returns counts dict.
    """
    to_rename = [m for m in matches if m["action"] == "rename"]
    to_drop = [m for m in matches if m["action"] == "drop"]

    counts = {"renamed": 0, "dropped": 0, "errors": 0}

    # ---- Drop orphan sbrefs FIRST (before any renames) ----
    # Must drop before renaming so that a renamed file (e.g. run-02 → run-01)
    # is not immediately deleted by the drop of the original run-01 path.
    for m in to_drop:
        sbref = m["sbref"]
        if not dry_run:
            try:
                for path in (sbref["nii_path"], sbref["json_path"]):
                    if op.exists(path):
                        os.remove(path)
                counts["dropped"] += 1
            except Exception as exc:
                console.print(f"  [red][ERROR] drop: {exc}[/]")
                counts["errors"] += 1
        else:
            counts["dropped"] += 1

    # ---- Phase 1 + 2: rename ----
    # Phase 1: move to temp names
    temp_records = []
    for m in to_rename:
        sbref = m["sbref"]
        temp_id = uuid.uuid4().hex[:8]
        func_dir = sbref["func_dir"]
        temp_base = f"._tmp_{temp_id}_sbref"

        temp_nii = op.join(func_dir, temp_base + ".nii.gz")
        temp_json = op.join(func_dir, temp_base + ".json")

        if not dry_run:
            try:
                if op.exists(sbref["nii_path"]):
                    os.rename(sbref["nii_path"], temp_nii)
                if op.exists(sbref["json_path"]):
                    os.rename(sbref["json_path"], temp_json)
            except Exception as exc:
                console.print(f"  [red][ERROR] phase-1 rename: {exc}[/]")
                counts["errors"] += 1
                temp_records.append(None)
                continue

        temp_records.append(
            {
                "temp_nii": temp_nii,
                "temp_json": temp_json,
                "new_base": m["new_name"],
                "func_dir": func_dir,
            }
        )

    # Phase 2: move from temp to final
    for rec in temp_records:
        if rec is None:
            continue
        final_nii = op.join(rec["func_dir"], rec["new_base"] + ".nii.gz")
        final_json = op.join(rec["func_dir"], rec["new_base"] + ".json")

        if not dry_run:
            try:
                if op.exists(rec["temp_nii"]):
                    os.rename(rec["temp_nii"], final_nii)
                if op.exists(rec["temp_json"]):
                    os.rename(rec["temp_json"], final_json)
                counts["renamed"] += 1
            except Exception as exc:
                console.print(f"  [red][ERROR] phase-2 rename: {exc}[/]")
                counts["errors"] += 1
        else:
            counts["renamed"] += 1

    return counts


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_ACTION_STYLE = {
    "ok": "green",
    "rename": "yellow",
    "drop": "red",
}


def _print_session_table(
    sub: str,
    ses: str,
    funcs: list[dict],
    sbrefs: list[dict],
    matches: list[dict],
    max_gap: int,
) -> None:
    """Print the func / sbref acq-time table with match annotations."""

    sbref_match: dict[str, dict] = {m["sbref"]["name"]: m for m in matches}

    all_rows = []
    for r in funcs:
        all_rows.append(("func", r["acq_sec"], r["acq_time"], r["name"], None))
    for r in sbrefs:
        m = sbref_match.get(r["name"])
        all_rows.append(("sbref", r["acq_sec"], r["acq_time"], r["name"], m))
    all_rows.sort(key=lambda x: x[1])

    n_sb_drop = sum(1 for m in matches if m["action"] == "drop")
    n_sb_rename = sum(1 for m in matches if m["action"] == "rename")

    parts = []
    if n_sb_drop:
        parts.append(f"[red]sbref-DROP {n_sb_drop}[/]")
    if n_sb_rename:
        parts.append(f"[yellow]sbref-RENAME {n_sb_rename}[/]")
    flag = "  " + "  ".join(parts) if parts else " [green]OK[/]"

    console.print(f"\n[bold cyan]sub-{sub}  ses-{ses}[/]{flag}")

    t = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 1))
    t.add_column("acq_time", justify="right", style="cyan")
    t.add_column("type", justify="center", style="dim")
    t.add_column("name")
    t.add_column("action", justify="center")
    t.add_column("Δs", justify="right")
    t.add_column("note", style="dim")

    for mod, _, acq_time, name, ann in all_rows:
        if ann is None:
            t.add_row(_fmt_time(acq_time) or "[dim]—[/]", mod, name, "", "", "")
            continue

        action = ann["action"]
        color = _ACTION_STYLE.get(action, "white")
        delta_str = str(int(ann["delta_sec"])) if ann["delta_sec"] is not None else "—"
        new_name = ann.get("new_name") or ""
        note_str = ann["func"]["name"] if ann.get("func") else "[dim]none[/dim]"
        action_str = f"[{color}]{action.upper()}[/]"
        name_str = (
            f"[yellow]{name}[/] → [green]{new_name}[/]"
            if action == "rename"
            else f"[{color}]{name}[/]"
        )
        t.add_row(
            _fmt_time(acq_time) or "[dim]—[/]",
            mod,
            name_str,
            action_str,
            delta_str,
            note_str,
        )

    console.print(t)


def _print_summary_table(session_results: list[dict]) -> None:
    t = Table(
        title="sbref drop/rename summary",
        show_header=True,
        header_style="bold magenta",
        box=None,
        padding=(0, 1),
    )
    t.add_column("sub", justify="right", style="cyan")
    t.add_column("ses", justify="right", style="cyan")
    t.add_column("funcs", justify="right")
    t.add_column("sbrefs", justify="right")
    t.add_column("sb-ok", justify="right", style="green")
    t.add_column("sb-rename", justify="right", style="yellow")
    t.add_column("sb-drop", justify="right", style="red")
    t.add_column("status", justify="center")

    for r in session_results:
        has_issues = r["n_drop"] or r["n_rename"]
        if r.get("error"):
            st = f"[red]{r['error']}[/]"
        elif has_issues:
            st = "[yellow]ISSUES[/]"
        else:
            st = "[green]OK[/]"
        t.add_row(
            r["sub"],
            r["ses"],
            str(r["n_funcs"]),
            str(r["n_sbrefs"]),
            str(r["n_ok"]),
            str(r["n_rename"]),
            str(r["n_drop"]),
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
    max_gap: int,
    dry_run: bool,
    verbose: bool,
    outdir: Optional[str],
) -> dict:
    """Run the full drop/rename workflow for one sub/ses.  Returns a result dict."""

    collected = _collect_func_and_sbref(bidsdir, sub, ses)
    funcs = collected["funcs"]
    sbrefs = collected["sbrefs"]

    if not collected["ses_dirs"]:
        console.print(
            f"[yellow]sub-{sub} ses-{ses}: session directory not found — skipped[/]"
        )
        return {
            "sub": sub,
            "ses": ses,
            "n_funcs": 0,
            "n_sbrefs": 0,
            "n_ok": 0,
            "n_rename": 0,
            "n_drop": 0,
            "error": "not found",
        }

    matches = _match_sbrefs(funcs, sbrefs, max_gap)

    n_ok = sum(1 for m in matches if m["action"] == "ok")
    n_rename = sum(1 for m in matches if m["action"] == "rename")
    n_drop = sum(1 for m in matches if m["action"] == "drop")

    has_issues = n_drop or n_rename
    if verbose or has_issues:
        _print_session_table(sub, ses, funcs, sbrefs, matches, max_gap)

    counts = _execute(matches, dry_run=dry_run)

    # Write TSV log if output dir given
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tsv = op.join(outdir, f"sbref_drop_sub-{sub}_ses-{ses}_{ts}.tsv")
        with open(tsv, "w", newline="") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow(["name", "action", "matched_func", "delta_sec", "new_name"])
            for m in matches:
                writer.writerow(
                    [
                        m["sbref"]["name"],
                        m["action"],
                        m["func"]["name"] if m["func"] else "",
                        int(m["delta_sec"]) if m["delta_sec"] is not None else "",
                        m["new_name"] or "",
                    ]
                )

    return {
        "sub": sub,
        "ses": ses,
        "n_funcs": len(funcs),
        "n_sbrefs": len(sbrefs),
        "n_ok": n_ok,
        "n_rename": n_rename,
        "n_drop": n_drop,
        "error": None,
        "counts": counts,
    }


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
        help="sub,ses pair e.g. -s 10,02  (repeatable: -s 10,02 -s 11,03).",
    ),
    subses_file: Optional[Path] = typer.Option(
        None, "--file", "-f", help="TSV/CSV with columns sub, ses."
    ),
    max_gap: int = typer.Option(
        DEFAULT_MAX_GAP,
        "--max-gap",
        help="Max time gap (s) between sbref and func to count as a match.",
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--execute",
        help="Print plan only (default) or execute renames/drops.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed acq-time table for every session (not just problem ones).",
    ),
    outdir: Optional[Path] = typer.Option(
        None, "--outdir", "-o", help="Write per-session TSV logs to this directory."
    ),
):
    """
    Check sbref ↔ bold/magnitude pairing by AcquisitionTime and drop/rename
    sbrefs that are orphaned or mismatched.

    Default: dry-run — prints the plan without touching any files.
    Use --execute to apply changes.

    Examples:
        python 01_drop_sbref.py -b /BIDS -s 10,02
        python 01_drop_sbref.py -b /BIDS -s 10,02 -s 11,03
        python 01_drop_sbref.py -b /BIDS -f subseslist.tsv --execute
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
    console.print(
        f"\n[bold]01_drop_sbref[/]  bidsdir={bidsdir}  "
        f"max-gap={max_gap}s  mode={mode_str}\n"
    )

    results = []
    for s, e in pairs:
        r = _process_session(
            str(bidsdir),
            s,
            e,
            max_gap=max_gap,
            dry_run=dry_run,
            verbose=verbose,
            outdir=str(outdir) if outdir else None,
        )
        results.append(r)

    console.print()
    _print_summary_table(results)

    if dry_run:
        console.print("\n[yellow]Dry-run complete.  Use --execute to apply changes.[/]")
    else:

        def _count(key: str) -> int:
            return sum(r["counts"].get(key, 0) for r in results if r.get("counts"))

        console.print(
            f"\n[green]Done.[/]"
            f"  sbref-renamed={_count('renamed')}  sbref-dropped={_count('dropped')}"
            f"  errors={_count('errors')}"
        )


if __name__ == "__main__":
    app()
