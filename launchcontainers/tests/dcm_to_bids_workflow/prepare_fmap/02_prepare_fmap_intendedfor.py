"""
prepare_fmap_intendedfor.py
---------------------------
For each sub/ses, populate the ``IntendedFor`` field in all fmap JSON sidecars.

How it works
------------
1. Read AcquisitionTime from every func JSON sidecar (bold + sbref).
2. Read AcquisitionTime from every fmap JSON sidecar (AP and PA epi).
3. Sort fmaps by AcquisitionTime.  For each fmap, collect every func file
   whose AcquisitionTime falls within ``--lookback`` minutes BEFORE that fmap
   or AFTER that fmap and BEFORE the next fmap (default lookback: 10 min).
4. Check if IntendedFor already exists and is correct (GOOD / WRONG / MISSING).
5. Remove any fmap run with no func files after it (delete JSON + nii.gz for
   both AP and PA directions).
6. Renumber the remaining fmaps 1, 2, ... in acquisition-time order,
   renaming both JSON and nii.gz files for each direction.
7. Write IntendedFor only to JSONs that are WRONG or MISSING.

Output layers
-------------
default  — one-liner per session + final summary table
-v       — adds per-session plan table + IntendedFor write actions
--debug  — adds timing check + file-level rename/delete details

IntendedFor paths are written relative to the subject directory, e.g.::

    ses-10/func/sub-06_ses-10_task-retfixRW_run-01_bold.nii.gz

Usage
-----
    python prepare_fmap_intendedfor.py --bidsdir /path/BIDS -s 06,10
    python prepare_fmap_intendedfor.py --bidsdir /path/BIDS -f subseslist.tsv -v
    python prepare_fmap_intendedfor.py --bidsdir /path/BIDS -f subseslist.tsv --execute
    python prepare_fmap_intendedfor.py --bidsdir /path/BIDS -f subseslist.tsv --execute -w 8
    python prepare_fmap_intendedfor.py --bidsdir /path/BIDS -s 06,10 --lookback 10
"""

from __future__ import annotations

import glob
import json
import os
import os.path as op
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from launchcontainers.utils import (
    atomic_rename_pairs,
    hms_to_sec,
    parse_hms,
    parse_subses_list,
    read_json_acqtime,
)

console = Console()
app = typer.Typer(pretty_exceptions_show_locals=False)

SESSION_GAP_MAX_SEC = (
    30 * 60
)  # 30 min — consecutive scan gap that indicates a session break
FMAP_FUNC_MAX_GAP_SEC = (
    8 * 60
)  # 8   min  — fmap must be followed by func within this window
FMAP_LOOKBACK_SEC = (
    10 * 60
)  # 10 min — how far before a fmap to look back for func files

# ---------------------------------------------------------------------------
# Verbosity gate — set once by CLI, read everywhere
# ---------------------------------------------------------------------------

_verbose: bool = False
_debug: bool = False


def _vprint(msg: str, level: int = 1) -> None:
    """
    Print msg only when verbosity is sufficient.

    level 0 — always (errors, per-session one-liner, summary)
    level 1 — verbose (-v): per-session plan table, IntendedFor write actions
    level 2 — debug (--debug): timing check, file-level rename/delete details
    """
    if level == 0:
        console.print(msg)
    elif level == 1 and _verbose:
        console.print(msg)
    elif level == 2 and _debug:
        console.print(msg)


# ---------------------------------------------------------------------------
# Core: read func and fmap files
# ---------------------------------------------------------------------------


def _read_func_files(bidsdir: str, sub: str, ses: str) -> list[dict]:
    """
    Return one entry per func JSON sidecar (bold + sbref), sorted by AcquisitionTime.
    """
    func_dir = op.join(bidsdir, f"sub-{sub}", f"ses-{ses}", "func")
    if not op.isdir(func_dir):
        return []

    rows = []
    for jf in sorted(
        glob.glob(op.join(func_dir, f"sub-{sub}_ses-{ses}_task-*_run-*_*.json"))
    ):
        basename = op.basename(jf)
        m = re.search(r"_(bold|sbref)\.json$", basename)
        if not m:
            continue
        nii_name = basename.replace(".json", ".nii.gz")
        acq_time = parse_hms(read_json_acqtime(jf))
        rows.append(
            {
                "basename": basename,
                "suffix": m.group(1),
                "acq_time": acq_time,
                "acq_sec": hms_to_sec(acq_time),
                "intended_for_path": f"ses-{ses}/func/{nii_name}",
            }
        )

    rows.sort(key=lambda r: r["acq_sec"])
    return rows


def _read_fmap_files(bidsdir: str, sub: str, ses: str) -> list[dict]:
    """
    Return fmap entries grouped by run number, sorted by AcquisitionTime.
    Tolerates optional acq-* entity in filename.
    """
    fmap_dir = op.join(bidsdir, f"sub-{sub}", f"ses-{ses}", "fmap")
    if not op.isdir(fmap_dir):
        return []

    by_run: dict[str, list[str]] = {}
    for jf in sorted(glob.glob(op.join(fmap_dir, f"sub-{sub}_ses-{ses}_*_epi.json"))):
        m = re.search(r"_run-(\d+)_epi\.json$", op.basename(jf))
        if not m:
            continue
        by_run.setdefault(str(int(m.group(1))), []).append(jf)  # 1-digit fmap run IDs

    rows = []
    for run, json_paths in by_run.items():
        acq_time = ""
        for jf in json_paths:
            raw = read_json_acqtime(jf)
            if raw:
                acq_time = parse_hms(raw)
                break
        rows.append(
            {
                "run": run,
                "acq_time": acq_time,
                "acq_sec": hms_to_sec(acq_time),
                "json_paths": json_paths,
            }
        )

    rows.sort(key=lambda r: r["acq_sec"])
    return rows


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _check_session_gap(func_files: list[dict], fmap_runs: list[dict]) -> list[str]:
    """
    For each fmap, find the first func that comes after it and check if the
    gap is > SESSION_GAP_MAX_SEC.  Func-to-func gaps are not checked.

    Returns a list of warning strings (empty if all OK).
    """
    warnings = []
    for fm in fmap_runs:
        after = [f for f in func_files if f["acq_sec"] > fm["acq_sec"]]
        if not after:
            continue
        first_func = min(after, key=lambda f: f["acq_sec"])
        gap = first_func["acq_sec"] - fm["acq_sec"]
        if gap > SESSION_GAP_MAX_SEC:
            warnings.append(
                f"  [red]⚠ SESSION GAP[/]  fmap run-{fm['run']} ({fm['acq_time']})"
                f"  →  {first_func['basename']} ({first_func['acq_time']})"
                f"  gap={gap / 60:.1f} min (>{SESSION_GAP_MAX_SEC // 60} min)"
            )
    return warnings


def _check_fmap_timing(func_files: list[dict], fmap_runs: list[dict]) -> None:
    """Debug-level: print nearest func file and gap for each fmap."""
    if not func_files or not fmap_runs:
        return
    for fm in fmap_runs:
        candidates = [f for f in func_files if f["acq_sec"] > fm["acq_sec"]]
        if not candidates:
            _vprint(
                f"    [yellow]~[/]  fmap run-{fm['run']}  acq={fm['acq_time']}"
                f"  →  no func after — will be removed",
                level=2,
            )
            continue
        nearest = min(candidates, key=lambda f: f["acq_sec"] - fm["acq_sec"])
        gap = nearest["acq_sec"] - fm["acq_sec"]
        colour = "bold red" if gap > FMAP_FUNC_MAX_GAP_SEC else "green"
        flag = f"  (>{FMAP_FUNC_MAX_GAP_SEC}s)" if gap > FMAP_FUNC_MAX_GAP_SEC else ""
        _vprint(
            f"    [{colour}]{'⚠' if gap > FMAP_FUNC_MAX_GAP_SEC else '✓'}[/]"
            f"  fmap run-{fm['run']}  acq={fm['acq_time']}"
            f"  →  {nearest['basename']}  acq={nearest['acq_time']}"
            f"  gap={gap:.0f}s{flag}",
            level=2,
        )


def _assign_intendedfor(
    func_files: list[dict], fmap_runs: list[dict], lookback_sec: int = FMAP_LOOKBACK_SEC
) -> list[dict]:
    for i, fm in enumerate(fmap_runs):
        # Look back up to lookback_sec, but not before the previous fmap (no overlap).
        prev_sec = fmap_runs[i - 1]["acq_sec"] if i > 0 else 0.0
        window_start = max(fm["acq_sec"] - lookback_sec, prev_sec)
        next_sec = (
            fmap_runs[i + 1]["acq_sec"] if i + 1 < len(fmap_runs) else float("inf")
        )
        funcs_in_window = [
            f for f in func_files if window_start <= f["acq_sec"] < next_sec
        ]
        fm["intended_for"] = [f["intended_for_path"] for f in funcs_in_window]
        # Store the acq_sec of the first func in the window for session-gap check
        fm["first_func_in_window_sec"] = (
            min(f["acq_sec"] for f in funcs_in_window) if funcs_in_window else None
        )
    return fmap_runs


def _prune_session_gap_fmaps(
    kept: list[dict], removed: list[dict]
) -> tuple[list[dict], list[dict]]:
    """
    Pass 3 — drop fmaps whose gap to the first func in their window exceeds
    SESSION_GAP_MAX_SEC.  This handles the case where a morning fmap is
    incorrectly assigned to cover an evening/part2 func block after a session
    break.  The dropped fmap's funcs then become NOT COVERED → ERROR.
    """
    still_kept: list[dict] = []
    for fm in kept:
        first_sec = fm.get("first_func_in_window_sec")
        if first_sec is not None:
            gap = first_sec - fm["acq_sec"]
            if gap > SESSION_GAP_MAX_SEC:
                fm["drop_reason"] = (
                    f"session gap to first func in window: {gap / 60:.1f} min "
                    f"(>{SESSION_GAP_MAX_SEC // 60} min)"
                )
                removed.append(fm)
                continue
        still_kept.append(fm)
    return still_kept, removed


def _prune_fmaps(fmap_runs: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Remove invalid fmap runs in two passes:

    Pass 1 — consecutive fmaps with no func between them:
        If fmap[i] and fmap[i+1] are back-to-back (no func in the window),
        drop fmap[i] (keep the later one, which is closer to the actual func run).

    Pass 2 — fmaps with no func at all after them (tail orphans):
        Any fmap whose intended_for is empty after pass 1 is also removed.

    Returns (kept_runs, removed_runs).
    """
    removed: list[dict] = []

    # Pass 1: drop the earlier of any consecutive fmap pair
    to_drop: set[int] = set()
    for i in range(len(fmap_runs) - 1):
        if not fmap_runs[i]["intended_for"]:
            # No func between fmap[i] and fmap[i+1] → drop fmap[i]
            to_drop.add(i)
            fmap_runs[i]["drop_reason"] = "consecutive (no func before next fmap)"

    after_pass1 = []
    for i, fm in enumerate(fmap_runs):
        if i in to_drop:
            removed.append(fm)
        else:
            after_pass1.append(fm)

    # Pass 2: drop remaining fmaps with empty intended_for (tail orphans)
    kept: list[dict] = []
    for fm in after_pass1:
        if fm["intended_for"]:
            kept.append(fm)
        else:
            fm.setdefault("drop_reason", "no func after")
            removed.append(fm)

    return kept, removed


def _check_first_func_coverage(
    fmap_runs: list[dict], func_files: list[dict], lookback_sec: int = FMAP_LOOKBACK_SEC
) -> list[str]:
    """
    Two checks focused on the FIRST functional run:

    Check A — first func outside lookback window of all fmaps (ERROR):
        If the earliest func is more than lookback_sec before the nearest fmap,
        it is not covered → ERROR (blocks writing IntendedFor).
        If it is within lookback_sec before the nearest fmap → OK (covered by
        the lookback window).

    Check B — fmap→first-func gap > FMAP_FUNC_MAX_GAP_SEC (WARNING):
        Only applies when first func comes AFTER its nearest fmap.

    Returns list of warning/error strings (empty if all OK).
    """
    if not func_files or not fmap_runs:
        return []

    first_func = min(func_files, key=lambda f: f["acq_sec"])
    first_fmap = min(fmap_runs, key=lambda fm: fm["acq_sec"])
    issues = []

    # First func comes before the first fmap.
    if first_func["acq_sec"] < first_fmap["acq_sec"]:
        gap = first_fmap["acq_sec"] - first_func["acq_sec"]
        if gap > lookback_sec:
            issues.append(
                f"  [bold red]✗ FIRST FUNC OUTSIDE LOOKBACK[/]"
                f"  {first_func['basename']} ({first_func['acq_time']})"
                f"  is {gap / 60:.1f} min before first fmap run-{first_fmap['run']}"
                f"  (>{lookback_sec // 60} min lookback)"
                f"  — IntendedFor cannot be assigned for the first run"
            )
        # else: within lookback window → covered, no issue
        return issues

    # First func comes after the first fmap — check gap.
    fmaps_before = [fm for fm in fmap_runs if fm["acq_sec"] < first_func["acq_sec"]]
    nearest_before = max(fmaps_before, key=lambda fm: fm["acq_sec"])
    gap = first_func["acq_sec"] - nearest_before["acq_sec"]
    if gap > FMAP_FUNC_MAX_GAP_SEC:
        issues.append(
            f"  [yellow]⚠ FMAP→FIRST-FUNC GAP[/]"
            f"  fmap run-{nearest_before['run']} ({nearest_before['acq_time']})"
            f"  →  {first_func['basename']} ({first_func['acq_time']})"
            f"  gap={gap / 60:.1f} min (>{FMAP_FUNC_MAX_GAP_SEC // 60} min)"
        )

    return issues


def _warn_fmap_func_gap(kept_runs: list[dict], func_files: list[dict]) -> list[str]:
    """
    For each kept fmap, find the nearest func run after it and warn if gap > 3 min.
    Returns list of warning strings.
    """
    warnings = []
    for fm in kept_runs:
        candidates = [f for f in func_files if f["acq_sec"] > fm["acq_sec"]]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda f: f["acq_sec"] - fm["acq_sec"])
        gap = nearest["acq_sec"] - fm["acq_sec"]
        if gap > FMAP_FUNC_MAX_GAP_SEC:
            warnings.append(
                f"  [yellow]⚠ FMAP→FUNC GAP[/]  fmap run-{fm['run']} ({fm['acq_time']})"
                f"  →  {nearest['basename']} ({nearest['acq_time']})"
                f"  gap={gap / 60:.1f} min (>{FMAP_FUNC_MAX_GAP_SEC // 60} min)"
            )
    return warnings


def _check_existing_intendedfor(kept_runs: list[dict]) -> tuple[str, dict]:
    """
    Compare computed IntendedFor against what is on disk for each kept JSON.

    Returns (session_status, per_json) where session_status is
    "GOOD" | "WRONG" | "MISSING" and per_json maps json_path → status.
    """
    per_json: dict[str, str] = {}
    for fm in kept_runs:
        computed = sorted(fm["intended_for"])
        for jf in fm["json_paths"]:
            try:
                with open(jf) as fh:
                    existing = json.load(fh).get("IntendedFor", None)
                if existing is None:
                    per_json[jf] = "MISSING"
                elif sorted(existing) == computed:
                    per_json[jf] = "GOOD"
                else:
                    per_json[jf] = "WRONG"
            except Exception:
                per_json[jf] = "MISSING"

    statuses = set(per_json.values())
    if "MISSING" in statuses:
        return "MISSING", per_json
    if "WRONG" in statuses:
        return "WRONG", per_json
    return "GOOD", per_json


def _check_func_coverage(kept_runs: list[dict], func_files: list[dict]) -> list[str]:
    """
    Verify that every func file (bold + sbref) appears in at least one fmap's
    computed IntendedFor list.  Returns a list of error strings for any func
    that is not covered.
    """
    covered: set[str] = set()
    for fm in kept_runs:
        covered.update(fm["intended_for"])

    issues = []
    for f in func_files:
        if f["intended_for_path"] not in covered:
            issues.append(
                f"  [bold red]✗ NOT COVERED[/]  {f['basename']}  ({f['acq_time']})"
                f"  — not in any fmap IntendedFor"
            )
    return issues


def _fmap_all_paths(fm: dict) -> list[tuple[str, str]]:
    return [(jf, re.sub(r"\.json$", ".nii.gz", jf)) for jf in fm["json_paths"]]


def _plan_renumber(kept_runs: list[dict]) -> list[tuple[str, str]]:
    ops: list[tuple[str, str]] = []
    for new_idx, fm in enumerate(kept_runs, start=1):
        new_run = str(new_idx)
        for jf, nii in _fmap_all_paths(fm):
            jf_new = re.sub(r"_run-\d+_epi\.json$", f"_run-{new_run}_epi.json", jf)
            nii_new = re.sub(
                r"_run-\d+_epi\.nii\.gz$", f"_run-{new_run}_epi.nii.gz", nii
            )
            if jf != jf_new:
                ops.append((jf, jf_new))
            if nii != nii_new:
                ops.append((nii, nii_new))
    return ops


def _update_json_paths_after_rename(kept_runs, rename_ops):
    rename_map = {old: new for old, new in rename_ops}
    for fm in kept_runs:
        fm["json_paths"] = [rename_map.get(jf, jf) for jf in fm["json_paths"]]
    return kept_runs


# ---------------------------------------------------------------------------
# Per-session: verbose tables
# ---------------------------------------------------------------------------


def _print_timeline_table(
    sub: str,
    ses: str,
    func_files: list[dict],
    fmap_runs: list[dict],
) -> None:
    """Verbose-level: all fmap and func acquisitions sorted by time."""
    rows = []
    for fm in fmap_runs:
        dirs = sorted(
            set(
                re.search(r"dir-(\w+)", op.basename(p)).group(1)
                for p in fm["json_paths"]
                if re.search(r"dir-(\w+)", op.basename(p))
            )
        )
        rows.append(
            {
                "acq_sec": fm["acq_sec"],
                "acq_time": fm["acq_time"],
                "type": "fmap",
                "label": f"run-{fm['run']}  ({'+'.join(dirs)})",
            }
        )
    for f in func_files:
        rows.append(
            {
                "acq_sec": f["acq_sec"],
                "acq_time": f["acq_time"],
                "type": f["suffix"],  # bold | sbref
                "label": f["basename"].replace(".json", ""),
            }
        )

    rows.sort(key=lambda r: r["acq_sec"])

    t = Table(
        show_header=True,
        header_style="bold magenta",
        box=None,
        title=f"sub-{sub}  ses-{ses}  acquisition timeline",
    )
    t.add_column("acq_time", justify="right")
    t.add_column("type", justify="center")
    t.add_column("file")

    for r in rows:
        if r["type"] == "fmap":
            colour = "cyan"
        elif r["type"] == "sbref":
            colour = "dim"
        else:
            colour = ""
        t.add_row(
            r["acq_time"],
            f"[{colour}]{r['type']}[/]" if colour else r["type"],
            f"[{colour}]{r['label']}[/]" if colour else r["label"],
        )

    _vprint("", level=1)
    if _verbose:
        console.print(t)


def _print_plan_table(
    sub: str,
    ses: str,
    fmap_runs: list[dict],
    kept_runs: list[dict],
    per_json_status: dict,
) -> None:
    """Verbose-level: fmap plan table with actions and IntendedFor status."""
    kept_new_ids = {id(fm): str(i) for i, fm in enumerate(kept_runs, start=1)}

    t = Table(
        show_header=True,
        header_style="bold magenta",
        box=None,
        title=f"sub-{sub}  ses-{ses}  fmap plan",
    )
    t.add_column("old_run", style="dim")
    t.add_column("new_run")
    t.add_column("action", justify="center")
    t.add_column("func_files", justify="right")
    t.add_column("IntendedFor", justify="center")

    for fm in fmap_runs:
        old_run = fm["run"]
        if id(fm) in kept_new_ids:
            new_run = kept_new_ids[id(fm)]
            action = (
                "[green]keep[/]"
                if old_run == new_run
                else f"[yellow]rename → run-{new_run}[/]"
            )
            n_func = str(len(fm["intended_for"]))
            run_s = {per_json_status.get(jf, "MISSING") for jf in fm["json_paths"]}
            if_cell = (
                "[red]MISSING[/]"
                if "MISSING" in run_s
                else "[yellow]WRONG[/]"
                if "WRONG" in run_s
                else "[green]GOOD[/]"
            )
        else:
            new_run = "—"
            action = "[red]delete (no func)[/]"
            n_func = "0"
            if_cell = "[dim]n/a[/]"
        t.add_row(
            f"run-{old_run}",
            f"run-{new_run}" if new_run != "—" else "—",
            action,
            n_func,
            if_cell,
        )

    _vprint("", level=1)
    console.print(t) if _verbose else None  # table needs direct print


# ---------------------------------------------------------------------------
# Lab-note: expected fmap count (lightweight, fmap only)
# ---------------------------------------------------------------------------


def _safe_zfill(v) -> str:
    s = str(v).strip()
    try:
        return str(int(float(s))).zfill(2)
    except (ValueError, TypeError):
        return s


def _load_labnote_fmap_counts(xlsx_path: str) -> dict[tuple[str, str], int]:
    """
    Read the lab note and return {(sub, ses): expected_fmap_count}.
    Only counts protocol_name rows matching fmap_NN after normalization.
    Skips rows with quality_mark == fail/failed/f.
    """
    try:
        import pandas as pd
    except ImportError:
        console.print("[red]pandas not installed — cannot read lab note.[/]")
        raise typer.Exit(1)

    _failed_re = re.compile(r"^(fail(ed)?|f)$", re.IGNORECASE)

    xls = pd.ExcelFile(xlsx_path)
    sheets = [s for s in xls.sheet_names if s.startswith("sub-")]
    parts = []
    for sheet in sheets:
        df = pd.read_excel(xls, sheet_name=sheet, header=0)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        if not {"sub", "ses", "protocol_name"}.issubset(df.columns):
            continue
        df = df.copy()
        df[["sub", "ses"]] = df[["sub", "ses"]].replace("", pd.NA).ffill()
        df = df.dropna(subset=["sub", "ses"])
        df["sub"] = df["sub"].apply(_safe_zfill)
        df["ses"] = df["ses"].apply(_safe_zfill)
        parts.append(df)

    if not parts:
        return {}

    big = pd.concat(parts, ignore_index=True)
    counts: dict[tuple[str, str], int] = {}
    for _, row in big.iterrows():
        proto = str(row.get("protocol_name", "")).strip()
        if not re.match(r"fmap_\d+$", proto, re.IGNORECASE):
            continue
        qual = str(row.get("quality_mark", "")).strip().lower()
        if qual and qual != "nan" and _failed_re.match(qual):
            continue
        key = (str(row["sub"]), str(row["ses"]))
        counts[key] = counts.get(key, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Process one session
# ---------------------------------------------------------------------------


def process_session(
    bidsdir: str,
    sub: str,
    ses: str,
    dry_run: bool = True,
    ln_fmap_counts: dict | None = None,
    lookback_sec: int = FMAP_LOOKBACK_SEC,
) -> dict:
    """
    Returns {sub, ses, n_fmaps_orig, n_removed, n_written, intendedfor_status, warnings}.
    """
    func_files = _read_func_files(bidsdir, sub, ses)
    fmap_runs = _read_fmap_files(bidsdir, sub, ses)

    if not fmap_runs:
        _vprint(f"[dim]  sub-{sub}  ses-{ses}  SKIP  (no fmap dir)[/]", level=0)
        return {
            "sub": sub,
            "ses": ses,
            "n_fmaps_orig": 0,
            "n_removed": 0,
            "n_written": 0,
            "intendedfor_status": "SKIP",
            "warnings": [],
        }

    if not func_files:
        _vprint(f"[dim]  sub-{sub}  ses-{ses}  SKIP  (no func dir)[/]", level=0)
        return {
            "sub": sub,
            "ses": ses,
            "n_fmaps_orig": len(fmap_runs),
            "n_removed": 0,
            "n_written": 0,
            "intendedfor_status": "SKIP",
            "warnings": [],
        }

    all_warnings: list[str] = []

    # --- 1. Session gap check (> 45 min between any two consecutive scans) ---
    gap_warns = _check_session_gap(func_files, fmap_runs)
    if gap_warns:
        all_warnings.extend(gap_warns)

    # --- 2. Lab-note fmap count vs BIDS fmap count ---
    if ln_fmap_counts is not None:
        expected = ln_fmap_counts.get((sub, ses))
        if expected is not None and expected != len(fmap_runs):
            w = (
                f"  [yellow]⚠ FMAP COUNT[/]  sub-{sub} ses-{ses}  "
                f"labnote={expected}  bids={len(fmap_runs)}"
            )
            all_warnings.append(w)

    # --- 3. debug: per-fmap timing detail ---
    _vprint(f"\n  [bold cyan]sub-{sub}  ses-{ses}[/]  timing check", level=2)
    _check_fmap_timing(func_files, fmap_runs)

    # --- 4. Check first-func coverage (ERROR if first func outside lookback window) ---
    first_func_issues = _check_first_func_coverage(fmap_runs, func_files, lookback_sec)
    has_coverage_error = any(
        "NO FMAP BEFORE FIRST FUNC" in w for w in first_func_issues
    )
    if first_func_issues:
        all_warnings.extend(first_func_issues)
        if has_coverage_error and not dry_run:
            _vprint(
                f"  [bold red]BLOCKED[/]  sub-{sub} ses-{ses}: "
                f"cannot write IntendedFor — no fmap before first func.",
                level=0,
            )
            return {
                "sub": sub,
                "ses": ses,
                "n_fmaps_orig": len(fmap_runs),
                "n_removed": 0,
                "n_written": 0,
                "intendedfor_status": "ERROR",
                "warnings": all_warnings,
            }

    # --- 5. Assign IntendedFor windows, then prune ---
    fmap_runs = _assign_intendedfor(func_files, fmap_runs, lookback_sec)
    kept_runs, removed_runs = _prune_fmaps(fmap_runs)
    kept_runs, removed_runs = _prune_session_gap_fmaps(kept_runs, removed_runs)

    # --- 6. Warn if any kept fmap → func gap > 3 min ---
    gap_func_warns = _warn_fmap_func_gap(kept_runs, func_files)
    if gap_func_warns:
        all_warnings.extend(gap_func_warns)

    session_status, per_json_status = _check_existing_intendedfor(kept_runs)
    func_coverage_issues = _check_func_coverage(kept_runs, func_files)
    if has_coverage_error or func_coverage_issues:
        session_status = "ERROR"
    rename_ops = _plan_renumber(kept_runs)

    # --- verbose: timeline + plan table ---
    if _verbose:
        _print_timeline_table(sub, ses, func_files, fmap_runs)
        _print_plan_table(sub, ses, fmap_runs, kept_runs, per_json_status)

    # --- debug: deletions ---
    if removed_runs:
        _vprint(
            f"\n  [yellow]Deleting {len(removed_runs)} invalid fmap(s):[/]", level=2
        )
        for fm in removed_runs:
            reason = fm.get("drop_reason", "no func after")
            _vprint(f"    [dim]reason: {reason}[/]", level=2)
        for fm in removed_runs:
            for jf, nii in _fmap_all_paths(fm):
                if dry_run:
                    _vprint(f"    [dim][DRY][/] delete  {op.basename(jf)}", level=2)
                    _vprint(f"    [dim][DRY][/] delete  {op.basename(nii)}", level=2)
                else:
                    for path in (jf, nii):
                        if op.exists(path):
                            os.remove(path)
                            _vprint(
                                f"    [red]✗[/] deleted  {op.basename(path)}", level=2
                            )
                        else:
                            _vprint(
                                f"    [yellow][WARN][/] not found: {op.basename(path)}",
                                level=0,
                            )

    # --- debug: renames ---
    if rename_ops:
        _vprint(f"\n  [yellow]Renaming {len(kept_runs)} fmap run(s):[/]", level=2)
        for old, new in rename_ops:
            _vprint(
                f"    {'[dim][DRY][/]' if dry_run else '[green][RENAME][/]'}"
                f" {op.basename(old)}  →  {op.basename(new)}",
                level=2,
            )
        if not dry_run:
            try:
                atomic_rename_pairs(
                    [(Path(old), Path(new)) for old, new in rename_ops],
                    dry_run=False,
                )
            except RuntimeError as exc:
                _vprint(f"  [red][ERROR][/] {exc}", level=0)

    if not dry_run and rename_ops:
        kept_runs = _update_json_paths_after_rename(kept_runs, rename_ops)

    # --- verbose: IntendedFor write actions ---
    n_written = 0
    for fm in kept_runs:
        intended_for = fm["intended_for"]
        for jf in fm["json_paths"]:
            jf_status = per_json_status.get(jf, "MISSING")
            if jf_status == "GOOD":
                _vprint(f"    [green]✓[/] already correct — {op.basename(jf)}", level=1)
                continue
            label = f"[yellow]{jf_status}[/]"
            if dry_run:
                _vprint(
                    f"\n    [dim][DRY][/] {label}  would write IntendedFor → "
                    f"[bold]{op.basename(jf)}[/]  ({len(intended_for)} entries)",
                    level=1,
                )
                for p in intended_for:
                    _vprint(f"        {p}", level=1)
            else:
                try:
                    with open(jf) as fh:
                        data = json.load(fh)
                    data["IntendedFor"] = intended_for
                    with open(jf, "w") as fh:
                        json.dump(data, fh, indent=2)
                    _vprint(
                        f"    [green]✓[/] IntendedFor written ({jf_status} → fixed)"
                        f" → {op.basename(jf)}  ({len(intended_for)} entries)",
                        level=1,
                    )
                    n_written += 1
                except Exception as exc:
                    _vprint(f"    [red]ERROR[/] {op.basename(jf)}: {exc}", level=0)

    # --- 7. Report uncovered func runs ---
    if func_coverage_issues:
        all_warnings.extend(func_coverage_issues)

    # --- streamline: one-liner per session (always printed first) ---
    sc = {
        "GOOD": "green",
        "WRONG": "yellow",
        "MISSING": "red",
        "ERROR": "bold red",
    }.get(session_status, "dim")
    rem = f"  removed={len(removed_runs)}" if removed_runs else ""
    warn = f"  [yellow]⚠ {len(all_warnings)} warning(s)[/]" if all_warnings else ""
    _vprint(
        f"  sub-{sub}  ses-{ses}  [{sc}]{session_status}[/]"
        f"  fmaps={len(fmap_runs)}{rem}{warn}",
        level=0,
    )

    # --- detail: print all warnings/issues after the one-liner ---
    for w in all_warnings:
        _vprint(w, level=0)

    return {
        "sub": sub,
        "ses": ses,
        "n_fmaps_orig": len(fmap_runs),
        "n_removed": len(removed_runs),
        "n_written": n_written,
        "intendedfor_status": session_status,
        "warnings": all_warnings,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    bidsdir: Path = typer.Option(..., "--bidsdir", "-b", help="BIDS root directory."),
    subses: Optional[str] = typer.Option(
        None,
        "--subses",
        "-s",
        help="Single sub,ses pair e.g. 06,10",
    ),
    subses_file: Optional[Path] = typer.Option(
        None,
        "--file",
        "-f",
        help="Subseslist TSV/CSV with sub and ses columns.",
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Write IntendedFor to JSON files. Default: dry-run.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Per-session plan table and IntendedFor write actions.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Add timing check and file-level rename/delete details.",
    ),
    workers: Optional[int] = typer.Option(
        None,
        "--workers",
        "-w",
        help="Number of parallel workers. Omit for serial execution.",
    ),
    labnote: Optional[Path] = typer.Option(
        None,
        "--labnote",
        "-l",
        help="Lab-note Excel (e.g. VOTCLOC_subses_list.xlsx). "
        "When supplied, compares expected fmap count from lab note against BIDS.",
    ),
    lookback: int = typer.Option(
        FMAP_LOOKBACK_SEC // 60,
        "--lookback",
        help="Minutes before a fmap to include func files in IntendedFor (default: 10).",
    ),
):
    """
    Populate IntendedFor in fmap JSONs based on AcquisitionTime ordering.

    Output layers
    -------------
    (default)  one-liner per session + final summary table
    -v         adds per-session plan table and IntendedFor write actions
    --debug    adds timing check and file-level rename/delete details
    """
    global _verbose, _debug
    _verbose = verbose or debug  # debug implies verbose
    _debug = debug
    lookback_sec = lookback * 60

    # Load lab-note fmap counts once if provided
    ln_fmap_counts: dict | None = None
    if labnote is not None:
        ln_fmap_counts = _load_labnote_fmap_counts(str(labnote))

    if subses_file is not None:
        pairs = parse_subses_list(subses_file)
    elif subses is not None:
        parts = [p.strip().zfill(2) for p in subses.split(",")]
        if len(parts) != 2:
            console.print("[red]--subses must be sub,ses e.g. 06,10[/]")
            raise typer.Exit(1)
        pairs = [(parts[0], parts[1])]
    else:
        console.print("[red]Provide --subses or --file.[/]")
        raise typer.Exit(1)

    dry_run = not execute
    if dry_run:
        console.print("\n[bold yellow]─── DRY-RUN ─── (no files changed)[/bold yellow]")
    else:
        console.print(
            "\n[bold green]─── EXECUTE ─── (writing IntendedFor)[/bold green]"
        )

    results: list[dict] = []
    if workers:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    process_session,
                    str(bidsdir),
                    sub,
                    ses,
                    dry_run,
                    ln_fmap_counts,
                    lookback_sec,
                ): (sub, ses)
                for sub, ses in pairs
            }
            for fut in as_completed(futures):
                sub, ses = futures[fut]
                try:
                    results.append(fut.result())
                except Exception as exc:
                    _vprint(f"  [red]ERROR[/]  sub-{sub}  ses-{ses}: {exc}", level=0)
        results.sort(key=lambda r: (r["sub"], r["ses"]))
    else:
        for sub, ses in pairs:
            results.append(
                process_session(
                    str(bidsdir),
                    sub,
                    ses,
                    dry_run=dry_run,
                    ln_fmap_counts=ln_fmap_counts,
                    lookback_sec=lookback_sec,
                )
            )

    # ── Summary table ─────────────────────────────────────────────────────────
    good = [
        (r["sub"], r["ses"]) for r in results if r.get("intendedfor_status") == "GOOD"
    ]
    wrong = [
        (r["sub"], r["ses"]) for r in results if r.get("intendedfor_status") == "WRONG"
    ]
    missing = [
        (r["sub"], r["ses"])
        for r in results
        if r.get("intendedfor_status") == "MISSING"
    ]
    error = [
        (r["sub"], r["ses"]) for r in results if r.get("intendedfor_status") == "ERROR"
    ]
    skipped = [
        (r["sub"], r["ses"]) for r in results if r.get("intendedfor_status") == "SKIP"
    ]
    total_removed = sum(r.get("n_removed", 0) for r in results)

    console.print(f"\n{'─' * 60}")
    console.print(f"[bold]Summary[/]  {len(results)} session(s) processed")
    console.print(f"{'─' * 60}")

    t = Table(show_header=True, header_style="bold magenta", box=None)
    t.add_column("status", justify="center")
    t.add_column("count", justify="right")
    t.add_column("sessions")
    t.add_row(
        "[green]GOOD[/]",
        str(len(good)),
        "  ".join(f"sub-{s} ses-{e}" for s, e in good) or "—",
    )
    t.add_row(
        "[yellow]WRONG[/]",
        str(len(wrong)),
        "  ".join(f"sub-{s} ses-{e}" for s, e in wrong) or "—",
    )
    t.add_row(
        "[red]MISSING[/]",
        str(len(missing)),
        "  ".join(f"sub-{s} ses-{e}" for s, e in missing) or "—",
    )
    if error:
        t.add_row(
            "[bold red]ERROR[/]",
            str(len(error)),
            "  ".join(f"sub-{s} ses-{e}" for s, e in error),
        )
    if skipped:
        t.add_row(
            "[dim]SKIP[/]",
            str(len(skipped)),
            "  ".join(f"sub-{s} ses-{e}" for s, e in skipped),
        )
    console.print(t)

    if total_removed:
        console.print(f"  [yellow]{total_removed} invalid fmap run(s) removed[/]")
    total_warnings = sum(len(r.get("warnings", [])) for r in results)
    if total_warnings:
        console.print(f"  [yellow]{total_warnings} warning(s) across all sessions[/]")

    console.print()
    if dry_run:
        console.print(
            "[dim]Pass [bold]--execute[/bold] to apply the changes above.[/dim]"
        )
    else:
        total_written = sum(r["n_written"] for r in results)
        console.print(f"[green]{total_written} fmap JSON file(s) updated.[/green]")


if __name__ == "__main__":
    app()
