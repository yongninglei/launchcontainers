"""
00_merge_split_ses_reording_runs.py
------------------------------------
Two related tasks in one script:

  1. Merge split BIDS sessions (e.g. ses-02 + ses-02part2) into one.
  2. Fix run-number gaps or non-1 starting indices in a single session.

Both are solved by the same mechanism: sort ALL files per group
(mod / task / suffix) by AcquisitionTime and assign run numbers 1, 2, 3 …
sequentially.  This automatically fills gaps (run-08 → run-10 becomes
run-08 → run-09) and renumbers any task that doesn't start from run-01.

Split-session case
------------------
When a session is split across ses-{ses} + ses-{ses}part2 (…), run indices
in the secondary directories restart from 01.  The script:

  1. Discovers all session directories matching ses-{ses}* by globbing.
  2. Reads AcquisitionTime from every JSON sidecar and builds a unified
     timeline.
  3. Assigns new run numbers via time-sorted sequential renumbering.
  4. Moves files into the primary directory (ses-{ses}) with updated names.
  5. Merges the scans.tsv files (updating filenames, sorting by acq_time).
  6. Removes now-empty secondary directories.

Single-session case
-------------------
When there is only one session directory, the same renumbering is applied.
Files whose run number already matches their sequential position are left
untouched (no rename plan entry generated for them).

Run-number formatting
---------------------
  fmap   → 1-digit  (run-1, run-2, …)
  others → 2-digit  (run-01, run-02, …)

Usage
-----
  # dry-run single session (default):
  python 00_merge_session.py -b /BIDS -s 10,02

  # execute:
  python 00_merge_session.py -b /BIDS -s 10,02 --execute

  # batch:
  python 00_merge_session.py -b /BIDS -f subseslist.tsv --execute -v
"""

from __future__ import annotations

import csv
import glob
import os
import os.path as op
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from launchcontainers.utils import parse_subses_list, read_json_acqtime, substitute_run

console = Console()
app = typer.Typer(pretty_exceptions_show_locals=False)

MODALITIES = ("anat", "func", "dwi", "fmap")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_sec(t: str | None) -> float:
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


def _run_fmt(mod: str, run_int: int) -> str:
    """Format a run number: 1-digit for fmap, 2-digit for everything else."""
    return str(run_int) if mod == "fmap" else f"{run_int:02d}"


def _find_ses_dirs(bidsdir: str, sub: str, ses: str) -> list[str]:
    """Return all session dirs matching ses-{ses}*, sorted by name."""
    sub_dir = op.join(bidsdir, f"sub-{sub}")
    matches = sorted(glob.glob(op.join(sub_dir, f"ses-{ses}*")))
    return [d for d in matches if op.isdir(d)]


# ---------------------------------------------------------------------------
# Step 1: collect all files across split dirs, sorted by acq_time
# ---------------------------------------------------------------------------


def _collect_all_files(bidsdir: str, sub: str, ses: str) -> list[dict]:
    """
    Collect every sidecar JSON (and its companions) across all split dirs,
    sorted by AcquisitionTime ascending.

    Each entry:
      ses_dir    — full path to the session directory it lives in
      ses_label  — e.g. "02" or "02part2"
      mod        — anat / func / dwi / fmap
      mod_dir    — full path to the modality subdirectory
      stem       — basename without extension (e.g. sub-10_ses-02_task-fLoc_run-01_bold)
      json_path  — full path to .json
      nii_path   — full path to .nii.gz  (may not exist for .bval/.bvec files)
      extra_paths— list of additional companion paths (bval, bvec for DWI)
      acq_time   — raw AcquisitionTime string
      acq_sec    — seconds since midnight (float)
      run_int    — integer run number from filename (None if no run entity)
      task       — task label (func only, else None)
    """
    ses_dirs = _find_ses_dirs(bidsdir, sub, ses)
    rows: list[dict] = []

    for ses_dir in ses_dirs:
        ses_label = op.basename(ses_dir).replace("ses-", "")
        for mod in MODALITIES:
            mod_dir = op.join(ses_dir, mod)
            if not op.isdir(mod_dir):
                continue
            for jf in sorted(glob.glob(op.join(mod_dir, "*.json"))):
                stem = re.sub(r"\.json$", "", op.basename(jf))
                acq_time = read_json_acqtime(jf)
                run_m = re.search(r"_run-(\d+)_", stem)
                task_m = re.search(r"_task-(\w+)_", stem)
                acq_m = re.search(r"_acq-(\w+)_", stem)

                # Companion files
                extra: list[str] = []
                if mod == "dwi":
                    for ext in (".bval", ".bvec"):
                        p = op.join(mod_dir, stem + ext)
                        if op.exists(p):
                            extra.append(p)

                # suffix = last BIDS entity (e.g. "magnitude", "sbref", "phase",
                #          "bold", "T1w", "dwi", "epi", …)
                suffix = stem.split("_")[-1]

                rows.append(
                    {
                        "ses_dir": ses_dir,
                        "ses_label": ses_label,
                        "mod": mod,
                        "mod_dir": mod_dir,
                        "stem": stem,
                        "suffix": suffix,
                        "json_path": jf,
                        "nii_path": op.join(mod_dir, stem + ".nii.gz"),
                        "extra_paths": extra,
                        "acq_time": acq_time,
                        "acq_sec": _to_sec(acq_time),
                        "run_int": int(run_m.group(1)) if run_m else None,
                        "task": task_m.group(1) if task_m else None,
                        "acq": acq_m.group(1) if acq_m else None,
                    }
                )

    # ── Pass 2: collect json-less NII files (e.g. gfactor) ──────────────────
    # Build acq_time lookup from JSON-backed rows so gfactor inherits timing.
    acq_lookup: dict[tuple, tuple[str, float]] = {}
    for r in rows:
        if r["run_int"] is not None and r.get("task"):
            key = (r["ses_label"], r["mod"], r["task"], r["run_int"])
            acq_lookup.setdefault(key, (r["acq_time"], r["acq_sec"]))

    json_stems = {r["stem"] for r in rows}
    for ses_dir in ses_dirs:
        ses_label = op.basename(ses_dir).replace("ses-", "")
        for mod in MODALITIES:
            mod_dir = op.join(ses_dir, mod)
            if not op.isdir(mod_dir):
                continue
            for nii in sorted(glob.glob(op.join(mod_dir, "*.nii.gz"))):
                stem_nii = re.sub(r"\.nii\.gz$", "", op.basename(nii))
                if stem_nii in json_stems:
                    continue
                if op.exists(op.join(mod_dir, stem_nii + ".json")):
                    continue
                run_m = re.search(r"_run-(\d+)_", stem_nii)
                task_m = re.search(r"_task-(\w+)_", stem_nii)
                acq_m = re.search(r"_acq-(\w+)_", stem_nii)
                run_int = int(run_m.group(1)) if run_m else None
                task = task_m.group(1) if task_m else None
                suffix = stem_nii.split("_")[-1]
                key = (ses_label, mod, task, run_int)
                acq_time, acq_sec = acq_lookup.get(key, ("", float("inf")))
                rows.append(
                    {
                        "ses_dir": ses_dir,
                        "ses_label": ses_label,
                        "mod": mod,
                        "mod_dir": mod_dir,
                        "stem": stem_nii,
                        "suffix": suffix,
                        "json_path": "",
                        "nii_path": nii,
                        "extra_paths": [],
                        "acq_time": acq_time,
                        "acq_sec": acq_sec,
                        "run_int": run_int,
                        "task": task,
                        "acq": acq_m.group(1) if acq_m else None,
                    }
                )

    rows.sort(key=lambda r: r["acq_sec"])
    return rows


# ---------------------------------------------------------------------------
# Step 2: compute run offsets for secondary dirs
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Step 2+3: build rename plan via time-sorted sequential renumbering
# ---------------------------------------------------------------------------


def _group_key(r: dict) -> tuple:
    """
    Grouping key for sequential run renumbering.

    func  → (mod, task, suffix)  — sbref / magnitude / phase counted separately
    fmap  → (mod, None, None)    — AP+PA share the same run; paired by (ses_label, run_int)
    dwi   → (mod, acq, None)     — AP+PA share the same run; different acq- labels
                                   (e.g. acq-b0 vs acq-b1000) are independent sequences;
                                   pairs resolved by (ses_label, run_int) inside the group
    anat  → (mod, None, suffix)
    """
    if r["mod"] == "func" and r["task"]:
        return (r["mod"], r["task"], r["suffix"])
    if r["mod"] == "fmap":
        return (r["mod"], None, None)
    if r["mod"] == "dwi":
        return (r["mod"], r["acq"], None)
    return (r["mod"], None, r["suffix"])


def _build_rename_plan(
    primary_label: str,
    all_files: list[dict],
) -> list[dict]:
    """
    Build a rename/move plan for every file that needs to change.

    Strategy
    --------
    Rather than computing offsets, we assign new run numbers directly by
    sorting ALL files (primary + secondary) within each group by
    AcquisitionTime and numbering them 1, 2, 3 … in that order.

    This automatically:
    • gives secondary files the correct run number continuation, and
    • fills any gaps (e.g. run-08 magnitude, gap, run-10 → becomes 08, 09).

    For fmap the AP and PA files of the same run share the same run_int and
    are renumbered together (both get the same new run number).

    A file is added to the plan only when it actually needs to change
    (session label rename and/or run number change).

    Returns list of plan dicts:
      file        — original entry from _collect_all_files
      new_stem    — new basename without extension
      dst_mod_dir — target directory (always inside the primary session dir)
      old_run     — original run_int (or None)
      new_run     — assigned run_int (or None)
    """
    primary_dir = next(
        (r["ses_dir"] for r in all_files if r["ses_label"] == primary_label), None
    )
    if primary_dir is None:
        return []

    plan: list[dict] = []

    # ── files WITH run numbers ──────────────────────────────────────────────
    from collections import defaultdict

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_files:
        if r["run_int"] is not None:
            groups[_group_key(r)].append(r)

    for gkey, files in groups.items():
        mod = gkey[0]

        if mod in ("fmap", "dwi"):
            # fmap: AP+PA of the same acquisition share the same run_int.
            # dwi:  dir-AP and dir-PA of the same acquisition also share run_int,
            #       but different acq- labels (e.g. acq-b0 vs acq-b1000) are
            #       already in separate groups via _group_key.
            # In both cases: group by (ses_label, run_int) so that run-1 from
            # ses-03 and run-1 from ses-03part2 are treated as SEPARATE logical
            # runs, then sort groups by their earliest acq_sec.
            run_to_files: dict[tuple, list[dict]] = defaultdict(list)
            for r in files:
                run_to_files[(r["ses_label"], r["run_int"])].append(r)

            sorted_runs = sorted(
                run_to_files.items(),
                key=lambda kv: min(r["acq_sec"] for r in kv[1]),
            )

            for new_run_int, ((_ses, _old_run), run_files) in enumerate(
                sorted_runs, start=1
            ):
                for r in run_files:
                    _maybe_add(plan, r, new_run_int, primary_label, primary_dir)
        else:
            # Sort by acq_time; for files without acq_time fall back to run_int
            files_sorted = sorted(files, key=lambda r: (r["acq_sec"], r["run_int"]))
            for new_run_int, r in enumerate(files_sorted, start=1):
                _maybe_add(plan, r, new_run_int, primary_label, primary_dir)

    # ── files WITHOUT run numbers (just need ses-label update if secondary) ─
    for r in all_files:
        if r["run_int"] is not None:
            continue
        if r["ses_label"] == primary_label:
            continue
        label = r["ses_label"]
        new_stem = r["stem"].replace(f"ses-{label}", f"ses-{primary_label}")
        plan.append(
            {
                "file": r,
                "new_stem": new_stem,
                "dst_mod_dir": op.join(primary_dir, r["mod"]),
                "old_run": None,
                "new_run": None,
            }
        )

    return plan


def _maybe_add(
    plan: list[dict],
    r: dict,
    new_run_int: int,
    primary_label: str,
    primary_dir: str,
) -> None:
    """Add r to plan if it needs a ses-label rename and/or a run-number change."""
    label = r["ses_label"]
    old_run = r["run_int"]
    new_stem = r["stem"].replace(f"ses-{label}", f"ses-{primary_label}")

    if old_run != new_run_int:
        zero_pad = 1 if r["mod"] == "fmap" else 2
        new_stem = substitute_run(new_stem, new_run_int, zero_pad=zero_pad)

    needs_change = (label != primary_label) or (old_run != new_run_int)
    if needs_change:
        plan.append(
            {
                "file": r,
                "new_stem": new_stem,
                "dst_mod_dir": op.join(primary_dir, r["mod"]),
                "old_run": old_run,
                "new_run": new_run_int,
            }
        )


# ---------------------------------------------------------------------------
# Step 4: merge scans.tsv
# ---------------------------------------------------------------------------


def _merge_scans_tsv(
    primary_label: str,
    all_files: list[dict],
    plan: list[dict],
    dry_run: bool,
) -> int:
    """
    Merge scans.tsv files from all split dirs into the primary one.
    Filenames in secondary tsv entries are updated using the rename plan.
    Returns number of rows in the merged tsv (0 if no tsv found).
    """
    # Build lookup: original stem → new stem
    stem_map: dict[str, str] = {p["file"]["stem"]: p["new_stem"] for p in plan}

    primary_dir = None
    for r in all_files:
        if r["ses_label"] == primary_label:
            primary_dir = r["ses_dir"]
            break
    if primary_dir is None:
        return 0

    all_dfs: list[list[dict]] = []

    for r in all_files:
        tsv_path = op.join(
            r["ses_dir"],
            f"sub-{r['stem'].split('_')[0].replace('sub-', '')}"
            f"_ses-{r['ses_label']}_scans.tsv",
        )
        if not op.exists(tsv_path) or any(
            d.get("_tsv") == tsv_path for d in (all_dfs[-1] if all_dfs else [])
        ):
            continue
        try:
            with open(tsv_path, newline="") as fh:
                rows = list(csv.DictReader(fh, delimiter="\t"))
            for row in rows:
                fn = row.get("filename", "")
                fn_stem = re.sub(r"\.(nii\.gz|json|bval|bvec)$", "", op.basename(fn))
                if fn_stem in stem_map:
                    new_fn_stem = stem_map[fn_stem]
                    row["filename"] = fn.replace(fn_stem, new_fn_stem)
                # Update ses label in filename
                row["filename"] = row["filename"].replace(
                    f"ses-{r['ses_label']}/", f"ses-{primary_label}/"
                )
            all_dfs.append(rows)
        except Exception:
            pass

    if not all_dfs:
        return 0

    merged = []
    seen_fns: set[str] = set()
    for rows in all_dfs:
        for row in rows:
            fn = row.get("filename", "")
            if fn not in seen_fns:
                seen_fns.add(fn)
                merged.append(row)

    merged.sort(key=lambda r: _to_sec(r.get("acq_time", "")))

    if not dry_run and merged:
        primary_tsv = op.join(
            primary_dir,
            f"sub-{all_files[0]['stem'].split('_')[0].replace('sub-', '')}"
            f"_ses-{primary_label}_scans.tsv",
        )
        fieldnames = list(merged[0].keys())
        with open(primary_tsv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(merged)

    return len(merged)


# ---------------------------------------------------------------------------
# Step 5: execute renames + moves
# ---------------------------------------------------------------------------


def _execute_plan(plan: list[dict], dry_run: bool) -> dict:
    """
    Two-phase rename/move: source → temp → destination.
    Returns counts dict.
    """
    counts = {"moved": 0, "errors": 0}

    # Phase 1: rename source files to temp names (avoids conflicts)
    temp_records: list[dict | None] = []
    for p in plan:
        r = p["file"]
        temp_id = uuid.uuid4().hex[:8]
        temp_stem = f"._tmp_{temp_id}"

        # Compute run-tagged prefix for proper companion (extra) naming
        old_pfx_m = re.search(r"^(.*?_run-\d+_)", r["stem"])
        new_pfx_m = re.search(r"^(.*?_run-\d+_)", p["new_stem"])
        old_pfx = old_pfx_m.group(1) if old_pfx_m else None
        new_pfx = new_pfx_m.group(1) if new_pfx_m else None

        paths: list[tuple[str, str, str]] = []  # (src, temp, final_name)
        slot = 0

        def _add(src: str, final_name: str) -> None:
            nonlocal slot
            if src and op.exists(src):
                ext = ".nii.gz" if src.endswith(".nii.gz") else op.splitext(src)[1]
                temp_dst = op.join(r["mod_dir"], f"{temp_stem}_{slot}{ext}")
                paths.append((src, temp_dst, final_name))
                slot += 1  # noqa: F821

        _add(r["json_path"], p["new_stem"] + ".json")
        _add(r["nii_path"], p["new_stem"] + ".nii.gz")
        for ep in r["extra_paths"]:
            ep_bn = op.basename(ep)
            ep_ext = ".nii.gz" if ep_bn.endswith(".nii.gz") else op.splitext(ep_bn)[1]
            if old_pfx and new_pfx and ep_bn.startswith(old_pfx):
                ep_final = new_pfx + ep_bn[len(old_pfx) :]
            else:
                ep_final = p["new_stem"] + ep_ext
            _add(ep, ep_final)

        if not dry_run:
            try:
                for src, tmp, _ in paths:
                    os.rename(src, tmp)
            except Exception as exc:
                console.print(f"  [red][ERROR] phase-1 move: {exc}[/]")
                counts["errors"] += 1
                temp_records.append(None)
                continue

        temp_records.append(
            {
                "paths": paths,
                "dst_mod_dir": p["dst_mod_dir"],
            }
        )

    # Phase 2: move temp → final destination
    for rec in temp_records:
        if rec is None:
            continue
        dst_dir = rec["dst_mod_dir"]
        if not dry_run:
            os.makedirs(dst_dir, exist_ok=True)
        ok = True
        for _src, tmp, final_name in rec["paths"]:
            final = op.join(dst_dir, final_name)
            if not dry_run:
                try:
                    shutil.move(tmp, final)
                except Exception as exc:
                    console.print(f"  [red][ERROR] phase-2 move: {exc}[/]")
                    counts["errors"] += 1
                    ok = False
        if ok and not dry_run:
            counts["moved"] += 1
        elif ok:
            counts["moved"] += 1

    return counts


# ---------------------------------------------------------------------------
# Step 6: clean up empty secondary directories
# ---------------------------------------------------------------------------


def _cleanup_secondary_dirs(
    primary_label: str,
    all_files: list[dict],
    dry_run: bool,
) -> list[str]:
    """Remove empty modality dirs and session dirs for secondary labels."""
    removed = []
    secondary_dirs = {
        r["ses_dir"] for r in all_files if r["ses_label"] != primary_label
    }
    for ses_dir in sorted(secondary_dirs):
        for mod in MODALITIES:
            mod_dir = op.join(ses_dir, mod)
            if op.isdir(mod_dir):
                if not dry_run:
                    try:
                        if not any(True for _ in os.scandir(mod_dir)):
                            os.rmdir(mod_dir)
                            removed.append(mod_dir)
                    except Exception:
                        pass
                else:
                    removed.append(mod_dir)
        if not dry_run:
            try:
                if not any(True for _ in os.scandir(ses_dir)):
                    os.rmdir(ses_dir)
                    removed.append(ses_dir)
            except Exception:
                pass
        else:
            removed.append(ses_dir)
    return removed


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _print_timeline(
    sub: str,
    ses: str,
    all_files: list[dict],
    plan: list[dict],
    primary_label: str,
    verbose: bool = False,
) -> None:
    """Print one-liner per session; full acq-time table only when verbose=True."""
    plan_map: dict[str, dict] = {p["file"]["stem"]: p for p in plan}

    n_move = len(plan)
    n_rename = sum(1 for p in plan if p["old_run"] != p["new_run"])
    parts = sorted({r["ses_label"] for r in all_files})
    split_note = " + ".join(f"ses-{lbl}" for lbl in parts)

    if n_move:
        flag = f"  [yellow]{n_move} files to move[/]  [yellow]{n_rename} run-renames[/]"
    else:
        flag = "  [green]OK (no changes)[/]"

    console.print(f"  sub-{sub}  ses-{ses}  ({split_note}){flag}")

    if not verbose:
        return

    t = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 1))
    t.add_column("acq_time", justify="right", style="cyan")
    t.add_column("ses_part", justify="center", style="dim")
    t.add_column("mod", justify="center", style="dim")
    t.add_column("name")
    t.add_column("run change", justify="center")

    seen_stems: set[str] = set()  # deduplicate (json + nii share same stem)
    for r in all_files:
        if r["stem"] in seen_stems:
            continue
        seen_stems.add(r["stem"])

        p = plan_map.get(r["stem"])
        part_style = "yellow" if r["ses_label"] != primary_label else "dim"
        name_short = re.sub(rf"^sub-\w+_ses-{r['ses_label']}_", "", r["stem"])

        if p is not None:
            new_short = re.sub(rf"^sub-\w+_ses-{primary_label}_", "", p["new_stem"])
            if p["old_run"] != p["new_run"]:
                run_change = f"[yellow]run-{_run_fmt(r['mod'], p['old_run'])} → run-{_run_fmt(r['mod'], p['new_run'])}[/]"
                name_col = f"[yellow]{name_short}[/] → [green]{new_short}[/]"
            else:
                run_change = "[dim]ses only[/]"
                name_col = f"[dim]{name_short}[/] → [green]{new_short}[/]"
        else:
            run_change = ""
            name_col = name_short

        t.add_row(
            _fmt_time(r["acq_time"]) or "[dim]—[/]",
            f"[{part_style}]{r['ses_label']}[/]",
            r["mod"],
            name_col,
            run_change,
        )

    console.print(t)


def _print_summary_table(results: list[dict]) -> None:
    t = Table(
        title="merge-session summary",
        show_header=True,
        header_style="bold magenta",
        box=None,
        padding=(0, 1),
    )
    t.add_column("sub", justify="right", style="cyan")
    t.add_column("ses", justify="right", style="cyan")
    t.add_column("parts", justify="right")
    t.add_column("files", justify="right")
    t.add_column("moved", justify="right", style="yellow")
    t.add_column("errors", justify="right", style="red")
    t.add_column("status", justify="center")

    for r in results:
        if r.get("error"):
            st = f"[red]{r['error']}[/]"
        elif r.get("counts", {}).get("errors", 0):
            st = "[red]ERRORS[/]"
        elif r.get("counts", {}).get("moved", 0) == 0:
            st = "[green]OK (no changes)[/]"
        else:
            st = "[green]OK[/]"
        t.add_row(
            r["sub"],
            r["ses"],
            str(r["n_parts"]),
            str(r["n_files"]),
            str(r.get("counts", {}).get("moved", 0)),
            str(r.get("counts", {}).get("errors", 0)),
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
    dry_run: bool,
    verbose: bool,
    outdir: Optional[str],
) -> dict:
    ses_dirs = _find_ses_dirs(bidsdir, sub, ses)
    primary_label = ses  # primary = the exact ses label (e.g. "02")

    if not ses_dirs:
        console.print(f"[yellow]sub-{sub} ses-{ses}: not found — skipped[/]")
        return {
            "sub": sub,
            "ses": ses,
            "n_parts": 0,
            "n_files": 0,
            "error": "not found",
        }

    all_files = _collect_all_files(bidsdir, sub, ses)
    plan = _build_rename_plan(primary_label, all_files)

    # Count unique file stems only (avoid counting json + nii separately)
    unique_stems = len({r["stem"] for r in all_files})

    _print_timeline(sub, ses, all_files, plan, primary_label, verbose=verbose)

    counts = _execute_plan(plan, dry_run=dry_run)
    _merge_scans_tsv(primary_label, all_files, plan, dry_run=dry_run)
    removed_dirs = _cleanup_secondary_dirs(primary_label, all_files, dry_run=dry_run)

    if not dry_run and removed_dirs:
        for d in removed_dirs:
            console.print(f"  [dim]removed {d}[/]")

    # Write audit log
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log = op.join(outdir, f"merge_sub-{sub}_ses-{ses}_{ts}.tsv")
        with open(log, "w", newline="") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow(
                ["ses_label", "mod", "old_stem", "new_stem", "old_run", "new_run"]
            )
            for p in plan:
                writer.writerow(
                    [
                        p["file"]["ses_label"],
                        p["file"]["mod"],
                        p["file"]["stem"],
                        p["new_stem"],
                        p["old_run"],
                        p["new_run"],
                    ]
                )

    return {
        "sub": sub,
        "ses": ses,
        "n_parts": len(ses_dirs),
        "n_files": unique_stems,
        "counts": counts,
        "error": None,
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
    dry_run: bool = typer.Option(
        True, "--dry-run/--execute", help="Print plan only (default) or execute moves."
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show timeline for sessions with nothing to merge too.",
    ),
    outdir: Optional[Path] = typer.Option(
        None,
        "--outdir",
        "-o",
        help="Write per-session audit TSV logs to this directory.",
    ),
):
    """
    Merge split BIDS session directories (ses-02 + ses-02part2 → ses-02)
    and/or fix run-number gaps and non-1 starting indices in a single session.

    Run numbers are assigned by sorting ALL files per group (mod/task/suffix)
    by AcquisitionTime and numbering 1, 2, 3 … sequentially.  This works for
    both split sessions and single sessions with gaps.

    Default: dry-run.  Use --execute to apply.

    Examples:
        python 00_merge_split_ses_reording_runs.py -b /BIDS -s 10,02
        python 00_merge_split_ses_reording_runs.py -b /BIDS -s 10,02 -s 11,03
        python 00_merge_split_ses_reording_runs.py -b /BIDS -f subseslist.tsv --execute
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
    console.print(f"\n[bold]00_merge_session[/]  bidsdir={bidsdir}  mode={mode_str}\n")

    results = []
    for s, e in pairs:
        r = _process_session(
            str(bidsdir),
            s,
            e,
            dry_run=dry_run,
            verbose=verbose,
            outdir=str(outdir) if outdir else None,
        )
        results.append(r)

    console.print()
    _print_summary_table(results)

    if dry_run:
        console.print("\n[yellow]Dry-run complete.  Use --execute to apply.[/]")
    else:
        total_moved = sum(r.get("counts", {}).get("moved", 0) for r in results)
        total_errors = sum(r.get("counts", {}).get("errors", 0) for r in results)
        console.print(f"\n[green]Done.[/]  moved={total_moved}  errors={total_errors}")


if __name__ == "__main__":
    app()
