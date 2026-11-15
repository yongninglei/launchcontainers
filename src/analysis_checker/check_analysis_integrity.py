#!/usr/bin/env python3
"""
Check Analysis Integrity (Generic Engine)
==========================================
Validates completeness of neuroimaging analysis folders.

All analysis-specific knowledge (file lists, folder structure, timing logic)
lives in analysis_checker/.  This file contains only:
  - AnalysisSpec ABC
  - Result dataclasses
  - Checker engine
  - Output writers
  - CLI

Usage:
    python check_analysis_integrity.py /path/to/analysis prfprepare
    python check_analysis_integrity.py /path/to/analysis prfanalyze -s 01,01 -s 01,02
    python check_analysis_integrity.py /path/to/analysis prfanalyze
                                        --subseslist-file subseslist.csv
    python check_analysis_integrity.py /path/to/analysis bids --verbose

Output:
    1. <type>_subses_summary.txt    — brief sub,ses,RUN for every sub/ses
    2. <type>_detailed.log          — indexed detailed log of missing files
    3. <type>_matrix_simple.csv     — pivot: subjects × sessions (1/0)
    4. <type>_matrix_detailed.csv   — pivot: 1 / 0.5 / 0 (complete/partial/missing)
    5. <type>_corrupted.txt         — absolute paths of corrupted files (if any)
    6. <type>_timing_mismatch.txt   — timing-mismatch details (if any)
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path

import h5py
import pandas as pd
import scipy.io
import typer
from rich.console import Console
from rich.table import Table

try:
    from .base import AnalysisSpec
    from .bids import DWINiiSpec, FuncSBRefSpec, BIDSfuncSpec, BIDSSpec
    from .fmriprep import FMRIPrepSpec
    from .glm import GLMSpec
    from .prf_analyze import PRFAnalyzeSpec
    from .prf_prepare import PRFPrepareSpec
    from .qmri import BIDSQMRISpec
    from .rtp import RTPSpec, RTP2PipelineSpec
except ImportError:
    import sys
    import os

    sys.path.insert(0, os.path.dirname(__file__))
    from base import AnalysisSpec
    from bids import DWINiiSpec, FuncSBRefSpec, BIDSfuncSpec, BIDSSpec
    from fmriprep import FMRIPrepSpec
    from glm import GLMSpec
    from prf_analyze import PRFAnalyzeSpec
    from prf_prepare import PRFPrepareSpec
    from qmri import BIDSQMRISpec
    from rtp import RTPSpec

# Import all specs from analysis_checker package.
# [DEV] When you add a new spec file, register it in analysis_checker/__init__.py
#       — nothing here needs to change.

app = typer.Typer(help="Check analysis folder integrity against expected file specs.")
console = Console()


# =============================================================================
# 1. RESULT DATA CLASSES
# =============================================================================
@dataclass
class GroupResult:
    """
    Result for one file group (e.g. one run prefix, one modality folder).

    ─── Extend here for new error categories ────────────────────────────────
    [DEV] Add a new list field below (e.g. `truncated_files: list[str]`)
          then update is_complete to include it.
    """

    group_label: str
    expected_files: list[str]
    found_files: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    corrupted_files: list[str] = field(default_factory=list)  # [DEV] corrupted
    timing_issues: list[str] = field(default_factory=list)  # [DEV] timing
    log_issues: list[str] = field(default_factory=list)  # [DEV] log completion
    # [DEV] add new error category fields here ↓

    @property
    def is_complete(self) -> bool:
        # [DEV] add new error list to this condition when you add a new field
        return (
            len(self.missing_files) == 0
            and len(self.corrupted_files) == 0
            and len(self.timing_issues) == 0
            and len(self.log_issues) == 0
        )


@dataclass
class SessionResult:
    """
    Aggregated result for one (sub, ses) pair.

    ─── Extend here for new error categories ────────────────────────────────
    [DEV] When you add a new list to GroupResult, add a matching
          total_<field>() property here so the reporting layer can find it.
    """

    sub: str
    ses: str
    session_dir_exists: bool
    missing_folders: list[str] = field(default_factory=list)
    groups: dict[str, GroupResult] = field(default_factory=dict)

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    @property
    def complete_groups(self) -> int:
        return sum(1 for g in self.groups.values() if g.is_complete)

    @property
    def total_missing(self) -> int:
        return sum(len(g.missing_files) for g in self.groups.values())

    @property
    def total_corrupted(self) -> int:
        # [DEV] corrupted — mirrors GroupResult.corrupted_files
        return sum(len(g.corrupted_files) for g in self.groups.values())

    @property
    def total_timing_issues(self) -> int:
        # [DEV] timing — mirrors GroupResult.timing_issues
        return sum(len(g.timing_issues) for g in self.groups.values())

    @property
    def total_log_issues(self) -> int:
        # [DEV] log completion — mirrors GroupResult.log_issues
        return sum(len(g.log_issues) for g in self.groups.values())

    @property
    def extra_groups(self) -> dict[str, "GroupResult"]:
        """Groups discovered beyond the expected set (label starts with EXTRA:)."""
        return {k: v for k, v in self.groups.items() if k.startswith("EXTRA:")}

    @property
    def total_extra_groups(self) -> int:
        return len(self.extra_groups)

    # [DEV] add total_<field>() properties here ↓

    @property
    def is_complete(self) -> bool:
        return (
            self.session_dir_exists
            and len(self.missing_folders) == 0
            and self.num_groups > 0
            and all(g.is_complete for g in self.groups.values())
        )

    @property
    def is_standard(self) -> bool:
        """True only if complete AND no extra groups beyond the expected spec."""
        return self.is_complete and self.total_extra_groups == 0


# =============================================================================
# 2. FILE INTEGRITY HELPERS + CHECKER ENGINE
# =============================================================================

# ── File-level integrity checks ───────────────────────────────────────────────
# [DEV] Add new check_broken_<ext>() helpers here following the same pattern:
#       input:  Path  →  output: (is_valid: bool, error_msg: str)
#       Register the new helper inside check_one_session() below.


def check_broken_mat(filepath: Path) -> tuple[bool, str]:
    """Validate .mat file (scipy v5-v7.2 first, h5py v7.3 fallback)."""
    try:
        data = scipy.io.loadmat(str(filepath))
        for key, val in data.items():
            if not key.startswith("__"):
                _ = val.shape if hasattr(val, "shape") else val
        return True, ""
    except Exception as e_scipy:
        try:
            with h5py.File(str(filepath), "r") as f:

                def _read_all(obj):
                    for key in obj:
                        item = obj[key]
                        if hasattr(item, "keys"):
                            _read_all(item)
                        else:
                            _ = item[()]

                _read_all(f)
            return True, ""
        except Exception:
            return False, str(e_scipy)


def check_broken_json(filepath: Path) -> tuple[bool, str]:
    """Validate JSON file (parse + non-empty check)."""
    try:
        data = json.loads(filepath.read_text())
        if not data:
            return False, "empty JSON object"
        return True, ""
    except json.JSONDecodeError as e:
        return False, f"invalid JSON: {e}"
    except Exception as e:
        return False, str(e)


def check_broken_nii(filepath: Path) -> tuple[bool, str]:
    """Validate .nii.gz header + first-volume decompression (fast)."""
    try:
        import nibabel as nib
        import numpy as np

        img = nib.load(str(filepath))
        shape = img.header.get_data_shape()
        if len(shape) == 0:
            return False, "empty shape in header"
        proxy = img.dataobj
        _ = np.asarray(proxy[..., 0]) if len(shape) == 4 else np.asarray(proxy)
        return True, ""
    except Exception as e:
        return False, str(e)


# ── Core engine ───────────────────────────────────────────────────────────────


def _check_file_exists(session_dir: Path, fname: str, use_glob: bool) -> bool:
    if use_glob and ("*" in fname or "?" in fname):
        fpath = Path(fname)
        parent = session_dir / fpath.parent if str(fpath.parent) != "." else session_dir
        return bool(list(parent.glob(fpath.name))) if parent.is_dir() else False
    return (session_dir / fname).exists()


def check_one_session(
    spec: AnalysisSpec,
    analysis_dir: Path,
    sub: str,
    ses: str,
    check_corruption: bool = False,
) -> SessionResult:
    """
    Check a single sub/ses against the spec.

    ─── Where to add new per-file checks ────────────────────────────────────
    [DEV] To wire in a new file-level check (e.g. truncation):
      1. Add check_broken_<ext>() helper above
      2. Add an elif branch here (same pattern as .mat / .json / .nii.gz)
      3. Append to the new error list on GroupResult instead of `corrupted`
    """
    session_dir = spec.get_session_dir(analysis_dir, sub, ses)
    result = SessionResult(
        sub=sub,
        ses=ses,
        session_dir_exists=session_dir.is_dir(),
    )

    for folder in spec.get_expected_subfolders(analysis_dir, sub, ses):
        if not folder.is_dir():
            try:
                rel = str(folder.relative_to(analysis_dir))
            except ValueError:
                rel = str(folder)
            result.missing_folders.append(rel)

    if not result.session_dir_exists:
        return result

    for group_label, expected_files in spec.get_expected_groups(session_dir).items():
        found, missing, corrupted = [], [], []
        is_extra = group_label.startswith("EXTRA:")

        for fname in expected_files:
            fpath = session_dir / fname
            if _check_file_exists(session_dir, fname, spec.uses_glob):
                # ── Integrity checks (only when --check-corrupted is set) ──
                # [DEV] add new elif branch here for a new extension
                if not check_corruption:
                    valid, err = True, ""
                elif fname.endswith(".mat"):
                    valid, err = check_broken_mat(fpath)
                elif fname.endswith(".json"):
                    valid, err = check_broken_json(fpath)
                elif fname.endswith(".nii.gz"):
                    valid, err = check_broken_nii(fpath)
                else:
                    valid, err = True, ""

                if not valid:
                    corrupted.append(f"{fname} ({err})")
                else:
                    found.append(fname)
            else:
                # Extra groups are informational — absent companion files are
                # not "missing" from the experiment, just not present for the run.
                if not is_extra:
                    missing.append(fname)

        result.groups[group_label] = GroupResult(
            group_label=group_label,
            expected_files=expected_files,
            found_files=found,
            missing_files=missing,
            corrupted_files=corrupted,
        )

    # ── Timing check — duck-typed: only specs that implement _check_timing ──
    # [DEV] For a new spec-level custom check (like timing), implement
    #       _check_timing(session_dir, group_label) -> str | None on your spec
    #       and append to the matching GroupResult list here.
    if hasattr(spec, "_check_timing"):
        for group_label, gresult in result.groups.items():
            if group_label.startswith("EXTRA:"):
                continue
            if gresult.missing_files:
                continue
            issue = spec._check_timing(session_dir, group_label)
            if issue:
                gresult.timing_issues.append(issue)

    # ── Log-completion check — duck-typed: only specs with _check_log_completion ──
    # [DEV] implement _check_log_completion(session_dir, group_label) -> str | None
    #       on a spec to verify log file content (e.g. exit signal).
    if hasattr(spec, "_check_log_completion"):
        for group_label, gresult in result.groups.items():
            if group_label.startswith("EXTRA:"):
                continue
            if gresult.missing_files:
                continue
            issue = spec._check_log_completion(session_dir, group_label)
            if issue:
                gresult.log_issues.append(issue)

    return result


def run_integrity_check_parallel(
    spec: AnalysisSpec,
    analysis_dir: Path,
    subses_list: list[tuple[str, str]],
    max_workers: int = 30,
    check_corruption: bool = False,
) -> list[SessionResult]:
    """Run integrity check across all sub/ses pairs in parallel (I/O-bound)."""

    def _check(sub_raw: str, ses_raw: str) -> SessionResult:
        sub = f"sub-{sub_raw}" if not sub_raw.startswith("sub-") else sub_raw
        ses = f"ses-{ses_raw}" if not ses_raw.startswith("ses-") else ses_raw
        return check_one_session(spec, analysis_dir, sub, ses, check_corruption)

    results: list[SessionResult | None] = [None] * len(subses_list)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_check, sub, ses): idx
            for idx, (sub, ses) in enumerate(subses_list)
        }
        with typer.progressbar(
            length=len(subses_list),
            label=f"Checking {spec.name} integrity",
            show_pos=True,
        ) as progress:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    sub, ses = subses_list[idx]
                    console.print(f"[red]Error checking sub-{sub}/ses-{ses}: {e}[/red]")
                    results[idx] = SessionResult(
                        sub=f"sub-{sub}",
                        ses=f"ses-{ses}",
                        session_dir_exists=False,
                    )
                progress.update(1)

    return results  # type: ignore[return-value]


def run_integrity_check_single(
    spec: AnalysisSpec,
    analysis_dir: Path,
    subses_list: list[tuple[str, str]],
    check_corruption: bool = False,
) -> list[SessionResult]:
    """Run integrity check sequentially (useful for debugging)."""
    results = []
    with typer.progressbar(
        subses_list,
        label=f"Checking {spec.name} integrity",
        show_pos=True,
        length=len(subses_list),
    ) as progress:
        for sub_raw, ses_raw in progress:
            sub = f"sub-{sub_raw}" if not sub_raw.startswith("sub-") else sub_raw
            ses = f"ses-{ses_raw}" if not ses_raw.startswith("ses-") else ses_raw
            results.append(
                check_one_session(spec, analysis_dir, sub, ses, check_corruption)
            )
    return results


# =============================================================================
# 4. OUTPUT WRITERS
# =============================================================================


def write_brief_csv(results: list[SessionResult], output_path: Path) -> pd.DataFrame:
    """Write brief CSV: sub, ses, RUN (True/False). Returns DataFrame."""
    rows = []
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sub", "ses", "RUN"])
        for r in results:
            sub_num = r.sub.replace("sub-", "")
            ses_num = r.ses.replace("ses-", "")
            writer.writerow([sub_num, ses_num, r.is_standard])
            rows.append({"sub": sub_num, "ses": ses_num, "RUN": r.is_standard})
    return pd.DataFrame(rows, columns=["sub", "ses", "RUN"])


def write_matrix_from_brief_csv(
    source: pd.DataFrame | Path,
    output_path: Path,
) -> None:
    """
    Pivot table CSV (subjects × sessions), values: 1 = complete, 0 = not.
    Accepts either a DataFrame or a path to a brief CSV file.
    """
    if isinstance(source, Path):
        df = pd.read_csv(source, dtype=str)
    else:
        df = source.copy().astype(str)

    subs = list(dict.fromkeys(df["sub"].tolist()))
    sessions = list(dict.fromkeys(df["ses"].tolist()))
    lookup = {
        (str(r["sub"]), str(r["ses"])): "1" if str(r["RUN"]).lower() == "true" else "0"
        for _, r in df.iterrows()
    }

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + [f"ses-{s}" for s in sessions])
        for sub in subs:
            writer.writerow(
                [f"sub-{sub}"] + [lookup.get((sub, ses), "") for ses in sessions]
            )

    console.print(f"[bold]Matrix (simple) CSV:[/bold] {output_path}")


def write_matrix_from_result(results: list[SessionResult], output_path: Path) -> None:
    """
    Pivot table CSV with three-level completeness:
      1   = complete
      0.5 = incomplete but has data
      0   = no directory / no groups
    """
    subs_seen: dict[str, None] = {}
    ses_seen: dict[str, None] = {}
    lookup: dict[tuple[str, str], float] = {}

    for r in results:
        sub_num = r.sub.replace("sub-", "")
        ses_num = r.ses.replace("ses-", "")
        subs_seen[sub_num] = None
        ses_seen[ses_num] = None
        if r.is_complete:
            lookup[(sub_num, ses_num)] = 1.0
        elif not r.session_dir_exists or r.num_groups == 0:
            lookup[(sub_num, ses_num)] = 0.0
        else:
            lookup[(sub_num, ses_num)] = 0.5

    def _fmt(v: float) -> str:
        return {1.0: "1", 0.5: "0.5", 0.0: "0"}.get(v, "")

    subs = list(subs_seen.keys())
    sessions = list(ses_seen.keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + [f"ses-{s}" for s in sessions])
        for sub in subs:
            writer.writerow(
                [f"sub-{sub}"]
                + [
                    _fmt(lookup[(sub, ses)]) if (sub, ses) in lookup else ""
                    for ses in sessions
                ],
            )

    console.print(f"[bold]Matrix (detailed) CSV:[/bold] {output_path}")


def write_detailed_log(
    results: list[SessionResult],
    spec: AnalysisSpec,
    output_path: Path,
) -> None:
    """
    Indexed detailed log [0], [1], ... matching brief CSV rows.

    ─── Where to add new error categories ───────────────────────────────────
    [DEV] Add a new `if gresult.<new_list>:` block here (copy the
          corrupted_files block as a template) to surface new error types.
    """
    with open(output_path, "w") as f:
        f.write(f"# {spec.name.upper()} Analysis Integrity — Detailed Log\n")
        f.write(f"# {spec.description}\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# {'=' * 70}\n\n")

        for idx, r in enumerate(results):
            sub_num = r.sub.replace("sub-", "")
            ses_num = r.ses.replace("ses-", "")
            status = "COMPLETE" if r.is_complete else "INCOMPLETE"
            f.write(f"[{idx}] sub={sub_num}, ses={ses_num} — {status}\n")

            if not r.session_dir_exists:
                f.write("      Session directory does not exist.\n\n")
                continue

            if r.missing_folders:
                f.write("      Missing folders:\n")
                for mf in r.missing_folders:
                    f.write(f"        - {mf}\n")

            if r.num_groups == 0:
                f.write("      No file groups discovered (empty directory?).\n\n")
                continue

            expected_complete = r.complete_groups - r.total_extra_groups
            expected_total = r.num_groups - r.total_extra_groups
            f.write(
                f"      Groups: {expected_complete}/{expected_total} complete  |  "
                f"Missing: {r.total_missing}  |  Corrupted: {r.total_corrupted}"
                + (
                    f"  |  Extra: {r.total_extra_groups}"
                    if r.total_extra_groups
                    else ""
                )
                + "\n",
            )
            # [DEV] add new totals to the summary line above ↑

            # Expected groups — show missing / corrupted / timing issues
            for glabel, gresult in sorted(r.groups.items()):
                if glabel.startswith("EXTRA:"):
                    continue
                if gresult.is_complete:
                    continue

                if gresult.missing_files:
                    f.write(
                        f"      [{glabel}] — {len(gresult.missing_files)} missing:\n"
                    )
                    for mf in gresult.missing_files:
                        f.write(f"        - {mf}\n")

                # [DEV] add new error-list blocks here ↓ (copy this pattern)
                if gresult.corrupted_files:
                    f.write(
                        f"      [{glabel}] — {len(gresult.corrupted_files)} corrupted:\n"
                    )
                    for cf in gresult.corrupted_files:
                        f.write(f"        ! {cf}\n")

                if gresult.timing_issues:
                    f.write(f"      [{glabel}] — timing mismatch:\n")
                    for ti in gresult.timing_issues:
                        f.write(f"        ~ {ti}\n")

                if gresult.log_issues:
                    f.write(f"      [{glabel}] — log incomplete:\n")
                    for li in gresult.log_issues:
                        f.write(f"        % {li}\n")

            # Extra groups — informational, separate section
            if r.extra_groups:
                f.write(f"      Extra runs found ({len(r.extra_groups)}):\n")
                for glabel, gresult in sorted(r.extra_groups.items()):
                    task_run = glabel.removeprefix("EXTRA:")
                    found_n = len(gresult.found_files)
                    total_n = len(gresult.expected_files)
                    f.write(
                        f"        + {task_run}  ({found_n}/{total_n} files present)\n"
                    )

            f.write("\n")


def write_corrupted_list(
    results: list[SessionResult],
    output_path: Path,
    spec: AnalysisSpec,
    analysis_dir: Path,
) -> None:
    """Write absolute paths of all corrupted files, one per line."""
    with open(output_path, "w") as f:
        for r in results:
            session_dir = spec.get_session_dir(analysis_dir, r.sub, r.ses)
            for glabel, gresult in r.groups.items():
                if glabel.startswith("EXTRA:"):
                    continue
                for cf in gresult.corrupted_files:
                    fname = cf.split(" (")[0]
                    f.write(f"{session_dir / fname}\n")


def write_time_mismatch_list(
    results: list[SessionResult],
    output_path: Path,
    spec: AnalysisSpec,
    analysis_dir: Path,
) -> None:
    """Write bold paths + timing issue details, one per line."""
    with open(output_path, "w") as f:
        for r in results:
            session_dir = spec.get_session_dir(analysis_dir, r.sub, r.ses)
            for glabel, gresult in r.groups.items():
                if glabel.startswith("EXTRA:"):
                    continue
                if gresult.timing_issues:
                    bold_path = session_dir / glabel
                    for issue in gresult.timing_issues:
                        f.write(f"{bold_path}  {issue.strip()}\n")


def write_log_issues_list(
    results: list[SessionResult],
    output_path: Path,
    spec: AnalysisSpec,
    analysis_dir: Path,
) -> None:
    """Write sub/ses pairs where the pipeline log did not finish cleanly."""
    with open(output_path, "w") as f:
        for r in results:
            session_dir = spec.get_session_dir(analysis_dir, r.sub, r.ses)
            for glabel, gresult in r.groups.items():
                if glabel.startswith("EXTRA:"):
                    continue
                if gresult.log_issues:
                    for issue in gresult.log_issues:
                        f.write(f"{session_dir}  {issue.strip()}\n")


def write_extra_runs_list(
    results: list[SessionResult],
    output_path: Path,
    spec: AnalysisSpec,
    analysis_dir: Path,
) -> None:
    """Write one line per extra run found beyond the expected set."""
    with open(output_path, "w") as f:
        f.write("# Extra runs — present in func/ but not in the expected spec\n")
        f.write("# sub  ses  task_run  files_found/files_expected\n\n")
        for r in results:
            if not r.extra_groups:
                continue
            session_dir = spec.get_session_dir(analysis_dir, r.sub, r.ses)
            sub_num = r.sub.replace("sub-", "")
            ses_num = r.ses.replace("ses-", "")
            for glabel, gresult in sorted(r.extra_groups.items()):
                task_run = glabel.removeprefix("EXTRA:")
                found_n = len(gresult.found_files)
                total_n = len(gresult.expected_files)
                f.write(f"{sub_num}  {ses_num}  {task_run}  {found_n}/{total_n}\n")
                for ff in gresult.found_files:
                    f.write(f"  {session_dir / ff}\n")


# =============================================================================
# 5. RICH CONSOLE OUTPUT
# =============================================================================


def print_summary(results: list[SessionResult], spec: AnalysisSpec) -> None:
    """
    Rich summary table + aggregate stats.

    ─── Where to add new columns ─────────────────────────────────────────────
    [DEV] To surface a new error category in the table:
      1. Add a table.add_column(...) call below
      2. Compute the per-row string inside the for loop
      3. Add it to table.add_row(...)
      4. Add an aggregate stat in the console.print() block at the bottom
    """
    n_complete = sum(1 for r in results if r.is_standard)
    n_incomplete = sum(1 for r in results if not r.is_standard)
    n_no_dir = sum(1 for r in results if not r.session_dir_exists)
    n_no_groups = sum(1 for r in results if r.session_dir_exists and r.num_groups == 0)
    total_groups = sum(r.num_groups for r in results)
    complete_groups = sum(r.complete_groups for r in results)
    total_corrupted = sum(r.total_corrupted for r in results)
    total_timing = sum(r.total_timing_issues for r in results)
    total_log_issues = sum(r.total_log_issues for r in results)

    table = Table(title=f"{spec.name.upper()} Integrity Summary")
    table.add_column("sub", style="cyan")
    table.add_column("ses", style="cyan")
    table.add_column("RUN", justify="center")
    table.add_column("Groups", justify="center")
    table.add_column("Missing", justify="center")
    table.add_column("Corrupted", justify="center")  # [DEV] new column template
    table.add_column("Timing", justify="center")  # [DEV] new column template
    table.add_column("Log", justify="center")  # [DEV] log completion column
    table.add_column("Extra", justify="center")
    # [DEV] add new columns here ↓

    for r in results:
        if r.is_standard:
            run_style = "green"
        else:
            run_style = "red bold"

        if not r.session_dir_exists:
            groups_str = "no dir"
            missing_str = "—"
            corrupted_str = "—"
            timing_str = "—"
            log_str = "—"
            extra_str = "—"
        elif r.num_groups == 0:
            groups_str = "0"
            missing_str = "—"
            corrupted_str = "—"
            timing_str = "—"
            log_str = "—"
            extra_str = "—"
        else:
            expected_n = r.num_groups - r.total_extra_groups
            groups_str = f"{r.complete_groups - r.total_extra_groups}/{expected_n}"
            missing_str = str(r.total_missing) if r.total_missing else "0"
            corrupted_str = (
                f"[red bold]{r.total_corrupted}[/red bold]"
                if r.total_corrupted
                else "0"
            )
            timing_str = (
                f"[yellow bold]{r.total_timing_issues}[/yellow bold]"
                if r.total_timing_issues
                else "0"
            )
            log_str = (
                f"[red bold]{r.total_log_issues}[/red bold]"
                if r.total_log_issues
                else "0"
            )
            extra_str = (
                f"[yellow]{r.total_extra_groups}[/yellow]"
                if r.total_extra_groups
                else "0"
            )
            # [DEV] compute new column cell value here ↓

        table.add_row(
            r.sub.replace("sub-", ""),
            r.ses.replace("ses-", ""),
            f"[{run_style}]{r.is_standard}[/{run_style}]",
            groups_str,
            missing_str,
            corrupted_str,
            timing_str,
            log_str,
            extra_str,
            # [DEV] add new cell value here ↓
        )

    console.print(table)
    console.print(
        f"\n[green]{n_complete} complete[/green] | "
        f"[red]{n_incomplete} incomplete[/red] out of {len(results)} sessions",
    )
    if total_groups > 0:
        console.print(f"Groups: {complete_groups}/{total_groups} complete overall")
    # [DEV] add new aggregate stats here ↓
    if total_corrupted:
        console.print(f"[red bold]Corrupted files: {total_corrupted}[/red bold]")
    if total_timing:
        console.print(f"[yellow bold]Timing issues: {total_timing}[/yellow bold]")
    if total_log_issues:
        console.print(f"[red bold]Log incomplete: {total_log_issues}[/red bold]")
    if n_no_dir:
        console.print(f"[yellow]Missing session dirs: {n_no_dir}[/yellow]")
    if n_no_groups:
        console.print(f"[yellow]Empty session dirs: {n_no_groups}[/yellow]")


def print_all_groups(results: list[SessionResult], spec: AnalysisSpec) -> None:
    """Print group breakdown for every session, complete or not."""
    for r in results:
        status = "[green]✓[/green]" if r.is_complete else "[red]✗[/red]"
        console.print(f"\n{status} [cyan]{r.sub}/{r.ses}[/cyan]")

        if not r.session_dir_exists:
            console.print("  [yellow]directory missing[/yellow]")
            continue

        for glabel, gresult in sorted(r.groups.items()):
            group_status = "[green]✓[/green]" if gresult.is_complete else "[red]✗[/red]"
            console.print(
                f"  {group_status} {glabel} "
                f"({len(gresult.found_files)}/{len(gresult.expected_files)} files)"
            )
            if gresult.missing_files:
                for mf in gresult.missing_files:
                    console.print(f"      [red]- {mf}[/red]")
            if gresult.corrupted_files:
                for cf in gresult.corrupted_files:
                    console.print(f"      [magenta]! {cf}[/magenta]")


def print_detailed_results(
    results: list[SessionResult],
    verbose: bool = False,
) -> None:
    """
    Print missing/corrupted/timing details for non-standard sessions.

    Covers:
    - Truly incomplete sessions (missing expected files)
    - Complete sessions with extra ret* runs (flagged red)

    ─── Where to add new error categories ───────────────────────────────────
    [DEV] Add a new `if gresult.<new_list>:` block here (copy the
          corrupted_files block as a template).
    """
    non_standard = [r for r in results if not r.is_standard]
    if not non_standard:
        console.print("\n[bold green]✓ All sessions complete![/bold green]")
        return

    # --- Sessions with extra ret* runs but otherwise complete ---
    extra_ret_sessions = [
        r for r in non_standard if r.is_complete and r.total_extra_groups > 0
    ]
    if extra_ret_sessions:
        console.print(
            f"\n[bold red]Sessions with extra ret* runs ({len(extra_ret_sessions)}):[/bold red]\n"
        )
        for r in extra_ret_sessions:
            # group EXTRA: labels by task
            by_task: dict[str, list[str]] = defaultdict(list)
            for glabel in r.extra_groups:
                m = re.search(r"task-(\w+)_run-(\d+)", glabel)
                if m:
                    by_task[m.group(1)].append(m.group(2))
            task_summary = ", ".join(
                f"task-{task}: {len(runs)} extra run(s) (run-{', run-'.join(sorted(runs))})"
                for task, runs in sorted(by_task.items())
            )
            console.print(f"  [red]{r.sub}/{r.ses}[/red] — {task_summary}")
        console.print()

    incomplete = [r for r in non_standard if not r.is_complete]
    if not incomplete:
        return

    console.print(f"\n[bold red]Incomplete Sessions ({len(incomplete)}):[/bold red]\n")
    for r in incomplete:
        if not r.session_dir_exists:
            console.print(f"  [yellow]{r.sub}/{r.ses}[/yellow] — directory missing")
            continue
        if r.num_groups == 0:
            console.print(f"  [yellow]{r.sub}/{r.ses}[/yellow] — no file groups found")
            continue

        console.print(
            f"  [cyan]{r.sub}/{r.ses}[/cyan] "
            f"({r.complete_groups}/{r.num_groups} groups complete, "
            f"{r.total_missing} missing, {r.total_corrupted} corrupted)",
        )
        for glabel, gresult in sorted(r.groups.items()):
            if gresult.is_complete:
                if verbose:
                    console.print(f"    [green]✓ {glabel}[/green]")
                continue

            if gresult.missing_files:
                console.print(
                    f"    [red]✗ {glabel}[/red] — {len(gresult.missing_files)} missing:"
                )
                for mf in gresult.missing_files:
                    console.print(f"      - {mf}")

            # [DEV] add new error-list blocks here ↓ (copy this pattern)
            if gresult.corrupted_files:
                console.print(
                    f"    [magenta]⚠ {glabel}[/magenta] — "
                    f"{len(gresult.corrupted_files)} corrupted:",
                )
                for cf in gresult.corrupted_files:
                    console.print(f"      - {cf}")

            if gresult.timing_issues:
                console.print(f"    [yellow]⏱ {glabel}[/yellow] — timing mismatch:")
                for ti in gresult.timing_issues:
                    console.print(f"      [yellow]{ti}[/yellow]")

            if gresult.log_issues:
                console.print(f"    [red]✗ {glabel}[/red] — log incomplete:")
                for li in gresult.log_issues:
                    console.print(f"      [red]{li}[/red]")

    console.print()


def print_group_distribution(results: list[SessionResult]) -> None:
    """Print histogram of how many groups each session has."""
    counts: dict[int, int] = defaultdict(int)
    for r in results:
        if r.num_groups > 0:
            counts[r.num_groups] += 1
    if not counts:
        return
    console.print("\n[bold]Group Distribution:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("# Groups", style="cyan")
    table.add_column("# Sessions", style="green")
    for n in sorted(counts.keys()):
        table.add_row(str(n), str(counts[n]))
    console.print(table)


# a more detailed summary for prfanalyze
def summarize_incomplete_by_dimension(
    results: list[SessionResult],
    spec: AnalysisSpec,
    output_dir: Path,
) -> None:
    """
    Generic dimensional summary for incomplete sessions.
    Dimension is defined per-spec via get_group_dimension().
    If spec returns None for all groups, prints a flat incomplete list instead.
    """
    # dim_value -> set of (sub, ses) for console dedup
    dim_subses: defaultdict[str, set] = defaultdict(set)
    # dim_value -> list of (sub, ses) for txt output
    dim_entries: defaultdict[str, list] = defaultdict(list)
    missing_dirs: list[tuple[str, str]] = []
    dim_name = "group"  # fallback label

    for r in results:
        if not r.session_dir_exists:
            missing_dirs.append((r.sub, r.ses))
            continue
        if r.is_complete:
            continue
        for glabel, gresult in r.groups.items():
            if gresult.is_complete:
                continue
            dim = spec.get_group_dimension(glabel)
            if dim is not None:
                dim_name, dim_val = dim
                dim_subses[dim_val].add((r.sub, r.ses))
                dim_entries[dim_val].append((r.sub, r.ses))
            else:
                # No dimension — bucket everything under the group label itself
                dim_subses[glabel].add((r.sub, r.ses))
                dim_entries[glabel].append((r.sub, r.ses))

    # ── Console print ──────────────────────────────────────────────────────
    console.print(f"\n[bold cyan]Incomplete by {dim_name}:[/bold cyan]")
    if dim_subses:
        for dim_val, subses_set in sorted(dim_subses.items()):
            subses_strs = ", ".join(
                f"[yellow]{sub}_{ses}[/yellow]" for sub, ses in sorted(subses_set)
            )
            console.print(f"  [bold]{dim_name}-{dim_val}[/bold]: {subses_strs}")
    else:
        console.print("  [green]None[/green]")

    if missing_dirs:
        console.print("\n[bold red]Missing Session Directories:[/bold red]")
        for sub, ses in sorted(missing_dirs):
            console.print(f"  [red]{sub}_{ses}[/red]")

    # ── Summary counts ─────────────────────────────────────────────────────
    console.print("\n[bold cyan]Summary:[/bold cyan]")
    console.print(
        f"  {dim_name}s with incomplete sessions : [red]{len(dim_subses)}[/red]"
    )
    all_incomplete = set().union(*dim_subses.values()) if dim_subses else set()
    console.print(
        f"  Unique incomplete sub/ses           : [red]{len(all_incomplete)}[/red]"
    )
    console.print(
        f"  Missing session directories         : [red]{len(missing_dirs)}[/red]"
    )
    total = sum(len(v) for v in dim_entries.values())
    console.print(f"  Total incomplete groups             : [red]{total}[/red]")

    # ── Write per-dimension txt files ──────────────────────────────────────
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for dim_val, entries in sorted(dim_entries.items()):
            out_file = output_dir / f"incomplete_{dim_name}-{dim_val}.txt"
            with open(out_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["sub", "ses", "RUN"])
                seen: set = set()
                for sub, ses in sorted(entries):
                    if (sub, ses) not in seen:
                        seen.add((sub, ses))
                        writer.writerow(
                            [
                                sub.replace("sub-", ""),
                                ses.replace("ses-", ""),
                                True,
                            ]
                        )
            console.print(f"  [dim]Wrote {out_file}[/dim]")


# =============================================================================
# 7. SPEC REGISTRY + CLI
# =============================================================================


SPEC_REGISTRY: dict[str, AnalysisSpec] = {
    "prfprepare": PRFPrepareSpec(),
    "prfanalyze": PRFAnalyzeSpec(),
    "bids": BIDSSpec(),
    "dwinii": DWINiiSpec(),
    "funcsbref": FuncSBRefSpec(),
    "bidsfunc": BIDSfuncSpec(),
    "qmri": BIDSQMRISpec(),
    "fmriprep": FMRIPrepSpec(),
    "glm": GLMSpec(),
    "rtp": RTPSpec(),
    "rtp2pipeline": RTP2PipelineSpec(),
    # [DEV] register new spec here ↓
}


def parse_subses(raw: str) -> tuple[str, str]:
    parts = raw.strip().split(",")
    if len(parts) != 2:
        raise typer.BadParameter(f"Expected 'sub,ses' but got: '{raw}'")
    return parts[0].strip().replace("sub-", ""), parts[1].strip().replace("ses-", "")


def load_subseslist_from_file(filepath: Path) -> list[tuple[str, str]]:
    pairs = []
    with open(filepath) as f:
        # Peek at the first non-comment line to detect whether a header exists.
        reader = csv.reader(f)
        first = None
        for row in reader:
            if row and row[0].strip().lower() not in ("#", ""):
                first = row
                break
        if first is None:
            return pairs

        has_header = first[0].strip().lower() in ("sub", "subject")
        if has_header:
            # Re-open with DictReader so we can filter on RUN by name.
            f.seek(0)
            for row in csv.DictReader(f):
                sub = (
                    str(row.get("sub", row.get("subject", "")))
                    .strip()
                    .replace("sub-", "")
                )
                ses = str(row.get("ses", "")).strip().replace("ses-", "")
                run = str(row.get("RUN", "True")).strip()
                if not sub or not ses:
                    continue
                if run != "True":
                    continue
                pairs.append((sub, ses))
        else:
            # No header — fall back to positional parsing (col 0=sub, 1=ses, 2=RUN).
            # Process the already-read first row, then continue the iterator.
            def _include(row):
                if len(row) >= 3:
                    return row[2].strip() == "True"
                return True  # no RUN column → include all

            if _include(first):
                pairs.append(
                    (
                        first[0].strip().replace("sub-", ""),
                        first[1].strip().replace("ses-", ""),
                    )
                )
            for row in reader:
                if not row or len(row) < 2:
                    continue
                if _include(row):
                    pairs.append(
                        (
                            row[0].strip().replace("sub-", ""),
                            row[1].strip().replace("ses-", ""),
                        )
                    )

    return pairs


@app.command()
def check(
    analysis_dir: Path = typer.Argument(
        ..., help="Path to analysis root directory.", exists=True
    ),
    analysis_type: str = typer.Argument(
        ...,
        help="Analysis type: prfprepare, prfanalyze, bids, bidsfunc, "
        "dwinii, funcsbref, qmri, fmriprep, glm, rtp, rtp2pipeline",
    ),
    subses: list[str] | None = typer.Option(
        None,
        "--subses",
        "-s",
        help="Sub,ses pairs: -s 01,02 -s 03,04",
    ),
    subseslist_file: Path | None = typer.Option(
        None,
        "--subseslist-file",
        "-f",
        help="CSV/txt with sub,ses columns",
    ),
    output_dir: Path = typer.Option(
        Path("."),
        "--output-dir",
        "-o",
        help="Directory for output reports",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show complete groups too"
    ),
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Show all groups (verbose + debug)"
    ),
    show_distribution: bool = typer.Option(
        True, "--show-distribution/--no-distribution"
    ),
    max_workers: int = typer.Option(
        30, "--workers", "-j", help="Parallel workers (default: 30)"
    ),
    check_corrupted: bool = typer.Option(
        False,
        "--check-corrupted",
        help="Check file integrity (slow)",
    ),
) -> None:
    """Check analysis integrity against expected file/folder specs."""

    if analysis_type not in SPEC_REGISTRY:
        console.print(
            f"[red]Error:[/red] Unknown type '{analysis_type}'. "
            f"Valid: {list(SPEC_REGISTRY.keys())}",
        )
        raise typer.Exit(code=1)

    spec = SPEC_REGISTRY[analysis_type]

    if subses:
        pairs = [parse_subses(s) for s in subses]
    elif subseslist_file:
        if not subseslist_file.exists():
            console.print(f"[red]Error:[/red] File not found: {subseslist_file}")
            raise typer.Exit(code=1)
        pairs = load_subseslist_from_file(subseslist_file)
    else:
        defaults = spec.get_default_combinations()
        if defaults:
            pairs = [
                (s.replace("sub-", ""), ss.replace("ses-", "")) for s, ss in defaults
            ]
            console.print(
                f"[dim]Using default sub/ses from {spec.name} spec "
                f"({len(pairs)} combinations)[/dim]",
            )
        else:
            console.print("[red]Error:[/red] Provide --subseslist or --subseslist-file")
            raise typer.Exit(code=1)

    if not pairs:
        console.print("[red]Error:[/red] No sub/ses pairs.")
        raise typer.Exit(code=1)

    console.print(
        f"\n[bold]{spec.name.upper()} integrity check[/bold] — {spec.description}"
    )
    console.print(f"Dir: {analysis_dir}")
    console.print(f"Sessions: {len(pairs)}\n")

    if max_workers:
        results = run_integrity_check_parallel(
            spec,
            analysis_dir,
            pairs,
            max_workers,
            check_corrupted,
        )
    else:
        results = run_integrity_check_single(spec, analysis_dir, pairs, check_corrupted)

    print_summary(results, spec)
    if show_distribution:
        print_group_distribution(results)
    if debug:
        print_all_groups(results, spec)
    if verbose:
        print_detailed_results(results, verbose)

    output_dir.mkdir(parents=True, exist_ok=True)

    brief_path = output_dir / f"{spec.name}_subses_summary.txt"
    detail_path = output_dir / f"{spec.name}_detailed.log"
    brief_df = write_brief_csv(results, brief_path)
    write_detailed_log(results, spec, detail_path)
    write_matrix_from_result(results, output_dir / f"{spec.name}_matrix_detailed.csv")
    write_matrix_from_brief_csv(brief_df, output_dir / f"{spec.name}_matrix_simple.csv")

    # additional summary for prfanalyze to break down incomplete by task
    summarize_incomplete_by_dimension(results, spec, output_dir)

    # [DEV] Add new conditional output writers here (copy this pattern)
    if any(r.total_corrupted > 0 for r in results):
        p = output_dir / f"{spec.name}_corrupted.txt"
        write_corrupted_list(results, p, spec, analysis_dir)
        console.print(f"[bold]Corrupted list:[/bold] {p}")

    if any(r.total_timing_issues > 0 for r in results):
        p = output_dir / f"{spec.name}_timing_mismatch.txt"
        write_time_mismatch_list(results, p, spec, analysis_dir)
        console.print(f"[bold]Timing mismatch list:[/bold] {p}")

    if any(r.total_log_issues > 0 for r in results):
        p = output_dir / f"{spec.name}_log_incomplete.txt"
        write_log_issues_list(results, p, spec, analysis_dir)
        console.print(f"[bold]Log incomplete list:[/bold] {p}")

    if any(r.total_extra_groups > 0 for r in results):
        p = output_dir / f"{spec.name}_extra_runs.txt"
        write_extra_runs_list(results, p, spec, analysis_dir)
        console.print(f"[bold]Extra runs:[/bold] {p}")

    console.print(f"\n[bold]Brief CSV:[/bold]    {brief_path}")
    console.print(f"[bold]Detailed log:[/bold] {detail_path}")


def main():
    app()


if __name__ == "__main__":
    main()
