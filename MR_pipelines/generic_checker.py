#!/usr/bin/env python3
"""
Check Analysis Integrity (Generic)
===================================
Validates completeness of neuroimaging analysis folders by comparing actual
files against expected file specs defined in AnalysisSpec subclasses.

Each analysis type subclasses AnalysisSpec and defines:
  - Folder structure (get_session_dir, get_expected_subfolders)
  - File grouping logic (get_expected_groups)
  - Default sub/ses combinations (get_default_combinations)

The checker engine reads these specs generically — no analysis-specific logic
in the engine.

Usage:
    python check_analysis_integrity.py /path/to/analysis prfprepare
    python check_analysis_integrity.py /path/to/analysis prfanalyze -s 01,01 -s 01,02
    python check_analysis_integrity.py /path/to/analysis prfanalyze --subseslist-file subseslist.csv
    python check_analysis_integrity.py /path/to/analysis bids --verbose

Output:
    1. <type>_incomplete.csv  — brief sub,ses,RUN for every sub/ses
    2. <type>_detailed.log    — indexed detailed log of missing files
"""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
import pandas as pd

import typer
from rich.console import Console
from rich.table import Table
import scipy.io
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed

app = typer.Typer(help="Check analysis folder integrity against expected file specs.")
console = Console()


# =============================================================================
# 1. ABSTRACT ANALYSIS SPEC
# =============================================================================


class AnalysisSpec(ABC):
    """
    Abstract base class defining what an analysis type expects.

    Subclass for each analysis type. The checker engine reads everything
    from these methods — zero special-casing in the engine.

    "Groups" are the logical units files are organized by:
      - prfanalyze:  run prefix  (files = prefix × hemi × suffix)
      - prfprepare:  maskinfo | events | bold  (each has own expected file list)
      - bids:        modality subfolder (anat, func, fmap, dwi)
      - fmriprep:    modality subfolder (anat, func)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name, e.g. 'prfanalyze', 'prfprepare'."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""
        ...

    @property
    def uses_glob(self) -> bool:
        """If True, file patterns may contain * or ? and will be glob-matched."""
        return False

    @abstractmethod
    def get_session_dir(self, analysis_dir: Path, sub: str, ses: str) -> Path:
        """Return the root directory to scan for this sub/ses."""
        ...

    @abstractmethod
    def get_expected_groups(self, session_dir: Path) -> dict[str, list[str]]:
        """
        Discover file groups and return expected files for each.

        Returns
        -------
        dict[str, list[str]]
            group_label -> list of expected file paths relative to session_dir
        """
        ...

    def get_expected_subfolders(
        self, analysis_dir: Path, sub: str, ses: str
    ) -> list[Path]:
        """Directories that must exist. Default: just the session dir."""
        return [self.get_session_dir(analysis_dir, sub, ses)]

    def get_default_combinations(self) -> list[tuple[str, str]] | None:
        """Default sub/ses list if no --subseslist given. None = require input."""
        return None


# =============================================================================
# 2. RESULT DATA CLASSES
# =============================================================================


@dataclass
class GroupResult:
    """Result for one file group."""

    group_label: str
    expected_files: list[str]
    found_files: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    # [DEV] appending different type of errors below
    corrupted_files: list[str] = field(default_factory=list)
    timing_issues: list[str] = field(default_factory=list)

    # [DEV] appending different type of errors below
    @property
    def is_complete(self) -> bool:
        return (
            len(self.missing_files) == 0
            and len(self.corrupted_files) == 0
            and len(self.timing_issues) == 0
        )


@dataclass
class SessionResult:
    """Result for one sub/ses pair."""

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
    def is_complete(self) -> bool:
        return (
            self.session_dir_exists
            and len(self.missing_folders) == 0
            and self.num_groups > 0
            and all(g.is_complete for g in self.groups.values())
        )

    @property
    def total_missing(self) -> int:
        return sum(len(g.missing_files) for g in self.groups.values())

    # [DEV] appending different type of errors below
    @property
    def total_corrupted(self) -> int:
        return sum(len(g.corrupted_files) for g in self.groups.values())

    @property
    def total_timing_issues(self) -> int:
        return sum(len(g.timing_issues) for g in self.groups.values())


# =============================================================================
# 3. GENERIC CHECKER ENGINE
# =============================================================================


def _check_file_exists(session_dir: Path, fname: str, use_glob: bool) -> bool:
    """Check if a file exists, with optional glob support."""
    if use_glob and ("*" in fname or "?" in fname):
        fpath = Path(fname)
        parent = session_dir / fpath.parent if str(fpath.parent) != "." else session_dir
        return len(list(parent.glob(fpath.name))) > 0 if parent.is_dir() else False
    else:
        return (session_dir / fname).exists()


# add the helper function to check matfile
def check_broken_mat(filepath: Path) -> tuple[bool, str]:
    """Try scipy (v5-v7.2) then h5py (v7.3). Returns (is_valid, error_msg)."""
    try:
        data = scipy.io.loadmat(str(filepath))
        # Force decompression — loadmat is lazy, corruption only surfaces on access
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


# add the helper function to check if json is broken
def check_broken_json(filepath: Path) -> tuple[bool, str]:
    """Check if a JSON file is valid and non-empty."""
    try:
        text = filepath.read_text()
        data = json.loads(text)
        if not data:
            return False, "empty JSON object"
        return True, ""
    except json.JSONDecodeError as e:
        return False, f"invalid JSON: {e}"
    except Exception as e:
        return False, str(e)


# add the helper function to check if nii.gz is broken
def check_broken_nii(filepath: Path) -> tuple[bool, str]:
    """Check if a .nii.gz has a valid header (fast) without loading all data."""
    try:
        import nibabel as nib

        img = nib.load(str(filepath))
        shape = img.header.get_data_shape()
        if len(shape) == 0:
            return False, "empty shape in header"
        # Read only first volume to test decompression without loading all data
        import numpy as np

        proxy = img.dataobj
        _ = np.asarray(proxy[..., 0]) if len(shape) == 4 else np.asarray(proxy)
        return True, ""
    except Exception as e:
        return False, str(e)


# helper function to read json
def read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


# helper function to get json sidercar for nii.gz
def json_for(nii: Path) -> Path:
    return nii.with_name(nii.name.replace(".nii.gz", ".json"))


# helper function to parse acq time from json or from scants.tsv rows
def parse_hms(ts: str) -> str:
    """
    Normalize any time string to zero-padded HH:MM:SS.
    Handles:
      '15:41:1.000000'             (no zero-padding, with sub-seconds)
      '9:05:38.297500'             (single-digit hour)
      '2025-06-19T15:41:01.212500' (ISO with sub-seconds)
      '2025-01-28T13:57:43'        (ISO without sub-seconds)
      '13:57:43'                   (plain HH:MM:SS)

    The output will be:
    plain HH:MM:SS, 17:50:30
    """
    s = str(ts).strip()
    if "T" in s:
        s = s.split("T")[1]
    s = s.split(".")[0]
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(s, fmt).strftime("%H:%M:%S")
        except ValueError:
            continue
    return s


# time match check
def times_match(t1: str, t2: str, max_diff_sec: int = 30) -> bool:
    """
    Compare two BIDS-style AcquisitionTime strings.

    Each SBRef should have been acquired just before its corresponding bold
    run (typically a few seconds to ~1 minute apart). This function checks
    that the absolute difference between two acquisition times is within an
    acceptable window, used to verify sbref↔bold pairing.

    Parameters
    ----------
    t1 : str
        AcquisitionTime string from the sbref JSON, e.g. "14:04:15.487500"
    t2 : str
        AcquisitionTime string from the bold JSON,  e.g. "14:04:29.487500"
    max_diff_sec : int
        Maximum allowed difference in seconds to consider the sbref and bold
        as a matched pair. Default is 30 s; raise to ~180 s if your protocol
        has a longer gap between SBRef and the start of the bold acquisition.

    Returns
    -------
    bool
        True if |t1 - t2| <= max_diff_sec, False otherwise.
    """
    if t1 is None or t2 is None:
        return False

    try:
        dt1 = datetime.strptime(parse_hms(t1), "%H:%M:%S")
        dt2 = datetime.strptime(parse_hms(t2), "%H:%M:%S")
    except ValueError:
        return False

    diff = abs((dt1 - dt2).total_seconds())
    return diff <= max_diff_sec


# check single session wrapper
def check_one_session(
    spec: AnalysisSpec,
    analysis_dir: Path,
    sub: str,
    ses: str,
    check_corruption: bool = False,
) -> SessionResult:
    """Check a single sub/ses against the spec."""
    session_dir = spec.get_session_dir(analysis_dir, sub, ses)
    result = SessionResult(
        sub=sub,
        ses=ses,
        session_dir_exists=session_dir.is_dir(),
    )

    # Check subfolders
    for folder in spec.get_expected_subfolders(analysis_dir, sub, ses):
        if not folder.is_dir():
            try:
                rel = str(folder.relative_to(analysis_dir))
            except ValueError:
                rel = str(folder)
            result.missing_folders.append(rel)

    if not result.session_dir_exists:
        return result

    # Check file groups
    for group_label, expected_files in spec.get_expected_groups(session_dir).items():
        found, missing, corrupted = [], [], []
        for fname in expected_files:
            fpath = session_dir / fname
            if _check_file_exists(session_dir, fname, spec.uses_glob):
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
                missing.append(fname)

        result.groups[group_label] = GroupResult(
            group_label=group_label,
            expected_files=expected_files,
            found_files=found,
            missing_files=missing,
            corrupted_files=corrupted,
        )
    # [DEV] appending different type of modality specific errors below
    # Timing check — only for specs that implement _check_timing
    if hasattr(spec, "_check_timing"):
        for group_label, gresult in result.groups.items():
            if gresult.missing_files:
                continue  # skip timing if files are already missing
            issue = spec._check_timing(session_dir, group_label)
            if issue:
                gresult.timing_issues.append(issue)

    return result


def run_integrity_check_single(
    spec: AnalysisSpec,
    analysis_dir: Path,
    subses_list: list[tuple[str, str]],
    check_corruption: bool = False,
) -> list[SessionResult]:
    """Run integrity check across all sub/ses pairs."""
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

    # Preserve original order in results
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
                    # Insert a failed placeholder so indexing stays intact
                    results[idx] = SessionResult(
                        sub=f"sub-{sub}", ses=f"ses-{ses}", session_dir_exists=False
                    )
                progress.update(1)

    return results  # type: ignore[return-value]


# =============================================================================
# 4. OUTPUT WRITERS
# =============================================================================


def write_brief_csv(results: list[SessionResult], output_path: Path) -> pd.DataFrame:
    """Write brief CSV: sub,ses,RUN — row index matches detailed log."""
    rows = []
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sub", "ses", "RUN"])
        for r in results:
            sub_num = r.sub.replace("sub-", "")
            ses_num = r.ses.replace("ses-", "")
            writer.writerow([sub_num, ses_num, r.is_complete])
            rows.append({"sub": sub_num, "ses": ses_num, "RUN": r.is_complete})

    return pd.DataFrame(rows, columns=["sub", "ses", "RUN"])


def write_matrix_from_brief_csv(
    source: "pd.DataFrame | Path",
    output_path: Path,
) -> None:
    """
    Write a pivot-table CSV from a brief CSV or DataFrame.

    Accepts either:
      - a DataFrame with columns [sub, ses, RUN]
      - a Path to a brief_csv file

    Rows = subjects, Columns = sessions.
    Values:
      1   = RUN=True
      0   = RUN=False
      ''  = not present in input
    """
    import pandas as pd

    # Load from file if path given
    if isinstance(source, Path):
        df = pd.read_csv(source, dtype=str)
    else:
        df = source.copy()
        df["sub"] = df["sub"].astype(str)
        df["ses"] = df["ses"].astype(str)
        df["RUN"] = df["RUN"].astype(str)

    # Preserve order of appearance
    subs = list(dict.fromkeys(df["sub"].tolist()))
    sessions = list(dict.fromkeys(df["ses"].tolist()))

    # Build lookup (sub, ses) -> value
    lookup: dict[tuple[str, str], str] = {}
    for _, row in df.iterrows():
        val = "1" if str(row["RUN"]).strip().lower() == "true" else "0"
        lookup[(str(row["sub"]), str(row["ses"]))] = val

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + [f"ses-{s}" for s in sessions])
        for sub in subs:
            row = [f"sub-{sub}"]
            for ses in sessions:
                row.append(lookup.get((sub, ses), ""))
            writer.writerow(row)

    console.print(f"[bold]Matrix CSV:[/bold]   {output_path}")


# [DEV] appending different type of errors below
def write_matrix_from_result(results: list[SessionResult], output_path: Path) -> None:
    """
    Write a pivot-table CSV for Google Sheets.

    Rows = subjects, Columns = sessions.
    Values:
      1   = complete (RUN=True)
      0.5 = incomplete but has data (RUN=False, missing files > 0)
      0   = no directory or no groups (RUN=False, missing = —)
    """
    # Collect all unique subs and sessions, preserving order
    subs_seen: dict[str, None] = {}
    ses_seen: dict[str, None] = {}
    lookup: dict[tuple[str, str], float] = {}

    for r in results:
        sub_num = r.sub.replace("sub-", "")
        ses_num = r.ses.replace("ses-", "")
        subs_seen[sub_num] = None
        ses_seen[ses_num] = None

        if r.is_complete:
            value = 1.0
        elif not r.session_dir_exists or r.num_groups == 0:
            value = 0.0
        else:
            # Has data but incomplete (missing files > 0)
            value = 0.5

        lookup[(sub_num, ses_num)] = value

    subs = list(subs_seen.keys())
    sessions = list(ses_seen.keys())

    def _fmt(v: float) -> str:
        if v == 1.0:
            return "1"
        elif v == 0.5:
            return "0.5"
        else:
            return "0"

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header: ,ses-01,ses-02,...  (empty first cell)
        writer.writerow([""] + [f"ses-{s}" for s in sessions])
        for sub in subs:
            row = [f"sub-{sub}"]
            for ses in sessions:
                val = lookup.get((sub, ses), "")
                row.append(_fmt(val) if val != "" else "")
            writer.writerow(row)


# [DEV] appending different type of errors below
def write_detailed_log(
    results: list[SessionResult],
    spec: AnalysisSpec,
    output_path: Path,
) -> None:
    """Write detailed log indexed [0], [1], ... matching CSV rows."""
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

            f.write(
                f"      Groups: {r.complete_groups}/{r.num_groups} complete  |  "
                f"Missing files: {r.total_missing}\n"
                f"Corrupted files: {r.total_corrupted}\n"
            )

            for glabel, gresult in sorted(r.groups.items()):
                if gresult.is_complete:
                    continue
                if gresult.missing_files:
                    f.write(
                        f"      [{glabel}] — {len(gresult.missing_files)} missing:\n"
                    )
                    for mf in gresult.missing_files:
                        f.write(f"        - {mf}\n")
                # [DEV] appending different type of errors below
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

            f.write("\n")


# add function to write corrupted file list only for review and remove later
def write_corrupted_list(
    results: list[SessionResult],
    output_path: Path,
    spec: AnalysisSpec,
    analysis_dir: Path,
) -> None:
    """Write full absolute paths of all corrupted files, one per line."""
    with open(output_path, "w") as f:
        for r in results:
            session_dir = spec.get_session_dir(analysis_dir, r.sub, r.ses)
            for gresult in r.groups.values():
                for cf in gresult.corrupted_files:
                    fname = cf.split(" (")[0]
                    f.write(f"{session_dir / fname}\n")


def write_time_mismatch_list(
    results: list[SessionResult],
    output_path: Path,
    spec: AnalysisSpec,
    analysis_dir: Path,
) -> None:
    """Write full absolute paths of all timing mismatched bold files, one per line."""
    with open(output_path, "w") as f:
        for r in results:
            session_dir = spec.get_session_dir(analysis_dir, r.sub, r.ses)
            for gresult in r.groups.values():
                if gresult.timing_issues:
                    # group_label is the bold filename for BIDSFuncSBRefSpec
                    bold_path = session_dir / gresult.group_label
                    for issue in gresult.timing_issues:
                        f.write(f"{bold_path}  {issue.strip()}\n")


# =============================================================================
# 5. RICH CONSOLE OUTPUT
# =============================================================================


# [DEV] appending different type of errors below
def print_summary(results: list[SessionResult], spec: AnalysisSpec) -> None:
    """Print summary table and statistics."""
    n_complete = sum(1 for r in results if r.is_complete)
    n_incomplete = sum(1 for r in results if not r.is_complete)
    n_no_dir = sum(1 for r in results if not r.session_dir_exists)
    n_no_groups = sum(1 for r in results if r.session_dir_exists and r.num_groups == 0)
    total_groups = sum(r.num_groups for r in results)
    complete_groups = sum(r.complete_groups for r in results)
    total_corrupted = sum(r.total_corrupted for r in results)

    table = Table(title=f"{spec.name.upper()} Integrity Summary")
    table.add_column("sub", style="cyan")
    table.add_column("ses", style="cyan")
    table.add_column("RUN", justify="center")
    table.add_column("Groups", justify="center")
    table.add_column("Missing", justify="center")
    # [DEV] appending different type of errors below
    table.add_column("Corrupted", justify="center")
    table.add_column("Timing", justify="center")
    total_timing = sum(r.total_timing_issues for r in results)

    for r in results:
        style = "green" if r.is_complete else "red bold"

        if not r.session_dir_exists:
            groups_str = "no dir"
            missing_str = "—"
            # [DEV] appending different type of errors below
            corrupted_str = "—"
            timing_str = "—"
        elif r.num_groups == 0:
            groups_str = "0"
            missing_str = "—"
            # [DEV] appending different type of errors below
            corrupted_str = "—"
            timing_str = "—"
        else:
            groups_str = f"{r.complete_groups}/{r.num_groups}"
            missing_str = str(r.total_missing) if r.total_missing > 0 else "0"
            # [DEV] appending different type of errors below
            corrupted_str = (
                f"[red bold]{r.total_corrupted}[/red bold]"
                if r.total_corrupted > 0
                else "0"
            )
            timing_str = (
                f"[yellow bold]{r.total_timing_issues}[/yellow bold]"
                if r.total_timing_issues > 0
                else "0"
            )
        table.add_row(
            r.sub.replace("sub-", ""),
            r.ses.replace("ses-", ""),
            f"[{style}]{r.is_complete}[/{style}]",
            groups_str,
            missing_str,
            # [DEV] appending different type of errors below
            corrupted_str,
            timing_str,
        )

    console.print(table)
    console.print(
        f"\n[green]{n_complete} complete[/green] | "
        f"[red]{n_incomplete} incomplete[/red] "
        f"out of {len(results)} sessions"
    )
    if total_groups > 0:
        console.print(f"Groups: {complete_groups}/{total_groups} complete overall")
    # [DEV] appending different type of errors below
    if total_corrupted > 0:
        console.print(f"[red bold]Corrupted .mat files: {total_corrupted}[/red bold]")
    if total_timing > 0:
        console.print(f"[yellow bold]Timing issues: {total_timing}[/yellow bold]")
    if n_no_dir > 0:
        console.print(f"[yellow]Missing session dirs: {n_no_dir}[/yellow]")
    if n_no_groups > 0:
        console.print(f"[yellow]Empty session dirs: {n_no_groups}[/yellow]")


# [DEV] appending different type of errors below
def print_detailed_results(results: list[SessionResult], verbose: bool = False) -> None:
    """Print missing file details for incomplete sessions."""
    incomplete = [r for r in results if not r.is_complete]
    if not incomplete:
        console.print("\n[bold green]✓ All sessions complete![/bold green]")
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
            f"{r.total_missing} files missing, "
            f"{r.total_corrupted} corrupted)"
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
            # [DEV] appending different type of errors below
            if gresult.corrupted_files:
                console.print(
                    f"    [magenta]⚠ {glabel}[/magenta] — {len(gresult.corrupted_files)} corrupted:"
                )
                for cf in gresult.corrupted_files:
                    console.print(f"      - {cf}")
            if gresult.timing_issues:
                console.print(f"    [yellow]⏱ {glabel}[/yellow] — timing mismatch:")
                for ti in gresult.timing_issues:
                    console.print(f"      [yellow]{ti}[/yellow]")
    console.print()


def print_group_distribution(results: list[SessionResult]) -> None:
    """Print distribution of groups per session."""
    counts = defaultdict(int)
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


# =============================================================================
# 6. SHARED CONSTANTS — Excluded sessions used by multiple specs
# =============================================================================

EXCLUDED_SESSIONS: set[tuple[str, str]] = {}


def default_combinations() -> list[tuple[str, str]]:
    """All 11 subs × 10 sessions minus excluded."""
    combos = []
    for sub_id in range(1, 12):
        for ses_id in range(1, 11):
            sub = f"{sub_id:02d}"
            ses = f"{ses_id:02d}"
            if (sub, ses) not in EXCLUDED_SESSIONS:
                combos.append((sub, ses))
    return combos


# =============================================================================
# 7. ANALYSIS SPEC: PRFAnalyze (was "prf")
# =============================================================================


class PRFAnalyzeSpec(AnalysisSpec):
    """
    PRF analyze results: analysis_dir / sub-XX / ses-XX / <files>

    Files grouped by run prefix (discovered from files on disk).
    Each run × hemi expects a fixed set of output suffixes.
    """

    HEMIS = ["hemi-L", "hemi-R"]

    EXPECTED_SUFFIXES = [
        "centerx0.nii.gz",
        "centery0.nii.gz",
        "estimates.json",
        "modelpred.nii.gz",
        "r2.nii.gz",
        "results.mat",
        "sigmamajor.nii.gz",
        "sigmaminor.nii.gz",
        "testdata.nii.gz",
        "theta.nii.gz",
    ]

    @property
    def name(self) -> str:
        return "prfanalyze"

    @property
    def description(self) -> str:
        return (
            f"PRF analyze outputs — {len(self.EXPECTED_SUFFIXES)} files "
            f"× {len(self.HEMIS)} hemis per run"
        )

    def get_session_dir(self, analysis_dir: Path, sub: str, ses: str) -> Path:
        return analysis_dir / sub / ses

    def get_expected_groups(self, session_dir: Path) -> dict[str, list[str]]:
        """Discover run prefixes, build expected = prefix × hemi × suffix."""
        if not session_dir.is_dir():
            return {}

        prefixes: set[str] = set()
        for f in session_dir.iterdir():
            if f.is_file() and "_hemi-" in f.name:
                prefixes.add(f.name.split("_hemi-")[0])

        groups: dict[str, list[str]] = {}
        for prefix in sorted(prefixes):
            expected = []
            for hemi in self.HEMIS:
                for suffix in self.EXPECTED_SUFFIXES:
                    expected.append(f"{prefix}_{hemi}_{suffix}")
            groups[prefix] = expected

        return groups

    def get_default_combinations(self) -> list[tuple[str, str]]:
        return default_combinations()


# =============================================================================
# 8. ANALYSIS SPEC: PRFPrepare
# =============================================================================


class PRFPrepareSpec(AnalysisSpec):
    """
    PRF prepare outputs: analysis_dir / sub-XX / ses-XX / <files>

    Three groups of files, each with different structure:
      1. maskinfo: per-hemi ROI mask JSON files (hemi-L and hemi-R have
         DIFFERENT ROI names due to lh./rh. prefix on VOTC parcels)
      2. events:   per-task per-run event TSVs (no hemi dimension)
      3. bold:     per-task per-run per-hemi surface BOLD (only "fix" tasks)

    All files are prefixed with {sub}_{ses} on disk.
    """

    # -- Maskinfo: explicit per-hemi suffix lists (ROI names differ!) --
    MASKINFO_SUFFIXES_HEMI_L = [
        "_hemi-L_desc-hFEF-wang_maskinfo.json",
        "_hemi-L_desc-hV4-benson_maskinfo.json",
        "_hemi-L_desc-hV4-wang_maskinfo.json",
        "_hemi-L_desc-IPS0-wang_maskinfo.json",
        "_hemi-L_desc-IPS1-wang_maskinfo.json",
        "_hemi-L_desc-IPS2-wang_maskinfo.json",
        "_hemi-L_desc-IPS3-wang_maskinfo.json",
        "_hemi-L_desc-IPS4-wang_maskinfo.json",
        "_hemi-L_desc-IPS5-wang_maskinfo.json",
        "_hemi-L_desc-lh.G_oc-temp_lat-fusifor-prf_VOTC_maskinfo.json",
        "_hemi-L_desc-lh.G_temporal_inf-prf_VOTC_maskinfo.json",
        "_hemi-L_desc-lh.S_oc-temp_lat-prf_VOTC_maskinfo.json",
        "_hemi-L_desc-LO1-benson_maskinfo.json",
        "_hemi-L_desc-LO1-wang_maskinfo.json",
        "_hemi-L_desc-LO2-benson_maskinfo.json",
        "_hemi-L_desc-LO2-wang_maskinfo.json",
        "_hemi-L_desc-PHC1-wang_maskinfo.json",
        "_hemi-L_desc-PHC2-wang_maskinfo.json",
        "_hemi-L_desc-SPL1-wang_maskinfo.json",
        "_hemi-L_desc-TO1-benson_maskinfo.json",
        "_hemi-L_desc-TO1-wang_maskinfo.json",
        "_hemi-L_desc-TO2-benson_maskinfo.json",
        "_hemi-L_desc-TO2-wang_maskinfo.json",
        "_hemi-L_desc-V1-benson_maskinfo.json",
        "_hemi-L_desc-V1d-wang_maskinfo.json",
        "_hemi-L_desc-V1v-wang_maskinfo.json",
        "_hemi-L_desc-V2-benson_maskinfo.json",
        "_hemi-L_desc-V2d-wang_maskinfo.json",
        "_hemi-L_desc-V2v-wang_maskinfo.json",
        "_hemi-L_desc-V3a-benson_maskinfo.json",
        "_hemi-L_desc-V3a-wang_maskinfo.json",
        "_hemi-L_desc-V3b-benson_maskinfo.json",
        "_hemi-L_desc-V3-benson_maskinfo.json",
        "_hemi-L_desc-V3b-wang_maskinfo.json",
        "_hemi-L_desc-V3d-wang_maskinfo.json",
        "_hemi-L_desc-V3v-wang_maskinfo.json",
        "_hemi-L_desc-VO1-benson_maskinfo.json",
        "_hemi-L_desc-VO1-wang_maskinfo.json",
        "_hemi-L_desc-VO2-benson_maskinfo.json",
        "_hemi-L_desc-VO2-wang_maskinfo.json",
    ]

    MASKINFO_SUFFIXES_HEMI_R = [
        "_hemi-R_desc-G_oc-temp_lat-fusifor-prf_VOTC_maskinfo.json",
        "_hemi-R_desc-G_temporal_inf-prf_VOTC_maskinfo.json",
        "_hemi-R_desc-hFEF-wang_maskinfo.json",
        "_hemi-R_desc-hV4-benson_maskinfo.json",
        "_hemi-R_desc-hV4-wang_maskinfo.json",
        "_hemi-R_desc-IPS0-wang_maskinfo.json",
        "_hemi-R_desc-IPS1-wang_maskinfo.json",
        "_hemi-R_desc-IPS2-wang_maskinfo.json",
        "_hemi-R_desc-IPS3-wang_maskinfo.json",
        "_hemi-R_desc-IPS4-wang_maskinfo.json",
        "_hemi-R_desc-IPS5-wang_maskinfo.json",
        "_hemi-R_desc-LO1-benson_maskinfo.json",
        "_hemi-R_desc-LO1-wang_maskinfo.json",
        "_hemi-R_desc-LO2-benson_maskinfo.json",
        "_hemi-R_desc-LO2-wang_maskinfo.json",
        "_hemi-R_desc-PHC1-wang_maskinfo.json",
        "_hemi-R_desc-PHC2-wang_maskinfo.json",
        "_hemi-R_desc-S_oc-temp_lat-prf_VOTC_maskinfo.json",
        "_hemi-R_desc-SPL1-wang_maskinfo.json",
        "_hemi-R_desc-TO1-benson_maskinfo.json",
        "_hemi-R_desc-TO1-wang_maskinfo.json",
        "_hemi-R_desc-TO2-benson_maskinfo.json",
        "_hemi-R_desc-TO2-wang_maskinfo.json",
        "_hemi-R_desc-V1-benson_maskinfo.json",
        "_hemi-R_desc-V1d-wang_maskinfo.json",
        "_hemi-R_desc-V1v-wang_maskinfo.json",
        "_hemi-R_desc-V2-benson_maskinfo.json",
        "_hemi-R_desc-V2d-wang_maskinfo.json",
        "_hemi-R_desc-V2v-wang_maskinfo.json",
        "_hemi-R_desc-V3a-benson_maskinfo.json",
        "_hemi-R_desc-V3a-wang_maskinfo.json",
        "_hemi-R_desc-V3b-benson_maskinfo.json",
        "_hemi-R_desc-V3-benson_maskinfo.json",
        "_hemi-R_desc-V3b-wang_maskinfo.json",
        "_hemi-R_desc-V3d-wang_maskinfo.json",
        "_hemi-R_desc-V3v-wang_maskinfo.json",
        "_hemi-R_desc-VO1-benson_maskinfo.json",
        "_hemi-R_desc-VO1-wang_maskinfo.json",
        "_hemi-R_desc-VO2-benson_maskinfo.json",
        "_hemi-R_desc-VO2-wang_maskinfo.json",
    ]

    # -- expected runs --
    EXPECTED_RUNS = ["01", "02", "0102avg"]

    # -- hemi for bold files --
    BOLD_HEMIS = ["hemi-L", "hemi-R"]

    @property
    def name(self) -> str:
        return "prfprepare"

    @property
    def description(self) -> str:
        n_mask = len(self.MASKINFO_SUFFIXES_HEMI_L) + len(self.MASKINFO_SUFFIXES_HEMI_R)
        return (
            f"PRF prepare outputs — {n_mask} maskinfo (fixed) + "
            f"events & bold per discovered task"
        )

    def get_session_dir(self, analysis_dir: Path, sub: str, ses: str) -> Path:
        return analysis_dir / sub / ses / "func"

    def _discover_tasks(
        self, session_dir: Path, prefix: str
    ) -> tuple[set[str], set[str]]:
        """
        Scan directory to discover which tasks exist.

        Returns
        -------
        event_tasks : set[str]
            Tasks that have at least one _events.tsv file.
        bold_tasks : set[str]
            Tasks that have at least one _bold.nii.gz file.
        """
        import re

        event_tasks: set[str] = set()
        bold_tasks: set[str] = set()

        task_pattern = re.compile(re.escape(prefix) + r"_task-([^_]+)_run-")

        for f in session_dir.iterdir():
            if not f.is_file():
                continue
            m = task_pattern.match(f.name)
            if not m:
                continue
            task = m.group(1)
            if f.name.endswith("_events.tsv"):
                event_tasks.add(task)
            elif f.name.endswith("_bold.nii.gz"):
                bold_tasks.add(task)

        return event_tasks, bold_tasks

    def get_expected_groups(self, session_dir: Path) -> dict[str, list[str]]:
        """
        Build expected files:
          maskinfo:           fixed list (80 files, hemi-specific ROI names)
          events-{task}:      discovered per task, expects 3 runs
          bold-{task}:        discovered per task, expects 3 runs × 2 hemis
        """
        # sub and ses are going with session_dir, if session_dir is func/ then parent is ses and grandparent is sub
        sub = session_dir.parent.parent.name
        ses = session_dir.parent.name
        prefix = f"{sub}_{ses}"

        groups: dict[str, list[str]] = {}

        # --- Maskinfo group (always fixed) ---
        maskinfo_files = []
        for suffix in self.MASKINFO_SUFFIXES_HEMI_L:
            maskinfo_files.append(f"{prefix}{suffix}")
        for suffix in self.MASKINFO_SUFFIXES_HEMI_R:
            maskinfo_files.append(f"{prefix}{suffix}")
        groups["maskinfo"] = maskinfo_files

        # --- Discover tasks from files on disk ---
        event_tasks, bold_tasks = self._discover_tasks(session_dir, prefix)

        # --- Events: one group per discovered event task ---
        for task in sorted(event_tasks):
            expected = []
            for run in self.EXPECTED_RUNS:
                expected.append(f"{prefix}_task-{task}_run-{run}_events.tsv")
            groups[f"events-{task}"] = expected

        # --- Bold: one group per discovered bold task ---
        for task in sorted(bold_tasks):
            expected = []
            for run in self.EXPECTED_RUNS:
                for hemi in self.BOLD_HEMIS:
                    expected.append(
                        f"{prefix}_task-{task}_run-{run}_{hemi}_bold.nii.gz"
                    )
            groups[f"bold-{task}"] = expected

        return groups

    def get_default_combinations(self) -> list[tuple[str, str]]:
        return default_combinations()


# =============================================================================
# 9. ANALYSIS SPEC: BIDS
# =============================================================================


class BIDSSpec(AnalysisSpec):
    """
    Raw BIDS: analysis_dir / sub-XX / ses-XX / {anat,func,fmap,dwi} / <files>
    Files grouped by modality subfolder, using glob patterns.
    """

    MODALITY_PATTERNS: dict[str, list[str]] = {
        "anat": ["{sub}_{ses}_*T1w.nii.gz"],
        "func": [
            "{sub}_{ses}_task-*_bold.nii.gz",
            "{sub}_{ses}_task-*_bold.json",
            "{sub}_{ses}_task-*_events.tsv",
        ],
        "fmap": ["{sub}_{ses}_*epi.nii.gz"],
        "dwi": ["{sub}_{ses}_*dwi.nii.gz"],
    }

    @property
    def name(self) -> str:
        return "bids"

    @property
    def description(self) -> str:
        return "Raw BIDS dataset — anat, func, fmap, dwi"

    @property
    def uses_glob(self) -> bool:
        return True

    def get_session_dir(self, analysis_dir: Path, sub: str, ses: str) -> Path:
        return analysis_dir / sub / ses

    def get_expected_subfolders(
        self, analysis_dir: Path, sub: str, ses: str
    ) -> list[Path]:
        session = analysis_dir / sub / ses
        return [session / mod for mod in self.MODALITY_PATTERNS]

    def get_expected_groups(self, session_dir: Path) -> dict[str, list[str]]:
        # sub and ses are going with session_dir, if session_dir is func/ then parent is ses and grandparent is sub
        sub = session_dir.parent.name
        ses = session_dir.name
        groups: dict[str, list[str]] = {}
        for modality, patterns in self.MODALITY_PATTERNS.items():
            expected = []
            for pat in patterns:
                resolved = pat.replace("{sub}", sub).replace("{ses}", ses)
                expected.append(f"{modality}/{resolved}")
            groups[modality] = expected
        return groups


class BIDSDWISpec(AnalysisSpec):
    """
    BIDS DWI check: analysis_dir / sub-XX / ses-XX / dwi / <files>

    Discovers acq- labels from files on disk, then for each acq expects:
      dir-AP_run-01:  .bval, .bvec, .json, .nii.gz
      dir-PA_run-01:  .json, .nii.gz
    """

    AP_EXTENSIONS = [".bval", ".bvec", ".json", ".nii.gz"]
    PA_EXTENSIONS = [".json", ".nii.gz"]

    @property
    def name(self) -> str:
        return "bidsdwi"

    @property
    def description(self) -> str:
        return (
            f"BIDS DWI — per acq: "
            f"AP ({len(self.AP_EXTENSIONS)} files) + PA ({len(self.PA_EXTENSIONS)} files)"
        )

    def get_session_dir(self, analysis_dir: Path, sub: str, ses: str) -> Path:
        return analysis_dir / sub / ses / "dwi"

    def get_expected_subfolders(
        self, analysis_dir: Path, sub: str, ses: str
    ) -> list[Path]:
        return [analysis_dir / sub / ses / "dwi"]

    def get_expected_groups(self, session_dir: Path) -> dict[str, list[str]]:
        """
        Discover acq- labels from files, then build expected files per acq.
        session_dir = .../sub-XX/ses-XX/dwi/
        """
        import re

        if not session_dir.is_dir():
            return {}

        # sub/ses from path: dwi parent = ses, grandparent = sub
        sub = session_dir.parent.parent.name
        ses = session_dir.parent.name
        prefix = f"{sub}_{ses}"

        # Discover unique acq labels
        acq_pattern = re.compile(re.escape(prefix) + r"_acq-([^_]+)_dir-")
        acq_labels: set[str] = set()
        for f in session_dir.iterdir():
            if f.is_file():
                m = acq_pattern.match(f.name)
                if m:
                    acq_labels.add(m.group(1))

        # Build expected files per acq
        groups: dict[str, list[str]] = {}
        for acq in sorted(acq_labels):
            expected = []
            # AP
            for ext in self.AP_EXTENSIONS:
                expected.append(f"{prefix}_acq-{acq}_dir-AP_run-01_dwi{ext}")
            # PA
            for ext in self.PA_EXTENSIONS:
                expected.append(f"{prefix}_acq-{acq}_dir-PA_run-01_dwi{ext}")
            groups[f"acq-{acq}"] = expected

        return groups

    def get_default_combinations(self) -> list[tuple[str, str]]:
        return default_combinations()


class BIDSFuncSBRefSpec(AnalysisSpec):
    """
    BIDS func sbref↔bold pairing check.

    For every bold.nii.gz (non-symlink) in the func dir, checks:
      1. A matching sbref.nii.gz exists
      2. Both have a corresponding .json
      3. AcquisitionTime gap between sbref and bold is <= max_diff_sec
    """

    def __init__(self, max_diff_sec: int = 30):
        self.max_diff_sec = max_diff_sec

    @property
    def name(self) -> str:
        return "bidsfuncsbref"

    @property
    def description(self) -> str:
        return f"BIDS func sbref↔bold pairing — timing gap <= {self.max_diff_sec}s"

    def get_session_dir(self, analysis_dir: Path, sub: str, ses: str) -> Path:
        return analysis_dir / sub / ses / "func"

    def get_expected_subfolders(
        self, analysis_dir: Path, sub: str, ses: str
    ) -> list[Path]:
        return [self.get_session_dir(analysis_dir, sub, ses)]

    def get_expected_groups(self, session_dir: Path) -> dict[str, list[str]]:
        """
        Each bold.nii.gz (non-symlink) becomes a group.
        Expected files per group: [sbref.nii.gz, bold.json, sbref.json]
        Timing mismatch is encoded as a fake missing file entry.
        """
        if not session_dir.is_dir():
            return {}

        groups: dict[str, list[str]] = {}

        bold_files = sorted(
            f
            for f in session_dir.iterdir()
            if f.is_file()  # excludes symlinks
            and not f.is_symlink()
            and f.name.endswith("_bold.nii.gz")
        )

        for bold in bold_files:
            sbref = Path(bold.name.replace("_bold.nii.gz", "_sbref.nii.gz"))
            bold_json = Path(bold.name.replace(".nii.gz", ".json"))
            sbref_json = Path(sbref.name.replace(".nii.gz", ".json"))

            # These are the files that must exist — engine checks presence
            expected = [str(sbref), str(bold_json), str(sbref_json)]
            groups[bold.name] = expected

        return groups

    def _check_timing(self, session_dir: Path, bold_name: str) -> str | None:
        """
        Returns a diagnostic string if timing is mismatched, else None.
        Called separately from validate_timing_mismatches().
        """
        sbref_json_path = session_dir / bold_name.replace("_bold.nii.gz", "_sbref.json")
        bold_json_path = session_dir / bold_name.replace(".nii.gz", ".json")

        sbref_meta = read_json(sbref_json_path)
        bold_meta = read_json(bold_json_path)

        t_sbref = sbref_meta.get("AcquisitionTime")
        t_bold = bold_meta.get("AcquisitionTime")

        if t_sbref is None or t_bold is None:
            return f"  AcquisitionTime missing (sbref={t_sbref}, bold={t_bold})"

        if not times_match(t_sbref, t_bold, self.max_diff_sec):
            try:
                dt1 = datetime.strptime(parse_hms(t_sbref), "%H:%M:%S")
                dt2 = datetime.strptime(parse_hms(t_bold), "%H:%M:%S")
                diff = abs((dt1 - dt2).total_seconds())
            except ValueError:
                diff = -1
            return (
                f"  timing mismatch: sbref={parse_hms(t_sbref)}, "
                f"bold={parse_hms(t_bold)}, gap={diff:.0f}s "
                f"(max={self.max_diff_sec}s)"
            )
        return None

    def get_default_combinations(self) -> list[tuple[str, str]]:
        return default_combinations()


class BIDSScanstsvSpec(AnalysisSpec):
    """
    BIDS scans.tsv ↔ file acquisition time consistency check.

    For each session, reads the scans.tsv and compares AcquisitionTime
    against the corresponding JSON sidecar for each file.

    Mapping rules:
      anat/*_T1w.nii.gz     → scans.tsv rows for _T1_inv1, _T1_inv2, _T1_uni
      anat/*_T2w.nii.gz     → scans.tsv row  for _T2w
      fmap/*_epi.nii.gz     → scans.tsv row  for same name (run single-digit)
      func/*_bold.nii.gz    → scans.tsv row  for _magnitude
      func/*_sbref.nii.gz   → scans.tsv row  for _sbref
    """

    def __init__(self, max_diff_sec: int = 30):
        self.max_diff_sec = max_diff_sec

    @property
    def name(self) -> str:
        return "bidsscantsv"

    @property
    def description(self) -> str:
        return (
            f"BIDS scans.tsv ↔ JSON AcquisitionTime consistency "
            f"(T1w, T2w, fmap, bold, sbref) — gap <= {self.max_diff_sec}s"
        )

    def get_session_dir(self, analysis_dir: Path, sub: str, ses: str) -> Path:
        sub_str = f"sub-{sub}" if not sub.startswith("sub-") else sub
        ses_str = f"ses-{ses}" if not ses.startswith("ses-") else ses
        return analysis_dir / sub_str / ses_str

    def get_expected_subfolders(
        self, analysis_dir: Path, sub: str, ses: str
    ) -> list[Path]:
        return [self.get_session_dir(analysis_dir, sub, ses)]

    # ------------------------------------------------------------------
    # scans.tsv reader
    # ------------------------------------------------------------------

    def _read_scans_tsv(self, session_dir: Path) -> dict[str, str]:
        """
        Read scans.tsv → {filename: AcquisitionTime}.
        filename column is like 'func/sub-01_ses-01_task-ret_run-01_bold.nii.gz'
        Returns {} if file missing or malformed.
        """
        import csv as _csv

        tsv_files = list(session_dir.glob("*_scans.tsv"))
        if not tsv_files:
            return {}

        result: dict[str, str] = {}
        try:
            with open(tsv_files[0]) as f:
                reader = _csv.DictReader(f, delimiter="\t")
                for row in reader:
                    fname = row.get("filename", "").strip()
                    acqtime = row.get("acq_time", "").strip()
                    if fname:
                        result[fname] = acqtime
        except Exception:
            pass
        return result

    # ------------------------------------------------------------------
    # Mapping helpers: BIDS file → expected scans.tsv filename(s)
    # ------------------------------------------------------------------

    def _tsv_keys_for(self, bids_file: Path, sub: str, ses: str) -> list[str]:
        """
        Given a BIDS nii.gz path, return the list of scans.tsv filename
        entries it should correspond to.

        Returns a list because T1w maps to 3 tsv rows (_inv1, _inv2, _uni).
        """
        name = bids_file.name
        modality = bids_file.parent.name  # 'anat', 'func', 'fmap'
        stem = name.replace(".nii.gz", "")

        # --- T1w: maps to inv1, inv2, uni ---
        if name.endswith("_T1w.nii.gz"):
            base = stem.replace("_T1w", "")
            return [
                f"anat/{base}_T1_inv1.nii.gz",
                f"anat/{base}_T1_inv2.nii.gz",
                f"anat/{base}_T1_uni.nii.gz",
            ]

        # --- T2w: direct match ---
        if name.endswith("_T2w.nii.gz"):
            return [f"anat/{name}"]

        # --- fmap epi: direct match, but run is single-digit in tsv ---
        if modality == "fmap" and name.endswith("_epi.nii.gz"):
            # normalise run-01 -> run-1 for tsv lookup
            import re

            normalised = re.sub(r"run-0*(\d+)", lambda m: f"run-{m.group(1)}", name)
            return [f"fmap/{normalised}"]

        # --- func bold: maps to _magnitude AND _phase ---
        if name.endswith("_bold.nii.gz"):
            mag = name.replace("_bold.nii.gz", "_magnitude.nii.gz")
            phase = name.replace("_bold.nii.gz", "_phase.nii.gz")
            return [f"func/{mag}", f"func/{phase}"]

        # --- func sbref: direct match ---
        if name.endswith("_sbref.nii.gz"):
            return [f"func/{name}"]

        return []

    # ------------------------------------------------------------------
    # Collect files to check
    # ------------------------------------------------------------------

    def _collect_bids_files(self, session_dir: Path) -> list[Path]:
        """
        Collect all non-symlink nii.gz files to check:
          anat: *_T1w.nii.gz, *_T2w.nii.gz
          fmap: *_epi.nii.gz
          func: *_bold.nii.gz, *_sbref.nii.gz
        """
        patterns = {
            "anat": ["*_T1w.nii.gz", "*_T2w.nii.gz"],
            "fmap": ["*_epi.nii.gz"],
            "func": ["*_bold.nii.gz", "*_sbref.nii.gz"],
        }
        files = []
        for modality, globs in patterns.items():
            mod_dir = session_dir / modality
            if not mod_dir.is_dir():
                continue
            for pattern in globs:
                for f in sorted(mod_dir.glob(pattern)):
                    if not f.is_symlink():
                        files.append(f)
        return files

    # ------------------------------------------------------------------
    # Core: get_expected_groups
    # Each BIDS file = one group; expected = tsv key(s) that must exist
    # Timing issues populated separately via _check_timing
    # ------------------------------------------------------------------

    @property
    def uses_glob(self) -> bool:
        return False

    def get_expected_groups(self, session_dir: Path) -> dict[str, list[str]]:
        if not session_dir.is_dir():
            return {}

        sub = session_dir.parent.name
        ses = session_dir.name
        tsv = self._read_scans_tsv(session_dir)

        if not tsv:
            return {"scans.tsv": ["*_scans.tsv"]}

        groups: dict[str, list[str]] = {}
        for bids_file in self._collect_bids_files(session_dir):
            rel = str(bids_file.relative_to(session_dir))
            tsv_keys = self._tsv_keys_for(bids_file, sub, ses)
            # Pre-resolve: only missing tsv keys become "expected" for the engine
            # Complete groups get empty list → engine marks them complete immediately
            missing_in_tsv = [k for k in tsv_keys if k not in tsv]
            groups[rel] = missing_in_tsv

        return groups

    # ------------------------------------------------------------------
    # Timing check (called by check_one_session via hasattr duck-typing)
    # ------------------------------------------------------------------

    def _check_timing(self, session_dir: Path, group_label: str) -> str | None:
        sub = session_dir.parent.name
        ses = session_dir.name
        tsv = self._read_scans_tsv(session_dir)

        bids_file = session_dir / group_label
        if not bids_file.exists():
            return None

        json_path = json_for(bids_file)
        json_meta = read_json(json_path) if json_path.exists() else {}
        t_json_raw = json_meta.get("AcquisitionTime")

        # Normalise to HH:MM:SS — strips date if ISO format, handles sub-seconds
        t_json = parse_hms(t_json_raw) if t_json_raw else None

        tsv_keys = self._tsv_keys_for(bids_file, sub, ses)

        issues = []
        for key in tsv_keys:
            t_tsv_raw = tsv.get(key)
            if t_tsv_raw is None:
                continue  # missing from tsv — already caught as missing file

            # Normalise tsv time the same way — tsv acq_time often has date
            t_tsv = parse_hms(t_tsv_raw) if t_tsv_raw else None

            if not times_match(t_json, t_tsv, self.max_diff_sec):
                try:
                    dt1 = datetime.strptime(t_json, "%H:%M:%S")
                    dt2 = datetime.strptime(t_tsv, "%H:%M:%S")
                    diff = abs((dt1 - dt2).total_seconds())
                except Exception:
                    diff = -1
                issues.append(
                    f"  {key}: "
                    f"json={t_json if t_json else 'missing'}, "
                    f"tsv={t_tsv if t_tsv else 'missing'}, "
                    f"gap={diff:.0f}s"
                )

        return "\n".join(issues) if issues else None

    def get_default_combinations(self) -> list[tuple[str, str]]:
        return default_combinations()


# =============================================================================
# 10. ANALYSIS SPEC: BIDS qMRI (MRF)
# =============================================================================


class BIDSQMRISpec(AnalysisSpec):
    """
    qMRI MRF outputs under BIDS anat:
      analysis_dir / sub-XX / ses-XX / anat / sub-XX_ses-XX / <MRF files>

    The MRF prefix (e.g. MRF_X4g_LONG_HIGHSNR_534) varies per session and is
    discovered from files on disk using known anchor suffixes. All files must
    be present.
    """

    # Suffixes appended to the per-session MRF prefix
    MRF_SUFFIXES: list[str] = [
        "_B0B1.mat",
        "_B1cor_PDmap.nii",
        "_B1cor_sync_DIR.nii",
        "_B1cor_sync_T1mprage.nii",
        "_B1cor_sync_T2flair.nii",
        "_B1cor_sync_T2W.nii",
        "_B1cor_T1map.nii",
        "_B1cor_T2map.nii",
        "_coeff.mat",
        "_navi.mat",
        "_PDmap.nii",
        "_T1map.nii",
        "_T1T2P_B1cor.mat",
        "_T1T2P.mat",
        "_T2map.nii",
    ]

    # Fixed filenames independent of the MRF prefix
    FIXED_FILES: list[str] = [
        "my_diary.txt",
        "myinfo.mat",
        "myscript_recon_v16.m",
    ]

    # acq label used in the subject/session-specific .dat filename
    DAT_ACQ: str = "MRFxa60"

    # Anchor suffixes used to discover the MRF prefix from whatever is on disk
    _PREFIX_ANCHORS: list[str] = ["_T1map.nii", "_T2map.nii", "_PDmap.nii", "_B0B1.mat"]

    @property
    def name(self) -> str:
        return "qmri"

    @property
    def description(self) -> str:
        n = len(self.MRF_SUFFIXES) + len(self.FIXED_FILES) + 1
        return f"qMRI MRF outputs — {n} files under anat/sub-XX_ses-XX/"

    def get_session_dir(self, analysis_dir: Path, sub: str, ses: str) -> Path:
        return analysis_dir / sub / ses / "anat" / f"{sub}_{ses}"

    def get_expected_subfolders(
        self, analysis_dir: Path, sub: str, ses: str
    ) -> list[Path]:
        return [
            analysis_dir / sub / ses / "anat",
            self.get_session_dir(analysis_dir, sub, ses),
        ]

    def _discover_mrf_prefix(self, session_dir: Path) -> str | None:
        """Return the MRF prefix by matching any known anchor suffix on disk."""
        for f in session_dir.iterdir():
            if not f.is_file():
                continue
            for anchor in self._PREFIX_ANCHORS:
                if f.name.endswith(anchor):
                    return f.name[: -len(anchor)]
        return None

    def get_expected_groups(self, session_dir: Path) -> dict[str, list[str]]:
        if not session_dir.is_dir():
            return {}

        # Path layout: .../sub-XX/ses-XX/anat/sub-XX_ses-XX
        ses = session_dir.parent.parent.name  # ses-XX
        sub = session_dir.parent.parent.parent.name  # sub-XX

        prefix = self._discover_mrf_prefix(session_dir)
        if prefix is None:
            # Directory exists but no MRF anchor file found — report as missing
            return {"mrf": ["<MRF_prefix>_T1map.nii"]}

        mrf_files = [f"{prefix}{s}" for s in self.MRF_SUFFIXES]
        dat_file = f"{sub}_{ses}_acq-{self.DAT_ACQ}_run-01.dat"

        return {"mrf": mrf_files + self.FIXED_FILES + [dat_file]}

    def get_default_combinations(self) -> list[tuple[str, str]]:
        return default_combinations()


# =============================================================================
# 11. ANALYSIS SPEC: fMRIPrep
# =============================================================================
class FMRIPrepSpec(AnalysisSpec):
    """fMRIPrep derivatives: analysis_dir / sub-XX / ses-XX / {anat,func,figures}"""

    MODALITY_PATTERNS: dict[str, list[str]] = {
        "anat": [
            "{sub}_{ses}_*desc-preproc_T1w.nii.gz",
            "{sub}_{ses}_*desc-brain_mask.nii.gz",
        ],
        "func": [
            "{sub}_{ses}_task-*_space-*_desc-preproc_bold.nii.gz",
            "{sub}_{ses}_task-*_desc-confounds_timeseries.tsv",
        ],
    }

    @property
    def name(self) -> str:
        return "fmriprep"

    @property
    def description(self) -> str:
        return "fMRIPrep preprocessed outputs"

    @property
    def uses_glob(self) -> bool:
        return True

    def get_session_dir(self, analysis_dir: Path, sub: str, ses: str) -> Path:
        return analysis_dir / sub / ses

    def get_expected_subfolders(
        self, analysis_dir: Path, sub: str, ses: str
    ) -> list[Path]:
        session = analysis_dir / sub / ses
        return [session / mod for mod in self.MODALITY_PATTERNS] + [session / "figures"]

    def get_expected_groups(self, session_dir: Path) -> dict[str, list[str]]:
        # sub and ses are going with session_dir, if session_dir is func/ then parent is ses and grandparent is sub
        sub = session_dir.parent.name
        ses = session_dir.name
        groups: dict[str, list[str]] = {}
        for modality, patterns in self.MODALITY_PATTERNS.items():
            expected = []
            for pat in patterns:
                resolved = pat.replace("{sub}", sub).replace("{ses}", ses)
                expected.append(f"{modality}/{resolved}")
            groups[modality] = expected
        return groups


# =============================================================================
# 12. ANALYSIS SPEC: GLM, RTP — placeholders
# =============================================================================


class GLMSpec(AnalysisSpec):
    """GLM analysis — customize EXPECTED_SUFFIXES for your pipeline."""

    EXPECTED_contrasts = [
        "AllvsNull",
        "PERvsNull",
        "LEXvsNull",
        "PERvsLEX",
        "RWvsLEX",
        "RWvsPER",
        "RWvsNull",
        "RWvsLEXPER",
        "RWvsAllnoWordnoLEX",
        "RWvsAllnoWord",
        "LEXvsAllnoLEXnoRW",
        "LEXvsAllnoLEXnoRWnoPER",
        "PERvsAllnoPERnoRW",
        "PERvsAllnoPERnoRWnoLEX",
        "CSvsFF",
        "FFvsCS",
        "FacesvsNull",
        "FacesvsLEX",
        "FacesvsPER",
        "FacesvsLEXPER",
        "FacesvsAllnoFace",
        "LimbsvsNull",
        "LimbsvsLEX",
        "LimbsvsPER",
        "LimbsvsLEXPER",
        "LimbsvsAllnoLimbs",
        "RWvsCS",
        "RWvsFF",
        "RWvsFace",
        "RWvsLimbs",
        "RWvsSD",
        "FacesvsLimbs",
    ]
    EXPECTED_SUFFIXES = [
        "stat-*.func.gii",  # placeholder — edit for your pipeline
    ]

    @property
    def name(self) -> str:
        return "glm"

    @property
    def description(self) -> str:
        return "GLM statistical analysis outputs"

    @property
    def uses_glob(self) -> bool:
        return True

    def get_session_dir(self, analysis_dir: Path, sub: str, ses: str) -> Path:
        return analysis_dir / sub / ses

    def get_expected_groups(self, session_dir: Path) -> dict[str, list[str]]:
        # sub and ses are going with session_dir, if session_dir is func/ then parent is ses and grandparent is sub
        sub = session_dir.parent.name
        ses = session_dir.name
        return {
            "glm_outputs": [
                f"{sub}_{ses}_*{c}_{s}"
                for c in self.EXPECTED_contrasts
                for s in self.EXPECTED_SUFFIXES
            ]
        }

    def get_default_combinations(self) -> list[tuple[str, str]]:
        return default_combinations()


class RTPSpec(AnalysisSpec):
    """RTP analysis — customize for your pipeline."""

    EXPECTED_SUFFIXES = ["rtp_*.nii.gz"]

    @property
    def name(self) -> str:
        return "rtp"

    @property
    def description(self) -> str:
        return "RTP retinotopy analysis outputs"

    @property
    def uses_glob(self) -> bool:
        return True

    def get_session_dir(self, analysis_dir: Path, sub: str, ses: str) -> Path:
        return analysis_dir / sub / ses

    def get_expected_groups(self, session_dir: Path) -> dict[str, list[str]]:
        # sub and ses are going with session_dir, if session_dir is func/ then parent is ses and grandparent is sub
        sub = session_dir.parent.name
        ses = session_dir.name
        return {"rtp_outputs": [f"{sub}_{ses}_{s}" for s in self.EXPECTED_SUFFIXES]}


# =============================================================================
# 13. SPEC REGISTRY
# =============================================================================
# [DEV] append here new modality
SPEC_REGISTRY: dict[str, AnalysisSpec] = {
    "prfprepare": PRFPrepareSpec(),
    "prfanalyze": PRFAnalyzeSpec(),
    "bids": BIDSSpec(),
    "bidsdwi": BIDSDWISpec(),
    "bidsfuncsbref": BIDSFuncSBRefSpec(),
    "bidsscantsv": BIDSScanstsvSpec(),
    "qmri": BIDSQMRISpec(),
    "fmriprep": FMRIPrepSpec(),
    "glm": GLMSpec(),
    "rtp": RTPSpec(),
}


# =============================================================================
# 14. CLI
# =============================================================================


def parse_subses(raw: str) -> tuple[str, str]:
    """Parse '01,02' or 'sub-01,ses-02' into (sub_num, ses_num)."""
    parts = raw.strip().split(",")
    if len(parts) != 2:
        raise typer.BadParameter(f"Expected 'sub,ses' but got: '{raw}'")
    return parts[0].strip().replace("sub-", ""), parts[1].strip().replace("ses-", "")


def load_subseslist_from_file(filepath: Path) -> list[tuple[str, str]]:
    """Load sub/ses from CSV. Skips header/comment rows."""
    pairs = []
    with open(filepath) as f:
        for row in csv.reader(f):
            if not row or row[0].strip().lower() in ("sub", "subject", "#", ""):
                continue
            if len(row) >= 2:
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
        ...,
        help="Path to analysis root directory.",
        exists=True,
    ),
    analysis_type: str = typer.Argument(
        ...,
        help=f"Analysis type: {', '.join(SPEC_REGISTRY.keys())}",
    ),
    subseslist: Optional[list[str]] = typer.Option(
        None,
        "--subseslist",
        "-s",
        help="Sub,ses pairs: -s 01,02 -s 03,04",
    ),
    subseslist_file: Optional[Path] = typer.Option(
        None,
        "--subseslist-file",
        "-f",
        help="CSV with sub,ses columns.",
    ),
    output_dir: Path = typer.Option(
        Path("."),
        "--output-dir",
        "-o",
        help="Directory for output reports.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show complete groups too.",
    ),
    show_distribution: bool = typer.Option(
        True,
        "--show-distribution/--no-distribution",
        help="Show group count distribution.",
    ),
    max_workers: int = typer.Option(
        30, "--workers", "-j", help="Parallel workers for I/O (default: 30)."
    ),
    check_corrupted: bool = typer.Option(
        False, "--check-corrupted", help="Check for corrupted files (slow)."
    ),
) -> None:
    """Check analysis integrity against expected file/folder specs."""

    if analysis_type not in SPEC_REGISTRY:
        console.print(
            f"[red]Error:[/red] Unknown type '{analysis_type}'. "
            f"Valid: {list(SPEC_REGISTRY.keys())}"
        )
        raise typer.Exit(code=1)

    spec = SPEC_REGISTRY[analysis_type]

    # Resolve sub/ses list
    if subseslist:
        pairs = [parse_subses(s) for s in subseslist]
    elif subseslist_file:
        if not subseslist_file.exists():
            console.print(f"[red]Error:[/red] File not found: {subseslist_file}")
            raise typer.Exit(code=1)
        pairs = load_subseslist_from_file(subseslist_file)
    else:
        defaults = spec.get_default_combinations()  # default_combinations()
        if defaults:
            pairs = [
                (s.replace("sub-", ""), ss.replace("ses-", "")) for s, ss in defaults
            ]
            console.print(
                f"[dim]Using default sub/ses from {spec.name} spec "
                f"({len(pairs)} combinations)[/dim]"
            )
        else:
            console.print("[red]Error:[/red] Provide --subseslist or --subseslist-file")
            raise typer.Exit(code=1)

    if not pairs:
        console.print("[red]Error:[/red] No sub/ses pairs.")
        raise typer.Exit(code=1)

    # Run
    console.print(
        f"\n[bold]{spec.name.upper()} integrity check[/bold] — {spec.description}"
    )
    console.print(f"Dir: {analysis_dir}")
    console.print(f"Sessions: {len(pairs)}\n")

    if max_workers:
        results = run_integrity_check_parallel(
            spec, analysis_dir, pairs, max_workers, check_corrupted
        )
    else:
        results = run_integrity_check_single(spec, analysis_dir, pairs, check_corrupted)
    # Output
    print_summary(results, spec)
    if show_distribution:
        print_group_distribution(results)
    print_detailed_results(results, verbose)

    output_dir.mkdir(parents=True, exist_ok=True)
    brief_path = output_dir / f"{spec.name}_subses_summary.txt"
    detail_path = output_dir / f"{spec.name}_detailed.log"
    brief_df = write_brief_csv(results, brief_path)
    write_detailed_log(results, spec, detail_path)
    # [DEV] add more outputs here as needed, e.g. corrupted list, timing mismatch list, matrix CSV, etc.]
    # write the corrupted txt
    corrupted_path = output_dir / f"{spec.name}_corrupted.txt"
    # write the corrupted txt
    if any(r.total_corrupted > 0 for r in results):
        corrupted_path = output_dir / f"{spec.name}_corrupted.txt"
        write_corrupted_list(results, corrupted_path, spec, analysis_dir)
        console.print(f"[bold]Corrupted list:[/bold] {corrupted_path}")

    # write the timing mismatched txt
    if any(r.total_timing_issues > 0 for r in results):
        timing_path = output_dir / f"{spec.name}_timing_mismatch.txt"
        write_time_mismatch_list(results, timing_path, spec, analysis_dir)
        console.print(f"[bold]Timing mismatch list:[/bold] {timing_path}")

    matrix_path = output_dir / f"{spec.name}_matrix_detailed.csv"
    write_matrix_from_result(results, matrix_path)
    matrix_path_simple = output_dir / f"{spec.name}_matrix_simple.csv"
    write_matrix_from_brief_csv(brief_df, matrix_path_simple)

    console.print(f"\n[bold]Brief CSV:[/bold]    {brief_path}")
    console.print(f"[bold]Detailed log:[/bold] {detail_path}")
    console.print(f"[bold]Matrix CSV:[/bold]   {matrix_path}")


if __name__ == "__main__":
    app()
