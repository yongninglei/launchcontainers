# analysis_checker/base.py
# All shared ABCs and utilities — no imports from within analysis_checker
from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path

from launchcontainers.utils import hms_to_sec, parse_hms, times_match  # noqa: F401


# =============================================================================
# 1. ABSTRACT ANALYSIS SPEC
# =============================================================================


class AnalysisSpec(ABC):
    """
    Abstract base class defining what an analysis type expects.

    Subclass this in analysis_checker/<name>.py and implement the abstract methods.
    The engine never needs to be modified when adding a new analysis type —
    just add the subclass and register it in analysis_checker/__init__.py.

    ─── How groups work ─────────────────────────────────────────────────────
    "Groups" are the logical units files are organised by. Examples:
      prfanalyze  → one group per run prefix  (prefix × hemi × suffix)
      prfprepare  → maskinfo | events-{task} | bold-{task}
      bids        → one group per modality subfolder (anat / func / fmap / dwi)
      fmriprep    → one group per modality subfolder (anat / func)

    ─── How to add a new check type ─────────────────────────────────────────
    [DEV] To add a new analysis type:
      1. Create  analysis_checker/<yourtype>.py  subclassing AnalysisSpec
      2. Implement all @abstractmethod methods (see below)
      3. Register in analysis_checker/__init__.py  SPEC_REGISTRY dict
      That's it — the engine picks it up automatically.

    ─── How to add a new file-level check ───────────────────────────────────
    [DEV] To add a new category of file-level error (beyond missing/corrupted/timing):
      1. Add a new list field to GroupResult  (e.g. truncated_files)
      2. Add a corresponding total_* property to SessionResult
      3. Add a check_broken_<ext>() helper in section 3 if needed
      4. Wire the check into check_one_session() in section 3
      5. Surface the new field in print_summary(), print_detailed_results(),
         and write_detailed_log() in sections 4–5
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'prfanalyze', 'bids'."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable one-liner shown in report headers."""
        ...

    @property
    def uses_glob(self) -> bool:
        """
        Set True in subclass if get_expected_groups() returns glob patterns.
        The engine will use Path.glob() instead of exact-name lookup.
        """
        return False

    @abstractmethod
    def get_session_dir(self, analysis_dir: Path, sub: str, ses: str) -> Path:
        """Return the root directory to scan for this sub/ses."""
        ...

    @abstractmethod
    def get_expected_groups(self, session_dir: Path) -> dict[str, list[str]]:
        """
        Discover file groups and return expected filenames for each group.

        Returns
        -------
        dict[str, list[str]]
            group_label → list of expected filenames relative to session_dir
        """
        ...

    def get_expected_subfolders(
        self,
        analysis_dir: Path,
        sub: str,
        ses: str,
    ) -> list[Path]:
        """
        Directories that must exist under the session dir.
        Default: only the session dir itself. Override if your analysis
        requires specific subdirectories (e.g. anat/, func/).
        """
        return [self.get_session_dir(analysis_dir, sub, ses)]

    def get_default_combinations(self) -> list[tuple[str, str]] | None:
        """
        Default (sub, ses) pairs used when no --subseslist is given.
        Return None to require explicit input from the user.
        """
        return None

    def get_group_dimension(self, group_label: str) -> tuple[str, str] | None:
        """
        Extract (dimension_name, dimension_value) from a group label for
        detailed summary. Return None to skip dimensional summary.

        Examples:
          prfanalyze: 'sub-01_ses-01_task-ret_run-01' -> ('task', 'ret')
          bidsdwi:    'acq-nordic'                    -> ('acq', 'nordic')
        """
        return None


# move these helpers here too
# =============================================================================
# 6. SHARED CONSTANTS
# =============================================================================

EXCLUDED_SESSIONS: set[tuple[str, str]] = {
    ("10", "02"),
    ("09", "01"),
    ("10", "05"),
    ("06", "10"),
    ("05", "08"),
    ("08", "04"),
}
# Path to the WC subseslist — two levels up from analysis_checker/
_WC_SUBSESLIST_PATH = (
    Path(__file__).parent.parent / "launchcontainers" / "tests" / "wc_subseslist.txt"
)


def _load_wc_sessions() -> set[tuple[str, str]]:
    """Load WC (wide-column / retfix) sessions from wc_subseslist.txt."""
    wc: set[tuple[str, str]] = set()
    if not _WC_SUBSESLIST_PATH.exists():
        return wc
    try:
        with open(_WC_SUBSESLIST_PATH, newline="") as fh:
            for row in csv.DictReader(fh):
                sub = str(row["sub"]).strip().zfill(2)
                ses = str(row["ses"]).strip().zfill(2)
                wc.add((sub, ses))
    except Exception:
        pass
    return wc


# Loaded once at import time.
WC_SESSIONS: set[tuple[str, str]] = _load_wc_sessions()


def default_combinations() -> list[tuple[str, str]]:
    """All 11 subs × 10 sessions, minus any excluded pairs."""
    return [
        (f"{sub:02d}", f"{ses:02d}")
        for sub in range(1, 12)
        for ses in range(1, 11)
        if (f"{sub:02d}", f"{ses:02d}") not in EXCLUDED_SESSIONS
    ]


# ── Shared time-parsing utilities (used by timing-aware specs) ────────────────


def read_json(p: Path) -> dict:
    """Safely read a JSON file; returns {} on any error."""
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def json_for(nii: Path) -> Path:
    """Return the JSON sidecar path for a .nii.gz file."""
    return nii.with_name(nii.name.replace(".nii.gz", ".json"))
