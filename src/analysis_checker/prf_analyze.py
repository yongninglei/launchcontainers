"""
analysis_specs/prf_analyze.py
==============================
PRF analyze output spec.

Expected structure:
    analysis_dir / sub-XX / ses-XX / {run_prefix}_{hemi}_{suffix}

Groups: one per run prefix (discovered from files on disk).
Each group = prefix × hemi × suffix.
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict

try:
    from .base import AnalysisSpec, default_combinations
except ImportError:
    import sys
    import os

    sys.path.insert(0, os.path.dirname(__file__))
    from base import AnalysisSpec, default_combinations


class PRFAnalyzeSpec(AnalysisSpec):
    """
    PRF analyze results: analysis_dir / sub-XX / ses-XX / <files>

    Files grouped by run prefix (discovered from files on disk).
    Each run × hemi expects a fixed set of output suffixes.

    ─── To change expected outputs ──────────────────────────────────────────
    [DEV] Edit HEMIS or EXPECTED_SUFFIXES below.
          No changes needed in the engine or registry.
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
    VALID_RUNS: list[str] = ["run-01", "run-02", "run-0102avg"]
    EXPECTED_N_TASKS: int = 3
    EXPECTED_N_GROUPS: int = 9  # 3 tasks × 3 runs (01, 02, 0102avg)

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

        # Discover all prefixes and group by task
        task_runs: dict[str, set[str]] = defaultdict(set)
        for f in session_dir.iterdir():
            if f.is_file() and "_hemi-" in f.name:
                prefix = f.name.split("_hemi-")[0]
                parts = prefix.split("_")
                task = next((p for p in parts if p.startswith("task-")), None)
                run = next((p for p in parts if p.startswith("run-")), None)
                if task and run:
                    task_runs[task].add(run)

        validation_errors: list[str] = []
        groups: dict[str, list[str]] = {}

        sub = session_dir.parent.name
        ses = session_dir.name
        prefix_base = f"{sub}_{ses}"

        for task in sorted(task_runs):
            runs = task_runs[task]

            # Flag unexpected run names
            unexpected = sorted(runs - set(self.VALID_RUNS))
            if unexpected:
                validation_errors.append(
                    f"Task {task} has unexpected runs: {unexpected}. "
                    f"Valid runs are: {self.VALID_RUNS}"
                )

            # Always build groups for ALL valid runs so missing ones surface as missing files
            for run in self.VALID_RUNS:
                group_label = f"{prefix_base}_{task}_{run}"
                groups[group_label] = [
                    f"{prefix_base}_{task}_{run}_{hemi}_{suffix}"
                    for hemi in self.HEMIS
                    for suffix in self.EXPECTED_SUFFIXES
                ]

        # Flag wrong total group count (expected exactly 3 tasks × 3 runs = 9)
        expected_n_groups = self.EXPECTED_N_TASKS * len(self.VALID_RUNS)  # 9
        if len(groups) != expected_n_groups:
            validation_errors.append(
                f"Expected {expected_n_groups} groups total "
                f"({self.EXPECTED_N_TASKS} tasks × {len(self.VALID_RUNS)} runs), "
                f"got {len(groups)} — tasks found: {sorted(task_runs.keys())}"
            )

        if validation_errors:
            groups["[validation-errors]"] = validation_errors

        return groups

    def get_group_dimension(self, group_label: str) -> tuple[str, str] | None:
        parts = group_label.split("_")
        task = next(
            (p.split("task-")[-1] for p in parts if p.startswith("task-")), None
        )
        if task:
            return ("task", task)
        return None

    def get_default_combinations(self) -> list[tuple[str, str]]:
        return default_combinations()
