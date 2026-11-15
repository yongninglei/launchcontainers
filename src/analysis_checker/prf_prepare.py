"""
analysis_checker/prf_prepare.py
================================
PRF prepare output spec.

Expected structure:
    analysis_dir / sub-XX / ses-XX / func / <files>

─── Groups per session (8 total) ────────────────────────────────────────────
  maskinfo-L          all left-hemi ROI maskinfo JSONs   (fixed list)
  maskinfo-R          all right-hemi ROI maskinfo JSONs  (fixed list)
  bold-{task}         run-01 / run-02 / run-0102avg × hemi-L / hemi-R  (3 groups)
  events-{task}       run-01 / run-02 only                              (3 groups)

─── Task categories (must not be mixed) ─────────────────────────────────────
  without-fix:  retFF  retRW  retCB
  with-fix:     retfixFF  retfixRW  retfixRWblock  retfixRWblock01  retfixRWblock02

  Exactly 3 tasks expected, all from the same category.
  If tasks from both categories are found → flagged as a mix error.

─── To change expected outputs ──────────────────────────────────────────────
[DEV] Edit MASKINFO_SUFFIXES_HEMI_L / _HEMI_R to update ROI lists.
[DEV] Edit TASKS_NO_FIX / TASKS_FIX to update valid task name sets.
[DEV] Edit BOLD_RUNS / EVENTS_RUNS / BOLD_HEMIS to change run/hemi structure.
"""

from __future__ import annotations

import re
from pathlib import Path

try:
    from .base import AnalysisSpec, default_combinations
except ImportError:
    import sys
    import os

    sys.path.insert(0, os.path.dirname(__file__))
    from base import AnalysisSpec, default_combinations


# ── Valid task name sets ──────────────────────────────────────────────────────
# [DEV] Add new task variants here
TASKS_NO_FIX: set[str] = {"retFF", "retRW", "retCB"}
TASKS_FIX: set[str] = {
    "retfixFF",
    "retfixRW",
    "retfixRWblock",
    "retfixRWblock01",
    "retfixRWblock02",
}
ALL_VALID_TASKS: set[str] = TASKS_NO_FIX | TASKS_FIX


class PRFPrepareSpec(AnalysisSpec):
    # ── Maskinfo ROI lists ────────────────────────────────────────────────────
    # [DEV] Add / remove ROI entries here as atlases change
    MASKINFO_SUFFIXES_HEMI_L: list[str] = [
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

    MASKINFO_SUFFIXES_HEMI_R: list[str] = [
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

    # ── Run / hemi structure ──────────────────────────────────────────────────
    # [DEV] Edit these to change which runs / hemis are expected
    BOLD_RUNS: list[str] = ["01", "02", "0102avg"]  # all 3 for bold nii.gz
    EVENTS_RUNS: list[str] = ["01", "02"]  # only real runs for events
    BOLD_HEMIS: list[str] = ["hemi-L", "hemi-R"]

    # ── Expected number of tasks ──────────────────────────────────────────────
    # [DEV] Change if your protocol uses a different number of tasks
    EXPECTED_N_TASKS: int = 3

    # [DEV] Sessions that only acquired retRW + retCB (no retFF) → 2 tasks expected.
    # Same set as BIDSSpec.REDUCED_RET_SESSIONS in bids.py.
    REDUCED_RET_SESSIONS: set[tuple[str, str]] = {
        ("sub-03", "ses-01"),
        ("sub-04", "ses-01"),
        ("sub-06", "ses-01"),
    }

    @property
    def name(self) -> str:
        return "prfprepare"

    @property
    def description(self) -> str:
        n_mask = len(self.MASKINFO_SUFFIXES_HEMI_L) + len(self.MASKINFO_SUFFIXES_HEMI_R)
        return (
            f"PRF prepare — {n_mask} maskinfo (L+R) + "
            f"{self.EXPECTED_N_TASKS} tasks × "
            f"({len(self.BOLD_RUNS)} runs × {len(self.BOLD_HEMIS)} hemis bold "
            f"+ {len(self.EVENTS_RUNS)} runs events)"
        )

    def get_session_dir(self, analysis_dir: Path, sub: str, ses: str) -> Path:
        return analysis_dir / sub / ses / "func"

    # ── Task discovery ────────────────────────────────────────────────────────

    def _expected_n_tasks(self, sub: str, ses: str) -> int:
        """Return the expected task count for this sub/ses.

        Reduced sessions (sub-03/04/06 ses-01) only acquired retRW + retCB,
        so only 2 tasks are expected instead of the usual 3.
        """
        sub_str = sub if sub.startswith("sub-") else f"sub-{sub}"
        ses_str = ses if ses.startswith("ses-") else f"ses-{ses}"
        if (sub_str, ses_str) in self.REDUCED_RET_SESSIONS:
            return 2
        return self.EXPECTED_N_TASKS

    def _discover_tasks(
        self,
        session_dir: Path,
        prefix: str,
        sub: str,
        ses: str,
    ) -> tuple[set[str], set[str], list[str]]:
        """
        Scan func/ directory to find which tasks exist.

        Returns
        -------
        bold_tasks  : set of task names that have at least one bold .nii.gz
        event_tasks : set of task names that have at least one events .tsv
        mix_errors  : list of human-readable error strings if fix/nofix mixed
        """
        bold_tasks: set[str] = set()
        event_tasks: set[str] = set()
        task_pattern = re.compile(re.escape(prefix) + r"_task-([^_]+)_run-")

        for f in session_dir.iterdir():
            if not f.is_file():
                continue
            m = task_pattern.match(f.name)
            if not m:
                continue
            task = m.group(1)
            if task not in ALL_VALID_TASKS:
                continue  # ignore unknown tasks
            if f.name.endswith("_bold.nii.gz"):
                bold_tasks.add(task)
            elif f.name.endswith("_events.tsv"):
                event_tasks.add(task)
        all_tasks = bold_tasks | event_tasks
        # ── Fix / no-fix mix check ────────────────────────────────────────────
        mix_errors: list[str] = []
        has_nofix = bool(all_tasks & TASKS_NO_FIX)
        has_fix = bool(all_tasks & TASKS_FIX)
        if has_nofix and has_fix:
            mix_errors.append(
                f"Mixed fix/no-fix tasks detected — "
                f"no-fix: {sorted(all_tasks & TASKS_NO_FIX)}, "
                f"fix: {sorted(all_tasks & TASKS_FIX)}. "
                f"All tasks must be from the same category."
            )

        # ── Wrong task count check ────────────────────────────────────────────
        expected_n = self._expected_n_tasks(sub, ses)
        if len(bold_tasks) != expected_n and len(bold_tasks) > 0:
            mix_errors.append(
                f"Expected {expected_n} tasks, "
                f"found {len(bold_tasks)}: {sorted(bold_tasks)}"
            )

        return bold_tasks, event_tasks, mix_errors

    # ── Group construction ────────────────────────────────────────────────────

    def get_expected_groups(self, session_dir: Path) -> dict[str, list[str]]:
        sub = session_dir.parent.parent.name
        ses = session_dir.parent.name
        prefix = f"{sub}_{ses}"
        groups: dict[str, list[str]] = {}

        # 1. Maskinfo (always fixed)
        groups["maskinfo-L"] = [f"{prefix}{s}" for s in self.MASKINFO_SUFFIXES_HEMI_L]
        groups["maskinfo-R"] = [f"{prefix}{s}" for s in self.MASKINFO_SUFFIXES_HEMI_R]

        if not session_dir.is_dir():
            return groups

        bold_tasks, event_tasks, mix_errors = self._discover_tasks(
            session_dir, prefix, sub, ses
        )

        # 2. Bold groups
        for task in sorted(bold_tasks):
            groups[f"bold-{task}"] = [
                f"{prefix}_task-{task}_run-{run}_{hemi}_bold.nii.gz"
                for run in self.BOLD_RUNS
                for hemi in self.BOLD_HEMIS
            ]

        # 3. Events groups
        for task in sorted(event_tasks):
            groups[f"events-{task}"] = [
                f"{prefix}_task-{task}_run-{run}_events.tsv" for run in self.EVENTS_RUNS
            ]

        # 4. Validate group count — expected 2 maskinfo + N_tasks × 2 (bold + events)
        expected_n_groups = 2 + self._expected_n_tasks(sub, ses) * 2
        if len(groups) != expected_n_groups:
            mix_errors.append(
                f"Expected {expected_n_groups} groups total, "
                f"got {len(groups)}: {list(groups.keys())}"
            )

        # 5. Surface all errors as a validation group
        # Each error string becomes a "missing file" in the report
        if mix_errors:
            groups["[validation-errors]"] = mix_errors

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
