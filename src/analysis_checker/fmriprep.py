"""
analysis_specs/fmriprep.py
===========================
fMRIPrep derivatives spec.

Expected structure:
    analysis_dir / sub-XX / ses-XX / {anat, func, figures} / <files>

[DEV] Edit MODALITY_PATTERNS to add/change expected output files.
"""

from __future__ import annotations

from pathlib import Path

try:
    from .base import AnalysisSpec, default_combinations
except ImportError:
    import sys
    import os

    sys.path.insert(0, os.path.dirname(__file__))
    from base import AnalysisSpec, default_combinations


class FMRIPrepSpec(AnalysisSpec):
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
        self,
        analysis_dir: Path,
        sub: str,
        ses: str,
    ) -> list[Path]:
        session = analysis_dir / sub / ses
        return [session / mod for mod in self.MODALITY_PATTERNS] + [session / "figures"]

    def get_expected_groups(self, session_dir: Path) -> dict[str, list[str]]:
        sub = session_dir.parent.name
        ses = session_dir.name
        return {
            modality: [
                f"{modality}/{pat.replace('{sub}', sub).replace('{ses}', ses)}"
                for pat in patterns
            ]
            for modality, patterns in self.MODALITY_PATTERNS.items()
        }

    def get_default_combinations(self) -> list[tuple[str, str]]:
        return default_combinations()
