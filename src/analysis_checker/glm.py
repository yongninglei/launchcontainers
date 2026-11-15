"""
analysis_specs/glm.py
======================
GLM analysis output spec.

[DEV] Edit EXPECTED_contrasts or EXPECTED_SUFFIXES to match your pipeline.
      Both support glob patterns (uses_glob = True).
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


class GLMSpec(AnalysisSpec):
    # [DEV] Add / remove contrast names here
    EXPECTED_CONTRASTS = [
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

    # [DEV] Change expected output file suffixes here
    EXPECTED_SUFFIXES = [
        "stat-*.func.gii",
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
        sub = session_dir.parent.name
        ses = session_dir.name
        return {
            "glm_outputs": [
                f"{sub}_{ses}_*{contrast}_{suffix}"
                for contrast in self.EXPECTED_CONTRASTS
                for suffix in self.EXPECTED_SUFFIXES
            ],
        }

    def get_default_combinations(self) -> list[tuple[str, str]]:
        return default_combinations()
