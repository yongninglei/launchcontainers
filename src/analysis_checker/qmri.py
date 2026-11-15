"""
analysis_checker/qmri.py
========================
qMRI (MRF) spec:
  BIDSQMRISpec  — MRF outputs under BIDS anat/sub-XX_ses-XX/

─── To change expected files ────────────────────────────────────────────────
[DEV] Edit MRF_SUFFIXES, FIXED_FILES, or DAT_ACQ in BIDSQMRISpec.
      No changes needed in the engine or registry.
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


class BIDSQMRISpec(AnalysisSpec):
    """
    qMRI MRF outputs under BIDS anat:
      analysis_dir / sub-XX / ses-XX / anat / sub-XX_ses-XX / <MRF files>

    The MRF prefix (e.g. MRF_X4g_LONG_HIGHSNR_534) varies per session and is
    discovered from files on disk using known anchor suffixes. All listed files
    must be present.

    [DEV] Adjust MRF_SUFFIXES / FIXED_FILES / DAT_ACQ to match your protocol.
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
        self,
        analysis_dir: Path,
        sub: str,
        ses: str,
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
