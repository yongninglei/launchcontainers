"""
analysis_specs/rtp.py
======================
RTP retinotopy analysis output spec.

[DEV] Edit EXPECTED_SUFFIXES to match your pipeline outputs.
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

# Last line expected in RTP_log.txt when the pipeline completes successfully.
_RTP_LOG_EXIT_SIGNAL = "Sending exit(0) signal."


class RTPSpec(AnalysisSpec):
    # [DEV] Change expected output file suffixes here
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
        sub = session_dir.parent.name
        ses = session_dir.name
        return {
            "rtp_outputs": [f"{sub}_{ses}_{s}" for s in self.EXPECTED_SUFFIXES],
        }

    def get_default_combinations(self) -> list[tuple[str, str]]:
        return default_combinations()


class RTP2PipelineSpec(AnalysisSpec):
    """
    Checker for rtp2-pipeline output completeness.

    Verifies two things per sub/ses:
      1. ``output/log/RTP_log.txt`` exists.
      2. The last non-empty line of that log is ``Sending exit(0) signal.``

    Usage example
    -------------
    python check_analysis_integrity.py \\
        /BIDS/derivatives/rtp2-pipeline-0.2.3_3.0.4/analysis-120_RTR_4TOI \\
        rtppipeline
    """

    _LOG_SUBPATH = Path("output") / "log" / "RTP_log.txt"

    @property
    def name(self) -> str:
        return "rtppipeline"

    @property
    def description(self) -> str:
        return "rtp2-pipeline run completion (checks RTP_log.txt exit signal)"

    def get_session_dir(self, analysis_dir: Path, sub: str, ses: str) -> Path:
        return analysis_dir / sub / ses

    def get_expected_subfolders(
        self, analysis_dir: Path, sub: str, ses: str
    ) -> list[Path]:
        session_dir = self.get_session_dir(analysis_dir, sub, ses)
        return [
            session_dir,
            session_dir / "output",
            session_dir / "output" / "log",
        ]

    def get_expected_groups(self, session_dir: Path) -> dict[str, list[str]]:
        return {
            "pipeline_log": [str(self._LOG_SUBPATH)],
        }

    def _check_log_completion(self, session_dir: Path, group_label: str) -> str | None:
        """
        Duck-typed hook called by the engine after file-existence checks.

        Returns a non-empty string describing the problem if the log did not
        end with the expected exit signal, or None if it looks complete.
        """
        if group_label != "pipeline_log":
            return None

        log_path = session_dir / self._LOG_SUBPATH
        try:
            lines = log_path.read_text(errors="replace").splitlines()
        except OSError as exc:
            return f"could not read log: {exc}"

        non_empty = [ln.rstrip() for ln in lines if ln.strip()]
        if not non_empty:
            return "log file is empty"

        last = non_empty[-1]
        if last != _RTP_LOG_EXIT_SIGNAL:
            preview = last[:120] + ("…" if len(last) > 120 else "")
            return f"unexpected last line: {preview!r}"

        return None

    def get_default_combinations(self) -> list[tuple[str, str]]:
        return default_combinations()
