"""Analyses that run as their own programs: pyGLM, autoROI, swi-prep, … (docs/plan_protocol.md).

launchcontainers never imports them and they never import launchcontainers.
The rules for running each one live here: which program, which flags, which
extra work-list columns become flags. Everything else comes from the
program's single-session command line:

    <exe> --config <lc_config.yaml> --sub S --ses E [flags]          run one session
    <exe> --config <lc_config.yaml> [--sub S --ses E] [flags] --plan  JSON, computes nothing

The program is found per host, because its environment differs per machine::

    host_options:
      BCBL:
        manager: sge
        exe:
          pyGLM: /home/tlei/soft/MRI-hub/pyGLM/.venv/bin/pyglm
          autoROI-scaling: /home/tlei/soft/MRI-hub/autoROI/.venv/bin/autoroi-scaling

Without ``host_options.<host>.exe.<tool>`` the default program name is looked
up on PATH. ``host_options`` is runtime-only, so changing a path never trips
the config guard.

lc's part: prepare = ``--plan`` problems + the inputs recorded (path, size,
mtime) in ``<analysis_dir>/logs/inputs/sub-S_ses-E.tsv``; run = the run
command wrapped by ``lc-session``; qc = the launch only (the session's
outputs exist and are not empty; ``lc qc`` adds exit code and log checks).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from launchcontainers.core import LCError
from launchcontainers.core.config import LCConfig
from launchcontainers.core.io import atomic_write_text, bare
from launchcontainers.core.tool import QCCheck, QCResult, SessionResult, ToolConfig

from launchcontainers import __version__ as LC_VERSION

PLAN_TIMEOUT_S = 600
INPUTS_DIR = Path("logs") / "inputs"

ExtraArgs = Callable[[ToolConfig, str, str], Sequence[str]]


def _no_extra(tc: ToolConfig, sub: str, ses: str) -> Sequence[str]:
    return ()


class ExternalTool:
    """One analysis program driven through its CLI and ``--plan``."""

    version = LC_VERSION  # of these rules; the program's own version is in tc.extra

    def __init__(
        self, name: str, default_exe: str, extra_args: ExtraArgs = _no_extra
    ) -> None:
        self.name = name
        self.default_exe = default_exe
        self.extra_args = extra_args

    def __repr__(self) -> str:
        return f"ExternalTool({self.name!r}, {self.default_exe!r})"

    # -- the program ----------------------------------------------------------
    def _exe(self, cfg: LCConfig) -> str:
        opts = cfg.host_options() if cfg.host else {}
        exe = str((opts.get("exe") or {}).get(self.name) or self.default_exe)
        found = Path(exe).expanduser() if "/" in exe else shutil.which(exe)
        if found is None or not Path(found).is_file():
            raise LCError(
                f"{self.name}: program {exe!r} not found. Install {self.name} in its own "
                f"environment and set host_options.{cfg.host or '<host>'}.exe.{self.name} "
                f"to its command (e.g. /path/to/repo/.venv/bin/{self.default_exe})"
            )
        return str(found)

    def plan(
        self, tc: ToolConfig, sub: str | None = None, ses: str | None = None
    ) -> dict[str, Any]:
        """The program's ``--plan`` JSON (docs/plan_protocol.md)."""
        exe, config = tc.extra["exe"], tc.general.source_path
        cmd = [exe, "--config", str(config)]
        if sub is not None:
            cmd += ["--sub", sub, "--ses", ses, *self.extra_args(tc, sub, ses)]
        cmd.append("--plan")
        try:
            cp = subprocess.run(
                cmd, capture_output=True, text=True, timeout=PLAN_TIMEOUT_S
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise LCError(f"{self.name}: {' '.join(cmd)}: {exc}") from None
        if cp.returncode != 0:
            msg = (cp.stderr or cp.stdout).strip().splitlines()
            raise LCError(
                f"{self.name} --plan exited {cp.returncode}: "
                + (msg[-1] if msg else "no output")
            )
        try:
            plan = json.loads(cp.stdout)
        except json.JSONDecodeError as exc:
            raise LCError(f"{self.name} --plan did not print JSON: {exc}") from None
        if not isinstance(plan, dict) or "analysis_dirs" not in plan:
            raise LCError(f"{self.name} --plan: no 'analysis_dirs' in its output")
        return plan

    # -- Tool interface -------------------------------------------------------
    def load(self, cfg: LCConfig) -> ToolConfig:
        tc = ToolConfig(general=cfg, params=cfg.tool_block(self.name))
        tc.extra["exe"] = self._exe(cfg)
        plan = self.plan(tc)
        tc.extra["analysis_dirs"] = [Path(d) for d in plan["analysis_dirs"]]
        tc.extra["container_version"] = str(plan.get("version") or "unknown")
        src = plan.get("source_datasets") or {}
        if not isinstance(src, dict) or not all(
            isinstance(k, str) and isinstance(v, str) and Path(v).is_absolute()
            for k, v in src.items()
        ):
            raise LCError(
                f"{self.name} --plan: 'source_datasets' must map link names to "
                f"absolute paths, got {src!r}"
            )
        tc.extra["source_datasets"] = {k: Path(v) for k, v in src.items()}
        return tc

    def analysis_dirs(self, tc: ToolConfig) -> list[Path]:
        return list(tc.extra["analysis_dirs"])

    def source_datasets(self, tc: ToolConfig) -> dict[str, Path]:
        """Datasets the analysis reads besides raw BIDS (an upstream analysis, fMRIPrep, …)."""
        return dict(tc.extra.get("source_datasets") or {})

    def prepare(self, tc: ToolConfig, sub: str, ses: str) -> SessionResult:
        """Check the session's inputs (``--plan``) and record them. Raises nothing."""
        sub, ses = bare(sub, "sub"), bare(ses, "ses")
        try:
            plan = self.plan(tc, sub, ses)
        except LCError as exc:
            return SessionResult(sub, ses, "failed", [str(exc)])
        problems, notes = (
            list(plan.get("problems") or []),
            list(plan.get("notes") or []),
        )
        if problems:
            return SessionResult(sub, ses, "blocked", problems + notes)
        inputs = [Path(p) for p in plan.get("inputs") or []]
        text = "path\tsize\tmtime_ns\n" + "".join(
            f"{p}\t{p.stat().st_size}\t{p.stat().st_mtime_ns}\n"
            if p.exists()
            else f"{p}\tmissing\t\n"
            for p in inputs
        )
        for d in self.analysis_dirs(tc):
            atomic_write_text(d / INPUTS_DIR / f"sub-{sub}_ses-{ses}.tsv", text)
        return SessionResult(
            sub, ses, "prepared", notes + [f"{len(inputs)} input file(s) recorded"]
        )

    def command(self, tc: ToolConfig, sub: str, ses: str) -> list[str]:
        sub, ses = bare(sub, "sub"), bare(ses, "ses")
        return [
            tc.extra["exe"],
            "--config",
            str(tc.general.source_path),
            "--sub",
            sub,
            "--ses",
            ses,
            *self.extra_args(tc, sub, ses),
        ]

    def qc(self, tc: ToolConfig, sub: str, ses: str) -> QCResult:
        """Launch QC: every session output exists and is not empty."""
        sub, ses = bare(sub, "sub"), bare(ses, "ses")
        res = QCResult(sub, ses)
        try:
            outputs = [
                Path(p) for p in self.plan(tc, sub, ses).get("session_outputs") or []
            ]
        except LCError as exc:
            res.checks.append(QCCheck("outputs", "fail", str(exc)))
            return res
        if not outputs:
            res.checks.append(
                QCCheck("outputs", "warn", "the program lists no session outputs")
            )
        for out in outputs:
            empty = out.is_dir() and not any(out.iterdir())
            if not out.exists() or empty:
                res.checks.append(
                    QCCheck(
                        f"output {out.name}",
                        "fail",
                        f"{'empty' if empty else 'missing'}: {out}",
                    )
                )
            else:
                n = sum(1 for _ in out.rglob("*")) if out.is_dir() else 1
                res.checks.append(
                    QCCheck(f"output {out.name}", "ok", f"{n} file(s) in {out}")
                )
        return res


def _autoroi_sources(tc: ToolConfig, sub: str, ses: str) -> Sequence[str]:
    """Pseudo-session: the work list's ``sources`` column becomes ``--sources a,b,c``."""
    row = tc.row(sub, ses)
    raw = (row.extras.get("sources") or "") if row is not None else ""
    sources = [
        bare(s.strip(), "ses") for s in raw.replace(";", ",").split(",") if s.strip()
    ]
    return ["--sources", ",".join(sources)] if sources else []


PYGLM = ExternalTool("pyGLM", "pyglm")
AUTOROI_SCALING = ExternalTool("autoROI-scaling", "autoroi-scaling", _autoroi_sources)
AUTOROI_CLUSTERING = ExternalTool("autoROI-clustering", "autoroi-clustering")
# swi-preproc's raw-QC stage: echo average + rigid MC of every SWI run to run-01.
# Needs greedy in lc's environment (cajal: module load vtk/9.1 itk/5.4.5 greedy/latest).
SWI_PREP = ExternalTool("swi-prep", "swi-prep")
# eyetracker_analysis: EyeLink EDFs paired with the BOLD runs by clock time, gaze
# reconstructed, per-run eye QC verdict (pass / fail / unreliable / no_data) for the GLM.
EYETRACK = ExternalTool("eyetrack", "eyetrack")
# swi-preproc's main stage (Faruk Gulban's procedure): crop, 2x, MC at 2x, composite of
# the runs chosen at raw QC, N4, registration to the T1w. Needs c3d, greedy and ANTs on
# lc's PATH (cajal: module load vtk/9.1 itk/5.4.5 c3d/latest greedy/latest ants/2.5.1).
SWI_PREPROC = ExternalTool("swi-preproc", "swi-preproc")
# swi-analysis (yongninglei/swi-analysis): every analysis of one SWI session in one flat
# folder derivatives/swi_analysis/analysis-<name>/ -- surface (A, dense x1 pial), envelope (B),
# display, vessels (VesSynth / VesselBoost, GPU). Replaces pial-vein (2026-11-19).
SWI_ANALYSIS = ExternalTool("swi-analysis", "swi-analysis")
# qMRI-preproc (MRI-hub/qMRI-preproc): one MRF qMRI session (BIDS_qMRI, from qMRI-to-bids) ->
# rigid ANTs to the anat_ref T1w, maps on the T1w grid, mri_vol2surf on fsnative / dense x1.
# Needs ANTs + FreeSurfer on PATH (cajal: module load ants/2.5.1; FreeSurfer 7.3.2).
QMRI_PREPROC = ExternalTool("qmri-preproc", "qmri-preproc")
# mri-registration (MRI-hub/mri-registration) `mrireg-bold2surf`: fMRIPrep's T1w-space BOLD
# sampled onto the dense x1 mesh (dense-surfaces), derivatives/bold_dense/analysis-<name>/.
# Needs FreeSurfer on PATH (mri_vol2surf). The program's config block is container_specific.bold2surf.
BOLD2SURF = ExternalTool("bold2surf", "mrireg-bold2surf")
EXTERNAL = {
    t.name: t
    for t in (
        PYGLM,
        AUTOROI_SCALING,
        AUTOROI_CLUSTERING,
        SWI_PREP,
        EYETRACK,
        SWI_PREPROC,
        SWI_ANALYSIS,
        QMRI_PREPROC,
        BOLD2SURF,
    )
}
