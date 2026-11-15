"""The Flywheel-style DWI containers as analysis-core Tools.

anatrois, freesurferator, rtppreproc, rtp2-preproc, rtp-pipeline, rtp2-pipeline.

This first port wraps the existing staging code (``prepare/dwi_prepare.py``,
``prepare/RTP2_prepare_input.py``, ``gen_jobscript/gen_container_cmd.py``) so
that `lc prepare` / `lc run` drive it through the Tool interface. The golden
snapshots in ``tests/golden`` pin its behaviour; the staging code can then be
rewritten container by container behind this interface.
"""

from __future__ import annotations

import os
import re
from argparse import Namespace
from pathlib import Path
from typing import Any

from analysis_core import AnalysisCoreError
from analysis_core.config import LCConfig
from analysis_core.tool import Figure, QCCheck, QCResult, SessionResult, ToolConfig

from launchcontainers import __version__ as LC_VERSION

DWI_CONTAINERS = (
    "anatrois",
    "freesurferator",
    "rtppreproc",
    "rtp2-preproc",
    "rtp-pipeline",
    "rtp2-pipeline",
)
# Containers whose staging reads the raw BIDS dataset through pybids.
_NEEDS_LAYOUT = {"anatrois", "freesurferator", "rtppreproc", "rtp2-preproc"}
# Containers that can stage a retest session as a link to a source session.
_HAS_SRC_SESSION = {"anatrois", "freesurferator"}


def _norm_ses(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).removeprefix("ses-")
    return s.zfill(2) if s.isdigit() else s


class DWIContainer:
    """One DWI container. ``version`` is launchcontainers' own version; the
    container's version comes from ``container_specific.<name>.version``."""

    version = LC_VERSION

    def __init__(self, name: str) -> None:
        if name not in DWI_CONTAINERS:
            raise ValueError(f"not a DWI container: {name}")
        self.name = name

    # -- load -------------------------------------------------------------
    def load(self, cfg: LCConfig) -> ToolConfig:
        block = cfg.tool_block(self.name)
        if not block.get("version"):
            raise AnalysisCoreError(
                f"{cfg.source_path}: container_specific.{self.name}.version is required"
            )
        general = cfg.raw.get("general") or {}
        if general.get("container", general.get("tool")) != self.name:
            raise AnalysisCoreError(
                f"{cfg.source_path}: general.container must be {self.name!r}"
            )
        if self.name in ("rtppreproc", "rtp2-preproc", "rtp-pipeline", "rtp2-pipeline"):
            # the staging code reads these keys directly
            for key in ("precontainer_anat", "anat_analysis_name"):
                if key not in block:
                    raise AnalysisCoreError(
                        f"{cfg.source_path}: container_specific.{self.name}.{key} is required"
                    )
        return ToolConfig(
            general=cfg,
            params=block,
            extra={"container_version": str(block["version"])},
        )

    # -- where it writes -------------------------------------------------
    def analysis_dirs(self, tc: ToolConfig) -> list[Path]:
        cfg = tc.general
        ver = tc.extra["container_version"]
        if (cfg.deriv_layout or "legacy") == "legacy":
            d = cfg.derivatives / f"{self.name}_{ver}" / f"analysis-{cfg.analysis_name}"
        else:
            d = cfg.derivatives / f"{self.name}-{ver}_{cfg.analysis_name}"
        return [d]

    def _namespace(self, tc: ToolConfig) -> Namespace:
        return Namespace(
            lc_config=str(tc.general.source_path),
            sub_ses_list=tc.extra.get("subseslist"),
            container_specific_config=tc.extra.get("container_config"),
        )

    def _src_session(self, tc: ToolConfig) -> str | None:
        if self.name not in _HAS_SRC_SESSION:
            return None
        return _norm_ses(tc.params.get("use_src_session"))

    # -- analysis-level staging (analysis_core.tool.prepare_analysis) ------
    def prepare_analysis(self, tc: ToolConfig, pairs: list[tuple[str, str]]):
        from launchcontainers import utils as do
        from launchcontainers.prepare import dwi_prepare

        ana = self.analysis_dirs(tc)[0]
        force = tc.general.force
        cc = tc.extra.get("container_config")
        if cc is not None:
            do.copy_file(str(cc), str(ana / f"{self.name}.json"), force)
        elif not (ana / f"{self.name}.json").exists():
            raise AnalysisCoreError(
                f"no {self.name}.json in {ana}: pass the container config with -cc"
            )
        config_json = dwi_prepare.copy_and_edit_config_json(
            self._namespace(tc), str(ana)
        )
        if not config_json:
            raise AnalysisCoreError(f"could not write {ana / (self.name + '.json')}")
        tc.extra["config_json"] = config_json
        if self.name in _NEEDS_LAYOUT:
            from bids import BIDSLayout

            tc.extra["layout"] = BIDSLayout(str(tc.general.bids_dir), validate=False)
        src = self._src_session(tc)
        if src is not None:
            # retest sessions link to the source session: stage sources first
            return sorted(pairs, key=lambda p: p[1] != src)
        return None

    # -- one session -----------------------------------------------------
    def prepare(self, tc: ToolConfig, sub: str, ses: str) -> SessionResult:
        from launchcontainers.prepare import dwi_prepare

        if "config_json" not in tc.extra:
            return SessionResult(sub, ses, "failed", ["prepare_analysis was not run"])
        ana = self.analysis_dirs(tc)[0]
        try:
            dwi_prepare._prepare_one_session(
                self._namespace(tc),
                str(ana),
                tc.general.raw,
                tc.extra["config_json"],
                tc.extra.get("layout"),
                self.name,
                tc.extra["container_version"],
                tc.general.force,
                self._src_session(tc),
                sub,
                ses,
            )
        except Exception as exc:  # noqa: BLE001 — the Tool contract: return a status
            return SessionResult(sub, ses, "failed", [f"{type(exc).__name__}: {exc}"])
        src = self._src_session(tc)
        msg = [f"linked to ses-{src}"] if src is not None and ses != src else []
        return SessionResult(
            sub, ses, "prepared", msg, [ana / f"sub-{sub}" / f"ses-{ses}"]
        )

    # -- before lc run (launchcontainers-only hook) -----------------------------
    def before_run(self, tc: ToolConfig, pairs, submit: bool) -> list[str]:
        """The container config must carry the staged inputs; on a real run,
        an old RTP_log.txt is renamed so the new run does not append to it."""
        import json
        from datetime import datetime

        ana = self.analysis_dirs(tc)[0]
        cc = ana / f"{self.name}.json"
        if not cc.is_file():
            raise AnalysisCoreError(f"{cc} missing: run `lc prepare` first")
        if "inputs" not in json.loads(cc.read_text()):
            raise AnalysisCoreError(f"{cc} has no 'inputs': run `lc prepare` first")
        notes = []
        if submit and self.name in ("rtp-pipeline", "rtp2-pipeline"):
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            for sub, ses in pairs:
                old = (
                    ana / f"sub-{sub}" / f"ses-{ses}" / "output" / "log" / "RTP_log.txt"
                )
                if old.exists():
                    old.rename(old.with_name(f"RTP_log_backup_at_{stamp}.txt"))
                    notes.append(f"sub-{sub} ses-{ses}: old RTP_log.txt backed up")
        return notes

    def shell_command(self, tc: ToolConfig, sub: str, ses: str) -> str:
        """The apptainer command line, with its own stdout/stderr redirection."""
        from launchcontainers.gen_jobscript.gen_container_cmd import gen_RTP2_cmd

        return gen_RTP2_cmd(tc.general.raw, sub, ses, str(self.analysis_dirs(tc)[0]))

    def command(self, tc: ToolConfig, sub: str, ses: str) -> list[str]:
        return ["bash", "-c", self.shell_command(tc, sub, ses)]

    # -- qc --------------------------------------------------------------
    def qc(self, tc: ToolConfig, sub: str, ses: str) -> QCResult:
        sdir = self.analysis_dirs(tc)[0] / f"sub-{sub}" / f"ses-{ses}"
        out = sdir / "output"
        checks = []
        if not sdir.exists():
            return QCResult(sub, ses, [QCCheck("prepared", "fail", f"{sdir} missing")])
        if sdir.is_symlink():
            checks.append(QCCheck("retest link", "ok", f"→ {os.readlink(sdir)}"))
            return QCResult(sub, ses, checks)
        produced = (
            [p for p in out.iterdir() if p.name not in ("log", "tmp")]
            if out.is_dir()
            else []
        )
        checks.append(
            QCCheck(
                "outputs",
                "ok" if produced else "fail",
                f"{len(produced)} entries in output/"
                if produced
                else "output/ is empty",
            )
        )
        errs = sorted((out / "log").glob(f"{self.name}-sub-{sub}_ses-{ses}_*.err"))
        if errs:
            text = errs[-1].read_text(errors="replace")
            bad = re.search(r"(?im)^.*\b(error|traceback|killed)\b.*$", text)
            checks.append(
                QCCheck(
                    "container log",
                    "warn" if bad else "ok",
                    bad.group(0)[:200] if bad else errs[-1].name,
                )
            )
        else:
            checks.append(QCCheck("container log", "warn", "no .err log yet"))
        if self.name in ("rtp-pipeline", "rtp2-pipeline"):
            checks.append(_rtp_log_check(out / "log" / "RTP_log.txt"))
        res = QCResult(sub, ses, checks)
        try:
            res.figures += self._figures(tc, sub, ses, out)
        except Exception as exc:  # noqa: BLE001 — a figure problem is a warning, not a crash
            checks.append(QCCheck("figures", "warn", f"{type(exc).__name__}: {exc}"))
        return res

    # -- pre-rendered QC images (analysis_core.slices) ---------------------------
    def _figures(self, tc: ToolConfig, sub: str, ses: str, out: Path) -> list[Figure]:
        """One rated item per session, rendered once (fingerprinted) under qc/<item>/:

        - anatrois / freesurferator  ``segmentation``: aparc+aseg (or the brain
          mask outline when there is no aparc+aseg) on the output T1
        - rtppreproc / rtp2-preproc   ``dwi_on_t1``: mean b0 on the output t1,
          i.e. the DWI→T1 registration
        - rtp(2)-pipeline: no figure yet (checks only)
        """
        from analysis_core.slices import Layer, render_item

        ana = self.analysis_dirs(tc)[0]
        overwrite = bool(tc.extra.get("qc_overwrite"))
        if self.name in ("anatrois", "freesurferator"):
            t1 = out / "T1.nii.gz"
            if not t1.is_file():
                return []
            layers = []
            seg = _first(out, ("aparc+aseg.nii.gz", "aseg.nii.gz"))
            if seg is not None:
                layers.append(
                    Layer("seg", seg.name.split(".")[0], "labels", seg, opacity=0.4)
                )
                layers.append(
                    Layer("edge", "label borders", "outline", seg, show=False)
                )
            mask = out / "brainmask.nii.gz"
            if mask.is_file():
                layers.append(
                    Layer(
                        "mask", "brain mask outline", "outline", mask, show=seg is None
                    )
                )
            if not layers:
                return []
            item, question = (
                "segmentation",
                "Is the segmentation / brain mask right on the T1?",
            )
            r = render_item(
                ana,
                item,
                sub,
                ses,
                base=t1,
                base_label="T1",
                layers=layers,
                mask=mask if mask.is_file() else None,
                question=question,
                overwrite=overwrite,
            )
        elif self.name in ("rtppreproc", "rtp2-preproc"):
            t1, dwi, bvals = out / "t1.nii.gz", out / "dwi.nii.gz", out / "dwi.bvals"
            if not (t1.is_file() and dwi.is_file() and bvals.is_file()):
                return []
            layers = [
                Layer(
                    "b0",
                    "mean b0",
                    "map",
                    lambda: _mean_b0(dwi, bvals),
                    cmap="hot",
                    opacity=0.5,
                )
            ]
            mask = _first(
                out,
                ("brainmask.nii.gz", "dwi_brainmask.nii.gz", "b0_brain_mask.nii.gz"),
            )
            if mask is not None:
                layers.append(Layer("mask", "DWI brain mask outline", "outline", mask))
            r = render_item(
                ana,
                "dwi_on_t1",
                sub,
                ses,
                base=t1,
                base_label="t1",
                layers=layers,
                inputs={
                    "t1": t1,
                    "dwi": dwi,
                    "bvals": bvals,
                    **({"mask": mask} if mask is not None else {}),
                },
                question="Is the DWI (mean b0) aligned with the T1?",
                overwrite=overwrite,
            )
        else:
            return []
        return [Figure(r.out.parts[-3], "", r.preview)] if r.preview else []


_RTP_EXIT = "Sending exit(0) signal."


def _rtp_log_check(log: Path) -> QCCheck:
    """rtp(2)-pipeline finished when RTP_log.txt ends with the exit(0) line
    (from analysis_checker.rtp.RTP2PipelineSpec)."""
    if not log.is_file():
        return QCCheck("RTP_log", "fail", "output/log/RTP_log.txt missing")
    lines = [
        ln.rstrip() for ln in log.read_text(errors="replace").splitlines() if ln.strip()
    ]
    if not lines:
        return QCCheck("RTP_log", "fail", "RTP_log.txt is empty")
    if lines[-1] != _RTP_EXIT:
        return QCCheck("RTP_log", "fail", f"last line: {lines[-1][:120]!r}")
    return QCCheck("RTP_log", "ok", _RTP_EXIT)


def _first(folder: Path, names) -> Path | None:
    return next((folder / n for n in names if (folder / n).is_file()), None)


def _mean_b0(dwi: Path, bvals: Path):
    """Mean of the b<=50 volumes (all volumes if none), as a 3-D nibabel image."""
    import nibabel as nib
    import numpy as np

    img = nib.load(str(dwi))
    data = np.asanyarray(img.dataobj)
    if data.ndim == 3:
        return img
    b = np.loadtxt(str(bvals)).ravel()
    sel = np.where(b <= 50)[0] if b.size == data.shape[3] else np.array([], int)
    vols = data[..., sel] if sel.size else data
    return nib.Nifti1Image(vols.astype(np.float32).mean(axis=3), img.affine)


ANATROIS = DWIContainer("anatrois")
FREESURFERATOR = DWIContainer("freesurferator")
RTPPREPROC = DWIContainer("rtppreproc")
RTP2_PREPROC = DWIContainer("rtp2-preproc")
RTP_PIPELINE = DWIContainer("rtp-pipeline")
RTP2_PIPELINE = DWIContainer("rtp2-pipeline")
