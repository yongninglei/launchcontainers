"""`lc qc` through the Tool interface: checks, pre-rendered PNGs, static report."""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest

import fakedata


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return fakedata.build_dataset(tmp_path_factory.mktemp("lcqc"))


def _prepared(root, container, name):
    from launchcontainers import lc_prepare

    cfg = fakedata.make_config(root, container, name, "legacy", "local")
    res = lc_prepare.prepare(
        cfg, root / "configs" / "subseslist.txt", root / "configs" / f"{container}.json"
    )
    assert res.ok
    return res.analysis_dirs[0]


def _head(shape=(20, 24, 18)):
    rng = np.random.default_rng(1)
    a = np.zeros(shape, np.float32)
    a[4:16, 4:20, 3:15] = 100 + rng.random((12, 16, 12)) * 50
    return a


def _fake_preproc_output(out):
    """t1 at 1 mm, a 4-volume DWI at 2 mm (two b0s), bvals."""
    out.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(_head(), np.eye(4)), out / "t1.nii.gz")
    low = _head()[::2, ::2, ::2]
    dwi = np.stack([low, low * 0.3, low, low * 0.3], axis=3)
    nib.save(nib.Nifti1Image(dwi, np.diag([2.0, 2.0, 2.0, 1.0])), out / "dwi.nii.gz")
    (out / "dwi.bvals").write_text("0 1000 5 1000\n")


def test_preproc_qc_renders_b0_on_t1(root):
    from launchcontainers import lc_qc

    wd = _prepared(root, "rtp2-preproc", "qc_pre")
    _fake_preproc_output(wd / "sub-01" / "ses-01" / "output")
    rep = lc_qc.qc(wd)
    by = {(r.sub, r.ses): r for r in rep.results}
    good = by[("01", "01")]
    assert [f.item for f in good.figures] == ["dwi_on_t1"]
    item = wd / "qc" / "dwi_on_t1" / "sub-01" / "ses-01"
    meta = json.loads((item / "session.json").read_text())
    assert [ly["key"] for ly in meta["layers"]] == ["base", "b0"]
    assert meta["resampled"] == ["b0"] and (item / "axial-3_b0.png").is_file()
    assert by[("01", "02")].figures == []  # no outputs: checks only
    assert (
        rep.index.is_file()
        and "preview.png" in (wd / "qc" / "sub-01_ses-01.html").read_text()
    )
    failed = rep.failed_list.read_text().splitlines()
    assert failed[0] == "sub,ses,RUN" and "01,02,True" in failed

    mtime = (item / "session.json").stat().st_mtime_ns
    lc_qc.qc(wd)  # unchanged inputs: not re-rendered
    assert (item / "session.json").stat().st_mtime_ns == mtime
    lc_qc.qc(wd, overwrite=True)
    assert (item / "session.json").stat().st_mtime_ns != mtime


def test_pipeline_qc_reads_rtp_log(root):
    from launchcontainers import lc_qc

    wd = _prepared(root, "rtp2-pipeline", "qc_pipe")
    log = wd / "sub-01" / "ses-01" / "output" / "log"
    log.mkdir(parents=True, exist_ok=True)
    (log / "RTP_log.txt").write_text("tracking...\nSending exit(0) signal.\n\n")
    rep = lc_qc.qc(wd)
    checks = {(r.sub, r.ses): {c.name: c.status for c in r.checks} for r in rep.results}
    assert checks[("01", "01")]["RTP_log"] == "ok"
    assert checks[("02", "01")]["RTP_log"] == "fail"
    assert checks[("01", "01")]["last run"] == "warn"  # never run through lc run
