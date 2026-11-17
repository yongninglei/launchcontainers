"""`lc qc`: launch checks only — last run, run log, the tool's expected outputs."""

from __future__ import annotations

import pytest

import fakedata


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    fakedata.require_extra()
    return fakedata.build_dataset(tmp_path_factory.mktemp("lcqc"))


def _prepared(root, container, name):
    from launchcontainers import lc_prepare

    cfg = fakedata.make_config(root, container, name, "legacy", "local")
    res = lc_prepare.prepare(
        cfg, root / "configs" / "subseslist.txt", root / "configs" / f"{container}.json"
    )
    assert res.ok
    return res.analysis_dirs[0]


def _checks(rep):
    return {(r.sub, r.ses): {c.name: c.status for c in r.checks} for r in rep.results}


def test_preproc_launch_qc(root):
    from launchcontainers import lc_qc

    wd = _prepared(root, "rtp2-preproc", "qc_pre")
    out = wd / "sub-01" / "ses-01" / "output"
    out.mkdir(parents=True, exist_ok=True)
    (out / "dwi.nii.gz").write_bytes(b"x")
    rep = lc_qc.qc(wd)
    checks = _checks(rep)
    assert checks[("01", "01")]["outputs"] == "ok"
    assert checks[("01", "02")]["outputs"] == "fail"
    assert checks[("01", "01")]["last run"] == "warn"  # never run through lc run
    assert not (
        wd / "qc" / "dwi_on_t1"
    ).exists()  # no images: that is the QC module's job
    assert rep.index.is_file()
    failed = rep.failed_list.read_text().splitlines()
    assert failed[0] == "sub,ses,RUN" and "01,02,True" in failed


def test_pipeline_qc_reads_rtp_log(root):
    from launchcontainers import lc_qc

    wd = _prepared(root, "rtp2-pipeline", "qc_pipe")
    log = wd / "sub-01" / "ses-01" / "output" / "log"
    log.mkdir(parents=True, exist_ok=True)
    (log / "RTP_log.txt").write_text("tracking...\nSending exit(0) signal.\n\n")
    checks = _checks(lc_qc.qc(wd))
    assert checks[("01", "01")]["RTP_log"] == "ok"
    assert checks[("02", "01")]["RTP_log"] == "fail"


def test_run_log_is_scanned(root, monkeypatch):
    from launchcontainers import lc_qc, lc_run
    from launchcontainers.tools import BUILTIN

    wd = _prepared(root, "rtp2-pipeline", "qc_log")
    monkeypatch.setattr(
        BUILTIN["rtp2-pipeline"],
        "command",
        lambda tc, sub, ses: [
            "bash",
            "-c",
            "echo 'ERROR: out of memory' ; exit 0" if sub == "02" else "echo fine",
        ],
    )
    lc_run.run(lc_run.plan(wd), submit=True)
    checks = _checks(lc_qc.qc(wd))
    assert (
        checks[("01", "01")]["last run"] == "ok"
        and checks[("01", "01")]["run log"] == "ok"
    )
    assert checks[("02", "01")]["run log"] == "warn"
