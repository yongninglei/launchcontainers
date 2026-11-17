"""`lc run` through the Tool interface: wrapped job lines, backends by manager,
and the per-session records the lc-session wrapper writes."""

from __future__ import annotations

import csv
import shlex

import pytest

import fakedata


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    fakedata.require_extra()
    return fakedata.build_dataset(tmp_path_factory.mktemp("lcrun"))


def _prepared(root, name, host, **over):
    from launchcontainers import lc_prepare

    cfg = fakedata.make_config(root, "rtp2-pipeline", name, "legacy", host, **over)
    res = lc_prepare.prepare(
        cfg,
        root / "configs" / "subseslist.txt",
        root / "configs" / "rtp2-pipeline.json",
    )
    assert res.ok
    return res.analysis_dirs[0]


def _sessions(workdir):
    with open(workdir / "logs" / "sessions.tsv", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_plan_wraps_every_session(root):
    from launchcontainers import lc_run

    wd = _prepared(root, "run_plan", "local")
    p = lc_run.plan(wd)
    assert p.manager == "local" and len(p.jobs) == 3
    argv = shlex.split(p.jobs[0].line)
    assert argv[0].endswith("lc-session")
    assert argv[argv.index("--analysis-dir") + 1] == str(wd)
    assert argv[argv.index("--tool") + 1] == "rtp2-pipeline"
    inner = argv[argv.index("--") + 1 :]
    assert inner[:2] == ["bash", "-c"] and "apptainer" in inner[2]


@pytest.mark.parametrize(
    "host,manager,marker",
    [("BCBL", "sge", "#$ -t 1-3"), ("DIPC", "slurm", "#SBATCH --array=1-3")],
)
def test_dry_run_writes_array_script(root, host, manager, marker):
    from launchcontainers import lc_run

    wd = _prepared(root, f"run_{manager}", host)
    p = lc_run.plan(wd)
    assert p.manager == manager
    res = lc_run.run(p, submit=False)
    assert res.run_dir.name.endswith("_dryrun")
    text = res.job_script.read_text()
    assert marker in text and str(res.run_dir / "commands.txt") in text
    assert len((res.run_dir / "commands.txt").read_text().splitlines()) == 3
    if manager == "slurm":
        assert "#SBATCH --qos=regular" in text and "#SBATCH --mem=32G" in text


def test_local_run_records_each_session(root, monkeypatch):
    from launchcontainers import lc_run
    from launchcontainers.tools import BUILTIN

    wd = _prepared(root, "run_local", "local")
    # stand-in for apptainer: sub-02 fails
    monkeypatch.setattr(
        BUILTIN["rtp2-pipeline"],
        "command",
        lambda tc, sub, ses: [
            "bash",
            "-c",
            f"echo hello {sub}-{ses}; exit {3 if sub == '02' else 0}",
        ],
    )
    res = lc_run.run(lc_run.plan(wd), submit=True)
    assert res.exit_codes == {("01", "01"): 0, ("01", "02"): 0, ("02", "01"): 3}
    last = {(r["sub"], r["ses"]): r for r in _sessions(wd)}
    assert last[("01", "01")]["status"] == "ok"
    assert last[("02", "01")]["status"] == "failed"
    assert "_sub-02_ses-01_" in last[("02", "01")]["run_id"]
    log = wd / "logs" / last[("01", "02")]["run_id"] / "sub-01_ses-02.log"
    assert "hello 01-02" in log.read_text()


def test_failed_prepare_is_skipped(root):
    from launchcontainers import lc_run

    wd = _prepared(root, "run_skip", "local")
    with open(wd / "logs" / "sessions.tsv", "a") as f:
        f.write("02\t01\tfailed\trun_skip_prepare_2099-01-01_00-00-00\t-\n")
    p = lc_run.plan(wd)
    assert [(j.sub, j.ses) for j in p.jobs] == [("01", "01"), ("01", "02")]
    assert p.skipped and "02" in p.skipped[0]


def test_unknown_manager_is_an_error(root):
    from launchcontainers.core import LCError

    from launchcontainers import lc_run

    wd = _prepared(root, "run_badmgr", "local")
    cfg = (wd / "lc_config.yaml").read_text().replace("manager: local", "manager: pbs")
    (wd / "lc_config.yaml").write_text(cfg)
    with pytest.raises(LCError, match="manager"):
        lc_run.plan(wd)
