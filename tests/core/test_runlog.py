from __future__ import annotations

import json
from pathlib import Path

from launchcontainers.core.runlog import (
    ENV_RUN_DIR,
    ENV_RUN_ID,
    MARKER,
    finish_run,
    start_run,
)


def _tsv(path: Path):
    lines = path.read_text().splitlines()
    cols = lines[0].split("\t")
    return [dict(zip(cols, line.split("\t"))) for line in lines[1:]]


def _setup(tmp_path):
    cfg = tmp_path / "lc_config.yaml"
    cfg.write_text("general: {a: 1}\n")
    sl = tmp_path / "subseslist.txt"
    sl.write_text("sub,ses,RUN\n01,01,True\n01,02,True\n")
    return tmp_path / "proj", [tmp_path / "ana1", tmp_path / "ana2"], cfg, sl


def test_batch_run(tmp_path):
    proj, anas, cfg, sl = _setup(tmp_path)
    run = start_run(
        "dummy",
        proj,
        "main",
        anas,
        cfg,
        sl,
        version="1.0",
        code_dir=tmp_path,
        packages=["pyyaml"],
    )
    assert run.run_dir.parent == proj / "logs" / "dummy" and run.run_id.startswith(
        "main_"
    )
    assert (run.run_dir / "lc_config.yaml").is_file() and (
        run.run_dir / "subseslist.txt"
    ).is_file()
    finish_run(
        run,
        1,
        [
            {"sub": "01", "ses": "01", "status": "ok"},
            {"sub": "01", "ses": "02", "ok": False},
        ],
    )
    rec = json.loads((run.run_dir / "run.json").read_text())
    assert (
        rec["ExitCode"] == 1 and rec["Packages"]["pyyaml"] and len(rec["Sessions"]) == 2
    )
    for ana in anas:
        runs = _tsv(ana / "logs" / "runs.tsv")
        assert len(runs) == 1 and runs[0]["n_ok"] == "1" and runs[0]["n_failed"] == "1"
        sess = _tsv(ana / "logs" / "sessions.tsv")
        assert [(s["ses"], s["status"]) for s in sess] == [
            ("01", "ok"),
            ("02", "failed"),
        ]
        assert (ana / "logs" / run.run_id / "run.json").is_file()
        assert (ana / "lc_config.yaml").read_text() == cfg.read_text()
        assert "01,02,True" in (ana / "subseslist.txt").read_text()
    assert run.markers()[0] == f"{MARKER} {run.run_dir}" and len(run.markers()) == 3

    # second batch: append-only TSVs, union subseslist, changed config → history
    cfg.write_text("general: {a: 2}\n")
    run2 = start_run("dummy", proj, "main", anas[:1], cfg, None, version="1.0")
    finish_run(run2, 0, [{"sub": "02", "ses": "01", "status": "ok"}])
    assert len(_tsv(anas[0] / "logs" / "runs.tsv")) == 2
    assert len(_tsv(anas[0] / "logs" / "sessions.tsv")) == 3
    text = (anas[0] / "subseslist.txt").read_text()
    assert "01,01,True" in text and "02,01,True" in text
    assert len(list((anas[0] / "logs" / "history").glob("lc_config_*.yaml"))) == 1


def test_single_session_scope_and_same_second(tmp_path):
    proj, anas, cfg, _ = _setup(tmp_path)
    r1 = start_run(
        "dummy", proj, "main", anas[:1], cfg, None, version="1.0", scope="sub-01_ses-02"
    )
    r2 = start_run(
        "dummy", proj, "main", anas[:1], cfg, None, version="1.0", scope="sub-01_ses-02"
    )
    assert r1.run_id.startswith("main_sub-01_ses-02_") and r1.run_id != r2.run_id
    finish_run(
        r1, 0, [{"sub": "01", "ses": "02", "status": "skipped"}], publish_config=False
    )
    assert _tsv(anas[0] / "logs" / "sessions.tsv")[0]["status"] == "skipped"
    assert not (anas[0] / "lc_config.yaml").exists()


def test_no_project_dir(tmp_path):
    _, anas, _cfg, _ = _setup(tmp_path)
    run = start_run("dummy", None, "main", anas[:1], None, None, version=None)
    assert run.run_dir.parent == anas[0] / "logs"
    finish_run(run, 0, [{"sub": "01", "ses": "01", "status": "ok"}])
    assert (
        run.copied_to == []
        and _tsv(anas[0] / "logs" / "runs.tsv")[0]["config_sha256"] == ""
    )


def test_nested_mode(tmp_path, monkeypatch):
    proj, anas, cfg, _sl = _setup(tmp_path)
    outer = start_run(
        "wrapper",
        None,
        "main",
        anas[:1],
        None,
        None,
        version=None,
        scope="sub-01_ses-01",
    )
    monkeypatch.setenv(ENV_RUN_ID, outer.run_id)
    monkeypatch.setenv(ENV_RUN_DIR, str(outer.run_dir))
    inner = start_run("dummy", proj, "main", anas, cfg, None, version="1.0")
    assert (
        inner.nested and inner.run_id == outer.run_id and inner.run_dir == outer.run_dir
    )
    assert not (proj / "logs").exists()  # no second run folder
    assert (
        outer.run_dir / "lc_config.yaml"
    ).is_file()  # config copied into the wrapper's run
    finish_run(inner, 0, [{"sub": "01", "ses": "01", "status": "skipped"}])
    assert not (anas[0] / "logs" / "runs.tsv").exists()  # nested finish appends nothing
    rec = json.loads((outer.run_dir / "tool_dummy.json").read_text())
    assert (
        rec["Nested"]
        and rec["ExitCode"] == 0
        and rec["Sessions"][0]["status"] == "skipped"
    )
    assert json.loads((outer.run_dir / "run.json").read_text())["Tool"] == "wrapper"
    assert inner.markers() == [f"{MARKER} {outer.run_dir}"]

    monkeypatch.delenv(ENV_RUN_ID)
    finish_run(outer, 0, [{"sub": "01", "ses": "01", "status": "ok"}])
    for ana in anas:  # inner analysis dirs get exactly one row
        assert len(_tsv(ana / "logs" / "runs.tsv")) == 1
        assert len(_tsv(ana / "logs" / "sessions.tsv")) == 1
    assert (anas[1] / "logs" / outer.run_id / "tool_dummy.json").is_file()
    assert (anas[1] / "lc_config.yaml").is_file()
