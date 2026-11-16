from __future__ import annotations

import json
import sys
from pathlib import Path

from typer.testing import CliRunner

from launchcontainers.core.session import app, run_session


def _sessions(ana: Path):
    lines = (ana / "logs" / "sessions.tsv").read_text().splitlines()
    return [line.split("\t") for line in lines[1:]]


def test_success(tmp_path, capfd):
    ana = tmp_path / "analysis-main"
    res = run_session(
        ana,
        "dummy",
        "1",
        "ses-2",
        [
            sys.executable,
            "-c",
            ("print('hello'); import sys; print('oops', file=sys.stderr)"),
        ],
    )
    assert res["exit_code"] == 0 and res["status"] == "ok"
    assert res["run_dir"].parent == ana / "logs" and res["run_dir"].name.startswith(
        "analysis-main_sub-01_ses-02_"
    )
    log = res["log"].read_text()
    assert "hello" in log and "oops" in log
    assert "hello" in capfd.readouterr().out
    row = _sessions(ana)[0]
    assert row[:3] == ["01", "02", "ok"]
    rec = json.loads((res["run_dir"] / "run.json").read_text())
    assert (
        rec["ExitCode"] == 0
        and rec["Command"][0] == sys.executable
        and rec["Sub"] == "01"
    )


def test_failure_cli(tmp_path):
    ana = tmp_path / "analysis-main"
    proj = tmp_path / "proj"
    result = CliRunner().invoke(
        app,
        [
            "--analysis-dir",
            str(ana),
            "--tool",
            "dummy",
            "--sub",
            "01",
            "--ses",
            "03",
            "--project-dir",
            str(proj),
            "--",
            sys.executable,
            "-c",
            "import sys; sys.exit(3)",
        ],
    )
    assert result.exit_code == 3, result.output
    assert _sessions(ana)[0][:3] == ["01", "03", "failed"]
    runs = (ana / "logs" / "runs.tsv").read_text().splitlines()
    assert runs[1].split("\t")[6] == "3"
    assert len(list((proj / "logs" / "dummy").iterdir())) == 1
    assert "PROVENANCE" in result.output


def test_missing_command(tmp_path):
    ana = tmp_path / "analysis-main"
    res = run_session(ana, "dummy", "01", "01", ["/nonexistent/binary-xyz"])
    assert res["exit_code"] == 127 and _sessions(ana)[0][2] == "failed"


def test_nested_tool_status(tmp_path):
    """A tool CLI inside the wrapper records into the wrapper's run; its status wins."""
    ana = tmp_path / "analysis-main"
    code = (
        "from pathlib import Path; from launchcontainers.core.runlog import start_run, finish_run;"
        f"r = start_run('dummy', None, 'main', [Path({str(ana)!r})], None, None, version='1');"
        "finish_run(r, 0, [{'sub': '01', 'ses': '01', 'status': 'skipped'}])"
    )
    res = run_session(ana, "dummy", "01", "01", [sys.executable, "-c", code])
    assert res["exit_code"] == 0 and res["status"] == "skipped"
    assert len(_sessions(ana)) == 1 and _sessions(ana)[0][2] == "skipped"
    assert len((ana / "logs" / "runs.tsv").read_text().splitlines()) == 2
    assert (res["run_dir"] / "tool_dummy.json").is_file()
