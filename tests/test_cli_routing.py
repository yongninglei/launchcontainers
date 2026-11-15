"""Where `lc` sends each config: Tools, GLM input prep (l1_surface), pyGLM."""

from __future__ import annotations

import pytest
import yaml
from typer.testing import CliRunner

from launchcontainers.cli import app

runner = CliRunner()


def _cfg(path, tool):
    path.mkdir(parents=True, exist_ok=True)
    (path / "lc_config.yaml").write_text(
        yaml.safe_dump({"general": {"container": tool, "basedir": str(path)}})
    )
    return path


@pytest.mark.parametrize("cmd", ["run", "qc"])
def test_l1_surface_run_and_qc_point_to_pyglm(tmp_path, cmd):
    wd = _cfg(tmp_path / "ana", "l1_surface")
    res = runner.invoke(app, [cmd, "-w", str(wd)])
    assert res.exit_code == 2
    assert "pyGLM" in res.output


def test_l1_surface_prepare_still_prepares_glm_inputs(tmp_path, monkeypatch):
    from launchcontainers import do_prepare

    calls = []
    monkeypatch.setattr(
        do_prepare, "main", lambda ns: calls.append(ns.lc_config) or (True, None)
    )
    wd = _cfg(tmp_path / "cfg", "l1_surface")
    ssl = tmp_path / "subseslist.txt"
    ssl.write_text("sub,ses,RUN\n01,01,True\n")
    monkeypatch.chdir(tmp_path)
    res = runner.invoke(
        app, ["prepare", "-lcc", str(wd / "lc_config.yaml"), "-ssl", str(ssl)]
    )
    assert res.exit_code == 0, res.output
    assert calls == [str(wd / "lc_config.yaml")]


def test_pyglm_is_an_lc_tool():
    pytest.importorskip("pyglm")
    from launchcontainers.lc_prepare import resolve_tool

    assert resolve_tool("pyGLM").name == "pyGLM"
