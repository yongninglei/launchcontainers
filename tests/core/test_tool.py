from __future__ import annotations

from importlib import metadata
from pathlib import Path

import pytest

from launchcontainers.core import LCError
from launchcontainers.core import tool as toolmod
from launchcontainers.core.config import build_params, load_lc_config
from launchcontainers.core.tool import (
    QCCheck,
    QCResult,
    SessionResult,
    Tool,
    ToolConfig,
    discover_tools,
    get_tool,
    tool_version,
)


class DummyTool:
    name = "dummy"
    version = "1.0"

    def load(self, cfg):
        from dataclasses import make_dataclass

        params = build_params(
            make_dataclass("P", [("alpha", int), ("beta", str)]),
            cfg.tool_block(self.name),
            f"container_specific.{self.name}",
        )
        return ToolConfig(cfg, params)

    def analysis_dirs(self, tc):
        return [
            tc.general.derivatives / self.name / f"analysis-{tc.general.analysis_name}"
        ]

    def prepare(self, tc, sub, ses):
        return SessionResult(sub, ses, "prepared")

    def command(self, tc, sub, ses):
        return ["echo", sub, ses]

    def qc(self, tc, sub, ses):
        return QCResult(sub, ses, [QCCheck("exists", "ok")])


TOOL = DummyTool()


class Broken:
    name = "broken"


@pytest.fixture
def fake_eps(monkeypatch):
    eps = metadata.EntryPoints(
        [
            metadata.EntryPoint("dummy", f"{__name__}:TOOL", toolmod.ENTRY_POINT_GROUP),
            metadata.EntryPoint(
                "dummycls", f"{__name__}:DummyTool", toolmod.ENTRY_POINT_GROUP
            ),
            metadata.EntryPoint(
                "broken", f"{__name__}:Broken", toolmod.ENTRY_POINT_GROUP
            ),
        ]
    )

    def fake(**kw):
        assert kw == {"group": toolmod.ENTRY_POINT_GROUP}
        return eps

    monkeypatch.setattr(metadata, "entry_points", fake)


def test_discover(fake_eps, config_factory):
    with pytest.warns(UserWarning, match="broken"):
        tools = discover_tools()
    assert sorted(tools) == ["dummy", "dummycls"]
    assert isinstance(tools["dummy"], Tool) and isinstance(tools["dummycls"], DummyTool)
    t = get_tool("dummy")
    tc = t.load(load_lc_config(config_factory()))
    assert tc.params.alpha == 1
    assert t.analysis_dirs(tc)[0].name == "analysis-main"
    assert t.prepare(tc, "01", "01").status == "prepared"
    assert t.qc(tc, "01", "01").status == "ok"
    assert tool_version("dummy") is None  # no distribution behind a fake entry point


def test_get_tool_errors(fake_eps):
    with pytest.raises(
        LCError, match=r"No tool named 'nope'.*Available: broken, dummy, dummycls"
    ):
        get_tool("nope")
    with pytest.raises(LCError, match="does not implement"):
        get_tool("broken")


def test_dataclass_validation():
    with pytest.raises(ValueError):
        SessionResult("01", "01", "done")
    with pytest.raises(ValueError):
        QCCheck("x", "bad")
    assert QCResult("01", "01").status == "none"
    assert (
        QCResult("01", "01", [QCCheck("a", "ok"), QCCheck("b", "warn")]).status
        == "warn"
    )
    assert Path  # keep import
