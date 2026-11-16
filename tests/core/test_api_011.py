"""API added in 0.1.1 at the request of pyGLM and autoROI."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import yaml

from launchcontainers.core import LCError
from launchcontainers.core.config import build_params, load_lc_config, parse_lc_config
from launchcontainers.core.provenance import guard_analysis_dir
from launchcontainers.core.subseslist import read_subseslist
from launchcontainers.core.tool import ToolConfig


@dataclass(frozen=True)
class P:
    names: tuple
    n: int = 1


def test_build_params_convert():
    p = build_params(P, {"names": ["a", "b"]}, "block", convert={"names": tuple})
    assert p.names == ("a", "b")
    with pytest.raises(LCError, match="Unknown key"):
        build_params(P, {"names": [], "typo": 1}, "block", convert={"names": tuple})


def test_default_tool_and_parse(config_factory):
    path = config_factory(general={"tool": None})
    raw = yaml.safe_load(path.read_text())
    raw["general"].pop("tool")
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(LCError, match="general.tool"):
        load_lc_config(path)
    assert load_lc_config(path, default_tool="dummy").tool == "dummy"
    assert parse_lc_config(raw, path, default_tool="dummy").tool_block() == {
        "alpha": 1,
        "beta": "x",
    }


def test_row_lineno_and_toolconfig_rows(tmp_path, config_factory):
    wl = tmp_path / "subseslist.txt"
    wl.write_text("sub,ses,RUN,sources\n# c\n01,avg,True,01;02\n02,01,False,\n")
    rows = read_subseslist(wl)
    assert rows[0].lineno == 3 and rows[0].extras["sources"] == "01;02"
    tc = ToolConfig(
        general=load_lc_config(config_factory()),
        params=None,
        rows={(r.sub, r.ses): r for r in rows},
    )
    assert tc.row("01", "avg").extras["sources"] == "01;02"
    assert tc.row("09", "01") is None


def test_guard_tool_runtime_only(tmp_path, config_factory):
    ana = tmp_path / "ana"
    guard_analysis_dir(ana, load_lc_config(config_factory()), "dummy", "1.0")
    changed = load_lc_config(config_factory(general={"use_dask": True}))
    with pytest.raises(LCError):
        guard_analysis_dir(ana, changed, "dummy", "1.0")
    guard_analysis_dir(ana, changed, "dummy", "1.0", runtime_only=("use_dask",))


def test_prepare_analysis_hook(tmp_path):
    import pytest

    from launchcontainers.core import LCError
    from launchcontainers.core.tool import prepare_analysis

    class NoHook:
        name = "nohook"

    class Hook:
        name = "hook"

        def __init__(self, out):
            self.out, self.seen = out, None

        def prepare_analysis(self, tc, pairs):
            self.seen = pairs
            return self.out

    pairs = [("01", "02"), ("01", "01")]
    assert prepare_analysis(NoHook(), None, pairs) == pairs
    assert prepare_analysis(Hook(None), None, pairs) == pairs
    assert prepare_analysis(Hook(sorted(pairs)), None, pairs) == sorted(pairs)
    with pytest.raises(LCError):
        prepare_analysis(Hook(pairs[:1]), None, pairs)
