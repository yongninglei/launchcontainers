from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
import yaml

from launchcontainers.core import LCError
from launchcontainers.core.config import build_params, load_lc_config, provenance_view
from launchcontainers.core.provenance import (
    bids_uri,
    environment_info,
    guard_analysis_dir,
    sidecar,
    software_info,
    write_dataset_description,
)


@dataclass(frozen=True)
class DummyParams:
    alpha: int
    beta: str = "x"


def test_load_config(config_factory, fake_bids):
    cfg = load_lc_config(config_factory())
    assert cfg.tool == "dummy" and cfg.analysis_name == "main" and cfg.host == "local"
    assert cfg.bids_dir == fake_bids / "BIDS"
    assert cfg.tool_block() == {"alpha": 1, "beta": "x"}
    assert cfg.host_options()["manager"] == "local"
    assert cfg.host_options("BCBL")["manager"] == "sge"
    with pytest.raises(LCError, match="host_options.nope"):
        cfg.host_options("nope")
    with pytest.raises(LCError, match="container_specific.nope"):
        cfg.tool_block("nope")


def test_container_alias(config_factory):
    cfg = load_lc_config(config_factory(general={"tool": None, "container": "dummy"}))
    assert cfg.tool == "dummy"
    with pytest.raises(LCError, match="disagree"):
        load_lc_config(config_factory(general={"container": "other"}))


@pytest.mark.parametrize(
    "general, msg",
    [
        ({"analysis_name": 1}, "must be a string"),
        ({"tool": None}, "missing general.tool"),
    ],
)
def test_config_errors(config_factory, general, msg):
    with pytest.raises(LCError, match=msg):
        load_lc_config(config_factory(general=general))


def test_build_params():
    assert build_params(DummyParams, {"alpha": 2}, "cs.dummy") == DummyParams(2)
    with pytest.raises(LCError, match=r"Unknown key\(s\) in cs.dummy: \['gamma'\]"):
        build_params(DummyParams, {"alpha": 1, "gamma": 3}, "cs.dummy")
    with pytest.raises(
        LCError, match=r"Missing required key\(s\) in cs.dummy: \['alpha'\]"
    ):
        build_params(DummyParams, {}, "cs.dummy")


def test_provenance_view_ignores_runtime():
    a = {
        "general": {"basedir": "/x", "host": "A", "force": False, "tool": "t"},
        "container_specific": {"t": {"k": 1}, "u": {"k": 1}},
        "host_options": {"A": {}},
    }
    b = {
        "general": {"basedir": "/x", "host": "B", "force": True, "container": "t"},
        "container_specific": {"t": {"k": 1}, "u": {"k": 2}},
        "host_options": {"B": {"q": 1}},
    }
    assert provenance_view(a, "t") == provenance_view(b, "t")


def _ana(cfg):
    return cfg.derivatives / "dummy" / f"analysis-{cfg.analysis_name}"


def test_guard_same_config_ok_and_host_change(config_factory):
    cfg = load_lc_config(config_factory())
    ana = _ana(cfg)
    assert guard_analysis_dir(ana, cfg, "dummy", "1.0") == []
    assert (ana / "lc_config.yaml").is_file() and (ana / ".bidsignore").is_file()
    assert guard_analysis_dir(ana, cfg, "dummy", "1.0") == []
    # host / force / host_options / another tool's block change: not computational
    cfg2 = load_lc_config(
        config_factory(
            general={"host": "BCBL"},
            host_options={"BCBL": {"manager": "slurm"}},
            container_specific={"dummy": {"alpha": 1, "beta": "x"}, "other": {"z": 9}},
        )
    )
    assert guard_analysis_dir(ana, cfg2, "dummy", "1.0") == []
    assert not (ana / "logs" / "history").exists()


def test_guard_changed_key_refused_then_forced(config_factory):
    cfg = load_lc_config(config_factory())
    ana = _ana(cfg)
    guard_analysis_dir(ana, cfg, "dummy", "1.0")
    changed = load_lc_config(config_factory(block={"alpha": 2}))
    with pytest.raises(LCError, match=r"container_specific\.dummy\.alpha"):
        guard_analysis_dir(ana, changed, "dummy", "1.0")
    forced = load_lc_config(config_factory(block={"alpha": 2}, general={"force": True}))
    notes = guard_analysis_dir(ana, forced, "dummy", "1.0")
    assert len(notes) == 1 and "config changed" in notes[0]
    hist = list((ana / "logs" / "history").glob("lc_config_*.yaml"))
    assert (
        len(hist) == 1
        and yaml.safe_load(hist[0].read_text())["container_specific"]["dummy"]["alpha"]
        == 1
    )
    assert (
        yaml.safe_load((ana / "lc_config.yaml").read_text())["container_specific"][
            "dummy"
        ]["alpha"]
        == 2
    )


def test_version_guard(config_factory, fake_bids):
    cfg = load_lc_config(config_factory())
    ana = _ana(cfg)
    guard_analysis_dir(ana, cfg, "dummy", "1.0")
    write_dataset_description(
        ana,
        "dummy analysis-main",
        "d",
        software_info("dummy", "1.0"),
        {"bids": cfg.bids_dir},
    )
    with pytest.raises(LCError, match="produced by dummy 1.0; this is 2.0"):
        guard_analysis_dir(ana, cfg, "dummy", "2.0")
    notes = guard_analysis_dir(ana, cfg, "dummy", "2.0", force=True)
    assert "version 1.0 → 2.0" in notes[0]
    assert list((ana / "logs" / "history").glob("dataset_description_*.json"))
    assert not (ana / "dataset_description.json").exists()


def test_description_uri_sidecar(config_factory):
    cfg = load_lc_config(config_factory())
    ana = _ana(cfg)
    ana.mkdir(parents=True)
    links = {"bids": cfg.bids_dir}
    dd = json.loads(
        write_dataset_description(
            ana, "n", "desc", software_info("dummy", "1.0", "https://x"), links
        ).read_text()
    )
    assert dd["SourceDatasets"] == [{"URI": "bids:bids:"}]
    assert dd["DatasetLinks"] == {"bids": "../../.."}
    src = cfg.bids_dir / "sub-01/ses-01/func/sub-01_ses-01_task-x_bold.nii.gz"
    assert (
        bids_uri(ana, links, "bids", src)
        == "bids:bids:sub-01/ses-01/func/sub-01_ses-01_task-x_bold.nii.gz"
    )
    assert (
        bids_uri(ana, links, "self", ana / "sub-01" / "f.json") == "bids::sub-01/f.json"
    )
    with pytest.raises(LCError, match="unknown dataset link"):
        bids_uri(ana, links, "nope", src)
    sc = sidecar(
        ana, links, [("bids", src)], {"alpha": 1}, software_info("dummy", "1.0")
    )
    assert sc["SourceChecksums"][sc["Sources"][0]].startswith("sha256:")
    env = environment_info(["pyyaml", "not-a-package"])
    assert env["packages"]["not-a-package"] is None and env["packages"]["pyyaml"]
