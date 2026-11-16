from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from launchcontainers.core.runlog import ENV_RUN_DIR, ENV_RUN_ID


@pytest.fixture(autouse=True)
def _no_outer_run(monkeypatch):
    """Tests never inherit a wrapper run from the environment."""
    monkeypatch.delenv(ENV_RUN_ID, raising=False)
    monkeypatch.delenv(ENV_RUN_DIR, raising=False)


@pytest.fixture
def fake_bids(tmp_path: Path) -> Path:
    """basedir with BIDS/sub-0X/ses-0Y/func/… placeholders."""
    base = tmp_path / "project"
    for sub in ("01", "02"):
        for ses in ("01", "02"):
            d = base / "BIDS" / f"sub-{sub}" / f"ses-{ses}" / "func"
            d.mkdir(parents=True)
            (d / f"sub-{sub}_ses-{ses}_task-x_bold.nii.gz").write_bytes(b"fake")
    (base / "BIDS" / "dataset_description.json").write_text('{"Name": "fake"}\n')
    return base


def make_config(path: Path, basedir: Path, **over) -> Path:
    general = {
        "basedir": str(basedir),
        "bidsdir_name": "BIDS",
        "tool": "dummy",
        "analysis_name": "main",
        "host": "local",
        "force": False,
    }
    block = {"alpha": 1, "beta": "x"}
    general.update(over.pop("general", {}))
    block.update(over.pop("block", {}))
    raw = {
        "general": general,
        "container_specific": {"dummy": block, "other": {"z": 1}},
        "host_options": {
            "local": {"manager": "local", "n_workers": 2},
            "BCBL": {"manager": "sge", "queue": "long.q"},
        },
    }
    raw.update(over)
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


@pytest.fixture
def config_factory(fake_bids, tmp_path):
    counter = {"n": 0}

    def factory(**over) -> Path:
        counter["n"] += 1
        return make_config(tmp_path / f"cfg{counter['n']}.yaml", fake_bids, **over)

    return factory
