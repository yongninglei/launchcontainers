"""The Tool-based `lc prepare` must stage exactly what the legacy code staged.

Goldens: tests/golden/data/*.json, captured by tests/golden/capture_legacy.py
from commit a7448d9 (the bug-fixed legacy code). Compared: every entry under
sub-*/ (links, dirs, files, small config contents), the top-level files the
legacy code wrote (except lc_config.yaml / subseslist.txt, now owned by the
launchcontainers.core guard and run log), and the run command per session.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import fakedata  # tests/conftest.py puts tests/ and tests/golden/ on sys.path
from scenarios import SCENARIOS
from snapshot import diff, norm, snapshot_dir

HERE = Path(__file__).resolve().parent

OWNED_BY_CORE = {
    "lc_config.yaml",
    "subseslist.txt",
    "dataset_description.json",
    ".bidsignore",
}


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return fakedata.build_dataset(tmp_path_factory.mktemp("lcfake"))


def _comparable(snap: dict) -> dict:
    def keep(k: str) -> bool:
        return k.startswith("sub-") or k not in OWNED_BY_CORE

    return {
        "tree": {k: v for k, v in snap["tree"].items() if keep(k)},
        "files": {k: v for k, v in snap["files"].items() if keep(k)},
        "commands": snap["commands"],
        "status": snap["status"],
    }


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_prepare_matches_golden(root, scenario):
    from launchcontainers import lc_prepare

    name, container, layout, host, overrides = scenario
    overrides = {
        k: (str(root / "configs" / "fsmask.nii.gz") if v == "FSMASK" else v)
        for k, v in overrides.items()
    }
    cfg = fakedata.make_config(root, container, name, layout, host, **overrides)
    res = lc_prepare.prepare(
        cfg, root / "configs" / "subseslist.txt", root / "configs" / f"{container}.json"
    )
    tool, tc, rows = lc_prepare.load(cfg, root / "configs" / "subseslist.txt")
    adir = res.analysis_dirs[0]
    snap = snapshot_dir(adir, root)
    snap["status"] = {"success": res.ok}
    snap["commands"] = {
        f"sub-{r.sub}_ses-{r.ses}": norm(tool.shell_command(tc, r.sub, r.ses), root)
        for r in rows
    }
    golden = json.loads((HERE / "golden" / "data" / f"{name}.json").read_text())
    assert norm(str(adir), root) == golden["analysis_dir"]
    problems = diff(_comparable(golden), _comparable(snap))
    # extra top-level entries from core (logs/ is skipped by the snapshot) are fine
    problems = [
        p
        for p in problems
        if "golden=None" not in p or p.startswith(("tree[sub-", "files[sub-"))
    ]
    assert not problems, "\n".join(problems)

    # provenance: the prepare run is recorded with one row per session
    sessions = (adir / "logs" / "sessions.tsv").read_text().splitlines()
    assert sessions[0].split("\t")[:3] == ["sub", "ses", "status"]
    assert len(sessions) - 1 == len(rows)
    assert all(line.split("\t")[2] == "prepared" for line in sessions[1:])
    assert (adir / "dataset_description.json").is_file()
