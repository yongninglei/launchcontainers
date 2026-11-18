"""Analyses that are their own programs (pyGLM, autoROI): lc drives them through
their CLI and ``--plan`` (docs/plan_protocol.md), never by importing them.

A fake program stands in for pyGLM: a Python script that answers ``--plan``
and, run for a session, writes its outputs.
"""

from __future__ import annotations

import json
import sys
import textwrap

import pytest
import yaml

FAKE = textwrap.dedent(
    """\
    #!{python}
    import json, sys, pathlib, argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config"); p.add_argument("--sub"); p.add_argument("--ses")
    p.add_argument("--sources"); p.add_argument("--plan", action="store_true")
    a = p.parse_args()
    root = pathlib.Path({root!r})
    if "BAD" in pathlib.Path(a.config).read_text():
        print("invalid config: BAD", file=sys.stderr); sys.exit(2)
    ana = [str(root / "out" / "analysis-main_base"), str(root / "out" / "analysis-main_ME")]
    inp = root / "inputs" / f"sub-{{a.sub}}_ses-{{a.ses}}_bold.nii.gz"
    if a.plan:
        plan = {{"tool": "pyGLM", "version": "9.9.9", "analysis_dirs": ana, "sub": a.sub, "ses": a.ses,
                "session_outputs": [], "inputs": [], "problems": [], "notes": [],
                "source_datasets": {{"fmriprep": str(root / "fmriprep")}}}}
        if a.sub:
            plan["session_outputs"] = [f"{{d}}/sub-{{a.sub}}/ses-{{a.ses}}" for d in ana]
            plan["inputs"] = [str(inp)]
            plan["problems"] = [] if inp.exists() else [f"no BOLD: {{inp}}"]
            plan["notes"] = [f"sources={{a.sources}}"] if a.sources else []
        print(json.dumps(plan)); sys.exit(0)
    for d in ana:
        out = pathlib.Path(d) / f"sub-{{a.sub}}" / f"ses-{{a.ses}}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "stat-t.func.gii").write_text("x")
    print("fitted", a.sub, a.ses)
    """
)


@pytest.fixture
def setup(tmp_path):
    exe = tmp_path / "bin" / "fake-pyglm"
    exe.parent.mkdir()
    exe.write_text(FAKE.format(python=sys.executable, root=str(tmp_path)))
    exe.chmod(0o755)
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "sub-01_ses-01_bold.nii.gz").write_text("bold")
    cfg = {
        "general": {
            "basedir": str(tmp_path),
            "bidsdir_name": "BIDS",
            "tool": "pyGLM",
            "analysis_name": "main",
            "host": "local",
        },
        "container_specific": {"pyGLM": {"task": "fLoc", "strategies": ["base", "ME"]}},
        "host_options": {"local": {"manager": "local", "exe": {"pyGLM": str(exe)}}},
    }
    lcc = tmp_path / "lc_config.yaml"
    lcc.write_text(yaml.safe_dump(cfg))
    ssl = tmp_path / "subseslist.txt"
    ssl.write_text("sub,ses,RUN\n01,01,True\n01,02,True\n")
    return tmp_path, lcc, ssl


def test_prepare_run_qc(setup):
    from launchcontainers import lc_prepare, lc_qc, lc_run

    root, lcc, ssl = setup
    res = lc_prepare.prepare(lcc, ssl)
    dirs = res.analysis_dirs
    assert [d.name for d in dirs] == ["analysis-main_base", "analysis-main_ME"]
    status = {(r.sub, r.ses): r.status for r in res.results}
    assert status == {("01", "01"): "prepared", ("01", "02"): "blocked"}
    for d in dirs:  # provenance: config guard, dataset_description, recorded inputs
        assert (d / "lc_config.yaml").is_file() and (
            d / "dataset_description.json"
        ).is_file()
        rec = (d / "logs" / "inputs" / "sub-01_ses-01.tsv").read_text().splitlines()
        assert rec[0] == "path\tsize\tmtime_ns" and rec[1].endswith(
            "\t4\t" + rec[1].split("\t")[2]
        )
    desc = json.loads((dirs[0] / "dataset_description.json").read_text())
    assert "9.9.9" in json.dumps(desc["GeneratedBy"])

    plan = lc_run.plan(dirs[0])
    assert [(j.sub, j.ses) for j in plan.jobs] == [("01", "01")]  # 01/02 was blocked
    assert (
        "fake-pyglm" in plan.jobs[0].line and "--sub 01 --ses 01" in plan.jobs[0].line
    )
    run = lc_run.run(plan, submit=True)
    assert run.exit_codes == {("01", "01"): 0}

    rep = lc_qc.qc(dirs[0], ssl)
    checks = {(r.sub, r.ses): {c.name: c.status for c in r.checks} for r in rep.results}
    assert checks[("01", "01")]["last run"] == "ok"
    assert checks[("01", "01")]["output ses-01"] == "ok"
    assert checks[("01", "02")]["output ses-02"] == "fail"


def test_missing_program_and_bad_config(setup):
    from launchcontainers import lc_prepare
    from launchcontainers.core import LCError

    root, lcc, ssl = setup
    text = lcc.read_text()
    lcc.write_text(
        text.replace(str(root / "bin" / "fake-pyglm"), str(root / "bin" / "nope"))
    )
    with pytest.raises(LCError, match="not found"):
        lc_prepare.load(lcc, ssl)
    lcc.write_text(text.replace("fLoc", "BAD"))
    with pytest.raises(LCError, match="invalid config: BAD"):
        lc_prepare.load(lcc, ssl)


def test_autoroi_sources_become_a_flag(setup):
    from launchcontainers import lc_prepare
    from launchcontainers.tools import BUILTIN

    root, lcc, ssl = setup
    cfg = yaml.safe_load(lcc.read_text())
    cfg["general"]["tool"] = "autoROI-scaling"
    cfg["container_specific"] = {"autoROI-scaling": {}}
    cfg["host_options"]["local"]["exe"] = {
        "autoROI-scaling": str(root / "bin" / "fake-pyglm")
    }
    lcc.write_text(yaml.safe_dump(cfg))
    ssl.write_text("sub,ses,RUN,sources\n01,avg,True,01;02\n01,01,True,\n")
    tool, tc, rows = lc_prepare.load(lcc, ssl)
    assert tool is BUILTIN["autoROI-scaling"]
    assert tool.command(tc, "01", "avg")[-2:] == ["--sources", "01,02"]
    assert "--sources" not in tool.command(tc, "01", "01")


def test_description_and_sources_in_dataset_description(setup):
    """general.description -> Description (rewording never trips the guard, the old
    description is archived); the program's source_datasets -> SourceDatasets."""
    from launchcontainers import lc_prepare

    root, lcc, ssl = setup
    cfg = yaml.safe_load(lcc.read_text())
    cfg["general"]["description"] = "Redo the  registration\n of analysis-v1"
    lcc.write_text(yaml.safe_dump(cfg))
    d = lc_prepare.prepare(lcc, ssl).analysis_dirs[0]
    dd = json.loads((d / "dataset_description.json").read_text())
    assert dd["Description"] == (
        "Redo the registration of analysis-v1 [pyGLM 9.9.9, prepared by launchcontainers]"
    )
    assert dd["SourceDatasets"] == [{"URI": "bids:raw:"}, {"URI": "bids:fmriprep:"}]
    assert dd["DatasetLinks"]["fmriprep"] == "../../fmriprep"

    cfg["general"]["description"] = "reworded"
    lcc.write_text(yaml.safe_dump(cfg))
    lc_prepare.prepare(lcc, ssl)  # no force needed: description is not computational
    dd = json.loads((d / "dataset_description.json").read_text())
    assert dd["Description"].startswith("reworded [")
    old = list((d / "logs" / "history").glob("dataset_description_*.json"))
    assert len(old) == 1 and "Redo the registration" in old[0].read_text()
