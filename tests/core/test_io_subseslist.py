from __future__ import annotations

import json

import pytest

from launchcontainers.core import LCError
from launchcontainers.core.io import (
    atomic_write_json,
    atomic_write_text,
    bare,
    sha256,
    sub_ses,
)
from launchcontainers.core.subseslist import (
    Row,
    read_pairs,
    read_subseslist,
    write_subseslist,
)


def test_labels():
    assert bare("sub-1", "sub") == "01"
    assert bare("7", "ses") == "07"
    assert bare("ses-5sesavg", "ses") == "5sesavg"
    assert bare("123", "sub") == "123"
    assert sub_ses("1", "ses-02") == ("sub-01", "ses-02")


def test_sha256_and_atomic(tmp_path):
    p = tmp_path / "a" / "x.txt"
    atomic_write_text(p, "abc")
    assert (
        sha256(p)
        == "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    atomic_write_json(tmp_path / "x.json", {"p": tmp_path})
    assert json.loads((tmp_path / "x.json").read_text()) == {"p": str(tmp_path)}
    assert [f.name for f in (tmp_path / "a").iterdir()] == [
        "x.txt"
    ]  # no temp files left


def _write(tmp_path, text):
    p = tmp_path / "subseslist.txt"
    p.write_text(text)
    return p


def test_read_basic(tmp_path):
    p = _write(
        tmp_path,
        "# comment\n\nsub,ses,RUN\nsub-1,ses-2,True\n01,03,false  # off\n2,1,TRUE\n",
    )
    assert read_pairs(p) == [("01", "02"), ("02", "01")]


def test_read_extras(tmp_path):
    p = _write(
        tmp_path, "sub,ses,run,sources,note\n03,01,True\n03,5sesavg,True,01;02;03,hi\n"
    )
    rows = read_subseslist(p, extra=("sources", "missing"))
    assert rows[0] == Row("03", "01")
    assert rows[0].extras == {"sources": "", "missing": "", "note": ""}
    assert rows[1].extras == {"sources": "01;02;03", "missing": "", "note": "hi"}


@pytest.mark.parametrize(
    "text, msg",
    [
        ("sub,ses\n01,01\n", "expected header"),
        ("01,01,True\n", "expected header"),
        ("sub,ses,RUN\n01,01\n", "malformed"),
        ("sub,ses,RUN\n01,01,True,extra\n", "malformed"),
        ("sub,ses,RUN\n,01,True\n", "malformed"),
        ("sub,ses,RUN\n01,01,yes\n", "RUN must be True or False"),
        ("sub,ses,RUN\n01,01,True\nsub-1,ses-01,False\n", "duplicate"),
        ("sub,ses,RUN,x,x\n01,01,True\n", "duplicate column"),
        ("# only a comment\n", "empty work list"),
    ],
)
def test_read_errors(tmp_path, text, msg):
    with pytest.raises(LCError, match=msg):
        read_subseslist(_write(tmp_path, text))


def test_missing_file(tmp_path):
    with pytest.raises(LCError, match="not found"):
        read_subseslist(tmp_path / "nope.txt")


def test_error_is_valueerror():
    assert issubclass(LCError, ValueError)


def test_write_roundtrip(tmp_path):
    p = tmp_path / "out" / "list.txt"
    write_subseslist(
        p, [("1", "2"), Row("03", "5sesavg", {"sources": "01;02"})], comment="seeded"
    )
    assert p.read_text().splitlines()[:2] == ["# seeded", "sub,ses,RUN,sources"]
    rows = read_subseslist(p)
    assert [(r.sub, r.ses, r.extras["sources"]) for r in rows] == [
        ("01", "02", ""),
        ("03", "5sesavg", "01;02"),
    ]
