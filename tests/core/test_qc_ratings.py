from __future__ import annotations

import pytest

from launchcontainers.core import LCError
from launchcontainers.core.qc import render_report
from launchcontainers.core.ratings import (
    latest_verdicts,
    load_all_ratings,
    load_ratings,
    ratings_path,
    save_rating,
)
from launchcontainers.core.tool import Figure, QCCheck, QCResult


def test_save_and_load(tmp_path):
    key = ("T1w", "01", "02", "run-01")
    row = save_rating(tmp_path, "tiger", key, False, "  bad\tregistration \n here ")
    assert row["comment"] == "bad registration here" and row["ok"] == "False"
    save_rating(tmp_path, "tiger", ("T1w", "01", "01", ""), True)
    save_rating(tmp_path, "tiger", key, True, "fixed")  # replace, not append
    path = ratings_path(tmp_path, "tiger")
    assert path == tmp_path / "qc" / "qc_rater-tiger.tsv"
    lines = path.read_text().splitlines()
    assert (
        lines[0] == "item\tsub\tses\trun\tok\tcomment\trater\ttime" and len(lines) == 3
    )
    rows = load_ratings(tmp_path, "tiger")
    assert rows[key]["ok"] == "True" and rows[("T1w", "01", "01", "")]["run"] == ""
    assert load_ratings(tmp_path, "nobody") == {}
    assert not list((tmp_path / "qc").glob(".*tmp"))


@pytest.mark.parametrize("rater", ["", "a b", "-x", "x" * 65, "../evil"])
def test_bad_rater(tmp_path, rater):
    with pytest.raises(LCError, match="rater name"):
        save_rating(tmp_path, rater, ("i", "01", "01", "r"), True)


def test_bad_rating(tmp_path):
    with pytest.raises(LCError, match="ok must be"):
        save_rating(tmp_path, "tiger", ("i", "01", "01", "r"), "yes")
    with pytest.raises(LCError, match="required"):
        save_rating(tmp_path, "tiger", ("", "01", "01", "r"), True)


def test_latest_over_raters(tmp_path):
    key = ("map", "01", "01", "run-1")
    save_rating(tmp_path, "ann", key, True)
    (tmp_path / "qc" / "qc_rater-bob.tsv").write_text(
        "item\tsub\tses\trun\tok\tcomment\trater\ttime\nmap\t01\t01\trun-1\tFalse\tlate\tbob\t2999-01-01T00:00:00\n"
    )
    assert sorted(load_all_ratings(tmp_path)) == ["ann", "bob"]
    assert latest_verdicts(tmp_path)[key]["rater"] == "bob"


def test_render_report(tmp_path):
    ana = tmp_path / "analysis-main"
    fig = ana / "qc" / "T1w" / "sub-01" / "ses-01" / "reg.png"
    fig.parent.mkdir(parents=True)
    fig.write_bytes(b"\x89PNG")
    frag = ana / "frag.html"
    frag.write_text("<p id='frag'>inline</p>")
    results = [
        QCResult(
            "01",
            "01",
            [QCCheck("outputs <present>", "ok", "all & fine")],
            [Figure("T1w", "run-01", fig), Figure("summary", "", frag)],
            html="<div id='extra'></div>",
        ),
        QCResult(
            "sub-01",
            "ses-02",
            [
                QCCheck("log", "fail", "<script>alert(1)</script>"),
                QCCheck("n", "warn", "tab\there"),
            ],
        ),
        QCResult("2", "1", []),
    ]
    save_rating(ana, "tiger", ("T1w", "01", "01", "run-01"), False, "<b>shifted</b>")
    index = render_report(ana, results, title="QC <dummy>")
    assert index == ana / "qc" / "index.html"
    text = index.read_text()
    assert "QC &lt;dummy&gt;" in text and "<dummy>" not in text
    assert (
        'href="sub-01_ses-01.html"' in text
        and 'class="cell fail"' in text
        and 'class="cell none"' in text
    )
    assert "✗ 1 / 1 rated" in text
    page = (ana / "qc" / "sub-01_ses-01.html").read_text()
    assert "outputs &lt;present&gt;" in page and "all &amp; fine" in page
    assert 'src="T1w/sub-01/ses-01/reg.png"' in page
    assert "<p id='frag'>inline</p>" in page and "<div id='extra'></div>" in page
    assert "&lt;b&gt;shifted&lt;/b&gt;" in page
    page2 = (ana / "qc" / "sub-01_ses-02.html").read_text()
    assert "<script>" not in page2 and "&lt;script&gt;" in page2
    assert (ana / "qc" / "sub-02_ses-01.html").is_file()
    tsv = (ana / "qc" / "qc.tsv").read_text().splitlines()
    assert tsv[0] == "sub\tses\tcheck\tstatus\tmessage" and len(tsv) == 4
    assert "01\t02\tn\twarn\ttab here" in tsv
