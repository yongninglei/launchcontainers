"""The launch-status report (core.qc.render_report): matrix, session pages, qc.tsv."""

from __future__ import annotations

from launchcontainers.core.qc import render_report
from launchcontainers.core.tool import QCCheck, QCResult


def test_render_report(tmp_path):
    ana = tmp_path / "analysis-main"
    results = [
        QCResult(
            "01",
            "01",
            [QCCheck("outputs <present>", "ok", "all & fine")],
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
    index = render_report(ana, results, title="QC <dummy>")
    assert index == ana / "qc" / "index.html"
    text = index.read_text()
    assert "QC &lt;dummy&gt;" in text and "<dummy>" not in text
    assert (
        'href="sub-01_ses-01.html"' in text
        and 'class="cell fail"' in text
        and 'class="cell none"' in text
    )
    page = (ana / "qc" / "sub-01_ses-01.html").read_text()
    assert (
        "outputs &lt;present&gt;" in page
        and "all &amp; fine" in page
        and "<div id='extra'></div>" in page
    )
    page2 = (ana / "qc" / "sub-01_ses-02.html").read_text()
    assert "<script>" not in page2 and "&lt;script&gt;" in page2
    assert "<img" not in page + page2  # no image QC in launchcontainers
    tsv = (ana / "qc" / "qc.tsv").read_text().splitlines()
    assert tsv[0] == "sub\tses\tcheck\tstatus\tmessage" and len(tsv) == 4
    assert "01\t02\tn\twarn\ttab here" in tsv
