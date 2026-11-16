"""Launch-status report: did each session's launch work (no image QC).

``render_report`` writes, under ``<analysis_dir>/qc/``:

- ``index.html``          sub × ses matrix; each cell coloured by the worst
                          check status and linking to the session page
- ``sub-XX_ses-YY.html``  the session's checks (exit code, log, outputs)
- ``qc.tsv``              sub ses check status message

Pages are self-contained (inline CSS); all text is escaped. Looking at the
images themselves is not launchcontainers' job: that is MRI-hub's image-QC
module (QC-registration), which renders slices and collects raters' verdicts.
"""

from __future__ import annotations

import datetime
import html
from collections.abc import Sequence
from pathlib import Path

from launchcontainers.core.io import PathLike, atomic_write_text, bare
from launchcontainers.core.tool import QCResult

QC_DIR = "qc"

_CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:24px;color:#1d1d1f;background:#fff}
h1{font-size:22px}h2{font-size:17px;margin-top:28px}
table{border-collapse:collapse;margin:8px 0}
th,td{border:1px solid #d0d0d5;padding:6px 10px;text-align:left;vertical-align:top;font-size:14px}
th{background:#f3f3f6}
td.cell{text-align:center;min-width:72px}
td.cell a{color:inherit;text-decoration:none;display:block}
.ok{background:#d9f2df}.warn{background:#fff1c2}.fail{background:#f9d0d0}.none{background:#ececf0}
.muted{color:#666;font-size:13px}
@media (prefers-color-scheme: dark){
 body{background:#161618;color:#e8e8ea}th{background:#26262a}th,td{border-color:#3a3a40}
 .ok{background:#1f4d2b}.warn{background:#5a4a12}.fail{background:#5c2020}.none{background:#2a2a2e}
 .muted{color:#aaa}}
"""

e = html.escape


def session_page_name(sub: str, ses: str) -> str:
    return f"sub-{bare(sub, 'sub')}_ses-{bare(ses, 'ses')}.html"


def _page(title: str, body: str) -> str:
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{e(title)}</title><style>{_CSS}</style></head>\n<body>\n{body}\n</body></html>\n"
    )


def _session_body(r: QCResult, title: str) -> str:
    sub, ses = bare(r.sub, "sub"), bare(r.ses, "ses")
    parts = [
        f'<p><a href="index.html">← {e(title)}</a></p>',
        f'<h1>sub-{e(sub)} ses-{e(ses)} <span class="{e(r.status)}">&nbsp;{e(r.status)}&nbsp;</span></h1>',
        "<h2>Checks</h2>",
    ]
    if r.checks:
        rows = "".join(
            f'<tr><td>{e(c.name)}</td><td class="{e(c.status)}">{e(c.status)}</td><td>{e(c.message)}</td></tr>'
            for c in r.checks
        )
        parts.append(
            f"<table><tr><th>check</th><th>status</th><th>message</th></tr>{rows}</table>"
        )
    else:
        parts.append('<p class="muted">No checks.</p>')
    if r.html:
        parts.append(r.html)
    return "\n".join(parts)


def _tsv_cell(text: str) -> str:
    return " ".join(str(text).split())


def render_report(
    analysis_dir: PathLike, results: Sequence[QCResult], title: str | None = None
) -> Path:
    """Write qc/index.html, one page per session and qc/qc.tsv; returns index.html."""
    ana = Path(analysis_dir)
    qc_dir = ana / QC_DIR
    qc_dir.mkdir(parents=True, exist_ok=True)
    title = title or f"Launch QC — {ana.name}"

    by_cell: dict[tuple[str, str], QCResult] = {
        (bare(r.sub, "sub"), bare(r.ses, "ses")): r for r in results
    }
    subs = sorted({s for s, _ in by_cell})
    sess = sorted({t for _, t in by_cell})

    tsv = ["sub\tses\tcheck\tstatus\tmessage"]
    for (sub, ses), r in sorted(by_cell.items()):
        atomic_write_text(
            qc_dir / session_page_name(sub, ses),
            _page(f"{title} — sub-{sub} ses-{ses}", _session_body(r, title)),
        )
        tsv += [
            f"{sub}\t{ses}\t{_tsv_cell(c.name)}\t{c.status}\t{_tsv_cell(c.message)}"
            for c in r.checks
        ]
    atomic_write_text(qc_dir / "qc.tsv", "\n".join(tsv) + "\n")

    head = "".join(f"<th>ses-{e(t)}</th>" for t in sess)
    rows = []
    for s in subs:
        cells = []
        for t in sess:
            r = by_cell.get((s, t))
            if r is None:
                cells.append('<td class="cell"></td>')
                continue
            n_fail = sum(c.status == "fail" for c in r.checks)
            n_warn = sum(c.status == "warn" for c in r.checks)
            tip = f"{len(r.checks)} checks, {n_fail} fail, {n_warn} warn"
            cells.append(
                f'<td class="cell {e(r.status)}" title="{e(tip)}"><a href="{e(session_page_name(s, t))}">{e(r.status)}</a></td>'
            )
        rows.append(f"<tr><th>sub-{e(s)}</th>{''.join(cells)}</tr>")
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    n = {
        k: sum(r.status == k for r in by_cell.values())
        for k in ("ok", "warn", "fail", "none")
    }
    body = (
        f"<h1>{e(title)}</h1>"
        f'<p class="muted">{len(by_cell)} sessions — ok {n["ok"]}, warn {n["warn"]}, fail {n["fail"]}, '
        f'no checks {n["none"]} · generated {e(stamp)} · <a href="qc.tsv">qc.tsv</a></p>'
        f"<table><tr><th></th>{head}</tr>{''.join(rows)}</table>"
    )
    index = qc_dir / "index.html"
    atomic_write_text(index, _page(title, body))
    return index
