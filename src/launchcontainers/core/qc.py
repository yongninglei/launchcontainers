"""Static QC report of automatic checks (ARCHITECTURE.md §6, layer 1).

``render_report`` writes, under ``<analysis_dir>/qc/``:

- ``index.html``          sub × ses matrix; each cell coloured by the worst
                          check status and linking to the session page; when
                          raters' ``qc_rater-*.tsv`` exist, the latest verdict too
- ``sub-XX_ses-YY.html``  checks table + figures (``<img>`` with a relative
                          path, or an ``.html`` fragment inlined) + QCResult.html
- ``qc.tsv``              sub ses check status message

Pages are self-contained (inline CSS). All text is escaped with html.escape;
only the tool's own HTML fragments are inserted as they are. The final look
(web-MRI style, following QC-registration) comes later.
"""

from __future__ import annotations

import datetime
import html
import os
from collections.abc import Sequence
from pathlib import Path

from launchcontainers.core.io import PathLike, atomic_write_text, bare
from launchcontainers.core.ratings import QC_DIR, latest_verdicts
from launchcontainers.core.tool import QCResult

_IMAGE_EXT = {".png", ".svg", ".jpg", ".jpeg", ".gif", ".webp"}
_HTML_EXT = {".html", ".htm"}
_CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:24px;color:#1d1d1f;background:#fff}
h1{font-size:22px}h2{font-size:17px;margin-top:28px}
table{border-collapse:collapse;margin:8px 0}
th,td{border:1px solid #d0d0d5;padding:6px 10px;text-align:left;vertical-align:top;font-size:14px}
th{background:#f3f3f6}
td.cell{text-align:center;min-width:72px}
td.cell a{color:inherit;text-decoration:none;display:block}
.ok{background:#d9f2df}.warn{background:#fff1c2}.fail{background:#f9d0d0}.none{background:#ececf0}
.rating{display:block;font-size:12px;color:#444}
.fig{margin:12px 0}.fig img{max-width:100%;border:1px solid #d0d0d5}
.muted{color:#666;font-size:13px}
@media (prefers-color-scheme: dark){
 body{background:#161618;color:#e8e8ea}th{background:#26262a}th,td{border-color:#3a3a40}
 .ok{background:#1f4d2b}.warn{background:#5a4a12}.fail{background:#5c2020}.none{background:#2a2a2e}
 .rating,.muted{color:#aaa}}
"""

e = html.escape


def session_page_name(sub: str, ses: str) -> str:
    return f"sub-{bare(sub, 'sub')}_ses-{bare(ses, 'ses')}.html"


def _page(title: str, body: str) -> str:
    return (
        f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{e(title)}</title><style>{_CSS}</style></head>\n<body>\n{body}\n</body></html>\n"
    )


def _figure_html(fig, qc_dir: Path) -> str:
    """A relative ``fig.path`` is taken relative to the analysis dir (qc_dir's parent)."""
    path = Path(fig.path)
    if not path.is_absolute():
        path = qc_dir.parent / path
    label = e(" · ".join(x for x in (fig.item, fig.run) if x))
    caption = f'<div class="muted">{label} — {e(path.name)}</div>'
    if path.suffix.lower() in _HTML_EXT and path.is_file():
        return f'<div class="fig">{caption}{path.read_text()}</div>'
    rel = os.path.relpath(path.resolve(), qc_dir.resolve()).replace(os.sep, "/")
    if path.suffix.lower() in _IMAGE_EXT:
        return f'<div class="fig">{caption}<img src="{e(rel)}" alt="{label}" loading="lazy"></div>'
    return f'<div class="fig">{caption}<a href="{e(rel)}">{e(path.name)}</a></div>'


def _rating_summary(rows: list[dict[str, str]]) -> tuple[str, str]:
    """(short text, tooltip) for one session's latest verdicts."""
    n_bad = sum(r.get("ok") != "True" for r in rows)
    last = max(rows, key=lambda r: r.get("time", ""))
    verdict = "✓" if n_bad == 0 else f"✗ {n_bad}"
    text = f"{verdict} / {len(rows)} rated"
    tip = f"latest: {last.get('item')} {last.get('run')} = {last.get('ok')} by {last.get('rater')} at {last.get('time')}"
    return text, tip


def _session_body(
    r: QCResult, qc_dir: Path, title: str, ratings: list[dict[str, str]]
) -> str:
    sub, ses = bare(r.sub, "sub"), bare(r.ses, "ses")
    parts = [
        f'<p><a href="index.html">← {e(title)}</a></p>',
        f'<h1>sub-{e(sub)} ses-{e(ses)} <span class="{e(r.status)}">&nbsp;{e(r.status)}&nbsp;</span></h1>',
        "<h2>Checks</h2>",
    ]
    if r.checks:
        rows = "".join(
            f'<tr><td>{e(c.name)}</td><td class="{e(c.status)}">{e(c.status)}</td>'
            f"<td>{e(c.message)}</td></tr>"
            for c in r.checks
        )
        parts.append(
            f"<table><tr><th>check</th><th>status</th><th>message</th></tr>{rows}</table>"
        )
    else:
        parts.append('<p class="muted">No checks.</p>')
    if ratings:
        rows = "".join(
            f"<tr><td>{e(x.get('item', ''))}</td><td>{e(x.get('run', ''))}</td>"
            f'<td class="{"ok" if x.get("ok") == "True" else "fail"}">{e(x.get("ok", ""))}</td>'
            f"<td>{e(x.get('comment', ''))}</td><td>{e(x.get('rater', ''))}</td>"
            f"<td>{e(x.get('time', ''))}</td></tr>"
            for x in sorted(
                ratings, key=lambda x: (x.get("item", ""), x.get("run", ""))
            )
        )
        parts.append(
            "<h2>Rater verdicts (latest per item)</h2><table><tr><th>item</th><th>run</th><th>ok</th>"
            f"<th>comment</th><th>rater</th><th>time</th></tr>{rows}</table>"
        )
    if r.figures:
        parts.append("<h2>Figures</h2>")
        parts += [_figure_html(f, qc_dir) for f in r.figures]
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
    title = title or f"QC — {ana.name}"

    verdicts: dict[tuple[str, str], list[dict[str, str]]] = {}
    for (_item, sub, ses, _run), row in latest_verdicts(ana).items():
        verdicts.setdefault((bare(sub, "sub"), bare(ses, "ses")), []).append(row)

    by_cell: dict[tuple[str, str], QCResult] = {}
    for r in results:
        by_cell[(bare(r.sub, "sub"), bare(r.ses, "ses"))] = r
    subs = sorted({s for s, _ in by_cell})
    sess = sorted({t for _, t in by_cell})

    tsv = ["sub\tses\tcheck\tstatus\tmessage"]
    for (sub, ses), r in sorted(by_cell.items()):
        page = _page(
            f"{title} — sub-{sub} ses-{ses}",
            _session_body(r, qc_dir, title, verdicts.get((sub, ses), [])),
        )
        atomic_write_text(qc_dir / session_page_name(sub, ses), page)
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
            rating = ""
            if verdicts.get((s, t)):
                text, rtip = _rating_summary(verdicts[(s, t)])
                rating = f'<span class="rating" title="{e(rtip)}">{e(text)}</span>'
            cells.append(
                f'<td class="cell {e(r.status)}" title="{e(tip)}"><a href="{e(session_page_name(s, t))}">'
                f"{e(r.status)}{rating}</a></td>"
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
