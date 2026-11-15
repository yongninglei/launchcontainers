"""`lc qc` on analysis-core: per-session ``tool.qc()`` → static HTML report.

For every session in the work list (``<workdir>/subseslist.txt`` or ``-ssl``):

- ``tool.qc(tc, sub, ses)`` (a crash becomes a ``fail`` check, never an abort);
  tools also pre-render their PNGs here (``analysis_core.slices``:
  ``qc/<item>/sub-XX/ses-YY/``, fingerprinted, so unchanged items are skipped);
- plus one launcher check, ``last run``: the latest non-prepare row for the
  session in ``logs/sessions.tsv`` (written by analysis-session).

Writes, via ``analysis_core.qc.render_report``, ``<workdir>/qc/index.html``,
``qc/sub-XX_ses-YY.html`` and ``qc/qc.tsv``; and ``qc/failed_subseslist.txt``
(``sub,ses,RUN`` with every session that has a ``fail``), ready for
``lc run -w <workdir> -ssl qc/failed_subseslist.txt``.

``lc qc --serve`` then opens the rating page (``analysis_core.viewer``) on the
pre-rendered items; verdicts go to ``qc/qc_rater-<name>.tsv`` and show up in
``qc/index.html`` on the next ``lc qc``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from analysis_core import AnalysisCoreError
from analysis_core.qc import render_report
from analysis_core.runlog import CONFIG_NAME, SESSIONS_TSV, SUBSES_NAME
from analysis_core.tool import QCCheck, QCResult

from launchcontainers import lc_prepare

FAILED_LIST = "failed_subseslist.txt"


@dataclass
class QCReport:
    tool: str
    workdir: Path
    results: list[QCResult] = field(default_factory=list)
    index: Path | None = None
    failed_list: Path | None = None

    def worst(self, r: QCResult) -> str:
        st = {c.status for c in r.checks}
        return "fail" if "fail" in st else "warn" if "warn" in st else "ok"

    @property
    def failed(self) -> list[QCResult]:
        return [r for r in self.results if self.worst(r) == "fail"]


def _last_run(workdir: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Latest sessions.tsv row per sub/ses that is not from `lc prepare`."""
    tsv = workdir / "logs" / SESSIONS_TSV
    out: dict[tuple[str, str], dict[str, str]] = {}
    if tsv.is_file():
        with open(tsv, newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if "_prepare_" not in (row.get("run_id") or ""):
                    out[(row["sub"], row["ses"])] = row
    return out


def _run_check(row: dict[str, str] | None) -> QCCheck:
    if row is None:
        return QCCheck("last run", "warn", "never run through `lc run`")
    status = row.get("status", "")
    level = "ok" if status in ("ok", "skipped") else "fail"
    return QCCheck(
        "last run", level, f"{status} — {row.get('run_id')} ({row.get('finished')})"
    )


def qc(
    workdir: Path,
    subseslist: Path | None = None,
    overwrite: bool = False,
    console=None,
) -> QCReport:
    """Checks + pre-rendered figures per session, then the static report.

    Figures are rendered once (``analysis_core.slices``: fingerprinted), so a
    second call only re-renders what changed; ``overwrite`` forces it.
    """
    workdir = Path(workdir).expanduser().resolve()
    lc_config = workdir / CONFIG_NAME
    if not lc_config.is_file():
        raise AnalysisCoreError(f"{workdir}: no {CONFIG_NAME}; run `lc prepare` first")
    ssl = (
        Path(subseslist).expanduser().resolve() if subseslist else workdir / SUBSES_NAME
    )
    if not ssl.is_file():
        raise AnalysisCoreError(f"work list not found: {ssl}")
    tool, tc, rows = lc_prepare.load(lc_config, ssl, None)
    tc.extra["qc_overwrite"] = overwrite
    rep = QCReport(tool.name, workdir)
    runs = _last_run(workdir)

    for r in rows:
        try:
            res = tool.qc(tc, r.sub, r.ses)
        except Exception as exc:  # noqa: BLE001 — one broken session must not stop the report
            res = QCResult(
                r.sub, r.ses, [QCCheck("qc", "fail", f"{type(exc).__name__}: {exc}")]
            )
        res.checks.insert(0, _run_check(runs.get((r.sub, r.ses))))
        rep.results.append(res)
        if console is not None:
            w = rep.worst(res)
            style = {"ok": "green", "warn": "yellow", "fail": "red"}[w]
            bad = [f"{c.name}: {c.message}" for c in res.checks if c.status != "ok"]
            console.print(
                f"  [{style}]{w:4}[/{style}] sub-{r.sub} ses-{r.ses}  "
                + "; ".join(bad),
                markup=True,
            )

    title = f"{tool.name} — {workdir.name}"
    rep.index = render_report(workdir, rep.results, title=title)
    rep.failed_list = rep.index.parent / FAILED_LIST
    rep.failed_list.write_text(
        "sub,ses,RUN\n" + "".join(f"{r.sub},{r.ses},True\n" for r in rep.failed)
    )
    return rep
