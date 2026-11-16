"""`lc qc`: did each session's launch work? (No image QC.)

For every session in the work list (``<workdir>/subseslist.txt`` or ``-ssl``):

- ``last run``: the latest non-prepare row for the session in
  ``logs/sessions.tsv`` (written by ``lc-session``): ok / failed / never run;
- ``run log``: that run's session log (``logs/<run_id>/sub-XX_ses-YY.log``)
  scanned for tracebacks and error lines;
- ``tool.qc(tc, sub, ses)``: the tool's own launch checks — its expected
  outputs exist and are not empty (a crash becomes a ``fail`` check).

Writes, via ``launchcontainers.core.qc.render_report``, ``<workdir>/qc/index.html``,
``qc/sub-XX_ses-YY.html`` and ``qc/qc.tsv``; and ``qc/failed_subseslist.txt``
(``sub,ses,RUN`` with every session that has a ``fail``), ready for
``lc run -w <workdir> -ssl qc/failed_subseslist.txt``.

Looking at the images is the image-QC module's job (QC-registration).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from launchcontainers.core import LCError
from launchcontainers.core.qc import render_report
from launchcontainers.core.runlog import CONFIG_NAME, SESSIONS_TSV, SUBSES_NAME
from launchcontainers.core.tool import QCCheck, QCResult

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


_ERROR_LINE = re.compile(
    r"(?im)^.*(traceback \(most recent call last\)|\berror\b|\bkilled\b|segmentation fault).*$"
)


def _log_check(
    workdir: Path, row: dict[str, str] | None, sub: str, ses: str
) -> QCCheck | None:
    """Scan the last run's session log (lc-session) for error lines."""
    if row is None:
        return None
    log = workdir / "logs" / row.get("run_id", "") / f"sub-{sub}_ses-{ses}.log"
    if not log.is_file():
        return QCCheck("run log", "warn", f"no log at {log}")
    bad = _ERROR_LINE.search(log.read_text(errors="replace"))
    if bad:
        return QCCheck("run log", "warn", bad.group(0).strip()[:200])
    return QCCheck("run log", "ok", log.name)


def qc(workdir: Path, subseslist: Path | None = None, console=None) -> QCReport:
    """Launch checks per session, then the static report and the failed list."""
    workdir = Path(workdir).expanduser().resolve()
    lc_config = workdir / CONFIG_NAME
    if not lc_config.is_file():
        raise LCError(f"{workdir}: no {CONFIG_NAME}; run `lc prepare` first")
    ssl = (
        Path(subseslist).expanduser().resolve() if subseslist else workdir / SUBSES_NAME
    )
    if not ssl.is_file():
        raise LCError(f"work list not found: {ssl}")
    tool, tc, rows = lc_prepare.load(lc_config, ssl, None)
    rep = QCReport(tool.name, workdir)
    runs = _last_run(workdir)

    for r in rows:
        try:
            res = tool.qc(tc, r.sub, r.ses)
        except Exception as exc:  # noqa: BLE001 — one broken session must not stop the report
            res = QCResult(
                r.sub, r.ses, [QCCheck("qc", "fail", f"{type(exc).__name__}: {exc}")]
            )
        last = runs.get((r.sub, r.ses))
        log_check = _log_check(workdir, last, r.sub, r.ses)
        res.checks[:0] = [_run_check(last)] + ([log_check] if log_check else [])
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
