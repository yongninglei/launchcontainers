"""Run logs and provenance — the master (and only) copy, moved from pyGLM.

While a batch runs, everything goes to the project:

    {project}/logs/{tool}/{analysis_name}[_sub-XX_ses-YY]_{stamp}/
    ├── lc_config.yaml      the config as launched
    ├── subseslist.txt      the work list as launched
    ├── run.json            command, host, user, versions, git commit, checksums,
    │                       start/end, exit code, per-session status
    ├── sub-XX_ses-YY.log   per-session logs (tools that write them)
    └── *.out / *.err       the bash launcher's logs, added after the batch exits

(``project_dir=None``: the run folder is ``{first analysis dir}/logs/<run_id>/``
directly, and is not copied onto itself.)

When the batch finishes (also when it fails), for every analysis dir it wrote:

    {analysis_dir}/
    ├── lc_config.yaml      latest config; a different earlier one → logs/history/
    ├── subseslist.txt      every session run in this analysis, over all batches
    └── logs/
        ├── {run_id}/       copy of the project run folder (one per batch)
        ├── runs.tsv        one row per batch: when, exit code, ok/failed, host, version, commit
        └── sessions.tsv    one row per session per batch; the last row for a sub/ses
                            names the batch that produced its current outputs

runs.tsv and sessions.tsv are append-only (one write per batch, so concurrent
batches never overwrite each other's rows) and subseslist.txt is rebuilt from
sessions.tsv rather than replaced by the latest batch's list. The batch then
prints one ``PROVENANCE <dir>`` line per destination (``Run.markers()``).

**Inside the session wrapper** (``lc-session``, which sets
``ANALYSIS_RUN_ID`` and ``ANALYSIS_RUN_DIR`` for its child), a tool's own
``start_run`` does not open a second run: it returns a *nested* Run that
writes ``tool_<tool>.json`` (same fields as run.json) into the wrapper's run
folder and copies its config there if the wrapper had none. ``finish_run`` on
a nested Run only closes that record — it appends no TSV rows and copies
nothing. When the wrapper finishes, it reads those records: their
AnalysisDirs are added to its destinations, and their per-session statuses
(e.g. ``skipped``) replace the plain ok/failed of the exit code. So every run
is recorded exactly once.
"""

from __future__ import annotations

import datetime
import getpass
import json
import os
import platform
import shutil
import socket
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from launchcontainers.core.io import atomic_write_json, atomic_write_text, sha256_hex
from launchcontainers.core.provenance import _git, package_versions

CONFIG_NAME = "lc_config.yaml"
SUBSES_NAME = "subseslist.txt"
RECORD_NAME = "run.json"
NESTED_PREFIX = "tool_"
RUNS_TSV = "runs.tsv"
SESSIONS_TSV = "sessions.tsv"
MARKER = "PROVENANCE"
ENV_RUN_ID = "ANALYSIS_RUN_ID"
ENV_RUN_DIR = "ANALYSIS_RUN_DIR"
_RUNS_COLS = (
    "run_id",
    "tool",
    "version",
    "commit",
    "started",
    "finished",
    "exit_code",
    "n_ok",
    "n_failed",
    "host",
    "user",
    "config_sha256",
    "subseslist_sha256",
)
_SESSIONS_COLS = ("sub", "ses", "status", "run_id", "finished")


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


_write_json = atomic_write_json


@dataclass
class Run:
    tool: str
    run_id: str
    run_dir: Path
    analysis_dirs: list[Path]
    record: dict[str, Any] = field(repr=False)
    copied_to: list[Path] = field(default_factory=list)
    nested: bool = False  # True: inside lc-session (see module docstring)
    record_path: Path | None = None  # run.json, or tool_<tool>.json when nested

    def __post_init__(self) -> None:
        if self.record_path is None:
            self.record_path = self.run_dir / RECORD_NAME

    def markers(self) -> list[str]:
        """``PROVENANCE <dir>`` lines for the bash launcher (project copy first)."""
        return [f"{MARKER} {d}" for d in [self.run_dir, *self.copied_to]]


def _file_info(path: Path | None) -> dict[str, str] | None:
    return (
        None
        if path is None
        else {"path": str(Path(path).resolve()), "sha256": sha256_hex(path)}
    )


def _exclusive(path: Path) -> Path:
    """Create ``path`` (or ``path-2``, …) as an empty placeholder and return it."""
    n = 1
    cand = path
    while True:
        try:
            with open(cand, "x"):
                return cand
        except FileExistsError:
            n += 1
            cand = path.with_name(f"{path.stem}-{n}{path.suffix}")


def start_run(
    tool: str,
    project_dir: Path | None,
    analysis_name: str,
    analysis_dirs: Sequence[Path],
    config: Path | None,
    subseslist: Path | None,
    *,
    version: str | None,
    code_dir: Path | None = None,
    packages: Iterable[str] = (),
    scope: str = "",
) -> Run:
    """Create the run folder and copy config + work list into it.

    ``scope``: e.g. ``sub-01_ses-02`` for a single-session run; empty for a batch.
    ``project_dir=None``: run folder directly under ``analysis_dirs[0]/logs/``.
    ``config=None``: nothing to copy (config fields of the record stay empty).
    With ``ANALYSIS_RUN_ID`` set, returns a nested Run (module docstring).
    """
    commit = _git(code_dir, "rev-parse", "HEAD") if code_dir is not None else None
    record: dict[str, Any] = {
        "Tool": tool,
        "Version": version,
        "CodeCommit": commit,
        "CodeDirty": bool(
            _git(code_dir, "status", "--porcelain", "--untracked-files=no")
        )
        if commit
        else None,
        "RunId": None,
        "Command": sys.argv,
        "WorkingDirectory": os.getcwd(),
        "Host": socket.gethostname(),
        "User": getpass.getuser(),
        "Python": sys.version.split()[0],
        "Platform": platform.platform(),
        "Packages": package_versions(packages),
        "Config": _file_info(config),
        "SubSesList": _file_info(subseslist),
        "AnalysisDirs": [str(a) for a in analysis_dirs],
        "Started": _now(),
        "Finished": None,
        "ExitCode": None,
        "Sessions": [],
    }

    env_id, env_dir = os.environ.get(ENV_RUN_ID), os.environ.get(ENV_RUN_DIR)
    if env_id and env_dir:
        run_dir = Path(env_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        if config is not None and not (run_dir / CONFIG_NAME).exists():
            shutil.copyfile(config, run_dir / CONFIG_NAME)
        if subseslist is not None and not (run_dir / SUBSES_NAME).exists():
            shutil.copyfile(subseslist, run_dir / SUBSES_NAME)
        record.update(RunId=env_id, Nested=True)
        path = _exclusive(run_dir / f"{NESTED_PREFIX}{tool}.json")
        _write_json(path, record)
        return Run(
            tool,
            env_id,
            run_dir,
            [Path(a) for a in analysis_dirs],
            record,
            nested=True,
            record_path=path,
        )

    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    if project_dir is not None:
        root = Path(project_dir) / "logs" / tool
    else:
        if not analysis_dirs:
            raise ValueError(
                "start_run: project_dir=None needs at least one analysis dir"
            )
        root = Path(analysis_dirs[0]) / "logs"
    base = f"{analysis_name}_{scope}" if scope else analysis_name
    run_id, n = f"{base}_{stamp}", 1
    while True:
        try:
            (root / run_id).mkdir(parents=True)
            break
        except FileExistsError:  # two runs started in the same second
            n += 1
            run_id = f"{base}_{stamp}-{n}"
    run_dir = root / run_id

    if config is not None:
        shutil.copyfile(config, run_dir / CONFIG_NAME)
    if subseslist is not None:
        shutil.copyfile(subseslist, run_dir / SUBSES_NAME)
    record["RunId"] = run_id
    _write_json(run_dir / RECORD_NAME, record)
    return Run(tool, run_id, run_dir, [Path(a) for a in analysis_dirs], record)


def _publish(src: Path, dst: Path, analysis_dir: Path) -> None:
    """Put ``src`` at ``dst``; a different existing copy moves to logs/history/."""
    if dst.exists():
        if dst.read_bytes() == src.read_bytes():
            return
        stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        hist = analysis_dir / "logs" / "history" / f"{dst.stem}_{stamp}{dst.suffix}"
        hist.parent.mkdir(parents=True, exist_ok=True)
        os.replace(dst, hist)
    shutil.copyfile(src, dst)


def _append_tsv(path: Path, cols: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    """Append rows in a single write (header first if the file is new)."""
    new = not path.exists()
    text = ("\t".join(cols) + "\n" if new else "") + "".join(
        "\t".join("" if v is None else str(v) for v in row) + "\n" for row in rows
    )
    with open(path, "a") as f:
        f.write(text)


def _status(s: dict[str, Any]) -> str:
    return str(s.get("status") or ("ok" if s.get("ok") else "failed"))


def _rebuild_subseslist(analysis_dir: Path) -> None:
    """Top-level subseslist.txt = every sub/ses in logs/sessions.tsv, RUN=True."""
    rows = (analysis_dir / "logs" / SESSIONS_TSV).read_text().splitlines()[1:]
    pairs = sorted({tuple(r.split("\t")[:2]) for r in rows if r.strip()})
    text = (
        "# Every session run in this analysis, over all batches (rebuilt by each batch).\n"
        "# Per-batch lists: logs/<run>/subseslist.txt; latest status per session: logs/sessions.tsv\n"
        "sub,ses,RUN\n" + "".join(f"{s},{e},True\n" for s, e in pairs)
    )
    atomic_write_text(analysis_dir / SUBSES_NAME, text)


def nested_records(run_dir: Path) -> list[dict[str, Any]]:
    """The ``tool_*.json`` records tools wrote into a wrapper's run folder."""
    out = []
    for p in sorted(Path(run_dir).glob(f"{NESTED_PREFIX}*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    return out


def finish_run(
    run: Run,
    exit_code: int,
    sessions: Sequence[dict[str, Any]] = (),
    *,
    copy_to_analysis: bool = True,
    publish_config: bool = True,
) -> Run:
    """Close the run record, then record the batch in every analysis dir.

    ``sessions``: one dict per session with ``sub``, ``ses`` and ``status``
    (or a boolean ``ok``). ``publish_config=False`` for tools that manage the
    top-level lc_config.yaml themselves (autoROI refuses a changed config).
    A nested Run only closes its own record (module docstring).
    """
    finished = _now()
    run.record.update(Finished=finished, ExitCode=exit_code, Sessions=list(sessions))
    if run.nested:
        run.record["PublishConfig"] = publish_config
        _write_json(run.record_path, run.record)
        return run
    _write_json(run.run_dir / RECORD_NAME, run.record)
    if not copy_to_analysis:
        return run

    # destinations: our analysis dirs + those of nested tool records (wrapper case)
    dests: dict[Path, bool] = {Path(a): publish_config for a in run.analysis_dirs}
    for inner in nested_records(run.run_dir):
        for a in inner.get("AnalysisDirs") or []:
            a = Path(a)
            dests[a] = dests.get(a, False) or (
                publish_config and inner.get("PublishConfig", True)
            )

    rec = run.record
    n_ok = sum(_status(s) == "ok" for s in sessions)
    run_row = (
        run.run_id,
        run.tool,
        rec["Version"],
        rec["CodeCommit"],
        rec["Started"],
        finished,
        exit_code,
        n_ok,
        len(sessions) - n_ok,
        rec["Host"],
        rec["User"],
        (rec["Config"] or {}).get("sha256"),
        (rec["SubSesList"] or {}).get("sha256"),
    )
    session_rows = [
        (s["sub"], s["ses"], _status(s), run.run_id, finished) for s in sessions
    ]
    for ana, publish in dests.items():
        (ana / "logs").mkdir(parents=True, exist_ok=True)
        dest = ana / "logs" / run.run_id
        if dest.resolve() != run.run_dir.resolve():
            shutil.copytree(run.run_dir, dest, dirs_exist_ok=True)
            run.copied_to.append(dest)
        _append_tsv(ana / "logs" / RUNS_TSV, _RUNS_COLS, [run_row])
        if session_rows:
            _append_tsv(ana / "logs" / SESSIONS_TSV, _SESSIONS_COLS, session_rows)
            _rebuild_subseslist(ana)
        if publish and (run.run_dir / CONFIG_NAME).exists():
            _publish(run.run_dir / CONFIG_NAME, ana / CONFIG_NAME, ana)
    return run
