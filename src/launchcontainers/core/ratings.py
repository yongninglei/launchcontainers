"""Raters' visual-QC verdicts: one TSV per rater (ported from QC-registration).

``<analysis_dir>/qc/qc_rater-<name>.tsv``, one row per rated (item, sub, ses, run)::

    item  sub  ses  run  ok  comment  rater  time

``ok`` is ``True`` / ``False``; ``run`` may be ``""`` for figures that are not
run-specific. A file per rater means two raters on two machines never write
the same file. Each save rewrites the file atomically (temp file + rename),
so a crash never leaves half a table.
"""

from __future__ import annotations

import csv
import datetime
import os
import re
import threading
from pathlib import Path

from launchcontainers.core import LCError
from launchcontainers.core.io import PathLike

COLUMNS = ("item", "sub", "ses", "run", "ok", "comment", "rater", "time")
RATER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
QC_DIR = "qc"
Key = tuple[str, str, str, str]  # item, sub, ses, run
_lock = threading.Lock()  # one viewer process serves several request threads


def check_rater(rater: str) -> str:
    rater = (rater or "").strip()
    if not RATER_RE.match(rater):
        raise LCError("rater name: letters, digits, '-' or '_' (max 64), e.g. tiger")
    return rater


def ratings_path(analysis_dir: PathLike, rater: str) -> Path:
    return Path(analysis_dir) / QC_DIR / f"qc_rater-{check_rater(rater)}.tsv"


def _read(path: Path) -> dict[Key, dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    return {(r["item"], r["sub"], r["ses"], r["run"] or ""): r for r in rows}


def load_ratings(analysis_dir: PathLike, rater: str) -> dict[Key, dict[str, str]]:
    """{(item, sub, ses, run): row} for one rater; {} if they have not rated yet."""
    path = ratings_path(analysis_dir, rater)
    return _read(path) if path.is_file() else {}


def load_all_ratings(analysis_dir: PathLike) -> dict[str, dict[Key, dict[str, str]]]:
    """{rater: {key: row}} for every ``qc_rater-*.tsv`` in ``<analysis_dir>/qc/``."""
    out = {}
    for path in sorted((Path(analysis_dir) / QC_DIR).glob("qc_rater-*.tsv")):
        out[path.stem[len("qc_rater-") :]] = _read(path)
    return out


def latest_verdicts(analysis_dir: PathLike) -> dict[Key, dict[str, str]]:
    """Per (item, sub, ses, run): the most recent row over all raters."""
    best: dict[Key, dict[str, str]] = {}
    for rows in load_all_ratings(analysis_dir).values():
        for key, row in rows.items():
            if key not in best or row.get("time", "") > best[key].get("time", ""):
                best[key] = row
    return best


def save_rating(
    analysis_dir: PathLike, rater: str, key: Key, ok: bool, comment: str = ""
) -> dict[str, str]:
    """Set (replace) one item's verdict and comment; returns the stored row."""
    if not isinstance(ok, bool):
        raise LCError("ok must be true or false")
    if len(key) != 4 or not all(isinstance(k, str) for k in key) or not all(key[:3]):
        raise LCError("item, sub and ses are required (run may be empty)")
    if any("\t" in k or "\n" in k for k in key):
        raise LCError("item/sub/ses/run must not contain tabs or newlines")
    rater = check_rater(rater)
    with _lock:
        rows = load_ratings(analysis_dir, rater)
        row = dict(
            zip(
                COLUMNS,
                (
                    *key,
                    str(ok),
                    " ".join(str(comment or "").split()),
                    rater,
                    datetime.datetime.now().isoformat(timespec="seconds"),
                ),
            )
        )
        rows[tuple(key)] = row
        path = ratings_path(analysis_dir, rater)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with tmp.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, COLUMNS, delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows[k] for k in sorted(rows))
        os.replace(tmp, path)
    return row
