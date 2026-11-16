"""``subseslist.txt`` — the work list. The only reader/writer in MRI-hub.

Format (comma-separated, header required)::

    sub,ses,RUN[,extra columns…]
    01,02,True
    sub-01,ses-03,false        # prefixes and any RUN case are accepted

Rules:

- the header's first three columns are ``sub,ses,RUN`` (case-insensitive);
  any further named columns are allowed (e.g. autoROI's ``sources``) and come
  back in ``Row.extras``;
- blank lines and ``#`` comments (whole-line or trailing) are ignored;
- ``sub-``/``ses-`` prefixes are stripped, numeric labels zero-padded to 2 digits;
- RUN must be true/false (any case); anything else is an error;
- a row with fewer than 3 or more cells than the header, an empty sub/ses, or a
  duplicate sub/ses (after normalisation, RUN=False rows included) is an error.

Only RUN=True rows are returned: a typo must stop the run, never silently drop
a session.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from launchcontainers.core import LCError
from launchcontainers.core.io import PathLike, atomic_write_text, bare

HEADER = ("sub", "ses", "RUN")


@dataclass(frozen=True)
class Row:
    sub: str  # bare, zero-padded: "01"
    ses: str
    extras: dict[str, str] = field(default_factory=dict, compare=False, hash=False)
    lineno: int = field(
        default=0, compare=False, hash=False
    )  # line in the file (for messages)


def read_subseslist(path: PathLike, extra: Sequence[str] = ()) -> list[Row]:
    """The RUN=True rows of ``path`` (see module docstring for the rules).

    ``extra``: extra columns the caller relies on. Each is guaranteed to be a
    key of ``Row.extras`` ("" when the file has no such column or the cell is
    empty). Undeclared extra columns are tolerated and also returned.
    """
    path = Path(path)
    if not path.is_file():
        raise LCError(f"subseslist not found: {path}")
    rows: list[Row] = []
    seen: set = set()
    header: list[str] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        cols = [c.strip() for c in line.split(",")]
        where = f"{path}:{lineno}"
        if not header:
            if [c.lower() for c in cols[:3]] != ["sub", "ses", "run"]:
                raise LCError(
                    f"{where}: expected header 'sub,ses,RUN[,extra…]', got {line!r}"
                )
            header = cols
            names = [c.lower() for c in header]
            if any(not c for c in header) or len(set(names)) != len(names):
                raise LCError(
                    f"{where}: empty or duplicate column name in header {line!r}"
                )
            continue
        if len(cols) < 3 or len(cols) > len(header) or not cols[0] or not cols[1]:
            raise LCError(f"{where}: malformed row {line!r} (want {','.join(header)})")
        run = cols[2].lower()
        if run not in ("true", "false"):
            raise LCError(f"{where}: RUN must be True or False, got {cols[2]!r}")
        sub, ses = bare(cols[0], "sub"), bare(cols[1], "ses")
        if not sub or not ses:
            raise LCError(f"{where}: empty sub or ses label in {line!r}")
        if (sub, ses) in seen:
            raise LCError(f"{where}: duplicate row for sub-{sub} ses-{ses}")
        seen.add((sub, ses))
        if run != "true":
            continue
        extras = {name: "" for name in extra}
        extras.update(
            {
                name: (cols[i] if i < len(cols) else "")
                for i, name in enumerate(header[3:], start=3)
            }
        )
        rows.append(Row(sub, ses, extras, lineno))
    if not header:
        raise LCError(f"{path}: empty work list (no 'sub,ses,RUN' header)")
    return rows


def read_pairs(path: PathLike) -> list[tuple[str, str]]:
    """The (sub, ses) pairs with RUN=True."""
    return [(r.sub, r.ses) for r in read_subseslist(path)]


def write_subseslist(
    path: PathLike, rows: Iterable[Row | tuple[str, str]], *, comment: str = ""
) -> Path:
    """Write a work list with every row RUN=True (seed it, then edit by hand).

    ``rows``: Row objects or (sub, ses) pairs. Extra columns are the union of
    the Rows' ``extras`` keys, in first-seen order. ``comment`` lines go on top
    as ``# …``. Written atomically.
    """
    norm: list[Row] = [r if isinstance(r, Row) else Row(*r) for r in rows]
    cols: list[str] = []
    for r in norm:
        cols += [k for k in r.extras if k not in cols]
    lines = [f"# {c}" for c in comment.splitlines()] if comment else []
    lines.append(",".join([*HEADER, *cols]))
    for r in norm:
        cells = [
            bare(r.sub, "sub"),
            bare(r.ses, "ses"),
            "True",
            *(str(r.extras.get(c, "")) for c in cols),
        ]
        if any("," in c or "\n" in c for c in cells):
            raise LCError(f"work-list cell contains ',' or a newline: {cells}")
        lines.append(",".join(cells))
    path = Path(path)
    atomic_write_text(path, "\n".join(lines) + "\n")
    return path
