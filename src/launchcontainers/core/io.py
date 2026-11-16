"""Small I/O helpers: subject/session labels, sha256, atomic writes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

PathLike = str | Path


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def bare(label: Any, prefix: str) -> str:
    """'sub-1' / '1' / '01' / 'sub-01' → '01'. Non-numeric labels are kept as they are.

    Numeric labels are zero-padded to two digits ('7' → '07'; '123' stays '123').
    """
    v = str(label).strip()
    if v.startswith(f"{prefix}-"):
        v = v[len(prefix) + 1 :]
    return v.zfill(2) if v.isdigit() else v


def norm_label(value: Any, prefix: str) -> str:
    """Alias of :func:`bare` (pyGLM's old name)."""
    return bare(value, prefix)


def sub_ses(sub: Any, ses: Any) -> tuple[str, str]:
    """Any form → ('sub-01', 'ses-01')."""
    return f"sub-{bare(sub, 'sub')}", f"ses-{bare(ses, 'ses')}"


# ---------------------------------------------------------------------------
# Checksums and atomic writes
# ---------------------------------------------------------------------------


def sha256(path: PathLike, _chunk: int = 1 << 20) -> str:
    """``"sha256:<hex>"`` of a file's content."""
    return "sha256:" + sha256_hex(path, _chunk)


def sha256_hex(path: PathLike, _chunk: int = 1 << 20) -> str:
    """Bare hex digest (the form runs.tsv stores)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(_chunk):
            h.update(block)
    return h.hexdigest()


def atomic_write(path: PathLike, write: Callable[[str], Any]) -> None:
    """Call ``write(tmp_path)`` on a temp file in the same dir, then rename it to ``path``.

    No half-written outputs. The temp name keeps the full extension, so writers
    that dispatch on it (nibabel) still work.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix="_" + path.name)
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):  # only when write() failed: our own temp file
            os.unlink(tmp)


# autoROI's private name
_atomic = atomic_write


def atomic_write_text(path: PathLike, text: str) -> None:
    atomic_write(path, lambda tmp: Path(tmp).write_text(text))


def atomic_write_json(path: PathLike, obj: Any) -> None:
    """Indented JSON + newline; non-JSON values (Path, datetime) go through ``str``."""
    atomic_write_text(path, json.dumps(obj, indent=2, default=str) + "\n")
