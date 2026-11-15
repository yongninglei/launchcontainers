"""Golden snapshots: what `lc prepare` + the run commands produce, normalised.

A snapshot of one analysis dir is a JSON dict:

- ``tree``: every entry under ``sub-*/`` as ``{relpath: "dir" | "file" | "link -> target"}``
  plus the top-level config copies; link targets and paths are rewritten with
  ``<ROOT>`` so snapshots compare across machines.
- ``files``: text content of the small config files prepare writes
  (``*.json``, ``*.yaml``, ``*.csv``, ``*.txt``, ``*.bval``, ``*.bvec``).
- ``commands``: the run command per ``sub-XX_ses-YY`` (timestamps → ``<TS>``).

Logs (``prepare_log/``, ``logs/``, ``job_script_dir_*``) are excluded: they are
timestamped and are the part the restructure is meant to change.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

TS = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}(-\d{2})?")
TEXT_SUFFIXES = {
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".txt",
    ".bval",
    ".bvec",
    ".bvals",
    ".bvecs",
}
SKIP_TOP = ("prepare_log", "logs", "job_script_dir_", "run_log", "qc")


def norm(s: str, root: Path) -> str:
    s = s.replace(str(Path(root).resolve()), "<ROOT>").replace(str(root), "<ROOT>")
    return TS.sub("<TS>", s)


def snapshot_dir(analysis_dir: Path, root: Path) -> dict:
    analysis_dir = Path(analysis_dir)
    tree: dict[str, str] = {}
    files: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(analysis_dir, followlinks=False):
        rel_dir = Path(dirpath).relative_to(analysis_dir)
        if rel_dir == Path("."):
            dirnames[:] = [d for d in dirnames if not d.startswith(SKIP_TOP)]
        names = list(dirnames) + list(filenames)
        for n in names:
            p = Path(dirpath) / n
            rel = str(rel_dir / n) if rel_dir != Path(".") else n
            if p.is_symlink():
                tree[rel] = "link -> " + norm(os.readlink(p), root)
            elif p.is_dir():
                tree[rel] = "dir"
            else:
                tree[rel] = "file"
                if p.suffix in TEXT_SUFFIXES and p.stat().st_size < 200_000:
                    files[rel] = norm(p.read_text(errors="replace"), root)
        # do not descend into linked dirs
        dirnames[:] = [d for d in dirnames if not (Path(dirpath) / d).is_symlink()]
    return {"tree": dict(sorted(tree.items())), "files": dict(sorted(files.items()))}


def dump(snap: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n")


def diff(a: dict, b: dict) -> list[str]:
    """Human-readable differences between two snapshots (empty = identical)."""
    out = []
    for key in ("tree", "files", "commands", "status"):
        x, y = a.get(key, {}), b.get(key, {})
        if isinstance(x, dict):
            for k in sorted(set(x) | set(y)):
                if x.get(k) != y.get(k):
                    out.append(
                        f"{key}[{k}]: golden={x.get(k)!r:.200} new={y.get(k)!r:.200}"
                    )
        elif x != y:
            out.append(f"{key}: golden={x!r} new={y!r}")
    return out
