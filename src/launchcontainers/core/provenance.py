"""Provenance of an analysis directory (ARCHITECTURE.md §5).

::

    <analysis dir>/
    ├── dataset_description.json   GeneratedBy, SourceDatasets / DatasetLinks (bids: URIs)
    ├── lc_config.yaml             the config it was made with (guarded)
    ├── .bidsignore
    └── logs/history/              earlier configs / descriptions — never deleted

One analysis dir = one config. :func:`guard_analysis_dir` refuses a config
whose computational keys differ (config.provenance_view), and a tool version
different from the one recorded in dataset_description.json, unless
``force`` — then the old file moves to ``logs/history/``. Nothing is deleted.
"""

from __future__ import annotations

import datetime
import getpass
import json
import os
import platform
import socket
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

from launchcontainers.core import LCError
from launchcontainers.core.config import RUNTIME_ONLY_KEYS, LCConfig, provenance_view
from launchcontainers.core.io import (
    PathLike,
    atomic_write_json,
    atomic_write_text,
    sha256,
)

CONFIG_NAME = "lc_config.yaml"
DESCRIPTION_NAME = "dataset_description.json"
BIDS_VERSION = "1.9.0"
BIDSIGNORE = ("logs/", "qc/", "lc_config.yaml", "subseslist.txt")


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _git(where: PathLike, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(where), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def software_info(
    name: str,
    version: str,
    code_url: str | None = None,
    code_dir: PathLike | None = None,
) -> dict[str, Any]:
    """BIDS ``GeneratedBy`` entry. CodeCommit/CodeDirty when ``code_dir`` is a git checkout."""
    info: dict[str, Any] = {"Name": name, "Version": version}
    if code_url:
        info["CodeURL"] = code_url
    if code_dir is not None:
        commit = _git(code_dir, "rev-parse", "HEAD")
        if commit:
            info["CodeCommit"] = commit
            info["CodeDirty"] = bool(
                _git(code_dir, "status", "--porcelain", "--untracked-files=no")
            )
    return info


def package_versions(packages: Iterable[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for pkg in packages:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def environment_info(packages: Iterable[str] = ()) -> dict[str, Any]:
    """Python, platform, host, user and the installed versions of ``packages``."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "packages": package_versions(packages),
    }


def flat(d: Any, prefix: str = "") -> dict[str, Any]:
    """Nested dict → {"a.b.c": value} (for naming the keys that differ)."""
    if not isinstance(d, dict):
        return {prefix: d}
    out: dict[str, Any] = {}
    for k, v in d.items():
        out.update(flat(v, f"{prefix}.{k}" if prefix else str(k)))
    return out


_flat = flat


def archive(path: PathLike, analysis_dir: PathLike) -> Path:
    """Move ``path`` into ``<analysis_dir>/logs/history/`` with a timestamp — never delete."""
    path = Path(path)
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    dest = Path(analysis_dir) / "logs" / "history" / f"{path.stem}_{stamp}{path.suffix}"
    n = 1
    while dest.exists():
        n += 1
        dest = dest.with_name(f"{path.stem}_{stamp}-{n}{path.suffix}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(path, dest)
    return dest


_archive = archive


def _recorded_version(dd: Mapping[str, Any], tool_name: str) -> str | None:
    gens = dd.get("GeneratedBy") or [{}]
    for g in gens:
        if isinstance(g, dict) and g.get("Name") == tool_name:
            return g.get("Version")
    return gens[0].get("Version") if isinstance(gens[0], dict) else None


def guard_analysis_dir(
    analysis_dir: PathLike,
    cfg: LCConfig,
    tool_name: str,
    version: str,
    force: bool | None = None,
    runtime_only: Sequence[str] = (),
) -> list[str]:
    """Create or validate ``analysis_dir``; return notes about what changed.

    - **config guard**: ``lc_config.yaml`` in the dir is compared with ``cfg`` on
      the computational keys only (runtime-only keys: general.host/force and
      host_options, plus the tool's own ``runtime_only`` general keys). Different → refused, unless ``force`` (default
      ``cfg.force``): the old copy moves to logs/history/ and ``cfg`` is copied in.
    - **version guard**: the ``tool_name`` version in dataset_description.json
      must equal ``version``; with ``force`` the old description is archived
      (write a new one with :func:`write_dataset_description`).
    - writes ``.bidsignore`` if missing.

    Safe to call from every single-session job: when nothing changed it only reads.
    """
    ana = Path(analysis_dir)
    force = cfg.force if force is None else force
    (ana / "logs").mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    cfg_copy = ana / CONFIG_NAME
    if cfg_copy.exists():
        skip = (*RUNTIME_ONLY_KEYS, *runtime_only)
        old_view = provenance_view(
            yaml.safe_load(cfg_copy.read_text()) or {}, tool_name, skip
        )
        new_view = provenance_view(cfg.raw, tool_name, skip)
        if old_view != new_view:
            a, b = flat(old_view), flat(new_view)
            diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            if not force:
                raise LCError(
                    f"{ana} was created with a different config (keys differ: {diff}). "
                    "Use a new analysis_name, or set general.force: true to overwrite "
                    "(the old config is kept in logs/history/)."
                )
            notes.append(
                f"config changed ({diff}); old copy → {archive(cfg_copy, ana)}"
            )
    if not cfg_copy.exists():
        atomic_write_text(cfg_copy, cfg.source_path.read_text())

    dd_path = ana / DESCRIPTION_NAME
    if dd_path.exists():
        old_version = _recorded_version(json.loads(dd_path.read_text()), tool_name)
        if old_version != version:
            if not force:
                raise LCError(
                    f"{ana} was produced by {tool_name} {old_version}; this is {version}. "
                    "Use a new analysis_name, or set general.force: true."
                )
            notes.append(
                f"{tool_name} version {old_version} → {version}; old description → {archive(dd_path, ana)}"
            )

    bidsignore = ana / ".bidsignore"
    if not bidsignore.exists():
        atomic_write_text(bidsignore, "\n".join(BIDSIGNORE) + "\n")
    return notes


# ---------------------------------------------------------------------------
# dataset_description.json and bids: URIs
# ---------------------------------------------------------------------------


def _rel(target: PathLike, start: PathLike) -> str:
    return os.path.relpath(Path(target).resolve(), Path(start).resolve())


def dataset_links(
    analysis_dir: PathLike, source_datasets: Mapping[str, PathLike]
) -> dict[str, str]:
    """{link: path of that dataset relative to analysis_dir}."""
    return {link: _rel(p, analysis_dir) for link, p in source_datasets.items()}


def write_dataset_description(
    analysis_dir: PathLike,
    name: str,
    description: str,
    generated_by: dict[str, Any] | Sequence[dict[str, Any]],
    source_datasets: Mapping[str, PathLike],
) -> Path:
    """BIDS-derivative dataset_description.json; rewritten only when its content changes.

    ``source_datasets``: {link name: dataset root}; becomes SourceDatasets
    ``bids:<link>:`` and DatasetLinks (relative paths). When the Description or the
    sources change, the old file is archived to ``logs/history/`` first, so what an
    analysis was said to be and built from is never lost.
    """
    path = Path(analysis_dir) / DESCRIPTION_NAME
    gens = [generated_by] if isinstance(generated_by, dict) else list(generated_by)
    dd = {
        "Name": name,
        "BIDSVersion": BIDS_VERSION,
        "DatasetType": "derivative",
        "Description": description,
        "GeneratedBy": gens,
        "SourceDatasets": [{"URI": f"bids:{link}:"} for link in source_datasets],
        "DatasetLinks": dataset_links(analysis_dir, source_datasets),
    }
    old = json.loads(path.read_text()) if path.exists() else None
    if old != dd:
        if old is not None and any(
            old.get(k) != dd[k]
            for k in ("Description", "SourceDatasets", "DatasetLinks")
        ):
            archive(path, analysis_dir)
        atomic_write_json(path, dd)
    return path


def bids_uri(
    analysis_dir: PathLike, links: Mapping[str, PathLike], link: str, path: PathLike
) -> str:
    """``bids:<link>:<path relative to that dataset>``; link "self" → ``bids::<path in analysis_dir>``."""
    if link == "self":
        return f"bids::{_rel(path, analysis_dir)}"
    if link not in links:
        raise LCError(
            f"unknown dataset link {link!r} (known: {sorted(links)} and 'self')"
        )
    return f"bids:{link}:{_rel(path, links[link])}"


def sidecar(
    analysis_dir: PathLike,
    links: Mapping[str, PathLike],
    sources: Iterable[tuple[str, PathLike]],
    parameters: Mapping[str, Any],
    generated_by: dict[str, Any],
    *,
    description: str = "",
    summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """JSON sidecar for one output: Sources (bids: URIs) + sha256 of each, parameters, software."""
    uris, checksums = [], {}
    for link, p in sources:
        uri = bids_uri(analysis_dir, links, link, p)
        uris.append(uri)
        checksums[uri] = sha256(p)
    out: dict[str, Any] = {}
    if description:
        out["Description"] = description
    out.update(
        {"Sources": uris, "SourceChecksums": checksums, "Parameters": dict(parameters)}
    )
    if summary is not None:
        out["Summary"] = dict(summary)
    out.update({"GeneratedBy": generated_by, "DateCreated": now()})
    return out
