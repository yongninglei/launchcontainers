"""The ``lc_config.yaml`` shape shared by launchcontainers and every tool.

::

    general:                 # where the data are, which tool, which analysis
      basedir: /path/to/project
      bidsdir_name: BIDS
      tool: pyGLM            # key into container_specific (`container:` accepted as alias)
      analysis_name: main
      host: BCBL             # key into host_options          (runtime only)
      force: false           #                                 (runtime only)
      deriv_layout: …        # optional
    container_specific:
      pyGLM: { … }           # validated by the tool's load() → build_params()
    host_options:            # runtime only
      BCBL: { manager: sge, … }

``general`` may hold further keys (e.g. launchcontainers' ``containerdir``);
they are kept in ``raw``. A tool's own block is strict: unknown or missing
keys are errors (:func:`build_params`), so a typo never falls back to a default.

The *computational* part of a config (:func:`provenance_view`) is what the
analysis-dir guard compares: everything in ``general`` except the runtime-only
keys, plus the tool's own block. ``host_options`` and other tools' blocks are
never part of it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

from launchcontainers.core import LCError
from launchcontainers.core.io import PathLike

# general keys that do not change what gets computed. The whole host_options
# section is runtime-only too (provenance_view never includes it).
RUNTIME_ONLY_KEYS = ("force", "host")

T = TypeVar("T")


def require_str(value: Any, key: str) -> str:
    """``value`` must be a YAML string (``analysis_name: 01`` would be the int 1)."""
    if not isinstance(value, str):
        raise LCError(
            f"{key} must be a string, got {value!r} ({type(value).__name__}). "
            f'Quote it in the YAML, e.g. {key}: "01".'
        )
    return value


_require_str = require_str  # autoROI's private name


def provenance_view(
    raw: Mapping[str, Any], tool: str, runtime_only: Sequence[str] = RUNTIME_ONLY_KEYS
) -> dict[str, Any]:
    """The part of a raw config dict that determines ``tool``'s outputs.

    ``container`` is folded into ``tool`` so the alias never looks like a change.
    """
    general = dict(raw.get("general") or {})
    if "container" in general:
        container = general.pop("container")
        if general.get("tool") is None:
            general["tool"] = container
    general = {k: v for k, v in general.items() if k not in runtime_only}
    block = (raw.get("container_specific") or {}).get(tool)
    return {"general": general, "container_specific": {tool: block}}


def build_params(
    cls: type[T],
    block: Mapping[str, Any] | None,
    where: str,
    convert: Mapping[str, Callable[[Any], Any]] | None = None,
) -> T:
    """Instantiate dataclass ``cls`` from ``block``; unknown or missing keys are errors.

    ``where`` names the block in messages, e.g. ``"container_specific.pyGLM"``.
    ``convert`` maps a key to a function applied to its value after the key
    check (e.g. ``{"strategies": tuple, "power": lambda d: build_params(Power, d, ...)}``).
    Range checks stay in the tool.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"build_params needs a dataclass type, got {cls!r}")
    if block is None:
        block = {}
    if not isinstance(block, Mapping):
        raise LCError(f"{where} must be a mapping, got {type(block).__name__}")
    fields = dataclasses.fields(cls)
    known = {f.name for f in fields if f.init}
    unknown = sorted(set(block) - known)
    if unknown:
        raise LCError(
            f"Unknown key(s) in {where}: {unknown}. Valid keys: {sorted(known)}"
        )
    missing = sorted(
        f.name
        for f in fields
        if f.init
        and f.name not in block
        and f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
    )
    if missing:
        raise LCError(f"Missing required key(s) in {where}: {missing}")
    values = dict(block)
    for key, fn in (convert or {}).items():
        if key in values:
            values[key] = fn(values[key])
    return cls(**values)


@dataclass(frozen=True)
class LCConfig:
    basedir: Path
    bidsdir_name: str
    tool: str
    analysis_name: str
    host: str | None
    force: bool
    source_path: Path
    raw: dict[str, Any] = field(repr=False)
    deriv_layout: str | None = None

    @property
    def bids_dir(self) -> Path:
        return self.basedir / self.bidsdir_name

    @property
    def derivatives(self) -> Path:
        return self.bids_dir / "derivatives"

    def tool_block(self, name: str | None = None) -> dict[str, Any]:
        """``container_specific.<name>`` (default: ``general.tool``)."""
        name = name or self.tool
        blocks = self.raw.get("container_specific") or {}
        if name not in blocks:
            raise LCError(
                f"{self.source_path}: missing 'container_specific.{name}' section "
                f"(present: {sorted(blocks)})"
            )
        return dict(blocks[name] or {})

    def host_options(self, host: str | None = None) -> dict[str, Any]:
        """``host_options.<host>`` (default: ``general.host``)."""
        host = host or self.host
        opts = self.raw.get("host_options") or {}
        if not host:
            raise LCError(f"{self.source_path}: general.host is not set")
        if host not in opts:
            raise LCError(
                f"{self.source_path}: host {host!r} has no 'host_options.{host}' section (present: {sorted(opts)})"
            )
        return dict(opts[host] or {})

    def provenance_view(self, tool: str | None = None) -> dict[str, Any]:
        return provenance_view(self.raw, tool or self.tool)


def load_lc_config(path: PathLike, default_tool: str | None = None) -> LCConfig:
    """Read ``lc_config.yaml`` and validate its ``general`` section.

    ``default_tool``: used when ``general`` names no tool — for a tool's own
    standalone CLI, whose config needs no ``general.tool``.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise LCError(f"Config not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise LCError(f"{path}: not valid YAML: {exc}") from None
    return parse_lc_config(raw, path, default_tool=default_tool)


def parse_lc_config(
    raw: Any, path: PathLike, default_tool: str | None = None
) -> LCConfig:
    """Validate an already-parsed config dict (``path`` is recorded and used in messages)."""
    path = Path(path).expanduser().resolve()
    if not isinstance(raw, dict):
        raise LCError(f"{path}: top level must be a mapping")
    general = raw.get("general")
    if not isinstance(general, dict):
        raise LCError(f"{path}: missing 'general' section")
    for key in ("basedir", "bidsdir_name", "analysis_name"):
        if key not in general:
            raise LCError(f"{path}: missing general.{key}")
    tool, container = general.get("tool"), general.get("container")
    if tool is not None and container is not None and tool != container:
        raise LCError(
            f"{path}: general.tool {tool!r} and general.container {container!r} disagree"
        )
    tool = tool if tool is not None else container
    if tool is None:
        tool = default_tool
    if tool is None:
        raise LCError(f"{path}: missing general.tool (or its alias general.container)")
    for sect in ("container_specific", "host_options"):
        if raw.get(sect) is not None and not isinstance(raw[sect], dict):
            raise LCError(f"{path}: '{sect}' must be a mapping")
    host = general.get("host")
    layout = general.get("deriv_layout")
    return LCConfig(
        basedir=Path(str(general["basedir"])).expanduser(),
        bidsdir_name=str(general["bidsdir_name"]),
        tool=require_str(tool, "general.tool"),
        analysis_name=require_str(general["analysis_name"], "general.analysis_name"),
        host=None if host is None else str(host),
        force=bool(general.get("force", False)),
        source_path=path,
        raw=raw,
        deriv_layout=None if layout is None else str(layout),
    )
