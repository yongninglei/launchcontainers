"""The Tool interface (ARCHITECTURE.md §2) and run-time discovery.

A software joins MRI-hub by implementing :class:`Tool` and declaring it in its
``pyproject.toml``::

    [project.entry-points."launchcontainers.tools"]
    pyGLM = "pyglm.lc_tool:TOOL"

The entry point may name a Tool instance or a zero-argument class.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from launchcontainers.core import LCError
from launchcontainers.core.config import LCConfig
from launchcontainers.core.subseslist import Row

ENTRY_POINT_GROUP = "launchcontainers.tools"
SESSION_STATUSES = ("ok", "prepared", "blocked", "failed", "skipped")
CHECK_STATUSES = ("ok", "warn", "fail")


@dataclass
class ToolConfig:
    general: LCConfig
    params: Any  # the tool's validated parameters (usually a dataclass)
    extra: dict[str, Any] = field(default_factory=dict)
    # The launcher's work list: (sub, ses) → Row, set before prepare/command/qc
    # are called, so a tool can read extra columns (e.g. autoROI's ``sources``).
    rows: dict[tuple[str, str], Row] = field(default_factory=dict)

    def row(self, sub: str, ses: str) -> Row | None:
        """The work-list row for ``sub``/``ses``, or None when run without a list."""
        return self.rows.get((sub, ses))


@dataclass
class SessionResult:
    sub: str
    ses: str
    status: str  # one of SESSION_STATUSES
    messages: list[str] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.status not in SESSION_STATUSES:
            raise ValueError(
                f"SessionResult.status must be one of {SESSION_STATUSES}, got {self.status!r}"
            )


@dataclass
class QCCheck:
    name: str
    status: str  # ok | warn | fail
    message: str = ""

    def __post_init__(self) -> None:
        if self.status not in CHECK_STATUSES:
            raise ValueError(
                f"QCCheck.status must be one of {CHECK_STATUSES}, got {self.status!r}"
            )


@dataclass
class Figure:
    """One rendered QC image (PNG/SVG) or HTML fragment, for the rater/viewer.

    ``item``: what is shown (a map, tract, registration…); ``run``: which run
    (``""`` when not run-specific). Rater verdicts are keyed by (item, sub, ses, run).
    """

    item: str
    run: str
    path: Path


@dataclass
class QCResult:
    sub: str
    ses: str
    checks: list[QCCheck] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    html: str = ""  # optional extra HTML fragment for the session page

    @property
    def status(self) -> str:
        """Worst check status; "none" without checks."""
        if not self.checks:
            return "none"
        return max((c.status for c in self.checks), key=CHECK_STATUSES.index)


@runtime_checkable
class Tool(Protocol):
    name: str  # key under container_specific in lc_config.yaml
    version: str

    def load(self, cfg: LCConfig) -> ToolConfig:
        """Validate container_specific.<name>; unknown or missing keys are errors."""
        ...

    def analysis_dirs(self, tc: ToolConfig) -> list[Path]:
        """Where this analysis writes (pyGLM: one per strategy)."""
        ...

    def prepare(self, tc: ToolConfig, sub: str, ses: str) -> SessionResult:
        """Stage inputs for one session. Idempotent; raises nothing — returns status."""
        ...

    def command(self, tc: ToolConfig, sub: str, ses: str) -> list[str]:
        """The one command that processes one session."""
        ...

    def qc(self, tc: ToolConfig, sub: str, ses: str) -> QCResult:
        """Checks + figures for one session's outputs."""
        ...

    # Optional (not part of the runtime check):
    #
    # def prepare_analysis(self, tc, pairs) -> list[tuple[str, str]] | None:
    #     """Analysis-level staging, called once by the launcher after the
    #     analysis dir is guarded and before any per-session prepare (e.g. write
    #     the container's config.json once). May return the sessions reordered
    #     (e.g. source sessions first); None keeps the order. Raising aborts
    #     the whole prepare."""


def prepare_analysis(
    tool: Tool, tc: ToolConfig, pairs: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Call the tool's optional ``prepare_analysis`` hook; return the session order to use."""
    hook = getattr(tool, "prepare_analysis", None)
    if hook is None:
        return list(pairs)
    out = hook(tc, list(pairs))
    if out is None:
        return list(pairs)
    out = list(out)
    if sorted(out) != sorted(pairs):
        raise LCError(
            f"{tool.name}.prepare_analysis must reorder the sessions, not add or drop any"
        )
    return out


def _entry_points() -> dict[str, metadata.EntryPoint]:
    return {ep.name: ep for ep in metadata.entry_points(group=ENTRY_POINT_GROUP)}


def _load(ep: metadata.EntryPoint) -> Tool:
    obj = ep.load()
    if isinstance(obj, type):
        obj = obj()
    if not isinstance(obj, Tool):
        raise LCError(
            f"entry point {ep.name} = {ep.value} does not implement the Tool interface"
        )
    return obj


def available_tools() -> list[str]:
    """Names of the registered tools, without importing them."""
    return sorted(_entry_points())


def tool_version(name: str) -> str | None:
    """Version of the distribution that provides tool ``name``, without importing it."""
    ep = _entry_points().get(name)
    dist = getattr(ep, "dist", None) if ep is not None else None
    return getattr(dist, "version", None)


def discover_tools() -> dict[str, Tool]:
    """Load every registered tool. A tool that fails to load is skipped with a warning."""
    tools: dict[str, Tool] = {}
    for name, ep in sorted(_entry_points().items()):
        try:
            tools[name] = _load(ep)
        except Exception as exc:  # noqa: BLE001 — a broken plug-in must not hide the others
            warnings.warn(
                f"launchcontainers.core: tool {name!r} ({ep.value}) failed to load: {exc}",
                stacklevel=2,
            )
    return tools


def get_tool(name: str) -> Tool:
    """The tool registered as ``name``; a clear error lists the available ones."""
    eps = _entry_points()
    if name not in eps:
        avail = ", ".join(sorted(eps)) or "none installed"
        raise LCError(
            f"No tool named {name!r} in entry-point group {ENTRY_POINT_GROUP!r}. Available: {avail}."
        )
    return _load(eps[name])
