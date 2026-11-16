"""Where a tool's per-session diagnostics go.

Tool code writes diagnostics to ``console``, a proxy whose target the caller
chooses:

- default: a quiet Console, so a ``run_session`` unit prints nothing on its own
- single-session CLI: ``with log_to(Console()):`` → stdout
- batch runner: ``with log_to(Console(file=open(log, "w"))):`` → one file per session

Only one session runs per process at a time (batch runners parallelise with
processes), so a process-wide target is enough. Threads inside one session
share the session's target.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Console


class ConsoleProxy:
    """Forwards every attribute to the current target Console."""

    def __init__(self) -> None:
        self._target = Console(quiet=True)

    def __getattr__(self, name: str):
        return getattr(self._target, name)


_ConsoleProxy = ConsoleProxy  # pyGLM's private name

console = ConsoleProxy()


@contextmanager
def log_to(target: Console | None, proxy: ConsoleProxy | None = None) -> Iterator[None]:
    """Send ``proxy`` (default: this module's ``console``) output to ``target`` (None = discard)."""
    proxy = console if proxy is None else proxy
    prev = proxy._target
    proxy._target = target if target is not None else Console(quiet=True)
    try:
        yield
    finally:
        proxy._target = prev
