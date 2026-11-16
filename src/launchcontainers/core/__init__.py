"""launchcontainers.core: the rules every MRI-hub tool shares, in one place.

Work list, config loader + guard, provenance, run logs, the Tool interface,
the session wrapper (``lc-session``) and the launch-status report. See
ARCHITECTURE.md. Runtime dependencies: pyyaml, rich, typer — nothing else.

Formerly the separate package ``analysis-core`` (MRI-hub/analysis-core, 0.1.4).
"""

from __future__ import annotations

from launchcontainers import __version__


class LCError(ValueError):
    """A user-facing error: bad work list, bad config, refused analysis dir, …"""


__all__ = ["LCError", "__version__"]
