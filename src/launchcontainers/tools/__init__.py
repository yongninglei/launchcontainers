"""Built-in tools: every analysis launchcontainers knows how to run.

- ``dwi.py``      the Flywheel-style DWI containers (apptainer images)
- ``external.py`` analyses that are their own programs (pyGLM, autoROI-scaling,
                  autoROI-clustering), run by their single-session CLI and
                  described by its ``--plan`` (docs/plan_protocol.md)

Other packages can still add tools under the ``launchcontainers.tools``
entry-point group; the built-ins are looked up first.
"""

from __future__ import annotations

from launchcontainers.tools.dwi import DWI_CONTAINERS, DWIContainer
from launchcontainers.tools.external import EXTERNAL, ExternalTool

BUILTIN = {**{name: DWIContainer(name) for name in DWI_CONTAINERS}, **EXTERNAL}

__all__ = ["BUILTIN", "DWI_CONTAINERS", "DWIContainer", "EXTERNAL", "ExternalTool"]
