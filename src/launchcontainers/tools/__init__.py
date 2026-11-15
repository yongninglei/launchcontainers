"""Built-in tools: the containers launchcontainers knows natively.

They are also registered under the ``analysis_core.tools`` entry-point group
(pyproject.toml), so any MRI-hub launcher finds them the same way it finds
pyGLM or autoROI.
"""

from __future__ import annotations

from launchcontainers.tools.dwi import DWI_CONTAINERS, DWIContainer

BUILTIN = {name: DWIContainer(name) for name in DWI_CONTAINERS}

__all__ = ["BUILTIN", "DWI_CONTAINERS", "DWIContainer"]
