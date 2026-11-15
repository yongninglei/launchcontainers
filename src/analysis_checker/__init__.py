"""
analysis_checker/__init__.py
==========================
Central registry of all AnalysisSpec subclasses.

─── How to add a new analysis type ──────────────────────────────────────────
[DEV] Steps to register a new spec:
  1. Create  analysis_checker/<yourtype>.py  (subclass AnalysisSpec)
  2. Import it here:   from .yourtype import YourTypeSpec
  3. Add to SPEC_REGISTRY:  "yourtype": YourTypeSpec()
  The CLI will pick it up automatically — nothing else needs changing.
─────────────────────────────────────────────────────────────────────────────
"""

# analysis_checker/__init__.py
from __future__ import annotations

from .check_analysis_integrity import SPEC_REGISTRY

__all__ = ["SPEC_REGISTRY"]
