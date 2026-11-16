"""Make the test helpers (fakedata, golden/*) importable as top-level modules."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE / "golden"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
