"""
Vercel entrypoint.

Vercel's Python runtime (in "other framework" mode, which is what this
project is on) only looks for Serverless Functions inside api/ — it
doesn't treat a root-level main.py as a function on its own. This file is
just a thin re-export so main.py stays the single source of truth for the
actual app; nothing here duplicates logic.
"""

import sys
from pathlib import Path

# main.py (and graph.py, which it imports) live one directory up, at the
# repo root — put that on the path so `from main import app` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import app  # noqa: E402  (import must follow the sys.path fix above)
