"""Put the project root on sys.path so these scripts can import config.

Run any script from the project root:
    .venv\\Scripts\\python.exe download\\step1_fetch_taxonomy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
