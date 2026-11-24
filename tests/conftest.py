"""Put the repository root on sys.path so `import app.main` works in CI.

GitHub Actions runs `pytest -q` (not `python -m pytest`), which does not add
the working directory to sys.path by itself.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
