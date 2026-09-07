import sys
from pathlib import Path

# Make `config` and `rag` importable when pytest runs from anywhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
