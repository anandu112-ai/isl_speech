"""
Thin wrapper around preprocess_dataset.py for extract_features CLI requirement.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.preprocess_dataset import main

if __name__ == "__main__":
    main()
