"""Compatibility script for fetching work usage data."""

from __future__ import annotations

import sys
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PYTHON_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lpt_stake.cli.fetch_work_data import main


if __name__ == "__main__":
    main()
