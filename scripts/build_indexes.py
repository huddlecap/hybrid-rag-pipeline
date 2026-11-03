#!/usr/bin/env python3
"""CLI entry point for building the dense and sparse indexes."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from indexer import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
