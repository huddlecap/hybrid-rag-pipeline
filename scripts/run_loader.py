#!/usr/bin/env python3
"""CLI entry point for the multi-format document loader."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from the repo root without installation.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from document_loader import LOADER_VERSION, main  # noqa: E402


def cli(argv=None):
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(cli())
