#!/usr/bin/env python3
"""Offline tests for citation verification and confidence fallback."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from generator import parse_response, verify_answer  # noqa: E402

NUM_CHUNKS = 5

PART_A_JSON = r'''{"answer": "You can start a FastAPI server for development using the command `uv run fastapi dev` [5]. To serve a FastAPI application for production, you can use `fastapi run main.py` or `uv run fastapi run` [2], [4], [5].", "confidence": 1.0}'''

INVALID_CITATION_JSON = r'''{"answer": "Start the server with `uv run fastapi dev` [6].", "confidence": 0.8}'''

UNKNOWN_JSON = r'''{"answer": "I don't know", "confidence": 0.9}'''


def test_part_a_valid_json() -> bool:
    parsed = parse_response(PART_A_JSON)
    verified = verify_answer(
        parsed["answer"], parsed["confidence"], NUM_CHUNKS
    )
    return (
        verified["answer"] == parsed["answer"]
        and verified["confidence"] == 1.0
        and verified["citations"] == [5, 2, 4, 5]
        and verified["invalid_citations"] == []
        and verified["fallback_applied"] is False
    )


def test_fallbacks() -> bool:
    invalid_parsed = parse_response(INVALID_CITATION_JSON)
    invalid_verified = verify_answer(
        invalid_parsed["answer"],
        invalid_parsed["confidence"],
        NUM_CHUNKS,
    )
    invalid_ok = (
        "[6]" not in invalid_verified["answer"]
        and invalid_verified["invalid_citations"] == [6]
        and invalid_verified["confidence"] == 0.8
        and invalid_verified["fallback_applied"] is False
    )

    unknown_parsed = parse_response(UNKNOWN_JSON)
    unknown_verified = verify_answer(
        unknown_parsed["answer"],
        unknown_parsed["confidence"],
        NUM_CHUNKS,
    )
    unknown_ok = (
        unknown_verified["confidence"] == 0.0
        and unknown_verified["fallback_applied"] is True
    )

    upper_verified = verify_answer("I DON'T KNOW", 0.9, NUM_CHUNKS)
    case_ok = (
        upper_verified["confidence"] == 0.0
        and upper_verified["fallback_applied"] is True
    )

    return invalid_ok and unknown_ok and case_ok


def main() -> int:
    results = [
        ("Test 1 (Part A valid JSON)", test_part_a_valid_json()),
        ("Test 2 (invalid [6] + I don't know)", test_fallbacks()),
    ]
    for name, passed in results:
        print(f"{name}: {'true' if passed else 'false'}")
    return 0 if all(passed for _, passed in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
