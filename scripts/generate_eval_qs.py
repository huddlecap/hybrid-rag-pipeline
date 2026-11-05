#!/usr/bin/env python3
"""Generate draft evaluation questions from structure-aware chunks."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

GEMINI_MODEL = "gemini-3.5-flash"

CHUNKS_PATH = ROOT / "data/chunks/structure_chunks.json"
CHUNKS_DIR = ROOT / "data/chunks/structure"
DEDUP_REPORT = ROOT / "data/index/dedup_report.json"
DEFAULT_OUTPUT = ROOT / "data/eval/draft_questions.txt"


def load_api_key() -> str:
    load_dotenv(dotenv_path=ROOT / ".env", override=False)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to .env or the environment."
        )
    return api_key


PROMPT = (
    "You are a Python developer. Read this documentation chunk and write ONE "
    "realistic, technical question that a user would ask where this chunk is "
    "the exact answer. Output ONLY the question."
)


def load_chunks() -> List[Dict[str, Any]]:
    if CHUNKS_PATH.exists():
        data = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        chunks = data.get("chunks", []) if isinstance(data, dict) else data
    else:
        chunks = []
        for path in sorted(CHUNKS_DIR.glob("*.json")):
            document = json.loads(path.read_text(encoding="utf-8"))
            chunks.extend(document.get("chunks", []))

    removed_ids = set()
    if DEDUP_REPORT.exists():
        report = json.loads(DEDUP_REPORT.read_text(encoding="utf-8"))
        for strategy in report.get("strategies", []):
            if strategy.get("strategy") == "structure":
                removed_ids = set(strategy.get("removed_ids", []))
                break

    if removed_ids:
        chunks = [
            chunk
            for chunk in chunks
            if chunk.get("chunk_id") not in removed_ids
        ]
    return chunks


def clean_question(raw: str) -> str:
    question = " ".join((raw or "").strip().split())
    if question.startswith("```") and question.endswith("```"):
        question = question.strip("`").strip()
    question = question.strip().strip('"').strip("'").strip()
    for prefix in ("Question:", "Q:"):
        if question.lower().startswith(prefix.lower()):
            question = question[len(prefix):].strip()
    return question


def generate_question(
    client: genai.Client,
    text: str,
    model: str,
    retries: int = 3,
) -> Optional[str]:
    if not text.strip():
        return None

    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=text,
                config=types.GenerateContentConfig(
                    system_instruction=PROMPT,
                    temperature=0.7,
                    max_output_tokens=200,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            question = clean_question(response.text or "")
            if question:
                return question
            print(f"      empty response; retry {attempt}/{retries}", flush=True)
        except Exception as exc:
            message = str(exc).splitlines()[0][:180]
            print(
                f"      error: {type(exc).__name__}: {message}",
                flush=True,
            )

        if attempt < retries:
            time.sleep(min(2 ** attempt, 8))

    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate draft evaluation questions from chunks"
    )
    parser.add_argument("--model", default=GEMINI_MODEL)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args(argv)

    chunks = load_chunks()
    if len(chunks) < args.count:
        raise RuntimeError(
            f"need {args.count} chunks, only found {len(chunks)}"
        )

    rng = random.Random(args.seed) if args.seed is not None else random
    rng.shuffle(chunks)
    selected = chunks[: args.count]
    backup = chunks[args.count:]

    print(
        f"[1/3] Loaded {len(chunks)} deduplicated structure chunks",
        flush=True,
    )
    print(f"[2/3] Sampling {args.count} chunks", flush=True)

    client = genai.Client(
        api_key=load_api_key(),
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1),
            timeout=30000,
        ),
    )

    questions: List[str] = []
    for index, chunk in enumerate(selected, start=1):
        source = chunk.get("source_file", "unknown")
        print(f"  [{index}/{args.count}] {source}", flush=True)
        started = time.time()
        question = generate_question(
            client, chunk.get("text", ""), args.model
        )

        while question is None and backup:
            extra = backup.pop()
            print(
                f"      using backup chunk: {extra.get('source_file', 'unknown')}",
                flush=True,
            )
            question = generate_question(
                client, extra.get("text", ""), args.model
            )

        if question is None:
            raise RuntimeError(
                f"could not generate question {index}/{args.count}"
            )

        questions.append(question)
        print(
            f"      ok ({time.time() - started:.1f}s): {question[:100]}",
            flush=True,
        )

        if args.delay:
            time.sleep(args.delay)

    print(f"[3/3] Writing {len(questions)} questions", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp_path.write_text(
        "\n".join(questions) + "\n", encoding="utf-8"
    )
    tmp_path.replace(args.output)

    print(f"Wrote {len(questions)} questions to {args.output}", flush=True)
    for question in questions[:3]:
        print(question)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
