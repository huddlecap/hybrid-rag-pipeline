#!/usr/bin/env python3
"""Interactively build a curated evaluation question set."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]

from dotenv import dotenv_values  # noqa: E402
from groq import Groq  # noqa: E402

CHUNKS_DIR = ROOT / "data/chunks/structure"
DEDUP_REPORT = ROOT / "data/index/dedup_report.json"
DEFAULT_OUTPUT = ROOT / "data/eval/curated_questions.txt"

MODEL = "qwen/qwen3.8-27b"
MIN_INTERVAL_SECONDS = 10.0
MAX_ATTEMPTS = 3
RETRYABLE_CODES = {429, 503, 504}

PROMPT = (
    "You are a Python developer. Read this documentation chunk and write ONE "
    "realistic, technical question that a user would ask where this chunk is "
    "the exact answer. Output ONLY the plain text question, no JSON, no "
    "formatting."
)

RETRY_MESSAGE_PATTERN = re.compile(
    r"(?:try again|retry)(?:\s+(?:in|after))?\s+"
    r"(\d+(?:\.\d+)?)\s*(s|sec(?:ond)?s?|m|min(?:ute)?s?)\b",
    re.IGNORECASE,
)


def load_groq_key() -> str:
    values = dotenv_values(ROOT / ".env")
    key = values.get("GROQ_API_KEY") or ""
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set in .env")
    return key


def load_chunks() -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
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


def error_code(exc: Exception) -> Optional[int]:
    for attr in ("status_code", "code"):
        code = getattr(exc, attr, None)
        if isinstance(code, int):
            return code
    match = re.search(r"\b(429|503|504)\b", str(exc))
    return int(match.group(1)) if match else None


def error_payload(exc: Exception) -> Any:
    for attr in ("body", "details", "response_json"):
        value = getattr(exc, attr, None)
        if value is not None:
            return value
    return None


def parse_delay(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.fullmatch(
            r"\s*(\d+(?:\.\d+)?)\s*(?:s|sec(?:ond)?s?)?\s*",
            value,
            re.IGNORECASE,
        )
        if match:
            return float(match.group(1))
    return None


def find_retry_delay(payload: Any) -> Optional[float]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized = key.replace("_", "").replace("-", "").lower()
            if normalized in {"retrydelay", "retryafter"}:
                delay = parse_delay(value)
                if delay is not None:
                    return delay
            delay = find_retry_delay(value)
            if delay is not None:
                return delay
    elif isinstance(payload, list):
        for value in payload:
            delay = find_retry_delay(value)
            if delay is not None:
                return delay
    return None


def parse_retry_message(text: str) -> Optional[float]:
    match = RETRY_MESSAGE_PATTERN.search(text or "")
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2).casefold()
    if unit.startswith("m"):
        return amount * 60.0
    return amount


def retry_after_seconds(exc: Exception) -> Optional[float]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        raw = headers.get("retry-after")
        if raw:
            delay = parse_delay(raw)
            if delay is not None:
                return delay
            try:
                when = parsedate_to_datetime(raw)
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                return max(
                    0.0,
                    (when - datetime.now(timezone.utc)).total_seconds(),
                )
            except (TypeError, ValueError, OverflowError):
                pass

    delay = find_retry_delay(error_payload(exc))
    if delay is not None:
        return delay

    message = f"{getattr(exc, 'message', '')} {exc}"
    return parse_retry_message(message)


class ApiPacer:
    def __init__(self, interval: float = MIN_INTERVAL_SECONDS) -> None:
        self.interval = interval
        self.last_request_at: Optional[float] = None

    def wait(self) -> None:
        now = time.monotonic()
        if self.last_request_at is not None:
            remaining = self.interval - (now - self.last_request_at)
            if remaining > 0:
                print(
                    f"    pacing: waiting {remaining:.1f}s before request ...",
                    flush=True,
                )
                time.sleep(remaining)
        self.last_request_at = time.monotonic()


def generate_question(
    client: Groq,
    text: str,
    pacer: ApiPacer,
) -> Optional[str]:
    if not text.strip():
        return None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        pacer.wait()
        print(
            f"    generating with {MODEL} "
            f"(attempt {attempt}/{MAX_ATTEMPTS}) ...",
            flush=True,
        )
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.7,
                max_tokens=200,
            )
            content = ""
            if response.choices:
                content = response.choices[0].message.content or ""
            question = clean_question(content)
            if question:
                return question
            print("    empty response", flush=True)
            return None
        except Exception as exc:
            code = error_code(exc)
            message = str(exc).splitlines()[0][:180]
            print(
                f"    error: {type(exc).__name__}: {message}",
                flush=True,
            )

            if code not in RETRYABLE_CODES:
                return None

            if attempt >= MAX_ATTEMPTS:
                print(
                    f"    giving up after {MAX_ATTEMPTS} attempts "
                    f"(last status {code})",
                    flush=True,
                )
                return None

            delay = retry_after_seconds(exc)
            if delay is None:
                delay = 30.0 + random.uniform(0.0, 5.0)
            print(
                f"    {code} received; waiting {delay:.1f}s "
                "before retry ...",
                flush=True,
            )
            time.sleep(delay)

    return None


def append_question(path: Path, question: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(question.rstrip("\n") + "\n")
        handle.flush()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Interactively curate evaluation questions"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    chunks = load_chunks()
    if not chunks:
        raise RuntimeError(f"no chunks found in {CHUNKS_DIR}")

    rng = random.Random(args.seed) if args.seed is not None else random
    remaining = chunks[:]
    rng.shuffle(remaining)

    client = Groq(
        api_key=load_groq_key(),
        timeout=30.0,
        max_retries=0,
    )
    pacer = ApiPacer()

    print(f"Loaded {len(chunks)} deduplicated structure chunks", flush=True)
    print(f"Model: {MODEL} (fixed)", flush=True)
    print(
        f"Pacing: at least {MIN_INTERVAL_SECONDS:.0f}s between requests",
        flush=True,
    )
    print(f"Saving kept questions to {args.output}", flush=True)
    print("Commands: y=keep, n=discard, q=quit, or type a rewrite", flush=True)

    saved = 0
    while True:
        if not remaining:
            remaining = chunks[:]
            rng.shuffle(remaining)
        chunk = remaining.pop()

        print("\n" + "=" * 72)
        print(f"Source: {chunk.get('source_file', 'unknown')}")
        print(f"Chunk:  {chunk.get('chunk_id', 'unknown')}")
        question = generate_question(
            client, chunk.get("text", ""), pacer
        )
        if question is None:
            print("Could not generate a question for this chunk.", flush=True)
            print("Enter to try another chunk, or q to quit: ", end="", flush=True)
            try:
                answer = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if answer == "q":
                break
            continue

        print("\nGenerated question:")
        print(question, flush=True)

        while True:
            print(
                "\n[y] keep  [n] discard  [q] quit  or type a rewrite: ",
                end="",
                flush=True,
            )
            try:
                answer = input().strip()
            except (EOFError, KeyboardInterrupt):
                print()
                answer = "q"

            if answer.lower() == "q":
                print(f"Saved {saved} question(s) to {args.output}")
                return 0
            if answer.lower() == "n":
                print("Discarded.")
                break
            if answer.lower() == "y":
                final = question
            elif answer:
                final = " ".join(answer.split())
            else:
                print("Enter y, n, q, or a rewritten question.")
                continue

            append_question(args.output, final)
            saved += 1
            print(f"Saved: {final}")
            break

    print(f"Saved {saved} question(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
