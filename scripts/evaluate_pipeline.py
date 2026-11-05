#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from groq import Groq

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from generator import generate
from retriever import search

DATASET_FILE = ROOT / "data/eval/golden_dataset.json"
CHECKPOINT_FILE = ROOT / "data/eval/checkpoint.json"
PREFERRED_JUDGE_MODELS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "llama3-8b-8192",
    "qwen-2.5-32b-it",
]
SCORE_KEYS = (
    "retrieval_relevance",
    "answer_correctness",
    "faithfulness",
    "citation_accuracy",
)

client: Optional[Groq] = None
JUDGE_MODEL = ""


def load_dataset() -> List[Dict[str, Any]]:
    return json.loads(DATASET_FILE.read_text(encoding="utf-8"))


def get_question_id(item: Dict[str, Any]) -> str:
    return item.get("id") or hashlib.sha256(item["question"].encode("utf-8")).hexdigest()


def load_checkpoint() -> Dict[str, Dict[str, int]]:
    if not CHECKPOINT_FILE.exists():
        return {}
    try:
        data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    checkpoint: Dict[str, Dict[str, int]] = {}
    for key, value in data.items():
        if isinstance(value, dict) and all(score_key in value for score_key in SCORE_KEYS):
            checkpoint[key] = {
                score_key: int(value[score_key]) for score_key in SCORE_KEYS
            }
    return checkpoint


def save_checkpoint(checkpoint: Dict[str, Dict[str, int]]) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_file = CHECKPOINT_FILE.with_suffix(".tmp")
    temp_file.write_text(
        json.dumps(checkpoint, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temp_file.replace(CHECKPOINT_FILE)


def resolve_judge_model(groq_client: Groq) -> str:
    available = {model.id for model in groq_client.models.list().data}
    for model in PREFERRED_JUDGE_MODELS:
        if model in available:
            return model
    raise RuntimeError(f"No preferred judge model available. Available: {sorted(available)}")


def ask_judge(prompt: str) -> int:
    if client is None:
        raise RuntimeError("Groq client is not initialized")
    last_content = ""
    for attempt in range(2):
        request_kwargs: Dict[str, Any] = {
            "model": JUDGE_MODEL,
            "messages": [
                {"role": "system", "content": "You are a strict evaluator. Return only 1 or 0."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 1024 if JUDGE_MODEL.startswith("openai/gpt-oss") else 8,
        }
        if JUDGE_MODEL.startswith("openai/gpt-oss"):
            request_kwargs["reasoning_effort"] = "low"
        response = client.chat.completions.create(**request_kwargs)
        content = (response.choices[0].message.content or "").strip()
        match = re.search(r"[01]", content)
        if match is not None:
            return int(match.group(0))
        last_content = content
        if attempt == 0:
            print("  judge returned no 1/0; retrying once ...", flush=True)
    raise RuntimeError(f"Judge returned an invalid response: {last_content!r}")


def average(scores: List[int]) -> float:
    return sum(scores) / len(scores) * 100 if scores else 0.0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline with a Groq judge")
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args(argv)

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    global client, JUDGE_MODEL

    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    client = Groq(api_key=api_key)
    JUDGE_MODEL = resolve_judge_model(client)
    print(f"Judge model: {JUDGE_MODEL}", flush=True)

    dataset = load_dataset()
    if not dataset:
        raise RuntimeError("dataset is empty")

    checkpoint = load_checkpoint()
    new_evaluated = 0
    total = len(dataset)

    print(
        f"Checkpoint: {len(checkpoint)}/{total} complete. "
        f"Evaluating up to {args.batch_size} new questions.",
        flush=True,
    )

    for index, item in enumerate(dataset, start=1):
        question_id = get_question_id(item)
        if question_id in checkpoint:
            continue

        question = item["question"]
        ground_truth_id = item["ground_truth_chunk_id"]
        ground_truth_text = item["ground_truth_text"]
        question_started = time.perf_counter()

        print(
            f"\n[{index}/{total}] new {new_evaluated + 1}/{args.batch_size} "
            f"| {question[:100]}",
            flush=True,
        )

        retrieval_started = time.perf_counter()
        print("  retrieval: ...", end="", flush=True)
        retrieval = search(question)
        retrieved_chunks = retrieval["results"]
        retrieved_ids = [chunk["chunk_id"] for chunk in retrieved_chunks]
        print(
            f" {len(retrieved_chunks)} chunks "
            f"({time.perf_counter() - retrieval_started:.1f}s)",
            flush=True,
        )

        generation_started = time.perf_counter()
        print("  generation: ...", end="", flush=True)
        generation = generate(question, retrieved_chunks, client=client)
        generated_answer = generation["parsed"]["answer"]
        print(
            f" {generation['model']} "
            f"({time.perf_counter() - generation_started:.1f}s)",
            flush=True,
        )

        retrieval_score = 1 if ground_truth_id in retrieved_ids else 0
        print(
            "  judging: correctness/faithfulness/citations ...",
            flush=True,
        )

        correctness_prompt = f"""Question: {question}

Ground truth:
{ground_truth_text}

Generated answer:
{generated_answer}

Is the generated answer factually correct based on the ground truth? Return 1 for yes, 0 for no. Return only 1 or 0."""
        correctness_score = ask_judge(correctness_prompt)

        context = "\n\n".join(
            f"[{chunk['chunk_id']}]\n{chunk['text']}" for chunk in retrieved_chunks
        )
        faithfulness_prompt = f"""Question: {question}

Generated answer:
{generated_answer}

Retrieved context:
{context}

Does the generated answer rely strictly on the provided context without hallucinating? Return 1 for yes, 0 for no. Return only 1 or 0."""
        faithfulness_score = ask_judge(faithfulness_prompt)

        citations = re.findall(r"\[([^\]]+)\]", generated_answer)
        if not citations:
            citation_score = 0
        else:
            citation_results: List[int] = []
            for citation in citations:
                chunk = None
                if citation.isdigit():
                    position = int(citation) - 1
                    if 0 <= position < len(retrieved_chunks):
                        chunk = retrieved_chunks[position]
                else:
                    chunk = next(
                        (
                            candidate
                            for candidate in retrieved_chunks
                            if candidate["chunk_id"] == citation
                        ),
                        None,
                    )
                if chunk is None:
                    citation_results.append(0)
                    continue
                citation_prompt = f"""Generated answer:
{generated_answer}

Citation: [{citation}]
Retrieved chunk text:
{chunk['text']}

Does the specific retrieved chunk support the claims attached to this citation? Return 1 for yes, 0 for no. Return only 1 or 0."""
                citation_results.append(ask_judge(citation_prompt))
            citation_score = 1 if all(score == 1 for score in citation_results) else 0

        checkpoint[question_id] = {
            "retrieval_relevance": retrieval_score,
            "answer_correctness": correctness_score,
            "faithfulness": faithfulness_score,
            "citation_accuracy": citation_score,
        }
        save_checkpoint(checkpoint)
        new_evaluated += 1

        print(
            f"  scores: retrieval={retrieval_score} "
            f"correctness={correctness_score} "
            f"faithfulness={faithfulness_score} "
            f"citation={citation_score} "
            f"| saved {len(checkpoint)}/{total} "
            f"| {time.perf_counter() - question_started:.1f}s",
            flush=True,
        )

        if new_evaluated >= args.batch_size:
            break

    completed = list(checkpoint.values())
    print(f"New questions evaluated this run: {new_evaluated}")
    print(f"Checkpointed questions total: {len(completed)}")
    print(
        "Retrieval Relevance: "
        f"{average([record['retrieval_relevance'] for record in completed]):.1f}%"
    )
    print(
        "Answer Correctness: "
        f"{average([record['answer_correctness'] for record in completed]):.1f}%"
    )
    print(
        "Faithfulness: "
        f"{average([record['faithfulness'] for record in completed]):.1f}%"
    )
    print(
        "Citation Accuracy: "
        f"{average([record['citation_accuracy'] for record in completed]):.1f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
