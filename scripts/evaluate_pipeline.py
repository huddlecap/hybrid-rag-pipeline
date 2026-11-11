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

STRATEGIES = ("fixed", "structure", "semantic")

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


def dataset_file_for_strategy(strategy: str) -> Path:
    return ROOT / "data/eval" / f"golden_dataset_{strategy}.json"


def checkpoint_file_for_strategy(strategy: str) -> Path:
    return ROOT / "data/eval" / f"checkpoint_{strategy}.json"


def load_dataset(strategy: str) -> List[Dict[str, Any]]:
    return json.loads(dataset_file_for_strategy(strategy).read_text(encoding="utf-8"))


def get_question_id(item: Dict[str, Any]) -> str:
    return item.get("id") or hashlib.sha256(item["question"].encode("utf-8")).hexdigest()


def load_checkpoint(checkpoint_file: Path) -> Dict[str, Dict[str, int]]:
    if not checkpoint_file.exists():
        return {}
    try:
        data = json.loads(checkpoint_file.read_text(encoding="utf-8"))
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


def save_checkpoint(checkpoint: Dict[str, Dict[str, int]], checkpoint_file: Path) -> None:
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = checkpoint_file.with_suffix(".tmp")
    temp_file.write_text(
        json.dumps(checkpoint, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temp_file.replace(checkpoint_file)


def resolve_judge_model(groq_client: Groq) -> str:
    available = {model.id for model in groq_client.models.list().data}
    for model in PREFERRED_JUDGE_MODELS:
        if model in available:
            return model
    raise RuntimeError(f"No preferred judge model available. Available: {sorted(available)}")


def ask_judge(question: str, generated_answer: str, context_text: str) -> Dict[str, Any]:
    """RAGAS-style claim-extraction judge.

    The LLM extracts atomic claims and returns integer counts; Python performs
    safe division to produce granular 0.0-1.0 floats.
    """
    if client is None:
        raise RuntimeError("Groq client is not initialized")
    system_prompt = (
        "You are a mathematical evaluator. Return ONLY a JSON object with exactly "
        'these keys: "atomic_claims" (list of strings), "total_claims" (integer), '
        '"supported_claims" (integer), "correct_claims" (integer), '
        '"unsupported_citations" (list of strings). '
        "Break the generated answer down into a list of discrete, atomic factual claims. "
        "1. atomic_claims: each distinct factual claim as a separate string. "
        "   total_claims must exactly equal len(atomic_claims). "
        "   If the answer is 'I don't know', output atomic_claims as [] and total_claims as 0. "
        "2. supported_claims: integer count of claims fully backed by the retrieved context. "
        "3. correct_claims: integer count of claims that correctly answer the user's prompt. "
        '4. unsupported_citations: list of citation strings like "[2]" NOT supported by the retrieved context. '
        "Output strictly as integers for counts and lists of strings for atomic_claims and unsupported_citations. "
        "Return only the JSON object, no markdown fences."
    )
    user_prompt = f"""Question: {question}

Generated answer:
{generated_answer}

Retrieved context:
{context_text}

Evaluate the generated answer. Return a JSON object with:
- atomic_claims: list of strings, each a discrete factual claim
- total_claims: integer count of claims (must equal len(atomic_claims); 0 if "I don't know")
- supported_claims: integer count of claims backed by retrieved context
- correct_claims: integer count of claims that correctly answer the user's prompt
- unsupported_citations: list of citation strings like "[2]" NOT supported by retrieved context

Return only the JSON object.
"""
    last_content = ""
    for attempt in range(2):
        request_kwargs: Dict[str, Any] = {
            "model": JUDGE_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 1024 if JUDGE_MODEL.startswith("openai/gpt-oss") else 256,
        }
        if JUDGE_MODEL.startswith("openai/gpt-oss"):
            request_kwargs["reasoning_effort"] = "low"
        response = client.chat.completions.create(**request_kwargs)
        content = (response.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()
        try:
            data = json.loads(content)
            if all(k in data for k in ("correctness", "faithfulness", "unsupported_citations")):
                return {
                    "correctness": int(data["correctness"]),
                    "faithfulness": int(data["faithfulness"]),
                    "unsupported_citations": list(data["unsupported_citations"]),
                }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        last_content = content
        if attempt == 0:
            print("  judge returned invalid JSON; retrying once ...", flush=True)
    raise RuntimeError(f"Judge returned invalid JSON after 2 attempts: {last_content!r}")


def average(scores: List[int]) -> float:
    return sum(scores) / len(scores) * 100 if scores else 0.0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline with a Groq judge")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--strategy",
        type=str,
        default="structure",
        choices=STRATEGIES,
        help="Chunking strategy to evaluate",
    )
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
    print(f"Strategy: {args.strategy}", flush=True)

    dataset = load_dataset(args.strategy)
    if not dataset:
        raise RuntimeError("dataset is empty")

    checkpoint_file = checkpoint_file_for_strategy(args.strategy)
    checkpoint = load_checkpoint(checkpoint_file)
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
        ground_truth_ids = item.get("ground_truth_chunk_ids", [item.get("ground_truth_chunk_id", "")])
        ground_truth_text = item["ground_truth_text"]
        question_started = time.perf_counter()

        print(
            f"\n[{index}/{total}] new {new_evaluated + 1}/{args.batch_size} "
            f"| {question[:100]}",
            flush=True,
        )

        retrieval_started = time.perf_counter()
        print("  retrieval: ...", end="", flush=True)
        retrieval = search(question, strategy=args.strategy)
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

        retrieval_score = 1 if any(gid in retrieved_ids for gid in ground_truth_ids) else 0
        print(
            "  judging: correctness/faithfulness/citations ...",
            flush=True,
        )

        context_text = "\n\n".join(
            f"[{i+1}] {chunk['text']}" for i, chunk in enumerate(retrieved_chunks)
        )
        judge_result = ask_judge(question, generated_answer, context_text)
        correctness_score = judge_result["correctness"]
        faithfulness_score = judge_result["faithfulness"]

        citations = re.findall(r"\[([^\]]+)\]", generated_answer)
        if not citations:
            citation_score = 0
        else:
            unsupported = set(judge_result["unsupported_citations"])
            citation_score = 1 if not (unsupported & set(citations)) else 0

        checkpoint[question_id] = {
            "retrieval_relevance": retrieval_score,
            "answer_correctness": correctness_score,
            "faithfulness": faithfulness_score,
            "citation_accuracy": citation_score,
        }
        save_checkpoint(checkpoint, checkpoint_file)
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
