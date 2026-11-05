#!/usr/bin/env python3
"""Build strategy-specific golden datasets for chunking comparison.

Maps each question in the 30-question subset to the correct chunk ID(s)
within each chunking strategy directory:
- data/chunks/fixed/
- data/chunks/structure/
- data/chunks/semantic/

Output:
- data/eval/golden_dataset_fixed.json
- data/eval/golden_dataset_structure.json
- data/eval/golden_dataset_semantic.json

Each entry: question, ground_truth_chunk_ids (list), ground_truth_text
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from fastembed import TextEmbedding

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ["fixed", "structure", "semantic"]
QUESTIONS_FILE = ROOT / "data/eval/curated_questions.txt"
SUBSET_SIZE = 30


def load_chunks(strategy: str) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    seen_ids = set()
    dir_path = ROOT / "data/chunks" / strategy
    for path in sorted(dir_path.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for chunk in document.get("chunks", []):
            chunk_id = chunk.get("chunk_id") or f"{chunk.get('source_file','unknown')}#{chunk.get('chunk_index','')}"
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            text = chunk.get("text", "")
            if not text:
                continue
            chunks.append({
                "chunk_id": chunk_id,
                "source_file": chunk.get("source_file", ""),
                "text": text,
            })
    return chunks


def load_questions() -> List[str]:
    questions: List[str] = []
    for line in QUESTIONS_FILE.read_text(encoding="utf-8").splitlines():
        question = line.strip()
        if question:
            questions.append(question)
    return questions[:SUBSET_SIZE]


def embed_texts(model: TextEmbedding, texts: List[str], label: str) -> np.ndarray:
    vectors: List[np.ndarray] = []
    total = len(texts)
    for index, vector in enumerate(model.embed(texts, batch_size=16), start=1):
        vectors.append(vector)
        if index % 10 == 0 or index == total:
            percent = index / total * 100
            print(f"\r{label}: {index}/{total} ({percent:.1f}%)", end="", flush=True)
    print()
    return np.asarray(vectors, dtype=np.float32)


def find_best_chunks(
    question_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunks: List[Dict[str, Any]],
    top_n: int = 3,
) -> List[str]:
    scores = chunk_embeddings @ question_embedding
    order = np.argsort(-scores)
    best_ids = []
    for idx in order[:top_n]:
        best_ids.append(chunks[idx]["chunk_id"])
    return best_ids


def main() -> int:
    questions = load_questions()
    if not questions:
        raise RuntimeError("no questions found")

    print(f"Loaded {len(questions)} questions (subset size {SUBSET_SIZE})")

    model = TextEmbedding(threads=1)

    for strategy in STRATEGIES:
        print(f"\nProcessing strategy: {strategy}")
        chunks = load_chunks(strategy)
        if not chunks:
            raise RuntimeError(f"no chunks found for {strategy}")

        chunk_embeddings = embed_texts(model, [c["text"] for c in chunks], f"{strategy} chunks")
        question_embeddings = embed_texts(model, questions, f"{strategy} questions")

        results: List[Dict[str, Any]] = []
        for question, question_embedding in zip(questions, question_embeddings):
            best_ids = find_best_chunks(question_embedding, chunk_embeddings, chunks, top_n=3)
            best_chunk = chunks[int(np.argmax(chunk_embeddings @ question_embedding))]
            results.append({
                "question": question,
                "ground_truth_chunk_ids": best_ids,
                "ground_truth_chunk_id": best_ids[0],
                "ground_truth_text": best_chunk["text"],
            })

        output = ROOT / "data/eval" / f"golden_dataset_{strategy}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Saved {len(results)} entries to {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
