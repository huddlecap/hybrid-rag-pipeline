#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from fastembed import TextEmbedding

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_DIR = ROOT / "data/chunks/structure"
QUESTIONS_FILE = ROOT / "data/eval/curated_questions.txt"
OUTPUT_FILE = ROOT / "data/eval/golden_dataset.json"


def load_chunks() -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    seen_ids = set()
    for path in sorted(CHUNKS_DIR.glob("*.json")):
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
    return questions


def embed_with_progress(model: TextEmbedding, texts: List[str], label: str, batch_size: int) -> np.ndarray:
    total = len(texts)
    vectors: List[np.ndarray] = []
    for index, vector in enumerate(model.embed(texts, batch_size=batch_size), start=1):
        vectors.append(vector)
        if index % 10 == 0 or index == total:
            percent = index / total * 100
            print(f"\r{label}: {index}/{total} ({percent:.1f}%)", end="", flush=True)
    print()
    return np.asarray(vectors, dtype=np.float32)


def main() -> int:
    chunks = load_chunks()
    if not chunks:
        raise RuntimeError("no chunks found")
    questions = load_questions()
    if not questions:
        raise RuntimeError("no questions found")

    corpus_texts = [chunk["text"] for chunk in chunks]
    print("Loading fastembed model...")
    model = TextEmbedding(threads=1)
    chunk_embeddings = embed_with_progress(model, corpus_texts, "Chunks", 16)
    question_embeddings = embed_with_progress(model, questions, "Questions", 16)

    results: List[Dict[str, str]] = []
    for question, question_embedding in zip(questions, question_embeddings):
        scores = np.dot(chunk_embeddings, question_embedding)
        best_index = int(np.argmax(scores))
        best_chunk = chunks[best_index]
        results.append({
            "question": question,
            "ground_truth_chunk_id": best_chunk["chunk_id"],
            "ground_truth_text": best_chunk["text"],
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"questions: {len(results)}")
    print(f"chunks: {len(chunks)}")
    print("model: fastembed (BAAI/bge-small-en-v1.5)")
    print(f"output: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
