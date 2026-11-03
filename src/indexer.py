"""Phase 1, Step 3: build dense (ChromaDB) and sparse (BM25) indexes.

Both indexes use the existing `chunk_id` from the chunk JSON files as the
single identity, so every dense row has exactly one BM25 row.

Layout:
  data/index/chroma/            persistent ChromaDB directory
  data/index/bm25/<strategy>.pkl  BM25 index + ids + texts (alongside chroma)

No chunk text is modified here; chunks are only embedded and indexed.
"""

from __future__ import annotations

import json
import logging
import pickle
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
STRATEGIES = ("fixed", "structure", "semantic")
TOKEN_RE = re.compile(r"[a-z0-9_]+")

logger = logging.getLogger("indexer")


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric/underscore tokens, keeping exact identifiers."""
    return TOKEN_RE.findall(text.lower())


def load_chunks(
    chunks_root: Path, strategy: str
) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    ids: List[str] = []
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for f in sorted((chunks_root / strategy).glob("*.json")):
        payload = json.loads(f.read_text(encoding="utf-8"))
        for chunk in payload.get("chunks", []):
            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text", "")
            if not chunk_id or not text.strip():
                skipped.append(
                    {
                        "source_file": payload.get("source_file"),
                        "chunk_index": chunk.get("chunk_index"),
                        "reason": "missing chunk_id" if not chunk_id else "empty text",
                    }
                )
                continue
            page = chunk.get("page_number")
            ids.append(chunk_id)
            texts.append(text)
            metadatas.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": chunk.get("doc_id", ""),
                    "source_file": chunk.get("source_file", ""),
                    "chunk_index": int(chunk.get("chunk_index", 0)),
                    "heading_path": " > ".join(chunk.get("heading_path") or []),
                    "page_number": int(page) if page is not None else -1,
                    "strategy": strategy,
                    "char_count": int(chunk.get("char_count", len(text))),
                    "token_count": int(chunk.get("token_count", 0)),
                }
            )
    return ids, texts, metadatas, skipped


def embed_texts(
    model: TextEmbedding, texts: List[str], batch_size: int = 64
) -> List[List[float]]:
    total = len(texts)
    embeddings: List[List[float]] = []
    for i, vector in enumerate(model.embed(texts, batch_size=batch_size), start=1):
        embeddings.append(vector.tolist())
        if i % batch_size == 0 or i == total:
            logger.info("embedded %d/%d (%.1f%%)", i, total, 100.0 * i / total)
    return embeddings


def reset_collection(client, name: str):
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.create_collection(name=name, metadata={"hnsw:space": "cosine"})


def add_in_batches(collection, ids, documents, metadatas, embeddings, size: int = 256) -> None:
    total = len(ids)
    for i in range(0, total, size):
        collection.add(
            ids=ids[i : i + size],
            documents=documents[i : i + size],
            metadatas=metadatas[i : i + size],
            embeddings=embeddings[i : i + size],
        )
        logger.info("chroma add %d/%d", min(i + size, total), total)


def build_bm25(
    strategy: str, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]
) -> Dict[str, Any]:
    tokenized = [tokenize(t) for t in texts]
    return {
        "strategy": strategy,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_ids": ids,
        "texts": texts,
        "metadatas": metadatas,
        "tokenized_corpus": tokenized,
        "bm25": BM25Okapi(tokenized),
        "count": len(ids),
    }


def ingest_strategy(
    strategy: str,
    chunks_root: Path,
    chroma_client,
    bm25_dir: Path,
    embedder: TextEmbedding,
    batch_size: int = 64,
) -> Dict[str, Any]:
    ids, texts, metadatas, skipped = load_chunks(chunks_root, strategy)
    logger.info("strategy=%s chunks=%d skipped=%d embedding...", strategy, len(ids), len(skipped))

    embeddings = embed_texts(embedder, texts, batch_size)

    logger.info("strategy=%s embedding done, writing to chroma", strategy)
    collection = reset_collection(chroma_client, strategy)
    add_in_batches(collection, ids, texts, metadatas, embeddings)

    bm25_data = build_bm25(strategy, ids, texts, metadatas)
    bm25_dir.mkdir(parents=True, exist_ok=True)
    bm25_path = bm25_dir / f"{strategy}.pkl"
    with bm25_path.open("wb") as f:
        pickle.dump(bm25_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return {
        "strategy": strategy,
        "json_chunks": len(ids),
        "chroma_count": collection.count(),
        "bm25_count": bm25_data["count"],
        "skipped": skipped,
        "bm25_path": str(bm25_path),
        "bm25_bytes": bm25_path.stat().st_size,
        "json_ids": ids,
        "json_texts": texts,
    }


def verify_strategy(
    strategy: str,
    chroma_client,
    bm25_path: Path,
    json_ids: List[str],
    json_texts: List[str],
    sample_size: int,
    seed: int,
) -> Dict[str, Any]:
    collection = chroma_client.get_collection(strategy)
    chroma_ids = collection.get(include=[])["ids"]

    with Path(bm25_path).open("rb") as f:
        bm25_data = pickle.load(f)
    bm25_ids = bm25_data["chunk_ids"]
    bm25_texts = bm25_data["texts"]

    json_text_by_id = dict(zip(json_ids, json_texts))
    duplicates = len(chroma_ids) - len(set(chroma_ids))

    counts_match = (
        collection.count() == bm25_data["count"] == len(json_ids)
    )
    id_sets_match = (
        set(chroma_ids) == set(bm25_ids) == set(json_ids)
    )

    k = min(sample_size, len(chroma_ids))
    sampled = random.Random(seed).sample(chroma_ids, k) if k else []
    samples: List[Dict[str, Any]] = []
    for chunk_id in sampled:
        rec = collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        chroma_text = rec["documents"][0]
        idx = bm25_ids.index(chunk_id)
        bm25_text = bm25_texts[idx]
        json_text = json_text_by_id.get(chunk_id)
        samples.append(
            {
                "chunk_id": chunk_id,
                "strategy": rec["metadatas"][0].get("strategy"),
                "source_file": rec["metadatas"][0].get("source_file"),
                "chroma_equals_bm25": chroma_text == bm25_text,
                "chroma_equals_json": chroma_text == json_text,
                "chars": len(chroma_text),
                "preview": chroma_text[:80],
            }
        )

    return {
        "strategy": strategy,
        "chroma_count": collection.count(),
        "bm25_count": bm25_data["count"],
        "json_count": len(json_ids),
        "counts_match": counts_match,
        "id_sets_match": id_sets_match,
        "duplicate_chroma_ids": duplicates,
        "samples": samples,
        "samples_all_match": all(
            s["chroma_equals_bm25"] and s["chroma_equals_json"] for s in samples
        ),
    }


def run(
    chunks_root: str | Path = "data/chunks",
    chroma_dir: str | Path = "data/index/chroma",
    bm25_dir: str | Path = "data/index/bm25",
    strategies: Tuple[str, ...] = STRATEGIES,
    sample_size: int = 5,
    seed: int = 42,
    batch_size: int = 64,
) -> Dict[str, Any]:
    chunks_root = Path(chunks_root)
    chroma_dir = Path(chroma_dir)
    bm25_dir = Path(bm25_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    logger.info("opening chroma at %s", chroma_dir)
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    logger.info("loading embedding model %s", EMBEDDING_MODEL)
    embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    logger.info("embedding model ready")

    ingested: List[Dict[str, Any]] = []
    verifications: List[Dict[str, Any]] = []

    for strategy in strategies:
        result = ingest_strategy(
            strategy, chunks_root, client, bm25_dir, embedder, batch_size
        )
        logger.info(
            "ingested strategy=%s json=%d chroma=%d bm25=%d pkl=%s bytes=%d",
            strategy, result["json_chunks"], result["chroma_count"],
            result["bm25_count"], result["bm25_path"], result["bm25_bytes"],
        )
        verification = verify_strategy(
            strategy, client, Path(result["bm25_path"]),
            result["json_ids"], result["json_texts"], sample_size, seed,
        )
        logger.info(
            "verify strategy=%s counts_match=%s ids_match=%s duplicates=%d samples_all_match=%s",
            strategy, verification["counts_match"], verification["id_sets_match"],
            verification["duplicate_chroma_ids"], verification["samples_all_match"],
        )
        for s in verification["samples"]:
            logger.info(
                "sample id=%s chroma==bm25=%s chroma==json=%s source=%s",
                s["chunk_id"], s["chroma_equals_bm25"],
                s["chroma_equals_json"], s["source_file"],
            )
        ingested.append({k: result[k] for k in
                         ("strategy", "json_chunks", "chroma_count", "bm25_count",
                          "skipped", "bm25_path", "bm25_bytes")})
        verifications.append(verification)

    return {
        "embedding_model": EMBEDDING_MODEL,
        "chroma_dir": str(chroma_dir),
        "bm25_dir": str(bm25_dir),
        "ingested": ingested,
        "verifications": verifications,
    }


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build ChromaDB + BM25 indexes")
    parser.add_argument("--chunks", default="data/chunks")
    parser.add_argument("--chroma", default="data/index/chroma")
    parser.add_argument("--bm25", default="data/index/bm25")
    parser.add_argument("--log", default="data/logs/indexer.log")
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.log, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    report = run(args.chunks, args.chroma, args.bm25,
                 sample_size=args.sample_size, seed=args.seed,
                 batch_size=args.batch_size)
    Path("data/index").mkdir(parents=True, exist_ok=True)
    Path("data/index/verification.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
