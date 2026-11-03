"""Phase 2, Step 1: hybrid retrieval with Reciprocal Rank Fusion.

Dense search queries the selected ChromaDB collection with the same local
fastembed model used for indexing. Sparse search runs BM25 over the matching
strategy pickle. The two ranked lists (top-k each) are merged with weighted
RRF:

    score(chunk) = vector_weight / (rrf_k + dense_rank)
                 + keyword_weight / (rrf_k + bm25_rank)

No cross-encoder reranker is applied here.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from fastembed import TextEmbedding

from indexer import STRATEGIES, tokenize

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_TOP_K = 10
DEFAULT_RRF_K = 60
DEFAULT_VECTOR_WEIGHT = 0.5
DEFAULT_KEYWORD_WEIGHT = 0.5

logger = logging.getLogger("retriever")


def load_bm25(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def dense_search(collection, embedder: TextEmbedding, query: str, k: int) -> List[Dict[str, Any]]:
    query_vector = next(embedder.query_embed(query)).tolist()
    count = collection.count()
    if count == 0:
        return []

    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(k, count),
        include=["documents", "metadatas", "distances"],
    )

    items = []
    for chunk_id, document, metadata, distance in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        items.append(
            {
                "chunk_id": chunk_id,
                "source_file": metadata.get("source_file", ""),
                "text": document,
                "distance": float(distance),
            }
        )

    items.sort(key=lambda item: (item["distance"], item["chunk_id"]))
    for rank, item in enumerate(items, start=1):
        item["rank"] = rank
    return items


def keyword_search(bm25_data: Dict[str, Any], query: str, k: int) -> List[Dict[str, Any]]:
    chunk_ids = bm25_data["chunk_ids"]
    texts = bm25_data["texts"]
    metadatas = bm25_data["metadatas"]
    scores = bm25_data["bm25"].get_scores(tokenize(query))

    order = sorted(
        range(len(chunk_ids)),
        key=lambda i: (-float(scores[i]), chunk_ids[i]),
    )[:k]

    items = []
    for rank, i in enumerate(order, start=1):
        items.append(
            {
                "chunk_id": chunk_ids[i],
                "source_file": metadatas[i].get("source_file", ""),
                "text": texts[i],
                "bm25_score": float(scores[i]),
                "rank": rank,
            }
        )
    return items


def fuse(
    dense_results: List[Dict[str, Any]],
    keyword_results: List[Dict[str, Any]],
    vector_weight: float,
    keyword_weight: float,
    rrf_k: int,
) -> List[Dict[str, Any]]:
    fused: Dict[str, Dict[str, Any]] = {}

    def entry(item: Dict[str, Any]) -> Dict[str, Any]:
        return fused.setdefault(
            item["chunk_id"],
            {
                "chunk_id": item["chunk_id"],
                "source_file": item["source_file"],
                "text": item["text"],
                "rrf_score": 0.0,
                "dense_rank": None,
                "dense_distance": None,
                "bm25_rank": None,
                "bm25_score": None,
            },
        )

    for item in dense_results:
        record = entry(item)
        record["dense_rank"] = item["rank"]
        record["dense_distance"] = item["distance"]
        record["rrf_score"] += vector_weight / (rrf_k + item["rank"])

    for item in keyword_results:
        record = entry(item)
        record["bm25_rank"] = item["rank"]
        record["bm25_score"] = item["bm25_score"]
        record["rrf_score"] += keyword_weight / (rrf_k + item["rank"])

    return sorted(
        fused.values(),
        key=lambda item: (-item["rrf_score"], item["chunk_id"]),
    )


def search(
    query: str,
    strategy: str = "structure",
    chroma_dir: str | Path = "data/index/chroma",
    bm25_dir: str | Path = "data/index/bm25",
    model_name: str = EMBEDDING_MODEL,
    top_k: int = DEFAULT_TOP_K,
    rrf_k: int = DEFAULT_RRF_K,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
) -> Dict[str, Any]:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if rrf_k < 0:
        raise ValueError("rrf_k must be >= 0")
    if vector_weight < 0 or keyword_weight < 0:
        raise ValueError("weights must be >= 0")
    if vector_weight == 0 and keyword_weight == 0:
        raise ValueError("at least one weight must be > 0")

    chroma_dir = Path(chroma_dir)
    bm25_path = Path(bm25_dir) / f"{strategy}.pkl"

    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(strategy)
    bm25_data = load_bm25(bm25_path)

    logger.info("loading embedding model %s", model_name)
    embedder = TextEmbedding(model_name=model_name)

    dense_results = dense_search(collection, embedder, query, top_k)
    keyword_results = keyword_search(bm25_data, query, top_k)
    results = fuse(dense_results, keyword_results, vector_weight, keyword_weight, rrf_k)

    logger.info(
        "query=%r strategy=%s dense=%d keyword=%d unique=%d",
        query, strategy, len(dense_results), len(keyword_results), len(results),
    )

    return {
        "query": query,
        "strategy": strategy,
        "embedding_model": model_name,
        "top_k": top_k,
        "rrf_k": rrf_k,
        "vector_weight": vector_weight,
        "keyword_weight": keyword_weight,
        "dense_count": len(dense_results),
        "keyword_count": len(keyword_results),
        "unique_chunks": len(results),
        "results": results,
    }


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid retrieval with RRF")
    parser.add_argument("query")
    parser.add_argument("--strategy", default="structure", choices=STRATEGIES)
    parser.add_argument("--chroma", default="data/index/chroma")
    parser.add_argument("--bm25", default="data/index/bm25")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--vector-weight", type=float, default=DEFAULT_VECTOR_WEIGHT)
    parser.add_argument("--keyword-weight", type=float, default=DEFAULT_KEYWORD_WEIGHT)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--log", default="data/logs/retriever.log")
    args = parser.parse_args(argv)

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log, encoding="utf-8")],
    )

    report = search(
        args.query,
        strategy=args.strategy,
        chroma_dir=args.chroma,
        bm25_dir=args.bm25,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        vector_weight=args.vector_weight,
        keyword_weight=args.keyword_weight,
    )

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    print(f"Query: {report['query']}")
    print(f"Strategy: {report['strategy']}  weights: {report['vector_weight']}/{report['keyword_weight']}")
    print(
        f"Dense: {report['dense_count']}  BM25: {report['keyword_count']}  "
        f"Unique after fusion: {report['unique_chunks']}"
    )
    for i, item in enumerate(report["results"], start=1):
        print(
            f"{i:2d}. {item['chunk_id']}  RRF {item['rrf_score']:.6f}  "
            f"dense #{item['dense_rank']}  bm25 #{item['bm25_rank']}  "
            f"source={item['source_file']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
