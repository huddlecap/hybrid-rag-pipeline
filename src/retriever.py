"""Phase 2: hybrid retrieval with RRF and a local cross-encoder reranker.

Dense search queries the selected ChromaDB collection with the same local
fastembed model used for indexing. Sparse search runs BM25 over the matching
strategy pickle. The two ranked lists (top-k each) are merged with weighted
RRF:

    score(chunk) = vector_weight / (rrf_k + dense_rank)
                 + keyword_weight / (rrf_k + bm25_rank)

The fused candidates are then scored by a local fastembed cross-encoder and
only the top reranked chunks are returned.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from indexer import STRATEGIES, tokenize

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"
DEFAULT_TOP_K = 10
DEFAULT_RRF_K = 60
DEFAULT_VECTOR_WEIGHT = 0.5
DEFAULT_KEYWORD_WEIGHT = 0.5
DEFAULT_RERANK_TOP_K = 5

logger = logging.getLogger("retriever")


def load_bm25(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


_embedders: Dict[str, TextEmbedding] = {}
_rerankers: Dict[str, TextCrossEncoder] = {}
_chroma_clients: Dict[str, Any] = {}
_collections: Dict[str, Any] = {}
_bm25_cache: Dict[str, Dict[str, Any]] = {}


def get_embedder(model_name: str = EMBEDDING_MODEL) -> TextEmbedding:
    embedder = _embedders.get(model_name)
    if embedder is None:
        logger.info("loading embedding model %s", model_name)
        embedder = TextEmbedding(model_name=model_name)
        _embedders[model_name] = embedder
    return embedder


def get_reranker(model_name: str = RERANK_MODEL) -> TextCrossEncoder:
    reranker = _rerankers.get(model_name)
    if reranker is None:
        logger.info("loading cross-encoder model %s", model_name)
        reranker = TextCrossEncoder(model_name=model_name)
        _rerankers[model_name] = reranker
    return reranker


def get_chroma_client(chroma_dir: str | Path = "data/index/chroma"):
    chroma_host = os.environ.get("CHROMA_HOST")
    chroma_port = os.environ.get("CHROMA_PORT")
    if chroma_host and chroma_port:
        key = f"http://{chroma_host}:{chroma_port}"
        chroma_client = _chroma_clients.get(key)
        if chroma_client is None:
            chroma_client = chromadb.HttpClient(
                host=chroma_host,
                port=int(chroma_port),
            )
            _chroma_clients[key] = chroma_client
        return chroma_client

    path = str(Path(chroma_dir).resolve())
    chroma_client = _chroma_clients.get(path)
    if chroma_client is None:
        chroma_client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )
        _chroma_clients[path] = chroma_client
    return chroma_client


def get_collection(chroma_dir: str | Path, strategy: str):
    path = str(Path(chroma_dir).resolve())
    key = f"{path}::{strategy}"
    collection = _collections.get(key)
    if collection is None:
        collection = get_chroma_client(path).get_collection(strategy)
        _collections[key] = collection
    return collection


def get_bm25(bm25_dir: str | Path, strategy: str) -> Dict[str, Any]:
    path = (Path(bm25_dir) / f"{strategy}.pkl").resolve()
    key = str(path)
    bm25_data = _bm25_cache.get(key)
    if bm25_data is None:
        bm25_data = load_bm25(path)
        _bm25_cache[key] = bm25_data
    return bm25_data


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


def rerank_results(
    query: str,
    results: List[Dict[str, Any]],
    reranker: TextCrossEncoder,
    top_k: int,
) -> List[Dict[str, Any]]:
    if not results:
        return []

    scores = [
        float(score)
        for score in reranker.rerank(query, [item["text"] for item in results])
    ]
    if len(scores) != len(results):
        raise RuntimeError("reranker returned a different number of scores than documents")

    for item, score in zip(results, scores):
        item["rerank_score"] = score

    return sorted(
        results,
        key=lambda item: (-item["rerank_score"], item["chunk_id"]),
    )[:top_k]


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
    rerank: bool = True,
    rerank_model: str = RERANK_MODEL,
    rerank_top_k: int = DEFAULT_RERANK_TOP_K,
    retrieval_mode: str = "hybrid",
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
    if rerank_top_k < 1:
        raise ValueError("rerank_top_k must be >= 1")
    if retrieval_mode not in ("hybrid", "dense"):
        raise ValueError(f"unknown retrieval_mode: {retrieval_mode}")

    collection = get_collection(chroma_dir, strategy)
    embedder = get_embedder(model_name)

    dense_results = dense_search(collection, embedder, query, top_k)

    # Initialize variables used by both paths
    fused_results = dense_results
    keyword_results = []
    rrf_top = None

    if retrieval_mode == "dense":
        # Dense-only path: skip BM25 and RRF fusion entirely
        results = dense_results
        rerank_model_used = None
        logger.info(
            "query=%r strategy=%s retrieval_mode=dense dense=%d returned=%d",
            query, strategy, len(dense_results), len(results),
        )
    else:
        # Hybrid path: dense + BM25 + RRF + rerank
        bm25_data = get_bm25(bm25_dir, strategy)
        keyword_results = keyword_search(bm25_data, query, top_k)
        fused_results = fuse(
            dense_results, keyword_results, vector_weight, keyword_weight, rrf_k
        )

        rrf_top = None
        if fused_results:
            rrf_top = {
                "chunk_id": fused_results[0]["chunk_id"],
                "source_file": fused_results[0]["source_file"],
                "rrf_score": fused_results[0]["rrf_score"],
            }

        if rerank:
            reranker = get_reranker(rerank_model)
            results = rerank_results(query, fused_results, reranker, rerank_top_k)
            rerank_model_used = rerank_model
        else:
            results = fused_results
            rerank_model_used = None

        logger.info(
            "query=%r strategy=%s retrieval_mode=hybrid dense=%d keyword=%d fused=%d returned=%d rerank=%s",
            query, strategy, len(dense_results), len(keyword_results),
            len(fused_results), len(results), rerank_model_used,
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
        "unique_chunks": len(fused_results),
        "fused_count": len(fused_results),
        "rerank": rerank,
        "rerank_model": rerank_model_used,
        "rerank_top_k": rerank_top_k,
        "rrf_top": rrf_top,
        "result_count": len(results),
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
    parser.add_argument("--rerank-model", default=RERANK_MODEL)
    parser.add_argument("--rerank-top-k", type=int, default=DEFAULT_RERANK_TOP_K)
    parser.add_argument("--no-rerank", action="store_true")
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
        rerank=not args.no_rerank,
        rerank_model=args.rerank_model,
        rerank_top_k=args.rerank_top_k,
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
    print(
        f"Reranker: {report['rerank_model'] or 'disabled'}  "
        f"Returned: {report['result_count']}"
    )
    for i, item in enumerate(report["results"], start=1):
        rerank_part = (
            f"  rerank {item['rerank_score']:.6f}"
            if "rerank_score" in item
            else ""
        )
        print(
            f"{i:2d}. {item['chunk_id']}  RRF {item['rrf_score']:.6f}"
            f"{rerank_part}  dense #{item['dense_rank']}  "
            f"bm25 #{item['bm25_rank']}  source={item['source_file']}"
        )
    if report["rerank"] and report["rrf_top"] and report["results"]:
        print(
            "Top chunk changed: "
            f"{report['rrf_top']['chunk_id'] != report['results'][0]['chunk_id']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
