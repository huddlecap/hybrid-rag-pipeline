"""Phase 1, Step 4: remove near-duplicate chunks from the indexes.

Scans each existing ChromaDB collection for chunk pairs whose cosine
similarity is above a threshold, keeps one chunk per near-duplicate group
(earliest by source_file, chunk_index, chunk_id), deletes the rest from
ChromaDB, and rebuilds the matching BM25 pickle.

The JSON chunk files are never opened for writing: they remain the raw
ground truth. ChromaDB is the source of embeddings and text for this step.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import numpy as np
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from indexer import STRATEGIES, tokenize

DEFAULT_THRESHOLD = 0.95

logger = logging.getLogger("deduper")


def load_collection(
    client, strategy: str
) -> Tuple[Any, List[str], List[str], List[Dict[str, Any]], np.ndarray]:
    collection = client.get_collection(strategy)
    record = collection.get(include=["embeddings", "documents", "metadatas"])

    order = sorted(range(len(record["ids"])), key=lambda i: record["ids"][i])
    ids = [record["ids"][i] for i in order]
    documents = [record["documents"][i] for i in order]
    metadatas = [record["metadatas"][i] for i in order]
    embeddings = np.asarray(record["embeddings"], dtype=np.float64)[order]

    return collection, ids, documents, metadatas, embeddings


def normalized(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def similarity_pairs(
    ids: List[str], embeddings: np.ndarray, threshold: float
) -> List[Tuple[str, str, float]]:
    matrix = normalized(embeddings)
    sims = matrix @ matrix.T
    np.fill_diagonal(sims, -1.0)

    rows, cols = np.triu_indices(len(ids), k=1)
    mask = sims[rows, cols] > threshold
    pairs = [
        (ids[rows[i]], ids[cols[i]], float(sims[rows[i], cols[i]]))
        for i in np.nonzero(mask)[0]
    ]
    pairs.sort(key=lambda p: (-p[2], p[0], p[1]))
    return pairs


def _keep_key(index: int, ids: List[str], metadatas: List[Dict[str, Any]]):
    meta = metadatas[index]
    return (
        str(meta.get("source_file", "")),
        int(meta.get("chunk_index", 0)),
        ids[index],
    )


def duplicate_groups(
    ids: List[str], metadatas: List[Dict[str, Any]], pairs: List[Tuple[str, str, float]]
) -> Tuple[List[str], int]:
    index_by_id = {chunk_id: i for i, chunk_id in enumerate(ids)}
    parent = list(range(len(ids)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b, _ in pairs:
        union(index_by_id[a], index_by_id[b])

    groups: Dict[int, List[int]] = {}
    for i in range(len(ids)):
        groups.setdefault(find(i), []).append(i)

    removed: List[str] = []
    group_count = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        group_count += 1
        keep = min(members, key=lambda i: _keep_key(i, ids, metadatas))
        removed.extend(ids[i] for i in members if i != keep)

    return sorted(removed), group_count


def update_bm25(bm25_path: Path, removed_ids: List[str]) -> Dict[str, Any]:
    with bm25_path.open("rb") as f:
        data = pickle.load(f)

    removed = set(removed_ids)
    keep = [i for i, chunk_id in enumerate(data["chunk_ids"]) if chunk_id not in removed]

    chunk_ids = [data["chunk_ids"][i] for i in keep]
    texts = [data["texts"][i] for i in keep]
    metadatas = [data["metadatas"][i] for i in keep]
    tokenized = [data["tokenized_corpus"][i] for i in keep]

    updated = dict(data)
    updated.update(
        {
            "chunk_ids": chunk_ids,
            "texts": texts,
            "metadatas": metadatas,
            "tokenized_corpus": tokenized,
            "bm25": BM25Okapi(tokenized),
            "count": len(chunk_ids),
        }
    )

    tmp_path = bm25_path.with_suffix(bm25_path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(updated, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(bm25_path)

    return {
        "before": data["count"],
        "after": updated["count"],
        "bytes": bm25_path.stat().st_size,
    }


def verify_sync(
    client,
    strategy: str,
    bm25_path: Path,
    sample_size: int,
    seed: int,
) -> Dict[str, Any]:
    collection = client.get_collection(strategy)
    chroma_ids = collection.get(include=[])["ids"]

    with bm25_path.open("rb") as f:
        bm25_data = pickle.load(f)
    bm25_ids = bm25_data["chunk_ids"]
    bm25_texts = bm25_data["texts"]

    duplicates = len(chroma_ids) - len(set(chroma_ids))
    counts_match = len(chroma_ids) == bm25_data["count"]
    id_sets_match = set(chroma_ids) == set(bm25_ids)

    k = min(sample_size, len(chroma_ids))
    sampled = random.Random(seed).sample(sorted(chroma_ids), k) if k else []
    bm25_index = {chunk_id: i for i, chunk_id in enumerate(bm25_ids)}
    bm25_text_by_id = {
        chunk_id: bm25_texts[bm25_index[chunk_id]] for chunk_id in bm25_ids
    }

    samples: List[Dict[str, Any]] = []
    for chunk_id in sampled:
        rec = collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        chroma_text = rec["documents"][0]
        bm25_text = bm25_text_by_id.get(chunk_id)
        samples.append(
            {
                "chunk_id": chunk_id,
                "strategy": rec["metadatas"][0].get("strategy"),
                "source_file": rec["metadatas"][0].get("source_file"),
                "chroma_equals_bm25": chroma_text == bm25_text,
                "chars": len(chroma_text),
                "preview": chroma_text[:80],
            }
        )

    return {
        "strategy": strategy,
        "chroma_count": len(chroma_ids),
        "bm25_count": bm25_data["count"],
        "counts_match": counts_match,
        "id_sets_match": id_sets_match,
        "duplicate_chroma_ids": duplicates,
        "samples": samples,
        "samples_all_match": all(s["chroma_equals_bm25"] for s in samples),
    }


def dedup_strategy(
    client,
    strategy: str,
    bm25_dir: Path,
    threshold: float,
    dry_run: bool,
) -> Dict[str, Any]:
    collection, ids, documents, metadatas, embeddings = load_collection(client, strategy)
    logger.info(
        "strategy=%s loading done chunks=%d embeddings=%s",
        strategy, len(ids), embeddings.shape,
    )

    pairs = similarity_pairs(ids, embeddings, threshold)
    removed, group_count = duplicate_groups(ids, metadatas, pairs)

    example = None
    if pairs:
        a, b, sim = pairs[0]
        ia, ib = ids.index(a), ids.index(b)
        example = {
            "chunk_id_a": a,
            "chunk_id_b": b,
            "similarity": round(sim, 6),
            "source_file_a": metadatas[ia].get("source_file"),
            "source_file_b": metadatas[ib].get("source_file"),
            "chunk_index_a": metadatas[ia].get("chunk_index"),
            "chunk_index_b": metadatas[ib].get("chunk_index"),
            "text_a": documents[ia],
            "text_b": documents[ib],
            "removed": a if a in set(removed) else b,
        }

    logger.info(
        "strategy=%s pairs>%.2f=%d groups=%d removed=%d",
        strategy, threshold, len(pairs), group_count, len(removed),
    )

    bm25_path = bm25_dir / f"{strategy}.pkl"
    bm25_result: Dict[str, Any] = {}
    if dry_run:
        bm25_result = {"before": len(ids), "after": len(ids) - len(removed)}
    else:
        if removed:
            collection.delete(ids=removed)
        bm25_result = update_bm25(bm25_path, removed)

    after_count = collection.count() if not dry_run else len(ids) - len(removed)
    logger.info(
        "strategy=%s chroma_before=%d chroma_after=%d bm25_before=%s bm25_after=%s",
        strategy, len(ids), after_count,
        bm25_result.get("before"), bm25_result.get("after"),
    )

    return {
        "strategy": strategy,
        "before": len(ids),
        "pairs_over_threshold": len(pairs),
        "duplicate_groups": group_count,
        "removed": len(removed),
        "removed_ids": removed,
        "after": after_count,
        "bm25_before": bm25_result.get("before"),
        "bm25_after": bm25_result.get("after"),
        "bm25_path": str(bm25_path),
        "example_pair": example,
    }


def run(
    chroma_dir: str | Path = "data/index/chroma",
    bm25_dir: str | Path = "data/index/bm25",
    strategies: Tuple[str, ...] = STRATEGIES,
    threshold: float = DEFAULT_THRESHOLD,
    sample_size: int = 5,
    seed: int = 42,
    dry_run: bool = False,
) -> Dict[str, Any]:
    chroma_dir = Path(chroma_dir)
    bm25_dir = Path(bm25_dir)

    logger.info("opening chroma at %s", chroma_dir)
    chroma_host = os.environ.get("CHROMA_HOST")
    chroma_port = os.environ.get("CHROMA_PORT")
    if chroma_host and chroma_port:
        client = chromadb.HttpClient(host=chroma_host, port=int(chroma_port))
    else:
        client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )

    results: List[Dict[str, Any]] = []
    verifications: List[Dict[str, Any]] = []

    for strategy in strategies:
        result = dedup_strategy(client, strategy, bm25_dir, threshold, dry_run)
        verification = verify_sync(
            client, strategy, Path(result["bm25_path"]), sample_size, seed
        )
        logger.info(
            "verify strategy=%s counts_match=%s ids_match=%s duplicates=%d samples_all_match=%s",
            strategy, verification["counts_match"], verification["id_sets_match"],
            verification["duplicate_chroma_ids"], verification["samples_all_match"],
        )
        for sample in verification["samples"]:
            logger.info(
                "sample id=%s chroma==bm25=%s source=%s",
                sample["chunk_id"], sample["chroma_equals_bm25"], sample["source_file"],
            )
        results.append(result)
        verifications.append(verification)

    return {
        "threshold": threshold,
        "dry_run": dry_run,
        "chroma_dir": str(chroma_dir),
        "bm25_dir": str(bm25_dir),
        "strategies": results,
        "verifications": verifications,
    }


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Remove near-duplicate chunks")
    parser.add_argument("--chroma", default="data/index/chroma")
    parser.add_argument("--bm25", default="data/index/bm25")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", action="append", choices=STRATEGIES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", default="data/index/dedup_report.json")
    parser.add_argument("--log", default="data/logs/deduper.log")
    args = parser.parse_args(argv)

    strategies = tuple(args.strategy) if args.strategy else STRATEGIES

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.log, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    report = run(
        args.chroma,
        args.bm25,
        strategies=strategies,
        threshold=args.threshold,
        sample_size=args.sample_size,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
