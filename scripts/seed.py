#!/usr/bin/env python3
"""Seed ChromaDB and the BM25 index on first container boot.

Flow:
1. Wait for ChromaDB to respond (30 attempts, 2s apart).
2. If any collections already exist, exit without touching data.
3. Generate a ~3,500-word synthetic "Huddl API Reference" markdown file.
4. Run the existing loader -> structure chunker -> indexer pipeline.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import chromadb  # noqa: E402
from chromadb.config import Settings  # noqa: E402

from chunker import run as run_chunker  # noqa: E402
from document_loader import LOADER_VERSION, build_document, write_processed  # noqa: E402
from indexer import run as run_indexer  # noqa: E402

DATA_DIR = ROOT / "data"
SEED_CORPUS = DATA_DIR / "seed_corpus.md"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"
CHROMA_DIR = DATA_DIR / "index" / "chroma"
BM25_DIR = DATA_DIR / "index" / "bm25"

STRATEGY = "structure"
MAX_RETRIES = 30
RETRY_SLEEP_SECONDS = 2
TARGET_WORDS = 3500


def make_client():
    host = os.environ.get("CHROMA_HOST")
    port = os.environ.get("CHROMA_PORT")
    if host and port:
        return chromadb.HttpClient(host=host, port=int(port))
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def wait_for_chroma():
    client = make_client()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client.heartbeat()
            print(f"[INFO] ChromaDB is ready (attempt {attempt}/{MAX_RETRIES}).")
            return client
        except Exception as exc:  # connection refused, DNS, etc.
            print(
                f"[INFO] Waiting for ChromaDB ({attempt}/{MAX_RETRIES}): "
                f"{type(exc).__name__}: {exc}"
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)
    raise SystemExit("[ERROR] ChromaDB did not become ready in time.")


def has_collections(client) -> bool:
    try:
        return len(client.list_collections()) > 0
    except Exception:
        return False


RESOURCES: List[Tuple[str, str, str]] = [
    ("workspace", "Workspaces", "a tenant boundary that groups projects, members, and billing settings"),
    ("project", "Projects", "a logical container for documents, collections, and retrieval profiles"),
    ("document", "Documents", "a source file that has been normalized, fingerprinted, and sectioned"),
    ("chunk", "Chunks", "a retrieval unit produced by the structure-aware chunking strategy"),
    ("collection", "Collections", "a named vector index with its own embedding model and metadata schema"),
    ("query", "Queries", "a retrieval request that fuses dense and sparse candidates before reranking"),
    ("answer", "Answers", "a grounded generation result with inline citations and confidence scores"),
    ("citation", "Citations", "a claim-to-source link that is verified by the evaluation judge"),
    ("evaluation", "Evaluations", "a scored run over a golden dataset with per-dimension metrics"),
    ("model", "Models", "a registered embedding, reranking, or generation model available to the workspace"),
    ("dataset", "Datasets", "a versioned set of questions and ground-truth chunk identifiers"),
    ("webhook", "Webhooks", "an outbound HTTP subscription for ingestion and evaluation events"),
    ("api-key", "API Keys", "a scoped credential used to authenticate server-to-server requests"),
    ("usage", "Usage", "token and request counters aggregated per workspace and per day"),
    ("conversation", "Conversations", "an ordered thread of questions and grounded answers"),
    ("feedback", "Feedback", "a thumbs-up or thumbs-down signal used to triage retrieval quality"),
]


def _json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2)


def render_resource(
    resource: str, plural: str, description: str, ordinal: int
) -> str:
    item_id = f"{resource}_{ordinal:04d}"
    list_payload = _json(
        {
            "object": "list",
            "data": [
                {
                    "id": item_id,
                    "status": "active",
                    "created_at": "2026-01-01T00:00:00Z",
                }
            ],
            "next_cursor": None,
        }
    )
    create_payload = _json(
        {
            "name": f"Example {plural[:-1]}",
            "description": description,
            "metadata": {"team": "platform", "environment": "production"},
        }
    )
    item_payload = _json(
        {
            "id": item_id,
            "object": resource,
            "name": f"Example {plural[:-1]}",
            "description": description,
            "status": "active",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-02T00:00:00Z",
            "metadata": {"team": "platform", "environment": "production"},
        }
    )
    return f"""## {plural}

A {resource} is {description}. Every {resource} belongs to exactly one workspace and inherits that workspace's rate-limit tier. All timestamps are returned in RFC 3339 format with a UTC offset, and every record exposes a stable identifier so that retries remain idempotent.

### List {plural}
`GET /v1/{resource}`

Returns a cursor-paginated list of {plural.lower()} visible to the authenticated workspace. Results are ordered by `created_at` in descending order unless the `sort` parameter is supplied. Deleted records are excluded by default and can be requested with `include_deleted=true`.

Query parameters:
- `limit` (integer): page size between 1 and 100. Defaults to 20.
- `cursor` (string): opaque cursor returned by the previous page.
- `status` (string): one of `active`, `paused`, or `archived`.

```json
{list_payload}
```

### Create {plural[:-1]}
`POST /v1/{resource}`

Creates a {resource} and returns the persisted record. Requests are idempotent when the `Idempotency-Key` header is present; replaying the same key within 24 hours returns the original response instead of creating a duplicate.

```json
{create_payload}
```

### Retrieve {plural[:-1]}
`GET /v1/{resource}/{{id}}`

Fetches a single {resource} by identifier. A missing identifier returns `404 not_found`; a record that exists outside the caller's workspace returns `403 forbidden` so that resource existence is not leaked across tenants.

```json
{item_payload}
```

### Delete {plural[:-1]}
`DELETE /v1/{resource}/{{id}}`

Archives a {resource} and schedules its dependent chunks for garbage collection. Deletion is soft by default: the record stays queryable with `include_deleted=true` until the retention window closes. Hard deletion requires the `admin` role and emits an `audit-log` entry.

```json
{{
  "id": "{item_id}",
  "deleted": true,
  "deleted_at": "2026-01-03T00:00:00Z"
}}
```
"""


INTRO = """# Huddl API Reference

The Huddl API is a REST interface for building retrieval-augmented applications over private documentation. It exposes ingestion, hybrid retrieval, grounded generation, and evaluation primitives. All endpoints are versioned under `/v1`, accept and return JSON, and use standard HTTP status codes.

## Authentication

Every request must include an `Authorization: Bearer <token>` header. Tokens are scoped to a single workspace and carry one or more roles: `reader`, `writer`, or `admin`. Requests without a token return `401 unauthorized`. Requests with a valid token but insufficient scope return `403 forbidden`.

## Rate limiting

Rate limits are applied per workspace and per route class. The default tier allows 120 requests per minute for read routes, 40 requests per minute for write routes, and 10 requests per minute for expensive evaluation routes. Bursts of up to 20 requests are tolerated within a rolling ten-second window.

When a limit is exceeded the API returns `429 too_many_requests` with a `Retry-After` header measured in seconds. Clients should implement exponential backoff with jitter, retrying only idempotent requests. The `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset` headers describe the current budget.

## Pagination

List endpoints use opaque cursors rather than numeric offsets. Pass the `cursor` value from a previous response to fetch the next page. A `null` cursor means the listing is exhausted. Page sizes are capped at 100 records.

## Errors

Errors use a consistent envelope with a machine-readable `type`, a human-readable `message`, and an optional `details` object. Validation failures return `422 validation_error` and include the offending field path. Transient provider failures return `503 service_unavailable` and are safe to retry with backoff.

```json
{
  "error": {
    "type": "validation_error",
    "message": "limit must be between 1 and 100",
    "details": {"field": "limit"}
  }
}
```

## Idempotency

Write requests accept an `Idempotency-Key` header. Keys are scoped per workspace and retained for 24 hours. Replaying a key with the same body returns the original response; replaying it with a different body returns `409 conflict`.
"""

OUTRO = """## Health and readiness

`GET /health` returns a lightweight liveness probe with no dependencies. `GET /ready` verifies that ChromaDB is reachable and that the embedding model is loaded. Readiness returns `503` until both checks pass, which lets orchestrators wait before routing traffic.

## Versioning policy

Additive changes such as new optional fields are considered backward compatible and ship without a version bump. Removing or renaming a field requires a new version prefix. Deprecated fields remain available for at least one release cycle and are announced through the changelog webhook.

## Support

For support, open a ticket from the workspace settings page or contact the platform team. Include the request identifier from the `X-Request-Id` response header so that logs can be correlated quickly. Production incidents are acknowledged within one business day.
"""


def generate_corpus() -> str:
    parts: List[str] = [INTRO]
    for ordinal, (resource, plural, description) in enumerate(RESOURCES, start=1):
        parts.append(render_resource(resource, plural, description, ordinal))
        if len("\n\n".join(parts).split()) >= TARGET_WORDS:
            break
    parts.append(OUTRO)
    return "\n\n".join(parts) + "\n"


def main() -> int:
    client = wait_for_chroma()

    if has_collections(client):
        print("[INFO] Database already seeded. Exiting.")
        return 0

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    corpus = generate_corpus()
    SEED_CORPUS.write_text(corpus, encoding="utf-8")
    words = len(corpus.split())
    print(f"[INFO] Generated synthetic corpus ({words} words) at {SEED_CORPUS}")

    print("[INFO] Ingesting 3,500-word synthetic corpus...")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    doc, _, _ = build_document(
        SEED_CORPUS, DATA_DIR, DATA_DIR, LOADER_VERSION, PROCESSED_DIR
    )
    write_processed(doc, PROCESSED_DIR)

    chunk_report = run_chunker(
        STRATEGY, input_root=PROCESSED_DIR, output_root=CHUNKS_DIR
    )
    print(f"[INFO] Chunked corpus: {chunk_report['chunks']} chunks ({STRATEGY})")

    index_report = run_indexer(
        chunks_root=CHUNKS_DIR,
        chroma_dir=CHROMA_DIR,
        bm25_dir=BM25_DIR,
        strategies=(STRATEGY,),
    )
    ingested = index_report["ingested"][0]
    print(
        f"[INFO] Indexed {ingested['chroma_count']} chunks into ChromaDB "
        f"and {ingested['bm25_count']} into {ingested['bm25_path']}"
    )

    print("[INFO] Seeding complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
