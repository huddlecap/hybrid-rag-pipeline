## Phase 5: API and Dashboard

### 1. Strict Data Contracts over Loose JSON

Why: By defining explicit Pydantic models (`QueryRequest`, `QueryResponse`) at the HTTP boundary, malformed requests are rejected with a `422` error before they ever reach the ChromaDB or Groq logic, preventing silent failures downstream.

### 2. Total Separation of Concerns

Why: The web layer is isolated entirely in the `api/` directory. The core RAG logic in `src/` remains a pure Python library, completely unaware of HTTP requests. This makes the system modular and much easier to test.

### 3. Auto-Generated Documentation (Swagger)

Why: By using FastAPI and defining explicit route tags, production-grade API documentation is generated automatically rather than written and maintained by hand.

### 4. Dynamic Retrieval Modes

Why: By updating the core `src/retriever.py` to accept a `retrieval_mode` argument, the backend can dynamically disable the BM25 reranking step for dense-only searches. This allows the API to support exact A/B testing without duplicating the retrieval codebase.

### 5. Dedicated Evaluator for Confidence Scores

Why: Instead of relying on the generator to self-grade or using raw vector distance heuristics, the `/v1/ask` route makes a secondary judge call. The judge is Cohere `command-r-plus-08-2024` (`api/main.py:56`), not the Phase 4 Groq judge, because the Groq-hosted Qwen model defaulted to binary verdicts. The Cohere judge extracts atomic claims and lets Python compute fractional scores for faithfulness, answer relevance, and retrieval relevance.

### 6. React Frontend with True Split-Screen Comparison

Why: Chose React (Vite/Tailwind) over Streamlit to support complex UI requirements like inline clickable citations. To satisfy the "side-by-side" comparison requirement, the UI implements a "Compare Mode" that splits the screen and fires two concurrent API requests (`hybrid` and `dense`), allowing real-time visual comparison of chunk relevance and answer quality.

### 7. Chain-of-Thought Judge Schema

Why: The judge returns a required `citation_analysis` field (enforced by `JudgeResponse` in `api/models.py:57`) before the final scores, and the prompt includes a few-shot out-of-bounds citation example. This makes the reasoning auditable instead of merely claimed.

### 8. Graded Retrieval Relevance

Why: A binary "found / not found" signal hides the difference between a strong and a marginal rerank hit. Hybrid retrieval maps `sigmoid(max_rerank_score / 3.0)` to a 0–1 score, keeping a binary fallback for dense-only mode where no rerank logit exists.

### 9. Explicit Chunk and Document Contracts

Why: `/v1/ask` returns `all_retrieved_chunks` with `display_score` and `is_percentage`, and `/v1/documents` returns nested `DocumentFile`/`DocumentChunk` models grouped by source file. Typed contracts let the dashboard render scores, chunk indices, and expansion state without guessing from metadata.

### 10. Four-Service Compose Topology

Why: `chromadb`, `api`, `seeder`, and `dashboard` are separate services with explicit dependencies instead of one monolith. Chroma keeps a named volume (`chroma_data`), while the API and seeder share `shared_data` for chunks and BM25 indexes. This lets the vector store persist across rebuilds and lets the seeder run to completion without becoming a long-lived process.

### 11. Build-Time Model Baking

Why: Downloading embedding and reranker weights on the first request caused cold-start latency and 504s behind a proxy. The API image downloads `BAAI/bge-small-en-v1.5` and `BAAI/bge-reranker-base` during `docker build` (`api/Dockerfile:8`), trading image size for deterministic first-query latency and no runtime dependency on Hugging Face.

### 12. Environment-Driven Chroma Client

Why: The same Python code must run against a local persisted store and a containerized server. `src/retriever.py`, `src/indexer.py`, and `src/deduper.py` choose `chromadb.HttpClient` when `CHROMA_HOST`/`CHROMA_PORT` are present and `PersistentClient` otherwise, avoiding a Docker-only code fork.

### 13. Idempotent One-Shot Seeder

Why: Reviewers should be able to run `docker compose up` and immediately query a populated store, but restarting the stack must not duplicate data. `scripts/seed.py` runs once, checks the collections, and skips when they are already populated. It is safe to re-run after a partial failure.

### 14. Nginx Same-Origin Reverse Proxy

Why: Serving the built SPA and proxying `/v1/` plus `/health` from the same Nginx origin removes CORS from the production path and matches the relative `API_BASE = ''` used by the dashboard. The API remains directly reachable on port 8000 for Swagger and debugging.

### 15. TCP-Socket Healthcheck

Why: The official Chroma image lacks `curl`/`wget` and hides Python from the healthcheck shell, so conventional checks fail or hang orchestration. A Bash `/dev/tcp/localhost/8000` check needs no external binary and gives Compose a reliable readiness signal (`docker-compose.yml:8-12`).

### 16. Track Inputs, Not Generated Artifacts

Why: `data/raw/` and `data/eval/` are inputs worth versioning, while `data/index/`, `data/processed/`, and `data/chunks/` are reproducible from the raw corpus and the seed script. Untracking the generated trees keeps the repository small and prevents environment-specific Chroma/BM25 state from leaking into commits.
