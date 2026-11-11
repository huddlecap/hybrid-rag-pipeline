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
