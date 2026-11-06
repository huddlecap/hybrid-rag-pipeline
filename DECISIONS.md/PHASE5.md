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

Why: Instead of relying on the generator to self-grade or using raw vector distance heuristics, the `/v1/ask` route makes a secondary call to the existing Phase 4 Groq judge. This provides rigorous, dimension-based confidence metrics (e.g., faithfulness, correctness) while keeping the system architecture DRY (Don't Repeat Yourself).

### 6. React Frontend with True Split-Screen Comparison

Why: Chose React (Vite/Tailwind) over Streamlit to support complex UI requirements like inline clickable citations. To satisfy the "side-by-side" comparison requirement, the UI implements a "Compare Mode" that splits the screen and fires two concurrent API requests (`hybrid` and `dense`), allowing real-time visual comparison of chunk relevance and answer quality.
