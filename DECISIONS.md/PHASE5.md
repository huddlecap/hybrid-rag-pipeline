## Phase 5: API and Dashboard

### 1. Strict Data Contracts over Loose JSON

Why: By defining explicit Pydantic models (`QueryRequest`, `QueryResponse`) at the HTTP boundary, malformed requests are rejected with a `422` error before they ever reach the ChromaDB or Groq logic, preventing silent failures downstream.

### 2. Total Separation of Concerns

Why: The web layer is isolated entirely in the `api/` directory. The core RAG logic in `src/` remains a pure Python library, completely unaware of HTTP requests. This makes the system modular and much easier to test.

### 3. Auto-Generated Documentation (Swagger)

Why: By using FastAPI and defining explicit route tags, production-grade API documentation is generated automatically rather than written and maintained by hand.