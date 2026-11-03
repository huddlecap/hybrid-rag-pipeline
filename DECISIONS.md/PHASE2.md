## Phase 2: Hybrid Retrieval

### 1. Reranker Choice: Local Cross-Encoder (`fastembed` with `bge-reranker`)

Why: Keeps the entire retrieval pipeline 100% free, runs locally, and avoids cloud API rate limits that would crash the heavy evaluation phase.
