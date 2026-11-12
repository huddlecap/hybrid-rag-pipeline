# High-Performance Hybrid RAG Pipeline

A production-grade, container-first Retrieval-Augmented Generation (RAG) architecture built with FastAPI, React, and ChromaDB. "Zero-config" means one command to boot; you only supply two API keys.

This pipeline implements a two-stage hybrid retrieval system (Vector + BM25) with cross-encoder reranking, completely containerized via Docker Compose for an instant, deterministic deployment.

## System Architecture

- **Frontend:** React / Vite dashboard served via Nginx.
- **Backend:** FastAPI REST API handling retrieval, reranking, generation, and evaluation.
- **Vector Store:** ChromaDB running in a persistent, decoupled container.
- **Generation:** Groq API (`llama-3.3-70b-versatile` with fallbacks) for ultra-low latency text generation.
- **Evaluation Judge:** Cohere `command-r-plus-08-2024` extracts atomic claims and grades faithfulness, answer relevance, and retrieval relevance.
- **Seeding Engine:** Isolated ingestion container that generates a synthetic corpus, computes embeddings, and populates the database on first boot.

## Key Engineering Decisions & Optimizations

### 1. Build-Time Model Caching (Zero Cold-Start Latency)

A common failure point in containerized ML applications is the 504 Gateway Timeout caused by models dynamically downloading weights on the first request.

- **The Fix:** The `Dockerfile` injects a build-time caching layer that explicitly downloads and bakes both the embedding (`BAAI/bge-small-en-v1.5`) and reranking (`BAAI/bge-reranker-base`) models directly into the image.
- **Result:** The API container is entirely immutable, network-independent for embeddings, and serves the very first query instantly.

### 2. Local Edge Inference vs. Cloud Embeddings

Instead of relying on rate-limited cloud embedding APIs (e.g., OpenAI, Cohere), this pipeline utilizes `fastembed` to run BAAI models locally on the CPU.

- **Result:** Bypasses network latency during large-scale ingestion, eliminates token-based API costs, and prevents throttling during parallel document chunking.

### 3. Zero-Dependency Container Health Checks

The official `chromadb` slim image strips standard networking tools (like `wget` and `curl`) and isolates Python from the default shell, which silently breaks Docker's default health checks and stalls container orchestration.

- **The Fix:** Implemented a zero-dependency health check using pure Bash built-in TCP sockets (`/dev/tcp/localhost/8000`) to guarantee deterministic boot sequencing for the API and Seeder containers.

## Quick Start (One-Command Deployment)

Docker and two API keys are the only local requirements. Embedding and reranking models are baked into the image at build time.

1. **Clone and Configure:**

   ```bash
   git clone git@github.com:huddlecap/hybrid-rag-pipeline.git
   cd hybrid-rag-pipeline
   cp .env.example .env
   # then set GROQ_API_KEY and COHERE_API_KEY in .env
   ```

2. **Build and Launch:**

   ```bash
   docker compose up --build -d
   ```

   _Note: The initial build takes a few minutes as it permanently caches the ML models into the API image. The one-shot `seeder` container indexes the sample corpus on first boot and skips work if the collections are already populated._

3. **Access the Pipeline:**
   - **React Dashboard:** `http://localhost:5173`
   - **FastAPI Swagger Docs:** `http://localhost:8000/docs`
   - **ChromaDB:** `http://localhost:8001` (host-mapped)

Generated artifacts (`data/index/`, `data/processed/`, `data/chunks/`) are untracked; the seeder recreates them. `data/raw/` and `data/eval/` remain in the repository.

## Local Development (Without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # set both keys
python scripts/seed.py
uvicorn api.main:app --reload --port 8000
```

In a second terminal:

```bash
cd dashboard
npm install
npm run dev
```

Vite proxies `/v1` and `/health` to `http://localhost:8000`, so the dashboard runs against the local API without CORS changes.

## Upcoming Roadmap

- **RAG Evaluation:** The custom judge-based harness has been built and run, but it is not yet a reproducible one-command suite: `scripts/evaluate_pipeline.py` has known parser/prompt bugs tracked in `DO_IT_LATER.md`. Repairing it and refreshing `comparison_report.md` is the remaining Phase 4 debt.
