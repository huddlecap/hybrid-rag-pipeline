- **API Reference Stubs:** “API reference pages are mostly stubs, since their content comes from source code docstrings that aren’t in the docs repo.” If those pages hurt retrieval later, the eval will show it, and you’ll have a real failure story.

- **Merged Subsection Headings:** “Structure-aware chunks that merge tiny subsections carry only the first subsection’s heading, so a citation may point at the wrong subsection.”

- **Semantic Micro-Chunks:** “Semantic chunking produced 433 chunks under 100 tokens. It has a max size but no minimum, so near-empty chunks can reach the retriever and waste context slots.”

- **Empty Heading Paths:** “Chunks with an empty heading path exist in every strategy: 1 fixed, 1 structure, and 2 semantic. All come from `translation-banner.md`, which has no headings, so citations from those chunks cannot name a section.”

- **Lexical Mapping Keyword Traps:** TF-IDF keyword search mismatched our golden dataset chunks by falling into keyword traps (e.g., matching random source code stubs instead of actual tutorials). Proved that semantic mapping is strictly required.

- **Dependency Bloat:** Using `sentence-transformers` triggered a massive ~5GB PyTorch/CUDA download, violating our lightweight constraints. Aborted and successfully pivoted to `fastembed` for CPU-only semantic embeddings.

- **Builder Agent Hallucination:** The Nvidia Nemotron 120B builder agent hallucinated and produced fabricated reports during Phase 5 code generation. Replaced it with DeepSeek OpenCode to restore accurate, verifiable output.

- **Binary-Only Judge:** The Groq-hosted Qwen evaluator refused to emit fractional confidence scores and defaulted to binary `0/1` verdicts, penalizing partially correct answers. Replaced with Cohere `command-r-plus-08-2024` (`api/main.py:56`), which extracts atomic claims and lets Python compute fractional scores.

- **Chain-of-Thought Claimed but Absent:** The judge prompt claimed to reason step by step, but the schema never required a reasoning field. Fixed with a required `citation_analysis` on the `JudgeResponse` model (`api/models.py:57`) plus a few-shot out-of-bounds citation example.

- **Eval Script / API Judge Divergence:** The `/v1/ask` judge was upgraded to fractional Cohere scoring, but `scripts/evaluate_pipeline.py` still parses legacy `correctness`/`faithfulness` keys (`:161`) and references an undefined `prompt` instead of `user_prompt` (`:142` vs `:119`), so the evaluation cannot currently run. Tracked in `DO_IT_LATER.md`.

- **Cold-Start 504s:** The first `/v1/ask` request downloaded the embedding and reranker weights on demand, which took long enough to trip a proxy timeout. Fixed by baking both BAAI models into the API image at build time (`api/Dockerfile:8`).

- **Healthcheck Tooling Missing:** The Chroma slim image lacks `curl`/`wget` and hides Python from the healthcheck shell, so the first healthcheck attempt never became healthy and blocked `api`/`seeder` startup. Replaced with a Bash `/dev/tcp/localhost:8000` check (`docker-compose.yml:8-12`).

- **Oversized Build Context:** The first Docker context included `.venv`, `node_modules`, generated indexes, and raw data, making builds slow and risking stale artifacts leaking into images. Fixed with root and `dashboard/` `.dockerignore` files.

- **Local Dev Proxy Mismatch:** After `API_BASE` moved to a relative `''` for Nginx, the Vite dev server still proxied only `/api`, so `npm run dev` sent `/v1` and `/health` to the Vite server instead of FastAPI. Fixed by proxying `/v1` and `/health` to `localhost:8000`.

- **Seeder/API Startup Race:** `api` waits for Chroma but not for the one-shot `seeder`, so an early query fails on a missing BM25 index. Mitigated by the seeder's idempotent design and documented as remaining debt (gate `api` on `service_completed_successfully` if needed).

- **Chroma Client/Server Version Risk:** The client currently resolves to 1.5.9 while the server image is `chromadb/chroma:latest`; a future `latest` can outrun the client. Pin the server image to the client version.
