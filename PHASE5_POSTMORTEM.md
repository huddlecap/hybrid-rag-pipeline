# Phase 5 Post-Mortem — API, Dashboard & Containerization

**Date:** November 2026
**Status:** Phase 5 Parts 1–3 complete
**Scope:** FastAPI service (`/v1/ask`, `/v1/documents`), React dashboard, and Docker Compose stack
**Outcome:** Judge granularity fixed, confidence split into three dimensions, dashboard shipped, and the full stack containerized with a one-shot seeder

## 1. Summary

Phase 5 exposed the RAG pipeline as a FastAPI service, built a React dashboard on top of it, and packaged the API, dashboard, ChromaDB, and seeder into a Docker Compose stack. Parts 1–3 are complete. The work also surfaced two AI-architecture failures (a hallucinating builder agent and a binary-only judge) and one reporting failure (the judge claimed Chain-of-Thought reasoning that its schema never emitted). Each was diagnosed and remediated.

Generation remains on Groq. Only the evaluation judge moved to Cohere — this is not a full provider swap.

## 2. Failure Report

### 2.1 Builder agent hallucination

The Nvidia Nemotron 120B builder agent hallucinated during code generation, producing fabricated reports rather than accurate output. The builder was replaced with DeepSeek OpenCode, which restored reliable, verifiable diffs.

### 2.2 Binary-only judge

The Groq-hosted Qwen judge failed to emit granular confidence scores, stubbornly defaulting to binary `0/1` evaluations. A binary judge penalizes partially correct answers, which is why the legacy `comparison_report.md` under-reports faithfulness. The evaluation judge was moved to Cohere `command-r-plus-08-2024` (`api/main.py:44`) via the Cohere `ClientV2` chat API (`get_cohere_client()` at `api/main.py:37`, model call at `api/main.py:115`), adding `cohere` to `requirements.txt` and `COHERE_API_KEY`.

### 2.3 Chain-of-Thought claimed but not emitted

A prior audit found the judge prompt claimed to use Chain-of-Thought reasoning but never produced a reasoning field. This was fixed structurally rather than cosmetically:

- New `JudgeResponse` model requires `citation_analysis`, `atomic_claims`, `total_claims`, `supported_claims`, `answer_relevance_score`, and `unsupported_citations` (`api/models.py:57`).
- `raw_judge_response` is now typed `Optional[JudgeResponse]` instead of a loose dict.
- The judge prompt demands a step-by-step `citation_analysis` field and includes a few-shot out-of-bounds citation example.
- `citation_analysis` is part of the validation keys and the output payload.
- `ask_judge()` (`api/main.py:47`) returns `tuple[Dict[str, float], Dict[str, Any]]`.

### 2.4 Granularity evidence

A controlled spectrum test confirmed fractional scoring:

| Case | Faithfulness | Answer relevance |
| --- | --- | --- |
| All claims supported | 1.0 | 0.80 |
| Half supported | 0.5 | 0.80 |
| Mostly unsupported | 0.25 | 0.20 |
| None supported | 0.0 | 0.00 |

### 2.5 Generation provider note

Generation is still Groq. `src/generator.py` defaults to `llama-3.3-70b-versatile` with a fallback chain that includes `openai/gpt-oss-120b` (`src/generator.py:20-26`). Only judging moved to Cohere.

## 3. Architectural & Backend Changes

- **Graded retrieval relevance:** Replaced the binary signal with `sigmoid(max_rerank_score / 3.0)` for hybrid retrieval, retaining a binary fallback for dense-only (`api/main.py:252-272`).
- **Expanded `/v1/ask` contract:** Returns `all_retrieved_chunks` (`rank`, `chunk_id`, `source_file`, `text`, `display_score`, `is_percentage`), a citation `index`, and the typed `raw_judge_response` (`api/main.py:252-278`). `display_score` is a sigmoid probability for reranked results and raw distance for dense results.
- **Rewritten `/v1/documents`:** Returns `List[DocumentFile]` grouped by `source_file` with nested `chunks`, reading `include=["metadatas", "documents"]` (`api/main.py:414`).
- **New Pydantic models:** `RetrievedChunk`, `DocumentChunk`, and `DocumentFile` (`api/models.py:37-55`) make the sidebar and chunk views type-safe.
- **Metric cleanup:** Removed the stale `correctness` metric. Confidence now covers faithfulness, answer relevance, and retrieval relevance.
- **Scoring definitions:** faithfulness = `supported_claims / total_claims` (1.0 when there are no claims); answer relevance = `answer_relevance_score / 10`; retrieval relevance as described above.

## 4. Dashboard UI & UX

- **Custom loading state:** Pure-CSS charcoal-to-black "gooey" animation (2.5s) driven by an inline SVG blur filter and high-contrast radial gradients for 3D droplet separation.
- **Button and timer:** Submit button changed from blue to black to match the loader. A live request timer sits beside the loader with `tabular-nums` typography and no emoji, preventing layout jitter while ticking. The loader renders outside the button, not inside it.
- **Chat input:** Removed the redundant outer box and "Question" label, disabled manual resize (`resize-none`), set `rows={3}`, and applied an ultra-diffuse custom shadow (`0_8px_30px_rgb(0,0,0,0.04)`) with a `gray-400` focus ring and deeper focus shadow.
- **Global scrollbars:** Slim 8px scrollbars with a transparent track, rounded `gray-300` thumb (padding-box clip), and `gray-400` on hover.
- **Sidebar assets:** Replaced the paper emoji with a solid Tailwind document SVG. Files expand to show chunk index and full text, with a true total-chunk count and truncation/tooltip fixes.
- **Retrieval modes:** A Standard/Compare segmented control replaces the old checkbox. Compare mode fires concurrent `hybrid` and `dense` requests and renders them side by side.
- **Citations:** Bracketed `[n]` citations are clickable and scroll to `#citation-${rank}`. Matching chunks get a "Cited" badge based on text equality.
- **Top Retrieved Chunks:** Accordion rows with expand/collapse, `line-clamp-3` when collapsed, and score shown as a percentage or raw distance.
- **Per-dimension confidence:** Answer Relevance, Faithfulness, and Retrieval Relevance are displayed as separate values, replacing the stale `correctness` field.
- **Layout:** `h-screen overflow-hidden` shell, full-height `#root`, independent sidebar/main scroll panes, corrected header sizing, and removal of legacy `h1`/`h2` CSS.

## 5. Containerization (Part 3)

Phase 5 Part 3 packages the API, dashboard, vector store, and seeder into a four-service Compose stack.

- **Topology:** `chromadb` (persistent `chroma_data` volume, host port 8001), `api` (host port 8000, `shared_data` volume), one-shot `seeder`, and `dashboard` (Nginx on host port 5173).
- **API image:** Python 3.11 slim; installs `requirements.txt`; bakes `BAAI/bge-small-en-v1.5` (embedding) and `BAAI/bge-reranker-base` (rerank) into the image at build time, eliminating first-request model downloads (`api/Dockerfile:8`).
- **Dashboard image:** multi-stage Node 18 build → `nginx:alpine`, serving the SPA and reverse-proxying `/v1/` and `/health` to `api:8000` (`dashboard/nginx.conf:12-24`).
- **Seeder:** `python scripts/seed.py` runs once, indexes the 3,797-word sample corpus (45 structure chunks), and is idempotent — it skips when the collections are already populated. It shares `shared_data` with the API for chunks and BM25 indexes.
- **Chroma client branch:** `src/retriever.py`, `src/indexer.py`, and `src/deduper.py` use `chromadb.HttpClient` when `CHROMA_HOST`/`CHROMA_PORT` are set, and `PersistentClient` otherwise, so the same code runs inside and outside Docker.
- **Healthcheck:** Chroma's Debian-based image hides Python from the healthcheck shell, so the check uses Bash's `/dev/tcp/localhost/8000` built-in instead of a Python or HTTP client (`docker-compose.yml:8-12`).
- **Secrets:** `GROQ_API_KEY` and `COHERE_API_KEY` are injected at runtime; `.env` is gitignored and excluded from the build context.
- **Build context:** root and `dashboard/` `.dockerignore` files exclude `.venv`, `node_modules`, `dist`, logs, and generated data so the images only receive source.

## 6. Verification

- `python -m py_compile` passes for `api/main.py` and `api/models.py`.
- `npm run build` passes for the dashboard.
- Live smoke checks of `/v1/ask` and `/v1/documents` confirmed the new response shapes.
- Spectrum test confirmed fractional judge output (Section 2.4).
- `scripts/seed.py` verified idempotent: 3,797-word corpus, 45 structure chunks, clean skip on a populated store.
- The Compose YAML was parsed statically.
- **Caveat:** Docker was not installed in the authoring environment, so `docker compose build` and `docker compose up` were not executed. Container boot is unverified end-to-end; this post-mortem does not claim otherwise.
- Caveat: there is no automated test suite yet; verification is compile, build, seeding, and live smoke testing.

## 7. Known Debt & Next Steps

- `scripts/evaluate_pipeline.py` is still legacy and currently broken: its prompt requests atomic-claim keys while the parser expects `correctness`/`faithfulness` (`scripts/evaluate_pipeline.py:161`), and it references an undefined `prompt` instead of `user_prompt` (`:142` vs `:119`). Tracked in `DO_IT_LATER.md`.
- Re-run the evaluation and refresh `comparison_report.md` after repairing the script.
- CORS origins are hardcoded to `localhost:5173`/`localhost:3000` (`api/main.py:177-183`). The containerized dashboard is same-origin through Nginx, so this only affects direct browser calls; make it configurable if a separate frontend origin is added.
- Pin `chromadb/chroma` to the client version used by `requirements.txt` (currently resolves to 1.5.9) instead of `latest` to avoid client/server skew.
- The `api` service depends on Chroma but not on the `seeder`, so a query sent before seeding finishes fails on a missing BM25 index. Gate `api` on `seeder` completing (`condition: service_completed_successfully`) if immediate first-query reliability matters.
- Generated artifacts (`data/index/`, `data/processed/`, `data/chunks/`) are no longer tracked; non-Docker users must run `scripts/seed.py` (or the ingestion pipeline) to recreate them. `data/raw/` and `data/eval/` remain tracked.

## 8. Lessons

- Validate that a judge's output schema actually matches its prompt; "it claims CoT" is not evidence.
- Prefer fractional claim-count scoring over binary verdicts; binary judges hide partial correctness.
- Evidence beats assertion: the spectrum test made the granularity claim defensible.
- Keep one source of truth for a judge. `api/main.py` and `scripts/evaluate_pipeline.py` diverged and the script silently rotted.
- UI details (time, focus states, scrollbars) are cheap to fix early and expensive to explain later.
- Bake ML models at build time; a healthcheck is only as good as the tools present inside the image, not the host.
- Make seeders idempotent and treat the shared data volume as an explicit contract between the seeder and the API.
- Keep raw and evaluation data in version control, but never commit generated indexes or chunks.
