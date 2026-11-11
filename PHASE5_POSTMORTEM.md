# Phase 5 Post-Mortem — API & Dashboard

**Date:** November 2026
**Status:** Phase 5 Parts 1–2 complete; Part 3 (Docker) pending
**Scope:** FastAPI service (`/v1/ask`, `/v1/documents`) and React dashboard
**Outcome:** Judge granularity fixed, confidence split into three dimensions, dashboard shipped and build-verified

## 1. Summary

Phase 5 exposed the RAG pipeline as a FastAPI service and built a React dashboard on top of it. Parts 1–2 are complete. The work also surfaced two AI-architecture failures (a hallucinating builder agent and a binary-only judge) and one reporting failure (the judge claimed Chain-of-Thought reasoning that its schema never emitted). Each was diagnosed and remediated.

Generation remains on Groq. Only the evaluation judge moved to Cohere — this is not a full provider swap.

## 2. Failure Report

### 2.1 Builder agent hallucination

The Nvidia Nemotron 120B builder agent hallucinated during code generation, producing fabricated reports rather than accurate output. The builder was replaced with DeepSeek OpenCode, which restored reliable, verifiable diffs.

### 2.2 Binary-only judge

The Groq-hosted Qwen judge failed to emit granular confidence scores, stubbornly defaulting to binary `0/1` evaluations. A binary judge penalizes partially correct answers, which is why the legacy `comparison_report.md` under-reports faithfulness. The evaluation judge was moved to Cohere `command-r-plus-08-2024` (`api/main.py:56`) via the Cohere `ClientV2` chat API (`api/main.py:46-53`, `:125`), adding `cohere>=4.0.0` (`requirements.txt:14`) and `COHERE_API_KEY`.

### 2.3 Chain-of-Thought claimed but not emitted

A prior audit found the judge prompt claimed to use Chain-of-Thought reasoning but never produced a reasoning field. This was fixed structurally rather than cosmetically:

- New `JudgeResponse` model requires `citation_analysis`, `atomic_claims`, `total_claims`, `supported_claims`, `answer_relevance_score`, and `unsupported_citations` (`api/models.py:57`).
- `raw_judge_response` is now typed `Optional[JudgeResponse]` instead of a loose dict.
- The judge prompt demands a step-by-step `citation_analysis` field and includes a few-shot out-of-bounds citation example.
- `citation_analysis` is part of the validation keys and the output payload.
- `ask_judge()` returns `tuple[Dict[str, float], Dict[str, Any]]`.

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

- **Graded retrieval relevance:** Replaced the binary signal with `sigmoid(max_rerank_score / 3.0)` for hybrid retrieval, retaining a binary fallback for dense-only (`api/main.py:259-261`).
- **Expanded `/v1/ask` contract:** Returns `all_retrieved_chunks` (`rank`, `chunk_id`, `source_file`, `text`, `display_score`, `is_percentage`), a citation `index`, and the typed `raw_judge_response` (`api/main.py:264-290`). `display_score` is a sigmoid probability for reranked results and raw distance for dense results.
- **Rewritten `/v1/documents`:** Returns `List[DocumentFile]` grouped by `source_file` with nested `chunks`, reading `include=["metadatas", "documents"]` (`api/main.py:425-459`).
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

## 5. Verification

- `python -m py_compile` passes for `api/main.py` and `api/models.py`.
- `npm run build` passes for the dashboard.
- Live smoke checks of `/v1/ask` and `/v1/documents` confirmed the new response shapes.
- Spectrum test confirmed fractional judge output (Section 2.4).
- Caveat: there is no automated test suite yet; verification is compile, build, and live smoke testing.

## 6. Known Debt & Next Steps

- `scripts/evaluate_pipeline.py` is still legacy and currently broken: its prompt requests atomic-claim keys while the parser expects `correctness`/`faithfulness` (`scripts/evaluate_pipeline.py:161`), and it references an undefined `prompt` instead of `user_prompt` (`:142` vs `:119`). Tracked in `DO_IT_LATER.md`.
- Re-run the evaluation and refresh `comparison_report.md` after repairing the script.
- CORS origins are hardcoded to `localhost:5173`/`localhost:3000` (`api/main.py:190-195`); make them configurable before Docker.
- Inject `GROQ_API_KEY` and `COHERE_API_KEY` via environment variables; never bake secrets into an image.
- Remove the now-unused Groq judge client in `api/main.py`.
- Proceed to Phase 5 Part 3: Docker/compose with a seed script.

## 7. Lessons

- Validate that a judge's output schema actually matches its prompt; "it claims CoT" is not evidence.
- Prefer fractional claim-count scoring over binary verdicts; binary judges hide partial correctness.
- Evidence beats assertion: the spectrum test made the granularity claim defensible.
- Keep one source of truth for a judge. `api/main.py` and `scripts/evaluate_pipeline.py` diverged and the script silently rotted.
- UI details (time, focus states, scrollbars) are cheap to fix early and expensive to explain later.
