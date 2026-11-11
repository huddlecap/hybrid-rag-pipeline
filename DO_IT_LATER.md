# TECHNICAL DEBT: RAG Evaluation Re-run & Metric Update

**Context:**
Deferred re-running the 30-question evaluation dataset to avoid the ~1 hour processing delay caused by Groq free-tier rate limits.

## Current Status (Phase 5 Part 2): Partially Done

- The `/v1/ask` judge in `api/main.py` is upgraded: Cohere `command-r-plus-08-2024`, RAGAS-style atomic-claim extraction, and fractional scores for faithfulness, answer relevance, and retrieval relevance.
- `scripts/evaluate_pipeline.py` is **not** upgraded. Its judge prompt requests `atomic_claims`/`total_claims`/`supported_claims`/`correct_claims`, but the parser still requires the legacy `correctness`/`faithfulness`/`unsupported_citations` keys (`scripts/evaluate_pipeline.py:161`).
- The same function builds `user_prompt` (`:119`) but passes the undefined `prompt` (`:142`), so calling the judge raises `NameError`.
- The previous "The Fix (Already Implemented)" claim in this file was therefore incorrect. Only the API judge was fixed; the evaluation script was not.

## The Problem

`comparison_report.md` still shows the legacy binary-judge numbers (faithfulness 26.7% fixed, 40.0% structure). Those are false negatives: a binary judge scores partially correct answers as 0%. Do not present them as final metrics.

## Action Items (Before Portfolio Finalization)

1. **Repair the eval judge:** pass `user_prompt`, reconcile the parser with the atomic-claim schema, and convert `SCORE_KEYS`/`average()` to fractional values.
2. **Implement rate limiting:** add exponential backoff (e.g. `tenacity`) and a delay inside the `scripts/evaluate_pipeline.py` loop to avoid HTTP 429 rate-limit crashes on the Groq free tier.
3. **Re-run the dataset:** execute all three chunking strategies (fixed, structure, semantic) over the 30-question dataset.
4. **Update the report:** overwrite `comparison_report.md` with the new fractional metric averages.
5. **Retire this file:** once the report is refreshed, delete `DO_IT_LATER.md`; the history remains in git and `PHASE5_POSTMORTEM.md`.
