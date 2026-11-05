# Phase 4 Evaluation Harness Post-Mortem

**Date:** November 2026  
**Status:** Resolved  
**Scope:** 50-question RAG evaluation run  
**Outcome:** 50/50 completed with no checkpoint data loss

## 1. Summary

The Phase 4 evaluation harness ran the Hybrid RAG pipeline against the 50-question golden dataset. The run was interrupted twice by third-party free-tier limits:

1. Gemini's daily generation quota was exhausted after 10 questions were checkpointed.
2. The Groq Qwen judge later hit its daily token limit at 38 questions.

The generator was migrated from Gemini to Groq, the judge was switched from Qwen to `openai/gpt-oss-20b`, and the evaluation resumed from `data/eval/checkpoint.json` until all 50 questions were complete.

The final metrics are:

- Retrieval relevance: **84.0%**
- Answer correctness: **68.0%**
- Faithfulness: **98.0%**
- Citation accuracy: **74.0%**

These are **mixed-judge metrics**: questions 1–38 were judged by Qwen, and questions 39–50 were judged by `openai/gpt-oss-20b`.

## 2. Timeline

1. The evaluation harness was connected to the real retriever and generator, with Groq used as the judge.
2. The retriever was refactored to lazily cache the embedder, reranker, Chroma client, collection, and BM25 index.
3. The first 10 questions were checkpointed. Gemini then returned a `429 RESOURCE_EXHAUSTED` free-tier daily quota error.
4. The run stopped and was resumed after the generator was migrated from Gemini to Groq.
5. Because the requested Llama models were unavailable, generation used `openai/gpt-oss-120b`.
6. Evaluation continued through question 38. The Qwen judge then hit its daily token limit.
7. The judge was switched to `openai/gpt-oss-20b`.
8. The first attempt with the new judge returned empty output at question 42 because the judge was limited to 8 output tokens.
9. The judge request was changed to `max_tokens=1024` with `reasoning_effort="low"` and one retry on empty/invalid output.
10. The remaining questions were completed, reaching 50/50.

## 3. Root Causes

### 3.1 Free-tier API quotas

Gemini's free-tier daily generation quota was exhausted. The error was explicit and immediate; the evaluation script did not hang or retry indefinitely. The Gemini client used by the evaluation script had no retry configuration, so the request failed on the first `429`.

Later, the Groq Qwen judge exhausted its daily token quota:

- Limit: 200,000 tokens/day
- Used: 199,479
- Requested: 1,786

The API suggested a short retry delay, but because this was a daily token limit, waiting briefly was not a reliable recovery path.

### 3.2 Sequential judge calls amplified latency

Each question requires two base judge calls plus one additional judge call per citation. Longer answers with multiple citations therefore produce several sequential Groq requests. This was the dominant source of per-question latency after retrieval caching was in place.

Observed post-caching timings for questions 11–50 averaged approximately **28.7 seconds per question**. Q16 took **72.8 seconds**, and Q46 took **83.1 seconds**. The latency was not primarily caused by retriever cold starts.

### 3.3 Judge token budget too small for a reasoning model

`openai/gpt-oss-20b` uses reasoning tokens. With `max_tokens=8`, the model could consume the entire budget before producing a final `1` or `0`, resulting in empty content. Raising the limit to `1024` and setting `reasoning_effort="low"` resolved this.

### 3.4 Mixed-judge baseline

The generator and judge changed during the run. The final metrics combine judgments from two different models, so they should not be interpreted as a single consistent baseline without that caveat.

## 4. Mitigations

### 4.1 Global model caching

`src/retriever.py` now uses module-level lazy caches for the embedder, reranker, Chroma client, collection, and BM25 data. Models are initialized once per process instead of once per query.

This removed repeated initialization overhead, but it did not eliminate the dominant judging latency.

### 4.2 Checkpointing

`scripts/evaluate_pipeline.py` writes results to `data/eval/checkpoint.json` after each completed question. The checkpoint key is a SHA-256 hash of the question text. On restart, completed questions are skipped.

This prevented data loss when the run was interrupted by quota errors.

### 4.3 Batched execution

The `--batch-size` argument limits how many new questions are evaluated per run. It does not add sleeps or automatic rate-limit cooldowns; it only stops the run after the requested number of new evaluations.

### 4.4 Generator migration

`src/generator.py` was migrated from `google.genai` to Groq. It initializes the client from `GROQ_API_KEY` and preserves the output shape expected by the evaluation script:

- `parsed.answer`
- `parsed.confidence`
- `verification.citations`

The requested Llama models were not available on the Groq key, so model selection fell back to `openai/gpt-oss-120b`.

### 4.5 Judge migration and retry

The evaluation judge was switched from Qwen to `openai/gpt-oss-20b` after Qwen hit its daily token limit. The judge now:

- Uses `max_tokens=1024` for GPT-OSS models
- Uses `reasoning_effort="low"`
- Retries once if the response contains no `1` or `0`

## 5. Final Outcome

- All 50 questions were evaluated.
- No checkpoint data was lost.
- The interrupted questions were rerun successfully.
- The final metric set was captured and is stored in `data/eval/checkpoint.json`.

## 6. Corrections to the Earlier Draft

The earlier Gemini-written report contained several inaccurate or overstated claims:

- **“The script stalled/hung with no fail-fast”** — False. Gemini returned an explicit `429` and the script stopped immediately. There was no retry loop.
- **“Both generator and judge were migrated to Groq mid-run”** — False. The judge was already on Groq/Qwen. Only the generator was migrated from Gemini. The judge was later switched from Qwen to `gpt-oss-20b`.
- **“Cold start caused the massive latency”** — Overstated. Caching was a real improvement, but post-caching runs still averaged about 28.7 seconds per question for Q11–50. Sequential judge calls were the main latency source.
- **“`--batch-size` throttles requests and respects cooling periods”** — False. It only limits the number of new questions per run; it does not sleep or throttle individual requests.
- **“The judge returned empty JSON strings”** — Inaccurate. It returned empty content with no `1` or `0`; the output was not malformed JSON.
- **“Safely paused”** — Misleading. The run terminated with an exception. Checkpointing made it safe to resume.
- **Final metrics presented without the mixed-judge caveat** — Material omission. Q1–38 used Qwen, while Q39–50 used `gpt-oss-20b`.

## 7. Lessons

- Checkpoint long-running API evaluations after every completed item.
- Treat provider quota limits as expected failure modes, not exceptional edge cases.
- Measure latency by stage so API judging cost is not mistaken for retrieval cost.
- Validate judge output budgets for reasoning models before starting a long run.
- Record which model produced each part of a metric when models change mid-evaluation.
