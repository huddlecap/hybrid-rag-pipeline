## Phase 4: Evaluation

### 1. Ground-Truth Dataset: Human-in-the-Loop Auto-Generation & Auto-Mapping

Why: Using the LLM to generate draft questions saves time. Human review guarantees real-world quality. Using the existing retrieval pipeline to auto-map the finalized questions to chunk IDs eliminates hours of tedious manual cross-referencing.

### 2. Auto-Mapper Search Strategy: Pure Semantic vs. Hybrid

Why: We opted for pure semantic vector search over a Hybrid (Keyword + Semantic) approach for the dataset mapping script. While hybrid is strictly better for the final retrieval pipeline, semantic-only mapping avoids the engineering overhead of building a Reciprocal Rank Fusion (RRF) algorithm for a single-use utility script while still providing near-perfect reverse-lookup accuracy.

### 3. Auto-Mapper Embedding Engine: fastembed vs. sentence-transformers

Why: We chose `fastembed` over Hugging Face's `sentence-transformers` for the script's local embedding generation. `fastembed` runs strictly on the CPU with minimal dependencies, purposefully avoiding the ~5GB PyTorch and NVIDIA CUDA bloat that standard ML frameworks force into the environment, keeping the project strictly lightweight.

### 4. Evaluation Harness: Real Pipeline Instead of Mocks

Why: The evaluation script calls the real retriever and generator so the metrics reflect the actual RAG pipeline. It measures retrieval relevance, answer correctness, faithfulness, and citation accuracy, using a Groq model as the binary judge.

### 5. Retriever Caching: Lazy Global Instances

Why: `search()` reuses module-level lazy caches for the embedder, reranker, Chroma client, collection, and BM25 data. Initializing these once per process removes repeated model-loading overhead; it does not eliminate the latency caused by sequential judge API calls.

### 6. Checkpointing: Save After Every Question

Why: Results are written immediately after each question, keyed by a SHA-256 hash of the question text. If the run fails on a quota or API error, completed questions are skipped on restart and no evaluation work is lost. There is one checkpoint file per chunking strategy: `data/eval/checkpoint_fixed.json`, `checkpoint_structure.json`, and `checkpoint_semantic.json`.

### 7. Batched Evaluation: `--batch-size`

Why: The `--batch-size` argument limits how many new questions are evaluated per run, with a default of 10. It does not add sleeps or automatic cooldowns; it simply creates a clean stopping point between resumable batches.

### 8. Generator Provider: Gemini to Groq

Why: Gemini's free-tier daily generation quota was exhausted, so `src/generator.py` was migrated to Groq using `GROQ_API_KEY`. The requested Llama models were unavailable on the key, so model selection falls back to `openai/gpt-oss-120b`. The generator preserves the existing `parsed` and `verification` output shape required by the evaluation script.

### 9. Judge Model: Qwen to `openai/gpt-oss-20b`

Why: The evaluation initially used Groq's `qwen/qwen3.8-27b` as the judge. Qwen reached its 200,000-token daily limit at 38/50 questions, so the judge was switched to `openai/gpt-oss-20b` for a separate quota. This completed the run but means the final metrics are mixed-judge metrics.

### 10. Judge Request Budget and Retry

Why: `openai/gpt-oss-20b` uses reasoning tokens, and an 8-token output budget sometimes produced no final `1` or `0`. The judge now uses `max_tokens=1024`, `reasoning_effort="low"`, and one retry when the response contains no valid score.

### 11. Final Evaluation Outcome

Why: All 50 questions were completed without checkpoint data loss. Final metrics are retrieval 84.0%, correctness 68.0%, faithfulness 98.0%, and citation 74.0%. Questions 1–38 were judged by Qwen and questions 39–50 by `openai/gpt-oss-20b`, so the final numbers are not a single consistent-judge baseline.

### 12. Strategy-Specific Golden Datasets

Why: Each chunking strategy (fixed, structure, semantic) has its own evaluation dataset at `data/eval/golden_dataset_{strategy}.json`. Each entry maps a question to its best-matching chunk ID(s) in that strategy using fastembed cosine similarity, with up to 3 acceptable chunk IDs stored in `ground_truth_chunk_ids` to allow for answers that span chunks.

### 13. Single-Pass Groq Judge

Why: The judge was refactored to evaluate correctness, faithfulness, and citation accuracy in a single API call. It returns JSON with keys `correctness` (0/1), `faithfulness` (0/1), and `unsupported_citations` (list of citation strings). Citation accuracy is computed deterministically in Python: 1 if no citations exist or all citations are supported, else 0. One retry is performed on invalid JSON output.

### 14. Strategy Routing with `--strategy`

Why: `scripts/evaluate_pipeline.py` accepts `--strategy fixed|structure|semantic`. This flag simultaneously routes: (1) the ChromaDB vector collection, (2) the BM25 index, (3) the strategy-specific golden dataset file, and (4) the strategy-specific checkpoint file (`checkpoint_{strategy}.json`).

### 15. Chunking Strategy Comparison Script

Why: `scripts/compare_chunking.py` reads the three checkpoint files and displays N (number of evaluations completed), per-metric averages, and paired wins/losses between strategy pairs. It explicitly marks partial results and refuses to declare an overall winner if N values are mismatched across strategies.

### 16. Chunking Strategy Comparison Report

Why: `comparison_report.md` at the repo root contains the final comparison table showing which chunking strategy wins on retrieval relevance, answer correctness, faithfulness, and citation accuracy. Strategies compared are fixed-size, structure-aware, and semantic chunking, each evaluated on 30 questions under identical pipeline conditions.
