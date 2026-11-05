**Part 1: Build the frame and the scoring math (using fake data).**

Create the evaluation harness scaffold in a new file `scripts/evaluate_pipeline.py`.

Requirements:

1. Load the dataset from `data/eval/golden_dataset.json`.
2. Create two mock functions to simulate the RAG pipeline for now:
   - `mock_retrieve(question, ground_truth_id)`: returns a list of 5 dummy chunk IDs. Make sure it includes the `ground_truth_id` ~80% of the time so we can test the scoring math.
   - `mock_generate(question, retrieved_ids)`: returns a dummy string representing an LLM answer.
3. Implement the evaluation loop iterating over the dataset.
4. Implement Metric 1: **Retrieval Relevance**.
   - Check if the `ground_truth_chunk_id` from the dataset exists in the list returned by `mock_retrieve`.
   - Score 1 if present, 0 if missing.
5. At the end of the loop, calculate and print the average Retrieval Relevance score (e.g., "Retrieval Relevance: 80.0%").

Report, short:

- Confirm the script is created, the dataset loads, and Metric 1 is implemented.
  Then STOP.

---

**Part 2: Build the AI judge (grading the fake data)**

Update `scripts/evaluate_pipeline.py` to implement the remaining 3 metrics using the Groq API as an LLM-as-a-judge.

Requirements:

1. Initialize the Groq client (ensure `groq` is installed, use `os.environ.get("GROQ_API_KEY")`).
2. Update `mock_generate` to output a fake answer that occasionally includes a dummy citation like "[chunk_123]".
3. Create a helper function `ask_judge(prompt)` that calls the Groq API (use `llama3-8b-8192` or `qwen-2.5-32b-it` depending on availability) with `temperature=0` and forces the model to return strictly "1" or "0".
4. Implement Metric 2 (Answer Correctness):
   - Prompt the judge to compare the `generated_answer` against the `ground_truth_text`. Is the answer factually correct based on the ground truth? (Return 1 or 0).
5. Implement Metric 3 (Faithfulness):
   - Prompt the judge to evaluate the `generated_answer` against the `retrieved_chunks`. Does the answer rely strictly on the provided context without hallucinating? (Return 1 or 0).
6. Implement Metric 4 (Citation Accuracy):
   - Use regex to extract any bracketed citations (e.g., `[chunk_id]`) from the `generated_answer`.
   - If citations exist, prompt the judge: "Does the specific retrieved chunk support the claims attached to this citation?" (Return 1 or 0). If no citations exist, score it as 0.
7. Track all scores in the loop and print the final averages for all 4 metrics at the end.

Report, short:

- Confirm the Groq integration is complete and all 4 metrics are printing.
  Then STOP.

---

**Part 3: Production Pipeline Integration**

Update `scripts/evaluate_pipeline.py` to use the real RAG pipeline instead of mock functions.

Requirements:

1. Remove `mock_retrieve` and `mock_generate`.
2. Import your actual retrieval logic from the project (e.g., the function that embeds the query and searches ChromaDB/Qdrant using Hybrid search or purely semantic fastembed).
3. Import your actual generation logic from the project (the function that builds the prompt with context and calls the LLM to generate an answer with citations).
4. Update the evaluation loop to pass the `question` to the real retriever to get `retrieved_chunks`.
5. Pass the `retrieved_chunks` and `question` to the real generator to get the `generated_answer`.
6. Keep the exact same 4-metric Groq judge evaluation logic.

Report, short:

- Confirm the mocks are replaced with the real pipeline imports and logic.
  Then STOP.

The diagnostic confirmed that `src/retriever.py` is reloading heavy models per query, causing massive slowdowns. Execute these specific optimizations:

1. **Refactor `src/retriever.py` for Caching:**
   - Move the initialization of `TextEmbedding`, `TextCrossEncoder`, `chromadb.PersistentClient`, and the BM25 file load out of the `search()` function body.
   - Store them as module-level global variables (e.g., using a singleton pattern or lazy-loading `get_embedder()` functions) so they are instantiated exactly ONCE when the module is imported, and reused by `search()`.

2. **Implement Checkpointing in `scripts/evaluate_pipeline.py`:**
   - Create a state-saving mechanism reading/writing to `data/eval/checkpoint.json`.
   - Before evaluating a question, check if its ID is in the checkpoint. If yes, SKIP it.
   - After evaluating a question, immediately append its ID and 4 scores to the JSON file.

3. **Implement Batching in `scripts/evaluate_pipeline.py`:**
   - Add a `--batch-size` CLI argument (default to 10).
   - Track how many _new_ questions have been evaluated in the current run. `break` the loop and exit gracefully after hitting the batch limit.

Report, short:

- Confirm `retriever.py` now reuses instantiated models, and checkpointing/batching are live.
  Then STOP.

The Gemini API has hit a hard rate limit. We need to swap the generator to use Groq instead so we can finish the evaluation run.

Modify `src/generator.py`:

1. Remove all `google.genai` dependencies and logic.
2. Initialize the `Groq` client using `os.environ.get("GROQ_API_KEY")`.
3. Refactor the `generate(question, retrieved_chunks)` function to use Groq (use the `llama-3.3-70b-versatile` or `llama-3.1-8b-instant` model).
4. Ensure the output format still matches what `scripts/evaluate_pipeline.py` expects (returning the generated text and extracting citations).

Report, short:

- Confirm `generator.py` is fully migrated to Groq and Gemini is removed.
  Then STOP.
