Build Phase 4, Part 1.1: Interactive Evaluation Question Builder.

- Create a new script `scripts/interactive_eval_builder.py`.
- Load the `.env` file to authenticate the Gemini/Groq API.
- Load the deduplicated chunks from the `data/chunks/structure/` directory.
- Build an interactive terminal loop that executes the following steps one at a time:
  1. Randomly pick one chunk.
  2. Send it to the LLM with the prompt: "You are a Python developer. Read this documentation chunk and write ONE realistic, technical question that a user would ask where this chunk is the exact answer. Output ONLY the plain text question, no JSON, no formatting."
  3. Print the generated question to the terminal clearly.
  4. Wait for user input: `y` (keep), `n` (discard/skip), `q` (quit and save), or any other typed text (treat as a manual rewrite of the question).
  5. If the user keeps or rewrites the question, append the final plain text string to `data/eval/curated_questions.txt`.
- Run the script and interact with it for 2 or 3 questions to verify the terminal flow works.

Report, short:

- Confirm the script successfully pauses for user input and correctly appends to the text file.
  Then STOP.

Part 1.2: Build the auto mapper using fast-embed local

Update `scripts/golden_dataset_mapper.py` to use `fastembed` for semantic vector search.

- Do NOT install `sentence-transformers` or `torch`. Rely entirely on the already-installed `fastembed` and `numpy`.
- Initialize `TextEmbedding()` from `fastembed` (this downloads a tiny <100MB model automatically, no GPU bloat).
- Generate embeddings for the 50 questions and all the chunk texts.
- Calculate cosine similarity using `numpy.dot` (fastembed returns normalized vectors) to find the 1 chunk with the highest similarity score for each question.
- Retain the exact same output format (`golden_dataset.json` with `question`, `ground_truth_chunk_id`, and `ground_truth_text`).

Report, short:

- Confirm the script is updated to use fastembed and numpy.
  Then STOP.
