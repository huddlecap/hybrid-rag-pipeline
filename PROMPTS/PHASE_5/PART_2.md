Execute Phase 5 Part 2 exactly as specified in the build plan. We are building a React frontend and making necessary backend adjustments to support side-by-side comparison and confidence scoring via our existing judge.

Step 1: Update Core Logic (`src/retriever.py`)

1. Modify the retrieval search function to accept a `retrieval_mode` argument (default="hybrid").
2. If `retrieval_mode == "dense"`, skip the BM25 reranking/fusion step entirely and return only the ChromaDB vector results.

Step 2: Update the API & Wire the Judge (`api/models.py` and `api/main.py`)

1. In `api/models.py`, add `retrieval_mode: Literal["hybrid", "dense"] = "hybrid"` to `QueryRequest`.
2. In `api/main.py`, update the `/v1/ask` route to pass `request.retrieval_mode` to the retriever.
3. In `/v1/ask`, after generating the answer with the LLM, make a secondary call to our existing Groq judge (from Phase 4 eval logic) passing the question, context, and generated answer.
4. Map the judge's evaluation metrics (e.g., faithfulness, correctness) into the `confidence_scores` dictionary in the `QueryResponse`.

Step 3: Initialize the React Frontend (`dashboard/` directory)

1. Inside a new `dashboard/` directory, initialize a React application using Vite and Tailwind CSS.
2. Ensure the app runs on a distinct port (e.g., 5173) and communicates with the FastAPI backend at `http://localhost:8000`.

Step 4: Build the React UI (`dashboard/src/App.jsx` or equivalent)

1. **Sidebar:** Display backend health status and a list of indexed documents (fetched from `GET /v1/documents`).
2. **Global Toggle:** A "Compare Mode" toggle switch.
3. **Standard Mode (Toggle Off):** A chat interface firing to `/v1/ask` with `mode="hybrid"`. Render the markdown answer, clickable inline citations mapping to chunks, visual confidence score gauges (from the judge), and an accordion/expander showing retrieved chunks ranked by relevance.
4. **Compare Mode (Toggle On):** The UI splits into two side-by-side columns. Submitting a question fires two concurrent requests to `/v1/ask` (one `hybrid`, one `dense`). Render both results side-by-side so we can visually compare the retrieved chunks, answers, and judge confidence scores.
