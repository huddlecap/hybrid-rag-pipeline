Build Phase 5, Part 1: FastAPI Evaluation RAG API.

Chunk 1 (Right now): Install dependencies and create the file structure (Steps 1 & 2).

1. **Install dependencies.** Ensure `fastapi`, `uvicorn`, `pydantic`, `python-dotenv` are installed. These are already in `requirements.txt`; install with `pip install -r requirements.txt` if not already present.

2. **Create file structure.** Create `api/main.py` and `api/models.py` under the project root. `__init__.py` may be empty. Keep the API layer entirely inside `api/` so the core RAG logic in `src/` stays untouched.

-- Stop here, run the file, and report back before continuing. --

--- Part 1.2: Write the exact Pydantic data contracts (Step 3).

3. **Define Pydantic models in `api/models.py`:**

   - `QueryRequest` — one required field: `question: str`.
   - `QueryResponse` — fields: `answer` (str), `citations` (list of `Citation`), `confidence_scores` (dict, optional).
   - `Citation` — fields: `text` (str), `metadata` (dict), `score` (optional float).
   - `IngestRequest` — fields: `text` (str), `metadata` (dict, default `{}`).

Do not skip this step. Strict Pydantic models are the gate that rejects malformed input with a 422 before it ever reaches the RAG pipeline.

-- Stop here, run the file, and report back before continuing. --

--- Part 1.3: Write the routes, connect your RAG pipeline, and boot the server (Steps 4 & 5).

4. **Write `api/main.py`.** Import from `src/retriever` (`search`) and `src/generator` (`generate`). Add `sys.path` insert for `src/` so the API can import the pure Python RAG library.

   Routes:
   - `POST /v1/ask` — takes `question`, calls `retriever.search(strategy="structure")`, then `generator.generate(question, retrieved_chunks)`. Map the generator's `parsed.answer`, `parsed.confidence`, `verification.citations`, and retrieved chunks into `QueryResponse`. Citation list: only include citations the judge marked valid; map 1-based indices into `retrieved_chunks`. Confidence scores dict includes `confidence`, `retrieval_relevance`, and `model`.
   - `POST /v1/ingest` — chunks raw text with `chunker.chunk_structure`, embeds with `fastembed`, stores in ChromaDB collection `structure`, updates `data/index/bm25/structure.pkl`.
   - `GET /v1/documents` — lists unique `source_file` values from ChromaDB metadata.
   - `GET /` — health check returning `{"status": "online", "model": "gpt-oss-120b"}`.

5. **Boot the server.** Run `uvicorn api.main:app --reload` (or `fastapi run`) and confirm the server starts without import errors. Verify `POST /v1/ask` returns a valid `QueryResponse`.

Remove all `# TODO` comments once wired.

-- Report, short:
- Confirm `api/main.py` imports and runs without errors.
- Confirm `POST /v1/ask` returns a valid `QueryResponse`.
- Confirm `GET /v1/documents` returns document metadata.
- Then STOP.