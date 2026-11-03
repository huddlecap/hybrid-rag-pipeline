Build Phase 1, Step 1.3: Syncing embeddings and BM25.

- Use local `fastembed` with `bge-small-en-v1.5` to embed the chunks.
- Create 3 separate ChromaDB collections (one for each strategy: fixed, structure, semantic).
- Use the existing `chunk_id` from the JSON files as the exact ID in both ChromaDB and BM25 to guarantee a 1:1 sync.
- Build a BM25 index (`rank_bm25`) for each strategy and save them to disk as `.pkl` files alongside the ChromaDB directory.
- Run the ingestion for all 3 strategies.
- Build a verification check: ensure total counts match between Chroma DB and BM25, and randomly sample 5 IDs from ChromaDB to prove they point to the exact same text in the JSON/BM25 data.

Report, short:

- Total chunks ingested per collection.
- The file paths and file sizes of the saved `.pkl` files.
- The exact result of the ID cross-check verification.
- Any errors or skipped chunks.
  Then STOP.
