Build Phase 2, Part 1: Hybrid Search and RRF Fusion.

- Create a new script for the Retriever pipeline.
- It must take a query, embed it using the same `fastembed` model (`bge-small-en-v1.5`), and fetch the top 10 chunks from ChromaDB.
- It must also fetch the top 10 chunks from the BM25 index using the exact text of the query.
- Combine both lists using Reciprocal Rank Fusion (RRF). Make the vector/keyword weights configurable (start with 0.5/0.5).
- Do NOT build the cross-encoder reranker yet.
- Run a quick test query: "How do I start a FastAPI server?" against the `structure` collection.

Report, short:

- The total number of unique chunks returned after fusion.
- The top 2 chunk IDs, their source files, and their final RRF scores.
  Then STOP.
