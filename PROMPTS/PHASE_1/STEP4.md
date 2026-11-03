Build Phase 1, Step 1.4: Deduplication.

- Write a post-processing script to scan the existing 3 ChromaDB collections.
- Find chunk pairs within the same collection that have a cosine similarity > 0.95.
- Remove the duplicates from ChromaDB AND remove their corresponding `chunk_id` from the BM25 `.pkl` files so the 1:1 sync is perfectly maintained.
- Leave the original JSON chunk files untouched (they are our raw ground truth).
- Run the script across all 3 strategies.

Report, short:

- Total duplicates found and removed per strategy (fixed, structure, semantic).
- 1 real example of a text pair that was flagged as a duplicate >0.95.
- Rerun the sync verification check (total counts match, 5/5 ID text match) and confirm it still passes.
  Then STOP.
