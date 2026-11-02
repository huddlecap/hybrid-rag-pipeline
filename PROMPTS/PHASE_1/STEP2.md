Task: chunking (Phase 1, Step 2). Follow BUILDPLAN.md (Phase 1, step 2) and FINAL_DECISIONS.md (Phase 1, Step 2). Do not guess; if something is unclear, ask me.

Rules for the whole task:

- Input: processed JSON files in data/processed/. Do not modify them or the loader.
- Three strategies, switchable with one config setting: fixed-size with overlap, structure-aware (by section headers), semantic.
- No vector store, BM25, deduplication, or saved embeddings. Those are later steps.
- Never cut a code block in half. If a code block is bigger than the max size, keep it whole and log it.
- Every chunk records: chunk id, doc_id, source_file, chunk index, heading path, page number, strategy name, character count, token count, text.
- Save chunks to data/chunks/<strategy>/, one file per document, so strategies never overwrite each other.
- Use LangChain text_splitters where it fits.

Now: do NOT build yet. Tell me briefly: libraries, how you will count tokens, how you will split sentences in text that contains code, and an estimate of the embedding cost for semantic chunking. Then STOP and wait for my approval.

Plan approved, with two changes:

1. Fixed-size is the naive baseline: use the standard splitter and do NOT protect code blocks. Report how many code blocks it cuts in half (a count, not an error). The "never cut a code block" rule applies only to structure-aware and semantic, where it must be 0.
2. Size limits as settings: structure-aware min 100 tokens, max 800 tokens. Semantic max 800 tokens.
   Now build fixed-size and structure-aware only, run both on all documents, and report the table of counts as described. Then STOP.

Good work. Two things before I approve:

1. Fix: when a chunk at the end of a document is under 100 tokens, merge it into the previous chunk if the combined size stays within 800 tokens. Apply to structure-aware only and re-run it.
2. Send the rest of the report I asked for, as a short table per strategy (structure-aware: after the fix): total chunks; average, min, max token size; chunks with a missing heading path; empty chunks; documents with zero chunks; chunks under 100 tokens (before and after the fix). Also confirm that running twice gives identical chunks, that switching the config relabels the strategy on every chunk, and show one real example chunk from each strategy for first-steps.md.
   Then STOP.

Two things:

1. The metric table came through without numbers. Resend it with numbers for both strategies.
2. Fix the fixed-size baseline: it should ignore document structure. For each document, join all section texts in order with blank lines, then split that single text every ~500 tokens with ~50 overlap (standard splitter, code not protected). For each chunk, record the heading_path of the section where the chunk starts, and the page_number of that section. Keep all other fields the same. Re-run fixed-size on all documents, then resend the full table for both strategies, including code blocks cut and the determinism check. Do not change structure-aware.
   Then STOP.

Now build the semantic strategy.

- Use OpenAI text-embedding-3-small only to find topic boundaries. Do not save the embeddings.
- Split at the top ~5% biggest meaning drops within each document, with a max chunk size of 800 tokens. Both are settings.
- Same chunk fields and save to data/chunks/semantic/. Like structure-aware, never cut a code block in half (an oversized one stays whole and is logged).
- Do not change the other two strategies.

Run it on all documents. Report, short:

- The actual embedding cost
- The same metric table as before, with numbers, for semantic only: documents, total chunks, average/min/max/median tokens, chunks with missing heading path, empty chunks, documents with zero chunks, chunks under 100 tokens, code blocks cut, oversized code blocks kept whole
- Confirm that running twice gives identical files
- One real example chunk for first-steps.md
  Then STOP.
