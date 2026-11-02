## Phase 1, Step 1: Build a multi-format Document Loader

### Corpus: FastAPI docs

**Why:** Technical docs full of exact function names and config keys. This is where keyword search (BM25) should beat vector search alone, which is the result this project is meant to measure.

### 1. Documents are stored as a list of sections

Each section keeps its own text, heading path, and page number.
**Why:** A single block of text has nowhere to hold headings or page numbers. The structure-aware chunker (step 1.2) and precise citations later both need the sections.

### 2. Keep code, headings, and tables as written; drop noise

Dropped: front matter, navigation, scripts, repeated PDF headers and footers.
**Why:** In technical docs the code is the answer. Altering it would weaken exact-term search. Special FastAPI syntax is handled only after the builder reports what it finds in real files.

### 3. Documents are identified by content fingerprint (hash)

The file name is kept alongside as a readable label.
**Why:** Loading the same file twice never creates a duplicate, and an edited file is recognized as changed. Name-based IDs can't do either.

### 4. A failed file is skipped, logged, and loading continues

A loaded/skipped summary is printed at the end.
**Why:** One bad file shouldn't block the rest, and nothing is hidden. The skipped list also feeds the "what did not work" section.

## Phase 1, Step 2: Configurable Chunking

### 1. Fixed-size baseline: ~500 tokens, ~50 tokens overlap

**Why:** The usual starting point, so it's a fair baseline to beat. Kept as a setting so the eval can test other sizes.

### 2. Structure-aware: split big sections, merge tiny ones

Every chunk keeps its heading path.
**Why:** Near-empty chunks waste search results, and oversized ones are what the baseline already does badly. This strategy has to fix both to be a fair competitor.

### 3. Semantic: split at the biggest meaning drops in each document (top ~5%), with a max chunk size

**Why:** No magic number to guess, and it behaves consistently across documents. The max size stops a chunk from growing huge when a document never changes topic.
