# PROJECT 6

## RAG Pipeline with Hybrid Search Over Internal Docs

**What You're Building:** A production-grade Retrieval-Augmented Generation system that ingests a company's internal documentation, indexes it with both dense vector and sparse keyword search, retrieves the most relevant context for any question, and generates grounded answers with inline source citations.

### WHY THIS PROJECT LANDS INTERVIEWS

RAG is the single most requested skill in AI engineering job descriptions. But most candidates build a toy demo with a single PDF. You're building a system with hybrid retrieval, chunking strategy decisions, and citation verification, the production concerns that separate a real RAG engineer from someone who followed a LangChain quickstart.

### TECH STACK

| COMPONENT        | TOOL / LIBRARY                | WHY THIS CHOICE                  |
| ---------------- | ----------------------------- | -------------------------------- |
| Language         | Python 3.11+                  | Ecosystem standard               |
| Embeddings       | fastembed (BAAI/bge-small-en-v1.5) | $0, fully local, no API key      |
| Vector Store     | ChromaDB or Qdrant            | File-based or containerized      |
| Sparse Search    | BM25 via rank_bm25            | Keyword matching for exact terms |
| LLM              | Groq (llama-3.3-70b-versatile / openai/gpt-oss-120b generator) + Cohere (command-r-plus-08-2024 judge) | Hosted, low-cost generation and auditing |
| Chunking         | Custom (fixed-size, structure-aware, semantic) | In-house, no LangChain dependency |
| API              | FastAPI                       | Async-native, production-grade   |
| Containerization | Docker                        | Reproducible deployment          |

### STEP-BY-STEP BUILD GUIDE

#### Phase 1: Build the Ingestion and Chunking Pipeline

**Day 1-3**

1. **Build a multi-format document loader.** Accept markdown, text, HTML, and PDF files. Normalize everything into clean plaintext with metadata (source file, section heading, page number). Store raw documents alongside processed versions so you can re-index without re-uploading.

2. **Implement configurable chunking.** Build three chunking strategies and make them switchable: fixed-size with overlap (baseline), recursive character splitting by section headers (structure-aware), and semantic chunking that splits on topic boundaries using embedding similarity. Track which strategy each chunk used.

3. **Generate and store embeddings.** Embed every chunk using the local fastembed BAAI/bge-small-en-v1.5 model (384 dimensions). Store in ChromaDB with metadata: source document, chunk index, section heading, chunking strategy, and character count. Build the BM25 index in parallel over the same chunks. Both indexes must stay in sync.

4. **Add deduplication.** Before inserting a chunk, check for near-duplicates (cosine similarity > 0.95 against existing chunks). Flag and skip duplicates. This prevents the retriever from wasting context window slots on redundant content when the same information appears in multiple docs.

#### Phase 2: Build the Hybrid Retrieval Engine

**Day 3-6**

1. **Implement dense retrieval.** Embed the user question with the same local fastembed model, then query the vector store. Return the top-k chunks ranked by cosine similarity. Start with k=10.

2. **Implement sparse retrieval.** Run the same query through BM25 over the chunk corpus. Return top-k by BM25 score. This catches exact keyword matches that semantic search might miss, critical for technical documentation with specific function names, config keys, or error codes.

3. **Build the fusion layer.** Implement Reciprocal Rank Fusion (RRF) to combine dense and sparse results into a single ranked list. RRF assigns scores based on rank position across both lists and merges them. Make the weighting configurable (e.g., 0.7 dense / 0.3 sparse) so you can tune it per use case.

4. **Add a reranker.** After fusion, send the top 20 candidates through a cross-encoder reranker (use a small model or LLM-as-judge) that scores each chunk's relevance to the actual question. Keep the top 5. This second pass dramatically improves precision and is a strong interview talking point.

#### Phase 3: Build the Generation and Citation Layer

**Day 6-9**

1. **Design the grounded generation prompt.** Construct a system prompt that instructs the LLM to answer only from the provided context, cite specific chunks using bracketed references ([1], [2]), and explicitly state when the context doesn't contain enough information to answer. Include the retrieved chunks as numbered context blocks.

2. **Implement citation verification.** After generation, parse the model's citations and verify each one. Does [1] actually support the claim it's attached to? Send each citation-claim pair to an LLM-as-judge for verification. Flag unsupported citations. This is the quality layer most RAG systems skip entirely.

3. **Build the answer confidence scorer.** Score each answer on: retrieval confidence (how relevant were the top chunks?), citation coverage (what percentage of claims have verified citations?), and answer completeness (did the response address all parts of the question?). Return a composite confidence score alongside the answer.

4. **Handle the "I don't know" case gracefully.** If retrieval confidence is below a threshold, don't hallucinate. Return a structured response that says what the system found, what it couldn't find, and which documents might be worth checking manually. This is more useful than a fabricated answer and signals production maturity.

#### Phase 4: Build the Evaluation Framework

**Day 9-11**

1. **Create a golden Q&A dataset.** Write 50+ question-answer pairs by hand, each tied to specific sections of your document corpus. Include straightforward lookups, multi-hop questions (answer requires combining information from two documents), questions with no answer in the corpus, and ambiguous questions.

2. **Implement automated eval metrics.** For each test case, measure: answer correctness (LLM-as-judge against golden answer), faithfulness (are all claims grounded in retrieved context?), retrieval relevance (were the right chunks retrieved?), and citation accuracy (do citations actually support claims?). Run the full suite on every pipeline change.

3. **Build a chunking strategy comparison.** ✅ COMPLETE. Ran the same eval suite across all three chunking strategies (fixed, structure, semantic). Generated comparison report showing which strategy wins on which metrics. Data drives architecture decisions and gives concrete numbers for interviews. See `comparison_report.md` and `scripts/compare_chunking.py`.

#### Phase 5: Expose as an API and Dashboard

**Day 11-13**

1. **Build the FastAPI service.** ✅ COMPLETE. POST /v1/ask accepts a question and returns the answer with citations, confidence scores, and source metadata. GET /v1/documents lists indexed documents grouped by source file. POST /v1/ingest accepts new documents for indexing. Include OpenAPI documentation.

2. **Build a simple query dashboard.** ✅ COMPLETE. React (Vite/Tailwind) frontend where you can ask questions and see: the generated answer with clickable citations, the retrieved chunks ranked by relevance, confidence scores broken down by dimension, and a Standard/Compare mode that runs hybrid vs. dense-only retrieval side by side.

3. **Containerize everything.** ✅ COMPLETE. Docker Compose runs `chromadb`, `api`, a one-shot `seeder`, and the Nginx-served dashboard. The API image bakes both BAAI models at build time, and the seeder indexes the sample corpus idempotently on first boot. See `PHASE5_POSTMORTEM.md` §5 and `DECISIONS/PHASE5.md` #10–15.

See `PHASE5_POSTMORTEM.md` for the Parts 1–3 update and failure report, and `DECISIONS/PHASE5.md` for the rationale behind each backend, UI, and container decision.

#### Phase 6: Polish for Portfolio

**Day 13-14**

1. **Record a demo walkthrough.** Show: ingesting a set of documents, asking questions of varying difficulty, the citation verification catching a hallucination, and the hybrid vs. dense-only comparison. Keep it under 4 minutes.

2. **Write the case study.** Frame it as: "I built a RAG system with hybrid search that achieves X% faithfulness and Y% citation accuracy on a 50-question eval suite." Lead with the numbers. Explain why hybrid beats dense-only for technical documentation. Show the chunking strategy comparison data.

**The rule:** a project only counts when it has an eval, a number, and a tradeoff you can defend out loud.

## What makes a project count

- A number. Every one of these projects produces a measured result: an accuracy delta, an availability figure, an agreement score, a break even volume. A project with no number is a tutorial you followed.

- A README that opens with the result. Benchmark table or demo video first, architecture second, setup instructions last. Nobody scrolls.

- A ninety second demo video. Most reviewers will never run your code. They will watch the video, read the top of the README, and decide.

- Honest failure notes. A section on what did not work and why is the single strongest credibility signal in a portfolio repository.

- Real data. Public filings, your own documents, open datasets. Toy data produces toy results and interviewers can tell instantly.

## Packaging the work

A finished project that nobody can evaluate in two minutes is an unfinished project. Every repository gets the same five things, in this order.

- One line at the top saying what it does and what it achieved, with the number in it.

- The result immediately after: a benchmark table, a dashboard screenshot, or an embedded demo video.

- An architecture diagram, one image, showing the data path rather than every class.

- A short design decisions section covering the two or three real trade offs you made and what you gave up.

- A what did not work section. This is the part experienced engineers read first and the part nobody writes.

## Talking about them in the interview

Lead with the problem and the number, not the stack. Not “I used LangGraph and Redis,” but “the runs were losing four minutes of work on every deploy, so I made the state durable and added a cost ceiling after one run cost eleven dollars.” The second version invites the follow up questions you have already prepared for.

Then have one failure ready per project. Every interviewer asks what went wrong, and the candidates who have a specific answer with a specific fix are the ones who get the offer.
