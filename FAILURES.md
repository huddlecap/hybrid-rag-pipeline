- **API Reference Stubs:** “API reference pages are mostly stubs, since their content comes from source code docstrings that aren’t in the docs repo.” If those pages hurt retrieval later, the eval will show it, and you’ll have a real failure story.

- **Merged Subsection Headings:** “Structure-aware chunks that merge tiny subsections carry only the first subsection’s heading, so a citation may point at the wrong subsection.”

- **Semantic Micro-Chunks:** “Semantic chunking produced 433 chunks under 100 tokens. It has a max size but no minimum, so near-empty chunks can reach the retriever and waste context slots.”

- **Empty Heading Paths:** “Chunks with an empty heading path exist in every strategy: 1 fixed, 1 structure, and 2 semantic. All come from `translation-banner.md`, which has no headings, so citations from those chunks cannot name a section.”

- **Lexical Mapping Keyword Traps:** TF-IDF keyword search mismatched our golden dataset chunks by falling into keyword traps (e.g., matching random source code stubs instead of actual tutorials). Proved that semantic mapping is strictly required.

- **Dependency Bloat:** Using `sentence-transformers` triggered a massive ~5GB PyTorch/CUDA download, violating our lightweight constraints. Aborted and successfully pivoted to `fastembed` for CPU-only semantic embeddings.
