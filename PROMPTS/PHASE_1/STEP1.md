Task: build ONLY the multi-format document loader (Phase 1, Step 1).
Follow the BUILDPLAN.md and FINAL_DECISIONS.md I attached. Do not guess; if something is unclear, ask me.

Scope

- Build only the loader. No chunking, embeddings, vector store, BM25, or extra features.
- Formats: markdown, text, HTML, PDF.
- Raw files go in data/raw/ and are never modified. Processed output goes in data/processed/, one file per document.
- Scanned PDFs with no extractable text are skipped and logged. No OCR.
- Each processed file records which loader version produced it.

Work in two stages

Before Stage A: download the English markdown docs from the public FastAPI GitHub repository into data/raw/. Download only the English documentation markdown files, nothing else from the repo. Do not modify them. Tell me how many files you got.

Stage A: open a sample of the real FastAPI docs in data/raw/ and report back every special markdown syntax you find (for example include directives, note boxes) and how you propose to handle each. Also tell me which libraries you plan to use, and confirm the PDF one gives text per page. Then STOP and wait for my approval.
Stage B: after I approve, build the loader.

Reporting
Keep reports short: critical points only.

Done means (show me the results of all four)

1. Run on one file of each format. Show the output for each.
2. Run the whole thing twice. No duplicate documents.
3. Run on a corrupt file. It is skipped with a logged reason and does not crash.
4. The counts add up: files in = documents loaded + files skipped.

Approved: all proposed handling in your report, with these decisions:

1. Code includes ({\* \*} and {! !}): do NOT drop them. Fetch the referenced target files from the same FastAPI repo into data/raw_includes/, keeping their relative paths and not modifying them. In the processed output, place each fetched code example inside the section where its pointer was, as a code block. If a target can't be found, skip it, log the file and reason, and report the count.
2. Use Python 3.11+ as in the build plan. Set it up and tell me if anything blocks you.
   Proceed to Stage B. Keep reports short, and show the four "done means" results at the end.

Final cleanup for the document loader, keep the report short.

1. Fix: inline pieces (plain text, inline code, links, abbreviations) are being joined with line breaks, splitting sentences mid-way. Join them with normal spacing within a paragraph. Keep line breaks only between paragraphs, headings, list items, and code blocks. Bump loader_version to 1.0.1 and re-run on all files.

2. Then audit and report counts only (a short table). Do not fix anything else without my approval:

- Processed files: total count, and how many have an empty section or zero sections.
- Leftover markers outside code blocks, with a count for each: ":::", "{\*", "///", "{%", "{{".
- Remaining mid-sentence line breaks: count, plus one example if any.
- data/raw: file count, and confirm every file is byte-identical to the source.
- data/raw_includes: file count.
- data/logs/loader.log: counts of skipped files, missing includes, and warnings.
- Confirm the four "done means" checks still pass.
