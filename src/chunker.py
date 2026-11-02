"""Configurable chunking (Phase 1, Step 2).

Reads processed sectioned documents from data/processed/ and writes one chunk
file per document into data/chunks/<strategy>/.

Strategies:
  fixed      Naive baseline: LangChain RecursiveCharacterTextSplitter with
             token sizing. Code blocks are NOT protected (by design).
  structure  Structure-aware: split oversized sections, merge tiny sibling
             sections, never cut a fenced code block.
  semantic   Reserved for the next step.

No vector store, BM25, deduplication or persisted embeddings here.
"""

from __future__ import annotations

import hashlib
import bisect
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNKER_VERSION = "1.0.0"

DEFAULT_CONFIG: Dict[str, Dict[str, int]] = {
    "fixed": {"chunk_size_tokens": 500, "chunk_overlap_tokens": 50},
    "structure": {"min_tokens": 100, "max_tokens": 800},
    "semantic": {"max_tokens": 800},
}

FENCE_RE = re.compile(r"(?ms)^(`{3,})[^\n]*\n.*?^\1[ \t]*$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[*`])")

logger = logging.getLogger("chunker")


def get_encoder():
    return tiktoken.get_encoding("cl100k_base")


def token_count(text: str, enc=None) -> int:
    enc = enc or get_encoder()
    return len(enc.encode(text))


def code_spans(text: str) -> List[Tuple[int, int]]:
    """Return (start, end) spans of fenced code blocks, fences included."""
    return [(m.start(), m.end()) for m in FENCE_RE.finditer(text)]


# ---------------------------------------------------------------------------
# Fixed-size baseline
# ---------------------------------------------------------------------------

def split_fixed_document(
    text: str, cfg: Dict[str, int]
) -> List[Tuple[str, int, int]]:
    """Split one whole document. Returns (chunk_text, start, end) offsets."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=cfg["chunk_size_tokens"],
        chunk_overlap=cfg["chunk_overlap_tokens"],
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )
    out: List[Tuple[str, int, int]] = []
    for d in splitter.create_documents([text]):
        start = d.metadata.get("start_index", 0)
        out.append((d.page_content, start, start + len(d.page_content)))
    return out


def chunk_fixed(doc: Dict[str, Any], cfg: Dict[str, int], enc) -> Tuple[List[Dict], int, int]:
    """Ignore structure: one text per document, split at ~chunk_size tokens.

    The heading_path/page_number of each chunk come from the section whose
    text contains the chunk's start offset.
    """
    sections = [s for s in doc["sections"] if (s.get("text") or "").strip()]
    parts: List[str] = []
    offsets: List[Tuple[int, List[str], Optional[int]]] = []
    pos = 0
    for section in sections:
        text = section["text"].strip()
        offsets.append((pos, section.get("heading_path", []), section.get("page_number")))
        parts.append(text)
        pos += len(text) + 2  # blank-line separator

    full_text = "\n\n".join(parts)
    if not full_text:
        return [], 0, 0

    spans = code_spans(full_text)
    pieces = split_fixed_document(full_text, cfg)

    code_cut = 0
    for start, end in spans:
        if any(start < cs < end or start < ce < end for _t, cs, ce in pieces):
            code_cut += 1

    starts = [o[0] for o in offsets]
    chunks: List[Dict] = []
    for piece, start, _end in pieces:
        if not piece.strip():
            continue
        i = bisect.bisect_right(starts, start) - 1
        if i < 0:
            i = 0
        _off, heading_path, page_number = offsets[i]
        chunks.append(
            _make_chunk(
                doc,
                "fixed",
                len(chunks),
                heading_path,
                page_number,
                piece,
                enc,
            )
        )
    return chunks, len(spans), code_cut


# ---------------------------------------------------------------------------
# Structure-aware
# ---------------------------------------------------------------------------

def atomic_units(text: str) -> List[Tuple[str, bool]]:
    """Split text into unbreakable units: whole code blocks and prose sentences."""
    units: List[Tuple[str, bool]] = []
    pos = 0
    for m in FENCE_RE.finditer(text):
        before = text[pos : m.start()]
        units.extend((s, False) for s in _split_prose(before))
        units.append((m.group(0), True))
        pos = m.end()
    units.extend((s, False) for s in _split_prose(text[pos:]))
    return [(u, is_code) for u, is_code in units if u.strip()]


def _split_prose(text: str) -> List[str]:
    if not text.strip():
        return []
    paragraphs = re.split(r"\n{2,}", text)
    sentences: List[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        sentences.extend(SENTENCE_SPLIT_RE.split(para))
    return [s.strip() for s in sentences if s.strip()]


def _split_prose_to_max(text: str, max_tokens: int, enc) -> List[str]:
    """Fallback for a single prose sentence larger than max: split on spaces."""
    words = text.split(" ")
    out: List[str] = []
    buf: List[str] = []
    for w in words:
        candidate = " ".join(buf + [w])
        if buf and token_count(candidate, enc) > max_tokens:
            out.append(" ".join(buf))
            buf = [w]
        else:
            buf.append(w)
    if buf:
        out.append(" ".join(buf))
    return out


def pack_units(
    units: List[Tuple[str, bool]], max_tokens: int, enc
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Pack units into chunks <= max_tokens; never split a code unit."""
    chunks: List[str] = []
    oversized: List[Dict[str, Any]] = []
    buf: List[str] = []
    buf_tokens = 0
    for unit, is_code in units:
        ut = token_count(unit, enc)
        if is_code and ut > max_tokens:
            if buf:
                chunks.append("\n".join(buf).strip())
                buf, buf_tokens = [], 0
            chunks.append(unit.strip())
            oversized.append({"tokens": ut, "chars": len(unit)})
            continue
        if not is_code and ut > max_tokens:
            if buf:
                chunks.append("\n".join(buf).strip())
                buf, buf_tokens = [], 0
            chunks.extend(_split_prose_to_max(unit, max_tokens, enc))
            continue
        if buf and buf_tokens + ut > max_tokens:
            chunks.append("\n".join(buf).strip())
            buf, buf_tokens = [], 0
        buf.append(unit)
        buf_tokens += ut
    if buf:
        chunks.append("\n".join(buf).strip())
    return [c for c in chunks if c.strip()], oversized


def _parent(path: List[str]) -> Tuple[str, ...]:
    return tuple(path[:-1])


def chunk_structure(
    doc: Dict[str, Any], cfg: Dict[str, int], enc
) -> Tuple[List[Dict], List[Dict]]:
    min_tokens, max_tokens = cfg["min_tokens"], cfg["max_tokens"]
    candidates: List[Dict[str, Any]] = []
    oversized: List[Dict[str, Any]] = []

    for section in doc["sections"]:
        text = section["text"]
        path = section.get("heading_path", [])
        page = section.get("page_number")
        if token_count(text, enc) <= max_tokens:
            candidates.append(
                {"text": text, "heading_path": path, "page_number": page,
                 "tokens": token_count(text, enc)}
            )
            continue
        texts, over = pack_units(atomic_units(text), max_tokens, enc)
        for o in over:
            oversized.append(
                {"source_file": doc["source_file"], "heading_path": path,
                 "page_number": page, **o}
            )
        for t in texts:
            candidates.append(
                {"text": t, "heading_path": path, "page_number": page,
                 "tokens": token_count(t, enc)}
            )

    # Merge tiny consecutive chunks: greedily group until the min is reached,
    # never exceeding the max and never crossing a page boundary.
    merged: List[Dict[str, Any]] = []
    buf: List[Dict[str, Any]] = []
    buf_tokens = 0
    buf_path: List[str] = []
    buf_page: Optional[int] = None

    def flush() -> None:
        nonlocal buf, buf_tokens, buf_path, buf_page
        if buf:
            text = "\n\n".join(x["text"].strip() for x in buf).strip()
            if text:
                merged.append(
                    {
                        "text": text,
                        "heading_path": list(buf_path),
                        "page_number": buf_page,
                        "tokens": token_count(text, enc),
                    }
                )
        buf, buf_tokens, buf_path, buf_page = [], 0, [], None

    for cand in candidates:
        if buf and (
            cand["page_number"] != buf_page
            or buf_tokens + cand["tokens"] > max_tokens
        ):
            flush()
        if not buf:
            buf_path = cand["heading_path"]
            buf_page = cand["page_number"]
        buf.append(cand)
        buf_tokens += cand["tokens"]
        if buf_tokens >= min_tokens:
            flush()
    flush()

    tiny_before = sum(1 for x in merged if x["tokens"] < min_tokens)
    # Trailing tiny chunk: merge into the previous chunk when it still fits max.
    while len(merged) >= 2 and merged[-1]["tokens"] < min_tokens:
        prev, last = merged[-2], merged[-1]
        if prev["page_number"] != last["page_number"]:
            break
        if prev["tokens"] + last["tokens"] > max_tokens:
            break
        prev["text"] = prev["text"].rstrip() + "\n\n" + last["text"].lstrip()
        prev["tokens"] = token_count(prev["text"], enc)
        merged.pop()
    tiny_after = sum(1 for x in merged if x["tokens"] < min_tokens)

    chunks: List[Dict] = []
    for item in merged:
        if not item["text"].strip():
            continue
        chunks.append(
            _make_chunk(
                doc,
                "structure",
                len(chunks),
                item["heading_path"],
                item["page_number"],
                item["text"],
                enc,
            )
        )
    return chunks, oversized, {"tiny_before": tiny_before, "tiny_after": tiny_after}


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

def _make_chunk(
    doc: Dict[str, Any],
    strategy: str,
    index: int,
    heading_path: List[str],
    page_number: Optional[int],
    text: str,
    enc,
) -> Dict[str, Any]:
    doc_id = doc["doc_id"]
    chunk_id = hashlib.sha1(
        f"{doc_id}:{strategy}:{index}:{text}".encode("utf-8")
    ).hexdigest()
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "source_file": doc["source_file"],
        "chunk_index": index,
        "heading_path": list(heading_path),
        "page_number": page_number,
        "strategy": strategy,
        "char_count": len(text),
        "token_count": token_count(text, enc),
        "text": text,
    }


def run(
    strategy: str,
    input_root: str | Path = "data/processed",
    output_root: str | Path = "data/chunks",
    config: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, Any]:
    if strategy not in ("fixed", "structure"):
        raise ValueError(f"strategy not built yet: {strategy}")
    cfg = {**DEFAULT_CONFIG[strategy], **(config or {}).get(strategy, {})}
    input_root = Path(input_root)
    out_dir = Path(output_root) / strategy
    out_dir.mkdir(parents=True, exist_ok=True)
    enc = get_encoder()

    files = sorted(input_root.glob("*.json"))
    total_chunks = 0
    total_code_blocks = 0
    total_code_cut = 0
    oversized: List[Dict[str, Any]] = []
    oversized: List[Dict[str, Any]] = []
    per_doc: List[Dict[str, Any]] = []
    tiny_before_total = 0
    tiny_after_total = 0

    for f in files:
        doc = json.loads(f.read_text(encoding="utf-8"))
        if strategy == "fixed":
            chunks, code_total, code_cut = chunk_fixed(doc, cfg, enc)
            total_code_blocks += code_total
            total_code_cut += code_cut
        else:
            chunks, over, stats = chunk_structure(doc, cfg, enc)
            oversized.extend(over)
            tiny_before_total += stats["tiny_before"]
            tiny_after_total += stats["tiny_after"]
            code_total = code_cut = 0
            for s in doc["sections"]:
                code_total += len(code_spans(s["text"]))
            total_code_blocks += code_total

        payload = {
            "doc_id": doc["doc_id"],
            "source_file": doc["source_file"],
            "strategy": strategy,
            "chunker_version": CHUNKER_VERSION,
            "chunk_count": len(chunks),
            "chunks": chunks,
        }
        (out_dir / f"{doc['doc_id']}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        total_chunks += len(chunks)
        per_doc.append(
            {"source_file": doc["source_file"], "chunks": len(chunks),
             "code_blocks": code_total, "code_cut": code_cut}
        )

    for o in oversized:
        logger.warning(
            "Oversized code block kept whole (%s tokens, %s chars) in %s",
            o["tokens"], o["chars"], o["source_file"],
        )

    return {
        "strategy": strategy,
        "config": cfg,
        "documents": len(files),
        "chunks": total_chunks,
        "code_blocks": total_code_blocks,
        "code_blocks_cut": total_code_cut,
        "oversized_code_blocks": len(oversized),
        "chunks_below_min_before": tiny_before_total,
        "chunks_below_min_after": tiny_after_total,
        "per_doc": per_doc,
    }


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Chunk processed documents")
    parser.add_argument("--strategy", required=True, choices=["fixed", "structure"])
    parser.add_argument("--input", default="data/processed")
    parser.add_argument("--output", default="data/chunks")
    parser.add_argument("--log", default="data/logs/chunker.log")
    parser.add_argument("--fixed-size", type=int, default=500)
    parser.add_argument("--fixed-overlap", type=int, default=50)
    parser.add_argument("--structure-min", type=int, default=100)
    parser.add_argument("--structure-max", type=int, default=800)
    args = parser.parse_args(argv)

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.log, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    config = {
        "fixed": {
            "chunk_size_tokens": args.fixed_size,
            "chunk_overlap_tokens": args.fixed_overlap,
        },
        "structure": {
            "min_tokens": args.structure_min,
            "max_tokens": args.structure_max,
        },
    }
    report = run(args.strategy, args.input, args.output, config)
    logger.info(
        "strategy=%s docs=%d chunks=%d code_blocks=%d code_blocks_cut=%d oversized_code_blocks=%d",
        report["strategy"], report["documents"], report["chunks"],
        report["code_blocks"], report["code_blocks_cut"],
        report["oversized_code_blocks"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
