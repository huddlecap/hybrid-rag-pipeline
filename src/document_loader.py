"""Multi-format document loader (Phase 1, Step 1).

Accepts markdown, text, HTML and PDF files. Normalizes each document into a
list of sections (text, heading path, page number) and writes one JSON file
per document into data/processed/. Raw files are never modified.

Documents are identified by a content fingerprint (SHA-256 of the raw bytes);
the file name is kept as a readable label. Loading the same content twice never
creates a duplicate document; an edited file produces a new fingerprint.

A failed/empty file is skipped, logged, and loading continues. A summary is
printed at the end.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup, NavigableString, Tag
from markdown_it import MarkdownIt
import pymupdf  # type: ignore
import yaml  # type: ignore

LOADER_VERSION = "1.0.1"

logger = logging.getLogger("document_loader")

HEADINGS = {f"h{i}" for i in range(1, 7)}
BLOCK_TAGS = {
    "p",
    "div",
    "section",
    "article",
    "main",
    "blockquote",
    "pre",
    "table",
    "ul",
    "ol",
    "li",
    "hr",
    "details",
    "summary",
}
# Tags whose rendered output should start on its own line.
BLOCKLIKE = BLOCK_TAGS | {"pre", "table", "ul", "ol", "hr", "blockquote", "details", "summary"}
# Inline tags: their children are joined with spaces, never mid-sentence newlines.
INLINE_TAGS = {
    "a",
    "code",
    "abbr",
    "dfn",
    "kbd",
    "span",
    "strong",
    "em",
    "b",
    "u",
    "i",
    "small",
    "font",
    "sub",
    "sup",
    "mark",
    "q",
    "cite",
}
SKIP_TAGS = {
    "script",
    "style",
    "noscript",
    "iframe",
    "svg",
    "form",
    "nav",
    "header",
    "footer",
    "aside",
}

EXT_LANGUAGE = {
    ".py": "python",
    ".ipynb": "json",
    ".js": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".json": "json",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".fish": "bash",
    ".txt": "text",
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "rst",
    ".dockerfile": "dockerfile",
    ".makefile": "makefile",
    ".cfg": "text",
    ".ini": "text",
    ".conf": "text",
    ".xml": "xml",
    ".sql": "sql",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def language_for_path(path: Path) -> str:
    return EXT_LANGUAGE.get(path.suffix.lower(), "text")


def max_backtick_run(text: str) -> int:
    best = cur = 0
    for ch in text:
        if ch == "`":
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def fence_code(content: str, language: str) -> str:
    runs = max_backtick_run(content)
    n = max(3, runs + 1)
    return "`" * n + language + "\n" + content.rstrip("\n") + "\n" + "`" * n + "\n"


def strip_html_tags(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text)


def is_console_block(el: Tag) -> bool:
    node: Optional[Tag] = el
    for _ in range(6):
        if node is None:
            break
        classes = " ".join(node.get("class", []))
        if "console" in classes or "termy" in classes:
            return True
        node = node.parent if isinstance(node.parent, Tag) else None
    return False


def render_pre(el: Tag) -> str:
    code = el.find("code", recursive=False)
    text = code.get_text() if code is not None else el.get_text()
    if is_console_block(el):
        text = strip_html_tags(text)
    lang_el = code if code is not None else el
    classes = " ".join(lang_el.get("class", []))
    lang = "text"
    m = re.search(r"language-(\S+)", classes)
    if m:
        lang = m.group(1)
    return fence_code(text, lang)


def render_code(el: Tag) -> str:
    return "`" + el.get_text(strip=True) + "`"


def render_table(el: Tag) -> str:
    lines: List[str] = []
    for i, tr in enumerate(el.find_all("tr", recursive=False)):
        cells = [
            c.get_text(" ", strip=True)
            for c in tr.find_all(["th", "td"], recursive=False)
        ]
        if not cells:
            continue
        lines.append(" | ".join(cells))
        if i == 0:
            lines.append(" | ".join(["---"] * len(cells)))
    return "\n".join(lines)


def render_list(el: Tag) -> str:
    items: List[str] = []
    start = 1
    if el.name == "ol":
        try:
            start = int(el.get("start", 1) or 1)
        except ValueError:
            start = 1
    for idx, li in enumerate(el.find_all("li", recursive=False), start):
        marker = f"{idx}." if el.name == "ol" else "-"
        inner = li.get_text(" ", strip=True)
        items.append(f"{marker} {inner}")
    return "\n".join(items)


def render_inline(el: Tag) -> str:
    """Render inline content as a single line; children joined with spaces."""
    parts: List[str] = []
    for child in el.children:
        if isinstance(child, NavigableString):
            t = str(child).strip()
            if t:
                parts.append(t)
        elif isinstance(child, Tag):
            if child.name in SKIP_TAGS:
                continue
            if child.name in INLINE_TAGS:
                parts.append(render_inline(child))
            elif child.name == "br":
                parts.append(" ")
            elif child.name == "img":
                alt = child.get("alt") or ""
                if alt:
                    parts.append(alt)
            elif child.name == "a":
                parts.append(render_inline(child))
    return " ".join(parts)


def process_element(el: Tag) -> str:
    """Render an element to plain text.

    Inline children are joined with spaces (no mid-sentence breaks). Block-like
    children are placed on their own lines. Line breaks appear only between
    paragraphs, headings, list items, tables, and code blocks.
    """
    name = el.name
    if name == "pre":
        return render_pre(el)
    if name == "code":
        return "`" + el.get_text(strip=True) + "`"
    if name == "table":
        return render_table(el)
    if name in ("ul", "ol"):
        return render_list(el)
    if name == "hr":
        return "---"
    if name == "blockquote":
        inner = render_inline(el)
        return "> " + inner if inner else ""
    if name == "a":
        return render_inline(el)
    if name == "img":
        return el.get("alt") or ""
    if name == "br":
        return "\n"

    seq: List[Tuple[str, str]] = []
    for child in el.children:
        if isinstance(child, NavigableString):
            t = str(child).strip()
            if t:
                seq.append(("text", t))
            continue
        if not isinstance(child, Tag):
            continue
        if child.name in SKIP_TAGS:
            continue
        if child.name in HEADINGS:
            seq.append(("block", child.get_text(" ", strip=True)))
            continue
        rendered = process_element(child)
        if not rendered or not rendered.strip():
            continue
        kind = "block" if (child.name in BLOCKLIKE or "\n" in rendered) else "inline"
        seq.append((kind, rendered.strip()))

    result_parts: List[str] = []
    pending_inline: List[str] = []

    def flush_inline() -> None:
        if pending_inline:
            result_parts.append(" ".join(pending_inline))
            pending_inline.clear()

    for kind, val in seq:
        if kind in ("text", "inline"):
            pending_inline.append(val)
        else:
            flush_inline()
            result_parts.append(val)
    flush_inline()
    return "\n".join(result_parts).strip()


def extract_sections_from_html(html_text: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html_text, "html.parser")
    body = soup.body if soup.body else soup

    # Drop noise containers (navigation, scripts, etc.) but keep text elsewhere.
    for tag in body.find_all(SKIP_TAGS):
        tag.decompose()
    for el in body.find_all(True):
        classes = " ".join(el.get("class", []))
        if "toc" in classes or "navigation" in classes or "sidebar" in classes:
            el.decompose()

    sections: List[Dict[str, Any]] = []
    path: List[str] = []
    current: Optional[Dict[str, Any]] = None

    def ensure_current() -> None:
        nonlocal current
        if current is None:
            current = {"heading_path": [], "text_parts": []}

    def finalize_section() -> None:
        nonlocal current
        if current is not None and current["text_parts"]:
            text = "\n".join(current["text_parts"]).strip()
            if text:
                sections.append(
                    {
                        "heading_path": list(current["heading_path"]),
                        "page_number": None,
                        "text": text,
                    }
                )
        current = None

    for child in body.children:
        if isinstance(child, NavigableString):
            t = str(child).strip()
            if t:
                ensure_current()
                assert current is not None
                current["text_parts"].append(t)
            continue
        if not isinstance(child, Tag):
            continue
        if child.name in SKIP_TAGS:
            continue
        if child.name in HEADINGS:
            finalize_section()
            lvl = int(child.name[1])
            heading_text = child.get_text(" ", strip=True)
            path = path[: lvl - 1] + [heading_text]
            current = {"heading_path": list(path), "text_parts": []}
            continue
        rendered = process_element(child)
        if rendered and rendered.strip():
            ensure_current()
            assert current is not None
            current["text_parts"].append(rendered.strip())
    finalize_section()
    return sections


def preprocess_markdown(text: str, source_path: Path, include_root: Path) -> Tuple[str, int, int]:
    """Apply FastAPI/MkDocs-specific preprocessing. Returns (text, missing_includes, include_seen)."""
    lines = text.split("\n")
    out: List[str] = []
    missing = 0
    seen = 0

    # 1. Strip YAML front matter (only at the very start).
    start = 0
    if lines and lines[0].strip() == "---":
        end = None
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end = i
                break
        if end is not None:
            start = end + 1
    lines = lines[start:]

    # 2. Remove whole comment-delimited blocks first.
    text_block = "\n".join(lines)
    text_block = re.sub(
        r"<!--\s*sponsors\s*-->.*?<!--\s*/sponsors\s*-->",
        "",
        text_block,
        flags=re.DOTALL,
    )
    text_block = re.sub(
        r"<!--\s*only-mkdocs\s*-->.*?<!--\s*/only-mkdocs\s*-->",
        "",
        text_block,
        flags=re.DOTALL,
    )
    # Remove any remaining HTML comments.
    text_block = re.sub(r"<!--.*?-->", "", text_block, flags=re.DOTALL)
    lines = text_block.split("\n")

    # 3. Jinja/template macros outside fenced code.
    jinja_depth = 0
    jinja_line_re = re.compile(r"\{\%\s*(for|if|endif|endfor|set|macro|endmacro|include|block|endblock)\b")
    jinja_close_re = re.compile(r"\{\%\s*(endfor|endif|endmacro|endblock)\b")
    jinja_expr_re = re.compile(r"\{\{.*?\}\}")

    # 4. Block / tab directives and includes are handled line by line.
    in_fence = False
    fence_char = ""
    include_re = re.compile(
        r"^\s*\{([*!])>?\s*(\S+)(.*?)\}\s*$",
        re.DOTALL,
    )

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        # Fence tracking protects inserted code blocks and literal fences.
        if stripped.startswith("```") or stripped.startswith("~~~"):
            if not in_fence:
                in_fence = True
                fence_char = stripped[:3]
            elif stripped.startswith(fence_char) and len(stripped) >= 3:
                in_fence = False
                fence_char = ""
            out.append(line)
            i += 1
            continue

        if in_fence:
            out.append(line)
            i += 1
            continue

        # Jinja control lines.
        m = jinja_line_re.search(line)
        if m:
            tag = m.group(1)
            if tag in ("for", "if", "set", "macro", "block"):
                jinja_depth += 1
                i += 1
                continue
            if tag in ("endfor", "endif", "endmacro", "endblock"):
                jinja_depth = max(0, jinja_depth - 1)
                i += 1
                continue
        if jinja_depth > 0:
            i += 1
            continue
        line = jinja_expr_re.sub("", line)

        # Include directives {* path *} / {! path !} / {!> path !}
        im = include_re.match(line)
        if im:
            seen += 1
            kind = im.group(1)
            target_raw = im.group(2)
            options = im.group(3) or ""
            target = re.sub(r"^(?:\.\./)+", "", target_raw.replace("\\", "/"))
            target_path = include_root / target
            if not target_path.is_file():
                logger.warning(
                    "Include target not found in %s: %s (referenced by %s)",
                    kind,
                    target,
                    source_path,
                )
                missing += 1
                i += 1
                continue
            content = target_path.read_text(encoding="utf-8")
            # Apply optional line selection ln[1:2,12:16,29,38:41]
            ln_m = re.search(r"ln\[([^\]]*)\]", options)
            if ln_m:
                selected: List[str] = []
                for token in ln_m.group(1).split(","):
                    token = token.strip()
                    if not token:
                        continue
                    if ":" in token:
                        a, b = token.split(":", 1)
                        a_i = int(a) if a else 1
                        b_i = int(b) if b else len(content.splitlines())
                        selected.extend(
                            content.splitlines()[a_i - 1 : b_i]
                        )
                    else:
                        n = int(token)
                        selected.append(content.splitlines()[n - 1])
                content = "\n".join(selected)
            language = language_for_path(target_path)
            indented = "".join(" " + c for c in content.splitlines(True))
            out.append(fence_code(indented, language))
            i += 1
            continue

        # Block directives: /// type | title ... ///   and   //// tab | Label ... ////
        close_block = re.match(r"^\s*/{3}\s*$", line)
        close_tab = re.match(r"^\s*/{4}\s*$", line)
        open_block = re.match(
            r"^\s*/{3}\s+([A-Za-z]+)(?:\s*\|\s*(.*?))?\s*$", line
        )
        open_tab = re.match(
            r"^\s*/{4}\s+tab\s*\|\s*(.*?)\s*$", line
        )
        if close_block or close_tab:
            # Close marker: drop it; add a blank separator.
            out.append("")
            i += 1
            continue
        if open_block:
            _kind, title = open_block.group(1), (open_block.group(2) or "").strip()
            if title:
                out.append(title + ":")
            i += 1
            continue
        if open_tab:
            label = (open_tab.group(1) or "").strip()
            out.append((label + ":") if label else "")
            i += 1
            continue

        # Heading anchors { #anchor } / {#anchor}
        line = re.sub(r"\s*\{\s*#\s*[^}]*\}", "", line)
        # Generic brace attributes { .class }
        line = re.sub(r"\s*\{[^}]*\}", "", line).rstrip()
        out.append(line)
        i += 1

    return "\n".join(out), missing, seen


def load_markdown(
    path: Path, source_root: Path, include_root: Path
) -> Tuple[List[Dict[str, Any]], int]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8 encoding: {exc}") from exc
    processed, missing, seen = preprocess_markdown(text, path, include_root)
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    html_text = md.render(processed)
    sections = extract_sections_from_html(html_text)
    return sections, missing, seen


def load_text(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8 encoding: {exc}") from exc
    text = text.strip()
    if not text:
        return [], 0
    return [{"heading_path": [], "page_number": None, "text": text}], 0


def load_html(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8 encoding: {exc}") from exc
    sections = extract_sections_from_html(text)
    return sections, 0


def _repeated_header_footer_lines(
    pages_text: List[str], page_count: int
) -> set[str]:
    if page_count < 3:
        return set()
    freq: Dict[str, int] = {}
    for ptext in pages_text:
        nonempty = [ln.strip() for ln in ptext.splitlines() if ln.strip()]
        edge = nonempty[:2] + nonempty[-2:]
        seen = set(edge)
        for ln in seen:
            freq[ln] = freq.get(ln, 0) + 1
    threshold = max(2, (page_count + 1) // 2)
    return {ln for ln, c in freq.items() if c >= threshold}


def load_pdf(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    sections: List[Dict[str, Any]] = []
    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:  # noqa: BLE001 - corrupt/encrypted PDF
        raise ValueError(f"cannot open PDF: {exc}") from exc

    try:
        page_count = doc.page_count
        pages_text: List[str] = []
        for pno in range(page_count):
            page = doc[pno]
            pages_text.append(page.get_text("text"))
    finally:
        doc.close()

    repeated = _repeated_header_footer_lines(pages_text, page_count)

    total_chars = 0
    for pno, ptext in enumerate(pages_text, start=1):
        lines = ptext.splitlines()
        if repeated:
            lines = [
                ln
                for idx, ln in enumerate(lines)
                if not (
                    ln.strip() in repeated
                    and (idx < 2 or idx >= len(lines) - 2)
                )
            ]
        text = "\n".join(lines).strip()
        total_chars += len(text)
        if text:
            sections.append(
                {
                    "heading_path": [],
                    "page_number": pno,
                    "text": text,
                }
            )

    if total_chars < 50:
        raise ValueError("no extractable text (likely a scanned PDF); skipping, no OCR")
    return sections, 0


FORMAT_LOADERS = {
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".txt": load_text,
    ".html": load_html,
    ".htm": load_html,
    ".pdf": load_pdf,
}


def classify_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".md", ".markdown"):
        return "markdown"
    if suffix in FORMAT_LOADERS:
        return suffix.lstrip(".")
    raise ValueError(f"unsupported format: {suffix}")


def write_processed(doc: Dict[str, Any], output_root: Path) -> bool:
    doc_id = doc["doc_id"]
    out_path = output_root / f"{doc_id}.json"
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        merged = False
        existing_files = set(existing.get("source_files", [existing.get("source_file", "")]))
        new_file = doc.get("source_file")
        if new_file and new_file not in existing_files:
            existing_files.add(new_file)
            existing["source_files"] = sorted(existing_files)
            existing["source_file"] = sorted(existing_files)[0]
            merged = True
        if merged:
            out_path.write_text(
                json.dumps(existing, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        return False  # not newly created
    out_path.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return True


def build_document(
    path: Path,
    source_root: Path,
    include_root: Path,
    loader_version: str,
    output_root: Path,
) -> Dict[str, Any]:
    fmt = classify_format(path)
    loader = FORMAT_LOADERS[path.suffix.lower()]
    include_missing = 0
    include_seen = 0
    if fmt == "markdown":
        sections, include_missing, include_seen = loader(path, source_root, include_root)
    else:
        sections, include_missing = loader(path)
    if not sections:
        raise ValueError("no extractable content")
    content_hash = sha256_file(path)
    sections_out = [
        {
            "heading_path": s.get("heading_path", []),
            "page_number": s.get("page_number"),
            "text": s["text"],
        }
        for s in sections
    ]
    char_count = sum(len(s["text"]) for s in sections_out)
    doc = {
        "doc_id": content_hash,
        "content_hash": content_hash,
        "source_file": str(path.relative_to(source_root)),
        "source_files": [str(path.relative_to(source_root))],
        "format": fmt,
        "loader_version": loader_version,
        "section_count": len(sections_out),
        "char_count": char_count,
        "sections": sections_out,
    }
    return doc, include_missing, include_seen


def load_documents(
    input_root: str | Path,
    output_root: str | Path,
    include_root: str | Path,
    loader_version: str = LOADER_VERSION,
) -> Dict[str, Any]:
    input_root = Path(input_root)
    output_root = Path(output_root)
    include_root = Path(include_root)
    output_root.mkdir(parents=True, exist_ok=True)

    supported = set(FORMAT_LOADERS)
    files = sorted(
        p for p in input_root.rglob("*")
        if p.is_file() and p.suffix.lower() in supported
    )

    report: Dict[str, Any] = {
        "loader_version": loader_version,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "include_root": str(include_root),
        "files_in": len(files),
        "documents_loaded": 0,
        "documents_new": 0,
        "documents_unchanged": 0,
        "documents_merged": 0,
        "files_skipped": 0,
        "unique_documents": 0,
        "include_directives_seen": 0,
        "include_targets_missing": 0,
        "skip_details": [],
        "include_missing_details": [],
    }

    existing_hashes: Dict[str, Path] = {}
    for existing in output_root.glob("*.json"):
        try:
            data = json.loads(existing.read_text(encoding="utf-8"))
        except Exception:
            continue
        h = data.get("content_hash") or data.get("doc_id")
        if h:
            existing_hashes[h] = existing

    for path in files:
        rel = str(path.relative_to(input_root))
        try:
            doc, include_missing, include_seen = build_document(
                path, input_root, include_root, loader_version, output_root
            )
        except Exception as exc:  # noqa: BLE001 - skip and continue
            logger.warning("Skipped %s: %s", rel, exc)
            report["files_skipped"] += 1
            report["skip_details"].append({"file": rel, "reason": str(exc)})
            continue

        report["include_directives_seen"] += include_seen
        report["include_targets_missing"] += include_missing
        if include_missing:
            report["include_missing_details"].append(
                {"file": rel, "missing_targets": include_missing}
            )

        doc_id = doc["doc_id"]
        was_new = write_processed(doc, output_root)
        if was_new:
            report["documents_new"] += 1
            existing_hashes[doc_id] = output_root / f"{doc_id}.json"
        else:
            existing = existing_hashes.get(doc_id)
            if existing is not None:
                report["documents_unchanged"] += 1
            else:
                # Same content from a different source file: merged alias.
                report["documents_merged"] += 1
        report["documents_loaded"] += 1

    report["unique_documents"] = len(list(output_root.glob("*.json")))
    return report


def summarize(report: Dict[str, Any]) -> str:
    lines = [
        f"loader version: {report['loader_version']}",
        f"files in: {report['files_in']}",
        f"documents loaded: {report['documents_loaded']}"
        f" (new={report['documents_new']}, unchanged={report['documents_unchanged']}, merged={report['documents_merged']})",
        f"unique documents: {report['unique_documents']}",
        f"files skipped: {report['files_skipped']}",
        f"include directives seen: {report['include_directives_seen']}",
        f"include targets missing: {report['include_targets_missing']}",
        f"counts ok: {report['files_in'] == report['documents_loaded'] + report['files_skipped']}",
    ]
    if report["skip_details"]:
        lines.append("skipped:")
        for d in report["skip_details"]:
            lines.append(f"  - {d['file']}: {d['reason']}")
    if report["include_missing_details"]:
        lines.append("include targets missing:")
        for d in report["include_missing_details"]:
            lines.append(f"  - {d['file']}: {d['missing_targets']} target(s)")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Multi-format document loader")
    parser.add_argument("--input", default="data/raw", help="raw documents root")
    parser.add_argument("--output", default="data/processed", help="processed output root")
    parser.add_argument(
        "--includes", default="data/raw_includes", help="fetched include targets root"
    )
    parser.add_argument(
        "--log", default="data/logs/loader.log", help="log file path"
    )
    parser.add_argument(
        "--version", default=LOADER_VERSION, help="loader version stamp"
    )
    args = parser.parse_args(argv)

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    report = load_documents(args.input, args.output, args.includes, args.version)
    logger.info("\n%s", summarize(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
