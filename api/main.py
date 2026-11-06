from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retriever import search, get_chroma_client, get_embedder
from generator import generate, get_client
from chunker import chunk_structure, get_encoder, DEFAULT_CONFIG
from indexer import tokenize
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
env_path = ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)

from api.models import QueryRequest, QueryResponse, IngestRequest, Citation

# Initialize Groq client for judge calls
_judge_client: Optional[Groq] = None


def get_judge_client() -> Groq:
    global _judge_client
    if _judge_client is None:
        _judge_client = get_client()
    return _judge_client


JUDGE_MODEL = "qwen/qwen3.8-27b"


def ask_judge(question: str, generated_answer: str, context_text: str) -> Dict[str, Any]:
    """Single-pass Groq judge: correctness, faithfulness, unsupported_citations."""
    client = get_judge_client()
    system_prompt = (
        "You are a strict evaluator. Return ONLY a JSON object with exactly "
        'these keys: "correctness" (0 or 1), "faithfulness" (0 or 1), '
        '"unsupported_citations" (list of strings). '
        "correctness=1 means the generated answer is factually correct based on the ground truth. "
        "faithfulness=1 means the answer relies strictly on the retrieved context without hallucination. "
        'unsupported_citations lists every citation string like "[2]" that is NOT supported by the retrieved chunk text. '
        "Return only the JSON object, no markdown fences."
    )
    user_prompt = f"""Question: {question}

Generated answer:
{generated_answer}

Retrieved context:
{context_text}

Evaluate the generated answer. Return a JSON object with:
- correctness: 1 if factually correct based on ground truth knowledge, else 0
- faithfulness: 1 if strictly grounded in retrieved context without hallucination, else 0
- unsupported_citations: list of citation strings like "[2]" that are NOT supported by the retrieved context

Return only the JSON object.
"""
    last_content = ""
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=1024,
            )
            content = (response.choices[0].message.content or "").strip()
            # Strip markdown fences if present
            if content.startswith("```"):
                lines = content.splitlines()
                if lines and lines[0].strip().startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                content = "\n".join(lines).strip()
            data = json.loads(content)
            if all(k in data for k in ("correctness", "faithfulness", "unsupported_citations")):
                return {
                    "correctness": int(data["correctness"]),
                    "faithfulness": int(data["faithfulness"]),
                    "unsupported_citations": list(data["unsupported_citations"]),
                }
        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
            pass
        last_content = content
    raise RuntimeError(f"Judge returned invalid JSON after 2 attempts: {last_content!r}")


app = FastAPI(
    title="Hybrid RAG Pipeline API",
    description="API for questioning documents using structure-aware chunking and Groq 120B.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
def health_check():
    """Verify the API is running."""
    return {"status": "online", "model": "gpt-oss-120b"}


@app.post("/v1/ask", response_model=QueryResponse, tags=["Query"])
def ask_question(request: QueryRequest):
    """Takes a question, retrieves chunks, and generates an answer."""
    try:
        # 1. Retrieval: hybrid dense + BM25 + RRF + rerank
        retrieval = search(
            request.question,
            strategy="structure",
            retrieval_mode=request.retrieval_mode,
            top_k=5,
            rerank=True,
            rerank_top_k=5,
        )
        retrieved_chunks = retrieval["results"]
        retrieved_ids = [chunk["chunk_id"] for chunk in retrieved_chunks]

        # 2. Generation: grounded LLM answer with citations
        generation = generate(request.question, retrieved_chunks)
        parsed = generation["parsed"]
        verification = generation["verification"]

        # 3. Map citations: verification["citations"] holds 1-based indices
        #    into retrieved_chunks. Only include citations the judge marked valid.
        valid_citation_indices = set(verification.get("citations", []))
        citations: List[Citation] = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            if idx not in valid_citation_indices:
                continue
            citations.append(
                Citation(
                    text=chunk.get("text", ""),
                    metadata={
                        "chunk_id": chunk.get("chunk_id", ""),
                        "source_file": chunk.get("source_file", ""),
                        "chunk_index": chunk.get("chunk_index", 0),
                        "heading_path": chunk.get("heading_path", []),
                        "page_number": chunk.get("page_number"),
                    },
                    score=chunk.get("distance") or chunk.get("rerank_score"),
                )
            )

        # 4. Judge evaluation for confidence scores
        context_text = "\n\n".join(
            f"[{i+1}] {chunk['text']}" for i, chunk in enumerate(retrieved_chunks)
        )
        judge_result = ask_judge(request.question, parsed["answer"], context_text)

        # 5. Build response
        return QueryResponse(
            answer=parsed["answer"],
            citations=citations,
            confidence_scores={
                "correctness": judge_result.get("correctness", 0),
                "faithfulness": judge_result.get("faithfulness", 0),
                "retrieval_relevance": 1.0 if retrieved_ids else 0.0,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/ingest", tags=["Data Management"])
def ingest_document(request: IngestRequest):
    """Chunks raw text using the structure-aware chunker and saves to ChromaDB + BM25."""
    try:
        # 1. Chunk the raw text using the structure-aware strategy
        enc = get_encoder()
        doc = {
            "doc_id": "ingested",
            "source_file": request.metadata.get("source_file", "ingested"),
            "sections": [
                {
                    "text": request.text,
                    "heading_path": [],
                    "page_number": None,
                }
            ],
        }
        cfg = DEFAULT_CONFIG["structure"]
        chunks, _, _ = chunk_structure(doc, cfg, enc)

        if not chunks:
            return {
                "status": "success",
                "message": "No chunks produced from input text.",
                "chunks_created": 0,
            }

        # 2. Embed chunks and add to ChromaDB
        client = get_chroma_client()
        collection = client.get_collection("structure")
        embedder = get_embedder()

        texts = [c["text"] for c in chunks]
        embeddings = embedder.embed(texts)
        ids = [c["chunk_id"] for c in chunks]
        metadatas = [
            {
                "chunk_id": c.get("chunk_id", ""),
                "source_file": c.get("source_file", ""),
                "chunk_index": c.get("chunk_index", 0),
                "heading_path": c.get("heading_path", []),
                "page_number": c.get("page_number"),
                "strategy": c.get("strategy", "structure"),
                "char_count": c.get("char_count", 0),
                "token_count": c.get("token_count", 0),
            }
            for c in chunks
        ]

        collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)

        # 3. Update BM25 index
        bm25_path = Path("data/index/bm25") / "structure.pkl"
        import pickle

        if bm25_path.exists():
            with bm25_path.open("rb") as f:
                bm25_data = pickle.load(f)

            existing_ids = list(bm25_data["chunk_ids"])
            existing_texts = list(bm25_data["texts"])
            existing_metadatas = list(bm25_data["metadatas"])

            new_ids = [c["chunk_id"] for c in chunks]
            existing_ids.extend(new_ids)
            existing_texts.extend(texts)
            existing_metadatas.extend(metadatas)

            tokenized = [tokenize(t) for t in existing_texts]
            new_bm25 = BM25Okapi(tokenized)

            updated = {
                "strategy": "structure",
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "chunk_ids": existing_ids,
                "texts": existing_texts,
                "metadatas": existing_metadatas,
                "tokenized_corpus": tokenized,
                "bm25": new_bm25,
                "count": len(existing_ids),
            }

            tmp_path = bm25_path.with_suffix(bm25_path.suffix + ".tmp")
            with tmp_path.open("wb") as f:
                pickle.dump(updated, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(bm25_path)
        else:
            # No existing BM25 index; create a new one from just these chunks
            bm25_path.parent.mkdir(parents=True, exist_ok=True)
            new_ids = [c["chunk_id"] for c in chunks]
            new_texts = [c["text"] for c in chunks]
            new_metadatas = [
                {
                    "chunk_id": c.get("chunk_id", ""),
                    "source_file": c.get("source_file", ""),
                    "chunk_index": c.get("chunk_index", 0),
                    "heading_path": c.get("heading_path", []),
                    "page_number": c.get("page_number"),
                    "strategy": c.get("strategy", "structure"),
                    "char_count": c.get("char_count", 0),
                    "token_count": c.get("token_count", 0),
                }
                for c in chunks
            ]
            tokenized = [tokenize(t) for t in new_texts]
            new_data = {
                "strategy": "structure",
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "chunk_ids": new_ids,
                "texts": new_texts,
                "metadatas": new_metadatas,
                "tokenized_corpus": tokenized,
                "bm25": BM25Okapi(tokenized),
                "count": len(new_ids),
            }
            with bm25_path.open("wb") as f:
                pickle.dump(new_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        return {
            "status": "success",
            "message": "Document successfully chunked and embedded.",
            "chunks_created": len(chunks),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/documents", tags=["Data Management"])
def list_documents():
    """Returns a list of all document metadata currently stored in ChromaDB."""
    try:
        client = get_chroma_client()
        collection = client.get_collection("structure")
        result = collection.get(include=["metadatas"])

        documents: Dict[str, Dict[str, Any]] = {}
        for meta in result.get("metadatas", []):
            key = meta.get("source_file", "unknown")
            if key not in documents:
                documents[key] = meta

        return {"documents": list(documents.values())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))