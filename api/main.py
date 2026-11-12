from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cohere
from pydantic import BaseModel, Field

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retriever import search, get_chroma_client, get_embedder
from generator import generate
from chunker import chunk_structure, get_encoder, DEFAULT_CONFIG
from indexer import tokenize
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
env_path = ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)

from api.models import QueryRequest, QueryResponse, IngestRequest, Citation, RetrievedChunk, DocumentFile, DocumentChunk

# Initialize Cohere client for judge calls
_cohere_client: Optional[cohere.ClientV2] = None


def get_cohere_client() -> cohere.ClientV2:
    global _cohere_client
    if _cohere_client is None:
        _cohere_client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    return _cohere_client


JUDGE_MODEL = "command-r-plus-08-2024"


def ask_judge(question: str, generated_answer: str, context_text: str) -> tuple[Dict[str, float], Dict[str, Any]]:
    """RAGAS-style judge with a Chain-of-Thought citation_analysis field.

    The LLM first reasons about citation validity inside ``citation_analysis``
    (an explicit CoT step), then extracts atomic claims, verifies bracketed
    citations against the retrieved context, and scores answer relevance.
    Python performs safe division to produce granular 0.0-1.0 floats.
    """
    system_prompt = (
        "You are a strict, mathematical RAG evaluator. You MUST first reason step-by-step "
        'in a field called "citation_analysis", then produce the final JSON scores. '
        "Return ONLY a single JSON object with exactly these keys: "
        '"citation_analysis" (string), "atomic_claims" (list of strings), '
        '"total_claims" (integer), "supported_claims" (integer), '
        '"answer_relevance_score" (integer 0-10), "unsupported_citations" (list of strings).\n\n'
        "DEFINITIONS:\n"
        "1. citation_analysis: your step-by-step Chain-of-Thought. First enumerate every "
        "context block index that actually exists (the context blocks are numbered [1]..[N]). "
        "Then list every bracketed citation used in the answer. For each citation state whether "
        "it is IN-BOUNDS (the index exists in the context) and whether that exact context block "
        "explicitly supports the attached claim. Any index greater than N (or less than 1) is "
        "OUT-OF-BOUNDS and therefore hallucinated.\n"
        "2. atomic_claims: each distinct factual claim as a separate string. "
        "total_claims MUST exactly equal len(atomic_claims). If the answer is \"I don't know\", "
        "return atomic_claims as [] and total_claims as 0.\n"
        "3. supported_claims: integer count of claims that are BOTH backed by an in-bounds "
        "context block AND carry no hallucinated citation. A claim citing an out-of-bounds "
        "index must NOT be counted as supported.\n"
        "4. answer_relevance_score: integer from 0 to 10 representing how well the generated "
        "answer addresses the user's original question.\n"
        "5. unsupported_citations: list of citation strings (e.g. \"[5]\") that are out-of-bounds "
        "or not supported by the retrieved context.\n\n"
        "FEW-SHOT EXAMPLE:\n"
        "Suppose the Retrieved context contains ONLY 3 blocks: [1], [2], [3]. "
        "The answer says: \"Docker uses containers [1]. Kubernetes orchestrates them [4]. "
        "It scales automatically [5].\"\n"
        "Correct output:\n"
        '{"citation_analysis": "Existing context indices: [1], [2], [3]. Citations used: [1] '
        '(in-bounds and supported by block [1]), [4] (OUT-OF-BOUNDS: no block [4] exists, '
        'hallucinated), [5] (OUT-OF-BOUNDS: no block [5] exists, hallucinated).", '
        '"atomic_claims": ["Docker uses containers", "Kubernetes orchestrates them", '
        '"It scales automatically"], "total_claims": 3, "supported_claims": 1, '
        '"answer_relevance_score": 8, "unsupported_citations": ["[4]", "[5]"]}\n\n'
        "Return only the JSON object, no markdown fences."
    )
    user_prompt = f"""Question: {question}

Generated answer:
{generated_answer}

Retrieved context:
{context_text}

Evaluate the generated answer. Return a JSON object with:
- citation_analysis: step-by-step reasoning that lists the existing context indices and verifies each bracketed citation ([n]) for in-bounds validity and support
- atomic_claims: list of strings, each a discrete factual claim
- total_claims: integer count of claims (must equal len(atomic_claims); 0 if "I don't know")
- supported_claims: integer count of claims backed by in-bounds retrieved context
- answer_relevance_score: integer from 0 to 10 representing how well the answer addresses the question
- unsupported_citations: list of citation strings like "[5]" that are out-of-bounds or not supported by retrieved context

Return only the JSON object.
"""
    last_content = ""
    for attempt in range(2):
        try:
            client = get_cohere_client()
            response = client.chat(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            raw_text = response.message.content[0].text
            content = raw_text.strip()
            if content.startswith("```"):
                lines = content.splitlines()
                if lines and lines[0].strip().startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                content = "\n".join(lines).strip()
            data = json.loads(content)
            if all(
                k in data
                for k in (
                    "citation_analysis",
                    "total_claims",
                    "supported_claims",
                    "answer_relevance_score",
                )
            ):
                total_claims = int(data["total_claims"])
                supported_claims = int(data["supported_claims"])
                answer_relevance_score = int(data["answer_relevance_score"])

                if total_claims > 0:
                    faithfulness_score = supported_claims / total_claims
                else:
                    faithfulness_score = 1.0

                answer_relevance = answer_relevance_score / 10.0

                confidence_scores = {
                    "faithfulness": faithfulness_score,
                    "answer_relevance": answer_relevance,
                }
                raw_judge_response = {
                    "citation_analysis": str(data.get("citation_analysis", "")),
                    "atomic_claims": list(data.get("atomic_claims", [])),
                    "total_claims": total_claims,
                    "supported_claims": supported_claims,
                    "answer_relevance_score": answer_relevance_score,
                    "unsupported_citations": list(data.get("unsupported_citations", [])),
                }
                return confidence_scores, raw_judge_response
        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
            pass
        last_content = content
    raise RuntimeError(f"Judge returned invalid JSON after 2 attempts: {last_content!r}")


app = FastAPI(
    title="Hybrid RAG Pipeline API",
    description="API for questioning documents using structure-aware chunking, Groq generation, and Cohere judging.",
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
                    index=idx,
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
        confidence_scores, raw_judge_response = ask_judge(request.question, parsed["answer"], context_text)
        rerank_scores = [
            chunk["rerank_score"]
            for chunk in retrieved_chunks
            if chunk.get("rerank_score") is not None
        ]
        if rerank_scores:
            max_score = max(rerank_scores)
            confidence_scores["retrieval_relevance"] = 1.0 / (1.0 + math.exp(-max_score / 3.0))
        else:
            confidence_scores["retrieval_relevance"] = 1.0 if retrieved_ids else 0.0

        # 5. Build response
        all_retrieved_chunks: List[RetrievedChunk] = []
        for rank, chunk in enumerate(retrieved_chunks, start=1):
            if chunk.get("rerank_score") is not None:
                display_score = 1.0 / (1.0 + math.exp(-chunk["rerank_score"] / 3.0))
                is_percentage = True
            elif chunk.get("distance") is not None:
                display_score = float(chunk["distance"])
                is_percentage = False
            else:
                display_score = 0.0
                is_percentage = False
            all_retrieved_chunks.append(
                RetrievedChunk(
                    rank=rank,
                    chunk_id=chunk.get("chunk_id"),
                    source_file=chunk.get("source_file"),
                    text=chunk.get("text", ""),
                    display_score=display_score,
                    is_percentage=is_percentage,
                )
            )

        return QueryResponse(
            answer=parsed["answer"],
            model=generation.get("model"),
            citations=citations,
            all_retrieved_chunks=all_retrieved_chunks,
            confidence_scores=confidence_scores,
            raw_judge_response=raw_judge_response,
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


@app.get("/v1/documents", response_model=List[DocumentFile], tags=["Data Management"])
def list_documents():
    """Returns every indexed chunk grouped by source file."""
    try:
        client = get_chroma_client()
        collection = client.get_collection("structure")
        result = collection.get(include=["metadatas", "documents"])

        ids = result.get("ids", []) or []
        texts = result.get("documents", []) or []
        metadatas = result.get("metadatas", []) or []

        grouped_files: Dict[str, List[DocumentChunk]] = {}
        for chunk_id, text, meta in zip(ids, texts, metadatas):
            meta = meta or {}
            source_file = meta.get("source_file", "unknown")
            grouped_files.setdefault(source_file, []).append(
                DocumentChunk(
                    chunk_id=chunk_id or "",
                    chunk_index=meta.get("chunk_index", 0),
                    text=text or "",
                )
            )

        return [
            DocumentFile(
                source_file=source_file,
                total_chunks=len(chunks),
                chunks=chunks,
            )
            for source_file, chunks in grouped_files.items()
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))