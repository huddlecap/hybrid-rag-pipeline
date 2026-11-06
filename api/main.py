from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from retriever import search, get_chroma_client, get_embedder
from generator import generate, get_client
from chunker import chunk_structure, get_encoder, DEFAULT_CONFIG
from indexer import tokenize
from rank_bm25 import BM25Okapi
from evaluate_pipeline import ask_judge

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
env_path = ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)

from api.models import QueryRequest, QueryResponse, IngestRequest, Citation

app = FastAPI(
    title="Hybrid RAG Pipeline API",
    description="API for questioning documents using structure-aware chunking and Groq 120B.",
    version="1.0.0",
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
        judge_prompt = f"""Question: {request.question}

Generated answer:
{parsed["answer"]}

Retrieved context:
{context_text}

Evaluate the generated answer. Return a JSON object with:
- correctness: 1 if factually correct based on ground truth knowledge, else 0
- faithfulness: 1 if strictly grounded in retrieved context without hallucination, else 0
- unsupported_citations: list of citation strings like "[2]" that are NOT supported by the retrieved context

Return only the JSON object.
"""
        judge_result = ask_judge(judge_prompt)

        # 5. Build response
        return QueryResponse(
            answer=parsed["answer"],
            citations=citations,
            confidence_scores={
                "correctness": judge_result.get("correctness", 0),
                "faithfulness": judge_result.get("faithfulness", 0),
                "retrieval_relevance": 1.0 if retrieved_ids else 0.0,
                "model": generation.get("model", ""),
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