import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

# Ensure root package is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    DEFAULT_TOP_K,
)
from ingest import (
    ingest_pdf_bytes,
    ingest_all_data_pdfs,
    get_db_stats,
)
from rag import RAGEngine

app = FastAPI(
    title="SupplyPilot API — AI Supply Chain Intelligence",
    description="REST API backend for SupplyPilot RAG document intelligence engine.",
    version="1.0.0"
)


class AskRequest(BaseModel):
    question: str = Field(..., example="Which supplier had the highest spend in Q1?")
    top_k: Optional[int] = Field(default=DEFAULT_TOP_K, example=5)


class SourceItem(BaseModel):
    file: str
    page: int


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceItem]
    refused: bool
    evidence_count: int


class StatsResponse(BaseModel):
    collection_name: str
    total_chunks: int
    document_count: int
    indexed_documents: List[str]
    embedding_model: str
    llm_model: str


@app.get("/")
def read_root():
    return {
        "app": "SupplyPilot API",
        "status": "online",
        "docs_url": "/docs"
    }


@app.get("/stats", response_model=StatsResponse)
def get_stats():
    """Return Knowledge Base vector store statistics."""
    stats = get_db_stats()
    return StatsResponse(
        collection_name=stats["collection_name"],
        total_chunks=stats["total_chunks"],
        document_count=stats["document_count"],
        indexed_documents=stats["indexed_documents"],
        embedding_model=stats["embedding_model"],
        llm_model=LLM_MODEL
    )


@app.post("/ingest")
async def ingest_documents(files: List[UploadFile] = File(None)):
    """
    Ingest uploaded PDF files or default data directory PDFs into ChromaDB.
    Returns dynamic file and chunk counts.
    """
    total_chunks_added = 0
    results = []
    
    if files and len(files) > 0:
        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                continue
            content = await file.read()
            res = ingest_pdf_bytes(content, file.filename)
            results.append(res)
            if res.get("status") == "success":
                total_chunks_added += res.get("chunks_added", 0)
    else:
        # Fallback to ingest data/ directory
        results = ingest_all_data_pdfs()
        for r in results:
            if r.get("status") == "success":
                total_chunks_added += r.get("chunks_added", 0)

    db_stats = get_db_stats()

    return {
        "files": len(results),
        "chunks": db_stats["total_chunks"],
        "new_chunks_added": total_chunks_added,
        "details": results
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    """
    Query SupplyPilot RAG engine with a question.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    rag_engine = RAGEngine(top_k=req.top_k or DEFAULT_TOP_K)
    res = rag_engine.answer_question(req.question, top_k=req.top_k)

    sources_formatted = [
        SourceItem(file=s["source"], page=s["page"]) for s in res.get("sources", [])
    ]

    return AskResponse(
        query=res["query"],
        answer=res["answer"],
        sources=sources_formatted,
        refused=res.get("refused", False),
        evidence_count=len(res.get("evidence", []))
    )
