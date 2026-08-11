import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pypdf
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    DATA_DIR,
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

load_dotenv()


def compute_file_hash(file_bytes: bytes) -> str:
    """Compute SHA-256 hash of file content to detect duplicates."""
    return hashlib.sha256(file_bytes).hexdigest()


class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    """Custom ChromaDB embedding function supporting Gemini and OpenAI APIs."""
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file.")

        cleaned_input = [text.replace("\n", " ") for text in input]

        # Primary: Google Generative AI embeddings (batch call)
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            res = genai.embed_content(
                model=self.model_name if "gemini" in self.model_name else "models/gemini-embedding-001",
                content=cleaned_input
            )
            return res["embedding"]
        except Exception as genai_err:
            # Fallback: OpenAI API client
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=cleaned_input
                )
                return [data.embedding for data in response.data]
            except Exception as openai_err:
                raise RuntimeError(f"Embedding error: {str(genai_err)} | OpenAI Fallback: {str(openai_err)}")


def get_chroma_client():
    """Get persistent ChromaDB client."""
    return chromadb.PersistentClient(path=str(CHROMA_DB_DIR))


def get_or_create_collection():
    """Get or create single persistent ChromaDB collection."""
    client = get_chroma_client()
    embedding_fn = GeminiEmbeddingFunction(EMBEDDING_MODEL)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"description": "Meridian Components Supply Chain Document Collection"}
    )


def extract_pdf_pages(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """
    Extract text page by page from PDF bytes using pypdf.
    Preserves page numbers (1-indexed) and table structures.
    """
    import io
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    extracted_pages = []
    
    for i, page in enumerate(reader.pages):
        page_num = i + 1
        page_text = page.extract_text() or ""
        if page_text.strip():
            extracted_pages.append({
                "page": page_num,
                "text": page_text,
                "source": filename
            })
            
    return extracted_pages


def ingest_pdf_file(file_path: Path) -> Dict[str, Any]:
    """Ingest a single PDF file from disk into ChromaDB."""
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    return ingest_pdf_bytes(file_bytes, file_path.name)


def ingest_pdf_bytes(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Process PDF bytes, split into chunks, embed, and store in persistent ChromaDB.
    Prevents duplicate indexing using SHA-256 hash.
    """
    doc_hash = compute_file_hash(file_bytes)
    collection = get_or_create_collection()
    
    # Check if document with this hash/filename is already indexed
    existing_results = collection.get(where={"doc_hash": doc_hash})
    if existing_results and len(existing_results.get("ids", [])) > 0:
        return {
            "status": "already_indexed",
            "message": f"Document '{filename}' is already indexed.",
            "filename": filename,
            "chunks_added": 0
        }
    
    # Also check by filename to avoid re-adding updated files with same name
    existing_filename = collection.get(where={"source": filename})
    if existing_filename and len(existing_filename.get("ids", [])) > 0:
        return {
            "status": "already_indexed",
            "message": f"Document '{filename}' is already indexed.",
            "filename": filename,
            "chunks_added": 0
        }

    pages = extract_pdf_pages(file_bytes, filename)
    if not pages:
        return {
            "status": "error",
            "message": f"No text could be extracted from '{filename}'.",
            "filename": filename,
            "chunks_added": 0
        }

    # Split using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    documents = []
    metadatas = []
    ids = []

    total_chunks = 0
    for p in pages:
        page_num = p["page"]
        page_text = p["text"]
        chunks = text_splitter.split_text(page_text)
        
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_hash[:12]}_p{page_num}_c{i}"
            documents.append(chunk_text)
            metadatas.append({
                "source": filename,
                "page": page_num,
                "doc_hash": doc_hash,
                "chunk_id": chunk_id,
                "chunk_index": i
            })
            ids.append(chunk_id)
            total_chunks += 1

    if documents:
        # Add to ChromaDB in batches to prevent payload limits
        batch_size = 50
        for b in range(0, len(documents), batch_size):
            collection.add(
                documents=documents[b:b+batch_size],
                metadatas=metadatas[b:b+batch_size],
                ids=ids[b:b+batch_size]
            )

    return {
        "status": "success",
        "message": f"Successfully indexed '{filename}' ({total_chunks} chunks).",
        "filename": filename,
        "pages": len(pages),
        "chunks_added": total_chunks
    }


def ingest_all_data_pdfs() -> List[Dict[str, Any]]:
    """Ingest all PDF files located in data directory."""
    results = []
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    for pdf_file in pdf_files:
        res = ingest_pdf_file(pdf_file)
        results.append(res)
    return results


def get_db_stats() -> Dict[str, Any]:
    """Retrieve dynamic database statistics from ChromaDB."""
    try:
        collection = get_or_create_collection()
        total_chunks = collection.count()
        
        # Fetch distinct sources
        indexed_docs = set()
        if total_chunks > 0:
            all_meta = collection.get(include=["metadatas"])
            for meta in all_meta.get("metadatas", []):
                if meta and "source" in meta:
                    indexed_docs.add(meta["source"])
                    
        return {
            "total_chunks": total_chunks,
            "document_count": len(indexed_docs),
            "indexed_documents": list(indexed_docs),
            "collection_name": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL
        }
    except Exception as e:
        return {
            "total_chunks": 0,
            "document_count": 0,
            "indexed_documents": [],
            "collection_name": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL,
            "error": str(e)
        }


def clear_knowledge_base():
    """Clear all indexed documents from ChromaDB collection."""
    client = get_chroma_client()
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    # Re-create empty collection
    get_or_create_collection()
    return {"status": "success", "message": "Knowledge base cleared successfully."}


if __name__ == "__main__":
    print("Testing Ingestion Pipeline...")
    res = ingest_all_data_pdfs()
    print("Ingestion Results:", res)
    print("DB Stats:", get_db_stats())
