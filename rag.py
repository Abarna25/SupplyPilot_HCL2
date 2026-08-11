import os
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

from config import (
    SYSTEM_PROMPT,
    LLM_MODEL,
    LLM_TEMPERATURE,
    DEFAULT_TOP_K,
    REFUSAL_MESSAGE,
)
from ingest import get_or_create_collection

load_dotenv()


def generate_llm_completion(user_message: str, system_prompt: str) -> str:
    """Generate completion using Gemini API with OpenAI fallback."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file.")

    # Primary: Google Generative AI Gemini API
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-flash-latest" if "gemini" in LLM_MODEL else LLM_MODEL,
            system_instruction=system_prompt
        )
        config = genai.types.GenerationConfig(
            temperature=LLM_TEMPERATURE,
            max_output_tokens=1000
        )
        resp = model.generate_content(user_message, generation_config=config)
        return resp.text.strip()
    except Exception as genai_err:
        # Fallback: OpenAI API
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=LLM_TEMPERATURE,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as openai_err:
            raise RuntimeError(f"Completion error: {str(genai_err)} | OpenAI Fallback: {str(openai_err)}")


class RAGEngine:
    """Core RAG retrieval and grounded answer generation engine."""

    def __init__(self, top_k: int = DEFAULT_TOP_K):
        self.top_k = top_k
        self.collection = get_or_create_collection()

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve top-K relevant chunks from ChromaDB.
        Returns list of chunks with text, metadata (source, page), and similarity distance.
        """
        k = top_k if top_k is not None else self.top_k
        
        # Check if collection is empty
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved_chunks = []
        if results and results.get("documents") and len(results["documents"]) > 0:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(docs)

            for doc_text, meta, dist in zip(docs, metas, distances):
                retrieved_chunks.append({
                    "text": doc_text,
                    "source": meta.get("source", "Unknown Document"),
                    "page": meta.get("page", 1),
                    "chunk_id": meta.get("chunk_id", ""),
                    "distance": round(float(dist), 4)
                })

        return retrieved_chunks

    def answer_question(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Full RAG flow: Retrieve chunks -> Construct prompt -> Call Gemini/GPT-4o -> Format Answer & Citations.
        """
        retrieved_chunks = self.retrieve(query, top_k=top_k)

        # Handle empty collection case
        if not retrieved_chunks:
            return {
                "query": query,
                "answer": "No documents have been indexed yet. Please upload and index documents into the knowledge base first.",
                "sources": [],
                "evidence": [],
                "refused": True
            }

        # Build context from retrieved chunks
        context_blocks = []
        unique_sources_map = {} # (source, page) -> True

        for idx, chunk in enumerate(retrieved_chunks, 1):
            source = chunk["source"]
            page = chunk["page"]
            text = chunk["text"]
            unique_sources_map[(source, page)] = True
            
            context_blocks.append(
                f"[DOCUMENT CHUNK {idx}]\n"
                f"Source File: {source}\n"
                f"PDF Page Number: Page {page}\n"
                f"Content:\n{text}\n"
            )

        formatted_context = "\n---\n".join(context_blocks)

        user_message = f"""USER QUESTION: {query}

RELEVANT DOCUMENT CONTEXT:
{formatted_context}

INSTRUCTIONS:
Answer the question strictly using ONLY the relevant document context provided above.
Include clear citations referencing the source document and page number for every key fact.
If the information is not supported by the document context, state exactly: "{REFUSAL_MESSAGE}"."""

        try:
            raw_answer = generate_llm_completion(user_message, SYSTEM_PROMPT)

            # Format source citations list
            sources_list = [
                {"source": src, "page": pg} for (src, pg) in sorted(unique_sources_map.keys())
            ]

            is_refusal = REFUSAL_MESSAGE.lower() in raw_answer.lower()

            return {
                "query": query,
                "answer": raw_answer,
                "sources": sources_list,
                "evidence": retrieved_chunks,
                "refused": is_refusal
            }

        except Exception as e:
            return {
                "query": query,
                "answer": f"An error occurred while generating answer: {str(e)}",
                "sources": [],
                "evidence": retrieved_chunks,
                "refused": True,
                "error": str(e)
            }


if __name__ == "__main__":
    print("Testing RAG Engine with Gemini API Key...")
    rag = RAGEngine()
    res = rag.answer_question("Which supplier had the highest spend in Q1?")
    print("Answer:", res["answer"])
    print("Sources:", res["sources"])
