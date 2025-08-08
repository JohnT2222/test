# file: C:\Users\johnn\Projects\AGENTICS AGENCY\AA_agentics_pipeline\rag_answer.py
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document

from backend.retriever import retrieve_similar_documents
from llm_client import LLMClient

def answer_with_llm(query: str, docs: list[Document], provider: str, model: str) -> str:
    llm = LLMClient(provider=provider, model=model)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = (
        "You are a concise assistant. Use ONLY the context below to answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    return llm.chat(prompt)

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="RAG answer using Chroma + LLMClient")
    parser.add_argument("--query", required=True, help="User query")
    parser.add_argument("--k", type=int, default=3, help="Top-k docs to retrieve")
    parser.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "groq"))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "qwen/qwen3-32b"))
    parser.add_argument("--db", default="chroma_db", help="Chroma persist directory")
    args = parser.parse_args()

    print(f"🔎 Query: {args.query}")
    docs = retrieve_similar_documents(args.query, persist_path=Path(args.db), k=args.k)
    print(f"📚 Retrieved {len(docs)} docs from {args.db}")

    print(f"🤖 LLM: {args.provider}:{args.model}")
    answer = answer_with_llm(args.query, docs, provider=args.provider, model=args.model)

    print("\n💬 Answer\n" + "-"*50)
    print(answer)
