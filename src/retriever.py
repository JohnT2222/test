import os
from pathlib import Path
import argparse
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# ------------------------------
# 🔧 Embedding + Retriever Setup
# ------------------------------
def get_retriever(persist_path: Path, k: int = 3):
    load_dotenv()
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=str(persist_path), embedding_function=embedding)
    return vectordb.as_retriever(search_kwargs={"k": k})

# ------------------------------
# 🔍 Query Execution
# ------------------------------
def retrieve_similar_documents(query: str, persist_path: Path, k: int = 3):
    retriever = get_retriever(persist_path, k)
    results = retriever.get_relevant_documents(query)
    return results

# ------------------------------
# 🧪 CLI Entry Point
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Chroma vector DB")
    parser.add_argument("--query", type=str, required=True, help="Your query string")
    parser.add_argument("--k", type=int, default=3, help="Number of top results to return")
    args = parser.parse_args()

    print(f"🔎 Query: {args.query}")
    docs = retrieve_similar_documents(args.query, persist_path=Path("chroma_db"), k=args.k)
    print(f"🔢 Top {args.k} results:\n" + "-" * 40)

    for i, doc in enumerate(docs, 1):
        print(f"[{i}]")
        print(f"📄 Source: {doc.metadata.get('source', 'unknown')}")
        print(f"📚 Content: {doc.page_content.strip()[:300]}...\n")
