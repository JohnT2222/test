import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from retriever import retrieve_similar_documents
from langchain_core.documents import Document

# -----------------------------
# 🔐 Groq API Client Init
# -----------------------------

# ✅ Option 1: HARD-CODED KEY (for testing only — remove after!)
# from groq import Groq
# api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # ⬅️ Paste your new key here
# client = Groq(api_key=api_key)

# ✅ Option 2: Load from .env (preferred in production)
from groq import Groq
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)


# ------------------------------
# 🤖 Answer Generation with Groq
# ------------------------------
def answer_query_with_groq(query: str, context_docs: list[Document], model: str = "mixtral-8x7b-32768"):
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not found or invalid")

    context = "\n\n".join([doc.page_content for doc in context_docs])
    prompt = f"""You are a helpful assistant. Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=512
    )

    return response.choices[0].message.content.strip()


# ------------------------------
# 🧪 CLI Entry Point
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG pipeline using Groq")
    parser.add_argument("--query", type=str, required=True, help="Query to answer")
    parser.add_argument("--k", type=int, default=3, help="Number of context chunks to retrieve")
    parser.add_argument("--model", type=str, default="mixtral-8x7b-32768", help="Groq model to use")

    args = parser.parse_args()

    print(f"🔎 Query: {args.query}")
    context_docs = retrieve_similar_documents(args.query, persist_path=Path("chroma_db"), k=args.k)
    print(f"📚 Retrieved {len(context_docs)} documents from Chroma")

    print("🤖 Generating answer from Groq...")
    answer = answer_query_with_groq(args.query, context_docs, model=args.model)
    print("\n💬 Answer:\n" + "-" * 50)
    print(answer)

