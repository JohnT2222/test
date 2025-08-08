from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from pathlib import Path

# 🔌 Import your modules
from backend.retriever import retrieve_similar_documents
from backend.rag_cli import answer_query_with_groq


load_dotenv()

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class RAGRequest(BaseModel):
    query: str

# Configuration
PERSIST_PATH = Path("C:/Users/johnn/Desktop/groq_compare/chroma_db")  # or just Path("chroma_db") if relative
TOP_K = 3
MODEL_NAME = "mistral-saba-24b"  # or whatever default Groq model you want

@app.post("/rag")
async def handle_rag(request: RAGRequest):
    query = request.query

    # 🔍 Retrieve relevant docs
    retrieved_docs = retrieve_similar_documents(query, persist_path=PERSIST_PATH, k=TOP_K)

    # 🧠 Generate RAG response using Groq
    rag_output = answer_query_with_groq(query, retrieved_docs, model=MODEL_NAME)

    # (Optional) You could later add a base model response for comparison

    return {
        "query": query,
        "retrieved_docs": [doc.page_content for doc in retrieved_docs],
        "rag_output": rag_output,
        "base_output": "⚠️ Base model response not configured (Groq-only mode)"
    }

