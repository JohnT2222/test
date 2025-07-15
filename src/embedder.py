import os
from pathlib import Path
from PyPDF2 import PdfReader

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document

CHUNK_SIZE = 512

def load_text(file_path):
    ext = os.path.splitext(file_path)[1]
    if ext == ".pdf":
        reader = PdfReader(file_path)
        return "".join(page.extract_text() or "" for page in reader.pages)
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def chunk_text(text, size=CHUNK_SIZE):
    return [text[i:i+size] for i in range(0, len(text), size)]

def run_embedding_pipeline(data_path: Path, persist_path: Path):
    all_chunks = []
    for filename in os.listdir(data_path):
        path = os.path.join(data_path, filename)
        print(f"📄 Loading: {filename}")
        try:
            raw_text = load_text(path)
            chunks = chunk_text(raw_text)
            docs = [Document(page_content=chunk, metadata={"source": filename}) for chunk in chunks]
            all_chunks.extend(docs)
            print(f"✅ {len(chunks)} chunks from {filename}")
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")

    print(f"📦 Total chunks: {len(all_chunks)}")

    embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(all_chunks, embedding=embedder, persist_directory=str(persist_path))
    vectorstore.persist()
    print("✅ Embeddings saved to", persist_path)

if __name__ == "__main__":
    print("✅ Starting embedder pipeline...")
    try:
        run_embedding_pipeline(
            data_path=Path("data"),
            persist_path=Path("chroma_db")
        )
    except Exception as e:
        print("❌ Exception occurred:", e)


