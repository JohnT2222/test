import streamlit as st
import requests
import json
from datetime import datetime

st.set_page_config(page_title="RAG MVP", layout="wide")

st.title("🔍 RAG vs. Base Model Comparison")

query = st.text_input("Enter a query:", placeholder="e.g. What is vector search in AI?")

if st.button("Run Query") and query:
    with st.spinner("Querying models..."):
        response = requests.post("http://localhost:8000/rag", json={"query": query})
        data = response.json()

        st.subheader("📥 Retrieved Documents")
        for doc in data["retrieved_docs"]:
            st.markdown(f"- {doc}")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🧠 Base Model Output")
            st.write(data["base_output"])
        with col2:
            st.subheader("🔗 RAG Model Output")
            st.write(data["rag_output"])

        st.subheader("📊 Provide Feedback")
        relevance = st.slider("Relevance of RAG output (0 = poor, 10 = excellent)", 0, 10, 5)
        clarity = st.slider("Clarity of response", 0, 10, 5)
        factual = st.slider("Factual accuracy", 0, 10, 5)

        if st.button("Submit Feedback"):
            feedback = {
                "timestamp": datetime.utcnow().isoformat(),
                "query": data["query"],
                "retrieved_docs": data["retrieved_docs"],
                "rag_output": data["rag_output"],
                "base_output": data["base_output"],
                "metrics": {
                    "relevance": relevance,
                    "clarity": clarity,
                    "factual": factual
                }
            }
            with open("storage/results.json", "a") as f:
                f.write(json.dumps(feedback) + "\n")
            st.success("✅ Feedback logged!")
 
