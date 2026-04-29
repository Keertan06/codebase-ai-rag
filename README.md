# 🧠 Codebase AI – RAG-based Code Understanding System

A Retrieval-Augmented Generation (RAG) system that analyzes large codebases and answers natural language questions using semantic search, graph relationships, and LLMs.

---

## 🚀 Overview

Understanding large codebases is difficult. This project solves that by combining:

- Semantic search (embeddings + FAISS)
- Code structure awareness (functions, classes, relationships)
- LLM-based explanations (Ollama)
- Robust fallback when LLM fails

---

## ⚙️ Architecture
User Query
↓
Embedding (Sentence Transformers)
↓
Vector Search (FAISS)
↓
Top-K Code Chunks
↓
Context Builder
↓
LLM (Ollama) OR Fallback Engine
↓
Answer

---

## 🔑 Key Features

- 🔍 **Semantic Code Search**  
  Understands meaning, not just keywords

- 🧠 **LLM-powered Answers**  
  Generates explanations using local models (Ollama)

- 🔗 **Graph-based Context**  
  Captures relationships between functions and classes

- ⚠️ **Graceful Fallback**  
  If LLM fails, system still returns useful context

- 🛠️ **Full Debug Logging**  
  Inspect every stage of the pipeline (retrieval, prompt, answer)

---

## 🧪 Example

### Query: What does include_router do in FastAPI?


### Retrieved:
- `fastapi/routing.py`
- `fastapi/applications.py`

### Output:
- Explanation (if LLM succeeds)
- OR relevant code context (fallback)

---

## 📦 Setup

1. Clone repo

git clone https://github.com/Keertan06/codebase-ai-rag.git
cd codebase-ai-rag

2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

## Index a Codebase
python3 main.py index test_repo

This will:
Scan files
Chunk code
Generate embeddings
Build FAISS index

## Ask Questions
python3 main.py ask "How are routes defined?" \
  --llm-provider ollama \
  --llm-model phi

## Without LLM (Debug Mode)
python3 main.py ask "include_router" --no-llm

Returns:
Retrieved chunks only
Useful for debugging retrieval

## Tech Stack
Python
FAISS (vector search)
Sentence Transformers (embeddings)
Ollama (local LLM)
Custom RAG pipeline

## Key Learnings
Retrieval quality matters more than LLM quality
Chunking strategy directly impacts accuracy
Small models require strict context limits
Observability (logging) is critical for debugging RAG systems

## Limitations
Local LLMs may fail on large prompts
Requires aggressive context trimming
Retrieval can be improved further

## Future Improvements
Web UI (Streamlit / React)
Hybrid search (keyword + semantic)
Function-level summarization
Flow tracing (call graph visualization)

