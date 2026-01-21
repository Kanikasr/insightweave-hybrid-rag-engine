# 🧠 InsightWeave — Hybrid RAG Search Engine

**InsightWeave** is a hybrid Retrieval-Augmented Generation (RAG) system that combines  
**semantic search over private documents** with **real-time web search** to deliver  
grounded, explainable, and up-to-date answers.

The system is designed to be **model-agnostic**, **cost-efficient**, and **enterprise-ready**,  
demonstrating how modern AI copilots blend internal knowledge bases with live external data.

---

## 🚀 Key Features

- 📄 **Multi-Document Semantic Search** (PDF / Text / Wikipedia)
- 🔍 **FAISS Vector Database** for fast similarity search
- 🌐 **Real-Time Web Search** using Tavily
- 🔀 **Hybrid Query Routing** (Document / Web / Hybrid)
- 🧠 **Retrieval-Augmented Generation (RAG)**
- 🧾 **Citation-Aware Answers**
- 🔍 **Transparent Evidence Display**
- 💻 **Interactive Streamlit UI**
- 💰 **No Paid APIs Required** (Local Embeddings + Local LLM)

---

## 🏗️ System Architecture

```
User Query
   │
   ▼
Query Router
(doc / web / hybrid)
   │
   ├──► FAISS Vector Search (Documents)
   │
   ├──► Tavily Web Search (Live Data)
   │
   ▼
Context Assembly
   │
   ▼
Context Sanitization
(remove source tags)
   │
   ▼
Local LLM (Flan-T5)
   │
   ▼
Answer Generation
   │
   ▼
Deterministic Source Attribution
   │
   ▼
Streamlit UI Output
```

---

## 🧠 Why Hybrid RAG?

Traditional LLMs rely solely on parametric memory, which:
- Becomes outdated
- Hallucinates facts
- Cannot access private data

InsightWeave solves this by:
- Retrieving **relevant documents at query time**
- Augmenting the prompt with **grounded context**
- Combining **private knowledge + live web data**
- Explicitly exposing **evidence used**

---

## 🧰 Tech Stack

| Component | Technology |
|--------|-----------|
| Language | Python |
| Orchestration | LangChain |
| Vector Store | FAISS |
| Embeddings | HuggingFace (Sentence Transformers) |
| LLM | Flan-T5 (Local) |
| Web Search | Tavily |
| UI | Streamlit |

---

## 📂 Project Structure

    insightweave_hybrid_rag/
    │
    ├── app.py                # Streamlit UI
    ├── ingestion.py          # Document loading & indexing
    ├── rag_pipeline.py       # Hybrid RAG logic
    ├── build_index.py        # FAISS index builder
    ├── schemas.py            # Data models
    ├── utils.py              # Environment loader
    │
    ├── data/
    │   └── docs/             # Input documents
    │
    ├── faiss_index/          # Vector index (local)
    ├── requirements.txt
    ├── .env                  # API keys (ignored)
    ├── .gitignore
    └── README.md

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository
```bash
git clone https://github.com/Kanikasr/insightweave-hybrid-rag-engine.git
cd insightweave_hybrid_rag
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
``` 
### 3️⃣ Install Dependencies 
```bash
pip install -r requirements.txt
```
### 4️⃣ Add Environment Variables
```bash
TAVILY_API_KEY=your_tavily_key_here
```
## 📥 Build Document Index

Add documents to:

    data/docs/

Then run:

    python build_index.py

This creates a local FAISS vector index.

---

## 🖥️ Run the Application

    streamlit run app.py

Open browser at:

    http://localhost:8501

---

## 🧪 Example Queries

### Document-based

    Explain retrieval augmented generation

### Web-based

    Latest developments in generative AI

### Hybrid

    How does RAG compare with current AI tools?

---

## 🧠 Design Decisions & Tradeoffs

### Why FAISS?

- Fast local vector search  
- No external service dependency  
- Production-proven  

### Why Local Embeddings & LLM?

- Zero API cost  
- Offline capability  
- Easy deployment  
- Architecture remains model-agnostic  

### Why Separate Evidence?

- Prevents hallucinations  
- Improves trust  
- Aligns with enterprise explainability standards  

---

## 🔮 Future Improvements

- Replace local LLM with GPT-4 / Claude  
- Add reranking (Cross-Encoders)  
- Persistent document upload in UI  
- User feedback loop  
- Streaming responses

