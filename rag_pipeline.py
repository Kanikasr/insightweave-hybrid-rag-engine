
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from utils import load_environment
from langchain_huggingface import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

INDEX_DIR = "faiss_index"


# --------- Query Classification ---------
import re

def strip_source_tags(text: str) -> str:
    """
    Remove [Doc X], [Web X], and wiki-style headers from context
    before sending to the LLM.
    """
    text = re.sub(r"\[(Doc|Web)\s*\d+\]", "", text)
    text = re.sub(r"==.*?==", "", text)
    return text

def classify_query(query: str) -> str:
    query_lower = query.lower()

    web_keywords = ["latest", "current", "recent", "today", "news", "now"]
    if any(word in query_lower for word in web_keywords):
        return "web"

    if "compare" in query_lower or "vs" in query_lower:
        return "hybrid"

    return "doc"


# --------- Load FAISS Index ---------

def load_faiss_index():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)


# --------- Document Retrieval ---------

def retrieve_documents(query: str, k: int = 4):
    db = load_faiss_index()
    return db.similarity_search(query, k=k)


# --------- Web Retrieval ---------

def retrieve_web(query: str):
    load_environment()
    tavily = TavilySearchResults(k=5)
    return tavily.run(query)


# --------- Context Assembly ---------

def assemble_context(query: str, route: str, use_web: bool = True):
    context_parts = []

    if route in ["doc", "hybrid"]:
        docs = retrieve_documents(query, k=2)   # LIMIT TO 3
        for i, doc in enumerate(docs):
            context_parts.append(f"[Doc {i+1}] {doc.page_content[:800]}")

    if use_web and route in ["web", "hybrid"]:
        web_results = retrieve_web(query)[:3]    # LIMIT TO 3
        for i, result in enumerate(web_results):
            context_parts.append(f"[Web {i+1}] {result['content'][:600]}")

    return "\n\n".join(context_parts)


# --------- Answer Generation ---------
def load_local_llm():
    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0
    )

    return HuggingFacePipeline(pipeline=pipe)

def answer_query(query: str, use_web: bool = True):

    route = classify_query(query)
    raw_context = assemble_context(query, route, use_web)
    context = strip_source_tags(raw_context)
    llm = load_local_llm()

    prompt = f"""
Summarize the answer in 4–5 concise sentences.

Rules:
- Do NOT include document labels like [Doc 1], [Doc 2].
- Do NOT repeat the same idea.
- Write in clean natural language.

Context:
{context}

Question:
{query}
"""

    raw_answer = llm.invoke(prompt).strip()

    # ----- Deterministic citation handling -----
    citations = []
    if route in ["doc", "hybrid"]:
        citations.append("📄 Documents")
    if route in ["web", "hybrid"] and use_web:
        citations.append("🌐 Web")

    citation_text = " | ".join(citations)

    final_answer = f"{raw_answer}\n\n**Sources:** {citation_text}"

    return {
        "answer": final_answer,
        "route": route
    }

