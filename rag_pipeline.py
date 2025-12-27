
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from utils import load_environment
from langchain_huggingface import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

INDEX_DIR = "faiss_index"


# --------- Query Classification ---------

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

def assemble_context(query: str, route: str):
    context = ""

    if route in ["doc", "hybrid"]:
        docs = retrieve_documents(query)
        for i, doc in enumerate(docs):
            context += f"[Doc {i+1}] {doc.page_content}\n\n"

    if route in ["web", "hybrid"]:
        web_results = retrieve_web(query)
        for i, result in enumerate(web_results):
            context += f"[Web {i+1}] {result['content']}\n\n"

    return context


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

def answer_query(query: str):
    route = classify_query(query)
    context = assemble_context(query, route)

    llm = load_local_llm()

    prompt = f"""
Answer the question using ONLY the context below.
Cite sources using [Doc X] and [Web Y].

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    return {
        "answer": response,
        "route": route
    }

