import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings


from langchain_community.vectorstores import FAISS

from utils import load_environment

DATA_DIR = "data/docs"
INDEX_DIR = "faiss_index"


def load_documents():
    documents = []

    # Load local documents
    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            documents.extend(loader.load())

        elif file.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            documents.extend(loader.load())

    # Load Wikipedia documents
    wiki_loader = WikipediaLoader(
        query="Retrieval Augmented Generation",
        load_max_docs=2
    )
    documents.extend(wiki_loader.load())

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)


def build_faiss_index(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)

    return vectorstore
