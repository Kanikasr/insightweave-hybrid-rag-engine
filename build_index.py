from ingestion import load_documents, split_documents, build_faiss_index

if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    build_faiss_index(chunks)
    print("FAISS index built successfully")
