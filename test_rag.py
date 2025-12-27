from rag_pipeline import answer_query

if __name__ == "__main__":
    query = "Explain retrieval augmented generation"
    result = answer_query(query)

    print("ROUTE:", result["route"])
    print("ANSWER:\n", result["answer"])
