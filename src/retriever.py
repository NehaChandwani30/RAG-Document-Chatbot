from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "rag_documents"


def get_retriever():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    return retriever


if __name__ == "__main__":
    retriever = get_retriever()

    question = "What is Generative AI?"

    results = retriever.invoke(question)

    print(f"Question: {question}")
    print(f"Retrieved chunks: {len(results)}")

    for i, document in enumerate(results):
        print(f"\n--- RESULT {i + 1} ---")
        print(document.page_content[:1000])
        print("\nPage:", document.metadata.get("page"))
        print("Source:", document.metadata.get("source"))