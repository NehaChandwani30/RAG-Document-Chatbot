from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from .document_loader import load_pdf
from .text_splitter import split_documents


CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "rag_documents"


def get_embedding_model():
    return OllamaEmbeddings(
        model="nomic-embed-text"
    )


def create_vector_store(chunks):
    embeddings = get_embedding_model()

    # Create a fresh vector store
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    # Remove old documents
    existing_data = vector_store.get()

    if existing_data["ids"]:
        vector_store.delete(
            ids=existing_data["ids"]
        )

    # Add new document chunks
    vector_store.add_documents(chunks)

    return vector_store


def process_document(pdf_path):

    documents = load_pdf(pdf_path)

    chunks = split_documents(documents)

    vector_store = create_vector_store(chunks)

    return vector_store, chunks, len(documents), len(chunks)


if __name__ == "__main__":

    pdf_path = "data/sample.pdf"

    vector_store, document_chunks, pages, chunks = process_document(pdf_path)

    print(f"Loaded {pages} pages")
    print(f"Created {chunks} chunks")
    print("Vector store created successfully!")