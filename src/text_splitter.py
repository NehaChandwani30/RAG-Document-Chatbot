from langchain_text_splitters import RecursiveCharacterTextSplitter
from .document_loader import load_pdf


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":
    pdf_path = "data/sample.pdf"

    documents = load_pdf(pdf_path)
    chunks = split_documents(documents)

    print(f"Number of pages: {len(documents)}")
    print(f"Number of chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- CHUNK {i + 1} ---")
        print(chunk.page_content[:500])
        print("Metadata:", chunk.metadata)