from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    return documents


if __name__ == "__main__":
    pdf_path = "data/sample.pdf"

    documents = load_pdf(pdf_path)

    print(f"Number of pages: {len(documents)}")

    for document in documents[:2]:
        print("\n--- PAGE ---")
        print(document.page_content[:1000])