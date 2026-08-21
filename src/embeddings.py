from langchain_ollama import OllamaEmbeddings


def get_embedding_model():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    return embeddings


if __name__ == "__main__":
    embeddings = get_embedding_model()

    text = "Generative AI can create new content."

    vector = embeddings.embed_query(text)

    print("Embedding created successfully!")
    print("Vector dimensions:", len(vector))
    print("First 10 values:", vector[:10])