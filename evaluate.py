
from src.vector_store import process_document
from src.hybrid_retriever import (
    create_bm25,
    tokenize,
    hybrid_search
)


# --------------------------------
# Configuration
# --------------------------------

PDF_PATH = "data/sample.pdf"
K = 5


# --------------------------------
# Test Questions
# --------------------------------
#
# expected_pages contains all pages
# where relevant information can appear.
#
# This is better than forcing every
# question to have exactly one page.
# --------------------------------

test_questions = [

    {
        "question": "What is Generative AI?",
        "expected_pages": [2]
    },

    {
        "question": "What types of content can Generative AI create?",
        "expected_pages": [2]
    },

    {
        "question": "How is Generative AI different from traditional AI?",
        "expected_pages": [3]
    },

    {
        "question": "What are the applications of Generative AI?",
        "expected_pages": [9, 10]
    },

    {
        "question": "How is Generative AI used in customer support?",
        "expected_pages": [9]
    },

    {
        "question": "How is Generative AI used for code generation?",
        "expected_pages": [9]
    },

    {
        "question": "What is the Transformer architecture?",
        "expected_pages": [5]
    },

    {
        "question": "When was the Transformer architecture introduced?",
        "expected_pages": [2]
    },

    {
        "question": "What is self-attention?",
        "expected_pages": [5]
    },

    {
        "question": "How does self-attention work?",
        "expected_pages": [5]
    },

    {
        "question": "What is an LLM?",
        "expected_pages": [4]
    },

    {
        "question": "What is retrieval augmented generation?",
        "expected_pages": [7, 8]
    },

    {
        "question": "Why is retrieval useful for LLMs?",
        "expected_pages": [7, 8]
    },

    {
        "question": "What is fine-tuning?",
        "expected_pages": [7, 8]
    },

    {
        "question": "What is tokenization?",
        "expected_pages": [6]
    },

    {
        "question": "What is a context window?",
        "expected_pages": [7]
    },

    {
        "question": "What are hallucinations in LLMs?",
        "expected_pages": [10, 11]
    }
]


# --------------------------------
# Helper Functions
# --------------------------------

def get_page(document):

    page = document.metadata.get("page")

    if page is not None:
        return page + 1

    return None


def print_results(name, documents):

    print(f"\n{name}")

    for i, document in enumerate(documents, start=1):

        page = get_page(document)

        text = document.page_content.replace(
            "\n",
            " "
        )

        # Make output easier to read
        if len(text) > 180:
            text = text[:180] + "..."

        print(
            f"  {i}. Page {page} | {text}"
        )


def check_hit(retrieved_pages, expected_pages):

    return any(
        page in retrieved_pages
        for page in expected_pages
    )


# --------------------------------
# Evaluation
# --------------------------------

def evaluate():

    print("\n==============================")
    print(" RAG RETRIEVAL EVALUATION")
    print("==============================\n")

    print("Loading document...")

    vector_store, chunks, pages, total_chunks = process_document(
        PDF_PATH
    )

    print(f"Pages: {pages}")
    print(f"Chunks: {total_chunks}\n")

    # Create BM25 index once
    bm25 = create_bm25(chunks)

    semantic_correct = 0
    bm25_correct = 0
    hybrid_correct = 0

    total = len(test_questions)

    for i, test in enumerate(test_questions, start=1):

        question = test["question"]
        expected_pages = test["expected_pages"]

        print("\n--------------------------------")
        print(f"Question {i}: {question}")
        print(f"Expected pages: {expected_pages}")


        # =================================
        # Semantic Search
        # =================================

        semantic_results = vector_store.similarity_search(
            question,
            k=K
        )

        semantic_pages = [
            get_page(document)
            for document in semantic_results
        ]

        semantic_hit = check_hit(
            semantic_pages,
            expected_pages
        )

        if semantic_hit:
            semantic_correct += 1

        print(
            f"\nSemantic pages: {semantic_pages}"
        )

        print(
            f"Semantic: "
            f"{'PASS' if semantic_hit else 'FAIL'}"
        )

        print_results(
            "Semantic retrieved chunks:",
            semantic_results
        )


        # =================================
        # BM25 Search
        # =================================

        bm25_results = bm25.get_top_n(
            tokenize(question),
            chunks,
            n=K
        )

        bm25_pages = [
            get_page(document)
            for document in bm25_results
        ]

        bm25_hit = check_hit(
            bm25_pages,
            expected_pages
        )

        if bm25_hit:
            bm25_correct += 1

        print(
            f"\nBM25 pages: {bm25_pages}"
        )

        print(
            f"BM25: "
            f"{'PASS' if bm25_hit else 'FAIL'}"
        )

        print_results(
            "BM25 retrieved chunks:",
            bm25_results
        )


        # =================================
        # Hybrid Search
        # =================================

        hybrid_results = hybrid_search(
            question,
            vector_store,
            chunks,
            k=K
        )

        hybrid_pages = [
            get_page(document)
            for document in hybrid_results
        ]

        hybrid_hit = check_hit(
            hybrid_pages,
            expected_pages
        )

        if hybrid_hit:
            hybrid_correct += 1

        print(
            f"\nHybrid pages: {hybrid_pages}"
        )

        print(
            f"Hybrid: "
            f"{'PASS' if hybrid_hit else 'FAIL'}"
        )

        print_results(
            "Hybrid retrieved chunks:",
            hybrid_results
        )


    # =================================
    # Final Results
    # =================================

    semantic_recall = semantic_correct / total
    bm25_recall = bm25_correct / total
    hybrid_recall = hybrid_correct / total

    print("\n\n==============================")
    print(" FINAL RESULTS")
    print("==============================")

    print(
        f"Semantic Recall@{K}: "
        f"{semantic_recall:.2%}"
    )

    print(
        f"BM25 Recall@{K}: "
        f"{bm25_recall:.2%}"
    )

    print(
        f"Hybrid Recall@{K}: "
        f"{hybrid_recall:.2%}"
    )

    print("==============================\n")


# --------------------------------
# Run Evaluation
# --------------------------------

if __name__ == "__main__":
    evaluate()

