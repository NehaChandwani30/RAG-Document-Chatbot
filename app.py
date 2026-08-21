import tempfile

import streamlit as st

from src.vector_store import process_document
from src.rag_pipeline import answer_question


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="📚",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------

st.title("📚 RAG Document Chatbot")

st.write(
    "Upload a PDF and ask questions about its content "
    "using Retrieval-Augmented Generation."
)


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.header("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )


# -----------------------------
# Process Uploaded PDF
# -----------------------------

if uploaded_file:

    st.success(f"Uploaded: {uploaded_file.name}")

    if (
        "processed_file" not in st.session_state
        or st.session_state.processed_file != uploaded_file.name
    ):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(uploaded_file.getvalue())
            temp_pdf_path = temp_file.name

        with st.spinner("Processing document..."):

            vector_store, document_chunks, pages, chunks = process_document(
                temp_pdf_path
            )

            st.session_state.vector_store = vector_store
            st.session_state.document_chunks = document_chunks
            st.session_state.processed_file = uploaded_file.name
            st.session_state.pages = pages
            st.session_state.chunks = chunks
            st.session_state.chat_history = []

        st.success(
            f"Document processed successfully! "
            f"{pages} pages → {chunks} chunks"
        )

    # -----------------------------
    # Initialize Chat History
    # -----------------------------

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


    # -----------------------------
    # Display Previous Conversation
    # -----------------------------

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.write(message["content"])

    # -----------------------------
    # Ask Question
    # -----------------------------

    question = st.chat_input(
        "Ask a question about your document..."
    )


    if question:

        # -----------------------------
        # Display User Message
        # -----------------------------

        with st.chat_message("user"):
            st.write(question)

        # -----------------------------
        # Generate Answer
        # -----------------------------

        with st.chat_message("assistant"):

            with st.spinner("Searching document..."):

                answer, documents, context, rewritten_question = answer_question(
                    question,
                    st.session_state.vector_store,
                    st.session_state.document_chunks,
                    st.session_state.chat_history
                )

            st.write(answer)

            # -----------------------------
            # Save Conversation
            # -----------------------------

            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })

            # -----------------------------
            # Debug Information
            # -----------------------------

            with st.expander("🔍 Debug: Retrieved Context"):

                st.write(
                    f"**Rewritten Question:** "
                    f"{rewritten_question}"
                )

                st.code(context)

            # -----------------------------
            # Sources
            # -----------------------------

            st.markdown("### 📚 Sources")

            shown_sources = set()

            for document in documents:

                page = document.metadata.get("page")

                if page is not None:
                    page_number = page + 1
                else:
                    page_number = "Unknown"

                if page_number not in shown_sources:

                    st.write(
                        f"📄 {uploaded_file.name} — "
                        f"Page {page_number}"
                    )

                    shown_sources.add(page_number)

else:

    st.info(
        "👈 Upload a PDF from the sidebar to get started."
    )