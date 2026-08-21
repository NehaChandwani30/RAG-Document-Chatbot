from langchain_ollama import ChatOllama
from .hybrid_retriever import hybrid_search


def answer_question(question, vector_store, chunks, chat_history=None):

    # -----------------------------
    # Conversation History
    # -----------------------------

    if chat_history is None:
        chat_history = []

    history_text = ""

    for message in chat_history:
        history_text += (
            f"{message['role'].upper()}: "
            f"{message['content']}\n"
        )

    # -----------------------------
    # Rewrite question using history
    # -----------------------------

    llm = ChatOllama(
        model="llama3.2:latest",
        temperature=0
    )

    rewrite_prompt = f"""
You are a question rewriting assistant.

Use the conversation history to understand the user's
latest question.

If the latest question depends on previous conversation,
rewrite it into a standalone question.

If it is already standalone, return it unchanged.

Do not answer the question.

CONVERSATION HISTORY:
{history_text}

LATEST QUESTION:
{question}

STANDALONE QUESTION:
"""

    rewritten_question = llm.invoke(rewrite_prompt).content.strip()

    # -----------------------------
    # Hybrid Retrieval
    # -----------------------------

    documents = hybrid_search(
        rewritten_question,
        vector_store,
        chunks,
        k=5
    )

    # -----------------------------
    # Build Context
    # -----------------------------

    context = "\n\n".join(
        f"--- DOCUMENT CHUNK {i + 1} ---\n"
        f"{document.page_content}"
        for i, document in enumerate(documents)
    )

    # -----------------------------
    # Generate Answer
    # -----------------------------

    answer_prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the information
contained in the document context below.

Use the conversation history only to understand
references such as "it", "they", "this", or "its".

Do not add information that is not supported by
the document.

If the document does not contain the answer, say:
"I couldn't find the answer in the provided document."

CONVERSATION HISTORY:
{history_text}

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Give a clear and complete answer.

ANSWER:
"""

    response = llm.invoke(answer_prompt)

    return response.content, documents, context, rewritten_question