# 🤖 RAG Document Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot that allows users to ask questions about their own PDF documents and receive **context-aware answers**.

The system combines 🔎 **Semantic Search** and 🔤 **BM25 keyword search** using a hybrid retrieval approach. Relevant document chunks are retrieved and provided as context to a local **Large Language Model (LLM)**, which generates the final response.

---

## ✨ Features

* 📄 PDF document processing
* ✂️ Text extraction and chunking
* 🧠 Vector embeddings
* 🗄️ ChromaDB vector database
* 🔎 Semantic similarity search
* 🔤 BM25 keyword-based retrieval
* 🔀 Hybrid retrieval
* 📊 Semantic and BM25 score normalization
* 🏆 Weighted hybrid ranking
* 🎯 Top-K document retrieval
* 🦙 Local LLM inference using Ollama
* 🧠 Llama 3.2 integration
* 💬 Conversational memory
* 🔄 Context-aware follow-up questions
* 🌐 Streamlit chatbot interface
* 📑 Source/page-aware retrieval
* 📈 Retrieval evaluation using Recall@5

---

## 🛠️ Technology Stack

| Component               | Technology            |
| ----------------------- | --------------------- |
| 💻 Programming Language | Python                |
| 🧠 LLM                  | Llama 3.2             |
| ⚙️ LLM Runtime          | Ollama                |
| 🔗 RAG Framework        | LangChain             |
| 🗄️ Vector Database     | ChromaDB              |
| 🧬 Embeddings           | Sentence Transformers |
| 🔤 Keyword Retrieval    | BM25                  |
| 📄 PDF Processing       | PyPDF                 |
| 🌐 Frontend             | Streamlit             |
| 📦 Version Control      | Git & GitHub          |

---

## 🔄 How It Works

### 1️⃣ Document Loading

The PDF document is loaded using **PyPDFLoader**. Each page is converted into a LangChain `Document` object containing the page content and metadata such as the page number.

### 2️⃣ Text Chunking

The extracted document text is divided into smaller chunks so that relevant sections can be retrieved efficiently.

### 3️⃣ Embedding Generation

Each document chunk is converted into a numerical vector using an embedding model. These vectors represent the semantic meaning of the document content.

### 4️⃣ Vector Storage

The generated embeddings are stored in **ChromaDB**, allowing the system to efficiently retrieve semantically similar document chunks.

### 5️⃣ Semantic Search 🔎

The user's question is converted into an embedding and compared with the stored document embeddings.

Semantic search helps retrieve relevant information even when the exact words used in the question do not appear in the document.

### 6️⃣ BM25 Retrieval 🔤

BM25 performs keyword-based retrieval and is particularly useful when important terms from the user's question appear directly in the document.

### 7️⃣ Hybrid Retrieval 🔀

The system combines semantic search and BM25 retrieval.

The scores from both methods are normalized and combined using weighted scoring:

```text
Hybrid Score =
    0.6 × Semantic Score
    +
    0.4 × BM25 Score
```

This allows the system to benefit from both **semantic understanding** and **exact keyword matching**.

### 8️⃣ Top-K Retrieval 🎯

The retrieved chunks are ranked using the combined hybrid score.

The current system retrieves:

```text
K = 5
```

relevant chunks.

### 9️⃣ Context Construction 📚

The retrieved document chunks are provided to the LLM as context.

### 🔟 Answer Generation 🤖

**Llama 3.2** generates an answer based on the retrieved document context.

### 1️⃣1️⃣ Conversational Memory 💬

The chatbot maintains conversation history so that follow-up questions can be understood using previous context.

Example:

```text
👤 User: What is Generative AI?

🤖 Bot: Generative AI is ...

👤 User: What are its applications?

🤖 Bot: Its applications include ...
```

The second question can be understood using the context of the previous conversation.

---

## 📈 Retrieval Evaluation

The retrieval system was evaluated using a set of document-based questions.

Three retrieval approaches were compared:

* 🔎 Semantic Search
* 🔤 BM25
* 🔀 Hybrid Retrieval

### 🏆 Results

| Retrieval Method        |    Recall@5 |
| ----------------------- | ----------: |
| 🔎 Semantic Search      |      94.12% |
| 🔤 BM25                 |      82.35% |
| 🏆 **Hybrid Retrieval** | **100.00%** |

### 📊 Result Analysis

**Semantic Search** achieved **94.12% Recall@5**, demonstrating strong performance in identifying relevant document chunks based on semantic similarity.

**BM25** achieved **82.35% Recall@5**, showing the usefulness of keyword-based retrieval while being less effective for questions requiring broader semantic understanding.

The **Hybrid Retriever** achieved **100.00% Recall@5** on the current evaluation dataset, outperforming both individual retrieval methods.

---

## 🤔 Why Hybrid Retrieval?

Semantic search and BM25 have different strengths.

### 🧠 Semantic Search

Effective when the **meaning** of the question is similar to the meaning of the document, even when the exact words differ.

### 🔤 BM25

Effective when important **keywords** from the question appear directly in the document.

### 🔀 Combined Approach

```text
🧠 Semantic Understanding
          +
🔤 Keyword Matching
          =
🏆 Hybrid Retrieval
```

The evaluation results show that the hybrid approach achieved **100.00% Recall@5** on the current test dataset.

---

## 📐 Evaluation Method

For each evaluation question, the system retrieves the **top 5 document chunks**.

A question is considered successfully retrieved when at least one relevant page appears within the top 5 results.

The evaluation uses **Recall@5**:

```text
Recall@5 =
Number of questions with a relevant result in Top 5
---------------------------------------------------
Total number of evaluation questions
```

### 📌 Current Results

```text
🔎 Semantic Recall@5: 94.12%
🔤 BM25 Recall@5:     82.35%
🏆 Hybrid Recall@5:  100.00%
```

---

## 📁 Project Structure

```text
RAG-Document-Chatbot/
│
├── 📂 src/
│   ├── __init__.py
│   ├── document_loader.py
│   ├── embeddings.py
│   ├── hybrid_retriever.py
│   ├── rag_pipeline.py
│   ├── retriever.py
│   ├── text_splitter.py
│   └── vector_store.py
│
├── app.py
├── evaluate.py
├── requirements.txt
├── README.md
└── .gitignore
```

### 🔧 Main Components

**`document_loader.py`**
📄 Loads PDF documents using `PyPDFLoader` and converts pages into LangChain `Document` objects.

**`text_splitter.py`**
✂️ Splits extracted document text into smaller chunks.

**`embeddings.py`**
🧠 Handles generation of vector embeddings.

**`vector_store.py`**
🗄️ Creates and manages the ChromaDB vector store.

**`retriever.py`**
🔎 Handles semantic retrieval from the vector database.

**`hybrid_retriever.py`**
🔀 Combines semantic search and BM25 retrieval using normalized weighted scores.

**`rag_pipeline.py`**
🧩 Connects document retrieval, context construction, LLM generation, and conversational processing.

**`app.py`**
🌐 Provides the Streamlit chatbot interface.

**`evaluate.py`**
📈 Evaluates Semantic Search, BM25, and Hybrid Retrieval using Recall@5.

---

## ⚙️ Setup

### 📋 Prerequisites

Make sure the following are installed:

* 🐍 Python 3.10+
* 🦙 Ollama
* 🔧 Git

### 🐍 Create Virtual Environment

Windows:

```powershell
python -m venv rag_env
```

Activate it:

```powershell
rag_env\Scripts\activate
```

### 📦 Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## 🦙 Ollama Configuration

The project uses **Ollama** for local LLM inference.

Make sure Ollama is installed and running.

Pull the required model:

```powershell
ollama pull llama3.2
```

Verify the model:

```powershell
ollama list
```

The `llama3.2` model should be available before running the chatbot.

---

## 📄 Adding a PDF

Place the PDF document you want to query inside the:

```text
data/
```

directory.

Example:

```text
data/
└── your_document.pdf
```

PDF documents are excluded from the Git repository to avoid uploading potentially large or private files.

---

## 🚀 Running the Chatbot

Start the Streamlit application:

```powershell
streamlit run app.py
```

The application will provide a local URL, usually:

```text
http://localhost:8501
```

Open the URL in your browser to interact with the chatbot.

---

## 🧪 Running Retrieval Evaluation

To evaluate the retrieval system:

```powershell
python evaluate.py
```

The evaluation compares:

```text
🔎 Semantic Search
🔤 BM25
🔀 Hybrid Retrieval
```

using **Recall@5**.

---

## 💡 Example Questions

The chatbot can answer questions based on the content of the uploaded document.

```text
❓ What is Generative AI?

❓ What types of content can Generative AI create?

❓ How is Generative AI different from traditional AI?

❓ What are the applications of Generative AI?

❓ What is the Transformer architecture?

❓ What is self-attention?

❓ What is an LLM?

❓ What is retrieval augmented generation?
```

💬 Follow-up questions can also be asked using conversational memory.

---

## ⚠️ Important Notes

### 🦙 Local LLM

The project uses **Ollama** for local LLM inference. The required Llama model must be installed locally before running the chatbot.

### 🗄️ ChromaDB

The ChromaDB vector store is generated locally and is excluded from the Git repository.

### 🐍 Virtual Environment

The Python virtual environment is excluded from the Git repository using `.gitignore`.

### 📄 PDF Documents

Local PDF documents are excluded from the repository to avoid uploading potentially large or private files.

---

## 🔮 Future Improvements

* 🔄 Reranking retrieved chunks
* 📈 Recall@1 and MRR evaluation
* 📝 Answer-level evaluation
* 🛡️ Hallucination detection
* 🔍 Improved query rewriting
* 📚 Support for multiple PDF documents
* 📂 Support for additional document formats
* 💬 Improved conversational context management
* ⚡ Streaming LLM responses
* ☁️ Web deployment
* 🔐 User authentication
* 🗄️ Cloud-based vector storage

---

## 👨‍💻 Author

**Neha Chandwani**

🎓 B.Tech Computer Science and Engineering

---

## 📜 License

This project is intended for **educational and portfolio purposes**.
