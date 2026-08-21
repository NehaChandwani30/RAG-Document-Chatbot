# 📚 RAG Document Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions about their content. The system combines semantic retrieval with BM25 keyword search to retrieve relevant document chunks and uses a local LLM to generate context-grounded answers.

## 🔎 Overview

Traditional LLMs may generate responses that are not grounded in a user's private documents. This project addresses that problem by retrieving relevant information from an uploaded PDF before generating an answer.

The system implements a hybrid retrieval pipeline that combines:

* Semantic similarity search using vector embeddings
* BM25 keyword-based retrieval
* Weighted hybrid ranking
* Context-aware question rewriting
* Local LLM-based answer generation
* Source and page-level retrieval
* Retrieval evaluation using Recall@5

## 🚀 Key Features

* PDF document ingestion using PyPDFLoader
* Recursive text chunking with configurable chunk size and overlap
* Local text embeddings using `nomic-embed-text`
* Persistent vector storage using ChromaDB
* Semantic similarity retrieval
* BM25 keyword retrieval
* Hybrid retrieval using normalized weighted scores
* Top-K document retrieval
* Conversation-aware question rewriting
* Local LLM inference using Llama 3.2 through Ollama
* Context-grounded answer generation
* Source page identification
* Streamlit-based chatbot interface
* Retrieval evaluation comparing Semantic, BM25, and Hybrid Search

## 🏗️ Architecture

```text
                    PDF Document
                         |
                         v
                  Document Loader
                    PyPDFLoader
                         |
                         v
                  Text Chunking
              RecursiveCharacterTextSplitter
                         |
                         v
                  Text Embeddings
                 nomic-embed-text
                         |
                         v
                     ChromaDB
                  Vector Database
                         |
             +-----------+-----------+
             |                       |
             v                       v
       Semantic Search           BM25 Search
             |                       |
             +-----------+-----------+
                         |
                         v
                  Score Normalization
                         |
                         v
                  Hybrid Ranking
             60% Semantic + 40% BM25
                         |
                         v
                    Top-K Chunks
                         |
                         v
                 Context Construction
                         |
                         v
                   Llama 3.2
                         |
                         v
                   Final Answer
                         |
                         v
                  Streamlit Interface
```

## ⚙️ How It Works

### 1. PDF Loading

The uploaded PDF is processed using `PyPDFLoader`.

Each page is converted into a LangChain `Document` containing the page text and metadata such as the source and page number.

### 2. Document Chunking

The extracted text is divided into smaller overlapping chunks using `RecursiveCharacterTextSplitter`.

Current configuration:

```text
Chunk size: 1000
Chunk overlap: 200
```

Chunking allows the retrieval system to search smaller and more relevant sections of the document instead of processing the entire document at once.

### 3. Embedding Generation

Each document chunk is converted into a vector representation using:

```text
nomic-embed-text
```

The embedding model runs locally through Ollama.

These vectors allow the system to identify chunks that are semantically similar to a user's question.

### 4. Vector Storage

Document embeddings are stored in ChromaDB.

ChromaDB provides persistent vector storage and allows the application to perform similarity searches against the processed document.

### 5. Semantic Retrieval

For semantic search, the user's question is converted into an embedding and compared against the document chunk embeddings stored in ChromaDB.

This allows the system to retrieve relevant information even when the wording of the question differs from the wording used in the document.

### 6. BM25 Retrieval

The project also implements BM25 keyword-based retrieval.

BM25 is useful when important keywords from the user's question appear directly in the document.

This complements semantic search by providing an additional lexical retrieval signal.

### 7. Hybrid Retrieval

The project combines semantic retrieval and BM25 retrieval using normalized scores.

The current weighting is:

```text
Hybrid Score =
0.6 × Semantic Score +
0.4 × BM25 Score
```

The combined score is used to rank the document chunks, and the top 5 chunks are selected as context for the LLM.

This approach combines semantic understanding with exact keyword matching.

### 8. Question Rewriting

The system uses conversation history to rewrite follow-up questions into standalone questions before retrieval.

For example:

```text
User: What is Generative AI?

User: What are its applications?
```

The second question can be rewritten as:

```text
What are the applications of Generative AI?
```

This improves retrieval for conversational queries that depend on previous messages.

### 9. Context Construction

The top retrieved chunks are combined into a context that is passed to the LLM.

The model is instructed to use the provided document context when generating the answer.

### 10. Answer Generation

Llama 3.2 is used through Ollama to generate the final response.

The system is designed to avoid introducing unsupported information and responds with:

```text
I couldn't find the answer in the provided document.
```

when the required information is not available in the retrieved context.

## 📊 Retrieval Evaluation

The project includes an evaluation script (`evaluate.py`) that compares three retrieval approaches:

1. Semantic Search
2. BM25
3. Hybrid Retrieval

The evaluation uses predefined document-based questions and checks whether at least one relevant page appears in the top 5 retrieved results.

### Recall@5

```text
Recall@5 =
Questions with a relevant result in Top 5
-----------------------------------------
Total evaluation questions
```

### Current Evaluation Results

| Retrieval Method | Recall@5 |
| ---------------- | -------: |
| Semantic Search  |   94.12% |
| BM25             |   82.35% |
| Hybrid Retrieval |  100.00% |

On the current evaluation dataset, the hybrid retriever achieved **100% Recall@5**, compared with **94.12%** for semantic search and **82.35%** for BM25.

These results are specific to the current test document and evaluation questions and should not be interpreted as a general benchmark.

## 🛠️ Technology Stack

| Technology       | Purpose                               |
| ---------------- | ------------------------------------- |
| Python           | Application development               |
| LangChain        | RAG and document-processing framework |
| Ollama           | Local model runtime                   |
| Llama 3.2        | Answer generation                     |
| nomic-embed-text | Text embeddings                       |
| ChromaDB         | Vector database                       |
| BM25             | Keyword retrieval                     |
| PyPDF            | PDF processing                        |
| Streamlit        | Web interface                         |
| Git & GitHub     | Version control                       |

## 📁 Project Structure

```text
RAG-Document-Chatbot/
│
├── src/
│   ├── __init__.py
│   ├── document_loader.py
│   ├── embeddings.py
│   ├── hybrid_retriever.py
│   ├── rag_pipeline.py
│   ├── retriever.py
│   ├── text_splitter.py
│   └── vector_store.py
│
├── data/
│   └── sample.pdf
│
├── app.py
├── evaluate.py
├── requirements.txt
├── README.md
└── .gitignore
```

### Main Components

**`document_loader.py`**

Loads PDF documents using `PyPDFLoader`.

**`text_splitter.py`**

Splits documents into overlapping chunks using `RecursiveCharacterTextSplitter`.

**`embeddings.py`**

Initializes the `nomic-embed-text` embedding model through Ollama.

**`vector_store.py`**

Processes documents, generates embeddings, and stores document vectors in ChromaDB.

**`retriever.py`**

Provides semantic retrieval from the ChromaDB vector store.

**`hybrid_retriever.py`**

Combines semantic similarity scores with BM25 scores using normalized weighted ranking.

**`rag_pipeline.py`**

Handles question rewriting, hybrid retrieval, context construction, and LLM-based answer generation.

**`app.py`**

Provides the Streamlit user interface for PDF uploads and conversational question answering.

**`evaluate.py`**

Evaluates Semantic Search, BM25, and Hybrid Retrieval using Recall@5.

## 💻 Installation

### Prerequisites

Make sure the following are installed:

* Python 3.10+
* Ollama
* Git

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/RAG-Document-Chatbot.git
cd RAG-Document-Chatbot
```

### 2. Create a Virtual Environment

#### Windows

```powershell
python -m venv rag_env
rag_env\Scripts\activate
```

#### Linux/macOS

```bash
python3 -m venv rag_env
source rag_env/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## 🤖 Ollama Setup

This project uses Ollama for local LLM and embedding inference.

Install Ollama and make sure it is running.

Pull the required models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Verify the installed models:

```bash
ollama list
```

Both models should be available before starting the application.

## ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the local URL displayed by Streamlit in your browser.

Upload a PDF document and start asking questions about its content.

## 🧪 Running the Evaluation

To evaluate the retrieval system:

```bash
python evaluate.py
```

The script compares:

```text
Semantic Search
BM25
Hybrid Retrieval
```

using Recall@5.

## ❓ Example Questions

After uploading a suitable PDF, questions can include:

```text
What is Generative AI?

What types of content can Generative AI create?

How is Generative AI different from traditional AI?

What are the applications of Generative AI?

What is the Transformer architecture?

What is self-attention?

What is an LLM?

What is Retrieval-Augmented Generation?

Why is retrieval useful for LLMs?

What is fine-tuning?

What are hallucinations in LLMs?
```

Follow-up questions can also be asked using the conversation history.

## 🧠 Design Decisions

### Why Hybrid Retrieval?

Semantic search is effective at understanding the meaning of a query, while BM25 is effective at matching important keywords.

Combining both methods provides two complementary retrieval signals:

```text
Semantic Similarity
        +
Keyword Matching
        =
Hybrid Retrieval
```

The current implementation uses a 60:40 weighting between semantic and BM25 scores.

### Why Local Models?

Ollama allows the LLM and embedding model to run locally.

Benefits include:

* No external LLM API dependency
* Local document processing
* Greater control over data
* No API key required for inference

## ⚠️ Limitations

* Current implementation processes one uploaded document at a time.
* Retrieval quality depends on chunking and embedding quality.
* BM25 uses simple whitespace-based tokenization.
* The current evaluation dataset is limited to the included test questions and document.
* Local LLM performance depends on available system hardware.

## 🔮 Future Improvements

* Add support for multiple documents
* Add document management and deletion
* Implement cross-encoder reranking
* Improve BM25 tokenization
* Add Precision@K, MRR, and NDCG evaluation
* Add answer-level evaluation
* Improve query rewriting
* Add streaming responses
* Add support for additional document formats
* Add web deployment
* Add authentication and user-specific document collections

## 📄 License

This project is intended for educational and portfolio purposes.

## 👩‍💻 Author

**Neha Chandwani**

B.Tech Computer Science and Engineering
