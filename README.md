# 🤖 Agentic RAG System

A production-ready Retrieval-Augmented Generation (RAG) system with agentic workflows, hybrid search, and automatic evaluation capabilities. Built with LangGraph, LangChain & FastAPI.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Agent Workflow](#-agent-workflow)
- [Tools](#-tools)
- [Evaluation Metrics](#-evaluation-metrics)
- [Examples](#-examples)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### Core Capabilities
- **Hybrid Search**: Combines FAISS (semantic) + BM25 (lexical) retrieval with cross-encoder reranking
- **Agentic Workflows**: Multi-step planning and execution using LangGraph
- **Multi-Tool Support**: 7 specialized tools including RAG, web search, calculations, and stock prices
- **Citation Generation**: Automatic citation with page numbers and relevance scores
- **Query Rewriting**: Automatic query optimization for better retrieval
- **Automatic Evaluation**: Built-in RAG quality metrics (context relevance, faithfulness, answer relevance, citation accuracy)
- **Multi-Threading**: Isolated document contexts per conversation thread
- **Persistent Memory**: SQLite-based conversation history

### Production Features
- RESTful API with FastAPI
- CORS and compression middleware
- Request validation with Pydantic
- Comprehensive logging
- Health check endpoints
- File size limits and validation
- Error handling and recovery

## 🏗️ Architecture

### System Overview

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       │ HTTP/REST
       │
┌──────▼──────────────────────────────────────────┐
│              FastAPI Server                      │
│  ┌────────────────────────────────────────────┐ │
│  │  Endpoints: /upload, /chat, /thread, etc   │ │
│  └────────────┬───────────────────────────────┘ │
└───────────────┼──────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────┐
│           LangGraph Agent (chatbot)              │
│  ┌──────────────────────────────────────────┐   │
│  │    Planner → Executor → Answer → Eval    │   │
│  └──────────────────────────────────────────┘   │
└───────────────┬──────────────────────────────────┘
                │
        ┌───────┴───────┐
        │               │
┌───────▼─────┐  ┌──────▼──────┐
│   Tools     │  │  Retrievers  │
│             │  │              │
│ • RAG       │  │ • FAISS      │
│ • Search    │  │ • BM25       │
│ • Calculator│  │ • Reranker   │
│ • Stock     │  │              │
│ • Summary   │  │ Per-Thread   │
│ • Citation  │  │ Storage      │
│ • Rewrite   │  │              │
└─────────────┘  └──────────────┘
```

### Agent Workflow

```
User Query
    │
    ▼
┌─────────────┐
│   PLANNER   │ ◄─── Checks PDF availability
└──────┬──────┘      Decides tool sequence
       │
       │ Plan (JSON)
       │
       ▼
┌─────────────┐
│  EXECUTOR   │ ◄─── Runs tools in sequence
└──────┬──────┘      Handles errors
       │
       │ Tool Results
       │
       ▼
┌─────────────┐
│   ANSWER    │ ◄─── Synthesizes final response
└──────┬──────┘      Formats citations
       │
       │ Final Answer
       │
       ▼
┌─────────────┐
│ EVALUATION  │ ◄─── Scores RAG quality
└──────┬──────┘      Logs metrics
       │
       ▼
   Response
```

### Hybrid Retrieval Pipeline

```
User Query
    │
    ├──────────────┬──────────────┐
    │              │              │
    ▼              ▼              ▼
┌────────┐   ┌─────────┐   ┌──────────┐
│ FAISS  │   │  BM25   │   │ Optional │
│ (k=20) │   │ (k=20)  │   │ Rewrite  │
└────┬───┘   └────┬────┘   └──────────┘
     │            │
     └──────┬─────┘
            │
            ▼
    ┌──────────────┐
    │ Deduplicate  │
    │   (~30 docs) │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  Cross-      │
    │  Encoder     │
    │  Reranking   │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  Top-K       │
    │  (k=4)       │
    └──────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip or conda
- OpenAI API key
- (Optional) Alpha Vantage API key for stock prices

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/agentic-rag-system.git
cd agentic-rag-system
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n rag-env python=3.9
conda activate rag-env
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**:
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-dotenv==1.0.0
pydantic==2.5.0
langchain==0.1.0
langchain-openai==0.0.2
langchain-community==0.0.10
langchain-text-splitters==0.0.1
langgraph==0.0.20
faiss-cpu==1.7.4
sentence-transformers==2.2.2
pypdf==3.17.0
duckduckgo-search==4.1.1
requests==2.31.0
python-multipart==0.0.6
```

### Step 4: Environment Setup

Create a `.env` file in the project root:

```bash
# OpenAI API
OPENAI_API_KEY=sk-your-openai-api-key-here

# Alpha Vantage (for stock prices)
ALPHA_VANTAGE_API_KEY=your-alpha-vantage-key

# Server Configuration
HOST=127.0.0.1
PORT=8000
CORS_ORIGINS=*

# File Upload Limits
MAX_FILE_SIZE_MB=10
```

## ⚙️ Configuration

### Model Configuration

Edit `rag_backend.py` to change models:

```python
# Language Model
llm = ChatOpenAI(model="gpt-4o-mini")  # or "gpt-4", "gpt-3.5-turbo"

# Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
```

### Retrieval Parameters

```python
# FAISS retriever
search_kwargs={"k": 20}  # Number of candidates

# BM25 retriever
bm25_retriever.k = 20

# Final reranked results
top_k = 4  # Number of chunks to use
```

### Chunking Strategy

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,        # Characters per chunk
    chunk_overlap=200,     # Overlap between chunks
    separators=["\n\n", "\n", " ", ""]
)
```

## 📖 Usage

### Starting the Server

```bash
# Development mode with auto-reload
python rag_app.py

# Or using uvicorn directly
uvicorn rag_app:app --reload --host 0.0.0.0 --port 8000
```

Server will start at `http://localhost:8000`

### Interactive API Documentation

Visit `http://localhost:8000/docs` for Swagger UI or `http://localhost:8000/redoc` for ReDoc.

### Basic Workflow

#### 1. Upload a PDF

```bash
curl -X POST "http://localhost:8000/upload?thread_id=thread123" \
  -F "file=@document.pdf"
```

**Response**:
```json
{
  "filename": "document.pdf",
  "thread_id": "thread123",
  "documents": 10,
  "chunks": 45,
  "processing_time_ms": 2341,
  "message": "Successfully indexed 45 chunks from 10 pages"
}
```

#### 2. Ask Questions

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the key findings in section 3?",
    "thread_id": "thread123"
  }'
```

**Response**:
```json
{
  "response": "Based on the document, section 3 presents three key findings...",
  "thread_id": "thread123",
  "processing_time_ms": 1823,
  "metadata": {
    "filename": "document.pdf",
    "documents": 10,
    "chunks": 45
  }
}
```

#### 3. Get Citations

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain the methodology with citations",
    "thread_id": "thread123"
  }'
```

The agent automatically uses the `cite_answer` tool when citations are requested.

#### 4. Check Thread Info

```bash
curl "http://localhost:8000/thread/thread123"
```

#### 5. List All Threads

```bash
curl "http://localhost:8000/threads"
```

#### 6. Delete Thread

```bash
curl -X DELETE "http://localhost:8000/thread/thread123"
```

## 🔌 API Reference

### Endpoints

#### `POST /upload`

Upload and index a PDF document.

**Parameters**:
- `thread_id` (query, required): Unique thread identifier
- `file` (form-data, required): PDF file (max 10MB)

**Response**: `UploadResponse`

---

#### `POST /chat`

Send a message and get AI response.

**Body**:
```json
{
  "message": "string (1-5000 chars)",
  "thread_id": "string (1-100 chars)"
}
```

**Response**: `ChatResponse`

---

#### `GET /thread/{thread_id}`

Get information about a thread.

**Response**: `ThreadInfoResponse`

---

#### `DELETE /thread/{thread_id}`

Delete a thread and its data.

**Response**: `204 No Content`

---

#### `GET /threads`

List all active threads.

**Response**: `List[ThreadInfoResponse]`

---

#### `GET /health`

Health check endpoint.

**Response**: `HealthResponse`

### Response Models

#### UploadResponse
```json
{
  "filename": "string",
  "thread_id": "string",
  "documents": "integer",
  "chunks": "integer",
  "processing_time_ms": "integer",
  "message": "string"
}
```

#### ChatResponse
```json
{
  "response": "string",
  "thread_id": "string",
  "processing_time_ms": "integer",
  "metadata": {
    "filename": "string",
    "documents": "integer",
    "chunks": "integer"
  }
}
```

## 🛠️ Tools

The system includes 7 specialized tools:

### 1. **rag_tool**
Retrieves relevant information from uploaded PDF using hybrid search.

**Input**:
```python
{
  "query": "What is the main topic?",
  "thread_id": "thread123"
}
```

**Output**:
```python
{
  "query": "What is the main topic?",
  "context": ["chunk1", "chunk2", ...],
  "metadata": [{"page": 1}, {"page": 2}, ...],
  "scores": [0.92, 0.87, ...],
  "source_file": "document.pdf"
}
```

### 2. **cite_answer**
Answers query with citations and page numbers.

**Input**:
```python
{
  "query": "Explain the methodology",
  "thread_id": "thread123"
}
```

**Output**:
```python
{
  "answer": "The methodology involves... [1] [2]",
  "citations": [
    {"id": 1, "page": 5, "relevance_score": 0.93},
    {"id": 2, "page": 6, "relevance_score": 0.89}
  ],
  "source_file": "document.pdf",
  "num_sources": 4
}
```

### 3. **summarize_pdf**
Generates document summary.

**Input**:
```python
{
  "thread_id": "thread123",
  "max_chunks": 20
}
```

**Output**:
```python
{
  "summary": "This document discusses...",
  "source_file": "document.pdf"
}
```

### 4. **rewrite_query**
Rewrites vague queries for better retrieval.

**Input**:
```python
{
  "query": "stuff about that thing"
}
```

**Output**:
```python
{
  "original_query": "stuff about that thing",
  "rewritten_query": "What are the key features and characteristics of..."
}
```

### 5. **search_tool** (DuckDuckGo)
Web search for current information.

**Input**: Query string

**Output**: Search results

### 6. **get_stock_price**
Fetches latest stock price.

**Input**:
```python
{
  "symbol": "AAPL"
}
```

**Output**: Stock data from Alpha Vantage

### 7. **calculator**
Basic arithmetic operations.

**Input**:
```python
{
  "first_num": 10,
  "second_num": 5,
  "operation": "add"  # add, sub, mul, div
}
```

**Output**:
```python
{
  "first_num": 10,
  "second_num": 5,
  "operation": "add",
  "result": 15
}
```

## 📊 Evaluation Metrics

The system automatically evaluates RAG quality using four metrics:

### 1. Context Relevance (0.0-1.0)
Measures how relevant retrieved chunks are to the query. Based on reranker scores.

**Formula**: Average of normalized reranker scores

### 2. Faithfulness (0.0-1.0)
Measures how well the answer is grounded in retrieved context.

**Evaluated by**: LLM scoring answer support from context

### 3. Answer Relevance (0.0-1.0)
Measures how well the answer addresses the original question.

**Evaluated by**: LLM scoring answer completeness

### 4. Citation Accuracy (0.0 or 1.0)
Binary check for proper citation usage when expected.

**Criteria**: Citations present when requested + citation markers in answer

### Overall Score
Average of all four metrics (0.0-1.0)

**Example Log Output**:
```
RAG EVALUATION → {
  'context_relevance': 0.891,
  'faithfulness': 0.95,
  'answer_relevance': 0.92,
  'citation_accuracy': 1.0,
  'overall_score': 0.940,
  'num_chunks': 4,
  'avg_reranker_score': 0.891
}
```

## 💡 Examples

### Example 1: Document Q&A

```python
import requests

# Upload document
files = {'file': open('research_paper.pdf', 'rb')}
response = requests.post(
    'http://localhost:8000/upload?thread_id=research1',
    files=files
)
print(response.json())

# Ask question
response = requests.post(
    'http://localhost:8000/chat',
    json={
        'message': 'What are the main conclusions?',
        'thread_id': 'research1'
    }
)
print(response.json()['response'])
```

### Example 2: Citation-Heavy Research

```python
response = requests.post(
    'http://localhost:8000/chat',
    json={
        'message': 'Explain the experimental setup with citations',
        'thread_id': 'research1'
    }
)

result = response.json()
print(result['response'])
# Output: The experiment used... [1] with controls for... [2][3]
```

### Example 3: Multi-Tool Query

```python
# Agent automatically combines tools
response = requests.post(
    'http://localhost:8000/chat',
    json={
        'message': 'Compare findings in the document with recent news about AI',
        'thread_id': 'research1'
    }
)
# Agent uses: rag_tool + search_tool + synthesis
```

### Example 4: Stock Analysis

```python
response = requests.post(
    'http://localhost:8000/chat',
    json={
        'message': 'What is the current price of TSLA stock?',
        'thread_id': 'finance1'
    }
)
```

## 📁 Project Structure

```
agentic-rag-system/
├── rag_app.py              # FastAPI application
├── rag_backend.py          # LangGraph agent & tools
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables
├── chatbot.db             # SQLite conversation history
├── rag.log                # Application logs
├── README.md              # This file
└── tests/                 # (Optional) Test suite
    ├── test_api.py
    ├── test_retrieval.py
    └── test_tools.py
```

## 🔒 Security Considerations

- **API Keys**: Never commit `.env` file. Use environment variables in production.
- **File Validation**: Only PDF files accepted, max 10MB.
- **Input Sanitization**: Pydantic models validate all inputs.
- **Rate Limiting**: Consider adding rate limiting middleware for production.
- **CORS**: Configure `CORS_ORIGINS` appropriately for your deployment.

## 🚀 Deployment

### Docker Deployment

**Dockerfile**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "rag_app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and run**:
```bash
docker build -t agentic-rag .
docker run -p 8000:8000 --env-file .env agentic-rag
```

### Cloud Deployment

Works with:
- **AWS**: EC2, ECS, Lambda (with API Gateway)
- **GCP**: Cloud Run, App Engine, Compute Engine
- **Azure**: App Service, Container Instances
- **Heroku**, **Railway**, **Render**

## 🐛 Troubleshooting

### Common Issues

**1. ImportError: No module named 'langchain_community'**
```bash
pip install langchain-community
```

**2. FAISS installation fails**
```bash
# Use CPU version
pip install faiss-cpu

# Or GPU version (requires CUDA)
pip install faiss-gpu
```

**3. OpenAI API errors**
- Check API key in `.env`
- Verify API key has credits
- Check rate limits

**4. PDF processing fails**
- Verify file is valid PDF
- Check file size < 10MB
- Ensure sufficient disk space

**5. Slow retrieval**
- Reduce `k` in FAISS/BM25 retrievers
- Use smaller embedding model
- Optimize chunk size

## 📈 Performance Tips

1. **Chunking**: Balance chunk_size (800) vs overlap (200) for your documents
2. **Reranking**: Adjust `top_k=4` based on context window limits
3. **Caching**: Add Redis for retriever caching in production
4. **Async**: Use `async/await` for I/O operations
5. **Batching**: Process multiple queries with batch embeddings

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Built with [LangChain](https://langchain.com/) and [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by [OpenAI](https://openai.com/)
- Search via [DuckDuckGo](https://duckduckgo.com/)
- Stock data from [Alpha Vantage](https://www.alphavantage.co/)

## 📞 Support
- **Email**: hiteshram321@gmail.com

---

