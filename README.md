# Python Q&A Assistant

AI-powered Python Q&A grounded in Stack Overflow data. Ask any Python question and get answers backed by real Stack Overflow threads.

## Architecture

| Layer | Technology |
|---|---|
| Frontend | Streamlit (port 8501) |
| Backend | FastAPI (port 8000) |
| RAG Pipeline | LangChain + LangGraph |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Embeddings | FastEmbed BAAI/bge-small-en |
| Vector DB | Qdrant Cloud (free tier, external) |

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Groq API key — free at https://console.groq.com
- Qdrant Cloud account + cluster — free at https://cloud.qdrant.io
- Dataset CSVs in `./data/`

### 1. Clone and configure

```bash
git clone <repo-url>
cd python_qna
cp .env.example .env
# Edit .env — set GROQ_API_KEY, QDRANT_HOST, QDRANT_API_KEY, QDRANT_USE_TLS=true
```

### 2. Place dataset files

```
data/
├── Questions.csv
├── Answers.csv
└── Tags.csv
```

### 3. Index the dataset (one-time, run locally)

This runs the indexer against your Qdrant Cloud cluster. Only needed once — the collection persists in Qdrant Cloud.

```bash
pip install -r backend/rag/indexer/requirements.txt
PYTHONPATH=backend DATA_DIR=data python -m rag.indexer.index_dataset
```

Takes ~15–30 min for the full dataset. Subsequent runs are skipped automatically (idempotency check).

### 4. Start backend + frontend

```bash
docker-compose up --build
```

### 5. Open the app

- **Chat UI**: http://localhost:8501
- **API docs**: http://localhost:8000/docs

---

## Local Development (without Docker)

```bash
# 1. Start Qdrant (or use Qdrant Cloud)
docker run -p 6333:6333 qdrant/qdrant

# 2. Index dataset (one-time)
pip install -r backend/rag/indexer/requirements.txt
PYTHONPATH=backend DATA_DIR=data QDRANT_HOST=localhost python -m rag.indexer.index_dataset

# 3. Start backend
pip install -r backend/requirements.txt
PYTHONPATH=backend GROQ_API_KEY=your_key QDRANT_HOST=localhost \
    uvicorn app.main:app --reload --port 8000

# 4. Start frontend (new terminal)
pip install -r frontend/requirements.txt
BACKEND_URL=http://localhost:8000 streamlit run frontend/app.py
```

---

## Running Tests

```bash
# Unit tests (no services required)
PYTHONPATH=backend pytest tests/ rag/tests/ -v -m "not integration"

# Integration tests (requires Qdrant + indexed data)
PYTHONPATH=backend pytest rag/tests/test_retriever.py -v -m integration
```

---

## API Reference

### POST /ask

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I use list comprehensions?", "session_id": "550e8400-e29b-41d4-a716-446655440000"}'
```

Response:
```json
{
  "answer": "List comprehensions provide a concise way...",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "sources": [
    {"question_title": "Python list comprehension", "question_id": 12345, "score": 42}
  ]
}
```

### GET /health

```bash
curl http://localhost:8000/health
```

Response:
```json
{"status": "ok", "qdrant": "connected", "version": "1.0.0"}
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | **required** | Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `QDRANT_HOST` | `localhost` | Qdrant hostname (`<cluster>.qdrant.io` for Qdrant Cloud) |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_API_KEY` | _(empty)_ | Qdrant Cloud API key — leave blank for local dev |
| `QDRANT_USE_TLS` | `false` | Set to `true` for Qdrant Cloud |
| `QDRANT_COLLECTION` | `python_qna` | Qdrant collection name |
| `ACTIVE_PROMPT_VERSION` | `v1` | Prompt version to load from `rag/prompts/` |
| `BACKEND_URL` | `http://backend:8000` | Backend URL (used by Streamlit) |

---

## Project Structure

```
python_qna/
├── frontend/      # Streamlit chat UI
├── backend/       # FastAPI service (POST /ask, GET /health)
├── rag/           # LangGraph RAG pipeline, Qdrant retriever, indexer
├── docs/          # Design spec, test report
├── data/          # Dataset CSVs (git-ignored)
└── docker-compose.yml
```

## Future Scope
- Message compaction: summarize first 15 messages when count > 20
- Router + RAG: LLM decides whether retrieval is needed per question
- Agent mode: Qdrant retrieval as a tool with re-query capability
- Redis sessions: persist conversation state across container restarts
- Tag-based filtering: extract tags from user query, use as Qdrant payload pre-filter before vector search; or expose as a user-controlled dropdown in the frontend
- Hybrid retrieval + reranking: combine dense vector search (current) with sparse BM25 keyword search, then rerank merged results with a cross-encoder (e.g. `BAAI/bge-reranker-base`) before passing context to the LLM
- Semantic response caching: embed incoming queries, store query vector + response in Redis; on new queries, check cosine similarity against cached vectors and return cached response if above threshold — cuts latency and LLM cost for near-duplicate questions
- Context recall eval: extend RAGAS suite with `ContextRecall` metric — uses Stack Overflow top-scored answer as ground truth to measure whether retrieved chunks contain sufficient information to answer the question
- Answer correctness eval: compare LLM-generated answer against Stack Overflow accepted answer using an LLM judge scoring factual overlap (no reference implementation needed — ground truth already in Answers.csv)
- Loop-until-dry adversarial eval: spawn multiple independent LLM judges per answer to refute faithfulness claims; flag answers where majority vote says hallucination — catches edge cases single-judge RAGAS misses
- Latency profiling: instrument retrieval and LLM stages separately; assert p95 retrieval < 500ms and p95 end-to-end < 5s as part of the eval suite
