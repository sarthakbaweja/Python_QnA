# Python Q&A Assistant — Spec

**Version:** 1.0.0  
**Status:** Approved

---

## Overview

An AI-powered Python Programming Q&A Assistant for data science learners. Answers are grounded in a Stack Overflow dataset (~1.7 GB of Python questions and answers) indexed into Qdrant Cloud. A LangGraph RAG pipeline serves answers via a FastAPI backend, with a Streamlit chat frontend. Everything runs via Docker Compose locally; production runs on Railway.

---

## Tech Stack

| Layer | Choice |
|---|---|
| LLM | Groq (`llama-3.1-70b-versatile`) |
| Embeddings | FastEmbed (`BAAI/bge-small-en`) — no API key required |
| Vector DB | Qdrant Cloud (free tier, external) |
| RAG framework | LangChain + LangGraph |
| Backend | FastAPI |
| Frontend | Streamlit |
| Containerization | Docker + Docker Compose (local) / Railway (production) |

---

## Requirements

### Functional

**API**
1. `POST /ask` — accepts `{ question: str, session_id: uuid }`, returns `{ answer, session_id, sources }`
2. `GET /health` — returns `{ status, qdrant, version }`
3. Multi-turn memory — last 20 messages per session, keyed by `session_id`
4. Grounded answers — responses cite Stack Overflow sources

**RAG Pipeline**
5. Retrieve top-5 relevant chunks from Qdrant for each question
6. Generate answers via Groq LLM with retrieved context injected into system prompt

**Dataset Indexing**
7. Index Stack Overflow Python Q&A dataset into Qdrant (one-time, run locally)
8. Document format: question + top answer (always) + up to 2 additional answers (score ≥ 10 only)
9. Idempotent — skip re-indexing if collection already exists

**Frontend**
10. Streamlit chat UI with persistent conversation display
11. Source citations shown per assistant response

### Non-Functional

- All services containerized (Docker Compose for local, Railway for prod)
- Secrets in `.env`; template in `.env.example`
- Prompt templates versioned in `rag/prompts/v1.py`
- Modular structure: `frontend/`, `backend/`, `rag/`
- Rate limiting: 20 req/min per IP (slowapi)
- CORS restricted to `FRONTEND_URL`
- UUID validation on `session_id`
- Non-root Docker user

---

## Architecture

### Directory Structure

```
python_qna/
├── frontend/                   # Streamlit app
│   ├── app.py
│   ├── Dockerfile
│   ├── railway.json
│   └── requirements.txt
│
├── backend/                    # FastAPI service
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── routes/ask.py
│   │   └── schemas/chat.py
│   ├── tests/
│   ├── Dockerfile
│   ├── railway.json
│   └── requirements.txt
│
├── rag/                        # RAG pipeline
│   ├── pipeline/
│   │   ├── graph.py            # LangGraph graph
│   │   ├── nodes.py            # retrieve + generate nodes
│   │   └── state.py            # GraphState TypedDict
│   ├── retriever/
│   │   └── qdrant_client.py
│   ├── prompts/
│   │   └── v1.py
│   ├── tests/
│   └── indexer/
│       ├── index_dataset.py
│       ├── chunk_formatter.py
│       ├── Dockerfile
│       └── requirements.txt
│
├── data/                       # .gitignored — CSVs live here
├── docs/
├── docker-compose.yml
├── .env                        # secrets — never committed
└── .env.example
```

### Services

**Local dev (Docker Compose):** Three containers, all connect to Qdrant Cloud via env vars.

| Service | Dockerfile | Port | Notes |
|---|---|---|---|
| `indexer` | `rag/indexer/Dockerfile` | — | One-shot, `restart: no` |
| `backend` | `backend/Dockerfile` | 8000 | |
| `frontend` | `frontend/Dockerfile` | 8501 | `depends_on: backend` |

**Production (Railway):** Two services — `backend` and `frontend`. Qdrant Cloud is external.

```
Qdrant Cloud (free tier)
       │ HTTPS + API key
Railway: backend (port 8000, private)
       │ Railway private network
Railway: frontend (port 8501, public domain)
```

The frontend is the only public-facing service. Backend communicates over Railway's private network (`http://backend.railway.internal:8000`) — no public backend URL needed.

---

## LangGraph RAG Pipeline

### Graph Topology

```
START → [retrieve] → [generate] → END
```

### GraphState

```python
class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # reducer appends, not replaces
    context:  list[str]
    sources:  list[dict]
    session_id: str
```

### Node: `retrieve`
- Extracts the latest HumanMessage from state
- Queries Qdrant via FastEmbed, returns top-5 chunks
- Returns `{"context": [...], "sources": [...]}`

### Node: `generate`
- Trims `messages` to last 20 before the LLM call
- Injects context into system prompt via `RETRIEVAL_CONTEXT_TEMPLATE`
- Calls Groq, returns `{"messages": [AIMessage(...)]}`

### Memory
- `MemorySaver` (in-process checkpointer) keyed by `thread_id` = `session_id`
- State survives across requests for the same session; lost on container restart (acceptable for demo)

---

## API Design

### `POST /ask`

**Request:**
```json
{ "question": "How do I use list comprehensions?", "session_id": "550e8400-..." }
```
Constraints: `question` min 1 / max 2000 chars; `session_id` must be valid UUID.

**Response:**
```json
{
  "answer": "...",
  "session_id": "550e8400-...",
  "sources": [{"question_title": "...", "question_id": 12345, "score": 42}]
}
```

### `GET /health`

```json
{ "status": "ok", "qdrant": "connected", "version": "1.0.0" }
```

### Error Codes

| Condition | HTTP |
|---|---|
| Empty / too-long question or invalid UUID | 422 |
| Qdrant unreachable | 503 (health) / 502 (ask) |
| Groq API error | 502 |
| Rate limit exceeded | 429 |

---

## Indexing Strategy

Dataset: `Questions.csv` + `Answers.csv` + `Tags.csv`

**Per question:**
1. Find all answers where `ParentId == question.Id`, sort by `Score DESC`
2. Always include answer #1 (top answer, regardless of score)
3. Include answers #2 and #3 only if `Score >= 10`
4. Strip HTML with BeautifulSoup
5. Format chunk:

```
Question: {Title}
{Body}

[TOP ANSWER (score: N)]
{answer_1_body}

[ANSWER 2 (score: N)]
{answer_2_body}

[ANSWER 3 (score: N)]
{answer_3_body}
```

**Qdrant payload:** `question_id`, `title`, `question_score`, `answer_count`, `text`  
**Batch size:** 256  
**Collection:** `python_qna`  
**Embedding:** `BAAI/bge-small-en` (384-dim, FastEmbed)

Indexing runs once locally, pointing at Qdrant Cloud. The collection persists permanently — no re-indexing needed.

---

## Prompt Versioning

Prompts live in `rag/prompts/v1.py`. New versions create `v2.py`; old versions are kept. Active version controlled by `ACTIVE_PROMPT_VERSION` env var.

```python
VERSION = "v1"
SYSTEM_PROMPT = """You are a Python programming assistant helping data science learners.
Answer questions grounded strictly in the retrieved Stack Overflow context provided below.
If the context does not contain enough information, say so clearly — do not hallucinate.
When referencing information, cite the Stack Overflow question title it came from."""
RETRIEVAL_CONTEXT_TEMPLATE = """--- Retrieved Context ---
{context}
--- End Context ---"""
```

---

## Deployment

### Local Dev

```bash
# 1. Configure env (fill in Qdrant Cloud + Groq credentials)
cp .env.example .env

# 2. Place dataset CSVs in data/
# data/Questions.csv, data/Answers.csv, data/Tags.csv

# 3. Index once (connects to Qdrant Cloud)
pip install -r rag/indexer/requirements.txt
PYTHONPATH=. DATA_DIR=data python -m rag.indexer.index_dataset

# 4. Start backend + frontend
docker-compose up --build
```

### Railway Production

**One-time setup:**
1. Create Qdrant Cloud cluster, run `index_dataset.py` locally against it
2. `railway init --name python-qna`
3. Create `backend` service — root dir: `backend/`, set env vars
4. Create `frontend` service — root dir: `frontend/`, set `BACKEND_URL=http://backend.railway.internal:8000`
5. Push to `main` → Railway auto-deploys both services

**Backend env vars (Railway dashboard):**
```
GROQ_API_KEY=<key>
QDRANT_HOST=<cluster>.qdrant.io
QDRANT_PORT=6333
QDRANT_API_KEY=<key>
QDRANT_USE_TLS=true
QDRANT_COLLECTION=python_qna
ACTIVE_PROMPT_VERSION=v1
```

**Frontend env vars:**
```
BACKEND_URL=http://backend.railway.internal:8000
```

**Estimated cost:** ~$1.50–2.50/month on Railway free tier; Qdrant Cloud free tier is external.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | **required** | Groq API key |
| `GROQ_MODEL` | `llama-3.1-70b-versatile` | Groq model |
| `QDRANT_HOST` | `localhost` | Qdrant hostname |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_API_KEY` | _(empty)_ | Qdrant Cloud API key |
| `QDRANT_USE_TLS` | `false` | Set `true` for Qdrant Cloud |
| `QDRANT_COLLECTION` | `python_qna` | Collection name |
| `ACTIVE_PROMPT_VERSION` | `v1` | Prompt version |
| `BACKEND_URL` | `http://backend:8000` | Backend URL (used by Streamlit) |
| `FRONTEND_URL` | `http://localhost:8501` | Allowed CORS origin |

---

## Testing

### Unit Tests (no services required)
```bash
PYTHONPATH=. pytest backend/tests/ rag/tests/ -v -m "not integration"
```

| Suite | Tests | Covers |
|---|---|---|
| `rag/tests/test_indexer.py` | 9 | strip_html, format_chunk, answer score filtering |
| `rag/tests/test_memory.py` | 4 | 20-message trim |
| `rag/tests/test_retriever.py` | 3 | search_similar mapping, collection check |
| `backend/tests/test_health.py` | 3 | GET /health shape, Qdrant disconnected |
| `backend/tests/test_ask.py` | 7 | POST /ask validation, multi-turn, 502 |

### Integration Tests (requires live Qdrant + indexed data)
```bash
PYTHONPATH=. pytest rag/tests/test_retriever.py -v -m integration
```

---

## Future Scope

1. **Message compaction** — when total count > 20, summarize messages 1–15 into a single summary before trimming
2. **Router + RAG** — LLM router decides whether retrieval is needed or the question can be answered from history
3. **Full agent with tools** — Qdrant retrieval as a callable tool, allowing re-query with refined search terms
4. **Redis sessions** — replace `MemorySaver` with a Redis checkpointer for persistence across restarts and horizontal scaling
