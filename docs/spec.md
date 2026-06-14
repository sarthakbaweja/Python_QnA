# Python Q&A Assistant — Spec

**Version:** 1.1.0  
**Status:** Approved

---

## Overview

An AI-powered Python Programming Q&A Assistant for data science learners. Answers are grounded in a Stack Overflow dataset indexed into Qdrant Cloud. A LangGraph RAG pipeline serves answers via a FastAPI backend, with a Streamlit chat frontend. Everything runs via Docker Compose locally; production runs on Railway.

---

## Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| LLM | Groq (`llama-3.3-70b-versatile`) | Free tier, fast inference, no cold starts |
| Embeddings | FastEmbed (`BAAI/bge-small-en`, 384-dim) | Runs locally, no API key, baked into Docker image at build time |
| Vector DB | Qdrant Cloud (free tier, external) | Managed, persistent across restarts, free at this scale |
| RAG Framework | LangChain + LangGraph | LangGraph gives explicit retrieve→generate graph with built-in memory checkpointing |
| Backend | FastAPI | Async, fast, native Pydantic validation |
| Frontend | Streamlit | Minimal code for a chat UI prototype |
| Containerization | Docker + Docker Compose (local), Railway (production) | |
| Rate Limiting | slowapi | Per-IP, wraps FastAPI with minimal overhead |

---

## Requirements

### Functional

**API**
1. `POST /ask` — accepts `{ question: str, session_id: uuid }`, returns `{ answer, session_id, sources }`
2. `GET /health` — returns `{ status, qdrant, version }`
3. Multi-turn memory — last 20 messages per session, keyed by `session_id`
4. Grounded answers — responses cite Stack Overflow sources with title, question_id, score, and tags

**RAG Pipeline**
5. Retrieve top-5 relevant chunks from Qdrant for each question
6. Generate answers via Groq LLM with retrieved context injected into system prompt

**Dataset Indexing**
7. Index Stack Overflow Python Q&A dataset into Qdrant (one-time, run locally)
8. Document format: question title + body + tags + top answer (always) + up to 2 additional answers (score ≥ 10 only)
9. Idempotent — skip re-indexing if collection already has vectors (checks `vectors_count > 0`, not just collection existence)

**Frontend**
10. Streamlit chat UI with persistent conversation display
11. Source citations shown per assistant response, including tags

### Non-Functional

- All services containerised (Docker Compose for local, Railway for prod)
- Secrets in `.env`; template in `.env.example`
- Prompt templates versioned in `rag/prompts/v1.py`
- Rate limiting: 20 req/min per IP
- CORS restricted to `FRONTEND_URL`
- UUID validation on `session_id`
- Non-root Docker user (backend)

---

## Architecture

### Directory Structure

```
python_qna/
├── frontend/                   # Streamlit app
│   ├── app.py
│   ├── Dockerfile              # Railway (build context: frontend/)
│   ├── Dockerfile.local        # Docker Compose (build context: repo root)
│   ├── railway.json
│   └── requirements.txt
│
├── backend/                    # FastAPI service + RAG pipeline
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── routes/ask.py
│   │   └── schemas/chat.py
│   ├── rag/
│   │   ├── pipeline/
│   │   │   ├── graph.py        # LangGraph graph definition
│   │   │   ├── nodes.py        # retrieve + generate nodes
│   │   │   └── state.py        # GraphState TypedDict
│   │   ├── retriever/
│   │   │   └── qdrant_client.py
│   │   ├── prompts/
│   │   │   └── v1.py
│   │   ├── indexer/
│   │   │   ├── index_dataset.py
│   │   │   ├── chunk_formatter.py
│   │   │   ├── Dockerfile
│   │   │   └── requirements.txt
│   │   └── tests/
│   │       ├── test_indexer.py
│   │       ├── test_memory.py
│   │       ├── test_retriever.py
│   │       ├── test_eval_retrieval.py   # eval suite
│   │       └── test_eval_ragas.py       # eval suite
│   ├── tests/
│   │   ├── test_ask.py
│   │   └── test_health.py
│   ├── Dockerfile
│   ├── pytest.ini
│   ├── railway.json
│   └── requirements.txt
│
├── data/                       # .gitignored — sampled CSVs (1000 questions)
├── data_full/                  # .gitignored — full original dataset
├── scripts/
│   └── sample_dataset.py       # reproducible dataset sampling
├── docs/
├── docker-compose.yml
├── pytest.ini                  # root-level pytest config
├── .env                        # secrets — never committed
└── .env.example
```

### Services

**Local dev (Docker Compose):** Three containers, all connect to Qdrant Cloud via env vars.

| Service | Dockerfile | Port | Notes |
|---|---|---|---|
| `indexer` | `backend/rag/indexer/Dockerfile` | — | One-shot, `restart: no` |
| `backend` | `backend/Dockerfile` | 8000 | Data dir mounted for evals |
| `frontend` | `frontend/Dockerfile.local` | 8501 | `depends_on: backend` |

**Production (Railway):** Two services — `backend` and `frontend`. Qdrant Cloud is external.

```
Qdrant Cloud (free tier, external)
       │ HTTPS + API key
Railway: backend (PORT dynamic, private network)
       │ railway.internal private networking
Railway: frontend (port 8501, public domain)
```

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
- Extracts the latest `HumanMessage` from state
- Queries Qdrant via FastEmbed, returns top-5 chunks
- Returns `{"context": [...], "sources": [...]}` — sources include `tags`

### Node: `generate`
- Trims `messages` to last 20 before the LLM call
- Injects context into system prompt via `RETRIEVAL_CONTEXT_TEMPLATE`
- Calls Groq, returns `{"messages": [AIMessage(...)]}`

### Memory
- `MemorySaver` (in-process SQLite checkpointer) keyed by `thread_id = session_id`
- State survives across requests within the same container run; lost on restart (acceptable for demo)

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
  "sources": [
    {
      "question_title": "Python list comprehension",
      "question_id": 12345,
      "score": 42,
      "tags": ["python", "list-comprehension"]
    }
  ]
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

## Dataset & Indexing

### Dataset

The raw Stack Overflow dataset (~1.7 GB) has three CSVs: `Questions.csv`, `Answers.csv`, `Tags.csv`. The full dataset is stored in `data_full/` (gitignored).

**For development and deployment, a reproducible 1000-question sample is used:**

```bash
python scripts/sample_dataset.py --n 1000 --seed 42
# Outputs: data/Questions.csv (1000 rows), data/Answers.csv (1648 rows), data/Tags.csv (3122 rows)
```

This was necessary because loading the full dataset into the indexer container caused OOM kills (Docker Desktop default memory + embedding model + pandas DataFrames exceeded available RAM).

### Chunk Format

```
Question: {Title}
{Body (HTML stripped)}
Tags: python, pandas, numpy

[TOP ANSWER (score: N)]
{answer_1_body}

[ANSWER 2 (score: N)]     ← only if score ≥ 10
{answer_2_body}

[ANSWER 3 (score: N)]     ← only if score ≥ 10
{answer_3_body}
```

Tags are embedded in the chunk text (passive signal for semantic search) **and** stored separately in the Qdrant payload (for future filtering).

### Qdrant Payload per Point

```json
{
  "text": "<full formatted chunk>",
  "question_id": 12345,
  "title": "How to reverse a list?",
  "question_score": 42,
  "answer_count": 7,
  "tags": ["python", "list"]
}
```

### Indexer Config

| Parameter | Value | Reason |
|---|---|---|
| Embedding model | `BAAI/bge-small-en` | 384-dim, fast, runs locally |
| Batch size | 32 | Reduced from 256 to prevent OOM in containers |
| Distance metric | Cosine | Standard for text embeddings |
| Model pre-download | At Docker build time | Avoids a 77 MB runtime download that caused OOM spikes |
| Idempotency check | `vectors_count > 0` | Handles the case where a prior run created the collection but failed before uploading any vectors |

---

## Prompt Versioning

Prompts live in `rag/prompts/v1.py`. New versions create `v2.py`; old versions are kept. Active version controlled by `ACTIVE_PROMPT_VERSION` env var.

```python
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
cp .env.example .env
# Fill in GROQ_API_KEY, QDRANT_HOST, QDRANT_API_KEY, QDRANT_USE_TLS=true

# Sample dataset (or place full CSVs in data/)
python scripts/sample_dataset.py

# Index once (connects to Qdrant Cloud)
docker-compose run --rm indexer

# Start backend + frontend
docker-compose up --build backend frontend
```

### Railway Production

**One-time setup:**
1. Run indexer locally against the Qdrant Cloud cluster
2. Create Railway project with two services
3. `backend` service: root dir = `backend/`, Dockerfile = `Dockerfile`
4. `frontend` service: root dir = `frontend/`, Dockerfile = `Dockerfile`
5. Set env vars per service (see below)
6. Push → Railway auto-deploys

**Backend env vars:**
```
GROQ_API_KEY, QDRANT_HOST, QDRANT_PORT=6333, QDRANT_API_KEY, QDRANT_USE_TLS=true,
QDRANT_COLLECTION=python_qna, ACTIVE_PROMPT_VERSION=v1
```

**Frontend env vars:**
```
BACKEND_URL=https://<backend-public-domain>.up.railway.app
```

Note: `BACKEND_URL` must be the public Railway domain, not `railway.internal`, because Streamlit runs in the user's browser and cannot reach Railway's private network.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | **required** | Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model |
| `QDRANT_HOST` | `localhost` | Qdrant hostname (no `https://` prefix) |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_API_KEY` | _(empty)_ | Qdrant Cloud API key |
| `QDRANT_USE_TLS` | `false` | Set `true` for Qdrant Cloud |
| `QDRANT_COLLECTION` | `python_qna` | Collection name |
| `ACTIVE_PROMPT_VERSION` | `v1` | Prompt version |
| `BACKEND_URL` | `http://backend:8000` | Backend URL (used by Streamlit) |
| `FRONTEND_URL` | `http://localhost:8501` | Allowed CORS origin |
| `DATA_DIR` | `/app/data` (container) | Path to CSV dataset |
| `INDEX_LIMIT` | `0` (no limit) | Cap number of indexed questions (useful for testing) |

---

## Testing

### Unit Tests (no services required)
```bash
PYTHONPATH=backend pytest backend/tests/ backend/rag/tests/ -v -m "not integration and not eval"
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
PYTHONPATH=backend pytest -m integration -s
```

### Eval Tests (requires live Qdrant + indexed data; RAGAS also needs GROQ_API_KEY)
```bash
# From inside the backend container:
docker exec python_qna-backend-1 python -m pytest -m eval -s -v

# Retrieval only (no LLM calls, ~40s for 100 questions):
docker exec python_qna-backend-1 python -m pytest rag/tests/test_eval_retrieval.py -m eval -s -v

# RAGAS only (~5 min for 10 questions, uses ~8k Groq tokens):
docker exec python_qna-backend-1 python -m pytest rag/tests/test_eval_ragas.py -m eval -s -v
```

**Eval results (1000-question sample, seed=42):**

| Metric | Value | Threshold |
|---|---|---|
| Hit@1 | 0.960 | — |
| Hit@3 | 0.980 | — |
| Hit@5 | 1.000 | ≥ 0.50 ✓ |
| MRR | 0.973 | ≥ 0.30 ✓ |

RAGAS (faithfulness + answer relevancy thresholds ≥ 0.70) — run after Groq daily quota resets (100k tokens/day on free tier).

---

## Design Decisions

### Qdrant Cloud over self-hosted
Qdrant Cloud free tier provides a managed, persistent cluster accessible from both local Docker Compose and Railway. Self-hosting Qdrant in Docker Compose would work locally but not on Railway without an additional service and persistent volume, adding complexity. The collection only needs to be indexed once, making the managed option low-cost.

### Tags used for embedding enrichment, not payload filtering
`Tags.csv` maps question IDs to Stack Overflow tags (e.g., `python`, `pandas`, `django`). Tags are appended to each chunk's text before embedding (`Tags: python, pandas`) and stored in the Qdrant payload. This passively improves semantic search — a query about "dataframes" pulls toward `pandas`-tagged chunks without any extra logic.

Active tag-based payload filtering (extracting tags from user query, then using Qdrant `must: tag in [...]` pre-filter) was considered but deferred. The semantic approach covers most cases adequately, and adding auto-tag extraction would introduce latency and complexity for marginal gain at this stage.

### No hybrid retrieval or reranking
Pure dense vector search with `BAAI/bge-small-en` was chosen for the initial implementation. Hybrid search (dense + sparse BM25) and cross-encoder reranking (`BAAI/bge-reranker-base`) would improve precision on keyword-heavy queries but add significant infrastructure complexity (a sparse index, an additional model, a merge step). The eval results (Hit@5=1.0, MRR=0.97) showed retrieval quality is already excellent on the current dataset, making this a lower priority.

### No semantic response caching
Caching semantically similar queries in Redis (embed query → check cosine similarity → return cached response if above threshold) would reduce latency and LLM cost for repeated or near-duplicate questions. Deferred because it requires a Redis service and adds complexity to the retrieval path. Will be more valuable at higher query volumes.

### Dataset sampled to 1000 questions
The full Stack Overflow Python dataset (~600k questions, ~1.7 GB) caused OOM kills in the Docker indexer container due to the combined memory footprint of: loading all answers into a dict, loading all tags into a dict, downloading the 77 MB FastEmbed model, and running batch embeddings. A reproducible 1000-question sample (`scripts/sample_dataset.py`, seed=42) is used for the current deployment. The full dataset can be indexed by pointing at `data_full/` and setting `INDEX_LIMIT=0`.

### rag/ moved inside backend/
Originally `rag/` was at the repo root. Railway builds each service from its own root directory — the `backend` service builds from `backend/`, which means any `COPY rag/` in the Dockerfile would fail because `rag/` is outside the build context. Moving `rag/` inside `backend/` lets both the backend app and the indexer share a single build context without path gymnastics.

### Two Dockerfiles for frontend
Railway builds from `frontend/` as the root, so paths like `COPY frontend/requirements.txt` fail. Docker Compose builds from the repo root, so paths without the `frontend/` prefix fail. Two Dockerfiles solve this: `Dockerfile` (Railway, paths relative to `frontend/`) and `Dockerfile.local` (Docker Compose, paths relative to repo root).

### Railway PORT handling via shell form CMD
Railway injects the port as `$PORT` at runtime. `CMD ["uvicorn", ..., "--port", "${PORT:-8000}"]` (exec form) does not expand shell variables — `${PORT:-8000}` is passed literally to uvicorn. Using `CMD ["sh", "-c", "uvicorn ... --port ${PORT:-8000}"]` (shell form) runs through a shell that expands the variable. `startCommand` in `railway.json` was also tried but bypasses the shell entirely, causing the same issue.

### Frontend uses public backend domain, not railway.internal
Streamlit renders in the user's browser. `railway.internal` is Railway's private DNS, resolvable only from within Railway's internal network (i.e., server-side). Since Streamlit makes API calls from the browser (client-side), `BACKEND_URL` must be the public `https://<backend>.up.railway.app` domain.

### Groq model: llama-3.3-70b-versatile
The originally specified `llama-3.1-70b-versatile` was decommissioned by Groq during development. Migrated to `llama-3.3-70b-versatile` which is the current recommended replacement with comparable quality.

### Evals in the test suite, not standalone scripts
Retrieval and RAGAS quality evals are implemented as `@pytest.mark.eval` tests inside `backend/rag/tests/` rather than standalone scripts. This lets them run with the same `pytest` invocation as unit and integration tests, assert quality thresholds (so they fail CI if quality regresses), and be skipped automatically when services or API keys are unavailable. The `ragas` and `pandas` dependencies are included in `backend/requirements.txt` so evals run inside the existing backend Docker image without a separate install step.

### RAGAS eval limitations on Groq free tier
RAGAS generates multiple LLM judge calls per sample (faithfulness NLI decomposition + answer relevancy synthetic question generation). For 25 samples × 2 metrics × ~3 calls each, this consumed ~100k tokens — Groq's full daily limit on the free tier. The eval was reduced to 10 samples with context chunks truncated to 600 chars to stay within both the per-minute (12k TPM) and daily (100k TPD) limits. Scores below 50% valid (NaN from rate-limit errors) trigger a `pytest.skip` rather than a hard failure.

---

## Future Scope

### Retrieval & Quality
- **Tag-based filtering** — extract tags from user query, use as Qdrant payload pre-filter before vector search; or expose as a user-controlled dropdown in the frontend
- **Hybrid retrieval + reranking** — combine dense vector search with sparse BM25, then rerank with a cross-encoder (`BAAI/bge-reranker-base`) before passing context to the LLM
- **Semantic response caching** — embed incoming queries, store vector + response in Redis; return cached response for near-duplicate queries above cosine similarity threshold

### Evals
- **Context recall** — extend RAGAS suite with `ContextRecall` metric using Stack Overflow top-scored answer as ground truth
- **Answer correctness** — compare LLM answer against Stack Overflow accepted answer using an LLM judge
- **Adversarial faithfulness** — spawn N independent judge agents to refute each answer; flag where majority vote says hallucination
- **Latency profiling** — instrument retrieval and LLM stages separately; assert p95 retrieval < 500ms and p95 end-to-end < 5s

### Infrastructure & Scale
- **Full dataset indexing** — index all ~600k questions using `data_full/` once compute and memory constraints are addressed
- **Redis sessions** — replace in-process `MemorySaver` with Redis checkpointer for persistence across restarts and horizontal scaling
- **Message compaction** — summarize first 15 messages when count exceeds 20

### Features
- **Persistent chat history** — store conversation turns in IndexedDB (browser-side) for restore on next visit without backend changes
- **Authentication** — JWT-based auth (FastAPI + python-jose); Streamlit login gate; chat history tied to authenticated user
- **LangSmith observability** — wrap LangGraph pipeline with LangSmith tracing for per-node latency, token counts, and context visibility; enabled via `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY`
- **Router + RAG** — LLM router decides whether retrieval is needed or the question can be answered from history alone
- **Agent mode** — Qdrant retrieval as a callable tool, allowing re-query with refined search terms
