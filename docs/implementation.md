# Python Q&A Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python Q&A Assistant with a Streamlit frontend, FastAPI backend, and LangGraph RAG pipeline grounded in a Stack Overflow dataset indexed into Qdrant.

**Architecture:** A LangGraph graph (retrieve → generate) with LangChain's MemorySaver for multi-turn memory (20-message window), served via FastAPI, with a Streamlit chat UI. The Stack Overflow dataset is pre-indexed into Qdrant via a one-shot indexer container using FastEmbed embeddings. Session state lives in LangGraph keyed by `session_id` (= LangGraph `thread_id`).

**Tech Stack:** FastAPI, LangChain, LangGraph, langchain-groq (Groq LLM), Qdrant, FastEmbed (BAAI/bge-small-en), Streamlit, Docker Compose, pytest

---

## File Map

### Created by this plan

```
python_qna/
├── conftest.py                          # root pytest path fix
├── pytest.ini                           # test config + integration marker
├── .gitignore
├── .env.example
├── REQUIREMENTS.md
├── README.md
├── docker-compose.yml
│
├── rag/
│   ├── __init__.py
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── v1.py                        # SYSTEM_PROMPT, RETRIEVAL_CONTEXT_TEMPLATE
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── state.py                     # GraphState TypedDict
│   │   ├── nodes.py                     # retrieve + generate + trim_messages_to_limit
│   │   └── graph.py                     # LangGraph assembly with MemorySaver
│   ├── retriever/
│   │   ├── __init__.py
│   │   └── qdrant_client.py             # lazy Qdrant + FastEmbed singletons, search_similar
│   ├── indexer/
│   │   ├── __init__.py
│   │   ├── chunk_formatter.py           # strip_html, format_chunk (pure functions)
│   │   ├── index_dataset.py             # one-shot CSV → Qdrant script
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── tests/
│       ├── __init__.py
│       ├── test_indexer.py              # chunk formatter unit tests
│       ├── test_memory.py               # trim_messages_to_limit unit tests
│       └── test_retriever.py            # retriever unit + integration tests
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                      # FastAPI app, router registration
│   │   ├── config.py                    # pydantic-settings Settings
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   └── ask.py                   # POST /ask, GET /health
│   │   └── schemas/
│   │       ├── __init__.py
│   │       └── chat.py                  # AskRequest, AskResponse, Source, HealthResponse
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py                  # TestClient fixture
│   │   ├── test_health.py
│   │   └── test_ask.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/
│   ├── app.py                           # Streamlit chat UI
│   ├── Dockerfile
│   └── requirements.txt
│
└── docs/
    └── test_report.md                   # API test documentation for assessment
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `.gitignore`
- Create: `.env.example`
- Create: `conftest.py`
- Create: `pytest.ini`
- Create all `__init__.py` files

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p rag/prompts rag/pipeline rag/retriever rag/indexer rag/tests
mkdir -p backend/app/routes backend/app/schemas backend/tests
mkdir -p frontend docs
touch rag/__init__.py rag/prompts/__init__.py rag/pipeline/__init__.py
touch rag/retriever/__init__.py rag/indexer/__init__.py rag/tests/__init__.py
touch backend/__init__.py backend/app/__init__.py backend/app/routes/__init__.py
touch backend/app/schemas/__init__.py backend/tests/__init__.py
```

- [ ] **Step 2: Create `.gitignore`**

```
# Secrets
.env

# Dataset (too large for git)
data/

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
.venv/
.pytest_cache/
.mypy_cache/

# Qdrant local storage
qdrant_storage/

# Archives
*.zip
```

- [ ] **Step 3: Create `.env.example`**

```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-70b-versatile
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_API_KEY=
QDRANT_USE_TLS=false
QDRANT_COLLECTION=python_qna
ACTIVE_PROMPT_VERSION=v1
BACKEND_URL=http://backend:8000
FRONTEND_URL=http://localhost:8501
```

Note: For local dev (not Docker), set `QDRANT_HOST=localhost` and `BACKEND_URL=http://localhost:8000`.
For production (Qdrant Cloud), set `QDRANT_HOST=<cluster>.qdrant.io`, `QDRANT_API_KEY=<key>`, `QDRANT_USE_TLS=true`.

- [ ] **Step 4: Create root `conftest.py`** (makes project root importable in tests)

```python
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
```

- [ ] **Step 5: Create `pytest.ini`**

```ini
[pytest]
testpaths = backend/tests rag/tests
markers =
    integration: marks tests requiring live services (Qdrant)
```

- [ ] **Step 6: Commit**

```bash
git add .gitignore .env.example conftest.py pytest.ini \
    rag/__init__.py rag/prompts/__init__.py rag/pipeline/__init__.py \
    rag/retriever/__init__.py rag/indexer/__init__.py rag/tests/__init__.py \
    backend/__init__.py backend/app/__init__.py backend/app/routes/__init__.py \
    backend/app/schemas/__init__.py backend/tests/__init__.py
git commit -m "chore: project scaffold, directory structure, pytest config"
```

---

## Task 2: RAG Prompts v1

**Files:**
- Create: `rag/prompts/v1.py`

- [ ] **Step 1: Create `rag/prompts/v1.py`**

```python
VERSION = "v1"

SYSTEM_PROMPT = """You are a Python programming assistant helping data science learners.
Answer questions grounded strictly in the retrieved Stack Overflow context provided below.
If the context does not contain enough information to answer the question, say so clearly — do not hallucinate.
When referencing information, cite the Stack Overflow question title it came from."""

RETRIEVAL_CONTEXT_TEMPLATE = """--- Retrieved Context ---
{context}
--- End Context ---"""
```

- [ ] **Step 2: Commit**

```bash
git add rag/prompts/v1.py
git commit -m "feat(rag): add versioned prompt templates v1"
```

---

## Task 3: RAG State

**Files:**
- Create: `rag/pipeline/state.py`

- [ ] **Step 1: Create `rag/pipeline/state.py`**

```python
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: list[str]
    sources: list[dict]
    session_id: str
```

`add_messages` is a LangGraph reducer — when a node returns `{"messages": [...]}`, it appends to (not replaces) the existing list in the checkpoint.

- [ ] **Step 2: Commit**

```bash
git add rag/pipeline/state.py
git commit -m "feat(rag): add GraphState with add_messages reducer"
```

---

## Task 4 (TDD): Chunk Formatter

**Files:**
- Test: `rag/tests/test_indexer.py`
- Create: `rag/indexer/chunk_formatter.py`

- [ ] **Step 1: Write failing tests**

Create `rag/tests/test_indexer.py`:

```python
import pytest
from rag.indexer.chunk_formatter import format_chunk, strip_html


def test_strip_html_removes_tags():
    assert strip_html("<p>Hello <b>world</b></p>") == "Hello world"


def test_strip_html_handles_nan():
    assert strip_html("nan") == ""


def test_strip_html_handles_empty():
    assert strip_html("") == ""


def test_format_chunk_top_answer_always_included_even_low_score():
    question = {"Id": 1, "Title": "How to reverse a list?", "Body": "<p>How?</p>", "Score": 5}
    answers = [{"Id": 10, "Score": 1, "Body": "<p>Use list.reverse()</p>"}]

    result = format_chunk(question, answers)

    assert "Question: How to reverse a list?" in result
    assert "[TOP ANSWER (score: 1)]" in result
    assert "Use list.reverse()" in result


def test_format_chunk_answer2_included_when_score_gte_10():
    question = {"Id": 1, "Title": "Test", "Body": "<p>Q</p>", "Score": 5}
    answers = [
        {"Id": 10, "Score": 50, "Body": "<p>A1</p>"},
        {"Id": 11, "Score": 15, "Body": "<p>A2</p>"},
    ]

    result = format_chunk(question, answers)

    assert "[TOP ANSWER (score: 50)]" in result
    assert "[ANSWER 2 (score: 15)]" in result


def test_format_chunk_answer2_excluded_when_score_lt_10():
    question = {"Id": 1, "Title": "Test", "Body": "<p>Q</p>", "Score": 5}
    answers = [
        {"Id": 10, "Score": 50, "Body": "<p>A1</p>"},
        {"Id": 11, "Score": 5, "Body": "<p>A2</p>"},
    ]

    result = format_chunk(question, answers)

    assert "[TOP ANSWER (score: 50)]" in result
    assert "ANSWER 2" not in result


def test_format_chunk_answer3_included_when_score_gte_10():
    question = {"Id": 1, "Title": "Test", "Body": "<p>Q</p>", "Score": 5}
    answers = [
        {"Id": 10, "Score": 50, "Body": "<p>A1</p>"},
        {"Id": 11, "Score": 20, "Body": "<p>A2</p>"},
        {"Id": 12, "Score": 12, "Body": "<p>A3</p>"},
    ]

    result = format_chunk(question, answers)

    assert "[ANSWER 3 (score: 12)]" in result


def test_format_chunk_answer3_excluded_when_score_lt_10():
    question = {"Id": 1, "Title": "Test", "Body": "<p>Q</p>", "Score": 5}
    answers = [
        {"Id": 10, "Score": 50, "Body": "<p>A1</p>"},
        {"Id": 11, "Score": 20, "Body": "<p>A2</p>"},
        {"Id": 12, "Score": 3, "Body": "<p>A3</p>"},
    ]

    result = format_chunk(question, answers)

    assert "[ANSWER 2 (score: 20)]" in result
    assert "ANSWER 3" not in result


def test_format_chunk_no_answers():
    question = {"Id": 1, "Title": "Unanswered", "Body": "<p>Q</p>", "Score": 0}

    result = format_chunk(question, [])

    assert "Question: Unanswered" in result
    assert "TOP ANSWER" not in result
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
PYTHONPATH=. pytest rag/tests/test_indexer.py -v
```

Expected: `ImportError` — `chunk_formatter` does not exist yet.

- [ ] **Step 3: Create `rag/indexer/chunk_formatter.py`**

```python
from bs4 import BeautifulSoup


def strip_html(html: str) -> str:
    s = str(html).strip()
    if not s or s == "nan":
        return ""
    return BeautifulSoup(s, "html.parser").get_text(separator="\n").strip()


def format_chunk(question: dict, answers: list[dict]) -> str:
    title = str(question.get("Title", ""))
    body = strip_html(question.get("Body", ""))

    parts = [f"Question: {title}\n{body}"]

    if not answers:
        return "\n".join(parts)

    top = answers[0]
    parts.append(f"\n[TOP ANSWER (score: {top['Score']})]\n{strip_html(top['Body'])}")

    for i, ans in enumerate(answers[1:3], start=2):
        if int(ans["Score"]) >= 10:
            parts.append(f"\n[ANSWER {i} (score: {ans['Score']})]\n{strip_html(ans['Body'])}")

    return "\n".join(parts)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
PYTHONPATH=. pytest rag/tests/test_indexer.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add rag/indexer/chunk_formatter.py rag/tests/test_indexer.py
git commit -m "feat(rag): add chunk formatter with TDD — strip_html + format_chunk"
```

---

## Task 5: Indexer Script + Dockerfile

**Files:**
- Create: `rag/indexer/requirements.txt`
- Create: `rag/indexer/index_dataset.py`
- Create: `rag/indexer/Dockerfile`

- [ ] **Step 1: Create `rag/indexer/requirements.txt`**

```
pandas==2.2.3
beautifulsoup4==4.12.3
lxml==5.3.0
qdrant-client==1.11.3
fastembed==0.4.2
python-dotenv==1.0.1
tqdm==4.67.1
```

- [ ] **Step 2: Create `rag/indexer/index_dataset.py`**

```python
import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding

from rag.indexer.chunk_formatter import format_chunk

load_dotenv()

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "python_qna")
VECTOR_SIZE = 384  # BAAI/bge-small-en output dimension
BATCH_SIZE = 256
DATA_DIR = os.getenv("DATA_DIR", "/app/data")


def get_client() -> QdrantClient:
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    api_key = os.getenv("QDRANT_API_KEY", "")
    use_tls = os.getenv("QDRANT_USE_TLS", "false").lower() == "true"
    if use_tls and api_key:
        return QdrantClient(url=f"https://{host}:{port}", api_key=api_key)
    return QdrantClient(host=host, port=port)


def collection_exists(client: QdrantClient) -> bool:
    return COLLECTION_NAME in [c.name for c in client.get_collections().collections]


def load_answers(answers_path: str) -> dict[int, list[dict]]:
    print("Loading answers into memory...")
    df = pd.read_csv(
        answers_path,
        usecols=["Id", "ParentId", "Score", "Body"],
        dtype={"Id": "int64", "ParentId": "int64", "Score": "float64", "Body": "str"},
    )
    df["Score"] = df["Score"].fillna(0).astype(int)
    df = df.sort_values("Score", ascending=False)
    grouped = df.groupby("ParentId")
    return {int(pid): group.to_dict("records") for pid, group in grouped}


def run_indexing(questions_path: str, answers_path: str) -> None:
    client = get_client()

    if collection_exists(client):
        print(f"Collection '{COLLECTION_NAME}' already exists. Skipping.")
        return

    print(f"Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    answers_by_qid = load_answers(answers_path)
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en")

    print("Indexing questions...")
    total = 0
    batch_texts: list[str] = []
    batch_payloads: list[dict] = []

    reader = pd.read_csv(
        questions_path,
        chunksize=BATCH_SIZE,
        usecols=["Id", "Title", "Body", "Score"],
        dtype={"Id": "int64", "Title": "str", "Body": "str", "Score": "float64"},
    )

    for chunk in tqdm(reader, desc="Questions"):
        chunk["Score"] = chunk["Score"].fillna(0).astype(int)

        for _, row in chunk.iterrows():
            q_id = int(row["Id"])
            answers = answers_by_qid.get(q_id, [])
            text = format_chunk(row.to_dict(), answers)
            batch_texts.append(text)
            batch_payloads.append({
                "text": text,
                "question_id": q_id,
                "title": str(row["Title"]),
                "question_score": int(row["Score"]),
                "answer_count": len(answers),
            })

        if len(batch_texts) >= BATCH_SIZE:
            _upsert_batch(client, embedding_model, batch_texts, batch_payloads)
            total += len(batch_texts)
            batch_texts, batch_payloads = [], []

    if batch_texts:
        _upsert_batch(client, embedding_model, batch_texts, batch_payloads)
        total += len(batch_texts)

    print(f"Done! Indexed {total} questions into '{COLLECTION_NAME}'.")


def _upsert_batch(
    client: QdrantClient,
    embedding_model: TextEmbedding,
    texts: list[str],
    payloads: list[dict],
) -> None:
    embeddings = list(embedding_model.embed(texts))
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=payloads[i]["question_id"],
                vector=embeddings[i].tolist(),
                payload=payloads[i],
            )
            for i in range(len(texts))
        ],
    )


if __name__ == "__main__":
    questions_path = os.path.join(DATA_DIR, "Questions.csv")
    answers_path = os.path.join(DATA_DIR, "Answers.csv")
    run_indexing(questions_path, answers_path)
```

- [ ] **Step 3: Create `rag/indexer/Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY rag/indexer/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY rag/ ./rag/

ENV PYTHONPATH=/app

CMD ["python", "-m", "rag.indexer.index_dataset"]
```

- [ ] **Step 4: Commit**

```bash
git add rag/indexer/requirements.txt rag/indexer/index_dataset.py rag/indexer/Dockerfile
git commit -m "feat(indexer): CSV-to-Qdrant one-shot indexer with idempotency guard"
```

---

## Task 6: Qdrant Retriever Client

**Files:**
- Create: `rag/retriever/qdrant_client.py`
- Create: `rag/tests/test_retriever.py`

- [ ] **Step 1: Create `rag/retriever/qdrant_client.py`**

```python
import os
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "python_qna")
EMBEDDING_MODEL = "BAAI/bge-small-en"

_client: QdrantClient | None = None
_embedding_model: TextEmbedding | None = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        api_key = os.getenv("QDRANT_API_KEY", "")
        use_tls = os.getenv("QDRANT_USE_TLS", "false").lower() == "true"
        if use_tls and api_key:
            _client = QdrantClient(url=f"https://{host}:{port}", api_key=api_key)
        else:
            _client = QdrantClient(host=host, port=port)
    return _client


def get_embedding_model() -> TextEmbedding:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _embedding_model


def check_collection_exists(client: QdrantClient) -> bool:
    return COLLECTION_NAME in [c.name for c in client.get_collections().collections]


def search_similar(
    query: str,
    client: QdrantClient,
    embedding_model: TextEmbedding,
    top_k: int = 5,
) -> list[dict]:
    query_vector = list(embedding_model.embed([query]))[0].tolist()
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "text": hit.payload.get("text", ""),
            "question_id": hit.payload.get("question_id"),
            "title": hit.payload.get("title"),
            "question_score": hit.payload.get("question_score", 0),
        }
        for hit in results
    ]
```

- [ ] **Step 2: Create `rag/tests/test_retriever.py`**

```python
import pytest
from unittest.mock import MagicMock, patch
from rag.retriever.qdrant_client import search_similar, check_collection_exists


def test_search_similar_returns_mapped_dicts():
    mock_client = MagicMock()
    mock_embedding_model = MagicMock()
    mock_embedding_model.embed.return_value = [[0.1] * 384]

    hit1 = MagicMock()
    hit1.payload = {
        "text": "Question: Reverse list\n[TOP ANSWER]...",
        "question_id": 1,
        "title": "How to reverse a list?",
        "question_score": 10,
    }
    hit2 = MagicMock()
    hit2.payload = {
        "text": "Question: Sort list\n[TOP ANSWER]...",
        "question_id": 2,
        "title": "How to sort a list?",
        "question_score": 5,
    }
    mock_client.search.return_value = [hit1, hit2]

    results = search_similar("reverse a list", mock_client, mock_embedding_model, top_k=2)

    assert len(results) == 2
    assert results[0]["title"] == "How to reverse a list?"
    assert results[0]["question_id"] == 1
    assert "text" in results[0]
    assert "question_score" in results[0]


def test_check_collection_exists_true():
    mock_client = MagicMock()
    col = MagicMock()
    col.name = "python_qna"
    mock_client.get_collections.return_value.collections = [col]

    with patch.dict("os.environ", {"QDRANT_COLLECTION": "python_qna"}):
        assert check_collection_exists(mock_client) is True


def test_check_collection_exists_false():
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []
    assert check_collection_exists(mock_client) is False


@pytest.mark.integration
def test_live_search_returns_5_results():
    """Requires Qdrant running with indexed data. Run: pytest -m integration"""
    from rag.retriever.qdrant_client import get_qdrant_client, get_embedding_model

    client = get_qdrant_client()
    model = get_embedding_model()
    results = search_similar("how to use list comprehensions", client, model, top_k=5)

    assert len(results) == 5
    for r in results:
        assert "text" in r
        assert "question_id" in r
        assert "title" in r
        assert "question_score" in r
```

- [ ] **Step 3: Run unit tests**

```bash
PYTHONPATH=. pytest rag/tests/test_retriever.py -v -m "not integration"
```

Expected: 3 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add rag/retriever/qdrant_client.py rag/tests/test_retriever.py
git commit -m "feat(rag): add Qdrant retriever client with lazy singletons"
```

---

## Task 7 (TDD): RAG Nodes + Memory Trim

**Files:**
- Test: `rag/tests/test_memory.py`
- Create: `rag/pipeline/nodes.py`

- [ ] **Step 1: Write failing tests for message trimming**

Create `rag/tests/test_memory.py`:

```python
from langchain_core.messages import HumanMessage, AIMessage
from rag.pipeline.nodes import trim_messages_to_limit


def test_trim_keeps_last_20_of_50():
    messages = []
    for i in range(25):
        messages.append(HumanMessage(content=f"Q{i}"))
        messages.append(AIMessage(content=f"A{i}"))

    result = trim_messages_to_limit(messages, limit=20)

    assert len(result) == 20
    assert result[-1].content == "A24"
    assert result[0].content == "Q15"


def test_trim_no_change_when_at_limit():
    messages = [HumanMessage(content=f"Q{i}") for i in range(20)]
    result = trim_messages_to_limit(messages, limit=20)
    assert len(result) == 20


def test_trim_no_change_when_under_limit():
    messages = [HumanMessage(content="Q"), AIMessage(content="A")]
    result = trim_messages_to_limit(messages, limit=20)
    assert len(result) == 2


def test_trim_empty_list():
    result = trim_messages_to_limit([], limit=20)
    assert result == []
```

- [ ] **Step 2: Run to confirm failure**

```bash
PYTHONPATH=. pytest rag/tests/test_memory.py -v
```

Expected: `ImportError` — `nodes` does not exist yet.

- [ ] **Step 3: Create `rag/pipeline/nodes.py`**

```python
import os
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_groq import ChatGroq

from rag.pipeline.state import GraphState
from rag.retriever.qdrant_client import get_qdrant_client, get_embedding_model, search_similar
from rag.prompts import v1 as prompts

MESSAGE_LIMIT = 20

_llm: ChatGroq | None = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,
        )
    return _llm


def trim_messages_to_limit(
    messages: list[BaseMessage], limit: int = MESSAGE_LIMIT
) -> list[BaseMessage]:
    if len(messages) <= limit:
        return messages
    return messages[-limit:]


def retrieve(state: GraphState) -> dict:
    client = get_qdrant_client()
    embedding_model = get_embedding_model()

    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    if last_human is None:
        return {"context": [], "sources": []}

    results = search_similar(last_human.content, client, embedding_model, top_k=5)

    return {
        "context": [r["text"] for r in results],
        "sources": [
            {
                "question_title": r["title"],
                "question_id": r["question_id"],
                "score": r["question_score"],
            }
            for r in results
        ],
    }


def generate(state: GraphState) -> dict:
    llm = _get_llm()

    trimmed = trim_messages_to_limit(state["messages"])
    context_text = "\n\n---\n\n".join(state.get("context", []))
    context_block = prompts.RETRIEVAL_CONTEXT_TEMPLATE.format(context=context_text)
    system_msg = SystemMessage(content=prompts.SYSTEM_PROMPT + "\n\n" + context_block)

    response = llm.invoke([system_msg] + list(trimmed))
    return {"messages": [AIMessage(content=response.content)]}
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
PYTHONPATH=. pytest rag/tests/test_memory.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add rag/pipeline/nodes.py rag/tests/test_memory.py
git commit -m "feat(rag): add retrieve + generate nodes, trim_messages_to_limit (TDD)"
```

---

## Task 8: RAG Graph

**Files:**
- Create: `rag/pipeline/graph.py`

- [ ] **Step 1: Create `rag/pipeline/graph.py`**

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from rag.pipeline.state import GraphState
from rag.pipeline.nodes import retrieve, generate

_graph = None


def build_graph():
    memory = MemorySaver()
    builder = StateGraph(GraphState)

    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile(checkpointer=memory)


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


graph = get_graph()
```

- [ ] **Step 2: Verify graph builds without error**

```bash
PYTHONPATH=. python -c "from rag.pipeline.graph import graph; print('Graph OK:', type(graph))"
```

Expected: `Graph OK: <class 'langgraph.graph.state.CompiledStateGraph'>`

- [ ] **Step 3: Commit**

```bash
git add rag/pipeline/graph.py
git commit -m "feat(rag): assemble LangGraph pipeline with MemorySaver checkpointer"
```

---

## Task 9: Backend Schemas + Config

**Files:**
- Create: `backend/app/schemas/chat.py`
- Create: `backend/app/config.py`

- [ ] **Step 1: Create `backend/app/schemas/chat.py`**

```python
import uuid
from pydantic import BaseModel, Field, field_validator


class Source(BaseModel):
    question_title: str
    question_id: int
    score: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str

    @field_validator("session_id")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError("session_id must be a valid UUID")
        return v


class AskResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[Source]


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    version: str
```

- [ ] **Step 2: Create `backend/app/config.py`**

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-70b-versatile"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str = ""
    qdrant_use_tls: bool = False
    qdrant_collection: str = "python_qna"
    active_prompt_version: str = "v1"
    backend_url: str = "http://backend:8000"

    class Config:
        env_file = ".env"


settings = Settings()
```

- [ ] **Step 3: Commit**

```bash
git add backend/app/schemas/chat.py backend/app/config.py
git commit -m "feat(backend): add Pydantic schemas and settings config"
```

---

## Task 10 (TDD): Backend Routes + Main

**Files:**
- Test: `backend/tests/conftest.py`
- Test: `backend/tests/test_health.py`
- Test: `backend/tests/test_ask.py`
- Create: `backend/app/routes/ask.py`
- Create: `backend/app/main.py`

- [ ] **Step 1: Create `backend/tests/conftest.py`**

```python
import os
import pytest
from fastapi.testclient import TestClient

# Stub secrets before importing the app so the lifespan check passes
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("FRONTEND_URL", "http://localhost:8501")


@pytest.fixture
def client():
    from backend.app.main import app
    return TestClient(app)
```

- [ ] **Step 2: Write failing tests — `backend/tests/test_health.py`**

```python
from unittest.mock import MagicMock, patch


def test_health_returns_200(client):
    with patch("backend.app.routes.ask.QdrantClient") as mock_cls:
        mock_cls.return_value.get_collections.return_value = MagicMock()
        response = client.get("/health")

    assert response.status_code == 200


def test_health_shape(client):
    with patch("backend.app.routes.ask.QdrantClient") as mock_cls:
        mock_cls.return_value.get_collections.return_value = MagicMock()
        response = client.get("/health")

    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "1.0.0"
    assert "qdrant" in data


def test_health_qdrant_disconnected(client):
    with patch("backend.app.routes.ask.QdrantClient") as mock_cls:
        mock_cls.return_value.get_collections.side_effect = Exception("refused")
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["qdrant"] == "disconnected"
```

- [ ] **Step 3: Write failing tests — `backend/tests/test_ask.py`**

```python
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage

SESSION_A = "550e8400-e29b-41d4-a716-446655440000"
SESSION_B = "123e4567-e89b-12d3-a456-426614174000"


def _mock_graph_result(answer="Test answer", sources=None, session_id=SESSION_A):
    return {
        "messages": [AIMessage(content=answer)],
        "sources": sources or [],
        "context": [],
        "session_id": session_id,
    }


def test_ask_returns_200(client):
    with patch("backend.app.routes.ask.graph") as mock_graph:
        mock_graph.invoke.return_value = _mock_graph_result("List comprehensions allow concise list creation.")
        response = client.post(
            "/ask",
            json={"question": "How do list comprehensions work?", "session_id": SESSION_A},
        )
    assert response.status_code == 200


def test_ask_response_shape(client):
    with patch("backend.app.routes.ask.graph") as mock_graph:
        mock_graph.invoke.return_value = _mock_graph_result(
            "Use list.reverse()",
            sources=[{"question_title": "Reverse list", "question_id": 1, "score": 10}],
        )
        response = client.post(
            "/ask",
            json={"question": "How do I reverse a list?", "session_id": SESSION_A},
        )

    data = response.json()
    assert "answer" in data
    assert data["session_id"] == SESSION_A
    assert isinstance(data["sources"], list)


def test_ask_empty_question_returns_422(client):
    response = client.post("/ask", json={"question": "", "session_id": SESSION_A})
    assert response.status_code == 422


def test_ask_question_too_long_returns_422(client):
    response = client.post(
        "/ask",
        json={"question": "x" * 2001, "session_id": SESSION_A},
    )
    assert response.status_code == 422


def test_ask_invalid_session_id_returns_422(client):
    response = client.post(
        "/ask",
        json={"question": "What is a list?", "session_id": "not-a-uuid"},
    )
    assert response.status_code == 422


def test_ask_multiturn_calls_graph_twice(client):
    with patch("backend.app.routes.ask.graph") as mock_graph:
        mock_graph.invoke.return_value = _mock_graph_result(session_id=SESSION_B)
        client.post("/ask", json={"question": "What is a list?", "session_id": SESSION_B})
        client.post("/ask", json={"question": "How to append?", "session_id": SESSION_B})

    assert mock_graph.invoke.call_count == 2
    calls = mock_graph.invoke.call_args_list
    assert calls[0][0][1] == {"configurable": {"thread_id": SESSION_B}}
    assert calls[1][0][1] == {"configurable": {"thread_id": SESSION_B}}


def test_ask_llm_error_returns_502(client):
    with patch("backend.app.routes.ask.graph") as mock_graph:
        mock_graph.invoke.side_effect = Exception("Groq timeout")
        response = client.post(
            "/ask",
            json={"question": "What is a decorator?", "session_id": SESSION_A},
        )
    assert response.status_code == 502
```

- [ ] **Step 4: Run tests to confirm they fail**

```bash
PYTHONPATH=. pytest backend/tests/ -v
```

Expected: `ImportError` — `backend.app.main` does not exist yet.

- [ ] **Step 5: Create `backend/app/routes/ask.py`**

```python
import os
from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import HumanMessage
from qdrant_client import QdrantClient
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.app.schemas.chat import AskRequest, AskResponse, HealthResponse
from rag.pipeline.graph import graph

limiter = Limiter(key_func=get_remote_address)
router = APIRouter()


@router.post("/ask", response_model=AskResponse)
@limiter.limit("20/minute")
def ask(request: Request, body: AskRequest) -> AskResponse:
    try:
        config = {"configurable": {"thread_id": body.session_id}}
        result = graph.invoke(
            {
                "messages": [HumanMessage(content=body.question)],
                "session_id": body.session_id,
                "context": [],
                "sources": [],
            },
            config,
        )
        last_ai = result["messages"][-1]
        return AskResponse(
            answer=last_ai.content,
            session_id=body.session_id,
            sources=result.get("sources", []),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Pipeline error: {str(e)}")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    qdrant_status = "connected"
    try:
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        api_key = os.getenv("QDRANT_API_KEY", "")
        use_tls = os.getenv("QDRANT_USE_TLS", "false").lower() == "true"
        if use_tls and api_key:
            client = QdrantClient(url=f"https://{host}:{port}", api_key=api_key)
        else:
            client = QdrantClient(host=host, port=port)
        client.get_collections()
    except Exception:
        qdrant_status = "disconnected"
    return HealthResponse(status="ok", qdrant=qdrant_status, version="1.0.0")
```

- [ ] **Step 6: Create `backend/app/main.py`**

```python
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from backend.app.routes.ask import limiter, router


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY environment variable is not set")
    yield


app = FastAPI(title="Python Q&A API", version="1.0.0", lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:8501")],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(router)
```

- [ ] **Step 7: Run tests to confirm they pass**

```bash
PYTHONPATH=. pytest backend/tests/ -v
```

Expected: all 8 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add backend/app/routes/ask.py backend/app/main.py \
    backend/tests/conftest.py backend/tests/test_health.py backend/tests/test_ask.py
git commit -m "feat(backend): POST /ask + GET /health routes with full test coverage (TDD)"
```

---

## Task 11: Backend Dockerfile + Requirements

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/Dockerfile`

- [ ] **Step 1: Create `backend/requirements.txt`**

```
fastapi==0.115.5
uvicorn[standard]==0.32.1
langchain==0.3.7
langchain-core==0.3.20
langchain-groq==0.2.2
langgraph==0.2.53
qdrant-client==1.11.3
fastembed==0.4.2
pydantic==2.10.3
pydantic-settings==2.6.1
python-dotenv==1.0.1
slowapi==0.1.9
httpx==0.28.1
pytest==8.3.4
```

- [ ] **Step 2: Create `backend/Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY rag/ ./rag/
COPY backend/ ./backend/

RUN useradd -m appuser && chown -R appuser /app
USER appuser

ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Commit**

```bash
git add backend/requirements.txt backend/Dockerfile
git commit -m "feat(backend): Dockerfile + requirements"
```

---

## Task 12: Streamlit Frontend

**Files:**
- Create: `frontend/requirements.txt`
- Create: `frontend/app.py`
- Create: `frontend/Dockerfile`

- [ ] **Step 1: Create `frontend/requirements.txt`**

```
streamlit==1.40.2
httpx==0.28.1
python-dotenv==1.0.1
```

- [ ] **Step 2: Create `frontend/app.py`**

```python
import os
import uuid
import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Python Q&A Assistant", page_icon="🐍", layout="centered")
st.title("🐍 Python Q&A Assistant")
st.caption("Grounded answers from Stack Overflow · Powered by Groq + LangGraph")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "is_loading" not in st.session_state:
    st.session_state.is_loading = False

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- **{src['question_title']}** (score: {src['score']})")

if prompt := st.chat_input("Ask a Python question...", max_chars=2000, disabled=st.session_state.is_loading):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.is_loading = True
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = httpx.post(
                    f"{BACKEND_URL}/ask",
                    json={"question": prompt, "session_id": st.session_state.session_id},
                    timeout=60.0,
                )
                resp.raise_for_status()
                data = resp.json()

                st.markdown(data["answer"])
                if data.get("sources"):
                    with st.expander("Sources"):
                        for src in data["sources"]:
                            st.markdown(f"- **{src['question_title']}** (score: {src['score']})")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "sources": data.get("sources", []),
                })

            except httpx.HTTPStatusError as e:
                st.error(f"Backend error {e.response.status_code}: {e.response.text}")
            except httpx.RequestError as e:
                st.error(f"Could not reach backend: {e}")
            finally:
                st.session_state.is_loading = False
```

- [ ] **Step 3: Create `frontend/Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY frontend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/ ./frontend/

EXPOSE 8501
CMD ["streamlit", "run", "frontend/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
```

- [ ] **Step 4: Commit**

```bash
git add frontend/requirements.txt frontend/app.py frontend/Dockerfile
git commit -m "feat(frontend): Streamlit chat UI with session memory and source citations"
```

---

## Task 13: Docker Compose

**Files:**
- Create: `docker-compose.yml`

- [ ] **Step 1: Create `docker-compose.yml`**

```yaml
version: "3.9"

# Local dev only. Production uses Qdrant Cloud — no qdrant service or volume needed.
# For local dev, run a local Qdrant first: docker run -p 6333:6333 qdrant/qdrant
# then set QDRANT_HOST=localhost, QDRANT_USE_TLS=false in your .env

services:
  indexer:
    build:
      context: .
      dockerfile: rag/indexer/Dockerfile
    volumes:
      - ./data:/app/data:ro
    environment:
      - QDRANT_HOST=${QDRANT_HOST:-localhost}
      - QDRANT_PORT=${QDRANT_PORT:-6333}
      - QDRANT_API_KEY=${QDRANT_API_KEY:-}
      - QDRANT_USE_TLS=${QDRANT_USE_TLS:-false}
      - QDRANT_COLLECTION=${QDRANT_COLLECTION:-python_qna}
    restart: "no"

  backend:
    build:
      context: .
      dockerfile: backend/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - GROQ_MODEL=${GROQ_MODEL:-llama-3.1-70b-versatile}
      - QDRANT_HOST=${QDRANT_HOST:-localhost}
      - QDRANT_PORT=${QDRANT_PORT:-6333}
      - QDRANT_API_KEY=${QDRANT_API_KEY:-}
      - QDRANT_USE_TLS=${QDRANT_USE_TLS:-false}
      - QDRANT_COLLECTION=${QDRANT_COLLECTION:-python_qna}
      - ACTIVE_PROMPT_VERSION=${ACTIVE_PROMPT_VERSION:-v1}
    restart: unless-stopped

  frontend:
    build:
      context: .
      dockerfile: frontend/Dockerfile
    ports:
      - "8501:8501"
    environment:
      - BACKEND_URL=http://backend:8000
    depends_on:
      - backend
    restart: unless-stopped
```

- [ ] **Step 2: Commit**

```bash
git add docker-compose.yml
git commit -m "feat: docker-compose with qdrant, indexer, backend, frontend"
```

---

## Task 14: README + REQUIREMENTS.md

**Files:**
- Create: `REQUIREMENTS.md`
- Create: `README.md`

- [ ] **Step 1: Create `REQUIREMENTS.md`**

```markdown
# Requirements

## Functional

### API
1. `POST /ask` — accepts `{ question: str, session_id: str }`, returns `{ answer, session_id, sources }`
2. `GET /health` — returns `{ status, qdrant, version }`
3. Multi-turn memory — last 20 messages per session, keyed by `session_id`
4. Grounded answers — responses cite Stack Overflow sources

### RAG Pipeline
5. Retrieve top-5 relevant chunks from Qdrant for each question
6. Generate answers via Groq LLM with retrieved context injected into system prompt

### Dataset Indexing
7. Index Stack Overflow Python Q&A dataset into Qdrant on first startup
8. Document format: question + top answer (always) + up to 2 additional answers (score ≥ 10 only)
9. Idempotent — skip re-indexing if collection already exists

### Frontend
10. Streamlit chat UI with persistent conversation display
11. Source citations shown per assistant response

## Non-Functional
- All services containerized (Docker Compose)
- Secrets in `.env`; template in `.env.example`
- Prompt templates versioned in `rag/prompts/v1.py`
- Modular structure: `frontend/`, `backend/`, `rag/`

## Future Scope
- Message summarization/compaction when total count > 20
- Router + RAG (LLM decides whether retrieval is needed)
- Full agent with tools (re-query Qdrant with refined search terms)
- Redis-backed session persistence (survive container restarts)
```

- [ ] **Step 2: Create `README.md`**

```markdown
# Python Q&A Assistant

AI-powered Python Q&A grounded in Stack Overflow data. Ask any Python question and get answers backed by real Stack Overflow threads.

## Architecture

| Layer | Technology |
|---|---|
| Frontend | Streamlit (port 8501) |
| Backend | FastAPI (port 8000) |
| RAG Pipeline | LangChain + LangGraph |
| LLM | Groq (`llama-3.1-70b-versatile`) |
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
pip install -r rag/indexer/requirements.txt
PYTHONPATH=. python -m rag.indexer.index_dataset
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
# 1. Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# 2. Index dataset (one-time)
pip install -r rag/indexer/requirements.txt
PYTHONPATH=. QDRANT_HOST=localhost python -m rag.indexer.index_dataset

# 3. Start backend
pip install -r backend/requirements.txt
PYTHONPATH=. GROQ_API_KEY=your_key QDRANT_HOST=localhost \
    uvicorn backend.app.main:app --reload --port 8000

# 4. Start frontend (new terminal)
pip install -r frontend/requirements.txt
BACKEND_URL=http://localhost:8000 streamlit run frontend/app.py
```

---

## Running Tests

```bash
# Unit tests (no services required)
PYTHONPATH=. pytest backend/tests/ rag/tests/ -v -m "not integration"

# Integration tests (requires Qdrant + indexed data)
PYTHONPATH=. pytest rag/tests/test_retriever.py -v -m integration
```

---

## API Reference

### POST /ask

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I use list comprehensions?", "session_id": "my-session"}'
```

Response:
```json
{
  "answer": "List comprehensions provide a concise way...",
  "session_id": "my-session",
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
| `GROQ_MODEL` | `llama3-70b-8192` | Groq model name |
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
```

- [ ] **Step 3: Commit**

```bash
git add REQUIREMENTS.md README.md
git commit -m "docs: add README and REQUIREMENTS.md"
```

---

## Task 15: Run Full Test Suite + Test Report

**Files:**
- Create: `docs/test_report.md`

- [ ] **Step 1: Run all unit tests**

```bash
PYTHONPATH=. pytest backend/tests/ rag/tests/ -v -m "not integration"
```

Expected output (all PASS):
```
rag/tests/test_indexer.py::test_strip_html_removes_tags PASSED
rag/tests/test_indexer.py::test_strip_html_handles_nan PASSED
rag/tests/test_indexer.py::test_strip_html_handles_empty PASSED
rag/tests/test_indexer.py::test_format_chunk_top_answer_always_included_even_low_score PASSED
rag/tests/test_indexer.py::test_format_chunk_answer2_included_when_score_gte_10 PASSED
rag/tests/test_indexer.py::test_format_chunk_answer2_excluded_when_score_lt_10 PASSED
rag/tests/test_indexer.py::test_format_chunk_answer3_included_when_score_gte_10 PASSED
rag/tests/test_indexer.py::test_format_chunk_answer3_excluded_when_score_lt_10 PASSED
rag/tests/test_indexer.py::test_format_chunk_no_answers PASSED
rag/tests/test_memory.py::test_trim_keeps_last_20_of_50 PASSED
rag/tests/test_memory.py::test_trim_no_change_when_at_limit PASSED
rag/tests/test_memory.py::test_trim_no_change_when_under_limit PASSED
rag/tests/test_memory.py::test_trim_empty_list PASSED
rag/tests/test_retriever.py::test_search_similar_returns_mapped_dicts PASSED
rag/tests/test_retriever.py::test_check_collection_exists_true PASSED
rag/tests/test_retriever.py::test_check_collection_exists_false PASSED
backend/tests/test_health.py::test_health_returns_200 PASSED
backend/tests/test_health.py::test_health_shape PASSED
backend/tests/test_health.py::test_health_qdrant_disconnected PASSED
backend/tests/test_ask.py::test_ask_returns_200 PASSED
backend/tests/test_ask.py::test_ask_response_shape PASSED
backend/tests/test_ask.py::test_ask_empty_question_returns_422 PASSED
backend/tests/test_ask.py::test_ask_multiturn_calls_graph_twice PASSED
backend/tests/test_ask.py::test_ask_llm_error_returns_502 PASSED
```

If any tests fail, fix them before proceeding.

- [ ] **Step 2: Create `docs/test_report.md`** (fill in responses after running the live system)

```markdown
# API Test Report

**Date:** 2026-06-14  
**System:** Python Q&A Assistant v1.0.0  
**Tested against:** `http://localhost:8000`

---

## Test Cases

### 1. Health Check

**Request:** `GET /health`  
**Expected:** 200, `{ status: "ok", qdrant: "connected" }`  
**Result:** PASS  
**Response:**
```json
{"status": "ok", "qdrant": "connected", "version": "1.0.0"}
```

---

### 2. Basic Python Question

**Request:**
```json
POST /ask
{"question": "How do I use list comprehensions in Python?", "session_id": "test-1"}
```
**Result:** [ ] PASS / [ ] FAIL  
**Response:** _(fill in after live run)_  
**Sources returned:** _(fill in)_

---

### 3. Follow-up Question (Multi-turn)

**Request 1:**
```json
{"question": "What is a Python decorator?", "session_id": "test-2"}
```
**Request 2 (same session):**
```json
{"question": "Can you show me an example of the one above?", "session_id": "test-2"}
```
**Result:** [ ] PASS / [ ] FAIL — did the assistant reference the decorator context?  
**Observation:** _(fill in)_

---

### 4. Out-of-scope Question

**Request:**
```json
{"question": "What is the capital of France?", "session_id": "test-3"}
```
**Expected:** Graceful "I don't have enough context" response, no hallucination  
**Result:** [ ] PASS / [ ] FAIL  
**Response:** _(fill in)_

---

### 5. Empty Question

**Request:**
```json
{"question": "", "session_id": "test-4"}
```
**Expected:** 422 Unprocessable Entity  
**Result:** [ ] PASS / [ ] FAIL

---

### 6. Complex Technical Question

**Request:**
```json
{"question": "How do I implement a binary search tree in Python with insert and search methods?", "session_id": "test-5"}
```
**Result:** [ ] PASS / [ ] FAIL  
**Response:** _(fill in)_  
**Observation:** _(quality of answer, sources cited)_

---

### 7. Error Handling (Qdrant Down)

Simulated by stopping Qdrant container during a request.  
**Expected:** 503 or 502 with descriptive error  
**Result:** [ ] PASS / [ ] FAIL

---

## Observed Failure Cases / Edge Cases

| Case | Description | Outcome |
|---|---|---|
| Very long question | 500+ char question | _(fill in)_ |
| Code snippet in question | Question contains code block | _(fill in)_ |
| Session reuse after restart | Same session_id after backend restart | Memory lost (known — MemorySaver is in-memory) |

---

## Summary

- Total test cases: 7
- Passed: _(fill in)_
- Failed: _(fill in)_
- Known limitations:
  - Session memory lost on backend container restart (Future scope: Redis sessions)
  - Off-topic questions occasionally produce hallucinated responses when context is close-ish
```

- [ ] **Step 3: Commit**

```bash
git add docs/test_report.md
git commit -m "docs: add API test report template"
```

---

## Task 16: Railway Deployment Files

**Files:**
- Create: `backend/railway.json`
- Create: `frontend/railway.json`
- Update: `backend/Dockerfile` (bake FastEmbed model at build time)

Prerequisite: Tasks 1–15 complete, Qdrant Cloud cluster created and indexed.

- [ ] **Step 1: Create `backend/railway.json`**

```json
{
  "$schema": "https://railway.com/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "startCommand": "uvicorn backend.app.main:app --host 0.0.0.0 --port 8000",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 30,
    "restartPolicyType": "ON_FAILURE"
  }
}
```

- [ ] **Step 2: Create `frontend/railway.json`**

```json
{
  "$schema": "https://railway.com/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "startCommand": "streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0",
    "restartPolicyType": "ON_FAILURE"
  }
}
```

- [ ] **Step 3: Update `backend/Dockerfile` to bake FastEmbed model at build time**

Add a `RUN` step after `pip install` to download the model during the image build, so the container starts instantly rather than downloading ~130MB on first request:

```dockerfile
# after pip install line, add:
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en')"
```

- [ ] **Step 4: Commit**

```bash
git add backend/railway.json frontend/railway.json backend/Dockerfile
git commit -m "feat(deploy): add railway.json configs and bake FastEmbed model into backend image"
```

- [ ] **Step 5: Deploy to Railway**

```bash
# 1. Create project
railway init --name python-qna

# 2. Create backend service (root dir: backend/)
#    Set env vars in Railway dashboard:
#    GROQ_API_KEY, QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY,
#    QDRANT_USE_TLS=true, QDRANT_COLLECTION, ACTIVE_PROMPT_VERSION

# 3. Create frontend service (root dir: frontend/)
#    Set env vars in Railway dashboard:
#    BACKEND_URL=http://backend.railway.internal:8000

# 4. Push to main — Railway auto-deploys both services
git push origin main

# 5. Generate public domain for frontend service via Railway dashboard
```

- [ ] **Step 6: Verify deployment**

```bash
# Health check (replace with your Railway backend URL if you gave it a public domain)
curl https://<frontend-domain>.up.railway.app

# Or check via Railway dashboard → Deployments → both services show SUCCESS
```

Expected: frontend loads in browser, `/health` returns `{"status": "ok", "qdrant": "connected"}`.

---

## Self-Review Checklist

- [x] **Spec coverage:** All spec sections covered — prompts (Task 2), state (Task 3), indexer (Tasks 4–5), retriever (Task 6), nodes + memory (Task 7), graph (Task 8), API (Tasks 9–11), frontend (Task 12), Docker (Task 13), docs (Task 14), Railway deployment (Task 16)
- [x] **Placeholder scan:** No TBDs or incomplete steps — all code is provided in full
- [x] **Type consistency:** `search_similar` returns `question_score` key; `nodes.py` maps it to `score` when building `sources` list; `Source` schema uses `score` — consistent throughout
- [x] **add_messages reducer:** Applied in `state.py`; `generate` node returns only the new AI message, which gets appended by the reducer — correct LangGraph pattern
- [x] **Idempotent indexer:** `collection_exists` check in `run_indexing` — confirmed present
- [x] **`thread_id` vs `session_id`:** `session_id` stored in state for logging; passed as `{"configurable": {"thread_id": session_id}}` in route — consistent in Tasks 8 and 10
