# API Test Report

**Date:** 2026-06-15
**System:** Python Q&A Assistant v1.0.0
**Tested against:** Unit tests (no live services)

---

## Unit Test Results

```
============================= test session starts ==============================
platform darwin -- Python 3.12.6, pytest-9.1.0, pluggy-1.6.0 -- /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12
cachedir: .pytest_cache
rootdir: /Users/sarthakbaweja/projects/python_qna
configfile: pytest.ini
plugins: anyio-4.4.0, langsmith-0.8.15
collecting ... collected 27 items / 1 deselected / 26 selected

backend/tests/test_ask.py::test_ask_returns_200 PASSED                   [  3%]
backend/tests/test_ask.py::test_ask_response_shape PASSED                [  7%]
backend/tests/test_ask.py::test_ask_empty_question_returns_422 PASSED    [ 11%]
backend/tests/test_ask.py::test_ask_question_too_long_returns_422 PASSED [ 15%]
backend/tests/test_ask.py::test_ask_invalid_session_id_returns_422 PASSED [ 19%]
backend/tests/test_ask.py::test_ask_multiturn_calls_graph_twice PASSED   [ 23%]
backend/tests/test_ask.py::test_ask_llm_error_returns_502 PASSED         [ 26%]
backend/tests/test_health.py::test_health_returns_200 PASSED             [ 30%]
backend/tests/test_health.py::test_health_shape PASSED                   [ 34%]
backend/tests/test_health.py::test_health_qdrant_disconnected PASSED     [ 38%]
rag/tests/test_indexer.py::test_strip_html_removes_tags PASSED           [ 42%]
rag/tests/test_indexer.py::test_strip_html_handles_nan PASSED            [ 46%]
rag/tests/test_indexer.py::test_strip_html_handles_empty PASSED          [ 50%]
rag/tests/test_indexer.py::test_format_chunk_top_answer_always_included_even_low_score PASSED [ 53%]
rag/tests/test_indexer.py::test_format_chunk_answer2_included_when_score_gte_10 PASSED [ 57%]
rag/tests/test_indexer.py::test_format_chunk_answer2_excluded_when_score_lt_10 PASSED [ 61%]
rag/tests/test_indexer.py::test_format_chunk_answer3_included_when_score_gte_10 PASSED [ 65%]
rag/tests/test_indexer.py::test_format_chunk_answer3_excluded_when_score_lt_10 PASSED [ 69%]
rag/tests/test_indexer.py::test_format_chunk_no_answers PASSED           [ 73%]
rag/tests/test_memory.py::test_trim_keeps_last_20_of_50 PASSED           [ 76%]
rag/tests/test_memory.py::test_trim_no_change_when_at_limit PASSED       [ 80%]
rag/tests/test_memory.py::test_trim_no_change_when_under_limit PASSED    [ 84%]
rag/tests/test_memory.py::test_trim_empty_list PASSED                    [ 88%]
rag/tests/test_retriever.py::test_search_similar_returns_mapped_dicts PASSED [ 92%]
rag/tests/test_retriever.py::test_check_collection_exists_true PASSED    [ 96%]
rag/tests/test_retriever.py::test_check_collection_exists_false PASSED   [100%]

======================= 26 passed, 1 deselected in 1.63s =======================
```

Total: 26 tests passed, 0 failed

---

## Test Cases

### 1. Health Check
**Request:** `GET /health`
**Expected:** 200, `{ status: "ok", qdrant: "connected" }`
**Result:** PASS (verified by test_health_returns_200, test_health_shape)

---

### 2. Basic Python Question
**Request:**
```json
POST /ask
{"question": "How do I use list comprehensions in Python?", "session_id": "550e8400-e29b-41d4-a716-446655440000"}
```
**Result:** [ ] PASS / [ ] FAIL
**Response:** _(fill in after live run)_
**Sources returned:** _(fill in)_

---

### 3. Follow-up Question (Multi-turn)
**Request 1:**
```json
{"question": "What is a Python decorator?", "session_id": "123e4567-e89b-12d3-a456-426614174000"}
```
**Request 2 (same session):**
```json
{"question": "Can you show me an example of the one above?", "session_id": "123e4567-e89b-12d3-a456-426614174000"}
```
**Result:** [ ] PASS / [ ] FAIL — did the assistant reference the decorator context?
**Observation:** _(fill in)_
**Automated coverage:** test_ask_multiturn_calls_graph_twice confirms the same thread_id is passed for both calls

---

### 4. Out-of-scope Question
**Request:**
```json
{"question": "What is the capital of France?", "session_id": "550e8400-e29b-41d4-a716-446655440001"}
```
**Expected:** Graceful "I don't have enough context" response, no hallucination
**Result:** [ ] PASS / [ ] FAIL
**Response:** _(fill in)_

---

### 5. Empty Question
**Request:**
```json
{"question": "", "session_id": "550e8400-e29b-41d4-a716-446655440000"}
```
**Expected:** 422 Unprocessable Entity
**Result:** PASS (verified by test_ask_empty_question_returns_422)

---

### 6. Question Too Long (>2000 chars)
**Request:** question with 2001 characters
**Expected:** 422 Unprocessable Entity
**Result:** PASS (verified by test_ask_question_too_long_returns_422)

---

### 7. Invalid session_id (not a UUID)
**Request:**
```json
{"question": "What is a list?", "session_id": "not-a-uuid"}
```
**Expected:** 422 Unprocessable Entity
**Result:** PASS (verified by test_ask_invalid_session_id_returns_422)

---

### 8. LLM/Pipeline Error
**Expected:** 502 Bad Gateway
**Result:** PASS (verified by test_ask_llm_error_returns_502)

---

### 9. Qdrant Disconnected
**Simulated:** Exception thrown from get_collections()
**Expected:** GET /health returns 200 with qdrant: "disconnected"
**Result:** PASS (verified by test_health_qdrant_disconnected)

---

### 10. Complex Technical Question
**Request:**
```json
{"question": "How do I implement a binary search tree in Python with insert and search methods?", "session_id": "550e8400-e29b-41d4-a716-446655440002"}
```
**Result:** [ ] PASS / [ ] FAIL
**Response:** _(fill in after live run)_
**Observation:** _(quality of answer, sources cited)_

---

## Observed Failure Cases / Edge Cases

| Case | Description | Outcome |
|---|---|---|
| Very long question | 2001+ char question | 422 (automated test passes) |
| Invalid UUID | Non-UUID session_id | 422 (automated test passes) |
| Session reuse after restart | Same session_id after backend restart | Memory lost (known — MemorySaver is in-memory) |

---

## Summary

- Total automated unit tests: 26
- Passed: 26
- Failed: 0
- Known limitations:
  - Session memory lost on backend container restart (Future scope: Redis sessions)
  - Live test cases (2, 3, 4, 10) require running backend with real Groq + Qdrant credentials
