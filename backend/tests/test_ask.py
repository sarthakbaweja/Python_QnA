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
