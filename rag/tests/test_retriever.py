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
