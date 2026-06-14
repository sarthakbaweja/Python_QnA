from unittest.mock import MagicMock, patch


def test_health_returns_200(client):
    mock_client = MagicMock()
    with patch("backend.app.routes.ask.get_qdrant_client", return_value=mock_client):
        response = client.get("/health")

    assert response.status_code == 200


def test_health_shape(client):
    mock_client = MagicMock()
    with patch("backend.app.routes.ask.get_qdrant_client", return_value=mock_client):
        response = client.get("/health")

    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "1.0.0"
    assert "qdrant" in data


def test_health_qdrant_disconnected(client):
    mock_client = MagicMock()
    mock_client.get_collections.side_effect = Exception("refused")
    with patch("backend.app.routes.ask.get_qdrant_client", return_value=mock_client):
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["qdrant"] == "disconnected"
