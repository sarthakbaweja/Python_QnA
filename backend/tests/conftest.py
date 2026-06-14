import os
import pytest
from fastapi.testclient import TestClient

# Stub secrets before importing the app so the lifespan check passes
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("FRONTEND_URL", "http://localhost:8501")


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)
