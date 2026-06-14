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
    raw = list(embedding_model.embed([query]))[0]
    query_vector = raw.tolist() if hasattr(raw, "tolist") else list(raw)
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
