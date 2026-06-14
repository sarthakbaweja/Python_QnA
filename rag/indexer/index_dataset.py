import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding

from rag.indexer.chunk_formatter import format_chunk

load_dotenv()

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "python_qna")
VECTOR_SIZE = 384  # BAAI/bge-small-en output dimension
BATCH_SIZE = 256
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
INDEX_LIMIT = int(os.getenv("INDEX_LIMIT", "0"))  # 0 = no limit


def get_client() -> QdrantClient:
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    api_key = os.getenv("QDRANT_API_KEY", "")
    use_tls = os.getenv("QDRANT_USE_TLS", "false").lower() == "true"
    if use_tls and api_key:
        return QdrantClient(url=f"https://{host}:{port}", api_key=api_key)
    return QdrantClient(host=host, port=port)


def collection_populated(client: QdrantClient) -> bool:
    names = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in names:
        return False
    count = client.get_collection(COLLECTION_NAME).vectors_count or 0
    return count > 0


def load_answers(answers_path: str) -> dict[int, list[dict]]:
    print("Loading answers into memory...")
    df = pd.read_csv(
        answers_path,
        usecols=["Id", "ParentId", "Score", "Body"],
        dtype={"Id": "int64", "ParentId": "int64", "Score": "float64", "Body": "str"},
        encoding="utf-8",
        encoding_errors="replace",
    )
    df["Score"] = df["Score"].fillna(0).astype(int)
    df = df.sort_values("Score", ascending=False)
    grouped = df.groupby("ParentId")
    return {int(pid): group.to_dict("records") for pid, group in grouped}


def run_indexing(questions_path: str, answers_path: str) -> None:
    client = get_client()

    if collection_populated(client):
        print(f"Collection '{COLLECTION_NAME}' already indexed. Skipping.")
        return

    print(f"Creating collection '{COLLECTION_NAME}'...")
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    except UnexpectedResponse as e:
        if e.status_code == 409:
            print(f"Collection '{COLLECTION_NAME}' already exists, continuing...")
        else:
            raise

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
        encoding="utf-8",
        encoding_errors="replace",
    )

    for chunk in tqdm(reader, desc="Questions"):
        chunk["Score"] = chunk["Score"].fillna(0).astype(int)

        for _, row in chunk.iterrows():
            if INDEX_LIMIT and total >= INDEX_LIMIT:
                break
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
