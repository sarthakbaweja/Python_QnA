"""
Retrieval hit-rate eval.

Queries Qdrant with each question's own title and checks whether the correct
question_id appears in the top-k results, using Stack Overflow IDs as ground truth.

Requires: live Qdrant with indexed data
Run:      PYTHONPATH=backend pytest rag/tests/test_eval_retrieval.py -v -m eval -s
"""
import os
from pathlib import Path

import pandas as pd
import pytest
from dotenv import load_dotenv

load_dotenv()

from rag.retriever.qdrant_client import get_embedding_model, get_qdrant_client, search_similar

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))

_N = 100
_TOP_K = 5
_SEED = 42

MIN_HIT_AT_5 = 0.50
MIN_MRR = 0.30


@pytest.mark.eval
def test_retrieval_hit_rate():
    df = pd.read_csv(
        os.path.join(DATA_DIR, "Questions.csv"),
        usecols=["Id", "Title"],
        encoding="utf-8",
        encoding_errors="replace",
    )
    sample = df.sample(n=min(_N, len(df)), random_state=_SEED).reset_index(drop=True)

    client = get_qdrant_client()
    model = get_embedding_model()

    hits = {k: 0 for k in range(1, _TOP_K + 1)}
    reciprocal_ranks: list[float] = []

    for _, row in sample.iterrows():
        q_id = int(row["Id"])
        results = search_similar(str(row["Title"]), client, model, top_k=_TOP_K)
        retrieved_ids = [r["question_id"] for r in results]

        for k in range(1, _TOP_K + 1):
            if q_id in retrieved_ids[:k]:
                hits[k] += 1

        rank = next((pos + 1 for pos, rid in enumerate(retrieved_ids) if rid == q_id), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

    n = len(sample)
    mrr = sum(reciprocal_ranks) / n

    print(f"\nRetrieval Hit-Rate ({n} questions, top_k={_TOP_K})")
    for k in range(1, _TOP_K + 1):
        print(f"  Hit@{k}: {hits[k] / n:.3f}  ({hits[k]}/{n})")
    print(f"  MRR:   {mrr:.3f}")

    assert hits[5] / n >= MIN_HIT_AT_5, f"Hit@5 {hits[5] / n:.3f} below threshold {MIN_HIT_AT_5}"
    assert mrr >= MIN_MRR, f"MRR {mrr:.3f} below threshold {MIN_MRR}"
