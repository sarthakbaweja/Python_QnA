"""
Retrieval hit-rate eval using Stack Overflow question IDs as ground truth.

For each sampled question, queries Qdrant with the question title and checks
whether the correct question_id appears in the top-k results.

Metrics:
  Hit@1, Hit@3, Hit@5  — fraction where correct doc is in top-k
  MRR                  — mean reciprocal rank

Usage:
    PYTHONPATH=backend python evals/eval_retrieval.py
    PYTHONPATH=backend python evals/eval_retrieval.py --n 200 --top-k 5
"""
import argparse
import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from rag.retriever.qdrant_client import get_embedding_model, get_qdrant_client, search_similar

DATA_DIR = os.getenv("DATA_DIR", "data")


def run(n: int, top_k: int, seed: int) -> None:
    df = pd.read_csv(
        os.path.join(DATA_DIR, "Questions.csv"),
        usecols=["Id", "Title"],
        encoding="utf-8",
        encoding_errors="replace",
    )
    sample = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    n_actual = len(sample)

    client = get_qdrant_client()
    model = get_embedding_model()

    hits = {k: 0 for k in range(1, top_k + 1)}
    reciprocal_ranks: list[float] = []

    print(f"Evaluating {n_actual} questions (top_k={top_k})...")
    for i, row in sample.iterrows():
        q_id = int(row["Id"])
        results = search_similar(str(row["Title"]), client, model, top_k=top_k)
        retrieved_ids = [r["question_id"] for r in results]

        for k in range(1, top_k + 1):
            if q_id in retrieved_ids[:k]:
                hits[k] += 1

        rank = next((pos + 1 for pos, rid in enumerate(retrieved_ids) if rid == q_id), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n_actual}")

    print(f"\nResults ({n_actual} questions, top_k={top_k}):")
    for k in range(1, top_k + 1):
        print(f"  Hit@{k}: {hits[k] / n_actual:.3f}  ({hits[k]}/{n_actual})")
    print(f"  MRR:   {sum(reciprocal_ranks) / n_actual:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Number of questions to evaluate")
    parser.add_argument("--top-k", type=int, default=5, help="Max rank to consider")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.n, args.top_k, args.seed)
