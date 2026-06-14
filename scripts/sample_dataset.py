"""
Sample N questions (and their answers + tags) from the full dataset.

Usage:
    python scripts/sample_dataset.py                  # 1000 questions, seed 42
    python scripts/sample_dataset.py --n 500
    python scripts/sample_dataset.py --n 1000 --seed 99
"""
import argparse
import os
import pandas as pd

FULL_DIR = os.path.join(os.path.dirname(__file__), "..", "data_full")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def sample(n: int, seed: int) -> None:
    print(f"Sampling {n} questions (seed={seed})...")

    questions = pd.read_csv(
        os.path.join(FULL_DIR, "Questions.csv"),
        usecols=["Id", "OwnerUserId", "CreationDate", "Score", "Title", "Body"],
        encoding="utf-8",
        encoding_errors="replace",
    )
    sampled = questions.sample(n=n, random_state=seed)
    ids = set(sampled["Id"].tolist())
    print(f"  Sampled {len(sampled)} questions")

    answers = pd.read_csv(
        os.path.join(FULL_DIR, "Answers.csv"),
        usecols=["Id", "OwnerUserId", "CreationDate", "ParentId", "Score", "Body"],
        encoding="utf-8",
        encoding_errors="replace",
    )
    sampled_answers = answers[answers["ParentId"].isin(ids)]
    print(f"  Matched {len(sampled_answers)} answers")

    tags = pd.read_csv(
        os.path.join(FULL_DIR, "Tags.csv"),
        usecols=["Id", "Tag"],
        encoding="utf-8",
        encoding_errors="replace",
    )
    sampled_tags = tags[tags["Id"].isin(ids)]
    print(f"  Matched {len(sampled_tags)} tag rows")

    os.makedirs(OUT_DIR, exist_ok=True)
    sampled.to_csv(os.path.join(OUT_DIR, "Questions.csv"), index=False)
    sampled_answers.to_csv(os.path.join(OUT_DIR, "Answers.csv"), index=False)
    sampled_tags.to_csv(os.path.join(OUT_DIR, "Tags.csv"), index=False)
    print(f"Saved to {os.path.abspath(OUT_DIR)}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sample(args.n, args.seed)
