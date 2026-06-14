"""
RAGAS eval — faithfulness and answer relevancy.

Samples N questions, runs retrieval + LLM generation, then scores with RAGAS
using Groq as the judge LLM and FastEmbed for embeddings.

Metrics:
  faithfulness      — fraction of answer claims supported by retrieved context
  answer_relevancy  — how well the answer addresses the original question

Usage:
    PYTHONPATH=backend python evals/eval_ragas.py
    PYTHONPATH=backend python evals/eval_ragas.py --n 25
"""
import argparse
import os

import pandas as pd
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness
from fastembed import TextEmbedding

load_dotenv()

from rag.prompts import v1 as prompts
from rag.retriever.qdrant_client import get_embedding_model, get_qdrant_client, search_similar

DATA_DIR = os.getenv("DATA_DIR", "data")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TOP_K = 5


class _FastEmbedLangchain(Embeddings):
    """Thin LangChain Embeddings wrapper around FastEmbed for RAGAS."""

    def __init__(self, model_name: str = "BAAI/bge-small-en"):
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [e.tolist() for e in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(self._model.embed([text]))[0].tolist()


def _generate_answer(llm: ChatGroq, question: str, contexts: list[str]) -> str:
    context_text = "\n\n---\n\n".join(contexts)
    context_block = prompts.RETRIEVAL_CONTEXT_TEMPLATE.format(context=context_text)
    system_msg = SystemMessage(content=prompts.SYSTEM_PROMPT + "\n\n" + context_block)
    response = llm.invoke([system_msg, HumanMessage(content=question)])
    return response.content


def run(n: int, seed: int) -> None:
    df = pd.read_csv(
        os.path.join(DATA_DIR, "Questions.csv"),
        usecols=["Id", "Title"],
        encoding="utf-8",
        encoding_errors="replace",
    )
    sample = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    n_actual = len(sample)

    client = get_qdrant_client()
    retrieval_model = get_embedding_model()
    llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.1)

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(_FastEmbedLangchain())

    print(f"Generating answers for {n_actual} questions...")
    ragas_samples: list[SingleTurnSample] = []

    for i, row in sample.iterrows():
        question = str(row["Title"])
        results = search_similar(question, client, retrieval_model, top_k=TOP_K)
        contexts = [r["text"] for r in results]
        answer = _generate_answer(llm, question, contexts)

        ragas_samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=contexts,
                response=answer,
            )
        )

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_actual} done")

    print("\nRunning RAGAS evaluation...")
    dataset = EvaluationDataset(samples=ragas_samples)
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), AnswerRelevancy()],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    print(f"\nRAGAS Results ({n_actual} questions):")
    print(f"  Faithfulness:     {result['faithfulness']:.3f}")
    print(f"  Answer Relevancy: {result['answer_relevancy']:.3f}")
    print(f"\nFull breakdown:\n{result.to_pandas()[['faithfulness', 'answer_relevancy']].to_string()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25, help="Number of questions to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.n, args.seed)
