"""
RAGAS quality eval — faithfulness and answer relevancy.

Requires: live Qdrant with indexed data, GROQ_API_KEY, ragas installed
Install:  pip install -r evals/requirements.txt
Run:      PYTHONPATH=backend pytest rag/tests/test_eval_ragas.py -v -m eval -s
"""
import os
from pathlib import Path

import pandas as pd
import pytest
from dotenv import load_dotenv

load_dotenv()

pytest.importorskip("ragas", reason="ragas not installed; run: pip install -r evals/requirements.txt")

from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from fastembed import TextEmbedding
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness

from rag.prompts import v1 as prompts
from rag.retriever.qdrant_client import get_embedding_model, get_qdrant_client, search_similar

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))

_N = 25
_TOP_K = 5
_SEED = 42

MIN_FAITHFULNESS = 0.70
MIN_ANSWER_RELEVANCY = 0.70


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
    return llm.invoke([system_msg, HumanMessage(content=question)]).content


@pytest.mark.eval
def test_ragas_faithfulness_and_relevancy():
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        pytest.skip("GROQ_API_KEY not set")

    df = pd.read_csv(
        os.path.join(DATA_DIR, "Questions.csv"),
        usecols=["Id", "Title"],
        encoding="utf-8",
        encoding_errors="replace",
    )
    sample = df.sample(n=min(_N, len(df)), random_state=_SEED).reset_index(drop=True)

    client = get_qdrant_client()
    retrieval_model = get_embedding_model()
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        api_key=groq_key,
        temperature=0.1,
    )
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(_FastEmbedLangchain())

    ragas_samples: list[SingleTurnSample] = []
    for _, row in sample.iterrows():
        question = str(row["Title"])
        results = search_similar(question, client, retrieval_model, top_k=_TOP_K)
        contexts = [r["text"] for r in results]
        answer = _generate_answer(llm, question, contexts)
        ragas_samples.append(
            SingleTurnSample(user_input=question, retrieved_contexts=contexts, response=answer)
        )

    dataset = EvaluationDataset(samples=ragas_samples)
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), AnswerRelevancy()],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    faithfulness = result["faithfulness"]
    answer_relevancy = result["answer_relevancy"]

    print(f"\nRAGAS Results ({len(sample)} questions):")
    print(f"  Faithfulness:     {faithfulness:.3f}")
    print(f"  Answer Relevancy: {answer_relevancy:.3f}")

    assert faithfulness >= MIN_FAITHFULNESS, (
        f"Faithfulness {faithfulness:.3f} below threshold {MIN_FAITHFULNESS}"
    )
    assert answer_relevancy >= MIN_ANSWER_RELEVANCY, (
        f"Answer Relevancy {answer_relevancy:.3f} below threshold {MIN_ANSWER_RELEVANCY}"
    )
