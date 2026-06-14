import os
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_groq import ChatGroq

from rag.pipeline.state import GraphState
from rag.retriever.qdrant_client import get_qdrant_client, get_embedding_model, search_similar
from rag.prompts import v1 as prompts

MESSAGE_LIMIT = 20

_llm: ChatGroq | None = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,
        )
    return _llm


def trim_messages_to_limit(
    messages: list[BaseMessage], limit: int = MESSAGE_LIMIT
) -> list[BaseMessage]:
    if len(messages) <= limit:
        return messages
    return messages[-limit:]


def retrieve(state: GraphState) -> dict:
    client = get_qdrant_client()
    embedding_model = get_embedding_model()

    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    if last_human is None:
        return {"context": [], "sources": []}

    results = search_similar(last_human.content, client, embedding_model, top_k=5)

    return {
        "context": [r["text"] for r in results],
        "sources": [
            {
                "question_title": r["title"],
                "question_id": r["question_id"],
                "score": r["question_score"],
            }
            for r in results
        ],
    }


def generate(state: GraphState) -> dict:
    llm = _get_llm()

    trimmed = trim_messages_to_limit(state["messages"])
    context_text = "\n\n---\n\n".join(state.get("context", []))
    context_block = prompts.RETRIEVAL_CONTEXT_TEMPLATE.format(context=context_text)
    system_msg = SystemMessage(content=prompts.SYSTEM_PROMPT + "\n\n" + context_block)

    response = llm.invoke([system_msg] + list(trimmed))
    return {"messages": [AIMessage(content=response.content)]}
