from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from rag.pipeline.state import GraphState
from rag.pipeline.nodes import retrieve, generate

_graph = None


def build_graph():
    memory = MemorySaver()
    builder = StateGraph(GraphState)

    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile(checkpointer=memory)


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


graph = get_graph()
