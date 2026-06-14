from langchain_core.messages import HumanMessage, AIMessage
from rag.pipeline.nodes import trim_messages_to_limit


def test_trim_keeps_last_20_of_50():
    messages = []
    for i in range(25):
        messages.append(HumanMessage(content=f"Q{i}"))
        messages.append(AIMessage(content=f"A{i}"))

    result = trim_messages_to_limit(messages, limit=20)

    assert len(result) == 20
    assert result[-1].content == "A24"
    assert result[0].content == "Q15"


def test_trim_no_change_when_at_limit():
    messages = [HumanMessage(content=f"Q{i}") for i in range(20)]
    result = trim_messages_to_limit(messages, limit=20)
    assert len(result) == 20


def test_trim_no_change_when_under_limit():
    messages = [HumanMessage(content="Q"), AIMessage(content="A")]
    result = trim_messages_to_limit(messages, limit=20)
    assert len(result) == 2


def test_trim_empty_list():
    result = trim_messages_to_limit([], limit=20)
    assert result == []
