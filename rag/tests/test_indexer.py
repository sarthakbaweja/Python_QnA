import pytest
from rag.indexer.chunk_formatter import format_chunk, strip_html


def test_strip_html_removes_tags():
    assert strip_html("<p>Hello <b>world</b></p>") == "Hello world"


def test_strip_html_handles_nan():
    assert strip_html("nan") == ""


def test_strip_html_handles_empty():
    assert strip_html("") == ""


def test_format_chunk_top_answer_always_included_even_low_score():
    question = {"Id": 1, "Title": "How to reverse a list?", "Body": "<p>How?</p>", "Score": 5}
    answers = [{"Id": 10, "Score": 1, "Body": "<p>Use list.reverse()</p>"}]

    result = format_chunk(question, answers)

    assert "Question: How to reverse a list?" in result
    assert "[TOP ANSWER (score: 1)]" in result
    assert "Use list.reverse()" in result


def test_format_chunk_answer2_included_when_score_gte_10():
    question = {"Id": 1, "Title": "Test", "Body": "<p>Q</p>", "Score": 5}
    answers = [
        {"Id": 10, "Score": 50, "Body": "<p>A1</p>"},
        {"Id": 11, "Score": 15, "Body": "<p>A2</p>"},
    ]

    result = format_chunk(question, answers)

    assert "[TOP ANSWER (score: 50)]" in result
    assert "[ANSWER 2 (score: 15)]" in result


def test_format_chunk_answer2_excluded_when_score_lt_10():
    question = {"Id": 1, "Title": "Test", "Body": "<p>Q</p>", "Score": 5}
    answers = [
        {"Id": 10, "Score": 50, "Body": "<p>A1</p>"},
        {"Id": 11, "Score": 5, "Body": "<p>A2</p>"},
    ]

    result = format_chunk(question, answers)

    assert "[TOP ANSWER (score: 50)]" in result
    assert "ANSWER 2" not in result


def test_format_chunk_answer3_included_when_score_gte_10():
    question = {"Id": 1, "Title": "Test", "Body": "<p>Q</p>", "Score": 5}
    answers = [
        {"Id": 10, "Score": 50, "Body": "<p>A1</p>"},
        {"Id": 11, "Score": 20, "Body": "<p>A2</p>"},
        {"Id": 12, "Score": 12, "Body": "<p>A3</p>"},
    ]

    result = format_chunk(question, answers)

    assert "[ANSWER 3 (score: 12)]" in result


def test_format_chunk_answer3_excluded_when_score_lt_10():
    question = {"Id": 1, "Title": "Test", "Body": "<p>Q</p>", "Score": 5}
    answers = [
        {"Id": 10, "Score": 50, "Body": "<p>A1</p>"},
        {"Id": 11, "Score": 20, "Body": "<p>A2</p>"},
        {"Id": 12, "Score": 3, "Body": "<p>A3</p>"},
    ]

    result = format_chunk(question, answers)

    assert "[ANSWER 2 (score: 20)]" in result
    assert "ANSWER 3" not in result


def test_format_chunk_no_answers():
    question = {"Id": 1, "Title": "Unanswered", "Body": "<p>Q</p>", "Score": 0}

    result = format_chunk(question, [])

    assert "Question: Unanswered" in result
    assert "TOP ANSWER" not in result
