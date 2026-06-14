from bs4 import BeautifulSoup


def strip_html(html: str) -> str:
    s = str(html).strip()
    if not s or s == "nan":
        return ""
    text = BeautifulSoup(s, "html.parser").get_text(separator=" ")
    return " ".join(text.split())


def format_chunk(question: dict, answers: list[dict], tags: list[str] | None = None) -> str:
    title = str(question.get("Title", ""))
    body = strip_html(question.get("Body", ""))

    parts = [f"Question: {title}\n{body}"]

    if tags:
        parts.append(f"Tags: {', '.join(tags)}")

    if not answers:
        return "\n".join(parts)

    top = answers[0]
    parts.append(f"\n[TOP ANSWER (score: {top['Score']})]\n{strip_html(top['Body'])}")

    for i, ans in enumerate(answers[1:3], start=2):
        if int(ans["Score"]) >= 10:
            parts.append(f"\n[ANSWER {i} (score: {ans['Score']})]\n{strip_html(ans['Body'])}")

    return "\n".join(parts)
