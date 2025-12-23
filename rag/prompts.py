from typing import List, Dict

DEFAULT_SYSTEM_INSTRUCTION = (
    "Ты — помощник службы поддержки. Формируй краткие, точные и понятные "
    "ответы пользователю на основании предоставленных выдержек из документации. "
    "Если ответа явно не содержится в источниках — честно скажи, что не нашёл информацию, "
    "и предложи шаги для дальнейших действий."
)

CONTEXT_SEPARATOR = "\n---\n"

def build_rag_prompt(
    question: str,
    retrieved_docs: List[Dict],
    system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
    max_context_chars: int = 2000,
) -> Dict:
    context_pieces = []
    for doc in retrieved_docs:
        meta = doc.metadata or {}
        title = meta.get("title")
        source = meta.get("source")
        text = doc.page_content

        header = f"Title: {title}" if title else ""
        src = f"Source: {source}" if source else ""
        snippet = text.strip()
        piece = " ".join([p for p in [header, src, snippet] if p])
        context_pieces.append(piece)

    context = CONTEXT_SEPARATOR.join(context_pieces)
    if len(context) > max_context_chars:
        context = context[: max_context_chars] + "\n[...context truncated...]"

    user_block = (
        "Вам приведены выдержки из документации (ниже). На их основе ответьте кратко и по делу.\n\n"
        f"Документация:\n{context}\n\n"
        f"Вопрос: {question}\n\n"
        "Ответ:"
    )

    return {"system": system_instruction, "user": user_block}
