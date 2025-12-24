def build_prompt(user_query: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(chunks)
    return f"""
You are a pump selection assistant.
Use only the context below to answer the question.
If the answer is not in the context, say you don't know.

CONTEXT:
{context}

QUESTION:
{user_query}

ANSWER:
""".strip()
