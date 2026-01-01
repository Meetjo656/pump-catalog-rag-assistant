def build_prompt(user_query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "Information not available for this pump."

    context = "\n\n---\n\n".join(
        c["text"] for c in chunks if "text" in c
    )

    return f"""You are a pump engineering expert.

Use ONLY the information provided below.
DO NOT infer, guess, or add specifications.

CONTEXT:
{context}

QUESTION:
{user_query}

FINAL ANSWER:
"""
