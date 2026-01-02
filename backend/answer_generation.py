from retriever import retrieve_top_k
from local_llm import local_generate
from generation import build_prompt

def generate_answer_local(user_query: str, k: int = 5) -> str:
    top_chunks = retrieve_top_k(user_query, k=k)
    if not top_chunks:
        return "No relevant pump data found. Check FAISS index and retriever.py."

    prompt = build_prompt(user_query, top_chunks)
    if not prompt:
        return "Information not available."

    return local_generate(prompt)

if __name__ == "__main__":
    q = "List technical specifications of pump P00001."
    answer = generate_answer_local(q, k=8)
    print("FINAL ANSWER:\n", answer)