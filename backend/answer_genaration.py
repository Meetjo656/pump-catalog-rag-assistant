from retriever import retrieve_top_k
from local_llm import local_generate
from generation import build_prompt
def generate_answer_local(user_query: str, k: int = 5) -> str:
    top_chunks = retrieve_top_k(user_query, k=k)
    chunk_texts = [c["text"] for c in top_chunks]
    prompt = build_prompt(user_query, chunk_texts)
    return local_generate(prompt)

if __name__ == "__main__":
    q = "Which pump has the highest discharge?"
    answer = generate_answer_local(q, k=3)
    print("FINAL ANSWER:\n", answer)
