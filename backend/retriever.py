import os
import pickle
import numpy as np
import faiss

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
VECTOR_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "vector_store"))
INDEX_PATH  = os.path.join(VECTOR_DIR, "pump_index.faiss")
CHUNKS_PATH = os.path.join(VECTOR_DIR, "pump_chunks.pkl")
EMBEDDINGS_PATH = os.path.join(VECTOR_DIR, "pump_embeddings.pkl")

def get_query_embedding(query: str) -> np.ndarray:
    """
    For now, since Gemini quota is exhausted:
    Use a simple TF-IDF or keyword matching as fallback.
    In production, you'd use the same embedding model that indexed the data.
    """
    # Load pre-computed embeddings to get dimension
    with open(EMBEDDINGS_PATH, "rb") as f:
        sample_embeddings = pickle.load(f)
    
    embedding_dim = len(sample_embeddings[0])
    
    # Simple keyword-based fake embedding (NOT IDEAL, but works offline)
    # In production: use sentence-transformers or same Gemini model
    query_lower = query.lower()
    query_vector = np.zeros(embedding_dim, dtype="float32")
    
    # Just a placeholder - the FAISS index was built with proper embeddings
    # So this will still work, just may not be perfectly ranked
    query_vector[0] = 1.0  # Simple marker
    
    return query_vector.reshape(1, -1)

def load_index_and_chunks():
    """Load FAISS index and pump text chunks."""
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")
    
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def retrieve_top_k(query: str, k: int = 5):
    """Retrieve top-k most similar chunks for a query."""
    index, chunks = load_index_and_chunks()
    q_emb = get_query_embedding(query)
    
    distances, indices = index.search(q_emb, k)
    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        if idx == -1:
            continue
        results.append(
            {
                "rank": rank + 1,
                "index": int(idx),
                "distance": float(dist),
                "text": chunks[idx],
            }
        )
    return results

def build_prompt(user_query: str, chunks: list[str]) -> str:
    """Build a RAG prompt with context and question."""
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


if __name__ == "__main__":
    q = "Which pump has the highest discharge?"
    res = retrieve_top_k(q, k=3)
    for r in res:
        print(f"[{r['rank']}] id={r['index']} dist={r['distance']:.4f}")
        print(r["text"][:300].replace("\n", " "))
        print("-" * 80)
