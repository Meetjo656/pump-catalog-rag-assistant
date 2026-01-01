import os
import pickle
import numpy as np
import faiss
from google import genai
from google.genai import types
from dotenv import load_dotenv

# =========================================================
# Environment & paths
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

VECTOR_DIR = os.path.join(BASE_DIR, "vector_store")
INDEX_PATH = os.path.join(VECTOR_DIR, "pump_index.faiss")
DOCS_PATH = os.path.join(VECTOR_DIR, "pump_documents.pkl")
METADATA_PATH = os.path.join(VECTOR_DIR, "pump_metadata.pkl")

EMBED_MODEL = "text-embedding-004"

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY missing")

client = genai.Client(api_key=api_key)


# =========================================================
# Helpers
# =========================================================

def normalize_model(name: str | None) -> str:
    """
    Normalize model identifiers so UI, CSV, and metadata match.
    """
    if not name:
        return ""
    return (
        str(name)
        .lower()
        .replace("(c.i.)", "")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "")
        .replace(" ", "")
        .strip()
    )


# =========================================================
# Core Retrieval
# =========================================================

def retrieve_top_k(
    query: str,
    k: int = 5,
    model_filter: str | None = None,
    chunk_type: str | None = None,
) -> list[dict]:
    """
    Semantic retrieval with strict metadata filtering.
    """

    # ---- Load vector store
    index = faiss.read_index(INDEX_PATH)

    with open(DOCS_PATH, "rb") as f:
        documents = pickle.load(f)

    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    # ---- Embed query
    embed_result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[query],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )

    q_embedding = np.array(
        [embed_result.embeddings[0].values],
        dtype="float32",
    )

    # ---- Search (over-fetch then filter)
    distances, indices = index.search(q_embedding, k * 3)

    results = []
    target_norm = normalize_model(model_filter) if model_filter else ""

    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        meta = metadata[idx]
        doc = documents[idx]

        # ---- Model filter
        if target_norm:
            if normalize_model(meta.get("model")) != target_norm:
                continue

        # ---- Chunk type filter
        if chunk_type and meta.get("chunk_type") != chunk_type:
            continue

        results.append({
            "rank": len(results) + 1,
            "text": doc,
            "distance": float(distances[0][i]),
            "model": meta.get("model", "ALL"),
            "chunk_type": meta.get("chunk_type", "general"),
            "source": meta.get("source", "unknown"),
            "category": meta.get("category", "general"),
        })

        if len(results) >= k:
            break

    return results


# =========================================================
# Prompt Builder
# =========================================================

def build_prompt(user_query: str, chunks: list[dict]) -> str:
    """
    Clean, model-strict RAG prompt.
    Forces structured technical output.
    """

    if not chunks:
        return (
            "No technical specifications were found for this pump model "
            "in the knowledge base."
        )

    # 🔒 Assume chunks are already filtered by model_id
    model_id = chunks[0]["model"]

    # Merge all retrieved text (usually 1 chunk after fix)
    context = "\n".join(c["text"] for c in chunks)

    return f"""You are a pump engineering expert.

Use ONLY the information provided below.
DO NOT infer, guess, or add specifications that are not present.

PUMP MODEL ID:
{model_id}

TECHNICAL SPECIFICATION DATA:
{context}

USER QUESTION:
{user_query}

INSTRUCTIONS FOR ANSWER:
- List specifications as clear bullet points
- Include units exactly as stated
- Do NOT mention other pump models
- Do NOT repeat the question
- Do NOT say "general specifications"

FINAL ANSWER:"""

