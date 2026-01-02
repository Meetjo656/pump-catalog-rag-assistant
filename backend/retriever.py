import os
import pickle
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

from pump_master import resolve_model_identifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

VECTOR_DIR = os.path.join(BASE_DIR, "vector_store")
INDEX_PATH = os.path.join(VECTOR_DIR, "pump_index.faiss")
DOCS_PATH = os.path.join(VECTOR_DIR, "pump_documents.pkl")
META_PATH = os.path.join(VECTOR_DIR, "pump_metadata.pkl")

EMBED_MODEL = "text-embedding-004"

_client: Optional[genai.Client] = None
_index = None
_docs: Optional[List[str]] = None
_meta: Optional[List[Dict[str, Any]]] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found in .env")
        _client = genai.Client(api_key=api_key)
    return _client


def _load_store() -> None:
    global _index, _docs, _meta

    if _index is not None and _docs is not None and _meta is not None:
        return

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH} (run ingest_pump_specifications.py)")

    if not os.path.exists(DOCS_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Documents/metadata not found in {VECTOR_DIR} (run ingest_pump_specifications.py)")

    _index = faiss.read_index(INDEX_PATH)

    with open(DOCS_PATH, "rb") as f:
        _docs = pickle.load(f)

    with open(META_PATH, "rb") as f:
        _meta = pickle.load(f)


def retrieve_top_k(
    query: str,
    k: int = 5,
    model_filter: Optional[str] = None,
    chunk_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Returns list of chunks:
      { "text": str, "model": str, "chunk_type": str, "category": str, "source": str }
    """
    if not isinstance(query, str) or not query.strip():
        return []

    _load_store()
    client = _get_client()

    resolved = resolve_model_identifier(model_filter) if model_filter else None

    emb = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[query],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )

    q = np.array([emb.embeddings[0].values], dtype="float32")
    D, I = _index.search(q, max(k * 6, k))

    out: List[Dict[str, Any]] = []

    for idx in I[0]:
        if idx < 0:
            continue
        if idx >= len(_docs):
            continue

        m = _meta[idx] or {}
        m_model = m.get("model")

        if resolved and m_model not in (resolved, "ALL"):
            continue

        if chunk_type and m.get("chunk_type") != chunk_type:
            continue

        out.append(
            {
                "text": _docs[idx],
                "model": m_model,
                "chunk_type": m.get("chunk_type"),
                "category": m.get("category"),
                "source": m.get("source"),
            }
        )

        if len(out) >= k:
            break

    return out
def retrieve_all_for_model(
    model_filter: str,
    chunk_type: str | None = None,
) -> list[dict]:
    """
    Deterministic retrieval (no embeddings):
    returns ALL chunks for a model (+ 'ALL' shared chunks).
    """
    _load_store()

    resolved = resolve_model_identifier(model_filter)
    if not resolved:
        return []

    out = []
    for i, m in enumerate(_meta):
        if not m:
            continue
        if chunk_type and m.get("chunk_type") != chunk_type:
            continue

        mm = m.get("model")
        if mm not in (resolved, "ALL"):
            continue

        out.append(
            {
                "text": _docs[i],
                "model": mm,
                "chunk_type": m.get("chunk_type"),
                "category": m.get("category"),
                "source": m.get("source"),
            }
        )

    # keep a stable order (category then source)
    out.sort(key=lambda x: (str(x.get("category") or ""), str(x.get("source") or ""), str(x.get("model") or "")))
    return out
