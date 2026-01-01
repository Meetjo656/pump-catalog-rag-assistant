from pump_master import get_model_id_by_name, get_model_name_by_id
from retriever import retrieve_top_k
from generation import build_prompt



# =========================================================
# Shared helper
# =========================================================

def _run_rag(question, model_id=None, chunk_type=None, k=6):
    chunks = retrieve_top_k(
        query=question,
        k=k,
        model_filter=model_id,
        chunk_type=chunk_type,
    )

    if not chunks:
        return {"answer": "Information not available for this pump."}

    prompt = build_prompt(question, chunks)
    return {"answer": prompt}


# =========================================================
# RAG handlers (MATCH app.py imports)
# =========================================================

def rag_view_specs(params: dict):
    # ---- Validate input
    model_name = (params.get("modelA") or "").strip()
    if not model_name:
        raise ValueError("modelA is required for view_specs")

    # ---- Resolve name → ID (NON-NEGOTIABLE)
    model_id = get_model_id_by_name(model_name)

    # ---- Build query (human-readable)
    question = f"List the technical specifications of the {model_name} pump."

    # ---- STRICT retrieval (ID ONLY)
    chunks = retrieve_top_k(
        query=question,
        k=1,  # exactly one consolidated chunk
        model_filter=model_id,
        chunk_type="specifications",
    )

    # ---- No data guard
    if not chunks:
        return {
            "intent": "view_specs",
            "model": model_name,
            "answer": "Information not available for this pump.",
        }

    # ---- Build grounded response
    prompt = build_prompt(question, chunks)

    return {
        "intent": "view_specs",
        "model": model_name,
        "answer": prompt,
    }


def rag_compare_models(params: dict):
    m1 = params.get("modelA", "")
    m2 = params.get("modelB", "")

    id1 = get_model_id_by_name(m1)
    id2 = get_model_id_by_name(m2)

    if not id1 or not id2:
        return {"answer": "One or both pump models not found."}

    question = f"Compare pump {m1} and pump {m2}."
    return _run_rag(question, model_id=None, chunk_type="specifications", k=10)


def rag_explain_suitability(params: dict):
    model_name = params.get("modelA", "")
    application = params.get("application", "general use")

    model_id = get_model_id_by_name(model_name)
    if not model_id:
        return {"answer": f"Pump {model_name} not found."}

    question = f"Explain why pump {model_name} is suitable for {application}."
    return _run_rag(question, model_id=model_id)


def rag_installation_guidance(params: dict):
    model_name = params.get("modelA", "")
    model_id = get_model_id_by_name(model_name)

    if not model_id:
        return {"answer": f"Pump {model_name} not found."}

    question = f"Provide installation and maintenance guidance for pump {model_name}."
    return _run_rag(question, model_id=model_id)


def rag_free_text(question: str):
    return _run_rag(question)
