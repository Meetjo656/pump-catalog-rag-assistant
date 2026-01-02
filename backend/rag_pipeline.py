# backend/rag_pipeline.py

from retriever import retrieve_all_for_model, retrieve_top_k
from generation import build_prompt, generate_answer
from pump_master import resolve_model_identifier


def _run(question: str, chunks):
    prompt = build_prompt(question, chunks)
    return generate_answer(prompt)


def _resolve_model(p: dict, *keys: str) -> str | None:
    """
    Accepts multiple possible param keys and resolves either model_id or model_name
    to the canonical model_id (Pxxxxx).
    """
    for k in keys:
        v = p.get(k)
        if not v:
            continue
        mid = resolve_model_identifier(v)
        if mid:
            return mid
    return None


def rag_view_specs(p):
    # accept: model_id / model_name / modelA
    mid = _resolve_model(p, "model_id", "model_name", "modelA")
    if not mid:
        return "Information not available."

    chunks = retrieve_all_for_model(mid, chunk_type="specifications")
    return _run(
        f"List ALL technical specifications of MODEL {mid}. Include every item from the context; do not summarize.",
        chunks,
    )


def rag_explain_suitability(p):
    # accept: model_id / model_name / modelA
    mid = _resolve_model(p, "model_id", "model_name", "modelA")
    app = p.get("application", "domestic")
    if not mid:
        return "Information not available."

    chunks = (
        retrieve_all_for_model(mid, chunk_type="features")
        + retrieve_all_for_model(mid, chunk_type="specifications")
    )

    question = (
        f"Explain suitability of MODEL {mid} for {app}.\n"
        "Use ONLY the context.\n"
        "Include:\n"
        "- Suitability score and 'Recommended' value if present.\n"
        "- Key reasons from applications.\n"
        "- Advantages/limitations from features.\n"
        "- A short final recommendation."
    )
    return _run(question, chunks)


def rag_compare_models(p):
    # accept for A: modelA OR modelA_id/modelA_name OR model_id/model_name
    m1 = _resolve_model(p, "modelA_id", "modelA_name", "modelA", "model_id", "model_name")
    # accept for B: modelB OR modelB_id/modelB_name
    m2 = _resolve_model(p, "modelB_id", "modelB_name", "modelB")

    if not m1 or not m2 or m1 == m2:
        return "Information not available."

    chunks = (
        retrieve_all_for_model(m1, "specifications")
        + retrieve_all_for_model(m1, "features")
        + retrieve_all_for_model(m2, "specifications")
        + retrieve_all_for_model(m2, "features")
    )

    question = (
        f"Compare MODEL {m1} vs MODEL {m2}. "
        "Use specs and features from context. Provide differences and a recommendation by application "
        "(domestic/agricultural/industrial)."
    )
    return _run(question, chunks)


def rag_installation_guidance(p):
    # accept: model_id / model_name / modelA
    mid = _resolve_model(p, "model_id", "model_name", "modelA")
    if not mid:
        return "Information not available."

    chunks = retrieve_all_for_model(mid, chunk_type="installation")
    question = (
        f"Provide installation guidance for MODEL {mid}.\n"
        "Use ONLY the context.\n"
        "Output as ordered steps.\n"
        "Include safety warnings where present."
    )
    return _run(question, chunks)


def rag_free_text(q: str):
    chunks = retrieve_top_k(q, model_filter=None, chunk_type=None, k=10)
    return _run(q, chunks)
