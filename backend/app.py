"""
RAG Chatbot Backend - Flask API
Intent router + RAG explanation layer
RAG is used ONLY for explanation and documentation reasoning
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

from retriever import retrieve_top_k
from generation import build_prompt
from local_llm import local_generate

app = Flask(__name__)
CORS(app)

# =========================================================
# Core RAG Helper (Explanation Layer Only)
# =========================================================

def run_rag_explanation(question: str, k: int = 3):
    """
    Shared RAG pipeline.
    This layer EXPLAINS using documentation.
    It does NOT invent or decide facts.
    """

    top_chunks = retrieve_top_k(question, k=k)

    if not top_chunks:
        return {
            "answer": "No relevant documentation was found to explain this request.",
            "sources": [],
            "question": question
        }

    chunk_texts = [c["text"] for c in top_chunks]

    guarded_question = (
        "Using ONLY the information provided below, "
        "explain the following request. "
        "If the documentation is insufficient, say so clearly.\n\n"
        f"REQUEST: {question}"
    )

    prompt = build_prompt(guarded_question, chunk_texts)
    answer = local_generate(prompt)

    sources = [
        {
            "rank": c["rank"],
            "index": c["index"],
            "distance": round(c["distance"], 4),
            "excerpt": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
        }
        for c in top_chunks
    ]

    return {
        "answer": answer,
        "sources": sources,
        "question": question
    }


# =========================================================
# Intent Handlers (Business Semantics)
# =========================================================

def handle_explain_suitability(params: dict):
    """
    intent: explain_suitability
    params: { modelA, application?, focus? }
    """
    model = (params.get("modelA") or "").strip()
    application = (params.get("application") or "general").strip()
    focus = (params.get("focus") or "overall suitability").strip()

    if not model:
        raise ValueError("modelA is required for explain_suitability")

    question = (
        f"Explain why pump {model} is suitable for {application} use, "
        f"with emphasis on {focus.replace('_', ' ')}."
    )

    result = run_rag_explanation(question, k=4)
    result.update({
        "intent": "explain_suitability",
        "model": model,
        "application": application,
        "focus": focus
    })

    return result


def handle_compare_models(params: dict):
    """
    intent: compare_models
    params: { modelA, modelB, application?, focus? }
    """
    model_a = (params.get("modelA") or "").strip()
    model_b = (params.get("modelB") or "").strip()
    application = (params.get("application") or "general").strip()
    focus = (params.get("focus") or "overall performance").strip()

    if not model_a or not model_b:
        raise ValueError("Both modelA and modelB are required for compare_models")

    if model_a == model_b:
        raise ValueError("Cannot compare the same pump model with itself")

    question = (
        f"Compare pumps {model_a} and {model_b} for {application} use, "
        f"highlighting differences in {focus.replace('_', ' ')}."
    )

    result = run_rag_explanation(question, k=5)
    result.update({
        "intent": "compare_models",
        "modelA": model_a,
        "modelB": model_b,
        "application": application,
        "focus": focus
    })

    return result


def handle_installation_guidance(params: dict):
    """
    intent: installation_guidance
    params: { modelA }
    """
    model = (params.get("modelA") or "").strip()

    if not model:
        raise ValueError("modelA is required for installation_guidance")

    question = (
        f"What installation guidelines and precautions are documented "
        f"for pump {model}?"
    )

    result = run_rag_explanation(question, k=4)
    result.update({
        "intent": "installation_guidance",
        "model": model
    })

    return result


def handle_explain_specs(params: dict):
    """
    intent: explain_specs
    params: { modelA }
    """
    model = (params.get("modelA") or "").strip()

    if not model:
        raise ValueError("modelA is required for explain_specs")

    question = (
        f"Explain the technical specifications of pump {model} "
        f"based on available documentation."
    )

    result = run_rag_explanation(question, k=3)
    result.update({
        "intent": "explain_specs",
        "model": model
    })

    return result


# =========================================================
# Intent Router
# =========================================================

def route_intent(intent: str, params: dict):
    if not intent:
        raise ValueError("Intent is required")

    params = params or {}

    print("INTENT:", intent, "| PARAMS:", params)

    if intent == "explain_suitability":
        return handle_explain_suitability(params)

    if intent == "compare_models":
        return handle_compare_models(params)

    if intent == "installation_guidance":
        return handle_installation_guidance(params)

    if intent in ("view_specs", "explain_specs"):
        return handle_explain_specs(params)

    raise ValueError(f"Unknown intent: {intent}")


# =========================================================
# API Routes
# =========================================================

@app.route("/ask", methods=["POST"])
def ask():
    """
    POST /ask

    Structured (recommended):
    {
      "intent": "compare_models",
      "params": {
        "modelA": "MDH-36A (C.I.)",
        "modelB": "LHP-3B",
        "application": "agricultural",
        "focus": "performance"
      }
    }

    Free-text fallback (debug / exploration only):
    {
      "question": "Explain why MDL-4SP/TC is suitable for domestic use."
    }
    """
    try:
        data = request.get_json(force=True) or {}

        # Preferred structured path
        if "intent" in data:
            intent = data.get("intent")
            params = data.get("params", {})
            result = route_intent(intent, params)
            return jsonify(result), 200

        # Controlled free-text fallback
        if "question" in data:
            question = (data.get("question") or "").strip()
            if not question:
                return jsonify({"error": "Question cannot be empty"}), 400

            result = run_rag_explanation(question, k=3)
            result["intent"] = "free_text"
            return jsonify(result), 200

        return jsonify({"error": "Request must contain 'intent' or 'question'"}), 400

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "name": "Pump Documentation Assistant",
        "version": "3.1 – Intent-driven RAG (Explanation Layer)",
        "principle": "RAG explains documentation; structured systems remain the source of truth.",
        "intents": [
            "explain_suitability",
            "compare_models",
            "installation_guidance",
            "explain_specs"
        ],
        "endpoints": {
            "POST /ask": "Execute structured intent or controlled free-text",
            "GET /health": "Health check"
        }
    }), 200


if __name__ == "__main__":
    print("🚀 Intent-driven RAG Backend starting...")
    print("📍 Ensure Ollama is running: ollama serve")
    print("🌐 API available at http://127.0.0.1:5000")
    print("-" * 60)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        threaded=True
    )
