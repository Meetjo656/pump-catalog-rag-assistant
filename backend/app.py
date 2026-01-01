from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import os
import pickle
from pump_master import get_all_pumps
from rag_pipeline import (
    rag_explain_suitability,
    rag_compare_models,
    rag_installation_guidance,
    rag_view_specs,
    rag_free_text,
)
from retriever import VECTOR_DIR  # uses same vector_store path


app = Flask(__name__)
CORS(app)

DATA_VERSION = "v2_structured_csv_2025"


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json or {}

    try:
        if "intent" in data:
            intent = data["intent"]
            params = data.get("params", {})

            if intent == "explain_suitability":
                answer = rag_explain_suitability(params)
            elif intent == "compare_models":
                answer = rag_compare_models(params)
            elif intent == "installation_guidance":
                answer = rag_installation_guidance(params)
            elif intent == "view_specs":
                answer = rag_view_specs(params)
            else:
                return jsonify({"error": f"Unknown intent: {intent}"}), 400

            return jsonify({
                "intent": intent,
                "answer": answer,
                "source": "rag",
            }), 200

        elif "question" in data:
            answer = rag_free_text(data["question"])
            return jsonify({
                "intent": "free_text",
                "answer": answer,
                "source": "rag",
            }), 200

        else:
            return jsonify({
                "error": "Request must contain 'intent' or 'question'"
            }), 400

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception:
        print("🔥 INTERNAL SERVER ERROR 🔥")
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500


@app.route("/models", methods=["GET"])
def list_models():
    try:
        pumps = get_all_pumps()
        return jsonify({
            "models": pumps,
            "count": len(pumps)
        }), 200
    except Exception as e:
        print("ERROR in /models:", e)
        traceback.print_exc()
        return jsonify({"models": [], "error": str(e)}), 500



@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "data_version": DATA_VERSION,
    }), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
