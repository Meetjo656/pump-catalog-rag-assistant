from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import traceback
import os
import json
from pump_master import get_all_pumps
from rag_pipeline import (
    rag_explain_suitability,
    rag_compare_models,
    rag_installation_guidance,
    rag_view_specs,
    rag_free_text,
)

app = Flask(__name__)
CORS(app)
DATA_VERSION = "v2_structured_csv_2025"

def bad_request(message: str):
    return jsonify({
        "status": "error",
        "message": message
    }), 400

FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "frontend"
)

@app.route("/")
def serve_frontend():
    return send_from_directory(FRONTEND_DIR, "static.html")

@app.route("/styles.css")
def serve_css():
    return send_from_directory(FRONTEND_DIR, "styles.css")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"answer": {"error": "Invalid JSON"}}), 400

        if "intent" in data:
            intent = data["intent"]
            params = data.get("params", {})
            if not isinstance(params, dict):
                return bad_request("'params' must be an object")

            if intent == "explain_suitability":
                answer = rag_explain_suitability(params)
            elif intent == "compare_models":
                answer = rag_compare_models(params)
            elif intent == "installation_guidance":
                answer = rag_installation_guidance(params)
            elif intent == "view_specs":
                answer = rag_view_specs(params)
            else:
                return bad_request(f"Unknown intent: {intent}")

            return jsonify({
                "status": "success",
                "intent": intent,
                "answer": answer,
                "source": "rag"
            }), 200

        if "question" in data:
            question = data["question"]
            if not isinstance(question, str) or not question.strip():
                return bad_request("'question' must be a non-empty string")
            answer = rag_free_text(question)
            return jsonify({
                "status": "success",
                "intent": "free_text",
                "answer": answer,
                "source": "rag"
            }), 200

        return bad_request("Request must include 'intent' or 'question'")

    except ValueError as ve:
        return bad_request(str(ve))
    except Exception:
        print("🔥 INTERNAL SERVER ERROR 🔥")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": "Internal server error"
        }), 500

@app.route("/models", methods=["GET"])
def list_models():
    try:
        pumps = get_all_pumps()
        return jsonify({
            "status": "success",
            "models": pumps,
            "count": len(pumps)
        }), 200
    except Exception as e:
        print("ERROR in /models:", e)
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "models": [],
            "message": str(e)
        }), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "data_version": DATA_VERSION
    }), 200

@app.route("/safe-questions", methods=["GET"])
def safe_questions():
    try:
        path = os.path.join(os.path.dirname(__file__), "safe_questions.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify({
            "status": "success",
            "data": data
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
