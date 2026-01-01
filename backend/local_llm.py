import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def local_generate(prompt: str) -> str:
    payload = {
        "model": "llama3.2:latest",   # must match `ollama list`
        "prompt": prompt,
        "stream": False
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)

    if resp.status_code != 200:
        return f"LLM error: {resp.status_code}"

    data = resp.json()

    # ✅ Correct field for /api/generate
    text = data.get("response", "").strip()

    if not text:
        return "Information not available for this pump."

    return text
