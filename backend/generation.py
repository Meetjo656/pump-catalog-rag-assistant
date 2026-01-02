import shutil
import subprocess
from collections import defaultdict
from typing import Any, Dict, List, Optional


def build_prompt(question: str, chunks: List[Dict[str, Any]]) -> Optional[str]:
    if not chunks:
        return None

    grouped = defaultdict(list)
    for c in chunks:
        if not isinstance(c, dict):
            continue
        model = (c.get("model") or "UNKNOWN")
        text = c.get("text", "")
        if isinstance(text, str) and text.strip():
            grouped[model].append(text.strip())

    if not grouped:
        return None

    # Keep order stable: UNKNOWN last
    model_order = sorted(grouped.keys(), key=lambda m: (m == "UNKNOWN", m))

    context_blocks = []
    for model in model_order:
        # De-duplicate lines to reduce repetition
        seen = set()
        lines = []
        for t in grouped[model]:
            if t in seen:
                continue
            seen.add(t)
            lines.append(t)

        if not lines:
            continue

        context_blocks.append(f"### MODEL: {model}\n" + "\n\n".join(lines))

    if not context_blocks:
        return None

    context = "\n\n".join(context_blocks)

    return (
        "You are Pump Info AI.\n"
        "You must answer using ONLY the CONTEXT.\n"
        "If the answer is not in the CONTEXT, reply exactly: Information not available.\n\n"
        "Formatting rules:\n"
        "- Output must be bullet points using '- '.\n"
        "- If the question asks for technical specifications, list EVERY spec item present in CONTEXT; do not omit items and do not summarize.\n"
        "- If multiple models appear, add headings 'MODEL <id>:' then bullets under each model.\n"
        "- Do not invent values.\n\n"
        "CONTEXT:\n"
        f"{context}\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "ANSWER:\n"
    )

def generate_answer(prompt: Optional[str], ollama_model: str = "llama3.1:8b") -> str:
    if not prompt:
        return "Information not available."

    if shutil.which("ollama") is None:
        return "Information not available."

    try:
        p = subprocess.run(
            ["ollama", "run", ollama_model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
        )
        text = (p.stdout or b"").decode("utf-8", errors="ignore").strip()

        if not text:
            # If ollama returned nothing, surface stderr in logs but keep safe response
            return "Information not available."

        return text
    except subprocess.TimeoutExpired:
        return "Information not available."
    except Exception:
        return "Information not available."
