import os
import csv
import pickle
import numpy as np
import faiss
from collections import defaultdict
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pump_master import resolve_model_identifier

# =========================================================
# Paths & env
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

CSV_DIR = os.path.join(PROJECT_ROOT, "csv")
VECTOR_DIR = os.path.join(BASE_DIR, "vector_store")
os.makedirs(VECTOR_DIR, exist_ok=True)

INDEX_PATH = os.path.join(VECTOR_DIR, "pump_index.faiss")
DOCS_PATH  = os.path.join(VECTOR_DIR, "pump_documents.pkl")
META_PATH  = os.path.join(VECTOR_DIR, "pump_metadata.pkl")

load_dotenv(os.path.join(BASE_DIR, ".env"))

EMBED_MODEL = "text-embedding-004"
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# =========================================================
# CSVs ENABLED FOR RAG
# =========================================================

RAG_FILES = [
    # --- Technical specifications ---
    "pump_specifications.csv",
    "pump-energy-sample.csv",
    "pump-cert-sample.csv",

    # --- Suitability / reasoning ---
    "pump-features-derived.csv",
    "pump-apps-sample.csv",

    # --- Operational guidance ---
    "installation-library.csv",
    "maintenance-library.csv",
    "troubleshooting-library.csv",
]

# =========================================================
# Chunk type inference (MUST MATCH rag_pipeline.py)
# =========================================================

def infer_chunk_type(filename: str) -> str | None:
    name = filename.lower()

    if "features" in name or "apps" in name:
        return "features"

    if "spec" in name or "energy" in name or "cert" in name:
        return "specifications"

    if "installation" in name:
        return "installation"

    if "maintenance" in name:
        return "maintenance"

    if "troubleshooting" in name:
        return "troubleshooting"

    return None


# =========================================================
# Model & category resolution
# =========================================================
def resolve_model_and_category(row):
    """
    Resolve model and category for ingestion.
    - model is ALWAYS internal model_id or 'ALL'
    - NEVER returns None
    """

    raw_model = (
        row.get("model_id")
        or row.get("model_name")
        or row.get("model")
    )

    # 🔒 HARD RULE
    model = resolve_model_identifier(raw_model) if raw_model else "ALL"

    category = (
        row.get("sub_category")
        or row.get("category")
        or row.get("spec_category")
        or row.get("feature_category")
        or "general"
    )

    return model, category


# =========================================================
# Row → text converters (per knowledge type)
# =========================================================

def row_to_text(filename: str, row: dict) -> str | None:
    fname = filename.lower()

    if "pump_specifications" in fname:
        return f"- {row['spec_name']}: {row['spec_value']} {row.get('unit','')}".strip()

    if "energy" in fname:
        return f"- Energy detail: {row.get('energy_feature') or row.get('feature_value')}"

    if "cert" in fname:
        return f"- Certification: {row.get('certification') or row.get('feature_value')}"

    if "features" in fname:
        return f"- {row['feature_category']}: {row['feature_name']} – {row['feature_value']}"

    if "apps" in fname:
        return f"- Application: {row.get('application') or row.get('use_case')}"

    if "installation" in fname:
        return f"- Installation step: {row.get('step') or row.get('instruction')}"

    if "maintenance" in fname:
        return f"- Maintenance: {row.get('task') or row.get('instruction')}"

    if "troubleshooting" in fname:
        return (
            f"- Issue: {row.get('symptom')} | "
            f"Cause: {row.get('cause')} | "
            f"Solution: {row.get('solution')}"
        )
    if "apps" in fname:
        return (
            f"- Application: {row.get('application')} | "
            f"Suitability score: {row.get('suitability_score')} | "
            f"Recommended: {row.get('is_recommended')} | "
            f"Reason: {row.get('key_reason')}"
        )
    if "installation" in fname:
        return (
            f"- {row.get('step_title')}: {row.get('step_description')} "
            f"(Safety: {row.get('safety_warning')}; "
            f"Time: {row.get('estimated_time_min')} min; "
            f"Skill: {row.get('skill_required')})"
        )

    return None


# =========================================================
# MAIN INGESTION
# =========================================================

def ingest_all():
    documents = []
    metadata  = []

    print("📂 CSV DIR:", CSV_DIR)
    print("📄 RAG FILES:", RAG_FILES)

    for file in RAG_FILES:
        path = os.path.join(CSV_DIR, file)
        if not os.path.exists(path):
            print(f"⚠️  Skipping {file} (not found)")
            continue

        chunk_type = infer_chunk_type(file)
        if not chunk_type:
            continue

        print(f"📄 Processing {file} as {chunk_type}")

        grouped = defaultdict(list)

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_id, category = resolve_model_and_category(row)
                grouped[(model_id, category)].append(row)

        for (model_id, category), rows in grouped.items():
            lines = [f"Information for pump model {model_id}:"]
            for r in rows:
                text = row_to_text(file, r)
                if text:
                    lines.append(text)

            if len(lines) == 1:
                continue

            documents.append("\n".join(lines))
            metadata.append({
                "model": model_id,
                "chunk_type": chunk_type,
                "category": category,
                "source": file,
            })

    if not documents:
        raise RuntimeError("❌ No documents ingested")

    print(f"✅ Total chunks: {len(documents)}")

    # =====================================================
    # Embedding
    # =====================================================

    vectors = []
    BATCH = 100

    for i in range(0, len(documents), BATCH):
        batch = documents[i:i+BATCH]
        resp = client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        vectors.extend(e.values for e in resp.embeddings)

    vectors = np.array(vectors, dtype="float32")

    # =====================================================
    # FAISS
    # =====================================================

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, INDEX_PATH)

    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)

    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("🎉 FULL RAG INGEST COMPLETE")
    print(f"📦 Vectors: {index.ntotal}")


if __name__ == "__main__":
    ingest_all()
