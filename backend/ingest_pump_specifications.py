import os
import csv
import pickle
import numpy as np
import faiss
from google import genai
from google.genai import types
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

load_dotenv(os.path.join(BASE_DIR, ".env"))

VECTOR_DIR = os.path.join(BASE_DIR, "vector_store")
os.makedirs(VECTOR_DIR, exist_ok=True)

INDEX_PATH = os.path.join(VECTOR_DIR, "pump_index.faiss")
DOCS_PATH = os.path.join(VECTOR_DIR, "pump_documents.pkl")
META_PATH = os.path.join(VECTOR_DIR, "pump_metadata.pkl")

SPEC_CSV = os.path.join(PROJECT_ROOT, "csv", "pump_specifications.csv")

EMBED_MODEL = "text-embedding-004"
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def ingest_pump_specifications():
    if not os.path.exists(SPEC_CSV):
        raise FileNotFoundError(SPEC_CSV)

    specs_by_model = {}

    with open(SPEC_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        print("CSV HEADERS:", reader.fieldnames)

        for row in reader:
            if not row.get("model_id"):
                raise RuntimeError(f"Invalid row: {row}")

            model_id = row["model_id"].strip()

            if "," in model_id:
                raise RuntimeError(
                    f"❌ model_id is polluted: {model_id}\n"
                    f"Row: {row}"
                )

            specs_by_model.setdefault(model_id, []).append(row)

    documents = []
    metadata = []

    for model_id, rows in specs_by_model.items():
        rows.sort(key=lambda r: int(r["spec_order"] or 999))

        lines = [f"Technical specifications for pump model {model_id}:"]
        for r in rows:
            value = r["spec_value"]
            unit = f" {r['unit']}" if r["unit"] else ""
            lines.append(f"- {r['spec_name']}: {value}{unit}")

        documents.append("\n".join(lines))
        metadata.append({
            "model": model_id,
            "chunk_type": "specifications",
            "category": "specifications",
            "source": "pump_specifications.csv",
        })

    print(f"MODELS INGESTED: {len(documents)}")
    print("\nSAMPLE DOC:\n", documents[0])
    print("\nSAMPLE META:\n", metadata[0])

    # ---- Embedding
    vectors = []
    for i in range(0, len(documents), 100):
        batch = documents[i:i+100]
        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        vectors.extend(e.values for e in result.embeddings)

    vectors = np.array(vectors, dtype="float32")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, INDEX_PATH)

    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)

    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("✅ INGEST COMPLETE")

if __name__ == "__main__":
    ingest_pump_specifications()
