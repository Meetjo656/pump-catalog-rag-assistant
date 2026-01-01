import os
import pickle
import warnings

import faiss
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv

from read_rows import read_csv
from row_to_text import row_to_text

warnings.filterwarnings("ignore", category=FutureWarning)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

CSV_DIR    = os.path.join(os.path.dirname(BASE_DIR), "csv")
VECTOR_DIR = os.path.join(BASE_DIR, "vector_store")

# All CSVs you want in RAG
RAG_FILES = [
    "pump-excel-samples.csv",     # master model list
    "pump-features-derived.csv",  # advantages / limitations / use cases
    "pump-specs-sample.csv",      # numeric specs
    "pump-tags-derived.csv",      # tags for suitability
    "series-defaults.csv",        # series-level defaults (no model_id)
    "installation-library.csv",
    "maintenance-library.csv",
    "troubleshooting-library.csv",
    "pump-apps-sample.csv",
    "pump-cert-sample.csv",
    "pump-energy-sample.csv",
]

EMBED_MODEL = "text-embedding-004"
api_key = os.getenv("GEMINI_API_KEY")
print("GEMINI_API_KEY =", "SET" if api_key else "MISSING")
client = genai.Client(api_key=api_key)


def infer_chunk_type(filename: str) -> str:
    """Infer chunk type from filename for metadata."""
    name = filename.lower()
    if "features" in name:
        return "pump_features"
    if "spec" in name:
        return "specifications"
    if "tag" in name:
        return "tags"
    if "installation" in name:
        return "installation_guide"
    if "maintenance" in name:
        return "maintenance"
    if "troubleshooting" in name:
        return "troubleshooting"
    if "apps" in name or "application" in name:
        return "applications"
    if "cert" in name:
        return "certifications"
    if "energy" in name:
        return "energy_data"
    if "series-defaults" in name:
        return "series_defaults"
    return "general"


def derive_model_and_category(file: str, row: dict) -> tuple[str, str]:
    """
    Normalize how we set metadata['model'] and metadata['category']
    across all CSVs.
    """
    name = file.lower()

    # For series-defaults: series_family only, no model_id
    if "series-defaults" in name:
        model = "ALL"
        category = row.get("series_family") or "general"
        return model, category

    # For most pump-level CSVs: model_id is the primary key
    model = (
        row.get("model_id")
        or row.get("model_name")   # just in case some file uses name as id
        or row.get("model")
        or row.get("pump_model")
        or row.get("pump_code")
        or row.get("pump_model_name")
        or "ALL"
    )

    # Category preference order tuned to your columns
    category = (
        row.get("sub_category")      # e.g. DRAINAGE PUMPS, SEWAGE PUMP
        or row.get("category")      # Domestic / Industrial if no sub_category
        or row.get("spec_category") # from specs
        or row.get("feature_category")
        or row.get("tag_category")
        or row.get("application")
        or "general"
    )

    return model, category


def main():
    print("CSV_DIR =", CSV_DIR)
    print("Files in CSV_DIR:", os.listdir(CSV_DIR))

    print("Step 1: Processing structured CSV knowledge into documents...")
    documents: list[str] = []
    doc_metadata: list[dict] = []

    for file in RAG_FILES:
        path = os.path.join(CSV_DIR, file)
        if not os.path.exists(path):
            print(f"⚠️  Skipping {file} - not found at {path}")
            continue

        chunk_type = infer_chunk_type(file)
        print(f"  📄 Processing {file} as {chunk_type}...")

        rows = read_csv(path)
        for i, row in enumerate(rows):
            text = row_to_text(file, row)
            if not text or not text.strip():
                continue

            model_name, category = derive_model_and_category(file, row)

            documents.append(text)
            doc_metadata.append(
                {
                    "source": file,
                    "chunk_type": chunk_type,
                    "model": model_name,
                    "category": category,
                    "row_id": i,
                }
            )

    print(f"✅ Loaded {len(documents)} chunks from CSVs")
    if not documents:
        print("❌ No documents found. Aborting.")
        return

    # ----- Batched embeddings -----
    print("\nStep 2: Creating embeddings (batched)...")
    embeddings: list[list[float]] = []
    BATCH_SIZE = 100

    for start in range(0, len(documents), BATCH_SIZE):
        batch = documents[start : start + BATCH_SIZE]
        print(f"  🔹 Embedding batch {start}–{start + len(batch) - 1}")
        resp = client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        embeddings.extend(e.values for e in resp.embeddings)

    print(f"✅ Created {len(embeddings)} embeddings (dim={len(embeddings[0])})")

    # ----- FAISS index -----
    print("\nStep 3: Building FAISS index...")
    embeddings_array = np.array(embeddings, dtype="float32")
    dimension = embeddings_array.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    print(f"✅ FAISS index contains {index.ntotal} vectors")

    # ----- Save vector store -----
    print("\nStep 4: Saving vector store...")
    os.makedirs(VECTOR_DIR, exist_ok=True)

    faiss.write_index(index, os.path.join(VECTOR_DIR, "pump_index.faiss"))
    with open(os.path.join(VECTOR_DIR, "pump_documents.pkl"), "wb") as f:
        pickle.dump(documents, f)
    with open(os.path.join(VECTOR_DIR, "pump_metadata.pkl"), "wb") as f:
        pickle.dump(doc_metadata, f)
    with open(os.path.join(VECTOR_DIR, "pump_embeddings.pkl"), "wb") as f:
        pickle.dump(embeddings, f)

    print("🎉 COMPLETE! Structured enterprise RAG index ready.")
    print(f"📁 Files saved in: {VECTOR_DIR}")
    print(f"  • pump_index.faiss ({index.ntotal} vectors)")
    print(f"  • pump_documents.pkl ({len(documents)} chunks)")
    print(f"  • pump_metadata.pkl ({len(doc_metadata)} entries)")


if __name__ == "__main__":
    main()
