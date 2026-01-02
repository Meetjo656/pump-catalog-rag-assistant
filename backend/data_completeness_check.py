import pickle
from collections import defaultdict
import os

# =========================================================
# Paths
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DIR = os.path.join(BASE_DIR, "vector_store")
METADATA_PATH = os.path.join(VECTOR_DIR, "pump_metadata.pkl")

# =========================================================
# Load metadata
# =========================================================

if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(
        "pump_metadata.pkl not found. "
        "Run ingestion before checking completeness."
    )

with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

# =========================================================
# Analyze coverage
# =========================================================

coverage = defaultdict(set)

for entry in metadata:
    model_id = entry.get("model")
    chunk_type = entry.get("chunk_type")
    if model_id and chunk_type:
        coverage[model_id].add(chunk_type)

# =========================================================
# Required data per intent
# =========================================================

INTENT_REQUIREMENTS = {
    "view_specs": {"specifications"},
    "explain_suitability": {"features"},
    "compare_models": {"specifications", "features"},
    "installation_guidance": {"installation"},
    "maintenance_guidance": {"maintenance"},
    "troubleshooting": {"troubleshooting"},
}

ALL_REQUIRED_CHUNKS = set().union(*INTENT_REQUIREMENTS.values())

# =========================================================
# Report
# =========================================================

print("\n📊 PUMP DATA COMPLETENESS REPORT\n")

for model_id in sorted(coverage.keys()):
    present = coverage[model_id]
    missing = ALL_REQUIRED_CHUNKS - present

    print(f"🔹 {model_id}")
    print(f"   ✔ Present : {', '.join(sorted(present))}")

    if missing:
        print(f"   ❌ Missing : {', '.join(sorted(missing))}")
    else:
        print("   ✅ Complete for all intents")

    # Intent-level availability
    for intent, required_chunks in INTENT_REQUIREMENTS.items():
        if required_chunks.issubset(present):
            print(f"     🟢 {intent}")
        else:
            print(f"     🔴 {intent}")

    print()

# =========================================================
# Summary
# =========================================================

print("Legend:")
print("  🟢 Intent fully supported")
print("  🔴 Intent NOT supported (missing data)")
print("\nDone.\n")
