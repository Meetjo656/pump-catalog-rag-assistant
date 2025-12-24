import os
import pickle
import numpy as np
import faiss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ../vector_store relative to backend/
VECTOR_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "vector_store"))

EMBEDDINGS_PATH = os.path.join(VECTOR_DIR, "pump_embeddings.pkl")
INDEX_PATH = os.path.join(VECTOR_DIR, "pump_index.faiss")

print("EMBEDDINGS_PATH =", EMBEDDINGS_PATH)
print("Exists? ", os.path.exists(EMBEDDINGS_PATH))

def build_faiss_index():
    print("Step 1: Loading Embeddings from disk...")
    with open(EMBEDDINGS_PATH, "rb") as f:
        embeddings = pickle.load(f)

    embeddings_array = np.array(embeddings).astype("float32")
    num_vectors, dim = embeddings_array.shape
    print(f"✅ Loaded {num_vectors} embeddings with dimension {dim}")

    print("Step 3: Creating FAISS IndexFlatL2...")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_array)
    print(f"✅ FAISS index created with {index.ntotal} vectors")

    print("Step 5: Saving FAISS index to disk...")
    os.makedirs(VECTOR_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    print(f"✅ Saved FAISS index to: {INDEX_PATH}")

if __name__ == "__main__":
    build_faiss_index()
