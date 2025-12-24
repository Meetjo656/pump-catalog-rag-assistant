import google.generativeai as genai
import warnings
import faiss
import numpy as np
import pickle
import os

warnings.filterwarnings('ignore', category=FutureWarning)

# Configure Gemini
genai.configure(api_key="AIzaSyAovKsl8nWrcq7Rky4kwUbvwjRejZ6y1qc")

print("Step 1: Reading pump descriptions...")
# Read chunks
with open(r"data\pump_descriptions.txt", "r", encoding="utf-8") as file:
    content = file.read()

chunks = content.split("Source: Domestic Pump Catalog")
chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

print(f"✅ Found {len(chunks)} pump descriptions")

print("\nStep 2: Creating embeddings...")
# Create embeddings
result = genai.embed_content(
    model="models/text-embedding-004",
    content=chunks,
    task_type="retrieval_document"
)

embeddings = result['embedding']
print(f"✅ Created {len(embeddings)} embeddings")
print(f"📊 Embedding dimension: {len(embeddings[0])}")

print("\nStep 3: Building FAISS index...")
# Convert to numpy array for FAISS
embeddings_array = np.array(embeddings).astype('float32')

# Create FAISS index (L2 distance)
dimension = len(embeddings[0])  # 768 for Gemini
index = faiss.IndexFlatL2(dimension)

# Add embeddings to index
index.add(embeddings_array)

print(f"✅ FAISS index created with {index.ntotal} vectors")

print("\nStep 4: Saving index and chunks...")
# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Save the FAISS index
faiss.write_index(index, "data/pump_index.faiss")
print("✅ Saved FAISS index to: data/pump_index.faiss")

# Save the chunks (for retrieval later)
with open("data/pump_chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)
print("✅ Saved pump chunks to: data/pump_chunks.pkl")

# Save embeddings as well (optional, for inspection)
with open("data/pump_embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)
print("✅ Saved embeddings to: data/pump_embeddings.pkl")

print("\n" + "="*80)
print("🎉 SUCCESS! Embeddings stored successfully!")
print("="*80)
print(f"\nSummary:")
print(f"  • Total chunks: {len(chunks)}")
print(f"  • Embedding dimension: {dimension}")
print(f"  • Index size: {index.ntotal} vectors")
print(f"\nFiles created:")
print(f"  • data/pump_index.faiss")
print(f"  • data/pump_chunks.pkl")
print(f"  • data/pump_embeddings.pkl")
print("\nYou can now run the search script!")