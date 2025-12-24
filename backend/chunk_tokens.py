import pandas as pd

# Read and split into chunks
with open(r"data\pump_descriptions.txt", "r", encoding="utf-8") as file:
    content = file.read()

# Split by "Source: Domestic Pump Catalog" to separate pump descriptions
chunks = content.split("Source: Domestic Pump Catalog")

# Clean up chunks - remove empty strings and strip whitespace
chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

# Add back the source information to each chunk (optional)
chunks_with_source = [chunk + "\nSource: Domestic Pump Catalog" for chunk in chunks]

print(f"Total chunks: {len(chunks)}")
print("\nFirst 3 chunks:")
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk)