import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Get API key from environment
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Rest of your code...
with open(r"data\pump_descriptions.txt", "r", encoding="utf-8") as file:
    content = file.read()

chunks = content.split("Source: Domestic Pump Catalog")
chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

print(f"Processing {len(chunks)} chunks...")

# Create embeddings
embeddings = []
for i, chunk in enumerate(chunks):
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=chunk,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT"
        )
    )
    embeddings.append(response.embeddings[0].values)
    
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(chunks)} chunks")

print(f"\n✅ Created {len(embeddings)} embeddings")
print(f"📊 Embedding dimension: {len(embeddings[0])}")