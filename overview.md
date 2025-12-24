## Architecture Overview

1. Pump catalog data is ingested from Excel/CSV files.
2. Specifications are converted into structured text chunks.
3. Sentence embeddings are generated offline.
4. FAISS is used for fast vector similarity search.
5. A local LLM (Ollama) generates answers strictly from retrieved context.
6. A Flask backend exposes the system to a frontend UI.
