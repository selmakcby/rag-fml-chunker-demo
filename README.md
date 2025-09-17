# LINDA Mini RAG Demo (Qdrant)

This is a minimal, **ready-to-run** setup to index your normalized chunks into Qdrant,
run filtered semantic search, and build a RAG prompt.

## 0) Prerequisites
- Docker (for Qdrant)
- Python 3.10+ with `pip`

## 1) Start Qdrant locally
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

## 2) Create a virtual env and install deps
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3) Put your normalized JSONs here
Copy your normalized files (the ones you just downloaded) into:
```
rag_demo/sample_chunks/
```
They should look like: `*.json` with fields like `chunk_id`, `level`, `summary_text`, etc.

## 4) Index chunks into Qdrant (embeddings + payload)
```bash
python index_chunks.py --chunks_dir sample_chunks --collection linda_chunks
```
Notes:
- By default this uses `sentence-transformers/all-MiniLM-L6-v2` (dim=384).
- If you prefer **Ollama** embeddings, run:
  ```bash
  export USE_OLLAMA=1
  export OLLAMA_URL=http://localhost:11434
  export EMBED_MODEL=nomic-embed-text
  python index_chunks.py --chunks_dir sample_chunks --collection linda_chunks
  ```

## 5) Try a couple of semantic+filter searches
```bash
python search_demo.py --collection linda_chunks
```

## 6) Build a simple RAG prompt with retrieved chunks
```bash
python rag_stub.py --collection linda_chunks --question "What characterizes the living area layout?"
```

## 7) Next steps
- Expand your chunk set and re-run the indexer.
- Add a quick evaluation (kNN room_type agreement) if useful.
- Optional: add image embeddings for schematic floorplans later.
