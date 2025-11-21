# Simple Intranet Chatbot (FastAPI + sentence-transformers)

This is a simple, free, local chatbot you can run on Windows using Docker Desktop + WSL2.
It uses:
- FastAPI backend (chat API)
- sentence-transformers (all-MiniLM-L6-v2) for embeddings (free)
- Simple on-disk JSON vector store (no external vector DB)
- Ingest worker scans `data/` folder for text files and upserts embeddings
- A small frontend widget (HTML/JS) to embed in Google Sites (iframe)

Quick start:
1. Install Docker Desktop (WSL2) and ensure Docker is running.
2. Extract this repo to C:\intranet-chatbot
3. Create data/ folder and put .txt files
4. Run: docker compose up --build
5. Open: http://localhost:8000/docs
