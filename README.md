# Study Assistant

A local RAG (retrieval-augmented generation) chatbot that answers questions about my lecture notes, with source citations. Everything runs on my own GPU using Ollama, so no data leaves my machine.

## How it works
1. PDFs in `notes/` are split into chunks (about 300 words, with overlap)
2. Each chunk is embedded with `nomic-embed-text` and stored in Chroma
3. A question is rewritten into a standalone question using chat history
4. The top matching chunks are retrieved and sent to `llama3.1:8b`
5. The answer is shown with the source file and page numbers

## Tech stack
Python 3.12, Ollama, ChromaDB, pypdf

## Setup
1. Install Ollama, then run:
   - `ollama pull llama3.1:8b`
   - `ollama pull nomic-embed-text`
2. Create a virtual environment and install packages:
   - `python -m venv .venv`
   - `.venv\Scripts\Activate.ps1`
   - `pip install -r requirements.txt`
3. Put your own PDFs in a `notes/` folder (not included in this repo)
4. Build the database: `python src/rag.py ingest`
5. Ask questions: `python src/rag.py`

