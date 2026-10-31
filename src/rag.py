import sys
from pathlib import Path

import chromadb
import ollama
from pypdf import PdfReader

CHAT_MODEL = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"
NOTES_DIR = Path("notes")
DB_DIR = "chroma_db"
CHUNK_WORDS = 300
OVERLAP_WORDS = 50
TOP_K = 4

client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(
    "notes", metadata={"hnsw:space": "cosine"}
)


def chunk_text(text):
    words = text.split()
    step = CHUNK_WORDS - OVERLAP_WORDS
    for start in range(0, len(words), step):
        piece = words[start:start + CHUNK_WORDS]
        if piece:
            yield " ".join(piece)


def ingest():
    pdfs = list(NOTES_DIR.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in notes/")
        return
    for pdf_path in pdfs:
        print(f"Ingesting {pdf_path.name}...")
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for i, chunk in enumerate(chunk_text(text)):
                emb = ollama.embed(model=EMBED_MODEL, input=chunk)["embeddings"][0]
                collection.upsert(
                    ids=[f"{pdf_path.name}-p{page_num}-c{i}"],
                    documents=[chunk],
                    embeddings=[emb],
                    metadatas=[{"source": pdf_path.name, "page": page_num}],
                )
    print(f"Done. {collection.count()} chunks stored.")


def ask(question):
    q_emb = ollama.embed(model=EMBED_MODEL, input=question)["embeddings"][0]
    results = collection.query(query_embeddings=[q_emb], n_results=TOP_K)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    if not docs:
        return "No notes stored yet. Run: python src/rag.py ingest", []

    context = "\n\n".join(
        f"[{m['source']} page {m['page']}]\n{d}" for d, m in zip(docs, metas)
    )
    system = (
        "You are a study assistant. Answer using ONLY the context provided. "
        "If the answer is not in the context, say you don't know based on the notes."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"
    reply = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return reply["message"]["content"], metas


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        ingest()
        return
    print("Ask a question about your notes (type 'quit' to exit).")
    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() in ("quit", "exit"):
            break
        if not q:
            continue
        answer, metas = ask(q)
        print(f"\n{answer}\n")
        sources = sorted({(m["source"], m["page"]) for m in metas})
        print("Sources:", ", ".join(f"{s} p{p}" for s, p in sources))


if __name__ == "__main__":
    main()