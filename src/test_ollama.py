import ollama

reply = ollama.chat(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    options={"num_ctx": 12288, "temperature": 0, "seed": 42},
)
print(reply["message"]["content"])

emb = ollama.embed(model="nomic-embed-text", input="test sentence")
print(len(emb["embeddings"][0]))