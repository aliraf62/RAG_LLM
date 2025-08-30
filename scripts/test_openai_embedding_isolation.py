"""
Test script to isolate OpenAI embedding API issues with random text chunks.
Respects VECTOR_STORE_BATCH_SIZE, VECTORSTORE_WORKERS, VECTOR_STORE_CHUNK_SIZE from config.
"""
import random
import string
from core.config.settings import settings
from core.utils.component_registry import create_component_instance
from tqdm import tqdm

# Generate random text chunks
def random_text(n=100, min_len=20, max_len=100):
    return [
        ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=random.randint(min_len, max_len)))
        for _ in range(n)
    ]

def main():
    batch_size = settings.get("VECTOR_STORE_BATCH_SIZE", 64)
    workers = settings.get("VECTORSTORE_WORKERS", 1)
    chunk_size = settings.get("VECTOR_STORE_CHUNK_SIZE", 100)
    print(f"[DEBUG] VECTOR_STORE_BATCH_SIZE={batch_size}, VECTORSTORE_WORKERS={workers}, VECTOR_STORE_CHUNK_SIZE={chunk_size}")

    # Generate random text chunks
    texts = random_text(200, 30, 120)  # 200 random chunks
    print(f"[DEBUG] Generated {len(texts)} random text chunks.")

    # Create embedder
    embedder_type = settings.get("EMBEDDER_PROVIDER", "openai")
    embedder = create_component_instance("embedder", embedder_type)
    print(f"[DEBUG] Using embedder: {embedder_type}")

    # Try embedding with progress bar
    embeddings = []
    try:
        for emb in tqdm(embedder.embed_text(texts), desc="Embedding", unit="vec", ncols=80):
            embeddings.append(emb)
        print(f"[DEBUG] Embedded {len(embeddings)} chunks successfully.")
    except Exception as e:
        print(f"[ERROR] Embedding failed: {e}")

if __name__ == "__main__":
    main()
