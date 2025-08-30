from core.llm.providers.openai import get_client
from core.pipeline.embedders import OpenAIEmbeddert
import os
client = initialize_client()
try:
    r = client.embeddings.create(input=["ping"], model=os.getenv("EMBED_MODEL") or None)
    print("OK, got vector:", r.data[0].embedding[:5])
except Exception as e:
    print("FAILED:", e)