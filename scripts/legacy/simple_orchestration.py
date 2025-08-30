from core.utils.component_registry import create_component_instance

extractor = create_component_instance("extractor", "excel", config)
rows = extractor.extract_rows()

cleaner = create_component_instance("cleaner", "html", config)
cleaned_rows = (cleaner.clean(row) for row in rows)

chunker = create_component_instance("chunker", "html", config)
chunks = (chunk for row in cleaned_rows for chunk in chunker.chunk(row))

embedder = create_component_instance("embedder", "openai", config)
embeddings = embedder.embed([chunk["text"] for chunk in chunks])

vector_store = create_component_instance("vectorstore", "faiss", config)
vector_store.add(embeddings, metadata)

retriever = create_component_instance("retriever", "faiss", config)
results = retriever.retrieve(query)