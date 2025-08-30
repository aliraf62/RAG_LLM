from core.config.paths import project_path
from core.pipeline.indexing.indexer import build_index_from_documents
html_dir = project_path("outputs/CSO_workflow_html_exports/html")
out_dir = project_path("vector_store/vector_store_all")

# Use in function calls
build_index_from_documents(
    html_dir=html_dir,
    out_dir=out_dir,
    worker_count=4,  # Enable multi-threading
    chunk_size=500, # Each worker processes 500 documents at a time
    limit=30
)