"""Default configuration values for the AI QnA Assistant.

Defines the baseline configuration for models, API, retrieval, formatting, and
other system parameters. These defaults are overridden by YAML config and
environment variables at runtime.
"""

# Default configuration dictionary
DEFAULT_CONFIG = {
    # -------------------------------------------------------------------------
    # OpenAI Client Configuration
    # -------------------------------------------------------------------------
    "CLIENT_ID": "",      # Set via environment variables
    "CLIENT_SECRET": "",  # Set via .env
    "SAND_TOKEN_URL": "",  # Default SAND token URL; set via .env
    "SCOPE": "COUPA",     # Default scope for authentication

    # -------------------------------------------------------------------------
    # Model and API Configuration
    # -------------------------------------------------------------------------
    "MODEL": "gpt-4o-mini",
    "VISION_MODEL": "gpt-4o-mini",
    "EMBED_MODEL": "text-embedding-3-small",
    "DEFAULT_TEMPERATURE": 0.3,  # Setting to 0.0 will make the model deterministic

    # -------------------------------------------------------------------------
    # Vectore store Configuration
    # -------------------------------------------------------------------------
    "VECTOR_STORE_PATH": "vector_store/vector_store_all",
    "VECTOR_STORE_BATCH_SIZE": 64,     # Batch size for embedding API calls
    "VECTORSTORE_WORKERS": 3,         # Number of parallel worker threads
    "VECTOR_STORE_CHUNK_SIZE": 100,   # Documents per worker thread
    "EMBEDDING_DIMENSION": 1536,       # Default dimension for embeddings
    "VECTOR_STORE_LOCK_TIMEOUT": 60,   # Seconds to wait for a lock before timing out
    "VECTOR_STORE_SIMILARITY_ALGORITHM": "cosine",  # Options: cosine, dot, euclidean
    # indexing configuration
    "INDEX_DEFAULTS": {
        "CHUNK_STRATEGY": "header",  # Options: header, fixed, semantic, etc.
        "BATCH_SIZE": 64,
    },

    # -------------------------------------------------------------------------
    # Metadata Framework Configuration
    # -------------------------------------------------------------------------
    # Canonical metadata keys that all documents should have where applicable
    "CANONICAL_METADATA_KEYS": [
        "source",
        "source_type",
        "title",
        "content_type",
        "language",
        "categories",
        "tags",
        "entity_name",
        "entity_id",
        "creation_date",
        "last_modified",
        "content_purpose",
        "author"
    ],

    # -------------------------------------------------------------------------
    # RAG Configuration
    # -------------------------------------------------------------------------
    # Retrieval parameters
    "TOP_K": 5,  # Number of documents to retrieve
    "SIMILARITY_THRESHOLD": 0.2,  # 0.0-1.0
    "FILTER_LOW_SCORES": True,

    # RAG system prompts
    "PROMPT_TEMPLATES": {
        "default": "You are an expert on products. Answer using only the provided Context.",
        "cso": "You are an expert on CSO (Sourcing Optimization). Answer using only the provided Context.",
        "sourcing": "You are an expert on Sourcing. Answer using only the provided Context."
    },

    # Token limits
    "MAX_TOKENS": 2048,
    "MAX_HISTORY_TOKENS": 1024,

    # Answer formatting
    "ENABLE_CITATIONS": True,
    "DEDUPLICATE_SOURCES": True,
    "INCLUDE_IMAGES": True,

    # Chunking and processing
    "DEFAULT_CHUNK_SIZE": 800,
    "DEFAULT_CHUNK_OVERLAP": 100,
    "FORCE_REGENERATE_DEFAULT": False,
    "FORCE_REGENERATE_VECTOR": False,

    # -------------------------------------------------------------------------
    # Classification Configuration
    # -------------------------------------------------------------------------
    "CLASSIFICATION_MATCH_WEIGHT": 1.0,
    "CLASSIFICATION_TITLE_BOOST": 2.0,
    "CLASSIFICATION_MAX_KEYPHRASES": 10,
    "CLASSIFICATION_KEYPHRASE_EXTRACTION_METHOD": "hybrid", # keybert, pke, tfidf, hybrid
    "CLASSIFICATION_OFFLINE_MODE": True, # Attempt to run classification models offline
}

# Add path configuration defaults with proper scoping
DEFAULT_CONFIG.update({
    # Path roots and structure
    "PATHS": {
        "OUTPUTS_ROOT": "outputs",
        "VECTOR_STORE_ROOT": "vector_store",
        "CUSTOMERS_ROOT": "customers",
        "DATASETS_ROOT": "datasets",
        "ASSETS_ROOT": "assets",
        "TEMP_ROOT": "temp",
        "CONFIG_ROOT": "config"
    },

    # Customer path structure
    "CUSTOMER_PATHS": {
        "outputs": "outputs",
        "vector_store": "vector_store",
        "datasets": "datasets",
        "assets": "assets",
        "config": "config",
        "temp": "temp"
    },

    # Isolation and scoping settings
    "PATH_SCOPES": {
        "VECTORSTORE_SCOPE": "dataset",  # Options: dataset, customer
        "ASSETS_SCOPE": "customer",      # Options: dataset, customer
        "OUTPUTS_SCOPE": "dataset",      # Options: dataset, customer
        "TEMP_SCOPE": "customer"         # Always customer-scoped
    },

    # Backend configuration
    "VECTOR_STORE": {
        "PROVIDER": "faiss",            # Vector store backend
        "ISOLATION_LEVEL": "dataset",   # How to isolate indices
        "OUTPUT_ROOT": None,            # Override default location
        "NAMING_TEMPLATE": "{customer_id}_{dataset_id}_{backend}"  # Index naming
    },

    # -------------------------------------------------------------------------
    # Customer Directory Structure
    # -------------------------------------------------------------------------
    "CUSTOMER_CONFIG_PATHS": {
        "CONFIG_DIR": "config",                    # Main config directory under customer root
        "DATASETS_DIR": "config/datasets",         # Dataset configurations directory
        "PIPELINES_DIR": "config/pipelines",       # Pipeline configurations directory
        "SCHEMAS_DIR": "config/schemas",           # Schema definitions directory
        "SESSIONS_DIR": "sessions",                # User sessions directory
        "CONVERSATIONS_DIR": "conversations",      # Conversation history directory
        "VECTOR_STORES_DIR": "vector_store",       # Vector store data directory
        "OUTPUTS_DIR": "outputs"                   # Processing outputs directory
    }
})

