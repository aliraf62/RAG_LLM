# AI-Powered Document Q&A Assistant

A production-ready **Modular RAG (Retrieval-Augmented Generation) Pipeline** that transforms enterprise documentation into an intelligent AI-powered knowledge assistant.

## 🚀 Features

- **Multi-format Document Processing**: Excel, HTML, and markdown support
- **Intelligent Document Chunking**: Advanced NLP pipeline with semantic segmentation
- **Vector Search**: FAISS-powered semantic search for accurate retrieval
- **RAG-based Q&A**: Context-aware responses with source citations
- **Modular Architecture**: Plugin-based design for extensibility
- **CLI Interface**: User-friendly command-line tools
- **Production Ready**: Comprehensive error handling and logging

## 🛠 Tech Stack

- **Backend**: Python 3.9+, FastAPI
- **AI/ML**: OpenAI GPT models, FAISS vector database
- **Document Processing**: BeautifulSoup4, Pandas, custom parsers
- **CLI Framework**: Typer
- **Architecture**: Modular plugin-based design

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/aliraf62/RAG_LLM.git
cd RAG_LLM

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY=your_openai_api_key_here
```

## 🔧 Quick Start

```bash
# Process documents and build index
python scripts/build_vector_index_from_html.py --html-root path/to/docs --out vector_store/

# Query the knowledge base
python -m cli query "Your question here"
```

## 🏗 Architecture Highlights

### Customer-Specific Adapters
The system supports **plugin-based customer configurations** in the `customers/` directory:

```
customers/myCustomer1/
├── myCustomer1.yaml              # Main customer configuration
├── classification_config.py      # Custom classification rules
├── config/
│   ├── workflow_relational_schema.yaml  # Data schema definitions
│   └── users.json                # User management
├── formatters/
│   └── workflow_formatter.py     # Custom content formatters
└── utils/
    ├── classifiers.py            # Domain-specific classifiers
    └── formatters.py             # Text processing utilities
```

### Modular Pipeline Architecture
- **Extractors**: Pull data from Excel, HTML, and other formats
- **Cleaners**: Normalize and clean document content  
- **Chunkers**: Intelligent text segmentation with overlap
- **Embedders**: Convert text to vector representations
- **Retrievers**: FAISS-powered semantic search
- **Generators**: RAG-based response generation with citations

### Enterprise Features
- **Multi-tenant support** with customer-specific configurations
- **Schema-driven data processing** for complex Excel workbooks
- **Asset management** for images and downloadable files
- **Metadata enrichment** with automatic classification and tagging
- **Conversation memory** for contextual multi-turn interactions
[project]
name = "ai_qna_assistant"
version = "0.1.0"
description = "Modular GenAI RAG pipeline for myCustomer1 guides"
authors = [
    { name = "Ali Rafieefar", email = "aliraf62@gmail.com" }
]
readme = "README.md"
requires-python = ">=3.9"

# ------------- dependencies (copied from requirements.txt) -------------
dependencies = [
    "beautifulsoup4",
    "lxml",
    "tiktoken",
    "faiss-cpu",
    "pandas",
    "pyarrow",
    "rich",
    "openai>=1.3.7",
    "pyxlsb",
    "orjson",
    "html2text",
    "graphviz",
    "tabulate",
    "openpyxl",
    "pytest",
    "langdetect",
    "langchain",
    "colorama",
]

# -----------------------------------------------------------------------
# Tell setuptools WHICH folders are code and which ones to ignore
# -----------------------------------------------------------------------
[tool.setuptools]
package-dir = {"" = "."}

[tool.setuptools.packages.find]
where    = ["."]
include  = [
    "cli*", "core*", "data_ingestion*", "vector_store*"
]
exclude  = [
    "data*", "outputs*", "notebooks*", "scripts*", "tests*", "docs*"
]

[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"
pip install -e .
to regenerate requirements.txt
pip-compile -o requirements.txt --strip-extras pyproject.toml
aiqa is added to pyproject.toml under [project.scripts] aiqa = 'cli.commands:app' then pip install -e . then aiqa ping or aiqa export-guides --help
aiqa export-guides --log-level INFO --workbook data/datasets/in-product-guides-Guide+Export/"Workflow Steps.xlsb" \
                   --assets   data/datasets/in-product-guides-Guide+Export/WorkflowSteps_unpacked \
                   --output   outputs/CSO_workflow_html_exports \
                   --limit    5


# Typer CLI
aiqa query "What is smart import?"
aiqa query --index-dir vector_store/my_other_index "Show me gating steps"

# Slash style (inside chat / or piping into commands.dispatch)
/query Where are Excel files stored?
/query /another/index | How do I pause an event?
