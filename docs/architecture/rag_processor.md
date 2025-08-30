# RAG Processor Architecture

This document describes the architecture and flow of the `rag_processor.py` module, which implements the core Retrieval-Augmented Generation (RAG) logic for the Coupa AI Assistant.

---

## Overview

The RAG processor is responsible for:
- Classifying user questions
- Retrieving relevant documents from a vector store
- Building prompts for the LLM using retrieved context
- Generating answers using an LLM (optionally with images)
- Formatting the final response, including citations and sources

---

## Main Components

### 1. `RAGResult` TypedDict

Defines the structure of the result returned by a RAG query:
- `question`: The original user question
- `classification`: Domain/type classification
- `index_path`: Path to the vector store used
- `docs`: List of retrieved documents (with content, score, source, etc.)
- `docs_count`: Number of retrieved documents
- `raw_output`: If True, skips LLM and returns only retrieved docs
- `answer`: The generated answer (if not in raw mode)

---

### 2. Classification

```python
classification = classify_question(question)
primary_domain = classification["primary_domain"]
prompt_type = classification["prompt_type"]
```
- Determines which vector store and prompt template to use.

---

### 3. Retrieval

```python
docs = retrieve_documents(
    question=question,
    index_path=index_path / "index.faiss",
    metadata_path=index_path / "metadata.jsonl",
    embed_fn=embed_string,
    top_k=retrieval_k,
    threshold=similarity_threshold,
)
```
- Retrieves top-K relevant documents using vector similarity.
- Applies a similarity threshold if configured.

---

### 4. Prompt Construction

```python
system_prompt, user_prompt = build_rag_prompt(
    question=question,
    docs=docs,
    primary_domain=primary_domain,
    system_prompt=system_prompt,
    include_images=include_images
)
```
- Builds the system and user prompts for the LLM, optionally including images.

---

### 5. Answer Generation

```python
answer = chat_completion(
    system=system_prompt,
    user=user_prompt,
    max_tokens=max_tokens,
    temperature=temperature
)
```
- Calls the LLM to generate an answer based on the constructed prompt.

---

### 6. Formatting the Response

```python
def format_rag_response(result: RAGResult) -> str:
    # ...
```
- Formats the answer and appends citations or sources as needed.
- If `raw_output` is True, returns a debug-friendly listing of retrieved docs.

---

## Configuration

- All behavior is controlled by the `config` object, which merges defaults, YAML, and environment variables.
- Key options: retrieval parameters, chunking, prompt templates, citation formatting, etc.

---

## Extensibility

- **Prompt Management**: Uses `prompt_manager` for domain- and type-specific prompts.
- **Citation Formatting**: Uses `get_citation_text` from `core.context_formatter`.
- **Pluggable Retrieval**: Retrieval and embedding functions can be swapped as needed.

---

## Typical Flow

1. **User Query** → `process_rag_query`
2. **Classification** → Selects vector store and prompt
3. **Retrieval** → Gets relevant docs from vector DB
4. **Prompt Build** → Assembles context and question
5. **LLM Call** → Generates answer (unless `raw=True`)
6. **Formatting** → Adds citations/sources, returns string

---

## References

- [core/rag_processor.py](../../core/rag_processor.py)
- [core/context_formatter.py](../../core/rag/context_formatter.py)
- [core/prompt_manager.py](../../core/rag/prompt_manager.py)
- [vector_store/retrieve.py](../../core/pipeline/retrievers/faiss_retriever.py)
