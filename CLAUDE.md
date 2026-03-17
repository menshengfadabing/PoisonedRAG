# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PoisonedRAG is a RAG (Retrieval-Augmented Generation) system with security protection against poisoning attacks. It's a graduation project that combines LangGraph + LangChain for the core framework.

## Commands

```bash
# Run Streamlit app
uv run streamlit run src/poisonedrag/app.py

# Run tests
uv run pytest

# Code linting
uv run ruff check .

# Install dependencies
uv pip install <package>
```

## Architecture

The system has three main layers:

1. **RAG Layer** (`rag/`): Retriever + Generator using LangGraph workflow
   - Retriever: Vector similarity search with ChromaDB
   - Generator: LangGraph state machine with retrieve → generate → validate nodes

2. **Security Layer** (`security/`): Content filtering + Response validation
   - `filter.py`: Keyword detection, semantic anomaly detection, source trust verification
   - `validator.py`: Fact consistency check, safety rule matching, confidence scoring

3. **Data Layer** (`data/`): Knowledge base + Poison corpus management
   - Knowledge documents in `data/knowledge/` (JSON format)
   - Poison samples in `data/poison/` for testing security features

## Key Configuration

Configuration is centralized in `config.py`:
- DeepSeek API (LLM): `deepseek-chat` model
- Ollama embedding: `qwen3-embedding:0.6b` on `localhost:11434`
- ChromaDB persisted at `data/chroma/`

Configuration uses singleton pattern via `get_config()`.

## Module Dependencies

```
config.py ← embeddings.py, llm.py, vectorstore.py
embeddings.py ← vectorstore.py
vectorstore.py ← rag/retriever.py
retriever.py ← rag/generator.py
security/filter.py ← rag/retriever.py (injected)
security/validator.py ← rag/generator.py (injected)
```

## Data Format

Knowledge documents (JSON):
```json
[{"content": "...", "metadata": {"source": "...", "category": "..."}}]
```

Poison samples (JSON):
```json
[{"content": "...", "poison_type": "misinformation|malicious_guidance|...", "description": "..."}]
```

## Environment

- WSL2 + Ubuntu 24.04
- Python 3.13
- UV package manager (use `uv` commands, not pip directly)
- Requires running Ollama service for embeddings